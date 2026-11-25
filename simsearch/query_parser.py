"""Plain-language search -> structured filters. This is the ONLY place an LLM is used.

"women over 60 with high BMI and rising glucose"
    -> {"gender": "female", "age_min": 60, "labs": {"bmi": "high"}, "trends": {"glucose": "rising"}}

Two parsers produce the same validated schema:
  * RuleParser  – deterministic keyword rules (always available, used when no key is set)
  * LLMParser   – Groq / Gemini / Ollama in JSON mode; any failure falls back to the rules

The parsed query then runs as normal hybrid search: age / sex / insurance become
hard filters; lab flags, trends and smoking become clinical-state tokens for BM25
and words for the embedding query; in strict mode results must satisfy them.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .index import Filters
from .llm import LLMError, get_llm

log = logging.getLogger(__name__)

LAB_KEYS = ["bmi", "systolic", "diastolic", "heart_rate", "glucose", "hba1c", "cholesterol", "ldl", "hdl",
            "triglycerides", "creatinine", "egfr", "hemoglobin", "microalbumin", "fev1_fvc", "pain", "phq2", "gad7"]
TREND_KEYS = ["bmi", "systolic", "glucose", "hba1c", "egfr", "cholesterol"]
LAB_WORDS = {  # longest phrases first when matching
    "bmi": ["body mass index", "bmi", "weight"], "systolic": ["blood pressure", "bp", "systolic"],
    "diastolic": ["diastolic"], "heart_rate": ["heart rate", "pulse"],
    "glucose": ["blood sugar", "blood glucose", "glucose", "sugar"], "hba1c": ["hba1c", "a1c", "haemoglobin a1c"],
    "cholesterol": ["total cholesterol", "cholesterol"], "ldl": ["ldl", "bad cholesterol"],
    "hdl": ["hdl", "good cholesterol"], "triglycerides": ["triglycerides", "triglyceride"],
    "creatinine": ["creatinine"], "egfr": ["egfr", "kidney function", "gfr"],
    "hemoglobin": ["hemoglobin", "haemoglobin"], "microalbumin": ["albumin", "microalbumin"],
    "fev1_fvc": ["lung function", "fev1"], "pain": ["pain"], "phq2": ["depression"], "gad7": ["anxiety"],
}
# Flags each parsed level accepts, per lab (matches the document builder's flags).
LEVEL_FLAGS = {"high": {"high", "obese", "overweight"}, "low": {"low", "underweight"}, "borderline": {"borderline", "overweight"},
               "normal": {None}}


class ParsedQuery(BaseModel):
    gender: Optional[Literal["female", "male"]] = None
    age_min: Optional[float] = Field(None, ge=0, le=120)
    age_max: Optional[float] = Field(None, ge=0, le=120)
    insurance: Optional[Literal["medicare", "medicaid", "private", "uninsured", "dual"]] = None
    smoking: Optional[Literal["current", "former", "never"]] = None
    labs: dict[str, Literal["high", "low", "borderline", "normal"]] = Field(default_factory=dict)
    trends: dict[str, Literal["rising", "falling"]] = Field(default_factory=dict)
    text: str = ""

    @field_validator("labs")
    @classmethod
    def _known_labs(cls, v):
        return {k: lvl for k, lvl in v.items() if k in LAB_KEYS}

    @field_validator("trends")
    @classmethod
    def _known_trends(cls, v):
        return {k: d for k, d in v.items() if k in TREND_KEYS}

    def filters(self) -> Filters:
        return Filters(gender=self.gender, age_min=self.age_min, age_max=self.age_max, insurance=self.insurance)

    def constraints(self) -> list[tuple[str, str, str]]:
        """Soft/strict conditions checked against each patient: (kind, key, value)."""
        out = [("lab", k, lvl) for k, lvl in self.labs.items()] + [("trend", k, d) for k, d in self.trends.items()]
        if self.smoking:
            out.append(("smoking", "smoking", self.smoking))
        return out

    def bm25_tokens(self) -> list[str]:
        toks = []
        for k, lvl in self.labs.items():
            flags = sorted(f for f in LEVEL_FLAGS[lvl] if f) if lvl != "normal" else ["normal"]
            toks += [f"{k}_{f}" for f in flags]
        toks += [f"{k}_{d}" for k, d in self.trends.items()]
        if self.gender:
            toks.append(f"sex_{self.gender}")
        if self.insurance:
            toks.append(f"insurance_{self.insurance}")
        if self.smoking:
            toks.append(f"smoker_{self.smoking}")
        if self.age_min is not None or self.age_max is not None:
            lo = int((self.age_min if self.age_min is not None else 0) // 10 * 10)
            hi = int((self.age_max if self.age_max is not None else 100) // 10 * 10)
            toks += [f"age_{a}s" for a in range(lo, min(hi, 100) + 1, 10)]
        return toks

    def sentence(self) -> str:
        """A short summary-like sentence for the embedding query."""
        bits = []
        if self.age_min is not None or self.age_max is not None:
            bits.append(f"age {int(self.age_min or 0)}-{int(self.age_max or 100)}")
        if self.gender:
            bits.append(self.gender)
        if self.insurance:
            bits.append(f"insurance {self.insurance}")
        if self.smoking:
            bits.append(f"smoking {self.smoking}")
        bits += [f"{k.replace('_', ' ')} {lvl}" for k, lvl in self.labs.items()]
        bits += [f"{k.replace('_', ' ')} {d}" for k, d in self.trends.items()]
        if self.text:
            bits.append(self.text)
        return ", ".join(bits)

    def chips(self) -> list[dict]:
        """Human-readable pieces the UI shows (and lets the user remove)."""
        c = []
        if self.gender:
            c.append({"field": "gender", "label": "Women" if self.gender == "female" else "Men"})
        if self.age_min is not None and self.age_max is not None:
            c.append({"field": "age", "label": f"Age {int(self.age_min)}–{int(self.age_max)}"})
        elif self.age_min is not None:
            c.append({"field": "age", "label": f"Age {int(self.age_min)}+"})
        elif self.age_max is not None:
            c.append({"field": "age", "label": f"Under {int(self.age_max)}"})
        if self.insurance:
            c.append({"field": "insurance", "label": {"uninsured": "No insurance", "dual": "Medicare + Medicaid"}.get(self.insurance, self.insurance.capitalize())})
        if self.smoking:
            c.append({"field": "smoking", "label": {"current": "Smoker", "former": "Ex-smoker", "never": "Never smoked"}[self.smoking]})
        for k, lvl in self.labs.items():
            c.append({"field": f"labs.{k}", "label": f"{LAB_LABEL.get(k, k)} {lvl}"})
        for k, d in self.trends.items():
            c.append({"field": f"trends.{k}", "label": f"{LAB_LABEL.get(k, k)} {d}"})
        return c


LAB_LABEL = {"bmi": "BMI", "systolic": "Blood pressure", "diastolic": "Diastolic BP", "heart_rate": "Heart rate",
             "glucose": "Blood sugar", "hba1c": "HbA1c", "cholesterol": "Cholesterol", "ldl": "LDL", "hdl": "HDL",
             "triglycerides": "Triglycerides", "creatinine": "Creatinine", "egfr": "Kidney function (eGFR)",
             "hemoglobin": "Hemoglobin", "microalbumin": "Urine albumin", "fev1_fvc": "Lung function",
             "pain": "Pain", "phq2": "Depression score", "gad7": "Anxiety score"}


# ----------------------------------------------------------------------------- rules
_HIGH = r"(?:high|higher|elevated|raised|increased|above normal|abnormal|uncontrolled|poorly controlled|bad|out of range)"
_LOW = r"(?:low|lower|reduced|decreased|below normal|poor)"
_BORDER = r"(?:borderline|slightly high|mildly high|pre-?diabetic)"
_NORMAL = r"(?:normal|healthy|good|controlled|well controlled|in range)"
_RISING = r"(?:rising|increasing|going up|getting worse|worsening|climbing|trending up)"
_CONJ = r"\b(?:and|with|but|or|who|plus|while)\b|,|;"
_WORSE_BETTER = r"\b(?:worsening|getting worse|deteriorating|improving|getting better)\b"
LOWER_IS_WORSE = {"egfr", "hdl", "hemoglobin", "fev1_fvc"}
_FALLING = r"(?:falling|decreasing|dropping|going down|improving|declining|trending down)"
NUM_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


class RuleParser:
    name = "rules"

    def parse(self, q: str) -> ParsedQuery:
        s = " " + re.sub(r"\s+", " ", q.lower()).strip() + " "
        for w, n in NUM_WORDS.items():
            s = re.sub(rf"\b{w}\b", str(n), s)
        out: dict = {"labs": {}, "trends": {}}
        used = []

        def take(pattern):
            m = re.search(pattern, s)
            if m:
                used.append(m.span())
            return m

        # sex (female before male: "female" contains "male")
        if take(r"\b(women|woman|female|females|ladies|lady|girls?)\b"):
            out["gender"] = "female"
        elif take(r"\b(men|man|male|males|gentlemen|boys?)\b"):
            out["gender"] = "male"
        # age
        if m := take(r"\b(?:between|aged|age)\s+(\d{1,3})\s*(?:and|to|-)\s*(\d{1,3})\b"):
            out["age_min"], out["age_max"] = float(m.group(1)), float(m.group(2))
        elif m := take(r"\b(\d{1,3})\s*(?:-|to)\s*(\d{1,3})\s*(?:years?|yrs?|y)?\b(?!\s*(?:mg|%|mmhg))"):
            a, b = float(m.group(1)), float(m.group(2))
            if 0 < a < b <= 120:
                out["age_min"], out["age_max"] = a, b
        else:
            if m := take(r"\b(?:over|above|older than|at least|aged over|age over|more than)\s+(\d{1,3})\b"):
                out["age_min"] = float(m.group(1))
            if m := take(r"\b(?:under|below|younger than|less than|aged under)\s+(\d{1,3})\b"):
                out["age_max"] = float(m.group(1))
            if m := take(r"\b(?:in (?:their|her|his) )?(\d)0s\b"):
                out["age_min"], out["age_max"] = float(m.group(1)) * 10, float(m.group(1)) * 10 + 9
            if m := take(r"\b(\d{2,3})\+"):
                out["age_min"] = float(m.group(1))
            if take(r"\b(elderly|older adults?|seniors?)\b") and "age_min" not in out:
                out["age_min"] = 65.0
            if take(r"\b(young adults?)\b") and "age_max" not in out:
                out["age_min"], out["age_max"] = 18.0, 35.0
        # insurance
        if take(r"\bdual(?:ly)?[ -]eligible\b|\bmedicare and medicaid\b|\bdual\b"):
            out["insurance"] = "dual"
        elif take(r"\bmedicaid\b"):
            out["insurance"] = "medicaid"
        elif take(r"\bmedicare\b"):
            out["insurance"] = "medicare"
        elif take(r"\b(uninsured|no insurance|without insurance|not insured)\b"):
            out["insurance"] = "uninsured"
        elif take(r"\bprivate(?:ly)?(?: insur\w*| plan| cover\w*)\b"):
            out["insurance"] = "private"
        # smoking
        if take(r"\b(ex-?smokers?|former smokers?|quit smoking|stopped smoking)\b"):
            out["smoking"] = "former"
        elif take(r"\b(non-?smokers?|never smoked|don'?t smoke|does not smoke|no smoking)\b"):
            out["smoking"] = "never"
        elif take(r"\b(smokers?|smoking|smokes|smoke)\b"):
            out["smoking"] = "current"
        # shorthand words for lab states
        if take(r"\b(obese|obesity|overweight)\b"):
            out["labs"]["bmi"] = "high"
        if take(r"\b(underweight|thin)\b"):
            out["labs"]["bmi"] = "low"
        if take(r"\b(hypertensive|high blood pressure|hypertension)\b"):
            out["labs"]["systolic"] = "high"
        if take(r"\b(anemic|anaemic)\b"):
            out["labs"]["hemoglobin"] = "low"
        if take(r"\b(diabetic|high blood sugar)\b"):
            out["labs"].setdefault("glucose", "high")
        # lab + level / trend, looked up only inside the lab's own clause
        # ("high BMI and rising glucose": "high" belongs to BMI, "rising" to glucose)
        for key, words in LAB_WORDS.items():
            for w in sorted(words, key=len, reverse=True):
                hits = list(re.finditer(rf"\b{re.escape(w)}\b", s))
                if not hits:
                    continue
                for m in hits:
                    a, b = m.span()
                    if key == "cholesterol" and re.search(r"\b(bad|good|ldl|hdl)\s+$", s[:a]):
                        continue
                    if key == "hemoglobin" and re.match(r"\s*a1c", s[b:]):
                        continue
                    before = re.split(_CONJ, s[max(0, a - 40):a])[-1]
                    after = re.split(_CONJ, s[b:b + 40])[0]
                    ctx = f"{before} | {after}"
                    if key in TREND_KEYS and key in LOWER_IS_WORSE and re.search(_WORSE_BETTER, ctx):
                        # "worsening kidney function" = eGFR falling; "improving" = rising
                        out["trends"][key] = "falling" if re.search(r"wors|deteriorat|getting worse", ctx) else "rising"
                    elif key in TREND_KEYS:
                        if re.search(rf"\b{_RISING}\b", ctx):
                            out["trends"][key] = "rising"
                        elif re.search(rf"\b{_FALLING}\b", ctx):
                            out["trends"][key] = "falling"
                    if re.search(rf"\b{_BORDER}\b", ctx):
                        out["labs"][key] = "borderline"
                    elif re.search(rf"\b{_HIGH}\b", ctx):
                        out["labs"][key] = "high"
                    elif re.search(rf"\b{_LOW}\b", ctx):
                        out["labs"][key] = "low"
                    elif re.search(rf"\b{_NORMAL}\b", ctx):
                        out["labs"][key] = "normal"
                break  # the longest matching synonym wins for this lab
        out["text"] = q.strip()
        return ParsedQuery(**out)


# ----------------------------------------------------------------------------- llm
SYSTEM_PROMPT = """You convert a clinician's plain-language patient search into JSON filters.
Return ONLY a JSON object with these optional keys:
  "gender": "female" | "male"
  "age_min": number, "age_max": number            (e.g. "over 60" -> age_min 60; "in their 40s" -> 40-49)
  "insurance": "medicare" | "medicaid" | "private" | "uninsured" | "dual"
  "smoking": "current" | "former" | "never"
  "labs":   {lab: "high" | "low" | "borderline" | "normal"}
  "trends": {lab: "rising" | "falling"}
Allowed lab keys: bmi, systolic (blood pressure), diastolic, heart_rate, glucose (blood sugar), hba1c, cholesterol,
ldl (bad cholesterol), hdl (good cholesterol), triglycerides, creatinine, egfr (kidney function), hemoglobin,
microalbumin, fev1_fvc (lung function), pain, phq2 (depression), gad7 (anxiety).
Trends are only allowed for: bmi, systolic, glucose, hba1c, egfr, cholesterol.
Rules: obese/overweight -> labs.bmi "high"; hypertensive/high blood pressure -> labs.systolic "high";
anemic -> labs.hemoglobin "low"; diabetic or high blood sugar -> labs.glucose "high".
Only include what the text actually asks for. Never invent values. Diagnoses that are not lab states
(e.g. asthma, migraine) are NOT filters: leave them out.
Example: "women over 60 with high BMI and rising glucose" ->
{"gender": "female", "age_min": 60, "labs": {"bmi": "high"}, "trends": {"glucose": "rising"}}"""


class LLMParser:
    def __init__(self, llm, fallback: RuleParser | None = None):
        self.llm = llm
        self.fallback = fallback or RuleParser()
        self.name = f"llm:{llm.name}"

    def parse(self, q: str) -> ParsedQuery:
        raw = self.llm.complete_json(SYSTEM_PROMPT, q)
        raw = {k: v for k, v in raw.items() if k in ParsedQuery.model_fields and v not in (None, "", {})}
        raw["text"] = q.strip()
        return ParsedQuery(**raw)


def get_parser():
    llm = get_llm()
    return LLMParser(llm) if llm is not None else RuleParser()


def parse_query(q: str, parser=None) -> tuple[ParsedQuery, dict]:
    """Parse with the LLM when configured; on any LLM/validation failure use the rules."""
    parser = parser or get_parser()
    t = time.perf_counter()
    try:
        pq = parser.parse(q)
        source = parser.name
    except (LLMError, ValueError) as e:
        log.warning("LLM parse failed (%s); using rules", e)
        pq = RuleParser().parse(q)
        source = "rules (LLM fallback)"
    return pq, {"parser": source, "parse_ms": round((time.perf_counter() - t) * 1000, 1)}


# ----------------------------------------------------------------------------- search
def satisfies(doc: dict, kind: str, key: str, value: str) -> bool:
    s = doc["structured"]
    if kind == "lab":
        v = s.get("latest", {}).get(key)
        if not v:
            return False
        return v["flag"] in LEVEL_FLAGS[value]
    if kind == "trend":
        t = s.get("trends", {}).get(key)
        return bool(t and t["direction"] == value)
    if kind == "smoking":
        sm = (s.get("smoking") or "").lower()
        if value == "never":
            return "never" in sm
        if value == "former":
            return "ex-" in sm or "former" in sm
        return bool(sm) and "never" not in sm and "ex-" not in sm and "former" not in sm
    return False


def run_query(engine, pq: ParsedQuery, k: int = 15, strict: bool = True, pool: int = 300) -> dict:
    """Hybrid search driven by the parsed filters. Strict: every condition must hold
    (patients meeting all conditions first, then partial matches only if too few)."""
    from .index import embed
    from .search import rrf
    from . import config

    t = time.perf_counter()
    idx = engine.index
    flt = pq.filters()
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]*", pq.text.lower())]
    bm = idx._bm25_rank(words + pq.bm25_tokens(), pool, flt)
    vec = idx.vector_search(embed([pq.sentence() or pq.text])[0], pool, flt)
    fused = sorted(rrf([[p for p, _ in bm], [p for p, _ in vec]], weights=[config.RRF_W_BM25, config.RRF_W_VEC]).items(),
                   key=lambda x: -x[1])
    cons = pq.constraints()
    docs = idx.all_docs()
    scored = []
    for pid, s in fused:
        d = docs[pid]
        met = [c for c in cons if satisfies(d, *c)]
        scored.append((len(met), s, pid, met))
    full = [x for x in scored if x[0] == len(cons)]
    if strict:
        # everyone meeting all conditions first (by search score), then the closest partial matches
        chosen = sorted(scored, key=lambda x: (-x[0], -x[1]))[:k]
    else:
        # soft: search score decides, conditions only nudge
        top = max((x[1] for x in scored), default=1.0) or 1.0
        chosen = sorted(scored, key=lambda x: -(x[1] / top + 0.15 * x[0] / max(1, len(cons))))[:k]
    results = []
    for n_met, s, pid, met in chosen:
        d = docs[pid]
        results.append({"short_id": d["short_id"], "age": d["age"], "gender": d["gender"], "insurance": d.get("insurance"),
                        "deceased": d["structured"].get("deceased", False), "chronic_conditions": d.get("chronic_conditions", []),
                        "matches_all": n_met == len(cons), "met": [_label(c) for c in met],
                        "missing": [_label(c) for c in cons if c not in met], "score": round(s, 5)})
    return {"parsed": pq.model_dump(exclude={"text"}), "chips": pq.chips(), "results": results,
            "n_full_matches": len(full), "strict": strict, "search_ms": round((time.perf_counter() - t) * 1000, 1)}


def _label(c) -> str:
    kind, key, value = c
    if kind == "smoking":
        return {"current": "smoker", "former": "ex-smoker", "never": "never smoked"}[value]
    return f"{LAB_LABEL.get(key, key)} {value}"
