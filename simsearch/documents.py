"""Build one patient document per patient from the multi-table Synthea EHR export.

Each document has:
  * ``text``       – a short natural-language summary used for search (demographics,
                     latest vitals and labs, trends, encounter history). Diagnoses,
                     medications, procedures and encounter reasons are deliberately
                     left OUT so they can serve as an unbiased answer key.
  * ``structured`` – the same facts as JSON (used by the learned ranker and the
                     risk model).
  * ``conditions`` / ``chronic_conditions`` – the answer key, never searched.

Two snapshots are produced, mirroring the temporal split of Assignment 2:
  * period 1 – everything recorded before TEMPORAL_SPLIT_DATE
  * period 2 – the full record (new patients + updated records of old ones)

Data-quality note: ``conditions.csv`` in this export is truncated (~108 patients),
so diagnoses are reconstructed from conditions + the SNOMED reason codes attached
to encounters, medications, procedures and care plans.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

READ_KW = dict(on_bad_lines="skip", low_memory=False, encoding="utf-8")

# ---------------------------------------------------------------------------
# Lab / vital definitions: (key, label, descriptions, unit, flag function)
# ---------------------------------------------------------------------------


def _flag_range(low=None, high=None, borderline=None):
    def f(v, age):
        if high is not None and v >= high:
            return "high"
        if borderline is not None and v >= borderline:
            return "borderline"
        if low is not None and v < low:
            return "low"
        return None

    return f


def _flag_bmi(v, age):
    if age is not None and age < 18:
        return None
    if v >= 30:
        return "obese"
    if v >= 25:
        return "overweight"
    if v < 18.5:
        return "underweight"
    return None


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    descriptions: tuple
    unit: str
    flag: object = None
    group: str = "labs"


MEASURES = [
    Measure("bmi", "BMI", ("Body mass index (BMI) [Ratio]",), "", _flag_bmi, "vitals"),
    Measure("weight", "weight", ("Body Weight",), "kg", None, "vitals"),
    Measure("systolic", "systolic BP", ("Systolic Blood Pressure",), "mmHg", _flag_range(high=140, borderline=130), "vitals"),
    Measure("diastolic", "diastolic BP", ("Diastolic Blood Pressure",), "mmHg", _flag_range(high=90, borderline=80), "vitals"),
    Measure("heart_rate", "heart rate", ("Heart rate",), "bpm", _flag_range(low=50, high=100), "vitals"),
    Measure("resp_rate", "respiratory rate", ("Respiratory rate",), "/min", _flag_range(high=20), "vitals"),
    Measure("glucose", "glucose", ("Glucose [Mass/volume] in Blood", "Glucose [Mass/volume] in Serum or Plasma"), "mg/dL", _flag_range(low=70, high=126, borderline=100)),
    Measure("hba1c", "HbA1c", ("Hemoglobin A1c/Hemoglobin.total in Blood",), "%", _flag_range(high=6.5, borderline=5.7)),
    Measure("cholesterol", "total cholesterol", ("Cholesterol [Mass/volume] in Serum or Plasma",), "mg/dL", _flag_range(high=240, borderline=200)),
    Measure("ldl", "LDL", ("Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay",), "mg/dL", _flag_range(high=160, borderline=130)),
    Measure("hdl", "HDL", ("Cholesterol in HDL [Mass/volume] in Serum or Plasma",), "mg/dL", _flag_range(low=40)),
    Measure("triglycerides", "triglycerides", ("Triglyceride [Mass/volume] in Serum or Plasma",), "mg/dL", _flag_range(high=200, borderline=150)),
    Measure("creatinine", "creatinine", ("Creatinine [Mass/volume] in Blood", "Creatinine [Mass/volume] in Serum or Plasma"), "mg/dL", _flag_range(high=1.3)),
    Measure("egfr", "eGFR", ("Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M",), "mL/min", _flag_range(low=60)),
    Measure("hemoglobin", "hemoglobin", ("Hemoglobin [Mass/volume] in Blood",), "g/dL", _flag_range(low=12, high=17.5)),
    Measure("leukocytes", "WBC", ("Leukocytes [#/volume] in Blood by Automated count",), "10^3/uL", _flag_range(low=4, high=11)),
    Measure("microalbumin", "urine albumin/creatinine", ("Microalbumin/Creatinine [Mass Ratio] in Urine",), "mg/g", _flag_range(high=30)),
    Measure("fev1_fvc", "FEV1/FVC", ("FEV1/FVC",), "%", _flag_range(low=70)),
    Measure("pain", "pain score", ("Pain severity - 0-10 verbal numeric rating [Score] - Reported",), "/10", _flag_range(high=7), "scores"),
    Measure("phq2", "PHQ-2", ("Patient Health Questionnaire 2 item (PHQ-2) total score [Reported]",), "", _flag_range(high=3), "scores"),
    Measure("gad7", "GAD-7", ("Generalized anxiety disorder 7 item (GAD-7) total score [Reported.PHQ]",), "", _flag_range(high=10), "scores"),
    Measure("audit_c", "AUDIT-C", ("Total score [AUDIT-C]",), "", _flag_range(high=4), "scores"),
]
DESC_TO_KEY = {d: m.key for m in MEASURES for d in m.descriptions}
MEASURE_BY_KEY = {m.key: m for m in MEASURES}
TREND_KEYS = ["bmi", "systolic", "glucose", "hba1c", "egfr", "cholesterol"]
ENCOUNTER_CLASSES = ["wellness", "ambulatory", "outpatient", "urgentcare", "emergency", "inpatient"]

# Diagnoses considered "acute / incidental" and therefore NOT part of the chronic
# answer key (sharing a cold or a sprained ankle does not make two patients similar).
ACUTE_PATTERN = re.compile(
    r"acute|viral|sinusitis|pharyngitis|bronchitis|gingiv|dental|caries|tooth|teeth|"
    r"fracture|sprain|laceration|injury|concussion|whiplash|contusion|dislocation|burn|"
    r"miscarriage|ovum|pregnan|ectopic|otitis|cystitis|streptococcal|appendicitis|"
    r"coronavirus|covid|pneumonia|wound|bite|abrasion|infection of|tear of|rupture|"
    r"sepsis|septic|drug overdose|poisoning|preeclampsia|eclampsia|neutropenia|"
    r"respiratory distress|hypoxemia|cardiac arrest|food allergy|allerg|overdose|molar|"
    r"alveolitis|torus|sudden cardiac|salivation|bleeding|pyelonephritis|fetus|"
    r"congenital uterine",
    re.IGNORECASE,
)
RISK_TARGET_KEYWORDS = ("diabetes", "hypertension", "prediabetes", "metabolic syndrome")

# Social-determinant (PRAPARE) answers -> short phrases. None = leave out of the text.
DECLINED = "I choose not to answer this question"
SOCIAL_ITEMS = {
    "Housing status": ("housing", lambda v: "housed" if v == "I have housing" else "homeless"),
    "Employment status - current": ("employment", lambda v: {
        "Full-time work": "full-time work", "Part-time or temporary work": "part-time work",
        "Unemployed": "unemployed"}.get(v, "not in labor force")),
    "Highest level of education": ("education", lambda v: {
        "More than high school": "college", "High school diploma or GED": "high school",
        "Less than high school degree": "no high school"}.get(v)),
    "Stress level": ("stress", lambda v: "stress " + {"Not at all": "none", "A little bit": "low", "Somewhat": "moderate",
                                                       "Quite a bit": "high", "Very much": "very high"}.get(v, "unknown")),
    "How often do you see or talk to people that you care about and feel close to [PRAPARE]":
        ("social_contact", lambda v: "socially isolated" if v == "Less than once a week" else None),
    "Has lack of transportation kept you from medical appointments  meetings  work  or from getting things "
    "needed for daily living": ("transport", lambda v: None if v == "No" else "transport barrier"),
    "Do you feel physically and emotionally safe where you currently live [PRAPARE]":
        ("safety", lambda v: "feels unsafe at home" if v == "No" else None),
    "Are you a refugee": ("refugee", lambda v: "refugee" if v == "Yes" else None),
}
SOCIAL_ORDER = ["housing", "employment", "education", "stress", "social_contact", "transport", "safety", "refugee"]
INSURANCE_ANSWERS = {"Medicare": "medicare", "Medicaid": "medicaid", "Private insurance": "private",
                     "None/uninsured": "uninsured"}
INSURANCE_TYPES = ("medicare", "medicaid", "private", "uninsured", "dual")


def short_id(patient_id: str) -> str:
    return "P-" + patient_id.replace("-", "")[:8].upper()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _to_dt(s: pd.Series, fmt: str | None = None) -> pd.Series:
    if fmt:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
    else:
        out = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        out = out.dt.tz_localize(None)
    return out


def load_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    d = Path(data_dir or config.DATA_DIR)
    patients = pd.read_csv(d / "patients.csv", **READ_KW)
    patients = patients[patients["Id"].astype(str).str.len() == 36].copy()
    patients["BIRTHDATE"] = _to_dt(patients["BIRTHDATE"])
    patients["DEATHDATE"] = _to_dt(patients["DEATHDATE"])
    for c in ("HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"):
        patients[c] = pd.to_numeric(patients[c], errors="coerce")

    enc = pd.read_csv(d / "encounters.csv", usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS",
                                                      "TOTAL_CLAIM_COST", "REASONCODE", "REASONDESCRIPTION"], **READ_KW)
    enc["DATE"] = _to_dt(enc["START"])
    enc["TOTAL_CLAIM_COST"] = pd.to_numeric(enc["TOTAL_CLAIM_COST"], errors="coerce")

    obs = pd.read_csv(d / "observations.csv", usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE", "TYPE"], **READ_KW)
    smoking = obs[obs["DESCRIPTION"] == "Tobacco smoking status"][["DATE", "PATIENT", "VALUE"]].copy()
    social = obs[obs["DESCRIPTION"].isin(SOCIAL_ITEMS) | (obs["DESCRIPTION"] == "Primary insurance")]
    social = social[social["VALUE"] != DECLINED][["DATE", "PATIENT", "DESCRIPTION", "VALUE"]].copy()
    social["DATE"] = _to_dt(social["DATE"])
    obs = obs[obs["DESCRIPTION"].isin(DESC_TO_KEY)].copy()
    obs["KEY"] = obs["DESCRIPTION"].map(DESC_TO_KEY)
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["VALUE"])
    obs["DATE"] = _to_dt(obs["DATE"])
    smoking["DATE"] = _to_dt(smoking["DATE"])

    # Diagnoses (answer key): conditions + reason codes from other tables + claim
    # diagnosis codes (claims.csv DIAGNOSIS1-8), mapped to descriptions via the
    # code/description pairs found in the other tables.
    frames, code_pairs = [], []
    try:
        cond = pd.read_csv(d / "conditions.csv", usecols=["START", "PATIENT", "CODE", "DESCRIPTION"], **READ_KW)
        cond["DATE"] = _to_dt(cond["START"], "%d-%m-%Y")
        frames.append(cond[["PATIENT", "DATE", "DESCRIPTION"]])
        code_pairs.append(cond[["CODE", "DESCRIPTION"]].set_axis(["code", "desc"], axis=1))
    except Exception:  # pragma: no cover - conditions file is known to be damaged
        pass
    frames.append(enc.rename(columns={"REASONDESCRIPTION": "DESCRIPTION"})[["PATIENT", "DATE", "DESCRIPTION"]])
    code_pairs.append(enc[["REASONCODE", "REASONDESCRIPTION"]].set_axis(["code", "desc"], axis=1))
    for name in ("medications", "procedures", "careplans"):
        t = pd.read_csv(d / f"{name}.csv", usecols=["START", "PATIENT", "REASONCODE", "REASONDESCRIPTION"], **READ_KW)
        t["DATE"] = _to_dt(t["START"])
        frames.append(t.rename(columns={"REASONDESCRIPTION": "DESCRIPTION"})[["PATIENT", "DATE", "DESCRIPTION"]])
        code_pairs.append(t[["REASONCODE", "REASONDESCRIPTION"]].set_axis(["code", "desc"], axis=1))
    pairs = pd.concat(code_pairs).dropna()
    pairs["code"] = _norm_code(pairs["code"])
    code2desc = pairs.drop_duplicates("code").set_index("code")["desc"]
    claims_path = d / "claims.csv"
    if claims_path.exists():
        dcols = [f"DIAGNOSIS{i}" for i in range(1, 9)]
        cl = pd.read_csv(claims_path, usecols=["PATIENTID", "SERVICEDATE"] + dcols, **READ_KW)
        cl = cl.melt(id_vars=["PATIENTID", "SERVICEDATE"], value_vars=dcols, value_name="code").dropna(subset=["code"])
        cl["DESCRIPTION"] = _norm_code(cl["code"]).map(code2desc)
        cl["DATE"] = _to_dt(cl["SERVICEDATE"])
        frames.append(cl.rename(columns={"PATIENTID": "PATIENT"})[["PATIENT", "DATE", "DESCRIPTION"]])
    dx = pd.concat(frames, ignore_index=True).dropna()
    dx = dx[dx["PATIENT"].isin(patients["Id"])]
    dx = dx[dx["DESCRIPTION"].str.contains(r"\((?:disorder|finding)\)", regex=True)]

    payers = pd.read_csv(d / "payer_transitions.csv", usecols=["PATIENT", "START_DATE", "END_DATE", "PAYER"], **READ_KW)
    payers["START"] = _to_dt(payers["START_DATE"])
    payers["END"] = _to_dt(payers["END_DATE"])
    payers["INSURANCE"] = payers["PAYER"].map(_payer_categories(payers, social))

    return {"patients": patients, "encounters": enc, "observations": obs, "smoking": smoking, "diagnoses": dx,
            "social": social, "payers": payers[["PATIENT", "START", "END", "INSURANCE"]]}


def _norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _payer_categories(payers: pd.DataFrame, social: pd.DataFrame) -> dict:
    """Payer IDs are anonymous; label each one by majority vote of the patients'
    self-reported "Primary insurance" answer while that payer was active."""
    ans = social[social["DESCRIPTION"] == "Primary insurance"].copy()
    ans["CAT"] = ans["VALUE"].map(INSURANCE_ANSWERS)
    m = ans.dropna(subset=["CAT"]).merge(payers, on="PATIENT")
    m = m[(m["DATE"] >= m["START"]) & ((m["DATE"] < m["END"]) | m["END"].isna())]
    out = {}
    for payer, cats in m.groupby("PAYER")["CAT"]:
        vc = cats.value_counts()
        if vc.iloc[0] / vc.sum() >= 0.9:
            out[payer] = vc.index[0]
        elif set(vc.index[:2]) == {"medicare", "medicaid"}:
            out[payer] = "dual"
    return out


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _trend_word(first: float, last: float) -> str:
    if first == 0:
        return "stable"
    change = (last - first) / abs(first)
    if change > 0.05:
        return "rising"
    if change < -0.05:
        return "falling"
    return "stable"


def is_chronic(description: str) -> bool:
    return description.endswith("(disorder)") and not ACUTE_PATTERN.search(description)


def clean_condition(description: str) -> str:
    return re.sub(r"\s*\((disorder|finding|situation)\)\s*$", "", description).strip()


def build_documents(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp, period: int) -> list[dict]:
    """Snapshot of every patient with at least one encounter before ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    pts = tables["patients"].set_index("Id")
    enc = tables["encounters"][tables["encounters"]["DATE"] < cutoff]
    obs = tables["observations"][tables["observations"]["DATE"] < cutoff].sort_values("DATE")
    smk = tables["smoking"][tables["smoking"]["DATE"] < cutoff].sort_values("DATE")
    dx = tables["diagnoses"][tables["diagnoses"]["DATE"] < cutoff]

    enc_groups = dict(tuple(enc.groupby("PATIENT")))
    obs_groups = dict(tuple(obs.groupby("PATIENT")))
    smk_last = smk.groupby("PATIENT")["VALUE"].last().to_dict()
    dx_groups = dx.groupby("PATIENT")["DESCRIPTION"].apply(lambda s: sorted(set(s))).to_dict()
    soc = tables["social"][tables["social"]["DATE"] < cutoff].sort_values("DATE")
    soc_last = soc.groupby(["PATIENT", "DESCRIPTION"])["VALUE"].last()
    social_by_pid: dict = {}
    for (pid_, desc), val in soc_last.items():
        social_by_pid.setdefault(pid_, {})[desc] = val
    pay = tables["payers"]
    pay = pay[pay["START"] < cutoff].sort_values("START").groupby("PATIENT")["INSURANCE"].last().to_dict()

    docs = []
    for pid, e in enc_groups.items():
        if pid not in pts.index:
            continue
        p = pts.loc[pid]
        ref = cutoff if pd.isna(p["DEATHDATE"]) or p["DEATHDATE"] > cutoff else p["DEATHDATE"]
        age = float((ref - p["BIRTHDATE"]).days / 365.25) if pd.notna(p["BIRTHDATE"]) else None
        gender = "female" if p["GENDER"] == "F" else "male"
        deceased = bool(pd.notna(p["DEATHDATE"]) and p["DEATHDATE"] <= cutoff)

        # --- measurements -------------------------------------------------
        o = obs_groups.get(pid)
        latest, trends = {}, {}
        if o is not None:
            g = o.groupby("KEY")["VALUE"]
            last, first, count = g.last(), g.first(), g.count()
            for k, v in last.items():
                m = MEASURE_BY_KEY[k]
                latest[k] = {"value": round(float(v), 2), "unit": m.unit,
                             "flag": m.flag(v, age) if m.flag else None}
            for k in TREND_KEYS:
                if k in last and count.get(k, 0) >= 2:
                    trends[k] = {"first": round(float(first[k]), 2), "last": round(float(last[k]), 2),
                                 "direction": _trend_word(first[k], last[k])}

        enc_counts = e["ENCOUNTERCLASS"].value_counts().to_dict()
        years = (e["DATE"].max() - e["DATE"].min()).days / 365.25
        dxs = dx_groups.get(pid, [])
        disorders = sorted({clean_condition(x) for x in dxs if x.endswith("(disorder)")})
        chronic = sorted({clean_condition(x) for x in dxs if is_chronic(x)})
        risk_target = int(any(k in x.lower() for x in dxs for k in RISK_TARGET_KEYWORDS))
        answers = social_by_pid.get(pid, {})
        social_vals = {}
        for desc, (key, fn) in SOCIAL_ITEMS.items():
            if desc in answers:
                phrase = fn(answers[desc])
                if phrase:
                    social_vals[key] = phrase
        insurance = pay.get(pid)
        if not isinstance(insurance, str):
            insurance = INSURANCE_ANSWERS.get(answers.get("Primary insurance"))

        structured = {
            "age": round(age, 1) if age is not None else None,
            "gender": gender,
            "race": str(p["RACE"]),
            "ethnicity": str(p["ETHNICITY"]),
            "deceased": deceased,
            "smoking": smk_last.get(pid),
            "insurance": insurance,
            "social": social_vals,
            "income": None if pd.isna(p["INCOME"]) else float(p["INCOME"]),
            "healthcare_expenses": None if pd.isna(p["HEALTHCARE_EXPENSES"]) else float(p["HEALTHCARE_EXPENSES"]),
            "healthcare_coverage": None if pd.isna(p["HEALTHCARE_COVERAGE"]) else float(p["HEALTHCARE_COVERAGE"]),
            "n_encounters": int(len(e)),
            "encounters_by_class": {k: int(v) for k, v in enc_counts.items()},
            "total_claim_cost": round(float(e["TOTAL_CLAIM_COST"].sum()), 2),
            "years_of_history": round(years, 1),
            "latest": latest,
            "trends": trends,
        }
        doc = {
            "patient_id": pid,
            "short_id": short_id(pid),
            "period": period,
            "snapshot_date": cutoff.date().isoformat(),
            "updated_at": e["DATE"].max().date().isoformat(),
            "age": structured["age"],
            "gender": gender,
            "insurance": insurance,
            "structured": structured,
            "conditions": disorders,
            "chronic_conditions": chronic,
            "risk_target": risk_target,
        }
        doc["text"] = render_text(doc)
        doc["content_hash"] = hashlib.sha1(doc["text"].encode()).hexdigest()
        docs.append(doc)
    return docs


def render_text(doc: dict) -> str:
    """Compact summary (units omitted, flags kept) so it fits a 256-token encoder."""
    s = doc["structured"]
    age = s["age"]
    parts = [f"{int(age) if age is not None else 'unknown'}-year-old {s['gender']}, {s['race']}, {s['ethnicity']}"
             + (", deceased" if s["deceased"] else "") + "."]
    if s.get("smoking"):
        parts.append(f"Smoking: {clean_condition(s['smoking']).lower()}.")
    if s.get("insurance"):
        parts.append(f"Insurance: {s['insurance']}.")
    soc = s.get("social") or {}
    if soc:
        parts.append("Social: " + ", ".join(soc[k] for k in SOCIAL_ORDER if k in soc) + ".")

    def block(group: str, title: str):
        items = []
        lat = s["latest"]
        both_bp = "systolic" in lat and "diastolic" in lat
        if group == "vitals" and both_bp:
            sy, di = lat["systolic"], lat["diastolic"]
            flag = sy["flag"] or di["flag"]
            items.append(f"BP {_fmt(sy['value'])}/{_fmt(di['value'])}" + (f" {flag}" if flag else ""))
        for m in MEASURES:
            if m.group != group or m.key not in lat or m.key == "weight":
                continue
            if m.key in ("systolic", "diastolic") and both_bp:
                continue
            v = lat[m.key]
            items.append(f"{m.label} {_fmt(v['value'])}" + (f" {v['flag']}" if v["flag"] else ""))
        if items:
            parts.append(f"{title}: " + ", ".join(items) + ".")

    block("vitals", "Vitals")
    block("labs", "Labs")
    if s["trends"]:
        tr = [f"{MEASURE_BY_KEY[k].label} {t['direction']}" + (f" {_fmt(t['first'])}->{_fmt(t['last'])}"
                                                                 if t["direction"] != "stable" else "")
              for k, t in s["trends"].items()]
        parts.append("Trends: " + ", ".join(tr) + ".")
    block("scores", "Scores")
    enc = ", ".join(f"{c} {s['encounters_by_class'][c]}" for c in ENCOUNTER_CLASSES if c in s["encounters_by_class"])
    parts.append(f"History: {s['years_of_history']:g} years, {s['n_encounters']} visits ({enc}).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_jsonl(docs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, default=_json_default) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def build_both_periods(data_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    tables = load_tables(data_dir)
    split = pd.Timestamp(config.TEMPORAL_SPLIT_DATE)
    end = tables["encounters"]["DATE"].max() + pd.Timedelta(days=1)
    return build_documents(tables, split, 1), build_documents(tables, end, 2)
