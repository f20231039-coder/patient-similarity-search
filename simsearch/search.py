"""Patient search: BM25 + vector search -> reciprocal rank fusion -> learned ranker.

The product uses one search, the default ("ltr"): BM25 top-50 and vector top-50
are merged with reciprocal rank fusion, then re-ordered by a LightGBM LambdaRank
model that also uses structured similarity (age, labs, vitals, trends).

The intermediate stages ("bm25", "vector", "hybrid") stay selectable only so the
evaluation can measure what each stage adds.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal, Optional

import joblib
import numpy as np

from . import config
from .documents import MEASURES
from .index import Filters, PatientIndex, embed

log = logging.getLogger(__name__)
Stage = Literal["bm25", "vector", "hybrid", "ltr"]
STAGES: tuple[str, ...] = ("bm25", "vector", "hybrid", "ltr")
DEFAULT = "ltr"
_pool = ThreadPoolExecutor(max_workers=4)


def rrf(rankings: list[list[str]], k: int | None = None, weights: list[float] | None = None) -> dict[str, float]:
    """(Weighted) reciprocal rank fusion: score(d) = sum over lists of w / (k + rank)."""
    k = config.RRF_K if k is None else k
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
    return scores


# ---------------------------------------------------------------- ranker features
LAB_KEYS = [m.key for m in MEASURES if m.key not in ("weight",)]
# Rough population scales so differences are comparable across labs.
LAB_SCALE = {"bmi": 6, "systolic": 15, "diastolic": 10, "heart_rate": 12, "resp_rate": 3, "glucose": 20,
             "hba1c": 0.8, "cholesterol": 35, "ldl": 30, "hdl": 12, "triglycerides": 50, "creatinine": 0.4,
             "egfr": 20, "hemoglobin": 1.5, "leukocytes": 2, "microalbumin": 30, "fev1_fvc": 10, "pain": 2,
             "phq2": 1.5, "gad7": 4, "audit_c": 2}
KEY_LABS = ["systolic", "diastolic", "bmi", "glucose", "hba1c", "cholesterol", "ldl", "hdl", "triglycerides",
            "creatinine", "egfr", "hemoglobin", "microalbumin", "fev1_fvc"]


def _structured(q: dict, c: dict) -> list[float]:
    qs, cs = q["structured"], c["structured"]
    feats = [
        abs((qs.get("age") or 0) - (cs.get("age") or 0)),
        float(qs.get("gender") == cs.get("gender")),
        float(qs.get("deceased") == cs.get("deceased")),
        float(qs.get("smoking") == cs.get("smoking")),
        abs(np.log1p(qs["n_encounters"]) - np.log1p(cs["n_encounters"])),
    ]
    flag_match = flag_mismatch = 0
    for k in LAB_KEYS:
        a, b = qs["latest"].get(k), cs["latest"].get(k)
        if a and b:
            feats.append(min(abs(a["value"] - b["value"]) / LAB_SCALE.get(k, 1.0), 10.0))
            feats.append(0.0)
            if a["flag"] or b["flag"]:
                flag_match += a["flag"] == b["flag"]
                flag_mismatch += a["flag"] != b["flag"]
        else:
            feats.append(0.0)
            feats.append(1.0 if (a or b) else 0.5)  # one side missing vs both missing
    for cls in ("emergency", "inpatient", "outpatient"):
        feats.append(abs(np.log1p(qs["encounters_by_class"].get(cls, 0)) - np.log1p(cs["encounters_by_class"].get(cls, 0))))
    feats += [flag_match, flag_mismatch]
    return feats


def ranker_features(q: dict, c: dict, rrf_score: float, bm25_rank, vec_rank, vec_score, bm25_score, bm25_max) -> list[float]:
    """Fusion signals + structured similarity + per-lab flag agreement (+1 same abnormal
    flag, -1 abnormal in only one) + shared trends, social context and insurance."""
    big = config.CANDIDATES_PER_RETRIEVER + 1
    feats = [rrf_score, bm25_rank or big, vec_rank or big, vec_score if vec_score is not None else 0.0,
             float(bm25_rank is not None and vec_rank is not None)] + _structured(q, c)
    qs, cs = q["structured"], c["structured"]
    ql, cl = qs.get("latest", {}), cs.get("latest", {})
    agree = []
    for k in KEY_LABS:
        a, b = ql.get(k), cl.get(k)
        fa = a and a["flag"] not in (None, "borderline")
        fb = b and b["flag"] not in (None, "borderline")
        agree.append(1.0 if (fa and fb and a["flag"] == b["flag"]) else -1.0 if (fa != fb and (a and b)) else 0.0)
    qt, ct = qs.get("trends", {}), cs.get("trends", {})
    trends = sum(1 for k, t in qt.items() if t["direction"] != "stable" and ct.get(k, {}).get("direction") == t["direction"])
    social = len(set((qs.get("social") or {}).values()) & set((cs.get("social") or {}).values()))
    return feats + agree + [(bm25_score or 0.0) / (bm25_max or 1.0), float(sum(x > 0 for x in agree)),
                            float(sum(x < 0 for x in agree)), float(trends), float(social),
                            float(qs.get("insurance") == cs.get("insurance")),
                            (qs.get("age") or 0) - (cs.get("age") or 0)]


@lru_cache(maxsize=1)
def get_ranker():
    """The trained LambdaRank model ({'model': LGBMRanker, ...}), or None if not trained."""
    if not config.LTR_MODEL_PATH.exists():
        return None
    return joblib.load(config.LTR_MODEL_PATH)


# ---------------------------------------------------------------- engine
class SearchEngine:
    def __init__(self, index: PatientIndex):
        self.index = index

    def _candidates(self, text, qvec, k, filters, exclude, timings, qdoc=None):
        def run_bm25():
            t = time.perf_counter()
            r = self.index.bm25_search(text, k, filters, exclude, query_doc=qdoc)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
            return r

        def run_vec():
            t = time.perf_counter()
            r = self.index.vector_search(qvec, k, filters, exclude)
            timings["vector_ms"] = (time.perf_counter() - t) * 1000
            return r

        fb, fv = _pool.submit(run_bm25), _pool.submit(run_vec)
        return fb.result(), fv.result()

    def search(self, query: str | None = None, patient_id: str | None = None, k: int = 10,
               filters: Filters | None = None, mode: Stage = DEFAULT, query_doc: dict | None = None) -> dict:
        """Return {results, timings, query_patient, mode}. Query by free text, by an
        indexed patient (whose summary becomes the query and who is excluded), or by
        a full patient document (``query_doc``). The learned ranker needs a query
        patient and a trained model; otherwise the search falls back to hybrid."""
        if mode not in STAGES:
            raise ValueError(f"unknown search stage {mode}")
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        qdoc, exclude, use_stored_vector = None, None, False
        if query_doc is not None:
            qdoc, exclude, text = query_doc, query_doc["patient_id"], query_doc["text"]
        elif patient_id:
            use_stored_vector = True
            qdoc = self.index.get(patient_id)
            if qdoc is None:
                raise KeyError(f"patient {patient_id} not found in index")
            exclude, text = qdoc["patient_id"], qdoc["text"]
        elif query:
            text = query
        else:
            raise ValueError("provide query or patient_id")
        if mode == "ltr" and (qdoc is None or get_ranker() is None):
            mode = "hybrid"

        n = config.CANDIDATES_PER_RETRIEVER
        if mode == "bm25":
            t = time.perf_counter()
            hits = self.index.bm25_search(text, k, filters, exclude, query_doc=qdoc)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
            ranked = [(pid, s, {}) for pid, s in hits]
        else:
            t = time.perf_counter()
            qvec = self.index.vector_by_id(exclude) if use_stored_vector else None
            if qvec is None:
                qvec = embed([text])[0]
            timings["embed_ms"] = (time.perf_counter() - t) * 1000
            if mode == "vector":
                t = time.perf_counter()
                hits = self.index.vector_search(qvec, k, filters, exclude)
                timings["vector_ms"] = (time.perf_counter() - t) * 1000
                ranked = [(pid, s, {}) for pid, s in hits]
            else:
                bm, vec = self._candidates(text, qvec, n, filters, exclude, timings, qdoc)
                t = time.perf_counter()
                bm_rank = {pid: r for r, (pid, _) in enumerate(bm, 1)}
                vec_rank = {pid: r for r, (pid, _) in enumerate(vec, 1)}
                vec_score = dict(vec)
                fused = rrf([[p for p, _ in bm], [p for p, _ in vec]], weights=[config.RRF_W_BM25, config.RRF_W_VEC])
                pool = sorted(fused.items(), key=lambda x: -x[1])[:n]
                timings["fusion_ms"] = (time.perf_counter() - t) * 1000
                if mode == "hybrid":
                    ranked = [(pid, s, {}) for pid, s in pool[:k]]
                else:
                    t = time.perf_counter()
                    docs = self.index.all_docs()
                    bm_score = dict(bm)
                    bm_max = max(bm_score.values(), default=1.0) or 1.0
                    X = np.array([ranker_features(qdoc, docs[pid], s, bm_rank.get(pid), vec_rank.get(pid),
                                                  vec_score.get(pid), bm_score.get(pid), bm_max) for pid, s in pool])
                    sc = get_ranker()["model"].predict(X) if len(X) else []
                    timings["ranker_ms"] = (time.perf_counter() - t) * 1000
                    order = np.argsort(-np.asarray(sc))[:k]
                    ranked = [(pool[i][0], float(sc[i]), {}) for i in order]

        docs = self.index.all_docs()
        results = []
        for rank, (pid, score, extra) in enumerate(ranked, 1):
            d = docs[pid]
            results.append({"rank": rank, "patient_id": pid, "short_id": d["short_id"], "score": round(score, 5),
                            "age": d["age"], "gender": d["gender"], "insurance": d.get("insurance"),
                            "period": d["period"], "updated_at": d["updated_at"], "summary": d["text"],
                            "conditions": d.get("chronic_conditions", []), **extra})
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {"mode": mode, "results": results, "timings": {k: round(v, 2) for k, v in timings.items()},
                "query_patient": qdoc["short_id"] if qdoc else None}
