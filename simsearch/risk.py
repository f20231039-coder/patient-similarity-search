"""Condition-risk classifier (the Assignment 2 models, shown in the app).

Target (same as Assignment 2): has a chronic metabolic/cardiovascular diagnosis –
diabetes, hypertension, prediabetes or metabolic syndrome.
Models: Decision Tree, SVM, MLP (as in Assignment 2). Trained on the period-1
snapshot, tested on period 2 (temporal shift); the best model on a period-1
validation split is kept.

Explanations are model-agnostic occlusion: each feature is replaced with its
training median and the change in predicted probability is reported.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

log = logging.getLogger(__name__)

LAB_FEATURES = ["bmi", "systolic", "diastolic", "heart_rate", "glucose", "hba1c", "cholesterol", "ldl", "hdl",
                "triglycerides", "creatinine", "egfr", "hemoglobin", "microalbumin"]
FEATURES = ["age", "is_male", "log_encounters", "emergency_visits", "inpatient_visits", "years_of_history",
            "log_claim_cost", "income", "healthcare_expenses", "healthcare_coverage"] + LAB_FEATURES
FEATURE_LABELS = {
    "age": "age", "is_male": "male sex", "log_encounters": "number of encounters",
    "emergency_visits": "emergency visits", "inpatient_visits": "inpatient stays",
    "years_of_history": "years of history", "log_claim_cost": "total claim cost", "income": "income",
    "healthcare_expenses": "healthcare expenses", "healthcare_coverage": "healthcare coverage",
    "bmi": "BMI", "systolic": "systolic BP", "diastolic": "diastolic BP", "heart_rate": "heart rate",
    "glucose": "glucose", "hba1c": "HbA1c", "cholesterol": "total cholesterol", "ldl": "LDL", "hdl": "HDL",
    "triglycerides": "triglycerides", "creatinine": "creatinine", "egfr": "eGFR", "hemoglobin": "hemoglobin",
    "microalbumin": "urine microalbumin",
}
TARGET_NAME = "chronic metabolic/cardiovascular condition (diabetes, hypertension, prediabetes or metabolic syndrome)"


def featurize(doc: dict) -> list[float]:
    s = doc["structured"]
    enc = s["encounters_by_class"]
    row = [s.get("age") or np.nan, float(s["gender"] == "male"), np.log1p(s["n_encounters"]),
           enc.get("emergency", 0), enc.get("inpatient", 0), s["years_of_history"],
           np.log1p(s["total_claim_cost"] or 0), s.get("income") or np.nan,
           s.get("healthcare_expenses") or np.nan, s.get("healthcare_coverage") or np.nan]
    for k in LAB_FEATURES:
        v = s["latest"].get(k)
        row.append(v["value"] if v else np.nan)
    return row


def _models() -> dict:
    return {
        "Decision Tree": make_pipeline(SimpleImputer(strategy="median"),
                                       DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10,
                                                              class_weight="balanced", random_state=42)),
        "SVM": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                                                        random_state=42), ensemble=False, cv=5)),
        "Neural Network": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=800,
                                                      early_stopping=True, random_state=42)),
    }


def _scores(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")}


def train(period1_docs: list[dict], period2_docs: list[dict]) -> dict:
    X1 = np.array([featurize(d) for d in period1_docs], dtype=float)
    y1 = np.array([d["risk_target"] for d in period1_docs])
    X2 = np.array([featurize(d) for d in period2_docs], dtype=float)
    y2 = np.array([d["risk_target"] for d in period2_docs])
    Xtr, Xva, ytr, yva = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=42)
    report = {"target": TARGET_NAME, "n_train_period1": len(y1), "n_test_period2": len(y2),
              "positive_rate_period1": float(y1.mean()), "positive_rate_period2": float(y2.mean()), "models": {}}
    best, best_auc = None, -1.0
    for name, model in _models().items():
        model.fit(Xtr, ytr)
        val = _scores(yva, model.predict_proba(Xva)[:, 1])
        model.fit(X1, y1)  # refit on all of period 1 before the temporal test
        test = _scores(y2, model.predict_proba(X2)[:, 1])
        report["models"][name] = {"validation_period1": val, "test_period2": test}
        if val["roc_auc"] > best_auc:
            best, best_auc = (name, model), val["roc_auc"]
    medians = np.nanmedian(X1, axis=0)
    joblib.dump({"name": best[0], "model": best[1], "medians": medians, "features": FEATURES}, config.RISK_MODEL_PATH)
    report["selected_model"] = best[0]
    load_model.cache_clear()
    return report


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not config.RISK_MODEL_PATH.exists():
        raise RuntimeError("risk model not trained: run scripts/train_risk_model.py")
    return joblib.load(config.RISK_MODEL_PATH)


def predict(doc: dict) -> dict:
    m = load_model()
    p = float(m["model"].predict_proba(np.array([featurize(doc)], dtype=float))[0, 1])
    return {"patient_id": doc["short_id"], "target": TARGET_NAME, "probability": round(p, 3),
            "risk_level": "high" if p >= 0.66 else "moderate" if p >= 0.33 else "low", "model": m["name"]}


def explain(doc: dict, top_n: int = 5) -> dict:
    m = load_model()
    x = np.array(featurize(doc), dtype=float)
    base = float(m["model"].predict_proba(x[None, :])[0, 1])
    rows = []
    for i, f in enumerate(FEATURES):
        if np.isnan(x[i]) or x[i] == m["medians"][i]:
            continue
        x2 = x.copy()
        x2[i] = m["medians"][i]
        delta = base - float(m["model"].predict_proba(x2[None, :])[0, 1])
        value = float(np.expm1(x[i])) if f.startswith("log_") else float(x[i])
        typical = float(np.expm1(m["medians"][i])) if f.startswith("log_") else float(m["medians"][i])
        rows.append({"feature": FEATURE_LABELS[f], "value": round(value, 2), "typical_value": round(typical, 2),
                     "effect": "raises risk" if delta > 0 else "lowers risk", "contribution": round(delta, 3)})
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return {"patient_id": doc["short_id"], "probability": round(base, 3), "model": m["name"],
            "method": "occlusion: change in probability when the feature is set to the training median",
            "top_features": rows[:top_n]}
