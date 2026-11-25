"""FastAPI service and web app.

  GET  /                      the web app (static/app.html)
  GET  /health                index size and models
  POST /search                similar patients (default search: hybrid + learned ranker)
  POST /query                 plain-language search -> filters (LLM or rules) -> search
  POST /report                everything the overview shows for one patient
  GET  /compare?a=..&b=..     two patients side by side
  GET  /patients/showcase     named demo cases
  GET  /patients/conditions   most common recorded conditions (browse chips)
  GET  /patients/browse       find patients by condition or description
  POST /reindex               upsert a newer snapshot by patient ID
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import config, risk
from .index import Filters, PatientIndex
from .query_parser import ParsedQuery, parse_query, run_query
from .reindex import build_period1, reindex
from .search import SearchEngine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("simsearch.api")
STATIC = Path(__file__).parent / "static"


# ------------------------------------------------------------------ requests
class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, max_length=4000)
    patient_id: Optional[str] = None
    filters: Optional[Filters] = None
    k: int = Field(10, ge=1, le=50)

    @model_validator(mode="after")
    def _one_of(self):
        if not self.query and not self.patient_id:
            raise ValueError("provide either query or patient_id")
        return self


class QueryRequest(BaseModel):
    q: str = Field("", max_length=500)
    parsed: Optional[dict] = None  # an edited parse from the UI (a chip removed); skips the parser
    k: int = Field(15, ge=1, le=50)
    strict: bool = True


class ReportRequest(BaseModel):
    patient_id: str
    filters: Optional[Filters] = None
    k: int = Field(10, ge=1, le=25)


class ReindexRequest(BaseModel):
    source: Literal["period1", "period2"] = "period2"


# ------------------------------------------------------------------ state
class State:
    index: PatientIndex
    engine: SearchEngine
    reindex_lock = threading.Lock()


state = State()


def init_state(index: PatientIndex | None = None) -> None:
    state.index = index or PatientIndex()
    if len(state.index) == 0 and os.getenv("AUTO_BUILD_INDEX", "1") == "1" and config.DOCS_PERIOD1.exists():
        log.info("index empty: building from period-1 documents")
        build_period1(state.index, reset=False)
    state.engine = SearchEngine(state.index)


def _warmup() -> None:
    try:
        state.engine.search(patient_id=next(iter(state.index.all_docs())), k=3)
    except Exception as e:  # noqa: BLE001
        log.warning("warmup failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(state, "index", None):
        init_state()
    if os.getenv("WARMUP_MODELS", "1") == "1" and len(state.index):
        await run_in_threadpool(_warmup)
    yield


app = FastAPI(title="Patient Similarity", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t) * 1000)
    return response


async def _with_timeout(fn, *args, **kwargs):
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args, **kwargs), timeout=config.REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, f"timed out after {config.REQUEST_TIMEOUT_S:.0f}s")


def _card(d: dict) -> dict:
    return {"short_id": d["short_id"], "age": d["age"], "gender": d["gender"], "insurance": d.get("insurance"),
            "period": d["period"], "deceased": d["structured"].get("deceased", False),
            "chronic_conditions": d.get("chronic_conditions", [])}


def _risk(doc: dict) -> tuple[dict, dict]:
    try:
        return risk.predict(doc), risk.explain(doc)
    except RuntimeError as e:  # risk model not trained
        return {"error": str(e)}, {"error": str(e)}


def _living_adults() -> list[dict]:
    return [d for d in state.index.all_docs().values()
            if d["age"] and 18 <= d["age"] <= 95 and not d["structured"].get("deceased")]


# ------------------------------------------------------------------ pages + health
@app.get("/")
def ui():
    return FileResponse(STATIC / "app.html")


@app.get("/health")
def health():
    docs = state.index.all_docs()
    periods = Counter(d["period"] for d in docs.values())
    return {"status": "ok", "patients_indexed": len(docs), "qdrant_points": state.index.count(),
            "records_by_period": {str(k): v for k, v in sorted(periods.items())},
            "embedding_model": "MiniLM fine-tuned on patients" if Path(config.EMBED_MODEL).exists() else config.EMBED_MODEL,
            "learned_ranker": config.LTR_MODEL_PATH.exists(),
            "qdrant": config.QDRANT_URL or "embedded"}


# ------------------------------------------------------------------ search
@app.post("/search")
async def search(req: SearchRequest):
    try:
        res = await _with_timeout(state.engine.search, query=req.query, patient_id=req.patient_id, k=req.k,
                                  filters=req.filters)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("search k=%d mode=%s timings=%s", req.k, res["mode"], res["timings"])
    return res


@app.post("/query")
async def plain_language_query(req: QueryRequest):
    """Plain-language search -> filters (LLM if configured, else rules) -> search."""
    if req.parsed is not None:
        try:
            pq = ParsedQuery(**{**req.parsed, "text": req.q})
        except ValueError as e:
            raise HTTPException(422, f"invalid filters: {e}")
        meta = {"parser": "edited by user", "parse_ms": 0.0}
    elif req.q.strip():
        pq, meta = await _with_timeout(parse_query, req.q)
    else:
        raise HTTPException(422, "type a search, e.g. 'women over 60 with high BMI and rising glucose'")
    res = await _with_timeout(run_query, state.engine, pq, req.k, req.strict)
    log.info("query parser=%s parse_ms=%s search_ms=%s full_matches=%s", meta["parser"], meta["parse_ms"],
             res["search_ms"], res["n_full_matches"])
    return {**res, **meta, "query": req.q}


def _report(req: ReportRequest) -> dict:
    doc = state.index.get(req.patient_id)
    if doc is None:
        raise KeyError(f"patient {req.patient_id} not found")
    res = state.engine.search(patient_id=doc["short_id"], k=req.k, filters=req.filters)
    docs = state.index.all_docs()
    matches = [{**_card(docs[r["patient_id"]]), "rank": r["rank"], "score": r["score"],
                "structured": docs[r["patient_id"]]["structured"]} for r in res["results"]]
    rk, ex = _risk(doc)
    return {"patient": {**_card(doc), "structured": doc["structured"], "summary": doc["text"], "updated_at": doc["updated_at"]},
            "matches": matches, "condition_counts": Counter(c for m in matches for c in m["chronic_conditions"]).most_common(),
            "risk": rk, "explanation": ex, "ranking": res["mode"],
            "filters": req.filters.model_dump(exclude_none=True) if req.filters else {},
            "timings": res["timings"], "index_size": len(docs)}


@app.post("/report")
async def report(req: ReportRequest):
    try:
        return await _with_timeout(_report, req)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))


@app.get("/compare")
def compare(a: str, b: str):
    """Two patients side by side: results, conditions, risk and reasons (no search involved)."""
    da, db = state.index.get(a), state.index.get(b)
    missing = [x for x, d in ((a, da), (b, db)) if d is None]
    if missing:
        raise HTTPException(404, f"patient {', '.join(missing)} not found")

    def profile(doc):
        rk, ex = _risk(doc)
        return {**_card(doc), "structured": doc["structured"], "summary": doc["text"], "updated_at": doc["updated_at"],
                "risk": rk, "explanation": ex}

    ca, cb = set(da.get("chronic_conditions", [])), set(db.get("chronic_conditions", []))
    return {"a": profile(da), "b": profile(db), "shared_conditions": sorted(ca & cb),
            "only_a": sorted(ca - cb), "only_b": sorted(cb - ca)}


# ------------------------------------------------------------------ finding a patient without an ID
SHOWCASE = [
    ("Diabetes, sugar not well controlled", lambda d, l: "Diabetes mellitus type 2" in d["chronic_conditions"]
     and l.get("hba1c", {}).get("flag") == "high"),
    ("Older man with heart disease", lambda d, l: d["gender"] == "male" and d["age"] >= 60
     and "Ischemic heart disease" in d["chronic_conditions"]),
    ("High blood pressure and high cholesterol", lambda d, l: "Essential hypertension" in d["chronic_conditions"]
     and l.get("ldl", {}).get("flag") in ("high", "borderline")),
    ("Kidney disease", lambda d, l: any("kidney" in c.lower() for c in d["chronic_conditions"])),
    ("Woman in her 30s-40s with asthma", lambda d, l: d["gender"] == "female" and 30 <= d["age"] < 50
     and any("asthma" in c.lower() for c in d["chronic_conditions"])),
    ("Healthy adult, nothing on record", lambda d, l: not d["chronic_conditions"] and 30 <= d["age"] <= 55
     and not any(v.get("flag") in ("high", "low") for v in l.values())),
]


@app.get("/patients/showcase")
def showcase():
    """Named demo cases so nobody has to remember an ID (stable choice per index)."""
    pool = sorted(_living_adults(), key=lambda d: d["patient_id"])
    out, used = [], set()
    for title, rule in SHOWCASE:
        for d in pool:
            if d["patient_id"] not in used and rule(d, d["structured"]["latest"]):
                used.add(d["patient_id"])
                out.append({"title": title, **_card(d)})
                break
    return out


@app.get("/patients/conditions")
def condition_list(limit: int = 12):
    """Most common recorded long-term conditions, for browse chips."""
    c = Counter(x for d in _living_adults() for x in d.get("chronic_conditions", []))
    return [{"name": n, "patients": k} for n, k in c.most_common(max(1, min(limit, 40)))]


@app.get("/patients/browse")
def browse_patients(condition: Optional[str] = None, q: Optional[str] = None, gender: Optional[str] = None,
                    age_min: Optional[float] = None, age_max: Optional[float] = None, limit: int = 15):
    """Find patients without knowing an ID: by recorded condition, a description, or an ID prefix."""
    limit = max(1, min(limit, 30))
    flt = Filters(gender=gender or None, age_min=age_min, age_max=age_max)
    if q and q.strip():
        text = q.strip().upper()
        if text.startswith("P-") and len(text) <= 10:
            docs = [d for d in state.index.all_docs().values() if d["short_id"].startswith(text)]
        else:
            hits = state.engine.search(query=q.strip(), k=60 if condition else limit, filters=flt, mode="hybrid")["results"]
            docs = [state.index.all_docs()[h["patient_id"]] for h in hits]
    else:
        docs = sorted((d for d in _living_adults() if flt.matches(d)), key=lambda d: d["patient_id"])
    if condition:
        docs = [d for d in docs if condition in d.get("chronic_conditions", [])]
    return [_card(d) for d in docs[:limit]]


# ------------------------------------------------------------------ re-index job
@app.post("/reindex")
async def reindex_endpoint(req: ReindexRequest = ReindexRequest()):
    if not state.reindex_lock.acquire(blocking=False):
        raise HTTPException(409, "a re-index is already running")
    try:
        if req.source == "period1":
            return await run_in_threadpool(build_period1, state.index)
        return await run_in_threadpool(reindex, state.index, config.DOCS_PERIOD2)
    finally:
        state.reindex_lock.release()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
