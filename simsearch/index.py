"""Qdrant vector index + in-memory BM25, kept in sync by patient ID.

Qdrant is the single source of truth: every point carries the summary text and
payload, so the BM25 index can always be rebuilt from it (after a restart or a
re-index). Points are keyed by the patient's UUID, so re-indexing a patient
*replaces* the old record instead of duplicating it.
"""
from __future__ import annotations

import atexit
import logging
import re
import threading
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from . import config

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z][a-z0-9\-]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _words(text: str) -> list[str]:
    """Words only: raw numbers ("glucose 89.4") are exact-match noise for BM25."""
    return [t for t in tokenize(text) if not t[0].isdigit()]


def _band(n: int, cuts: tuple[int, int]) -> str:
    return "none" if n == 0 else "some" if n <= cuts[0] else "many" if n > cuts[1] else "several"


def bm25_doc_tokens(doc: dict) -> list[str]:
    """BM25 vocabulary for a patient: the summary's
    words without raw numbers, plus composite tokens that make clinical state an
    exact match: one per lab/vital with its flag, trends, age band, encounter
    pattern, insurance, smoking and social context."""
    s = doc["structured"]
    toks = _words(doc["text"])
    for k, v in s.get("latest", {}).items():
        toks.append(f"{k}_{v['flag'] or 'normal'}")
    for k, t in s.get("trends", {}).items():
        if t["direction"] != "stable":
            toks.append(f"{k}_{t['direction']}")
    if s.get("age") is not None:
        toks.append(f"age_{int(s['age'] // 10) * 10}s")
    toks.append(f"sex_{s.get('gender')}")
    if s.get("insurance"):
        toks.append(f"insurance_{s['insurance']}")
    if s.get("smoking"):
        sm = s["smoking"].lower()
        toks.append("smoker_never" if "never" in sm else "smoker_former" if "ex-" in sm or "former" in sm else "smoker_current")
    for phrase in (s.get("social") or {}).values():
        toks.append("social_" + re.sub(r"[^a-z]+", "_", phrase.lower()).strip("_"))
    enc = s.get("encounters_by_class", {})
    for cls, cuts in (("emergency", (2, 6)), ("inpatient", (1, 3)), ("urgentcare", (2, 6))):
        toks.append(f"{cls}_{_band(int(enc.get(cls, 0)), cuts)}")
    n = s.get("n_encounters", 0)
    toks.append("visits_" + ("few" if n < 15 else "moderate" if n < 50 else "many"))
    return toks


def bm25_query_tokens(text: str, query_doc: dict | None = None) -> list[str]:
    return bm25_doc_tokens(query_doc) if query_doc else _words(text)


class Filters(BaseModel):
    """Personalisation filters, applied as Qdrant payload filters (and to BM25)."""

    age_min: Optional[float] = Field(None, ge=0)
    age_max: Optional[float] = Field(None, ge=0)
    gender: Optional[str] = Field(None, pattern="^(male|female)$")
    period: Optional[int] = Field(None, ge=1, le=2)
    insurance: Optional[str] = Field(None, pattern="^(medicare|medicaid|private|uninsured|dual)$")

    def to_qdrant(self) -> Optional[models.Filter]:
        must = []
        if self.age_min is not None or self.age_max is not None:
            must.append(models.FieldCondition(key="age", range=models.Range(gte=self.age_min, lte=self.age_max)))
        if self.gender:
            must.append(models.FieldCondition(key="gender", match=models.MatchValue(value=self.gender)))
        if self.period:
            must.append(models.FieldCondition(key="period", match=models.MatchValue(value=self.period)))
        if self.insurance:
            must.append(models.FieldCondition(key="insurance", match=models.MatchValue(value=self.insurance)))
        return models.Filter(must=must) if must else None

    def matches(self, payload: dict) -> bool:
        age = payload.get("age")
        if self.age_min is not None and (age is None or age < self.age_min):
            return False
        if self.age_max is not None and (age is None or age > self.age_max):
            return False
        if self.gender and payload.get("gender") != self.gender:
            return False
        if self.period and payload.get("period") != self.period:
            return False
        if self.insurance and payload.get("insurance") != self.insurance:
            return False
        return True


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model %s", config.EMBED_MODEL)
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)


_clients: dict[str, QdrantClient] = {}
_client_lock = threading.Lock()


@atexit.register
def _close_clients() -> None:
    """Close embedded Qdrant cleanly before interpreter shutdown (avoids noisy __del__ errors)."""
    for c in _clients.values():
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
    _clients.clear()


def get_client() -> QdrantClient:
    """One client per location (embedded Qdrant allows a single client per path)."""
    key = config.QDRANT_URL or str(config.QDRANT_PATH)
    with _client_lock:
        if key not in _clients:
            if config.QDRANT_URL:
                _clients[key] = QdrantClient(url=config.QDRANT_URL, timeout=30)
            else:
                config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
                _clients[key] = QdrantClient(path=str(config.QDRANT_PATH))
        return _clients[key]


PAYLOAD_KEYS = ("patient_id", "short_id", "period", "snapshot_date", "updated_at", "age", "gender", "insurance",
                "text", "structured", "conditions", "chronic_conditions", "risk_target", "content_hash")


class PatientIndex:
    def __init__(self, client: QdrantClient | None = None, collection: str = config.COLLECTION):
        self.client = client or get_client()
        self.collection = collection
        self._lock = threading.RLock()
        self._docs: dict[str, dict] = {}
        self._short: dict[str, str] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self.ensure_collection()
        self.refresh()

    # ------------------------------------------------------------------ setup
    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            m = get_embedder()
            dim = (m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension")
                   else m.get_sentence_embedding_dimension())
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            if config.QDRANT_URL:  # payload indexes are a server feature
                self.client.create_payload_index(self.collection, "age", models.PayloadSchemaType.FLOAT)
                self.client.create_payload_index(self.collection, "gender", models.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(self.collection, "period", models.PayloadSchemaType.INTEGER)
                self.client.create_payload_index(self.collection, "insurance", models.PayloadSchemaType.KEYWORD)

    def reset(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.ensure_collection()
            # Embedded Qdrant on Windows can resurrect old points when a collection is
            # dropped and recreated under the same name; clear explicitly and verify.
            if self.client.count(self.collection, exact=True).count:
                self.client.delete(self.collection, wait=True,
                                   points_selector=models.FilterSelector(filter=models.Filter()))
            if self.client.count(self.collection, exact=True).count:
                raise RuntimeError(f"could not clear collection {self.collection}")
            self.refresh()

    # ----------------------------------------------------------------- writes
    def upsert(self, docs: Iterable[dict], batch_size: int = 256) -> dict:
        """Add new patients and replace changed ones, keyed by patient ID.

        * added       – new patient: embedded and inserted
        * updated     – summary text changed: re-embedded and replaced
        * payload_only – text identical but metadata (period, dates, labels) changed:
                        payload overwritten in place, no re-embedding
        * unchanged   – nothing to do
        """
        docs = list(docs)
        stats = {"added": 0, "updated": 0, "payload_only": 0, "unchanged": 0}
        with self._lock:
            todo = []
            for d in docs:
                old = self._docs.get(d["patient_id"])
                if old is None:
                    stats["added"] += 1
                    todo.append(d)
                elif old.get("content_hash") != d["content_hash"]:
                    stats["updated"] += 1
                    todo.append(d)
                elif any(old.get(k) != d.get(k) for k in PAYLOAD_KEYS if k != "structured") \
                        or old.get("structured") != d.get("structured"):
                    stats["payload_only"] += 1
                    self.client.overwrite_payload(self.collection, payload={k: d.get(k) for k in PAYLOAD_KEYS},
                                                  points=[d["patient_id"]], wait=True)
                else:
                    stats["unchanged"] += 1
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                vecs = embed([d["text"] for d in chunk])
                points = [models.PointStruct(id=d["patient_id"], vector=v.tolist(),
                                             payload={k: d.get(k) for k in PAYLOAD_KEYS})
                          for d, v in zip(chunk, vecs)]
                self.client.upsert(self.collection, points=points, wait=True)
            self.refresh()
        return stats

    # ------------------------------------------------------------------ reads
    def refresh(self) -> None:
        """Reload payloads from Qdrant and rebuild BM25."""
        with self._lock:
            docs, offset = {}, None
            while True:
                pts, offset = self.client.scroll(self.collection, limit=1024, offset=offset,
                                                 with_payload=True, with_vectors=False)
                for p in pts:
                    docs[p.payload["patient_id"]] = p.payload
                if offset is None:
                    break
            self._docs = docs
            self._short = {d["short_id"]: pid for pid, d in docs.items()}
            self._bm25_ids = list(docs)
            self._bm25 = BM25Okapi([bm25_doc_tokens(docs[i]) for i in self._bm25_ids]) if docs else None

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def __len__(self) -> int:
        return len(self._docs)

    def resolve_id(self, ident: str) -> Optional[str]:
        ident = ident.strip()
        if ident in self._docs:
            return ident
        return self._short.get(ident.upper())

    def get(self, ident: str) -> Optional[dict]:
        pid = self.resolve_id(ident)
        return self._docs.get(pid) if pid else None

    def all_docs(self) -> dict[str, dict]:
        return self._docs

    def vector_by_id(self, pid: str) -> Optional[np.ndarray]:
        pts = self.client.retrieve(self.collection, ids=[pid], with_vectors=True)
        return np.asarray(pts[0].vector) if pts else None

    def vector_search(self, vector: np.ndarray, k: int, filters: Filters | None = None,
                      exclude: str | None = None) -> list[tuple[str, float]]:
        flt = filters.to_qdrant() if filters else None
        if exclude:
            cond = models.HasIdCondition(has_id=[exclude])
            flt = flt or models.Filter()
            flt.must_not = [cond]
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k,
                                       query_filter=flt, with_payload=False).points
        return [(str(p.id), float(p.score)) for p in res]

    def bm25_search(self, text: str, k: int, filters: Filters | None = None, exclude: str | None = None,
                    query_doc: dict | None = None) -> list[tuple[str, float]]:
        return self._bm25_rank(bm25_query_tokens(text, query_doc), k, filters, exclude)

    def _bm25_rank(self, tokens: list[str], k: int, filters: Filters | None = None,
                    exclude: str | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)
        out = []
        for i in order:
            pid = self._bm25_ids[i]
            if pid == exclude or (filters and not filters.matches(self._docs[pid])):
                continue
            out.append((pid, float(scores[i])))
            if len(out) >= k:
                break
        return out
