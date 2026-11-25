"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a local .env (gitignored). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "csv"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Same temporal split as the original Assignment 2 pipeline.
TEMPORAL_SPLIT_DATE = os.getenv("TEMPORAL_SPLIT_DATE", "2015-01-01")

# Documents produced by scripts/build_documents.py
DOCS_PERIOD1 = ARTIFACT_DIR / "documents_period1.jsonl"
DOCS_PERIOD2 = ARTIFACT_DIR / "documents_period2.jsonl"

# Qdrant: set QDRANT_URL (e.g. http://qdrant:6333) to use a server, otherwise an
# embedded on-disk instance under artifacts/qdrant is used (no Docker needed).
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", ARTIFACT_DIR / "qdrant"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "patients")

# Default: MiniLM fine-tuned on training patients (evals/finetune.py; LambdaRank nDCG@10 0.439 -> 0.458).
# Falls back to the base model when the fine-tuned one isn't present (e.g. CI).
_FT_EMBEDDER = ARTIFACT_DIR / "models" / "embed-ft"
EMBED_MODEL = os.getenv("EMBED_MODEL") or (str(_FT_EMBEDDER) if (_FT_EMBEDDER / "model.safetensors").exists()
                                           else "sentence-transformers/all-MiniLM-L6-v2")

CANDIDATES_PER_RETRIEVER = int(os.getenv("CANDIDATES_PER_RETRIEVER", "50"))
# Tuned on training queries (k=30 beat 60 and 100; equal weights were best).
RRF_K = int(os.getenv("RRF_K", "30"))
# Weighted reciprocal rank fusion: score = w_bm25/(k+rank_bm25) + w_vec/(k+rank_vec).
RRF_W_BM25 = float(os.getenv("RRF_W_BM25", "1.0"))
RRF_W_VEC = float(os.getenv("RRF_W_VEC", "1.0"))

RISK_MODEL_PATH = ARTIFACT_DIR / "risk_model.joblib"
LTR_MODEL_PATH = ARTIFACT_DIR / "learned_ranker.joblib"

# LLM: used ONLY to turn a plain-language search into filters.
# Provider: groq | gemini | ollama | openai | none (none = rule-based parser only).
# With just LLM_API_KEY set, the provider is detected from the key's prefix.
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()


def _detect_provider(key: str) -> str:
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-"):
        return "openai"
    return "none"


LLM_PROVIDER = os.getenv("LLM_PROVIDER") or _detect_provider(LLM_API_KEY)
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
