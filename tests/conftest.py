"""Shared fixtures: a temporary embedded Qdrant + risk model built from the small
committed sample in tests/fixtures (no CSVs, Docker or API keys needed)."""
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from simsearch import config  # noqa: E402

_tmp = Path(tempfile.mkdtemp(prefix="simsearch-test-"))
config.QDRANT_URL = None
config.QDRANT_PATH = _tmp / "qdrant"
config.RISK_MODEL_PATH = _tmp / "risk_model.joblib"
config.LTR_MODEL_PATH = _tmp / "learned_ranker.joblib"
config.DOCS_PERIOD1 = FIX / "documents_period1.jsonl"
config.DOCS_PERIOD2 = FIX / "documents_period2.jsonl"
# Tests never call a real LLM or use a private key (.env): the rule parser answers,
# and LLM behaviour is tested with fake LLM objects.
config.LLM_PROVIDER = "none"
config.LLM_API_KEY = ""

from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex  # noqa: E402
from simsearch.search import SearchEngine  # noqa: E402


@pytest.fixture(scope="session")
def docs1():
    return load_jsonl(config.DOCS_PERIOD1)


@pytest.fixture(scope="session")
def docs2():
    return load_jsonl(config.DOCS_PERIOD2)


@pytest.fixture(scope="session")
def risk_model(docs1, docs2):
    from simsearch import risk
    return risk.train(docs1, docs2)


@pytest.fixture(scope="session")
def full_index(docs1, docs2):
    """Period 1 then period 2 upserted (the state after re-indexing)."""
    idx = PatientIndex(collection="test_full")
    idx.reset()
    idx.upsert(docs1)
    idx.upsert(docs2)
    return idx


@pytest.fixture(scope="session")
def engine(full_index):
    return SearchEngine(full_index)
