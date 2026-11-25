"""CI evaluation on the committed sample (tests/fixtures): re-index without duplicates and
retrieval metrics for every mode. Fails the build if quality drops below the thresholds."""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals"))

from simsearch import config
tmp = Path(tempfile.mkdtemp(prefix="ci-eval-"))
config.QDRANT_URL, config.QDRANT_PATH = None, tmp / "qdrant"
config.RISK_MODEL_PATH, config.LTR_MODEL_PATH = tmp / "risk.joblib", tmp / "ltr.joblib"
config.DOCS_PERIOD1 = ROOT / "tests/fixtures/documents_period1.jsonl"
config.DOCS_PERIOD2 = ROOT / "tests/fixtures/documents_period2.jsonl"

from simsearch import risk
from simsearch.documents import load_jsonl
from simsearch.index import PatientIndex
from simsearch.reindex import build_period1, reindex
from simsearch.search import SearchEngine
from retrieval_eval import evaluate, pick_queries

OUT = ROOT / "evals/results/ci"; OUT.mkdir(parents=True, exist_ok=True)
p1, p2 = load_jsonl(config.DOCS_PERIOD1), load_jsonl(config.DOCS_PERIOD2)
risk.train(p1, p2)
idx = PatientIndex(collection="ci")
build_period1(idx)
stats = reindex(idx)
assert stats["qdrant_points"] == len({d["patient_id"] for d in p1 + p2}), "duplicate points after re-index"

truth = {d["patient_id"]: d["chronic_conditions"] for d in p2}
queries = pick_queries(p2, 60, 42)
engine = SearchEngine(idx)
retrieval = [evaluate(engine, queries, truth, m) for m in ("bm25", "vector", "hybrid")]
for r in retrieval:
    print(f"{r['system']:35s} R@10 {r['recall@10']:.3f} MRR {r['mrr']:.3f} nDCG {r['ndcg@10']:.3f}")

for r in retrieval:
    r.pop("_per_query", None)
(OUT / "retrieval.json").write_text(json.dumps({"reindex": stats, "retrieval": retrieval}, indent=2))

failures = []
best = max(r["ndcg@10"] for r in retrieval)
if best < 0.15:
    failures.append(f"best nDCG@10 {best:.3f} < 0.15 on the sample")
if min(r["ndcg@10"] for r in retrieval) <= 0:
    failures.append("a retrieval mode returned no relevant results")
if failures:
    sys.exit("EVAL REGRESSION: " + "; ".join(failures))
print("CI eval passed")
