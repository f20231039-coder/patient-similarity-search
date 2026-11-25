"""Build or update the search index (embedded Qdrant under artifacts/qdrant, or QDRANT_URL).

  python scripts/build_index.py                 # period 1, then the period-2 re-index (upsert by patient ID)
  python scripts/build_index.py --period1-only  # period 1 only (POST /reindex adds period 2 later)
  python scripts/build_index.py --if-empty      # skip when the index already has patients (container start)
  python scripts/build_index.py --with-eval     # also rebuild the 'eval_patients' collection used by evals/
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simsearch import config  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex  # noqa: E402
from simsearch.reindex import build_period1, reindex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--period1-only", action="store_true")
ap.add_argument("--if-empty", action="store_true")
ap.add_argument("--with-eval", action="store_true")
a = ap.parse_args()

print(f"embedder: {config.EMBED_MODEL}")
idx = PatientIndex()
if a.if_empty and len(idx):
    print(f"index already has {len(idx)} patients; skipping")
else:
    print(json.dumps(build_period1(idx)))
    if not a.period1_only:
        print(json.dumps(reindex(idx)))
if a.with_eval:
    ev = PatientIndex(collection="eval_patients")
    ev.reset()
    print("eval_patients:", json.dumps(ev.upsert(load_jsonl(config.DOCS_PERIOD2))))
