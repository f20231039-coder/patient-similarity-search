"""Step 1: build period-1 and period-2 patient documents (text + structured JSON)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simsearch import config
from simsearch.documents import build_both_periods, save_jsonl

t = time.time()
p1, p2 = build_both_periods()
save_jsonl(p1, config.DOCS_PERIOD1)
save_jsonl(p2, config.DOCS_PERIOD2)
ids1 = {d["patient_id"]: d["content_hash"] for d in p1}
new = sum(d["patient_id"] not in ids1 for d in p2)
changed = sum(d["patient_id"] in ids1 and ids1[d["patient_id"]] != d["content_hash"] for d in p2)
print(f"period 1: {len(p1)} docs | period 2: {len(p2)} docs ({new} new, {changed} changed, "
      f"{len(p2) - new - changed} unchanged) | {time.time() - t:.1f}s")
print(f"with >=1 chronic condition: p1={sum(bool(d['chronic_conditions']) for d in p1)} "
      f"p2={sum(bool(d['chronic_conditions']) for d in p2)}")
print("example:\n", p2[0]["short_id"], p2[0]["text"], "\n answer key:", p2[0]["chronic_conditions"])
