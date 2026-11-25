"""Clean latency measurement (run with nothing else using the CPU).

Per search mode: median and 95th-percentile end-to-end time and per-stage time,
over 100 patient queries, after a warm-up. CPU only.

Run:  python evals/latency.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from simsearch import config  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex  # noqa: E402
from simsearch.search import SearchEngine  # noqa: E402
from retrieval_eval import RESULTS, pick_queries  # noqa: E402


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main() -> None:
    p2 = load_jsonl(config.DOCS_PERIOD2)
    queries = pick_queries(p2, 100, 7)
    engine = SearchEngine(PatientIndex(collection="eval_patients"))
    out = {"cpu_threads": os.cpu_count(), "n_queries": len(queries), "embedder": config.EMBED_MODEL, "modes": {}}
    for mode in ("bm25", "vector", "hybrid", "ltr"):
        for q in queries[:5]:  # warm-up (model loading, caches)
            engine.search(patient_id=q["short_id"], k=10, mode=mode)
        totals, stages = [], {}
        for q in queries:
            t = engine.search(patient_id=q["short_id"], k=10, mode=mode)["timings"]
            totals.append(t["total_ms"])
            for k, v in t.items():
                if k != "total_ms":
                    stages.setdefault(k, []).append(v)
        out["modes"][mode] = {"median_ms": round(statistics.median(totals), 1), "p95_ms": round(pct(totals, 95), 1),
                              "stages_median_ms": {k: round(statistics.median(v), 1) for k, v in stages.items()}}
        print(mode, out["modes"][mode], flush=True)
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    names = {"bm25": "BM25 only", "vector": "Vector only", "hybrid": "Hybrid (RRF)", "ltr": "Hybrid + learned ranker (default)"}
    md = [f"## Latency (CPU, {out['cpu_threads']} threads, 100 patient queries, nothing else running)", "",
          "| System | Median | 95th percentile | Main stages (median) |", "|---|---|---|---|"]
    for m, v in out["modes"].items():
        st = ", ".join(f"{k.replace('_ms', '')} {x:.0f} ms" for k, x in v["stages_median_ms"].items())
        md.append(f"| {names[m]} | {v['median_ms']:.0f} ms | {v['p95_ms']:.0f} ms | {st} |")
    (RESULTS / "latency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
