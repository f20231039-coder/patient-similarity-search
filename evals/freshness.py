"""Changing-index check, done properly: what does a stale index get wrong?

The held-out evaluation patients whose records are new or changed in period 2, stale index (period-1 snapshot)
vs re-indexed index (period 2), default ranking (learned ranker). Metrics:

  coverage            share of each query's relevant patients that exist in the index at all
  stale_results       share of the top 10 whose stored record is out of date (old labs/visits shown)
  new_patients_shown  share of the top 10 that are period-2 newcomers (a stale index can never show them)
  fixed_pool_ndcg     nDCG@10 when both indexes may only return the 1,803 patients that exist in
                      both, so only the freshness of their records differs
  ndcg                ordinary nDCG@10 (the earlier, not like-for-like comparison)

Run:  python evals/freshness.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from simsearch import config  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex  # noqa: E402
from simsearch.metrics import bootstrap_ci, ndcg_at_k, paired_bootstrap, relevance  # noqa: E402
from simsearch.reindex import build_period1, reindex  # noqa: E402
from simsearch.search import SearchEngine  # noqa: E402
from retrieval_eval import RESULTS, pick_queries  # noqa: E402


def measure(engine, queries, truth, p2_hash, p1_ids, mode):
    docs = engine.index.all_docs()
    old_pool = [pid for pid in p1_ids]
    rows = {"coverage": [], "stale_results": [], "new_patients_shown": [], "fixed_pool_ndcg": [], "ndcg": []}
    for q in queries:
        qc, pid = truth[q["patient_id"]], q["patient_id"]
        relevant = [o for o, c in truth.items() if o != pid and relevance(qc, c) > 0]
        rows["coverage"].append(sum(o in docs for o in relevant) / max(1, len(relevant)))
        res = engine.search(query_doc=q, k=40, mode=mode)["results"]
        top = res[:10]
        rows["stale_results"].append(sum(docs[r["patient_id"]].get("content_hash") != p2_hash[r["patient_id"]] for r in top) / max(1, len(top)))
        rows["new_patients_shown"].append(sum(r["patient_id"] not in p1_ids for r in top) / max(1, len(top)))
        all_rels = [relevance(qc, c) for o, c in truth.items() if o != pid]
        rels = [relevance(qc, truth.get(r["patient_id"], [])) for r in top]
        rows["ndcg"].append(ndcg_at_k(rels, all_rels, 10))
        fixed = [r for r in res if r["patient_id"] in p1_ids][:10]
        pool_rels = [relevance(qc, truth[o]) for o in old_pool if o != pid]
        rows["fixed_pool_ndcg"].append(ndcg_at_k([relevance(qc, truth.get(r["patient_id"], [])) for r in fixed], pool_rels, 10))
    return rows


def main() -> None:
    p1, p2 = load_jsonl(config.DOCS_PERIOD1), load_jsonl(config.DOCS_PERIOD2)
    truth = {d["patient_id"]: d["chronic_conditions"] for d in p2}
    p1_hash = {d["patient_id"]: d["content_hash"] for d in p1}
    p2_hash = {d["patient_id"]: d["content_hash"] for d in p2}
    p1_ids = set(p1_hash)
    # Only the 200 held-out evaluation patients (never used to train the embedder or the ranker),
    # restricted to those whose record is new or changed in period 2.
    held_out = pick_queries(p2, 200, 42)
    queries = [q for q in held_out if p1_hash.get(q["patient_id"]) != q["content_hash"]]
    mode = "ltr" if config.LTR_MODEL_PATH.exists() else "hybrid"

    idx = PatientIndex(collection="freshness_eval")
    engine = SearchEngine(idx)
    build_period1(idx)
    stale = measure(engine, queries, truth, p2_hash, p1_ids, mode)
    stats = reindex(idx)
    fresh = measure(engine, queries, truth, p2_hash, p1_ids, mode)

    out = {"mode": mode, "n_queries": len(queries), "n_new_patient_queries": sum(q["patient_id"] not in p1_ids for q in queries),
           "reindex": stats, "metrics": {}}
    for m in stale:
        s, f = statistics.mean(stale[m]), statistics.mean(fresh[m])
        out["metrics"][m] = {"stale": round(s, 4), "fresh": round(f, 4), "stale_ci95": [round(x, 4) for x in bootstrap_ci(stale[m])],
                             "fresh_ci95": [round(x, 4) for x in bootstrap_ci(fresh[m])], "paired": paired_bootstrap(stale[m], fresh[m])}
    (RESULTS / "freshness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    names = {"coverage": "Relevant patients present in the index", "stale_results": "Top-10 results showing an outdated record",
             "new_patients_shown": "Top-10 results that are new (period-2) patients",
             "fixed_pool_ndcg": "nDCG@10, same 1,803-patient pool (record freshness only)", "ndcg": "nDCG@10, full pool (not like-for-like)"}
    md = [f"## Changing-index check: stale vs re-indexed ({len(queries)} held-out period-2 queries, learned ranker)", "",
          "| Metric | Stale index (period 1) | Re-indexed (period 2) | Difference (95% CI) |", "|---|---|---|---|"]
    for m, v in out["metrics"].items():
        pct = m in ("coverage", "stale_results", "new_patients_shown")
        fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
        d = v["paired"]
        md.append(f"| {names[m]} | {fmt(v['stale'])} | {fmt(v['fresh'])} | {d['diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) |")
    md += ["", f"Re-index: {stats['added']} added, {stats['updated']} re-embedded, {stats.get('payload_only', 0)} payload-only, "
               f"{stats['qdrant_points']} points, no duplicates. {out['n_new_patient_queries']} of the queries are new patients."]
    (RESULTS / "freshness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
