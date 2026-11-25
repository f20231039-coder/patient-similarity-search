"""Retrieval evaluation of the final search, stage by stage.

200 query patients, top 10. Two patients count as similar if they share a diagnosed
chronic condition (nDCG gain = number shared); conditions are never in the
searched text. Reports Recall@10, MRR, nDCG@10 (with a 95% bootstrap CI), P@10 and
median latency for each stage of the default search:

  bm25    keyword search on words + clinical-state tokens
  vector  embedding search (fine-tuned MiniLM, Qdrant)
  hybrid  both, merged with reciprocal rank fusion
  ltr     hybrid + LambdaRank learned ranker  (the default)

Also used as a library by the other eval scripts (pick_queries, evaluate).

Run:  python evals/retrieval_eval.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from simsearch import config  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex  # noqa: E402
from simsearch.metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, relevance  # noqa: E402
from simsearch.search import STAGES, SearchEngine  # noqa: E402

RESULTS = ROOT / "evals" / "results"
LABELS = {"bm25": "BM25 (words + clinical flags)", "vector": "Vector search (fine-tuned embedder)",
          "hybrid": "Hybrid (RRF)", "ltr": "Hybrid + learned ranker (default)"}


def pick_queries(docs: list[dict], n: int, seed: int, exclude: set[str] = frozenset()) -> list[dict]:
    pool = [d for d in docs if d["chronic_conditions"] and d["patient_id"] not in exclude]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def evaluate(engine: SearchEngine, queries: list[dict], truth: dict[str, list[str]], mode: str, k: int = 10) -> dict:
    """Score top-k results. ``truth`` maps every patient to its current chronic conditions."""
    rec, mrr, ndcg, prec, lat = [], [], [], [], []
    for q in queries:
        t = time.perf_counter()
        res = engine.search(query_doc=q, k=k, mode=mode)
        lat.append((time.perf_counter() - t) * 1000)
        qc = truth[q["patient_id"]]
        rels = [relevance(qc, truth.get(r["patient_id"], [])) for r in res["results"]]
        all_rels = [relevance(qc, truth.get(pid, [])) for pid in engine.index.all_docs() if pid != q["patient_id"]]
        n_rel = sum(1 for r in all_rels if r > 0)
        rec.append(recall_at_k(rels, n_rel, k))
        mrr.append(mrr_at_k(rels, k))
        ndcg.append(ndcg_at_k(rels, all_rels, k))
        prec.append(precision_at_k(rels, k))
    return {"mode": mode, "system": LABELS[mode], "n_queries": len(queries),
            "recall@10": round(float(np.mean(rec)), 4), "mrr": round(float(np.mean(mrr)), 4),
            "ndcg@10": round(float(np.mean(ndcg)), 4), "precision@10": round(float(np.mean(prec)), 4),
            "latency_median_ms": round(statistics.median(lat), 1),
            "_per_query": {"ndcg@10": [round(x, 5) for x in ndcg], "recall@10": [round(x, 5) for x in rec]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    p2 = load_jsonl(config.DOCS_PERIOD2)
    truth = {d["patient_id"]: d["chronic_conditions"] for d in p2}
    queries = pick_queries(p2, a.n, a.seed)
    idx = PatientIndex(collection="eval_patients")
    if len(idx) != len(p2):
        idx.reset()
        idx.upsert(p2)
    engine = SearchEngine(idx)

    rows = []
    for m in STAGES:
        r = evaluate(engine, queries, truth, m)
        lo, hi = bootstrap_ci(r.pop("_per_query")["ndcg@10"])
        r["ndcg_ci95"] = [round(lo, 4), round(hi, 4)]
        rows.append(r)
        print(f"{r['system']:40s} R@10 {r['recall@10']:.3f} MRR {r['mrr']:.3f} nDCG {r['ndcg@10']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  {r['latency_median_ms']:.0f} ms", flush=True)
    others = list(idx.all_docs())
    base = float(np.mean([sum(relevance(truth[q["patient_id"]], truth.get(o, [])) > 0 for o in others if o != q["patient_id"])
                          / (len(others) - 1) for q in queries]))
    out = {"answer_key": "relevant = shares >=1 chronic condition; nDCG gain = number shared",
           "n_index": len(p2), "embedder": config.EMBED_MODEL, "rows": rows, "random_ranking_precision": round(base, 4)}
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"## Retrieval, stage by stage ({len(queries)} query patients, top 10, {len(p2)} patients indexed)", "",
          "| Stage | Recall@10 | MRR | nDCG@10 (95% CI) | Latency (median) |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['system']} | {r['recall@10']:.3f} | {r['mrr']:.3f} | {r['ndcg@10']:.3f} "
                  f"({r['ndcg_ci95'][0]:.3f}–{r['ndcg_ci95'][1]:.3f}) | {r['latency_median_ms']:.0f} ms |")
    md += ["", f"A random ranking gives P@10 = {base:.3f}."]
    (RESULTS / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
