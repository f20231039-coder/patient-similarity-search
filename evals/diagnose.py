"""Why are the scores where they are? Ceilings and bottlenecks for the current system.

1. Metric ceiling: nDCG@10 if all 10 results were relevant but shared only ONE
   condition each (the ideal ranking puts patients sharing 3-6 conditions first).
2. Candidate ceiling: how many of a query's relevant patients reach the top-50
   pool the re-ranker sees, and the best nDCG@10 a perfect re-ranker could get
   from that pool.
3. Per-condition hit rate: for each query condition, the share of the final top 10
   that also has it. Conditions with no trace in labs/vitals stay near chance.

Run:  python evals/diagnose.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import numpy as np  # noqa: E402

from simsearch import config  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex, embed  # noqa: E402
from simsearch.metrics import dcg, ndcg_at_k, relevance  # noqa: E402
from simsearch.search import SearchEngine, rrf  # noqa: E402
from retrieval_eval import RESULTS, pick_queries  # noqa: E402


def main() -> None:
    p2 = load_jsonl(config.DOCS_PERIOD2)
    truth = {d["patient_id"]: d["chronic_conditions"] for d in p2}
    eval_q = pick_queries(p2, 200, 42)
    idx = PatientIndex(collection="eval_patients")
    engine = SearchEngine(idx)
    ids = list(idx.all_docs())
    n = config.CANDIDATES_PER_RETRIEVER

    one_shared, pool_recall, pool_ceiling, final_ndcg, n_rel = [], [], [], [], []
    per_cond = defaultdict(lambda: [0, 0])  # condition -> [hits in top10, slots]
    prevalence = defaultdict(int)
    for d in p2:
        for c in d["chronic_conditions"]:
            prevalence[c] += 1

    for q in eval_q:
        pid, qc = q["patient_id"], truth[q["patient_id"]]
        all_rels = [relevance(qc, truth.get(o, [])) for o in ids if o != pid]
        ideal = sorted(all_rels, reverse=True)
        n_rel.append(sum(r > 0 for r in all_rels))
        one_shared.append(dcg([1] * 10, 10) / dcg(ideal, 10))

        qvec = embed([q["text"]])[0]
        bm = idx.bm25_search(q["text"], n, None, pid, query_doc=q)
        vec = idx.vector_search(qvec, n, None, pid)
        pool = list(rrf([[p for p, _ in bm], [p for p, _ in vec]], weights=[config.RRF_W_BM25, config.RRF_W_VEC]))
        pool = sorted(pool, key=lambda p: -relevance(qc, truth.get(p, [])))
        rels_pool = [relevance(qc, truth.get(p, [])) for p in pool]
        pool_recall.append(sum(r > 0 for r in rels_pool) / max(1, n_rel[-1]))
        pool_ceiling.append(ndcg_at_k(rels_pool, all_rels, 10))

        res = engine.search(query_doc=q, k=10, mode="ltr")["results"]
        rels = [relevance(qc, truth.get(r["patient_id"], [])) for r in res]
        final_ndcg.append(ndcg_at_k(rels, all_rels, 10))
        for c in qc:
            per_cond[c][0] += sum(c in truth.get(r["patient_id"], []) for r in res)
            per_cond[c][1] += len(res)

    total = len(p2)
    cond_rows = sorted(((c, h / s, prevalence[c] / total, s // 10) for c, (h, s) in per_cond.items() if s >= 50),
                       key=lambda x: -(x[1] / max(x[2], 1e-9)))
    out = {
        "mean_relevant_per_query": float(np.mean(n_rel)),
        "ndcg_if_all10_share_one_condition": float(np.mean(one_shared)),
        "candidate_pool_size": "hybrid top-50 union",
        "share_of_relevant_reaching_pool": float(np.mean(pool_recall)),
        "perfect_reranker_ndcg_from_pool": float(np.mean(pool_ceiling)),
        "current_final_ndcg": float(np.mean(final_ndcg)),
        "per_condition": [{"condition": c, "hit_rate_top10": round(h, 3), "prevalence": round(p, 3),
                           "lift": round(h / p, 2), "queries": nq} for c, h, p, nq in cond_rows],
    }
    (RESULTS / "diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_condition"}, indent=2))
    print("\ncondition | hit rate in top10 | prevalence (random) | lift | queries")
    for r in out["per_condition"]:
        print(f"{r['condition'][:45]:45s} | {r['hit_rate_top10']:.2f} | {r['prevalence']:.2f} | {r['lift']:.1f}x | {r['queries']}")


if __name__ == "__main__":
    main()
