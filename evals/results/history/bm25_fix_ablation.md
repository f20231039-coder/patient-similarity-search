## Retrieval fixes, one step at a time (200 evaluation queries, top 10)

| Step | System | Recall@10 | MRR | nDCG@10 | Latency |
|---|---|---|---|---|---|
| 0 | Learned ranker (v2 baseline) | 0.641 | 0.768 | 0.338 | 205 ms |
| 0 | BM25, raw tokens (baseline) | 0.479 | 0.601 | 0.188 | 81 ms |
| 0 | Vector search (unchanged by these fixes) | 0.578 | 0.677 | 0.274 | 72 ms |
| 0 | Hybrid RRF, raw BM25 (baseline) | 0.539 | 0.680 | 0.255 | 141 ms |
| 1 | BM25 on words + clinical flags | 0.612 | 0.722 | 0.326 | 63 ms |
| 1 | Hybrid RRF, flag BM25 | 0.593 | 0.698 | 0.307 | 164 ms |
| 2 | Hybrid weighted RRF (w_bm25=1.0, k=30), flag BM25 | 0.597 | 0.695 | 0.308 | 193 ms |
| 3 | Learned ranker, retrained on the new candidates | 0.698 | 0.803 | 0.385 | 242 ms |

Weights tuned on 150 training queries (never the evaluation set). Best: w_bm25=1.0, k=30 (train nDCG@10 0.310).

| w_bm25 \ k | 30 | 60 | 100 |
|---|---|---|---|
| 0.25 | 0.288 | 0.297 | 0.303 |
| 0.5 | 0.300 | 0.307 | 0.307 |
| 0.75 | 0.306 | 0.305 | 0.305 |
| 1.0 | 0.310 | 0.308 | 0.307 |
