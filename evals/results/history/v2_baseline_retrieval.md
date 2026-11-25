## Retrieval comparison (200 query patients, top 10, up-to-date index)

| System | Recall@10 | MRR | nDCG@10 | P@10 | Latency (median) |
|---|---|---|---|---|---|
| BM25 only | 0.479 | 0.601 | 0.188 | 0.479 | 57 ms |
| Vector search only | 0.578 | 0.677 | 0.274 | 0.578 | 83 ms |
| Hybrid (RRF) | 0.539 | 0.680 | 0.255 | 0.539 | 170 ms |
| Hybrid + cross-encoder re-rank | 0.519 | 0.590 | 0.212 | 0.519 | 6013 ms |
| Hybrid + learned ranker (structured features) | 0.641 | 0.768 | 0.338 | 0.641 | 205 ms |

Random ranking would give P@10 = 0.281 (the share of the index that is relevant to a query, on average).

## Changing-index check (period-2 queries)

| System | nDCG@10 before re-index | after | Recall@10 before | after |
|---|---|---|---|---|
| Vector search only | 0.286 | 0.271 | 0.605 | 0.589 |
| Hybrid (RRF) | 0.288 | 0.252 | 0.608 | 0.567 |
| Hybrid + cross-encoder re-rank | 0.244 | 0.239 | 0.578 | 0.550 |
| Hybrid + learned ranker (structured features) | 0.337 | 0.345 | 0.666 | 0.656 |

Re-index: 472 added, 1618 re-embedded, 185 payload-only updates, 0 unchanged; 1803 -> 2275 patients (2275 Qdrant points, no duplicates).
