## Latency (CPU, 8 threads, 100 patient queries, nothing else running)

| System | Median | 95th percentile | Main stages (median) |
|---|---|---|---|
| BM25 only | 76 ms | 108 ms | bm25 76 ms |
| Vector only | 20 ms | 29 ms | embed 0 ms, vector 19 ms |
| Hybrid (RRF) | 116 ms | 181 ms | embed 0 ms, vector 32 ms, bm25 115 ms, fusion 0 ms |
| Hybrid + learned ranker (default) | 151 ms | 185 ms | embed 1 ms, vector 42 ms, bm25 141 ms, fusion 0 ms, ranker 9 ms |
