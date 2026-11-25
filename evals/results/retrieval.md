## Retrieval, stage by stage (200 query patients, top 10, 2275 patients indexed)

| Stage | Recall@10 | MRR | nDCG@10 (95% CI) | Latency (median) |
|---|---|---|---|---|
| BM25 (words + clinical flags) | 0.612 | 0.722 | 0.326 (0.292–0.361) | 88 ms |
| Vector search (fine-tuned embedder) | 0.593 | 0.687 | 0.274 (0.242–0.308) | 107 ms |
| Hybrid (RRF) | 0.624 | 0.729 | 0.344 (0.310–0.380) | 221 ms |
| Hybrid + learned ranker (default) | 0.714 | 0.808 | 0.458 (0.414–0.502) | 259 ms |

A random ranking gives P@10 = 0.281.
