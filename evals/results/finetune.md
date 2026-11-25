## Fine-tuning round (200 evaluation queries, top 10; 95% bootstrap CI on nDCG@10)

| Stage | System | Recall@10 | MRR | nDCG@10 | 95% CI | Latency |
|---|---|---|---|---|---|---|
| before | Vector search (current embedder) | 0.578 | 0.677 | 0.274 | 0.239–0.311 | 92 ms |
| before | Hybrid RRF, flag BM25 (current) | 0.597 | 0.695 | 0.308 | 0.272–0.346 | 182 ms |
| before | Learned ranker (current default) | 0.699 | 0.799 | 0.439 | 0.397–0.483 | 218 ms |
| embed | Vector search, fine-tuned embedder | 0.593 | 0.687 | 0.274 | 0.242–0.308 | 87 ms |
| embed | Hybrid RRF, fine-tuned embedder | 0.624 | 0.729 | 0.344 | 0.310–0.380 | 184 ms |
| ltr | LambdaRank learned ranker (fine-tuned embedder) | 0.714 | 0.808 | 0.458 | 0.414–0.502 | 222 ms |

Paired tests (same queries): difference in nDCG@10, 95% CI, one-sided p (b not better than a).

| a | b | Δ nDCG@10 | 95% CI | p |
|---|---|---|---|---|
| Vector search (current embedder) | Vector search, fine-tuned embedder | +0.000 | -0.020 to +0.021 | 0.493 |
| Hybrid RRF, flag BM25 (current) | Hybrid RRF, fine-tuned embedder | +0.036 | +0.019 to +0.054 | 0.000 |
| Learned ranker (current default) | LambdaRank learned ranker (fine-tuned embedder) | +0.019 | +0.003 to +0.035 | 0.009 |

- 200 evaluation queries; 1200 training queries; the evaluation patients are excluded from all training data (queries, candidates, positives, negatives).
- Embedder: all-MiniLM-L6-v2 fine-tuned 2 epochs (7,817 s on CPU) on 3,385 (anchor, positive, hard negative) triplets from training patients only; positives share the most conditions.
- fine-tuned embedder: LambdaRank trained on 1200 training queries / 53140 pairs; CV-chosen params {'num_leaves': 31, 'learning_rate': 0.08, 'n_estimators': 700, 'min_child_samples': 60} (CV nDCG@10 0.650).
