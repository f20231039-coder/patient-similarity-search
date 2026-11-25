## Changing-index check: stale vs re-indexed (177 held-out period-2 queries, learned ranker)

| Metric | Stale index (period 1) | Re-indexed (period 2) | Difference (95% CI) |
|---|---|---|---|
| Relevant patients present in the index | 94.7% | 100.0% | +0.053 (+0.041 to +0.067) |
| Top-10 results showing an outdated record | 47.5% | 0.0% | -0.475 (-0.510 to -0.441) |
| Top-10 results that are new (period-2) patients | 0.0% | 9.0% | +0.090 (+0.062 to +0.120) |
| nDCG@10, same 1,803-patient pool (record freshness only) | 0.415 | 0.462 | +0.047 (+0.021 to +0.073) |
| nDCG@10, full pool (not like-for-like) | 0.415 | 0.462 | +0.047 (+0.022 to +0.073) |

Re-index: 472 added, 1618 re-embedded, 185 payload-only, 2275 points, no duplicates. 15 of the queries are new patients.
