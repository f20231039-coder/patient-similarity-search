## Search parser: plain language -> filters (40 queries)

| Parser | Precision | Recall | F1 | F1 (answered only) | Exact match | Failed calls | Median latency |
|---|---|---|---|---|---|---|---|
| rules | 1.000 | 0.991 | 0.996 | 0.996 | 0.975 | 0 | 1 ms |
| llm:gemini:gemini-3.5-flash-lite | 0.982 | 1.000 | 0.991 | 0.991 | 0.975 | 0 | 766 ms |
| llm:gemini:gemini-3.5-flash | 0.949 | 0.330 | 0.490 | 0.974 | 0.350 | 25 | 50962 ms |

Age bounds within 1 year count as correct ('under 45' = 44 or 45). The rule parser was written alongside these queries, so its score is an upper bound; the LLM was not tuned on them. Failed calls (timeouts / rate limits) count as empty answers in F1; 'answered only' excludes them.
