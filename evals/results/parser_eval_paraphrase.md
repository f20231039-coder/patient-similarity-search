## Search parser: plain language -> filters (15 queries, paraphrase)

| Parser | Precision | Recall | F1 | F1 (answered only) | Exact match | Failed calls | Median latency |
|---|---|---|---|---|---|---|---|
| rules | 0.875 | 0.326 | 0.475 | 0.475 | 0.000 | 0 | 1 ms |
| llm:gemini:gemini-3.5-flash-lite | 0.889 | 0.930 | 0.909 | 0.909 | 0.733 | 0 | 1410 ms |

Age bounds within 1 year count as correct ('under 45' = 44 or 45). The rule parser was written alongside these queries, so its score is an upper bound; the LLM was not tuned on them. Failed calls (timeouts / rate limits) count as empty answers in F1; 'answered only' excludes them.
