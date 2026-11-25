"""Retrieval metrics with a condition-based answer key.

Two patients are similar if they share a diagnosed chronic condition; the graded
relevance used by nDCG is the number of chronic conditions they share. Conditions
never appear in the searchable text, so this is a fair test.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def relevance(query_conditions: Iterable[str], cand_conditions: Iterable[str]) -> int:
    return len(set(query_conditions) & set(cand_conditions))


def recall_at_k(rels: Sequence[int], n_relevant: int, k: int = 10) -> float:
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for r in rels[:k] if r > 0)
    return hits / min(k, n_relevant)


def mrr_at_k(rels: Sequence[int], k: int = 10) -> float:
    for i, r in enumerate(rels[:k], 1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(rels: Sequence[int], all_rels: Sequence[int], k: int = 10) -> float:
    ideal = dcg(sorted(all_rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def precision_at_k(rels: Sequence[int], k: int = 10) -> float:
    return sum(1 for r in rels[:k] if r > 0) / k


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean (resampling queries)."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n: int = 2000, seed: int = 0) -> dict:
    """Is system b better than a on the same queries? Mean difference, 95% CI, one-sided p."""
    import numpy as np
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_value": float((means <= 0).mean())}
