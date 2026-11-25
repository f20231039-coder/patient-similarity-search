import math

from simsearch.documents import render_text, short_id
from simsearch.metrics import mrr_at_k, ndcg_at_k, recall_at_k, relevance
from simsearch.search import rrf


def test_conditions_never_in_search_text(docs2):
    for d in docs2:
        low = d["text"].lower()
        for c in d["conditions"]:
            assert c.lower() not in low, (d["short_id"], c)


def test_short_ids_unique_and_render_deterministic(docs2):
    ids = [d["short_id"] for d in docs2]
    assert len(ids) == len(set(ids))
    assert all(s == short_id(d["patient_id"]) for s, d in zip(ids, docs2))
    assert render_text(docs2[0]) == docs2[0]["text"]


def test_metrics_known_values():
    rels = [0, 2, 0, 1]
    assert relevance(["a", "b"], ["b", "c"]) == 1
    assert mrr_at_k(rels) == 0.5
    assert recall_at_k(rels, n_relevant=2, k=4) == 1.0
    assert recall_at_k(rels, n_relevant=40, k=4) == 0.5
    assert math.isclose(ndcg_at_k([2, 1], [2, 1, 0]), 1.0)
    assert ndcg_at_k([0, 0], [1]) == 0.0


def test_rrf_rewards_agreement():
    s = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] > s["c"] > 0
    assert s["c"] == s["d"]
