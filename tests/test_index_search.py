import pytest

from simsearch.index import Filters, PatientIndex
from simsearch.search import SearchEngine


def test_reindex_upserts_without_duplicates(docs1, docs2):
    idx = PatientIndex(collection="test_reindex")
    idx.reset()
    s1 = idx.upsert(docs1)
    assert s1 == {"added": len(docs1), "updated": 0, "payload_only": 0, "unchanged": 0}
    assert idx.count() == len(docs1)

    s2 = idx.upsert(docs2)
    ids1 = {d["patient_id"]: d["content_hash"] for d in docs1}
    assert s2["added"] == sum(d["patient_id"] not in ids1 for d in docs2)
    assert s2["added"] + s2["updated"] + s2["payload_only"] + s2["unchanged"] == len(docs2)
    assert s2["updated"] == sum(d["patient_id"] in ids1 and ids1[d["patient_id"]] != d["content_hash"] for d in docs2)
    assert idx.count() == len({d["patient_id"] for d in docs1} | {d["patient_id"] for d in docs2})
    # every re-indexed record now carries the newer snapshot
    assert all(idx.get(d["patient_id"])["period"] == 2 for d in docs2)

    s3 = idx.upsert(docs2)  # idempotent: nothing re-embedded
    assert s3["unchanged"] == len(docs2) and s3["added"] == s3["updated"] == s3["payload_only"] == 0


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "ltr"])
def test_modes_return_k_results_excluding_self(engine, docs2, mode):
    q = docs2[0]
    res = engine.search(patient_id=q["short_id"], k=5, mode=mode)
    ids = [r["patient_id"] for r in res["results"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert q["patient_id"] not in ids
    assert res["timings"]["total_ms"] > 0


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "ltr"])
def test_filters_are_respected(engine, mode):
    f = Filters(age_min=50, gender="female")
    res = engine.search(query="high blood pressure, obese, HbA1c high", k=8, filters=f, mode=mode)
    assert res["results"], "expected some matches"
    for r in res["results"]:
        assert r["age"] >= 50 and r["gender"] == "female"


def test_text_query_and_unknown_patient(engine):
    assert engine.search(query="child with asthma", k=3, mode="hybrid")["results"]
    with pytest.raises(KeyError):
        engine.search(patient_id="P-00000000", mode="hybrid")
    with pytest.raises(ValueError):
        engine.search(mode="hybrid")


def test_reset_really_clears(docs1):
    idx = PatientIndex(collection="test_reset")
    idx.reset()
    idx.upsert(docs1[:20])
    assert idx.count() == 20
    idx.reset()
    assert idx.count() == 0 and len(idx) == 0


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_insurance_filter(engine, mode):
    res = engine.search(query="older patient with high blood pressure", k=5,
                        filters=Filters(insurance="medicare"), mode=mode)
    assert res["results"] and all(r["insurance"] == "medicare" for r in res["results"])


def test_insurance_filter_rejects_unknown_value():
    with pytest.raises(ValueError):
        Filters(insurance="gold-plan")


def test_flag_bm25_and_weighted_rrf(engine, docs2):
    from simsearch import config
    from simsearch.index import bm25_doc_tokens
    from simsearch.search import rrf
    toks = bm25_doc_tokens(docs2[0])
    assert toks and not any(t[0].isdigit() for t in toks)
    assert any(t.startswith("age_") for t in toks) and any(t.startswith("sex_") for t in toks)
    old = config.RRF_W_BM25
    try:
        config.RRF_W_BM25 = 0.5
        for mode in ("bm25", "hybrid"):
            res = engine.search(patient_id=docs2[0]["short_id"], k=5, mode=mode, filters=Filters(gender="female"))
            assert res["results"] and all(r["gender"] == "female" for r in res["results"])
            assert docs2[0]["patient_id"] not in [r["patient_id"] for r in res["results"]]
    finally:
        config.RRF_W_BM25 = old
    s = rrf([["a", "b"], ["b", "a"]], k=60, weights=[0.5, 1.0])
    assert s["b"] > s["a"]  # the more heavily weighted list wins ties in rank
