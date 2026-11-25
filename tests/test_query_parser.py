import pytest

from simsearch.query_parser import LLMParser, ParsedQuery, RuleParser, parse_query, run_query, satisfies
from simsearch.llm import LLMError


@pytest.mark.parametrize("q,expect", [
    ("women over 60 with high BMI and rising glucose",
     {"gender": "female", "age_min": 60, "labs": {"bmi": "high"}, "trends": {"glucose": "rising"}}),
    ("obese male smokers in their 50s",
     {"gender": "male", "age_min": 50, "age_max": 59, "smoking": "current", "labs": {"bmi": "high"}}),
    ("diabetic patients with worsening kidney function", {"labs": {"glucose": "high"}, "trends": {"egfr": "falling"}}),
    ("patients with high bad cholesterol", {"labs": {"ldl": "high"}}),
])
def test_rule_parser(q, expect):
    got = RuleParser().parse(q).model_dump(exclude={"text"}, exclude_none=True)
    got = {k: v for k, v in got.items() if v != {}}
    assert got == expect


def test_schema_drops_unknown_labs():
    pq = ParsedQuery(labs={"bmi": "high", "made_up": "high"}, trends={"hdl": "rising", "glucose": "falling"})
    assert pq.labs == {"bmi": "high"} and pq.trends == {"glucose": "falling"}


class BrokenLLM:
    name = "broken"

    def complete_json(self, system, user):
        raise LLMError("boom")


class FakeLLM:
    name = "fake"

    def complete_json(self, system, user):
        return {"gender": "female", "labs": {"bmi": "high", "nonsense": "x"}, "extra_key": 1}


def test_llm_failure_falls_back_to_rules():
    pq, meta = parse_query("men over 50 with high LDL", parser=LLMParser(BrokenLLM()))
    assert meta["parser"] == "rules (LLM fallback)" and pq.gender == "male" and pq.labs == {"ldl": "high"}


def test_llm_output_is_validated():
    with pytest.raises(ValueError):  # invalid lab level -> schema error -> caller falls back
        LLMParser(FakeLLM()).parse("x")
    pq, meta = parse_query("women", parser=LLMParser(FakeLLM()))
    assert meta["parser"] == "rules (LLM fallback)" and pq.gender == "female"


def test_strict_search_meets_all_conditions(engine):
    pq = ParsedQuery(gender="female", labs={"bmi": "high"}, text="obese women")
    res = run_query(engine, pq, k=5, strict=True)
    docs = {d["short_id"]: d for d in engine.index.all_docs().values()}
    full = [r for r in res["results"] if r["matches_all"]]
    assert res["n_full_matches"] >= len(full) > 0
    for r in full:
        d = docs[r["short_id"]]
        assert d["gender"] == "female" and satisfies(d, "lab", "bmi", "high")
    assert [c["label"] for c in res["chips"]] == ["Women", "BMI high"]
