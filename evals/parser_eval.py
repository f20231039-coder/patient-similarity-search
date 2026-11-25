"""Search-parser evaluation: plain-language query -> filters.

40 hand-written queries with expected filters (evals/parser_queries.json). Every
expected or predicted value is an atomic fact (gender=female, labs.bmi=high, ...);
we report precision, recall and F1 over facts, the share of queries parsed
exactly, and latency. The rule-based parser always runs; the LLM parser runs
when a key is configured (.env or environment).

End-to-end check: for each query, the share of the top 10 strict-mode results
that satisfy every expected lab/trend/smoking condition and filter.

Run:  python evals/parser_eval.py [--no-search]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsearch import config  # noqa: E402
from simsearch.llm import LLMError, get_llm  # noqa: E402
from simsearch.query_parser import LLMParser, ParsedQuery, RuleParser, run_query, satisfies  # noqa: E402

QUERIES = ROOT / "evals" / "parser_queries.json"
RESULTS = ROOT / "evals" / "results"


def facts(d: dict) -> set[tuple]:
    out = set()
    for k in ("gender", "insurance", "smoking"):
        if d.get(k):
            out.add((k, d[k]))
    for k in ("age_min", "age_max"):
        if d.get(k) is not None:
            out.add((k, round(float(d[k]))))
    for k, v in (d.get("labs") or {}).items():
        out.add((f"labs.{k}", v))
    for k, v in (d.get("trends") or {}).items():
        out.add((f"trends.{k}", v))
    return out


def align_ages(p: set, e: set) -> set:
    """'under 45' may fairly mean age <= 44 or <= 45: accept age bounds within 1 year."""
    out = set()
    for k, v in p:
        if k in ("age_min", "age_max"):
            match = next((ev for ek, ev in e if ek == k and abs(ev - v) <= 1), None)
            out.add((k, match if match is not None else v))
        else:
            out.add((k, v))
    return out


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def score(parser, queries, delay: float = 0.0) -> dict:
    tp = fp = fn = exact = 0
    a_tp = a_fp = a_fn = 0  # answered queries only (failures excluded)
    lat, errors, rows = [], 0, []
    for i, item in enumerate(queries):
        if delay and i:
            time.sleep(delay)
        t = time.perf_counter()
        failed = False
        try:
            pred = parser.parse(item["q"]).model_dump(exclude={"text"})
        except (LLMError, ValueError) as e:
            pred, errors, failed = {}, errors + 1, True
            rows.append({"q": item["q"], "error": str(e)})
        lat.append((time.perf_counter() - t) * 1000)
        e = facts(item["expect"])
        p = align_ages(facts(pred), e)
        tp += len(p & e)
        fp += len(p - e)
        fn += len(e - p)
        if not failed:
            a_tp += len(p & e)
            a_fp += len(p - e)
            a_fn += len(e - p)
        exact += p == e
        if p != e and not failed:
            rows.append({"q": item["q"], "missing": sorted(map(str, e - p)), "extra": sorted(map(str, p - e))})
    prec, rec, f1 = _prf(tp, fp, fn)
    _, _, f1_answered = _prf(a_tp, a_fp, a_fn)
    return {"parser": parser.name, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "f1_answered_only": round(f1_answered, 3), "exact_match": round(exact / len(queries), 3),
            "errors": errors, "latency_median_ms": round(statistics.median(lat), 1), "mistakes": rows}


def end_to_end(engine, queries) -> dict:
    """Strict search driven by the EXPECTED filters: share of top-10 meeting them all."""
    shares, full = [], []
    for item in queries:
        pq = ParsedQuery(**item["expect"], text=item["q"])
        res = run_query(engine, pq, k=10, strict=True)
        docs = {d["short_id"]: d for d in engine.index.all_docs().values()}
        ok = 0
        for r in res["results"]:
            d = docs[r["short_id"]]
            f = pq.filters()
            ok += f.matches(d) and all(satisfies(d, *c) for c in pq.constraints())
        shares.append(ok / max(1, len(res["results"])))
        full.append(res["n_full_matches"])
    return {"top10_all_conditions_met": round(statistics.mean(shares), 3),
            "queries_with_10_plus_full_matches": sum(n >= 10 for n in full), "n_queries": len(queries)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated model names to compare (same provider)")
    ap.add_argument("--delay", type=float, default=4.5, help="seconds between LLM calls (free-tier rate limits)")
    ap.add_argument("--queries", default=str(QUERIES), help="query file")
    ap.add_argument("--tag", default="", help="suffix for the result files")
    a = ap.parse_args()
    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    out = {"n_queries": len(queries), "parsers": [score(RuleParser(), queries)]}
    llm = get_llm()
    if llm is not None:
        from simsearch.llm import OpenAICompatLLM
        models = [m for m in a.models.split(",") if m] or [llm.model]
        for m in models:
            print(f"scoring {llm.provider}:{m} ...", flush=True)
            out["parsers"].append(score(LLMParser(OpenAICompatLLM(llm.provider, m, config.LLM_API_KEY, config.LLM_BASE_URL,
                                                                    timeout=90, retries=3)), queries, delay=a.delay))
    else:
        out["note"] = "No LLM key configured: only the rule-based parser was scored. Paste a key into .env and rerun."
    if not a.no_search:
        from simsearch.index import PatientIndex
        from simsearch.search import SearchEngine
        out["end_to_end"] = end_to_end(SearchEngine(PatientIndex(collection="eval_patients")), queries)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"parser_eval{a.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"## Search parser: plain language -> filters ({len(queries)} queries{', ' + a.tag.strip('_') if a.tag else ''})", "",
          "| Parser | Precision | Recall | F1 | F1 (answered only) | Exact match | Failed calls | Median latency |",
          "|---|---|---|---|---|---|---|---|"]
    for p in out["parsers"]:
        md.append(f"| {p['parser']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['f1_answered_only']:.3f} | "
                  f"{p['exact_match']:.3f} | {p['errors']} | {p['latency_median_ms']:.0f} ms |")
    md += ["", "Age bounds within 1 year count as correct ('under 45' = 44 or 45). The rule parser was written alongside "
               "these queries, so its score is an upper bound; the LLM was not tuned on them. Failed calls (timeouts / "
               "rate limits) count as empty answers in F1; 'answered only' excludes them."]
    if "note" in out:
        md += ["", out["note"]]
    if "end_to_end" in out:
        e = out["end_to_end"]
        md += ["", f"End to end (strict mode, expected filters): {e['top10_all_conditions_met']:.1%} of top-10 results meet "
                   f"every condition; {e['queries_with_10_plus_full_matches']}/{e['n_queries']} queries have at least 10 full matches."]
    (RESULTS / f"parser_eval{a.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    for p in out["parsers"]:
        for m in p["mistakes"][:12]:
            print(p["parser"], "|", m)


if __name__ == "__main__":
    main()
