# Patient Similarity Search

Find patients who look like a given patient in an electronic health record, see what conditions those look-alikes have, estimate the patient's chronic-disease risk, and compare any two patients side by side.

It's built on hybrid retrieval (BM25 + fine-tuned embeddings in Qdrant), reciprocal rank fusion and a LightGBM LambdaRank re-ranker, and every step is measured with a leakage-free evaluation. An LLM is used for exactly one job: turning a plain-language search into filters.

> All data is **synthetic** (a Synthea EHR export from a university ML assignment). No real patients. Not medical advice.

---

## Highlights

- **Ranking quality more than doubled:** nDCG@10 went from **0.188 → 0.458** over keyword search alone. Each step is measured on the same 200 held-out patients, with 95% bootstrap confidence intervals and paired tests.
- **Fine-tuned models:**
  - a **LambdaRank learned ranker** trained on 1,200 patients (+0.055 nDCG, p < 0.001);
  - a **fine-tuned MiniLM embedder** trained on 3,385 patient triplets (+0.019, p = 0.009).
- **Measured freshness:** re-indexing by patient ID removes **47.5% stale results** and improves ranking (+0.047 nDCG) on held-out patients.
- **Plain-language search:** "women over 60 with high BMI and rising glucose" becomes filters.
  - On everyday wording, Gemini Flash-Lite reaches **F1 0.909** versus 0.475 for keyword rules, at under a second per search.
  - The rules remain the instant fallback, so search never breaks.
- **Fast:** the default search answers in about **150 ms** (median) on CPU.

## Results

### Improvement, step by step (200 query patients, top 10)

| Step | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Keyword search (BM25 on raw summaries) | 0.479 | 0.601 | 0.188 |
| + BM25 on clinical-state tokens (`hba1c_high`, `age_60s`, …) | 0.612 | 0.722 | 0.326 |
| + vector search, reciprocal rank fusion and a pointwise ranker | 0.698 | 0.803 | 0.385 |
| + **LambdaRank** ranker (graded relevance, 1,200 training patients) | 0.699 | 0.799 | 0.439 |
| + **fine-tuned embedder** (final system) | **0.714** | **0.808** | **0.458** |

- **Answer key:** two patients are "similar" if they share a diagnosed chronic condition, and nDCG gains count how many they share. Diagnoses are never part of the searched text.
- **Random baseline:** a random ranking scores P@10 = 0.281.
- **Leakage rule:** the evaluation patients are excluded from all training data.

### Final search, stage by stage

| Stage | Recall@10 | MRR | nDCG@10 (95% CI) | Median latency |
|---|---|---|---|---|
| BM25 (words + clinical flags) | 0.612 | 0.722 | 0.326 (0.292–0.361) | 76 ms |
| Vector search (fine-tuned MiniLM) | 0.593 | 0.687 | 0.274 (0.242–0.308) | 20 ms |
| Hybrid (RRF, k = 30) | 0.624 | 0.729 | 0.344 (0.310–0.380) | 116 ms |
| **Hybrid + LambdaRank (default)** | **0.714** | **0.808** | **0.458 (0.414–0.502)** | **151 ms** |

**Where the ceiling is:**
- A perfect re-ranker on the same top-50 candidates would reach 0.798, so the ranking step still has headroom.
- Conditions with a lab signature are found reliably: type 2 diabetes and high triglycerides (99% of top-10 results share them), and diabetic kidney disease (93%).
- Conditions with no trace in labs or vitals (scoliosis, dermatitis, epilepsy) stay near chance.

### Keeping the index fresh (177 held-out patients with new or changed records)

| What a stale index gets wrong | Stale (period 1) | Re-indexed (period 2) |
|---|---|---|
| Top-10 results showing an outdated record | 47.5% | 0% |
| Relevant patients missing from the index | 5.3% | 0% |
| New patients that can appear in results | never | 9% of top 10 |
| nDCG@10 | 0.415 | 0.462 (+0.047, CI +0.021 to +0.073) |

The re-index upserts by patient ID:
- 472 new patients added;
- 1,618 changed records re-embedded;
- 185 metadata-only updates written without re-embedding;
- no duplicates.

### Plain-language search parser: the only LLM use

| Test set | Keyword rules | Gemini 3.5 Flash-Lite |
|---|---|---|
| 40 standard searches (rules written alongside them) | F1 0.996 | F1 0.991 · 0 failures · 766 ms |
| **15 differently-worded searches** | **F1 0.475** | **F1 0.909** · 1.4 s |

- **Validation and fallback:** the LLM's output is validated against a Pydantic schema. If the call fails, times out or returns invalid JSON, the rule parser answers instead.
- **Model choice:** Gemini 3.5 Flash was tested too. It was too slow (5–50 s) and ran out of free-tier quota.

### Risk model (trained before 2015, tested on later patients)

| Model | ROC-AUC (later period) | F1 |
|---|---|---|
| **Decision Tree (selected)** | **0.903** | 0.790 |
| SVM (calibrated) | 0.869 | 0.794 |
| MLP | 0.841 | 0.757 |

- **Target:** diabetes, hypertension, prediabetes or metabolic syndrome.
- **Explanations:** occlusion. Each feature is set to its typical value and the change in risk is reported.

## How it works

```
csv tables ─► patient summaries (vitals, labs, trends, visits, insurance, social context)
            ─► Qdrant (fine-tuned MiniLM vectors, upsert by patient ID) + BM25 (clinical-state tokens)

query ─► BM25 top-50 ──┐
      └► vector top-50 ┴► reciprocal rank fusion ─► LambdaRank re-ranker ─► top 10
                                                    (age, lab and vital distances, flag agreement, trends)

"women over 60 with high BMI" ─► Gemini (or rules) ─► {sex, age, insurance, lab flags, trends} ─► filtered search
```

**The answer key is rebuilt from several sources.** `conditions.csv` in the export was truncated (108 patients). Diagnoses are therefore rebuilt from conditions, the SNOMED reason codes on encounters, medications, procedures and care plans, and the diagnosis codes on insurance claims, for 1,443 labelled patients.

**The web app** works without ever needing a patient ID:
- start from named demo cases, a plain-language search, or a condition;
- read an overview with risk and its reasons, the 10 closest patients with plain-language "why similar", and their common conditions;
- compare any two patients, with a verdict and dot plots showing normal ranges;
- keep a shortlist and a recently-viewed list.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python scripts/build_documents.py      # csv/ -> artifacts/documents_period{1,2}.jsonl
python scripts/train_risk_model.py     # Decision Tree / SVM / MLP, best kept
python evals/finetune.py               # LambdaRank ranker + fine-tuned embedder (CPU: ~2.5 h)
python scripts/build_index.py          # period 1, then the period-2 re-index
uvicorn simsearch.api:app --port 8000  # app at http://localhost:8000
```

**Optional LLM parser:** copy `.env.example` to `.env` and add a Gemini or Groq key. The provider is detected from the key. Check it with `python scripts/check_llm.py`. Without a key, the rule parser is used.

**Tests and evaluations:**

```bash
pytest -q                                                  # 36 tests on a small synthetic fixture
python evals/retrieval_eval.py                             # stage-by-stage table
python evals/freshness.py                                  # stale vs re-indexed
python evals/parser_eval.py --models gemini-3.5-flash-lite # rules vs LLM parser
python evals/latency.py                                    # CPU latency per stage
```

CI (GitHub Actions) runs the tests and a retrieval check on every push.

## API

| Endpoint | Purpose |
|---|---|
| `POST /search` | Similar patients for a patient ID or text, with optional age / sex / insurance filters |
| `POST /query` | Plain-language search → filters → results, with matched / missing conditions |
| `POST /report` | One patient's overview: matches, shared conditions, risk and reasons |
| `GET /compare?a=&b=` | Two patients side by side |
| `GET /patients/showcase` · `/patients/browse` · `/patients/conditions` | Find patients without an ID |
| `POST /reindex` | Upsert a newer snapshot by patient ID |
| `GET /health` | Index size and models |

## Project structure

```
simsearch/   documents, index (Qdrant + BM25), search, risk, query_parser, llm, api, static/ (web app)
scripts/     build_documents, train_risk_model, build_index, check_llm, ci_eval
evals/       retrieval_eval, finetune, freshness, parser_eval, latency, diagnose; results/ (+ history/)
tests/       unit, search and API tests with a 240-patient synthetic fixture
```

## Limitations and future work

- **Synthetic data, no clinical validation.** The results show the method works, not that it has clinical value.
- **Some conditions can't be found from labs and vitals alone,** which caps recall on this answer key. Medications and procedures are deliberately left out of the search text to keep the evaluation fair.
- **Training ran on CPU,** so the embedder took about 2 h. On a GPU it would take minutes, and allow larger models and longer training.
- **Next:** per-condition reporting with confidence intervals; a structured-only kNN baseline; a larger candidate pool for the re-ranker; and a scale test with more synthetic patients.

**Stack:** Python · FastAPI · Qdrant · sentence-transformers · rank-bm25 · LightGBM · scikit-learn · Gemini (one task) · Docker · GitHub Actions
