"""Fine-tuning round. Every stage is measured on the same 200 evaluation queries,
with 95% bootstrap CIs and a paired test against the system before it.

Leakage rule: the 200 evaluation patients are removed from ALL training data —
never a training query, never a training candidate, never a positive or negative.

Stages (run all, or pick with --stages):
  ltr    LightGBM LambdaRank learned ranker: graded labels (number of shared
         conditions), ~1,200 training queries, richer features, grouped-CV tuning
  embed  fine-tune the MiniLM bi-encoder (MultipleNegativesRankingLoss on patients
         sharing conditions, hard negatives from the hybrid pool), rebuild the
         index, retrain the ranker on the new candidates

(A cross-encoder fine-tuning stage was dropped: the base cross-encoder scored nDCG@10 0.212
vs 0.439 for the learned ranker at ~25x the latency, and CPU fine-tuning cost ~2 hours.)

Run:  python evals/finetune.py [--stages ltr,embed]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from simsearch import config  # noqa: E402
from simsearch import index as index_mod  # noqa: E402
from simsearch import search as search_mod  # noqa: E402
from simsearch.documents import load_jsonl  # noqa: E402
from simsearch.index import PatientIndex, embed  # noqa: E402
from simsearch.metrics import bootstrap_ci, paired_bootstrap, relevance  # noqa: E402
from simsearch.search import SearchEngine, ranker_features, rrf  # noqa: E402
from retrieval_eval import RESULTS, evaluate, pick_queries  # noqa: E402

log = logging.getLogger("finetune")
MODELS = config.ARTIFACT_DIR / "models"
OUT = {"rows": [], "tests": [], "notes": []}


# ----------------------------------------------------------------------------- helpers
def record(engine, eval_q, truth, mode, label, stage, compare_to=None):
    r = evaluate(engine, eval_q, truth, mode)
    pq = r.pop("_per_query")
    lo, hi = bootstrap_ci(pq["ndcg@10"])
    r.update(system=label, stage=stage, ndcg_ci95=[round(lo, 4), round(hi, 4)], _ndcg=pq["ndcg@10"])
    OUT["rows"].append(r)
    msg = f"{label:62s} R@10 {r['recall@10']:.3f} MRR {r['mrr']:.3f} nDCG {r['ndcg@10']:.3f} [{lo:.3f}, {hi:.3f}] {r['latency_median_ms']:.0f} ms"
    if compare_to is not None:
        t = paired_bootstrap(compare_to["_ndcg"], pq["ndcg@10"])
        t.update(a=compare_to["system"], b=label)
        OUT["tests"].append(t)
        msg += f" | vs '{compare_to['system'][:30]}': {t['diff']:+.3f} CI [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['p_value']:.3f}"
    log.info(msg)
    save()
    return r


def save():
    rows = [{k: v for k, v in r.items() if k != "_ndcg"} for r in OUT["rows"]]
    (RESULTS / "finetune.json").write_text(json.dumps({**OUT, "rows": rows}, indent=2), encoding="utf-8")
    md = ["## Fine-tuning round (200 evaluation queries, top 10; 95% bootstrap CI on nDCG@10)", "",
          "| Stage | System | Recall@10 | MRR | nDCG@10 | 95% CI | Latency |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['stage']} | {r['system']} | {r['recall@10']:.3f} | {r['mrr']:.3f} | {r['ndcg@10']:.3f} | "
                  f"{r['ndcg_ci95'][0]:.3f}–{r['ndcg_ci95'][1]:.3f} | {r['latency_median_ms']:.0f} ms |")
    if OUT["tests"]:
        md += ["", "Paired tests (same queries): difference in nDCG@10, 95% CI, one-sided p (b not better than a).", "",
               "| a | b | Δ nDCG@10 | 95% CI | p |", "|---|---|---|---|---|"]
        for t in OUT["tests"]:
            md.append(f"| {t['a']} | {t['b']} | {t['diff']:+.3f} | {t['ci95'][0]:+.3f} to {t['ci95'][1]:+.3f} | {t['p_value']:.3f} |")
    if OUT["notes"]:
        md += [""] + [f"- {n}" for n in OUT["notes"]]
    (RESULTS / "finetune.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def pool_for(idx, q, n):
    qvec = embed([q["text"]])[0]
    bm = idx.bm25_search(q["text"], n, None, q["patient_id"], query_doc=q)
    vec = idx.vector_search(qvec, n, None, q["patient_id"])
    bm_rank = {p: r for r, (p, _) in enumerate(bm, 1)}
    vec_rank = {p: r for r, (p, _) in enumerate(vec, 1)}
    fused = sorted(rrf([[p for p, _ in bm], [p for p, _ in vec]], weights=[config.RRF_W_BM25, config.RRF_W_VEC]).items(),
                   key=lambda x: -x[1])[:n]
    return fused, bm, bm_rank, vec_rank, dict(vec)


# ----------------------------------------------------------------------------- stage: ltr
def build_ltr_data(idx, train_q, truth, banned, n):
    docs = idx.all_docs()
    X, y, groups = [], [], []
    for q in train_q:
        fused, bm, bm_rank, vec_rank, vec_score = pool_for(idx, q, n)
        bm_score = dict(bm)
        bm_max = max(bm_score.values(), default=1.0) or 1.0
        rows = [(c, s) for c, s in fused if c not in banned]
        if len(rows) < 5:
            continue
        for c, s in rows:
            X.append(ranker_features(q, docs[c], s, bm_rank.get(c), vec_rank.get(c), vec_score.get(c), bm_score.get(c), bm_max))
            y.append(min(relevance(truth[q["patient_id"]], truth.get(c, [])), 10))
        groups.append(len(rows))
    return np.array(X), np.array(y), np.array(groups)


def train_lambdarank(X, y, groups, seed=42):
    import lightgbm as lgb
    from sklearn.metrics import ndcg_score

    qid = np.repeat(np.arange(len(groups)), groups)
    folds = np.random.default_rng(seed).integers(0, 3, size=len(groups))[qid]
    grid = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
            for nl, lr, ne, mc in itertools.product([15, 31], [0.03, 0.08], [300, 700], [20, 60])]
    label_gain = [2 ** i - 1 for i in range(11)]
    best, best_score = None, -1
    for params in grid:
        scores = []
        for f in range(3):
            tr, va = folds != f, folds == f
            g_tr = np.bincount(qid[tr])[np.unique(qid[tr])]
            m = lgb.LGBMRanker(objective="lambdarank", label_gain=label_gain, random_state=seed, verbose=-1,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **params)
            m.fit(X[tr], y[tr], group=g_tr)
            pv = m.predict(X[va])
            for g in np.unique(qid[va]):
                sel = qid[va] == g
                if y[va][sel].max() > 0:
                    scores.append(ndcg_score([2 ** y[va][sel] - 1], [pv[sel]], k=10))
        s = float(np.mean(scores))
        log.info("cv %s -> nDCG@10 %.4f", params, s)
        if s > best_score:
            best, best_score = params, s
    m = lgb.LGBMRanker(objective="lambdarank", label_gain=label_gain, random_state=seed, verbose=-1,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **best)
    m.fit(X, y, group=groups)
    return m, best, best_score


def stage_ltr(idx, engine, eval_q, train_q, truth, banned, tag, baseline_row):
    n = config.CANDIDATES_PER_RETRIEVER
    t = time.time()
    X, y, groups = build_ltr_data(idx, train_q, truth, banned, n)
    log.info("ltr data: %d queries, %d pairs, positive %.3f (%.0fs)", len(groups), len(y), (y > 0).mean(), time.time() - t)
    model, params, cv = train_lambdarank(X, y, groups)
    OUT["notes"].append(f"{tag}: LambdaRank trained on {len(groups)} training queries / {len(y)} pairs; "
                        f"CV-chosen params {params} (CV nDCG@10 {cv:.3f}).")
    joblib.dump({"version": 2, "model": model, "params": params, "pool": n}, config.LTR_MODEL_PATH)
    search_mod.get_ranker.cache_clear()
    return record(engine, eval_q, truth, "ltr", f"LambdaRank learned ranker ({tag})", "ltr", baseline_row)


# ----------------------------------------------------------------------------- stage: embed
def embed_triplets(train_docs, idx, truth, banned, rng, per_query=3):
    from sentence_transformers import InputExample
    by_id = {d["patient_id"]: d for d in train_docs}
    cond_index = {}
    for d in train_docs:
        for c in d["chronic_conditions"]:
            cond_index.setdefault(c, []).append(d["patient_id"])
    examples = []
    for q in train_docs:
        qc = truth[q["patient_id"]]
        if not qc:
            continue
        cands = {p for c in qc for p in cond_index.get(c, []) if p != q["patient_id"]}
        if not cands:
            continue
        scored = sorted(cands, key=lambda p: -relevance(qc, truth[p]))
        top = [p for p in scored if relevance(qc, truth[p]) >= relevance(qc, truth[scored[0]])] or scored[:5]
        fused, *_ = pool_for(idx, q, 30)
        negs = [c for c, _ in fused if c not in banned and c in by_id and relevance(qc, truth[c]) == 0]
        for pos in rng.sample(top, min(per_query, len(top))):
            neg = rng.choice(negs) if negs else rng.choice(list(by_id))
            examples.append(InputExample(texts=[q["text"], by_id[pos]["text"], by_id[neg]["text"]]))
    return examples


def stage_embed(p2, eval_q, train_q, train_docs, truth, banned, before_rows, reuse=False):
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    out = MODELS / "embed-ft"
    if reuse and (out / "model.safetensors").exists():
        log.info("reusing the already fine-tuned embedder at %s", out)
        OUT["notes"].append("Embedder: all-MiniLM-L6-v2 fine-tuned 2 epochs (7,817 s on CPU) on 3,385 (anchor, positive, "
                            "hard negative) triplets from training patients only; positives share the most conditions.")
        return _eval_embedder(out, p2, eval_q, train_q, truth, banned, before_rows)
    rng = random.Random(7)
    base_idx = PatientIndex(collection="eval_patients")
    t = time.time()
    examples = embed_triplets(train_docs, base_idx, truth, banned, rng)
    log.info("embed triplets: %d (%.0fs)", len(examples), time.time() - t)
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    model.max_seq_length = 256
    loader = DataLoader(examples, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)
    t = time.time()
    ckpt = MODELS / "embed-ft-checkpoints"
    log.info("training: %d triplets, batch 32, 2 epochs = %d steps; progress bar below, checkpoint every 25 steps in %s",
             len(examples), 2 * len(loader), ckpt)
    # progress is printed per step; checkpoints let a stopped run be recovered (keep the latest 3)
    model.fit(train_objectives=[(loader, loss)], epochs=2, warmup_steps=20, show_progress_bar=True,
              checkpoint_path=str(ckpt), checkpoint_save_steps=25, checkpoint_save_total_limit=3)
    out = MODELS / "embed-ft"
    model.save(str(out))
    log.info("embedder fine-tuned in %.0fs -> %s", time.time() - t, out)
    OUT["notes"].append(f"Embedder: all-MiniLM-L6-v2 fine-tuned 2 epochs on {len(examples)} (anchor, positive, hard negative) "
                        "triplets from training patients only; positives share the most conditions with the anchor.")

    return _eval_embedder(out, p2, eval_q, train_q, truth, banned, before_rows)


def _eval_embedder(out, p2, eval_q, train_q, truth, banned, before_rows):
    config.EMBED_MODEL = str(out)
    index_mod.get_embedder.cache_clear()
    idx = PatientIndex(collection="eval_ft")
    idx.reset()
    idx.upsert(p2)
    engine = SearchEngine(idx)
    v = record(engine, eval_q, truth, "vector", "Vector search, fine-tuned embedder", "embed", before_rows["vector"])
    h = record(engine, eval_q, truth, "hybrid", "Hybrid RRF, fine-tuned embedder", "embed", before_rows["hybrid"])
    l = stage_ltr(idx, engine, eval_q, train_q, truth, banned, "fine-tuned embedder", before_rows["ltr"])
    return idx, engine, {"vector": v, "hybrid": h, "ltr": l}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="ltr,embed")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-embedder", action="store_true", help="evaluate artifacts/models/embed-ft without retraining")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stages = a.stages.split(",")
    MODELS.mkdir(parents=True, exist_ok=True)

    p2 = load_jsonl(config.DOCS_PERIOD2)
    truth = {d["patient_id"]: d["chronic_conditions"] for d in p2}
    eval_q = pick_queries(p2, 200, a.seed)
    banned = {q["patient_id"] for q in eval_q}
    train_docs = [d for d in p2 if d["patient_id"] not in banned]
    train_q = pick_queries(train_docs, a.n_train, a.seed + 1)
    OUT["notes"].append(f"{len(eval_q)} evaluation queries; {len(train_q)} training queries; the evaluation patients "
                        "are excluded from all training data (queries, candidates, positives, negatives).")

    idx = PatientIndex(collection="eval_patients")
    engine = SearchEngine(idx)
    before = {
        "vector": record(engine, eval_q, truth, "vector", "Vector search (current embedder)", "before"),
        "hybrid": record(engine, eval_q, truth, "hybrid", "Hybrid RRF, flag BM25 (current)", "before"),
        "ltr": record(engine, eval_q, truth, "ltr", "Learned ranker (current default)", "before"),
    }
    best_ltr = before["ltr"]
    if "ltr" in stages:
        best_ltr = stage_ltr(idx, engine, eval_q, train_q, truth, banned, "current embedder", before["ltr"])
    if "embed" in stages:
        lambdarank_path = config.LTR_MODEL_PATH.with_name("learned_ranker_previous.joblib")
        if config.LTR_MODEL_PATH.exists():
            shutil.copy(config.LTR_MODEL_PATH, lambdarank_path)
        ft_rows = {"vector": before["vector"], "hybrid": before["hybrid"], "ltr": best_ltr}
        idx, engine, after = stage_embed(p2, eval_q, train_q, train_docs, truth, banned, ft_rows, reuse=a.reuse_embedder)
        if after["ltr"]["ndcg@10"] < best_ltr["ndcg@10"]:
            OUT["notes"].append("Fine-tuned embedder did not beat the base embedder end to end; keeping the base one.")
            shutil.copy(lambdarank_path, config.LTR_MODEL_PATH)
            search_mod.get_ranker.cache_clear()
            config.EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            index_mod.get_embedder.cache_clear()
            idx = PatientIndex(collection="eval_patients")
            engine = SearchEngine(idx)
        else:
            best_ltr = after["ltr"]
    save()
    print((RESULTS / "finetune.md").read_text(encoding="utf-8").encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
