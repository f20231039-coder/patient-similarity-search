"""Assemble the Hugging Face Space folder (build/space/) from this repo.

The Space gets the code, Dockerfile and the trained artifacts (documents, risk
model, learned ranker, fine-tuned embedder: about 100 MB, which stay out of the
GitHub repo). Secrets are NOT copied: set LLM_API_KEY as a Space secret.

  python scripts/prepare_space.py
  hf upload <user>/<space-name> build/space --repo-type=space
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"
ARTIFACTS = ["documents_period1.jsonl", "documents_period2.jsonl", "risk_model.joblib", "learned_ranker.joblib"]

SPACE_README = """---
title: Patient Similarity Search
emoji: 🩺
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Find look-alike patients, risk and reasons (synthetic EHR)
---

# Patient Similarity Search

Find patients who look like a given patient, see what conditions those look-alikes have, estimate
chronic-disease risk with reasons, and compare any two patients side by side.

Hybrid retrieval (BM25 + fine-tuned MiniLM in Qdrant), reciprocal rank fusion and a LightGBM LambdaRank
re-ranker: nDCG@10 0.458 on 200 held-out patients (2.4x keyword search). An LLM (Gemini Flash-Lite) is
used only to turn a plain-language search into filters, with keyword rules as fallback.

All data is synthetic (Synthea). No real patients. Not medical advice.
"""


def main() -> None:
    missing = [a for a in ARTIFACTS if not (ROOT / "artifacts" / a).exists()]
    if missing or not (ROOT / "artifacts/models/embed-ft/model.safetensors").exists():
        sys.exit(f"missing artifacts: {missing or ['models/embed-ft']}; build them first (see README)")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "artifacts").mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "simsearch", OUT / "simsearch", ignore=ignore)
    (OUT / "scripts").mkdir()
    for s in ("build_index.py", "check_llm.py"):
        shutil.copy(ROOT / "scripts" / s, OUT / "scripts" / s)
    for f in ("Dockerfile", "requirements.txt", ".dockerignore"):
        shutil.copy(ROOT / f, OUT / f)
    for a in ARTIFACTS:
        shutil.copy(ROOT / "artifacts" / a, OUT / "artifacts" / a)
    shutil.copytree(ROOT / "artifacts/models/embed-ft", OUT / "artifacts/models/embed-ft", ignore=ignore)
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    (OUT / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n"
                                        "tokenizer.json filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    assert not list(OUT.rglob(".env")), "a .env file must never be uploaded"
    files = [p for p in OUT.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"{OUT}: {len(files)} files, {size / 1e6:.1f} MB")
    for p in sorted(files, key=lambda p: -p.stat().st_size)[:6]:
        print(f"  {p.stat().st_size / 1e6:7.1f} MB  {p.relative_to(OUT)}")


if __name__ == "__main__":
    main()
