# Patient Similarity Search: one container (FastAPI app + embedded Qdrant index).
# Works on Hugging Face Spaces (port 7860, non-root user 1000) and anywhere Docker runs.
#
# Needs the trained artifacts in ./artifacts (documents, risk model, learned ranker,
# fine-tuned embedder): run scripts/build_documents.py, scripts/train_risk_model.py
# and evals/finetune.py first, or use the folder made by scripts/prepare_space.py.
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user simsearch ./simsearch
COPY --chown=user scripts ./scripts
COPY --chown=user artifacts ./artifacts

# Build the search index into the image (period 1 + period-2 re-index), so the app
# starts in seconds instead of re-embedding every patient on each restart.
RUN python scripts/build_index.py

EXPOSE 7860
CMD ["sh", "-c", "uvicorn simsearch.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
