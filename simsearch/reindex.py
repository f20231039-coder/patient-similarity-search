"""Index jobs: initial period-1 build and incremental period-2 re-index."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import config
from .documents import load_jsonl
from .index import PatientIndex

log = logging.getLogger(__name__)


def build_period1(index: PatientIndex, reset: bool = True) -> dict:
    """Load the period-1 snapshot into a fresh collection."""
    t = time.perf_counter()
    if reset:
        index.reset()
    stats = index.upsert(load_jsonl(config.DOCS_PERIOD1))
    stats.update(total=len(index), seconds=round(time.perf_counter() - t, 2), source="period1")
    log.info("period-1 build: %s", stats)
    return stats


def reindex(index: PatientIndex, path: Path | None = None) -> dict:
    """Upsert a newer snapshot by patient ID: new patients are added, changed
    records replaced (re-embedded), unchanged ones skipped. Never duplicates."""
    t = time.perf_counter()
    before = len(index)
    stats = index.upsert(load_jsonl(path or config.DOCS_PERIOD2))
    stats.update(total_before=before, total_after=len(index), qdrant_points=index.count(),
                 seconds=round(time.perf_counter() - t, 2), source=str(path or config.DOCS_PERIOD2.name))
    log.info("re-index: %s", stats)
    return stats
