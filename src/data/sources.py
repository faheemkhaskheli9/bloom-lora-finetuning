"""Raw dataset sources: local JSONL (offline / tests) and Hugging Face Hub.

The Hugging Face path imports ``datasets`` lazily so the rest of the pipeline
(and the test suite) has no heavyweight dependency and runs fully offline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Mapping

logger = logging.getLogger(__name__)


class DatasetSourceError(RuntimeError):
    """Raised when a dataset source cannot be read."""


def iter_jsonl(path: str | Path, *, strict: bool = False) -> Iterator[Mapping[str, object]]:
    """Yield one dict per non-blank line of a JSONL file.

    A missing file is always a hard error. A line that is not valid JSON is
    logged at WARNING and skipped (bulk ingestion should not die on one bad
    row); pass ``strict=True`` to raise instead.
    """
    p = Path(path)
    if not p.is_file():
        raise DatasetSourceError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise DatasetSourceError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
                logger.warning("%s:%d: skipping line that is not valid JSON: %s", p, lineno, exc)


def iter_hf_dataset(
    name: str, *, split: str = "train", config: str | None = None, limit: int | None = None
) -> Iterator[Mapping[str, object]]:
    """Yield examples from a public Hugging Face dataset (network required)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise DatasetSourceError(
            "the 'datasets' package is required for Hugging Face sources; "
            "install it with `pip install datasets`"
        ) from exc

    ds = load_dataset(name, config, split=split, streaming=True)
    for i, example in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield example
