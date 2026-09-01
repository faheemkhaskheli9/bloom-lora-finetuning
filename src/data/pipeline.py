"""End-to-end dataset preparation: source -> clean -> atomic JSONL + stats."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from src.data.cleaning import CleaningConfig, CleanResult, clean_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareConfig:
    cleaning: CleaningConfig
    hf_name: str = "tatsu-lab/alpaca"
    hf_split: str = "train"
    hf_config: str | None = None


def load_config(path: str | Path | None) -> PrepareConfig:
    """Load a YAML pipeline config.

    Robustness rule 7: a *missing default* path returns built-in defaults; a
    path the caller explicitly passed that does not exist is a hard error.
    """
    if path is None:
        return PrepareConfig(cleaning=CleaningConfig())
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    clean_raw = dict(raw.get("cleaning", {}))
    if "field_map" in clean_raw and clean_raw["field_map"] is None:
        clean_raw.pop("field_map")
    cleaning = CleaningConfig(**clean_raw)
    hf = raw.get("huggingface", {}) or {}
    return PrepareConfig(
        cleaning=cleaning,
        hf_name=hf.get("name", "tatsu-lab/alpaca"),
        hf_split=hf.get("split", "train"),
        hf_config=hf.get("config"),
    )


def _atomic_write_jsonl(records, dest: Path) -> None:
    """Write records to ``dest`` via a temp file + ``os.replace`` (rule 1).

    A crash mid-write leaves the temp file, never a truncated ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def prepare_dataset(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
    cfg: PrepareConfig | None = None,
) -> CleanResult:
    cfg = cfg or PrepareConfig(cleaning=CleaningConfig())
    result = clean_dataset(rows, cfg.cleaning)

    dest = Path(output_path)
    _atomic_write_jsonl(result.records, dest)

    stats = {
        "total": result.total,
        "kept": result.kept,
        "dropped": dict(result.dropped),
        "output": str(dest),
    }
    stats_path = dest.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote %d cleaned records to %s", result.kept, dest)
    return result
