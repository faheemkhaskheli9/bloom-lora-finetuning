"""Deterministic train/validation split for the cleaned instruction dataset.

Phase 1 scope: partition an already-cleaned list of ``InstructionRecord``s
into disjoint train/validation subsets, with the split ratio and random seed
configurable via YAML (the ``split:`` section of ``configs/dataset.yaml``).
The split is index-based -- every record lands in exactly one of the two
returned lists -- so there is no leakage between train and validation by
construction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar

import yaml

from src.data.pipeline import atomic_write_jsonl
from src.data.schema import InstructionRecord

T = TypeVar("T")

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitConfig:
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")


@dataclass(frozen=True)
class SplitResult(Generic[T]):
    train: list[T]
    val: list[T]


def load_split_config(path: str | Path | None) -> SplitConfig:
    """Load the ``split:`` section of a dataset YAML config.

    Robustness rule 7: a missing *default* path yields the built-in default
    config; a path the caller explicitly passed that does not exist is a
    hard error, never a silent fall-through to defaults.
    """
    if path is None:
        return SplitConfig()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    split_raw = dict(raw.get("split", {}))
    return SplitConfig(**split_raw)


def split_records(records: Sequence[T], config: SplitConfig | None = None) -> SplitResult[T]:
    """Partition ``records`` into disjoint train/validation subsets.

    The split shuffles indices (seeded by ``config.seed``) and slices the
    shuffled order -- every index ends up in exactly one of the two returned
    lists, so train and validation never share a record. With more than one
    record and a nonzero ``val_fraction``, at least one record is always held
    out for validation (rounding a tiny fraction down to zero would defeat
    the point of a split).
    """
    config = config or SplitConfig()
    n = len(records)
    if n <= 1:
        return SplitResult(train=list(records), val=[])

    n_val = round(n * config.val_fraction)
    n_val = min(max(n_val, 1), n - 1)

    indices = list(range(n))
    random.Random(config.seed).shuffle(indices)
    val_indices = set(indices[:n_val])

    train = [records[i] for i in range(n) if i not in val_indices]
    val = [records[i] for i in indices[:n_val]]
    return SplitResult(train=train, val=val)


def write_split(result: SplitResult[InstructionRecord], output_dir: str | Path) -> dict[str, Path]:
    """Atomically write ``result.train``/``result.val`` under ``output_dir``.

    Returns the paths written, keyed ``"train"``/``"val"``.
    """
    out_dir = Path(output_dir)
    paths = {"train": out_dir / "train.jsonl", "val": out_dir / "val.jsonl"}
    atomic_write_jsonl(result.train, paths["train"])
    atomic_write_jsonl(result.val, paths["val"])
    return paths
