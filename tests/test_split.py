"""Tests for the Phase 1 train/validation split."""
from __future__ import annotations

import json

import pytest

from src.data.schema import InstructionRecord
from src.data.split import SplitConfig, load_split_config, split_records, write_split

RECORDS = [
    InstructionRecord(instruction=f"Q{i}", response=f"A{i}") for i in range(20)
]


def test_split_sizes_match_configured_fraction():
    result = split_records(RECORDS, SplitConfig(val_fraction=0.25, seed=1))
    assert len(result.val) == 5
    assert len(result.train) == 15


def test_split_is_disjoint_and_covers_every_record():
    result = split_records(RECORDS, SplitConfig(val_fraction=0.2, seed=7))
    train_hashes = {r.content_hash() for r in result.train}
    val_hashes = {r.content_hash() for r in result.val}
    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes | val_hashes == {r.content_hash() for r in RECORDS}
    assert len(result.train) + len(result.val) == len(RECORDS)


def test_split_is_deterministic_given_the_same_seed():
    a = split_records(RECORDS, SplitConfig(val_fraction=0.3, seed=99))
    b = split_records(RECORDS, SplitConfig(val_fraction=0.3, seed=99))
    assert [r.content_hash() for r in a.val] == [r.content_hash() for r in b.val]


def test_different_seeds_can_produce_different_splits():
    a = split_records(RECORDS, SplitConfig(val_fraction=0.3, seed=1))
    b = split_records(RECORDS, SplitConfig(val_fraction=0.3, seed=2))
    assert {r.content_hash() for r in a.val} != {r.content_hash() for r in b.val}


def test_tiny_dataset_still_holds_out_at_least_one_validation_record():
    tiny = [InstructionRecord(instruction="only one relevant", response="x") for _ in range(3)]
    result = split_records(tiny, SplitConfig(val_fraction=0.01, seed=0))
    assert len(result.val) >= 1
    assert len(result.train) >= 1


def test_single_record_dataset_goes_entirely_to_train():
    result = split_records([RECORDS[0]], SplitConfig())
    assert len(result.train) == 1
    assert result.val == []


def test_empty_dataset_produces_empty_split():
    result = split_records([], SplitConfig())
    assert result.train == []
    assert result.val == []


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_is_rejected(val_fraction):
    with pytest.raises(ValueError):
        SplitConfig(val_fraction=val_fraction)


def test_load_split_config_missing_explicit_path_is_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split_config(tmp_path / "nope.yaml")


def test_load_split_config_none_returns_defaults():
    cfg = load_split_config(None)
    assert cfg.val_fraction == 0.1
    assert cfg.seed == 42


def test_load_shipped_split_config():
    from pathlib import Path

    cfg = load_split_config(Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml")
    assert cfg.val_fraction == 0.1
    assert cfg.seed == 42


def test_write_split_writes_two_jsonl_files(tmp_path):
    result = split_records(RECORDS, SplitConfig(val_fraction=0.2, seed=3))
    paths = write_split(result, tmp_path)

    train_lines = paths["train"].read_text(encoding="utf-8").strip().splitlines()
    val_lines = paths["val"].read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == len(result.train)
    assert len(val_lines) == len(result.val)
    assert json.loads(train_lines[0])["instruction"].startswith("Q")


def test_write_split_is_atomic_on_failure(tmp_path, monkeypatch):
    import src.data.pipeline as pipe

    monkeypatch.setattr(pipe.os, "replace", lambda s, d: (_ for _ in ()).throw(RuntimeError("boom")))
    result = split_records(RECORDS, SplitConfig())
    with pytest.raises(RuntimeError):
        write_split(result, tmp_path)

    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "train.jsonl.tmp").exists()
