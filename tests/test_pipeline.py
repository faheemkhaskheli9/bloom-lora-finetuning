"""Tests for the prepare_dataset pipeline (source -> clean -> atomic write)."""
from __future__ import annotations

import json

import pytest

from src.data.pipeline import PrepareConfig, load_config, prepare_dataset
from src.data.cleaning import CleaningConfig

ROWS = [
    {"instruction": "Name a planet.", "input": "", "output": "Mars."},
    {"instruction": "Name a planet.", "input": "", "output": "Mars."},
    {"instruction": "", "input": "", "output": "drop me"},
]


def test_prepare_writes_jsonl_and_stats(tmp_path):
    out = tmp_path / "processed" / "clean.jsonl"
    result = prepare_dataset(iter(ROWS), out, PrepareConfig(cleaning=CleaningConfig()))

    assert result.kept == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["response"] == "Mars."

    stats = json.loads((tmp_path / "processed" / "clean.stats.json").read_text())
    assert stats["kept"] == 1
    assert stats["dropped"]["duplicate"] == 1


def test_prepare_is_atomic_on_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "clean.jsonl"
    import src.data.pipeline as pipe

    monkeypatch.setattr(pipe.os, "replace", lambda s, d: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        prepare_dataset(iter(ROWS), out, PrepareConfig(cleaning=CleaningConfig()))

    assert not out.exists()
    assert not out.with_suffix(".jsonl.tmp").exists()


def test_load_config_missing_explicit_path_is_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_none_returns_defaults():
    cfg = load_config(None)
    assert cfg.hf_name == "tatsu-lab/alpaca"
    assert cfg.cleaning.drop_exact_duplicates is True


def test_load_shipped_config(tmp_path):
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml")
    assert cfg.cleaning.field_map["response"] == "output"
    assert cfg.hf_split == "train"
