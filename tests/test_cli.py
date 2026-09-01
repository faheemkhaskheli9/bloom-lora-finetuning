"""Tests for the Phase 1 CLI (`python -m src.main prepare`)."""
from __future__ import annotations

from pathlib import Path

from src.main import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_from_shipped_example(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "clean.jsonl"
    rc = main(
        [
            "prepare",
            "--source",
            "jsonl",
            "--input",
            "examples/sample_raw.jsonl",
            "--output",
            str(out),
            "--config",
            "configs/dataset.yaml",
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 0
    assert out.is_file()
    kept = len(out.read_text(encoding="utf-8").strip().splitlines())
    # 7 clean unique rows survive; duplicate / empty / echo / non-JSON rows drop.
    assert kept == 7
    assert "kept" in captured


def test_prepare_jsonl_requires_input(capsys):
    rc = main(["prepare", "--source", "jsonl", "--output", "x.jsonl"])
    assert rc == 1
    assert "--input is required" in capsys.readouterr().err
