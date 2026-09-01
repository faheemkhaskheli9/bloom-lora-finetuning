"""Tests for dataset sources."""
from __future__ import annotations

import pytest

from src.data.sources import DatasetSourceError, iter_jsonl


def test_iter_jsonl_skips_blank_and_bad_lines(tmp_path, caplog):
    p = tmp_path / "raw.jsonl"
    p.write_text(
        '{"a": 1}\n\n   \nnot-json\n{"b": 2}\n', encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        rows = list(iter_jsonl(p))
    assert rows == [{"a": 1}, {"b": 2}]
    assert "not valid JSON" in caplog.text


def test_iter_jsonl_strict_raises_on_bad_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text("nope\n", encoding="utf-8")
    with pytest.raises(DatasetSourceError, match="invalid JSON"):
        list(iter_jsonl(p, strict=True))


def test_iter_jsonl_missing_file_is_hard_error(tmp_path):
    with pytest.raises(DatasetSourceError, match="not found"):
        list(iter_jsonl(tmp_path / "missing.jsonl"))


def test_shipped_example_file_parses(tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "sample_raw.jsonl"
    rows = list(iter_jsonl(example))
    assert len(rows) >= 10
