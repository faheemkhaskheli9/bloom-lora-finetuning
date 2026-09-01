"""Tests for instruction-dataset cleaning logic."""
from __future__ import annotations

from src.data.cleaning import CleaningConfig, clean_dataset, clean_example, normalize_text


def test_normalize_collapses_whitespace_and_strips_controls():
    assert normalize_text("a\x00b   c\r\nd  ") == "ab c\nd"
    assert normalize_text("café") == "café"
    assert normalize_text(None) == ""


def test_clean_example_accepts_well_formed_row():
    rec, reason = clean_example(
        {"instruction": " List primary colors ", "input": "", "output": "Red, blue, yellow."},
        CleaningConfig(),
    )
    assert reason is None
    assert rec.instruction == "List primary colors"
    assert rec.response == "Red, blue, yellow."


def test_clean_dataset_drops_malformed_and_dedupes():
    rows = [
        {"instruction": "Name a planet.", "input": "", "output": "Mars."},
        {"instruction": "Name a planet.", "input": "", "output": "Mars."},  # duplicate
        {"instruction": "", "input": "", "output": "no instruction"},       # empty_instruction
        {"instruction": "no response", "input": "", "output": "   "},       # empty_response
        {"instruction": "Echo.", "input": "", "output": "Echo."},           # response echoes
        ["not", "a", "mapping"],                                             # not_a_mapping
        {"instruction": "x" * 5, "input": "", "output": "y" * 9000},        # too_long
    ]
    result = clean_dataset(rows, CleaningConfig(min_instruction_chars=3))
    assert result.total == 7
    assert result.kept == 1
    assert result.dropped["duplicate"] == 1
    assert result.dropped["empty_instruction"] == 1
    assert result.dropped["empty_response"] == 1
    assert result.dropped["response_echoes_instruction"] == 1
    assert result.dropped["not_a_mapping"] == 1
    assert result.dropped["too_long"] == 1
    assert "kept" in result.summary()


def test_binary_like_rows_are_rejected():
    junk = "".join(chr(b) for b in range(0, 20)) * 5
    rec, reason = clean_example({"instruction": junk, "input": "", "output": "ok"}, CleaningConfig())
    assert rec is None
    assert reason == "binary_like_instruction"


def test_drop_counts_are_per_record_not_deduplicated():
    """Regression for robustness rule 14: 50 identical bad rows -> count of 50."""
    rows = [{"instruction": "", "input": "", "output": "x"} for _ in range(50)]
    result = clean_dataset(rows, CleaningConfig())
    assert result.dropped["empty_instruction"] == 50


def test_custom_field_map():
    cfg = CleaningConfig(field_map={"instruction": "q", "context": "ctx", "response": "a"})
    rec, reason = clean_example({"q": "What is 2+2?", "ctx": "", "a": "4"}, cfg)
    assert reason is None
    assert rec.instruction == "What is 2+2?"
    assert rec.response == "4"
