"""Tests for the BLOOM tokenization pipeline.

Uses the real ``bigscience/bloomz-560m`` tokenizer (downloaded once via
``transformers`` and cached by Hugging Face locally) rather than a mock,
since the acceptance criterion is specifically about using BLOOM's tokenizer.
"""
from __future__ import annotations

import pytest

from src.data.schema import InstructionRecord
from src.data.tokenization import (
    format_prompt,
    load_bloom_tokenizer,
    tokenize_record,
    tokenize_text,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_bloom_tokenizer()


def test_tokenize_pads_to_exact_max_length(tokenizer):
    example = tokenize_text("hello world", tokenizer=tokenizer, max_length=16)
    assert len(example.input_ids) == 16
    assert len(example.attention_mask) == 16


def test_tokenize_truncates_long_text_to_max_length(tokenizer):
    long_text = "word " * 500
    example = tokenize_text(long_text, tokenizer=tokenizer, max_length=32)
    assert len(example.input_ids) == 32


def test_round_trip_recovers_the_original_text(tokenizer):
    original = "The quick brown fox jumps over the lazy dog."
    example = tokenize_text(original, tokenizer=tokenizer, max_length=32)
    # Drop padding before decoding so the comparison is against real content.
    real_tokens = [
        tid for tid, mask in zip(example.input_ids, example.attention_mask) if mask == 1
    ]
    decoded = tokenizer.decode(real_tokens, skip_special_tokens=True)
    assert decoded.strip() == original


def test_invalid_max_length_is_rejected(tokenizer):
    with pytest.raises(ValueError):
        tokenize_text("hi", tokenizer=tokenizer, max_length=0)


def test_tokenizer_is_cached_across_calls():
    first = load_bloom_tokenizer()
    second = load_bloom_tokenizer()
    assert first is second


def test_format_prompt_includes_instruction_and_response():
    record = InstructionRecord(instruction="Summarize this.", response="A summary.")
    prompt = format_prompt(record)
    assert "Summarize this." in prompt
    assert "A summary." in prompt


def test_format_prompt_includes_context_when_present():
    record = InstructionRecord(instruction="Summarize.", context="Some passage.", response="OK.")
    prompt = format_prompt(record)
    assert "Some passage." in prompt


def test_tokenize_record_end_to_end(tokenizer):
    record = InstructionRecord(instruction="Say hi.", response="Hi!")
    example = tokenize_record(record, tokenizer=tokenizer, max_length=64)
    assert len(example.input_ids) == 64
