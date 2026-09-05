"""Tokenization pipeline matched to the BLOOM tokenizer.

Uses the real Hugging Face BLOOM tokenizer (``bigscience/bloomz-560m`` by
default -- BLOOM's own smallest checkpoint, so only tokenizer files are
downloaded, not the larger base-BLOOM weights) rather than a mock, since it is
a free public download with no paid API or special hardware involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.data.schema import InstructionRecord

DEFAULT_TOKENIZER_NAME = "bigscience/bloomz-560m"


@lru_cache(maxsize=4)
def load_bloom_tokenizer(model_name: str = DEFAULT_TOKENIZER_NAME):
    """Load (and cache) the BLOOM tokenizer for ``model_name``.

    Cached only on ``model_name`` -- its only input -- so different callers
    asking for the same tokenizer share one load (rule 11).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # BLOOM's tokenizer has no pad token by default; padding needs one.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(record: InstructionRecord) -> str:
    """Render one instruction-tuning record as a single training string."""
    if record.context:
        return (
            f"### Instruction:\n{record.instruction}\n\n"
            f"### Context:\n{record.context}\n\n"
            f"### Response:\n{record.response}"
        )
    return f"### Instruction:\n{record.instruction}\n\n### Response:\n{record.response}"


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask must be the same length")


def tokenize_text(text: str, *, tokenizer: Any, max_length: int) -> TokenizedExample:
    """Tokenize ``text``, truncating/padding to exactly ``max_length`` tokens."""
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return TokenizedExample(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )


def tokenize_record(
    record: InstructionRecord, *, tokenizer: Any, max_length: int
) -> TokenizedExample:
    return tokenize_text(format_prompt(record), tokenizer=tokenizer, max_length=max_length)
