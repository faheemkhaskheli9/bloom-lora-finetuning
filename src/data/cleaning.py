"""Cleaning logic for instruction fine-tuning data.

The pipeline turns arbitrary raw dicts into validated :class:`InstructionRecord`
objects, dropping malformed / empty / duplicate / degenerate examples and
returning per-reason drop counts so a caller can log an operational summary
(robustness rule 14: counts go through ``logging``/return values, not
``warnings.warn`` which the interpreter de-duplicates).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from src.data.schema import InstructionRecord

logger = logging.getLogger(__name__)

# Control characters except tab / newline / carriage-return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

DEFAULT_FIELD_MAP: dict[str, str] = {
    "instruction": "instruction",
    "context": "input",
    "response": "output",
}


@dataclass(frozen=True)
class CleaningConfig:
    min_instruction_chars: int = 3
    min_response_chars: int = 1
    max_chars: int = 8000
    max_control_char_ratio: float = 0.02
    drop_exact_duplicates: bool = True
    field_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_MAP))


@dataclass
class CleanResult:
    records: list[InstructionRecord] = field(default_factory=list)
    total: int = 0
    dropped: Counter = field(default_factory=Counter)

    @property
    def kept(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        parts = [f"{self.kept}/{self.total} kept"]
        for reason, n in sorted(self.dropped.items()):
            parts.append(f"{reason}={n}")
        return ", ".join(parts)


def normalize_text(value: object) -> str:
    """NFKC-normalize, strip control chars, and collapse redundant whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _control_char_ratio(raw: str) -> float:
    if not raw:
        return 0.0
    controls = len(_CONTROL_RE.findall(raw))
    return controls / len(raw)


def clean_example(
    raw: Mapping[str, object], cfg: CleaningConfig
) -> tuple[InstructionRecord | None, str | None]:
    """Return ``(record, None)`` if the example is usable, else ``(None, reason)``."""
    if not isinstance(raw, Mapping):
        return None, "not_a_mapping"

    fm = cfg.field_map
    raw_instruction = raw.get(fm["instruction"])
    raw_response = raw.get(fm["response"])
    raw_context = raw.get(fm.get("context", "context"), "")

    if _control_char_ratio(str(raw_instruction or "")) > cfg.max_control_char_ratio:
        return None, "binary_like_instruction"
    if _control_char_ratio(str(raw_response or "")) > cfg.max_control_char_ratio:
        return None, "binary_like_response"

    instruction = normalize_text(raw_instruction)
    response = normalize_text(raw_response)
    context = normalize_text(raw_context)

    if len(instruction) < cfg.min_instruction_chars:
        return None, "empty_instruction"
    if len(response) < cfg.min_response_chars:
        return None, "empty_response"
    if len(instruction) + len(context) + len(response) > cfg.max_chars:
        return None, "too_long"
    if response == instruction:
        return None, "response_echoes_instruction"

    return InstructionRecord(instruction=instruction, response=response, context=context), None


def clean_dataset(rows: Iterable[Mapping[str, object]], cfg: CleaningConfig | None = None) -> CleanResult:
    cfg = cfg or CleaningConfig()
    result = CleanResult()
    seen: set[str] = set()

    for row in rows:
        result.total += 1
        record, reason = clean_example(row, cfg)
        if record is None:
            result.dropped[reason] += 1
            continue
        if cfg.drop_exact_duplicates:
            h = record.content_hash()
            if h in seen:
                result.dropped["duplicate"] += 1
                continue
            seen.add(h)
        result.records.append(record)

    logger.info("cleaning complete: %s", result.summary())
    return result
