"""Record types for the instruction fine-tuning dataset pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class InstructionRecord:
    """A single instruction-tuning example (Alpaca-style fields).

    ``context`` is optional extra input the model conditions on (e.g. a passage
    to summarize); ``response`` is the target completion.
    """

    instruction: str
    response: str
    context: str = ""

    def content_hash(self) -> str:
        """Stable hash over the normalized text, used for exact deduplication."""
        payload = "\x1f".join((self.instruction, self.context, self.response))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "context": self.context,
            "response": self.response,
        }
