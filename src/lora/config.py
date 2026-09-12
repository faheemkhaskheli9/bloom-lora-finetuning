"""LoRA adapter configuration, loaded from ``configs/lora.yaml`` (issue #4).

Kept separate from ``src/lora/model.py`` so the config shape is unit-testable
without importing torch/peft.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "lora.yaml"
DEFAULT_BASE_MODEL_NAME = "bigscience/bloomz-560m"


@dataclass(frozen=True)
class LoraSettings:
    """Mirrors the fields ``peft.LoraConfig`` needs, plus the base model
    name this adapter is meant to attach to."""

    base_model_name: str = DEFAULT_BASE_MODEL_NAME
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = field(default_factory=lambda: ("query_key_value",))

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"r (rank) must be > 0, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")

    def to_peft_kwargs(self) -> dict:
        """Kwargs for ``peft.LoraConfig(**...)`` — kept as a plain dict here
        so this module has no ``peft`` import (see module docstring)."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": list(self.target_modules),
        }


def load_lora_settings(path: str | Path | None = None) -> LoraSettings:
    """Load LoRA settings from a YAML file (default: ``configs/lora.yaml``)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lora_section = raw.get("lora", {})

    kwargs: dict = {}
    if "base_model_name" in raw:
        kwargs["base_model_name"] = raw["base_model_name"]
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "task_type"):
        if key in lora_section:
            kwargs[key] = lora_section[key]
    if "target_modules" in lora_section:
        kwargs["target_modules"] = tuple(lora_section["target_modules"])

    return LoraSettings(**kwargs)
