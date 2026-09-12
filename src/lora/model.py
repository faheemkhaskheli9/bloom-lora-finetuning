"""Attach a configurable LoRA adapter to a BLOOM base model (issue #4).

No GPU is available in this environment (or CI), and downloading the full
``bigscience/bloomz-560m`` pretrained weights (~1GB) on every test run would
make the test suite slow and network-flaky for no benefit -- the LoRA
attachment mechanics don't depend on the base model's weights actually being
pretrained, only on its architecture/module names matching. So tests build a
tiny, randomly-initialized model from ``transformers.BloomConfig`` (same
architecture class, no download) instead of ``from_pretrained``; the real
training path (issue #5) calls ``load_base_model`` and does download the
real checkpoint, which is a free public download with no paid API involved.
That is the mock boundary this module's Definition of Done calls for.
"""
from __future__ import annotations

import logging

from src.lora.config import LoraSettings, load_lora_settings

logger = logging.getLogger(__name__)


def load_base_model(model_name: str):
    """Load the real pretrained BLOOM base model by name (network download,
    CPU-only). Not used by the test suite -- see the module docstring."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name)


def attach_lora(model, settings: LoraSettings):
    """Wrap ``model`` with a LoRA adapter configured from ``settings``.

    Raises whatever ``peft`` raises (e.g. target module not found) rather
    than swallowing it -- an adapter that silently fails to attach to any
    module is worse than a loud error.
    """
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(**settings.to_peft_kwargs())
    return get_peft_model(model, peft_config)


def count_trainable_parameters(model) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_trainable_fraction(model) -> float:
    """Log the trainable-parameter fraction and return it (issue #4
    acceptance criterion: trainable parameter count logged)."""
    trainable, total = count_trainable_parameters(model)
    fraction = trainable / total if total else 0.0
    logger.info(
        "LoRA trainable parameters: %s / %s (%.4f%% of total)",
        trainable,
        total,
        fraction * 100,
    )
    return fraction


def build_lora_model(config_path: str | None = None, model=None):
    """End-to-end: load settings, attach LoRA to ``model`` (or the real
    base model named in the settings if ``model`` is omitted), and log the
    trainable-parameter fraction."""
    settings = load_lora_settings(config_path)
    base_model = model if model is not None else load_base_model(settings.base_model_name)
    lora_model = attach_lora(base_model, settings)
    log_trainable_fraction(lora_model)
    return lora_model


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    build_lora_model()
