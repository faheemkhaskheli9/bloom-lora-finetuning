"""Tests for LoRA adapter attachment.

Uses a tiny, randomly-initialized ``transformers.BloomConfig`` model instead
of downloading the real ``bigscience/bloomz-560m`` pretrained weights -- see
src/lora/model.py's module docstring for why (no GPU/network-heavy download
in the test suite; the mock boundary this issue's Definition of Done calls
for). ``torch``/``transformers``/``peft`` come from the `train` extra
(``pip install -e .[train]``) and are skipped here if not installed, since
the base ``requirements.txt``/CI intentionally stay lean per pyproject.toml.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from src.lora.config import LoraSettings
from src.lora.model import attach_lora, count_trainable_parameters, log_trainable_fraction


@pytest.fixture
def tiny_bloom_model():
    config = transformers.BloomConfig(
        vocab_size=128,
        hidden_size=32,
        n_layer=2,
        n_head=2,
        intermediate_size=64,
    )
    return transformers.BloomForCausalLM(config)


def test_lora_attaches_cleanly_to_a_bloom_model(tiny_bloom_model):
    settings = LoraSettings(target_modules=("query_key_value",))
    lora_model = attach_lora(tiny_bloom_model, settings)
    assert lora_model is not None


def test_trainable_parameters_are_a_small_fraction_of_total(tiny_bloom_model):
    settings = LoraSettings(r=4, target_modules=("query_key_value",))
    lora_model = attach_lora(tiny_bloom_model, settings)

    trainable, total = count_trainable_parameters(lora_model)
    assert 0 < trainable < total
    assert trainable / total < 0.5  # LoRA params are a small slice of the whole model


def test_larger_rank_means_more_trainable_parameters(tiny_bloom_model):
    small = attach_lora(tiny_bloom_model, LoraSettings(r=2, target_modules=("query_key_value",)))
    small_trainable, _ = count_trainable_parameters(small)

    config = transformers.BloomConfig(
        vocab_size=128, hidden_size=32, n_layer=2, n_head=2, intermediate_size=64
    )
    fresh_model = transformers.BloomForCausalLM(config)
    large = attach_lora(fresh_model, LoraSettings(r=16, target_modules=("query_key_value",)))
    large_trainable, _ = count_trainable_parameters(large)

    assert large_trainable > small_trainable


def test_unknown_target_module_raises_not_silently_ignored(tiny_bloom_model):
    settings = LoraSettings(target_modules=("this_module_does_not_exist",))
    with pytest.raises(ValueError):
        attach_lora(tiny_bloom_model, settings)


def test_log_trainable_fraction_returns_fraction_in_unit_interval(tiny_bloom_model, caplog):
    import logging

    settings = LoraSettings(target_modules=("query_key_value",))
    lora_model = attach_lora(tiny_bloom_model, settings)

    with caplog.at_level(logging.INFO):
        fraction = log_trainable_fraction(lora_model)
    assert 0.0 < fraction < 1.0
    assert "trainable parameters" in caplog.text.lower()
