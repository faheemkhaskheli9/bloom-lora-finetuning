"""Tests for LoRA settings loading (no torch/peft import needed here)."""
from __future__ import annotations

import pytest

from src.lora.config import DEFAULT_CONFIG_PATH, LoraSettings, load_lora_settings


def test_default_settings_are_valid():
    settings = LoraSettings()
    assert settings.r > 0
    assert settings.target_modules


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank"):
        LoraSettings(r=0)


def test_alpha_must_be_positive():
    with pytest.raises(ValueError, match="lora_alpha"):
        LoraSettings(lora_alpha=0)


def test_dropout_must_be_in_range():
    with pytest.raises(ValueError, match="lora_dropout"):
        LoraSettings(lora_dropout=1.0)


def test_target_modules_must_not_be_empty():
    with pytest.raises(ValueError, match="target_modules"):
        LoraSettings(target_modules=())


def test_to_peft_kwargs_shape():
    settings = LoraSettings(r=4, lora_alpha=8, target_modules=("query_key_value",))
    kwargs = settings.to_peft_kwargs()
    assert kwargs["r"] == 4
    assert kwargs["lora_alpha"] == 8
    assert kwargs["target_modules"] == ["query_key_value"]


def test_load_lora_settings_reads_the_shipped_config():
    settings = load_lora_settings()
    assert settings.base_model_name == "bigscience/bloomz-560m"
    assert settings.target_modules == ("query_key_value",)
    assert settings.r == 8


def test_load_lora_settings_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_lora_settings("configs/does-not-exist.yaml")


def test_shipped_config_file_actually_exists():
    assert DEFAULT_CONFIG_PATH.is_file()


def test_load_lora_settings_custom_path(tmp_path):
    custom = tmp_path / "custom_lora.yaml"
    custom.write_text(
        "base_model_name: some/other-model\n"
        "lora:\n"
        "  r: 2\n"
        "  lora_alpha: 4\n"
        "  target_modules: [dense]\n",
        encoding="utf-8",
    )
    settings = load_lora_settings(custom)
    assert settings.base_model_name == "some/other-model"
    assert settings.r == 2
    assert settings.target_modules == ("dense",)
