"""Tests for the `python -m src.main tokenize` CLI subcommand."""
from __future__ import annotations

from src.main import main


def test_tokenize_cli_runs_without_a_config_option(capsys):
    # Regression test: the "tokenize" subcommand has no --config option, so
    # main()'s default-config fallback must not assume every subcommand does.
    rc = main(["tokenize", "--text", "hello", "--max-length", "8"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "input_ids (8)" in out
    assert "attention_mask (8)" in out


def test_tokenize_cli_requires_text_or_text_file(capsys):
    rc = main(["tokenize", "--max-length", "8"])
    assert rc == 1
    assert "--text" in capsys.readouterr().err


def test_tokenize_cli_reads_from_text_file(tmp_path, capsys):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("some sample text")
    rc = main(["tokenize", "--text-file", str(text_file), "--max-length", "8"])
    assert rc == 0
    assert "input_ids" in capsys.readouterr().out
