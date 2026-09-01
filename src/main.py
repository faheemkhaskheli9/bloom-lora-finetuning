"""CLI entrypoint for the BLOOM LoRA fine-tuning dataset pipeline.

Phase 1 command: prepare a cleaned instruction dataset from a local JSONL file
or a public Hugging Face dataset.

    python -m src.main prepare --source jsonl \\
        --input examples/sample_raw.jsonl \\
        --output data/processed/clean.jsonl \\
        --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data.pipeline import load_config, prepare_dataset
from src.data.sources import DatasetSourceError, iter_hf_dataset, iter_jsonl

DEFAULT_CONFIG = Path("configs/dataset.yaml")
DEFAULT_OUTPUT = Path("data/processed/clean.jsonl")


def _cmd_prepare(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.source == "jsonl":
        if not args.input:
            print("error: --input is required for --source jsonl", file=sys.stderr)
            return 1
        rows = iter_jsonl(args.input)
    else:
        rows = iter_hf_dataset(
            args.hf_name or cfg.hf_name, split=cfg.hf_split, config=cfg.hf_config, limit=args.limit
        )

    try:
        result = prepare_dataset(rows, args.output, cfg)
    except DatasetSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.summary())
    print(f"wrote {result.kept} records -> {args.output}")
    return 0 if result.kept > 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloom-lora", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Clean a raw dataset into JSONL for fine-tuning.")
    p.add_argument("--source", choices=["jsonl", "hf"], default="jsonl")
    p.add_argument("--input", type=Path, help="Raw JSONL path (for --source jsonl).")
    p.add_argument("--hf-name", dest="hf_name", help="Hugging Face dataset id (for --source hf).")
    p.add_argument("--limit", type=int, default=None, help="Cap examples pulled from HF.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.set_defaults(func=_cmd_prepare)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    # A default config path that does not exist -> fall back to defaults (rule 7).
    if args.config == DEFAULT_CONFIG and not args.config.is_file():
        args.config = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
