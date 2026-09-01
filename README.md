# BLOOM LoRA Fine-Tuning

> NLP portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Full fine-tuning of large language models is expensive; parameter-efficient fine-tuning (LoRA) offers a practical alternative worth demonstrating end-to-end.

## 2. Architecture

```text
Dataset -> Tokenization -> BLOOM Base Model + LoRA Adapters -> Fine-Tuning -> Inference -> Base vs Fine-Tuned Comparison
```

## 3. Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT library
- bitsandbytes

## 4. Feature List

- Dataset preparation
- Tokenization pipeline
- PEFT (Parameter-Efficient Fine-Tuning) setup
- LoRA adapter training
- Inference with fine-tuned adapters
- Base vs. fine-tuned evaluation comparison

## 5. Implementation Plan

1. Phase 1: Dataset preparation and tokenization
2. Phase 2: LoRA adapter configuration and training
3. Phase 3: Inference pipeline and base-vs-fine-tuned evaluation

## 6. Repository Structure

```text
bloom-lora-finetuning/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd bloom-lora-finetuning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Document the commands used to run training, ingestion, or the main pipeline, e.g.:

```bash
# Phase 1: build a cleaned instruction dataset from the bundled example
python -m src.main prepare --source jsonl \
    --input examples/sample_raw.jsonl \
    --output data/processed/clean.jsonl \
    --config configs/dataset.yaml

# or pull a public dataset from the Hugging Face Hub
python -m src.main prepare --source hf --hf-name tatsu-lab/alpaca --limit 2000 \
    --output data/processed/clean.jsonl --config configs/dataset.yaml
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t bloom-lora-finetuning .
docker run -p 8000:8000 bloom-lora-finetuning
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
