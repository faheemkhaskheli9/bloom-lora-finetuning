# Architecture Notes: BLOOM LoRA Fine-Tuning

## Pipeline

```text
Dataset -> Tokenization -> BLOOM Base Model + LoRA Adapters -> Fine-Tuning -> Inference -> Base vs Fine-Tuned Comparison
```

## Components

- Dataset preparation
- Tokenization pipeline
- PEFT (Parameter-Efficient Fine-Tuning) setup
- LoRA adapter training
- Inference with fine-tuned adapters
- Base vs. fine-tuned evaluation comparison

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
