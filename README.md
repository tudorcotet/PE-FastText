# Positionally-Enhanced FastText (PE‑FastText)

Light-weight, position-aware embeddings for biological sequences (DNA/RNA/proteins). FastText semantics + optional positional encodings (sinusoid, learned, RoPE, ALiBi) and simple downstream predictors.

This repo contains the core library (`src/pe_fasttext`) and two experiment suites:
- Protein tasks: `protein_experiments/`
- Genomic (DNA) tasks: `DNA_experiments/`
- Sanity checks: `sanity_check/`

---

## Install

Python 3.9+.

```bash
# Install the library so experiments can import pe_fasttext
pip install -e .
```

Each subproject documents its own extras.

---

## How to run experiments

- Protein experiments (local and Modal): see `protein_experiments/README.md`.
  - Local: single runs, config runs, batch comparisons, pretraining
  - Modal: dataset download, pretrain, full grid, summaries

- DNA experiments (local): see `DNA_experiments/README.md`.
  - Train tokenizer and models; run benchmarks for k‑mer and BPE with positional encodings

- Sanity checks (quick visuals and UMAPs): see `sanity_check/README.md`.

---

## Data sources

Protein tasks are pulled from HuggingFace (see `protein_experiments/src/data.py`). DNA benchmarks are local CSVs under `DNA_experiments/benchmark_datasets/`.

---
