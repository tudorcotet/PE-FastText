# Positionally-Enhanced FastText (PE-FastText)

PE-FastText is a **light-weight, position-aware embedding generator for biological sequences** (DNA, RNA, proteins).
It augments the sub-word semantics of [FastText](https://arxiv.org/abs/1607.04606) with positional encodings (sinusoidal, RoPE, ALiBi or learned) so that each embedding captures both **"what motif is this?"** and **"where does it sit?"**.

With a single lookup table PE-FastText can scan gigabases of raw sequence in linear time, flagging only the most promising regions for deeper analysis by heavy transformer models such as Evo 2 or HyenaDNA.

---

## Highlights

* FastText-style **n-gram semantics** for k-mers (3-7).
* Pluggable **positional encoders**: sinusoid, RoPE, ALiBi, learned.
* Two fusion rules – **`add`** (same dimensionality) or **`concat`** (semantic ∥ position).
* Out-of-the-box **FAISS** similarity search, light downstream heads, ablation grid.
* Remote execution on **[Modal](https://modal.com/)** for large corpus training & evaluation.
* Shipping as a standard **PEP-621** package (`pyproject.toml`), installable with `pipx` or `poetry`.

---

## Installation

```bash
# Option 1 – Poetry (recommended for development)
poetry install --with gpu  # add --with gpu to install torch extras

# Option 2 – pip
pip install pe-fasttext              # cpu-only
pip install pe-fasttext[gpu]          # + PyTorch for RoPE / GPU
```

Python ≥ 3.9 is required.

---

## Quick start

### 1. Download corpora (≈ 100 GB)

```bash
peft-download  \  # <- CLI entry-point
  --corpus uniref50 \
  --corpus mgnify \
  --output ~/data/peft_corpora
```

Each corpus is streamed & deduplicated on the fly (see `pe_fasttext/dataset.py`). You can customise the list with repeated `--corpus` flags.

### 2. Train FastText (semantics only)

```bash
peft-train \
  --corpus ~/data/peft_corpora/uniref50.fasta \
  --kmer 3 4 5 \
  --dim 512 \
  --workers 32 \
  --epochs 10 \
  --output ~/models/peft/fasttext_protein.bin
```

Training runs locally by default. Add `--modal` to spin up a remote Modal GPU/CPU job (see `modal/fasttext.py`).

### 3. Attach positional module & write embeddings

```bash
peft-embed \
  --model ~/models/peft/fasttext_protein.bin \
  --pos-encoder sinusoid \
  --fusion add \
  --fasta path/to/query_sequences.fasta \
  --index-out query.faiss
```

Embeddings are streamed into a FAISS (L2 or cosine) index ready for k-NN look-ups.

### 4. Benchmarks & ablations

```bash
# Suite of unsupervised + supervised benchmarks (Table 1–2 in the paper)
peft-bench run --config config/default.yaml --modal

# 5-axis ablation grid (Table 3)
peft-ablate grid --config config/ablation.yaml --modal
```

Results are written to `results/` as CSV + interactive HTML plots.

---

## Project layout

```
pe-fasttext/
├── src/pe_fasttext/        # main library
│   ├── tokenization.py     # k-mer streaming
│   ├── position_encodings.py
│   ├── fasttext_utils.py   # wrapper around gensim
│   ├── model.py            # fuse semantics + position
│   ├── dataset.py          # corpus download / parsing
│   ├── benchmark.py        # unsupervised + supervised tasks
│   ├── ablation.py         # grid search utilities
│   ├── cli.py              # Typer app exposing 5 commands
│   └── …
├── modal/
│   ├── fasttext.py         # Modal image & training function
│   ├── benchmarks.py       # Modal stubs for long jobs
│   └── …
├── scripts/                # ad-hoc helpers
├── config/                 # YAML config templates
├── examples/               # notebooks & minimal pipelines
├── tests/                  # pytest unit tests (soon)
├── pyproject.toml
└── README.md
```

---

## Modal remote execution

Large-scale training (≈ billions of k-mers) and evaluation are wrapped in [Modal](https://modal.com/) functions so you can run:

```bash
peft-train --modal …        # dispatch to cloud CPUs/GPUs
peft-bench run --modal
```

Set your token once (`modal token new`) and the CLI will forward jobs.

---

## Configuration

YAML files in `config/` fully describe a run – corpora, hyper-parameters, evaluation datasets. All CLI commands take a `--config` override.

---

## Citing

If you use PE-FastText in academic work, please cite the accompanying MSc thesis:

```
@misc{bohl2024peft,
  title   = {Positionally-enhanced FastText embeddings for biological sequences},
  author  = {Bohl, Michael and Cotet, Tudor-Stefan},
  year    = {2024},
  note    = {MSc thesis, ETH Zürich}
}
```

---

## License

MIT. See `LICENSE` file. 