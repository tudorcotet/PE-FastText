# Positionally-Enhanced FastText (PE‑FastText)

Light-weight, position-aware embeddings for biological sequences (DNA/RNA/proteins). FastText semantics + optional positional encodings (sinusoid, learned, RoPE, ALiBi) and simple downstream predictors.

This repo contains the core library (`src/pe_fasttext`) and a reproducible experiment suite for proteins in `protein_experiments/`.

---

## Install

Python 3.9+.

```bash
# 1) Install the library so experiments can import pe_fasttext
pip install -e .  # from repo root

# 2) Install experiment extras
pip install -r protein_experiments/requirements.txt
# Optional: if you plan to use ESM2 or RoPE on GPU, ensure a working PyTorch install
```

---

## Run locally

All commands are run from `protein_experiments/`.

```bash
cd protein_experiments
```

- Single experiment (FastText baseline):

```bash
python run_experiment.py --single --task fluorescence --embedder fasttext
```

- With positional encoding (example: RoPE) or residue tokenization:

```bash
python run_experiment.py --single --task fluorescence --embedder fasttext --pos-encoder rope
python run_experiment.py --single --task fluorescence --embedder fasttext --tokenization residue
```

- With ESM2 (requires torch+transformers):

```bash
python run_experiment.py --single --task fluorescence --embedder esm2
```

- From a config file:

```bash
python run_experiment.py --config configs/example_fasttext_rope.yaml
```

- Batch comparison across tasks/embedders (writes to `results/`):

```bash
python run_experiment.py --compare --output-dir results/comparison
```

### Pre-training (optional)

- Quick pre-train on a small HF dataset (minutes):

```bash
python pretrain_sample.py --dataset tattabio/OG_prot --max-sequences 10000 --output models/sample_pretrained.bin
```

- UniRef50 pre-train via HF streaming (hours to days depending on split):

```bash
python pretrain_uniref50.py --train-split 0.01 --epochs 5 --output models/uniref50_pretrained_kmer5.bin
```

Use the resulting `--pretrained_path` with `run_experiment.py` configs to freeze or fine‑tune.

---

## Run on Modal (cloud)

Great for UniRef50 pre-training and large experiment grids. First-time setup:

```bash
pip install modal
modal token new
```

Commands below are run from `protein_experiments/` and use named volumes:
- datasets: `pe-fasttext-datasets`
- models: `pe-fasttext-models`
- results: `pe-fasttext-results7`

- Download UniRef50 into the datasets volume and/or decompress:

```bash
modal run modal_app.py::download_uniref50
# If you already have uniref50.fasta.gz in the volume, you can just decompress:
# modal run modal_app.py::decompress_uniref50
```

- Pre-train on UniRef50 (customize args as needed):

```bash
modal run modal_app.py::pretrain_uniref50_fasta --train-split 0.10 --tokenization kmer --k 5 --dim 128 --epochs 10
```

- Run experiment grids (CPU workers in parallel):

```bash
# All experiments (skips runs already completed in the results volume)
modal run modal_app.py::run_all_experiments

# Subsets
modal run modal_app.py::run_ssp_experiments
modal run modal_app.py::run_deeploc_experiments
modal run modal_app.py::run_esm2_experiments
```

- Summarize many JSON results into a single Parquet file:

```bash
modal run modal_app.py::create_summary_parquet
```

Results live under the `pe-fasttext-results7` volume (`/results/...`). You can browse/download them from the Modal UI or via the `modal volume` CLI.

---

## Data sources

Downstream tasks are pulled automatically from HuggingFace datasets in `src/data.py`:
- `InstaDeepAI/true-cds-protein-tasks` (fluorescence, stability, melting_point, beta_lactamase_complete, SSP)
- `bloyal/deeploc` (multilabel subcellular localization)

UniRef50 for pre-training is streamed from HF or downloaded via Modal helpers.

---

## Citation

If you use PE‑FastText, please cite the accompanying MSc thesis (placeholder):

```
@misc{pefasttext2024,
  title   = {Positionally-enhanced FastText embeddings for biological sequences},
  author  = {Bohl, Michael and Cotet, Tudor-Stefan},
  year    = {2024}
}
```

---

## License

MIT. See `LICENSE`.