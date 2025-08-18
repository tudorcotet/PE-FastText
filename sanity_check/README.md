# Sanity checks (proteins)

Small, visual checks that PE‑FastText captures sensible protein structure signals and positional effects. Produces UMAP plots and a short PDF report.

## Install

```bash
# From repo root
pip install -e .
pip install -r sanity_check/requirements.txt
```

## Run on Modal (recommended)

These functions use volumes named: datasets=`pe-fasttext-datasets`, models=`pe-fasttext-models`, results=`pe-fasttext-results6`.

```bash
pip install modal
modal token new

# Generate plots + PDF and compute UniRef50 stats in parallel
modal run sanity_check/modal_app.py::run_sanity_checks_and_stats
```

Requirements:
- The models volume should contain the pre‑trained FastText models expected by the app:
  - `/models/uniref50_pretrained_10pct_kmer.epoch6.bin`
  - `/models/uniref50_pretrained_full_residue.bin`
- The datasets volume should contain `uniref50.fasta` if you want dataset stats:
  - You can create it with the protein experiments Modal helpers:
    - `modal run protein_experiments/modal_app.py::download_uniref50`
    - (optional) `modal run protein_experiments/modal_app.py::decompress_uniref50`

Outputs:
- Plots in the results volume under `/results/sanity_checks_protein/figures`
- PDF report `/results/sanity_checks_protein/protein_sanity_checks_report.pdf`
- Dataset stats CSVs under `/results/dataset_stats`

## Benchmarks included
- Real vs. shuffled proteins
- Low‑complexity vs. real
- Helix‑like vs. real
- Sheet‑like vs. real
- Disordered‑like vs. real
- Positional motif detection (start/middle/end)


