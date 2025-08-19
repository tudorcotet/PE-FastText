# Sanity checks (proteins)

Small, visual checks that PE‑FastText captures sensible protein structure signals and positional effects. Produces UMAP plots and a short PDF report.

## Install

```bash
# From repo root
pip install -e .
pip install -r sanity_check/requirements.txt
```

## Run locally

You can run any benchmark locally without Modal. Example (k-mer model, one benchmark):

```bash
python - <<'PY'
from pathlib import Path
from gensim.models import FastText
from sanity_check.run_checks import get_shuffled_vs_real_protein, run_benchmark

# Load a FastText model (train one via protein_experiments/pretrain_sample.py if needed)
model = FastText.load('models/sample_pretrained.bin')  # update path if different

out_dir = Path('sanity_check_local/figures'); out_dir.mkdir(parents=True, exist_ok=True)
run_benchmark(model, get_shuffled_vs_real_protein, 'Shuffled vs. Real Proteins', k=6, output_dir=out_dir, tokenization='kmer')
print('Done. See', out_dir)
PY
```

Notes:
- For residue tokenization, use `tokenization='residue'` and set `k=1`.
- Benchmarks available: `get_shuffled_vs_real_protein`, `get_low_complexity_vs_real_protein`, `get_helix_vs_real_protein`, `get_sheet_vs_real_protein`, `get_disordered_vs_real_protein`, `get_positional_motifs_protein`.

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


