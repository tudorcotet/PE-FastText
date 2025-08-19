# Protein Embedding Experiments

Clean, modular framework for comparing embedding methods on protein prediction tasks.

## Overview

The framework compares:
- **FastText** embeddings with optional positional encodings (sinusoidal, learned, RoPE, ALiBi)
- **ESM2** protein language model embeddings (optional, requires PyTorch)

On protein tasks from HuggingFace:
- Fluorescence (regression)
- Stability (regression)
- MPP - Membrane Protein Prediction (regression)
- Beta-lactamase activity (regression)
- SSP - Secondary Structure Prediction (classification)

## Installation

```bash
# From the repo root
pip install -e .  # or pip install -e '.[gpu]' for RoPE/GPU

# Experiment extras
pip install -r protein_experiments/requirements.txt
# Optional: pip install torch transformers  # for ESM2 support
```

## Quick Start

### 1. Pre-train on UniRef50 (Optional)

Pre-training on large protein corpora can improve downstream task performance:

```bash
# Pre-train on sample dataset (quick)
python pretrain_sample.py --dataset tattabio/OG_prot --max-sequences 10000

# Pre-train on UniRef50 (requires download)
python pretrain_uniref50.py --train-split 0.01 --max-sequences 50000
```

### 2. Run a Single Experiment

```bash
# FastText baseline
python run_experiment.py --single --task fluorescence --embedder fasttext

# FastText with pre-trained model
python run_experiment.py --single --task fluorescence --embedder fasttext \
    --config '{"pretrained_path": "models/sample_pretrained.bin", "fine_tune": true}'

# FastText with RoPE positional encoding
python run_experiment.py --single --task fluorescence --embedder fasttext --pos-encoder rope

# FastText with residue tokenization
python run_experiment.py --single --task fluorescence --embedder fasttext --tokenization residue

# ESM2 (if available)
python run_experiment.py --single --task fluorescence --embedder esm2
```

### 3. Run from Configuration File

```bash
# Example configs
python run_experiment.py --config configs/example_fasttext_baseline.yaml
python run_experiment.py --config configs/example_fasttext_rope.yaml
python run_experiment.py --config configs/example_pretrained_frozen.yaml
python run_experiment.py --config configs/example_pretrained_finetuned.yaml
```

### 4. Run Full Comparison

```bash
# Compare all embedders across all tasks
python run_experiment.py --compare --output-dir results/comparison
```

## Configuration

Experiment configurations are defined in YAML files. See `configs/schema.yaml` for the full schema.

Example configuration:

```yaml
task: fluorescence

embedder:
  type: fasttext
  model_path: models/fasttext_model.bin
  pretrained_path: models/uniref50_pretrained.bin  # Optional pre-trained model
  fine_tune: true  # Whether to fine-tune or freeze pre-trained model
  tokenization: kmer  # or "residue"
  k: 6
  dim: 128
  pos_encoder: rope  # Optional: sinusoid, learned, rope, alibi, ft_alibi
  fusion: add  # or "concat"

predictor:
  type: rf  # or "linear"
  n_estimators: 100
  max_depth: 20

data:
  train_size: 0.5
  val_split: 0.1
  max_length: 500
```

## Project Structure

```
protein_experiments/
├── src/
│   ├── __init__.py
│   ├── data.py          # Unified data loading
│   ├── experiment.py    # Core experiment logic
│   ├── pretrain.py      # UniRef50 pre-training
│   └── embedders/       # Embedder implementations
│       ├── __init__.py
│       ├── base.py      # Abstract base class
│       ├── fasttext.py  # FastText embedder
│       └── esm2.py      # ESM2 embedder (optional)
├── configs/             # Configuration files
│   ├── schema.yaml      # Configuration schema
│   └── example_*.yaml   # Example configs
├── run_experiment.py    # Local experiment runner
├── pretrain_sample.py   # Pre-train on small datasets
├── pretrain_uniref50.py # Pre-train on UniRef50
├── modal_app.py         # Modal cloud app (functions)
└── README.md
```

## Pre-training

The framework supports pre-training FastText models on large protein corpora (e.g., UniRef50) before fine-tuning on specific tasks:

### Pre-training Workflow

1. **Pre-train on UniRef50**:
   ```bash
   python pretrain_uniref50.py --output models/uniref50_pretrained.bin \
       --train-split 0.01 --max-sequences 100000 --epochs 5
   ```

2. **Use in experiments**:
   - **Frozen**: Use pre-trained embeddings without fine-tuning
   - **Fine-tuned**: Continue training on task-specific data
   
3. **Benefits**:
   - Better generalization from learning general protein patterns
   - Faster convergence on downstream tasks
   - Especially helpful for small datasets

### Pre-training Options

- `pretrained_path`: Path to pre-trained FastText model
- `fine_tune`: Whether to fine-tune (true) or freeze (false) embeddings
- Works with all positional encodings (RoPE, sinusoidal, etc.)

## Options at a glance

- Positional encoders: `sinusoid`, `learned`, `rope`, `alibi`, `ft_alibi`
- Tokenization: `kmer` (k=3–7) or `residue`
- Predictors: `rf`, `linear`, `mlp`, `xgboost`

## Reproduce the full grid locally (what the Modal app runs)

This mirrors the large grid from `modal_app.py`. Place the two pre-trained models in `protein_experiments/models/` with these names:
- `uniref50_pretrained_residue.bin`
- `uniref50_pretrained_kmer.bin`

Then run locally on CPU (sequential):

```bash
python - <<'PY'
import json
from pathlib import Path
from modal_app import generate_all_experiment_configs
from src.experiment import Experiment

models_dir = Path('models')
results_root = Path('results/runs')
results_root.mkdir(parents=True, exist_ok=True)

configs = generate_all_experiment_configs()

# Optional: narrow the grid for a quick local run
# configs = [c for c in configs if c['task'] in {'fluorescence','stability'} and c['predictor']['type']=='rf' and c['embedder']['type']=='fasttext']

for cfg in configs:
    ep = cfg.get('embedder', {}).get('pretrained_path')
    if ep:
        cfg['embedder']['pretrained_path'] = str(models_dir / Path(ep).name)
    run_name = f"{cfg['task']}_{cfg['embedder']['type']}_{cfg.get('id','exp')}"
    out_dir = results_root / run_name
    cfg['output_dir'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = Experiment(cfg).run()
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(res, f, indent=2)
print('Done. Results in results/runs/*')
PY
```

Summarize all `results.json` to a single Parquet locally (optional):

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

rows = []
for run_dir in Path('results/runs').glob('*'):
    f = run_dir / 'results.json'
    if f.exists():
        data = json.load(open(f))
        rows.append(data)
if rows:
    df = pd.json_normalize(rows)
    out = Path('results/results_summary.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print('Wrote', out)
else:
    print('No results found')
PY
```

## Results Format

Experiments output JSON files with:
- Task and embedder configuration
- Performance metrics (R²/MSE for regression, accuracy/F1 for classification)
- Timing information
- Train/test set sizes

## Modal (Cloud) Execution

For faster pre-training and parallel experiments, use Modal.

### Setup

```bash
pip install modal
modal token new
```

### Dataset helpers (volumes)

These functions use volumes named: datasets=`pe-fasttext-datasets`, models=`pe-fasttext-models`, results=`pe-fasttext-results`.

```bash
# From protein_experiments/
modal run modal_app.py::download_uniref50
# Optional (if .gz already present in the volume):
# modal run modal_app.py::decompress_uniref50
```

### Pre-training on Modal

```bash
modal run modal_app.py::pretrain_uniref50_fasta --train-split 0.01 --tokenization kmer --k 5 --dim 128 --epochs 5
```

### Running Experiments on Modal

```bash
# Full grid (skips runs already completed in results volume)
modal run modal_app.py::run_all_experiments

# Subsets
modal run modal_app.py::run_ssp_experiments
modal run modal_app.py::run_deeploc_experiments
modal run modal_app.py::run_esm2_experiments

# Summaries
modal run modal_app.py::create_summary_parquet
modal run modal_app.py::status
```

## Troubleshooting

- **ESM2 not available**: Install PyTorch and transformers: `pip install torch transformers`
- **Out of memory**: Reduce `batch_size` in ESM2 config or use smaller `train_size`
- **Slow training**: Use the tiny_test.yaml config for quick testing
- **Modal issues**: Check credentials with `modal token validate`

## Citation

If you use this code, please cite the PE-FastText repository.