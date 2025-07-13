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
# Clone the PE-FastText repository first
cd ..
pip install -e .  # or pip install -e ".[gpu]" for RoPE support

# Install experiment dependencies
cd protein_experiments
pip install scikit-learn datasets pyyaml tqdm
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
# Run tiny test configuration
python run_experiment.py --config configs/tiny_test.yaml

# Run full experiment
python run_experiment.py --config configs/example_fasttext_rope.yaml
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
├── modal_app.py         # Modal cloud app
├── run_modal.py         # Modal wrapper script
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

## Implementation Notes

### Positional Encodings

- **Sinusoidal**: Classic transformer-style positional encoding
- **Learned**: Simple embedding table for positions
- **RoPE**: Rotary positional encoding (requires PyTorch)
- **ALiBi**: Linear bias - implemented as surrogate for FastText
- **ft_alibi**: FastText-compatible ALiBi using residue distances

### Tokenization

- **K-mer**: Sliding window of k amino acids (default k=6)
- **Residue**: Individual amino acids as tokens

### Technical Details

1. **ALiBi Implementation**: Since FastText doesn't have attention mechanisms, ALiBi is implemented as a position-aware encoding that approximates the distance-based biasing concept.

2. **Memory Efficiency**: All data processing uses generators to handle large datasets without loading everything into memory.

3. **Modular Design**: Embedders and predictors are easily extensible through abstract base classes.

## Results Format

Experiments output JSON files with:
- Task and embedder configuration
- Performance metrics (R²/MSE for regression, accuracy/F1 for classification)
- Timing information
- Train/test set sizes

## Modal (Cloud) Execution

For faster pre-training and parallel experiments, use Modal:

### Setup

```bash
# Install Modal
pip install modal

# Authenticate (first time only)
modal token new
```

### Pre-training on Modal

```bash
# Pre-train on 1% of UniRef50 (~2-3 hours)
python run_modal.py pretrain --split 0.01

# Pre-train on 10% of UniRef50 (~1 day)
python run_modal.py pretrain --split 0.10
```

### Running Experiments on Modal

```bash
# Run single experiment
python run_modal.py experiment --config configs/example_fasttext_rope.yaml

# Run full comparison (parallel)
python run_modal.py experiment --compare
```

### Deploy as Modal App

```bash
# Deploy for scheduled runs
python run_modal.py deploy
```

## Troubleshooting

- **ESM2 not available**: Install PyTorch and transformers: `pip install torch transformers`
- **Out of memory**: Reduce `batch_size` in ESM2 config or use smaller `train_size`
- **Slow training**: Use the tiny_test.yaml config for quick testing
- **Modal issues**: Check credentials with `modal token validate`

## Citation

If you use this code, please cite the PE-FastText repository.