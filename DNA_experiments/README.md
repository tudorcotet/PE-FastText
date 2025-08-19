# Genomic Sequence Embedding and Benchmarking

End-to-end workflow for training and evaluating FastText models on genomic data (k-mer and BPE), including positional encodings.

## Install

```bash
# From repo root (for shared utilities)
pip install -e .

# Extra packages used by these scripts
pip install gensim tokenizers umap-learn matplotlib seaborn pandas scikit-learn tqdm genomic-benchmarks
```

## Workflow

The workflow is divided into five main steps, executed by the corresponding Python scripts.

### Step 1: Train BPE Tokenizer

**Script:** `1_train_BPE_tokenizer.py`

This script trains a Byte-Pair Encoding (BPE) tokenizer on the human genome (GRCh38). The BPE tokenizer learns to segment DNA sequences into subword units, which can be more effective than fixed-size k-mers.

- **Input:** `data/GCA_000001405.15_GRCh38_genomic.fna`
- **Output:** `hg38_tokenizer.json` (The trained BPE tokenizer)

**Usage:**
```bash
python 1_train_BPE_tokenizer.py
```

### Step 2: Hyperparameter Tuning (Optional)

**Script:** `2_hyper_tuning.py`

This script helps you find good hyperparameters for both k-mer and BPE FastText models. It iterates through different values for k-mer size, window size, and epochs. For each combination, it trains a model and evaluates its ability to distinguish between real and randomly generated DNA sequences by measuring the cosine distance of their embeddings.

**Usage:**
```bash
python 2_hyper_tuning.py
```

### Step 3: Visualize Embeddings (Optional)

**Script:** `3_real_vs_fake.py`

This script provides a visual sanity check. It trains a FastText model (k-mer or BPE) and generates UMAP plots for real vs. synthetic DNA. It also compares positional encodings (Sinusoidal, RoPE, ALiBi).

- **Output:** UMAP plots saved under `plots/` (e.g., `plots/umap_real_vs_fake_kmer_multi_pe.png`).

**Usage:**
```bash
# For k-mer model
python 3_real_vs_fake.py --model_type kmer

# For BPE model
python 3_real_vs_fake.py --model_type bpe
```

### Step 4: Train Final Models

**Script:** `4_train_models.py`

This script trains the final k-mer and BPE FastText models on the entire genomic dataset using a set of chosen hyperparameters.

- **Input:** `data/GCA_000001405.15_GRCh38_genomic.fna`, `hg38_tokenizer.json`
- **Output:** Trained models in the `models/` directory (e.g., `fasttext_kmer_hg38.bin`, `fasttext_bpe_hg38.bin`).

**Usage:**
```bash
python 4_train_models.py --vector-size 128 --window 10 --epochs_kmer 10 --epochs_bpe 20
```

### Step 5: Run Benchmark

**Script:** `5_benchmark.py`

This is the final step where the trained models are evaluated on various downstream genomic classification tasks. The script compares the performance of standard FastText embeddings against embeddings enhanced with different positional encodings.

- **Input:** Trained models from Step 4, benchmark datasets from `benchmark_datasets/`.
- **Output:**
    - `benchmark_results_<model_type>.csv` (at repo folder root): results per encoder/dataset/seed
    - `figures/benchmark_comparison_<model_type>_{accuracy,auc,average_precision}.png`: grouped bar charts

**Usage:**
```bash
# To run on all datasets with default models
python 5_benchmark.py --datasets "demo_coding_vs_intergenomic_seqs,human_enhancers_cohn,human_nontata_promoters,human_ocr_ensembl"
```

## Notes

- Input FASTA path defaults to `./data/GCA_000001405.15_GRCh38_genomic.fna`. Adjust via CLI flags if your dataset lives elsewhere.
- For BPE flows, run step 1 first to produce `hg38_tokenizer.json` used by steps 3–5.

## Reproduce the benchmark locally (as in the figures)

1) Train both models (or use your own trained models):

```bash
python 4_train_models.py --vector-size 128 --window 10 --epochs_kmer 10 --epochs_bpe 20
```

2) Run benchmarks for both model types with common encoders and multiple seeds:

```bash
# K-mer
python 5_benchmark.py \
  --kmer_model_path models/fasttext_kmer_hg38.bin \
  --datasets "demo_coding_vs_intergenomic_seqs,human_enhancers_cohn,human_nontata_promoters,human_ocr_ensembl" \
  --encodings sinusoid,rope,alibi \
  --seeds 1,2,3,4,5

# BPE
python 5_benchmark.py \
  --bpe_model_path models/fasttext_bpe_hg38.bin \
  --bpe_tokenizer_path hg38_tokenizer.json \
  --datasets "demo_coding_vs_intergenomic_seqs,human_enhancers_cohn,human_nontata_promoters,human_ocr_ensembl" \
  --encodings sinusoid,rope,alibi \
  --seeds 1,2,3,4,5
```

3) The script writes CSVs like `benchmark_results_{kmer|bpe}.csv` and plots under `figures/`.
