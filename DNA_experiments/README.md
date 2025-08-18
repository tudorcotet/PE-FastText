# Genomic Sequence Embedding and Benchmarking

This project provides a complete workflow for training and evaluating FastText models on genomic data for classification tasks. It includes scripts for tokenization, hyperparameter tuning, model training, and benchmarking with positional encodings. Most of the scripts have modifiable hyperparameters to allow for for testing different configurations but the default settings should are what was used in the final report.

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

This script provides visual sanity check to verify that the embeddings make sense. It trains a FastText model (either k-mer or BPE) and then generates UMAP plots to visualize the separation between embeddings of real and synthetic DNA sequences. It also explores the effect of different positional encodings (Sinusoidal, RoPE, ALiBi) on the embeddings.

- **Output:** UMAP plots saved in the `figures/` directory.

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
    - `figures/benchmark_results_<model_type>.csv`: CSV file with benchmark results.
    - `figures/benchmark_comparison_<model_type>.png`: Plot comparing the performance of different methods.

**Usage:**
```bash
# To run on all datasets with default models
python 5_benchmark.py --datasets "demo_coding_vs_intergenomic_seqs,human_enhancers_cohn,human_nontata_promoters,human_ocr_ensembl"
```

## Dependencies

You will need to install the required Python packages. You can typically find these at the top of the Python scripts. Key dependencies include:

- `gensim`
- `tokenizers`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `umap-learn`
- `tqdm`
- `genomic-benchmarks`
