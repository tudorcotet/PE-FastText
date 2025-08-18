"""Benchmark vanilla FastText vs positional-enhanced variants (sinusoid, RoPE, ALiBi)
for genomic classification datasets that suit non-contextual (bag/pooled) embeddings.

Usage examples:

# K-mer model (default paths)
python benchmark_positional_fasttext.py --model_type kmer \
  --model_path models/grch38_fasttext_kmer.model \
  --datasets human_ocr_ensembl,human_enhancers_cohn,human_nontata_promoters,demo_coding_vs_intergenomic_seqs

# BPE model
python benchmark_positional_fasttext.py --model_type bpe \
  --model_path models/fasttext_bpe_hg38.bin \
  --bpe_tokenizer_path hg38_tokenizer.json \
  --datasets human_ocr_ensembl,human_enhancers_cohn,human_nontata_promoters,demo_coding_vs_intergenomic_seqs

Outputs:
  CSV: positional_ft_results_<model_type>.csv
  Plot: positional_ft_comparison_<model_type>.png
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from gensim.models.fasttext import FastText  # type: ignore

warnings.filterwarnings("ignore")

# Genomic benchmarks
try:
    from genomic_benchmarks.loc2seq import download_dataset
except ImportError:
    print("Please install genomic-benchmarks: pip install genomic-benchmarks")
    raise

#############################################
# Positional Encoding Implementations
#############################################
class BasePositionalEncoding:
    name: str = "base"
    def __init__(self, dim: int):
        self.dim = dim
    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

class SinusoidalEncoding(BasePositionalEncoding):
    name = "sinusoid"
    def __call__(self, positions):
        positions = np.asarray(positions)[:, None]
        d = np.arange(self.dim)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (d // 2)) / self.dim)
        angle_rads = positions * angle_rates
        sines = np.sin(angle_rads[:, 0::2])
        coses = np.cos(angle_rads[:, 1::2])
        return np.concatenate([sines, coses], axis=-1)

class RoPEEncoding(BasePositionalEncoding):
    name = "rope"
    def __init__(self, dim: int):
        if dim % 2 != 0:
            raise ValueError("RoPE requires even dimension")
        super().__init__(dim)
        self.half_dim = dim // 2
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, self.half_dim, 2) / self.half_dim))
    def __call__(self, positions):
        positions = np.asarray(positions)
        freqs = np.einsum("i,j->ij", positions, self.inv_freq)
        emb = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
        return np.tile(emb, (1, 2))

class ALiBiEncoding(BasePositionalEncoding):
    name = "alibi"
    def __call__(self, positions):
        positions = np.asarray(positions)[:, None]
        slopes = np.linspace(0, 1, self.dim)[None, :]
        return positions * slopes

ENCODERS = {c.name: c for c in [SinusoidalEncoding, RoPEEncoding, ALiBiEncoding]}

def build_positional_encoder(name: str, dim: int):
    return ENCODERS[name](dim)

#############################################
# Tokenizers
#############################################
class SimpleTokenizer:
    def __init__(self, k=6):
        self.k = k
    def tokenize(self, sequence: str):
        if self.k == 1:
            return list(sequence)
        L = len(sequence)
        if L < self.k:
            return []
        return [sequence[i:i+self.k] for i in range(L - self.k + 1)]

class HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def tokenize(self, text: str):
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

#############################################
# Core Benchmark Class
#############################################
class FastTextPositionalBenchmark:
    def __init__(self, model_path: str, model_type: str, k: int = 6, chunk_size: int = 1000, bpe_tokenizer_path: str = "hg38_tokenizer.json"):
        self.model_type = model_type
        self.chunk_size = chunk_size
        print(f"Loading FastText model from {model_path} ...")
        self.model: FastText = FastText.load(model_path)
        self.vector_size = self.model.vector_size
        print(f"Loaded. Vector size={self.vector_size}")
        if model_type == 'kmer':
            self.tokenizer = SimpleTokenizer(k=k)
            self.use_kmer = True
        else:
            self.tokenizer = self._init_bpe(bpe_tokenizer_path)
            self.use_kmer = False

    def _init_bpe(self, path: str):
        try:
            if os.path.exists(path):
                from tokenizers import Tokenizer  # type: ignore
                print(f"Loading BPE tokenizer JSON: {path}")
                return HFTokenizerAdapter(Tokenizer.from_file(path))
        except Exception as e:
            print(f"Failed to load HF tokenizer ({path}): {e}")
        for cand in ['bpe_tokenizer.pkl', 'bpe_tokenizer2.pkl']:
            if os.path.exists(cand):
                try:
                    import pickle
                    with open(cand, 'rb') as f:
                        tok = pickle.load(f)
                    if hasattr(tok, 'tokenize'):
                        print(f"Loaded pickled tokenizer: {cand}")
                        return tok
                    return HFTokenizerAdapter(tok)
                except Exception as e:
                    print(f"Failed to load pickle {cand}: {e}")
        print("Falling back to character tokenizer (k=1)")
        return SimpleTokenizer(k=1)

    def _embed_tokens(self, tokens, encoding: str):
        if not tokens:
            return np.zeros(self.vector_size)
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vecs:
            return np.zeros(self.vector_size)
        mat = np.vstack(vecs)
        if encoding != 'none':
            try:
                encoder = build_positional_encoder(encoding, self.vector_size)
                pos_enc = encoder(np.arange(mat.shape[0]))
                mat = mat + pos_enc
            except Exception as e:
                print(f"Positional encoding error ({encoding}): {e}")
        return mat.mean(axis=0)

    def embed_sequence(self, seq: str, encoding: str):
        if self.use_kmer:
            tokens = self.tokenizer.tokenize(seq)
            return self._embed_tokens(tokens, encoding)
        # BPE: chunk then average chunk embeddings
        chunk_embs = []
        for i in range(0, len(seq), self.chunk_size):
            chunk = seq[i:i+self.chunk_size]
            if len(chunk) <= 10:
                continue
            toks = self.tokenizer.tokenize(chunk)
            emb = self._embed_tokens(toks, encoding)
            if emb is not None:
                chunk_embs.append(emb)
        if not chunk_embs:
            return np.zeros(self.vector_size)
        # Optional per-chunk positional encoding (treat each chunk as a position)
        if encoding != 'none':
            try:
                encoder = build_positional_encoder(encoding, self.vector_size)
                pos_enc = encoder(np.arange(len(chunk_embs)))
                chunk_embs = [e + p for e, p in zip(chunk_embs, pos_enc)]
            except Exception as e:
                print(f"Chunk-level positional encoding error ({encoding}): {e}")
        return np.vstack(chunk_embs).mean(axis=0)

    #############################
    # Data Loading (classification-style tasks)
    #############################
    def load_dataset(self, dataset_name: str):
        print(f"Loading dataset: {dataset_name}")
        benchmark_dir = "benchmark_datasets"
        train_csv = os.path.join(benchmark_dir, f"{dataset_name}_train_df.csv")
        test_csv = os.path.join(benchmark_dir, f"{dataset_name}_test_df.csv")
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            tr = pd.read_csv(train_csv)
            te = pd.read_csv(test_csv)
            train = [(r.sequence, r.label) for r in tr.itertuples()]
            test = [(r.sequence, r.label) for r in te.itertuples()]
            print(f"Loaded from {benchmark_dir}: train={len(train)} test={len(test)}")
            return train, test
        else:
            print(f"Could not find {train_csv} or {test_csv}")
            return None, None

    def _load_from_folders(self, base_path: str):
        import glob
        data = []
        pos_dir = os.path.join(base_path, 'positive')
        neg_dir = os.path.join(base_path, 'negative')
        for label, d in [(1, pos_dir), (0, neg_dir)]:
            if os.path.exists(d):
                for fp in glob.glob(os.path.join(d, '*.txt')):
                    try:
                        with open(fp, 'r') as f:
                            seq = f.read().strip()
                        if seq:
                            data.append((seq, label))
                    except Exception as e:
                        print(f"Read error {fp}: {e}")
        print(f"Loaded {len([x for x in data if x[1]==1])} pos / {len([x for x in data if x[1]==0])} neg from {base_path}")
        return data

    #############################
    # Evaluation
    #############################
    def evaluate(self, train_data, test_data, encoding: str, seed: int):
        print(f"Encoding={encoding} seed={seed}")
        X_train, y_train = [], []
        for seq, lab in tqdm(train_data, desc=f"Train {encoding}"):
            X_train.append(self.embed_sequence(seq, encoding))
            y_train.append(lab)
        X_test, y_test = [], []
        for seq, lab in tqdm(test_data, desc=f"Test {encoding}"):
            X_test.append(self.embed_sequence(seq, encoding))
            y_test.append(lab)
        X_train = np.vstack(X_train); y_train = np.array(y_train)
        X_test = np.vstack(X_test); y_test = np.array(y_test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:,1]
        preds = clf.predict(X_test_s)
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        acc = (preds == y_test).mean()
        return dict(encoding=encoding, auc=auc, average_precision=ap, accuracy=acc, random_state=seed)

    def run(self, datasets, seeds, encodings):
        rows = []
        for seed in seeds:
            print(f"==== Seed {seed} ====")
            for ds in datasets:
                tr, te = self.load_dataset(ds)
                if not tr or not te:
                    print(f"Skipping {ds} (data missing)")
                    continue
                for enc in encodings:
                    try:
                        res = self.evaluate(tr, te, enc, seed)
                        res['dataset'] = ds
                        res['model_type'] = self.model_type
                        rows.append(res)
                        print(f"{ds} {enc}: AUC={res['auc']:.4f} AP={res['average_precision']:.4f} Acc={res['accuracy']:.4f}")
                    except Exception as e:
                        print(f"Error {ds} {enc}: {e}")
        return pd.DataFrame(rows)

#############################################
# Plotting
#############################################
def create_grouped_plot(ax, metric_name, metric_mean_col, metric_sem_col, y_label, encoders_list, complete_summary, datasets):
    # Define proper display names for legend
    encoder_display_names = {
        'none': 'None',
        'sinusoid': 'Sinusoidal',
        'rope': 'RoPE',
        'alibi': 'AliBi',
        'HyenaDNA': 'HyenaDNA'
    }
    
    encoder_colors = {
        'none': '#3f4a8c',          # Navy blue
        'sinusoid': '#d4a31a',      # Yellow/Gold
        'rope': '#1a4d66',          # Dark blue
        'alibi': '#5fa052',         # Green
        'HyenaDNA': '#8b1538'       # Dark red
    }

    # Calculate bar width based on number of encoders
    width = 0.8 / len(encoders_list)
    dataset_positions = np.arange(len(datasets))
    
    plotted_encoders = []
    
    for i, encoder in enumerate(encoders_list):
        means = []
        errors = []
        valid_data_exists = False
        
        for dataset in datasets:
            data_row = complete_summary[
                (complete_summary['encoding'] == encoder) &
                (complete_summary['dataset'] == dataset)
            ]
            
            if not data_row.empty:
                mean_val = data_row[metric_mean_col].values[0]
                
                if not pd.isna(mean_val):
                    means.append(mean_val)
                    valid_data_exists = True
                    
                    sem_val = data_row[metric_sem_col].values[0]
                    if not pd.isna(sem_val) and sem_val > 0:
                        errors.append(sem_val)
                    else:
                        errors.append(0)
                else:
                    means.append(np.nan)
                    errors.append(0)
            else:
                means.append(np.nan)
                errors.append(0)
        
        if valid_data_exists:
            offset = (i - (len(encoders_list) - 1) / 2) * width
            bar_positions = dataset_positions + offset
            
            valid_mask = ~np.isnan(means)
            
            if np.any(valid_mask):
                valid_positions = bar_positions[valid_mask]
                valid_means = np.array(means)[valid_mask]
                valid_errors = np.array(errors)[valid_mask]
                
                ax.bar(valid_positions, valid_means, width,
                       yerr=valid_errors, capsize=3,
                       label=encoder_display_names.get(encoder, encoder),
                       color=encoder_colors.get(encoder),
                       error_kw={'linewidth': 1})
                
                for pos, mean_val, error_val in zip(valid_positions, valid_means, valid_errors):
                    text_y = mean_val + error_val + 0.01
                    ax.text(pos, text_y, f'{mean_val:.4f}',
                           ha='center', va='bottom', fontsize=8,
                           color='black', weight='normal')
                
                plotted_encoders.append(encoder)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'Performance by Dataset - {metric_name}', fontsize=14)
    
    ax.set_xticks(dataset_positions)
    display_datasets = [d.replace('demo_', '') for d in datasets]
    ax.set_xticklabels(display_datasets, rotation=45, ha='right')
    
    ax.set_ylim(0.5, 1)
    
    if plotted_encoders:
        ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1))

def plot_results(df: pd.DataFrame, model_type: str, out_prefix: str):
    if df.empty:
        print("No results to plot")
        return

    # Clean the data
    df_clean = df.dropna(subset=['dataset', 'encoding']).copy()

    # Hardcoded HyenaDNA data for comparison
    hyena_data_static = {
        'dataset': [
            'demo_coding_vs_intergenomic_seqs', 'human_enhancers_cohn',
            'human_nontata_promoters','human_ocr_ensembl'
        ],
        'accuracy': [91.3, 74.2, 96.6, 80.9]
    }
    hyena_df = pd.DataFrame(hyena_data_static)
    # Keep only datasets present in this run (robust to varying dataset selections)
    hyena_df = hyena_df[hyena_df['dataset'].isin(df_clean['dataset'].unique())].copy()
    hyena_df['accuracy'] = hyena_df['accuracy'] / 100
    hyena_df['encoding'] = 'HyenaDNA'

    # Calculate mean and standard error for regular encoders
    summary = df_clean.groupby(['encoding', 'dataset']).agg({
        'auc': ['mean', 'sem'],
        'average_precision': ['mean', 'sem'],
        'accuracy': ['mean', 'sem']
    }).reset_index()

    summary.columns = ['encoding', 'dataset',
                       'auc_mean', 'auc_sem',
                       'average_precision_mean', 'average_precision_sem',
                       'accuracy_mean', 'accuracy_sem']

    # Prepare HyenaDNA summary
    hyena_summary_list = []
    for _, row in hyena_df.iterrows():
        hyena_summary_list.append({
            'encoding': 'HyenaDNA',
            'dataset': row['dataset'],
            'auc_mean': np.nan, 'auc_sem': np.nan,
            'average_precision_mean': np.nan, 'average_precision_sem': np.nan,
            'accuracy_mean': row['accuracy'], 'accuracy_sem': 0,
        })
    hyena_summary = pd.DataFrame(hyena_summary_list)

    complete_summary = pd.concat([summary, hyena_summary], ignore_index=True)

    # Get unique encoders and datasets
    encoders_auc_ap = ['none', 'sinusoid', 'rope', 'alibi']
    encoders_acc = ['none', 'sinusoid', 'rope', 'alibi', 'HyenaDNA']
    datasets = sorted(list(df_clean['dataset'].unique()))

    metrics_to_plot = [
        {'metric': 'AUC', 'mean_col': 'auc_mean', 'sem_col': 'auc_sem', 'encoders': encoders_auc_ap, 'y_label': 'AUC'},
        {'metric': 'Average Precision', 'mean_col': 'average_precision_mean', 'sem_col': 'average_precision_sem', 'encoders': encoders_auc_ap, 'y_label': 'Average Precision'},
        {'metric': 'Accuracy', 'mean_col': 'accuracy_mean', 'sem_col': 'accuracy_sem', 'encoders': encoders_acc, 'y_label': 'Accuracy'}
    ]

    for plot_info in metrics_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        create_grouped_plot(ax, plot_info['metric'], plot_info['mean_col'], plot_info['sem_col'], plot_info['y_label'], plot_info['encoders'], complete_summary, datasets)
        
        fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        metric_name_file = plot_info['metric'].lower().replace(' ', '_')
        fig_path = f"{out_prefix}_comparison_{model_type}_{metric_name_file}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {fig_path}")

#############################################
# Main
#############################################

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark vanilla FastText vs positional encodings for k-mer and BPE models.")
    p.add_argument('--kmer_model_path', default='models/fasttext_kmer_hg38.bin', help='Path to k-mer FastText model file')
    p.add_argument('--bpe_model_path', default='models/fasttext_bpe_hg38.bin', help='Path to BPE FastText model file')
    p.add_argument('--bpe_tokenizer_path', default='hg38_tokenizer.json')
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--chunk_size', type=int, default=1000)
    p.add_argument('--datasets', default = "demo_coding_vs_intergenomic_seqs, human_enhancers_cohn,human_nontata_promoters,human_ocr_ensembl", help='Comma-separated dataset names')
    p.add_argument('--seeds', default='42', help='Comma-separated random seeds')
    p.add_argument('--encodings', default='sinusoid,rope,alibi', help='Comma-separated positional encodings to test (baseline none always added)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    pos_list = [e.strip() for e in args.encodings.split(',') if e.strip()]
    # Ensure only supported
    for e in pos_list:
        if e not in ENCODERS:
            raise ValueError(f"Unsupported encoding: {e}. Choices: {list(ENCODERS.keys())}")
    encodings = ['none'] + pos_list

    model_configs = [
        {
            'model_type': 'bpe',
            'model_path': args.bpe_model_path,
            'bpe_tokenizer_path': args.bpe_tokenizer_path,
        },
        {
            'model_type': 'kmer',
            'model_path': args.kmer_model_path,
        }        
    ]

    for config in model_configs:
        model_type = config['model_type']
        print(f"\n\n{'='*20} Processing for {model_type.upper()} model {'='*20}")

        out_csv = f"benchmark_results_{model_type}.csv"
        results = None

        if os.path.exists(out_csv):
            print(f"Found existing results file: {out_csv}. Loading...")
            results = pd.read_csv(out_csv)
        else:
            print(f"No results file found at {out_csv}. Running benchmark...")
            if not os.path.exists(config['model_path']):
                print(f"Model not found: {config['model_path']}. Skipping.")
                continue

            params = {
                'model_path': config['model_path'],
                'model_type': model_type,
                'k': args.k,
                'chunk_size': args.chunk_size,
            }
            if 'bpe_tokenizer_path' in config:
                params['bpe_tokenizer_path'] = config['bpe_tokenizer_path']

            bench = FastTextPositionalBenchmark(**params)
            results = bench.run(datasets, seeds, encodings)
            
            if results.empty:
                print(f"No results produced for {model_type}.")
            else:
                results.to_csv(out_csv, index=False)
                print(f"Saved results to {out_csv}")

        if results is not None and not results.empty:
            plot_results(results, model_type, out_prefix='figures/benchmark')
        else:
            print(f"No results to plot for {model_type}.")