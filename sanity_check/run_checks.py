from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
import umap
import random

import sys
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent / 'src'))
from pe_fasttext.utils import kmerize
from pe_fasttext.model import PEFastText
from plotting import plot_umap_comparison

PROTEIN_ALPHABET = "ARNDCEQGHILKMFPSTWYV"
HELIX_ALPHABET = "ALMEQ"       # Strong helix formers
SHEET_ALPHABET = "VIFTYW"      # Strong sheet formers
DISORDER_ALPHABET = "PGRQSEK"  # Disorder-promoting residues

# --- Data Loading ---
def load_protein_sequences(n_seqs=100):
    """Loads a list of protein sequences from the Tattabio OG dataset."""
    print(f"Loading {n_seqs} real protein sequences...")
    ds = load_dataset('tattabio/OG', 'train', split='train', streaming=True)
    sequences = []
    for row in tqdm(ds.take(n_seqs * 5), total=n_seqs): # Fetch more to get enough valid seqs
        if 'IGS_seqs' in row and row['IGS_seqs']:
            for seq in row['IGS_seqs']:
                if seq:
                    sequences.append(seq)
                    if len(sequences) >= n_seqs:
                        return sequences
    return sequences

# --- Embedding ---
def get_embeddings(model, sequences, k, pe_type='baseline', fusion='add', tokenization='kmer'):
    """Generate embeddings for a list of sequences for a given PE type."""
    if pe_type == 'baseline':
        tokens = [kmerize(s, k) for s in sequences] if tokenization == 'kmer' else [list(s) for s in sequences]
        vectors = []
        for sent in tokens:
            vecs = [model.wv[kmer] for kmer in sent if kmer in model.wv]
            vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
        return np.vstack(vectors)
    else:
        pe_model = PEFastText(fasttext_model=model, pos_encoder=pe_type, fusion=fusion)
        return pe_model.embed(sequences, k=k, average_sequences=True, tokenization=tokenization)

# --- Benchmark Suite ---
def run_benchmark(model, benchmark_fn, benchmark_name, k, output_dir, tokenization):
    """Runs a benchmark: generates sequences, embeds them, and plots the UMAP."""
    print(f"\n--- Running benchmark: {benchmark_name} ---")
    sequences, labels = benchmark_fn()
    
    pos_encoders = ['baseline', 'sinusoid', 'rope', 'ft_alibi']
    embeds_dict = {}

    for pe in pos_encoders:
        print(f"Generating embeddings for {pe}...")
        embeds = get_embeddings(model, sequences, k, pe_type=pe, tokenization=tokenization)
        embeds_std = StandardScaler().fit_transform(embeds)
        embeds_dict[pe] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeds_std)
        
    plot_umap_comparison(
        embeds_dict,
        labels,
        output_dir / f"{benchmark_name.lower().replace(' ', '_')}.png",
        main_title=f"{benchmark_name} ({tokenization} model)"
    )

# --- Benchmark Definitions ---
def get_shuffled_vs_real_protein(n_seqs=50):
    """Benchmark: Real vs. shuffled protein sequences."""
    real_seqs = load_protein_sequences(n_seqs)
    shuffled_seqs = ["".join(random.sample(s, len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['shuffled'] * len(shuffled_seqs)
    return real_seqs + shuffled_seqs, labels

def get_low_complexity_vs_real_protein(n_seqs=50):
    """Benchmark: Low-complexity vs. real protein sequences."""
    real_seqs = load_protein_sequences(n_seqs)
    low_complexity_seqs = ["".join(np.random.choice(list("AG"), len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['low_complexity'] * len(low_complexity_seqs)
    return real_seqs + low_complexity_seqs, labels

def get_helix_vs_real_protein(n_seqs=50):
    """Benchmark: Alpha-helix-like vs. real protein sequences."""
    real_seqs = load_protein_sequences(n_seqs)
    helix_seqs = ["".join(np.random.choice(list(HELIX_ALPHABET), len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['helix_like'] * len(helix_seqs)
    return real_seqs + helix_seqs, labels

def get_sheet_vs_real_protein(n_seqs=50):
    """Benchmark: Beta-sheet-like vs. real protein sequences."""
    real_seqs = load_protein_sequences(n_seqs)
    sheet_seqs = ["".join(np.random.choice(list(SHEET_ALPHABET), len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['sheet_like'] * len(sheet_seqs)
    return real_seqs + sheet_seqs, labels

def get_disordered_vs_real_protein(n_seqs=50):
    """Benchmark: Disordered-region-like vs. real protein sequences."""
    real_seqs = load_protein_sequences(n_seqs)
    disordered_seqs = ["".join(np.random.choice(list(DISORDER_ALPHABET), len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['disordered_like'] * len(disordered_seqs)
    return real_seqs + disordered_seqs, labels

def get_positional_motifs_protein(n_per_class=20, seq_len=100):
    """Benchmark: Same motif at different positions."""
    motif = "MVLSPADKTN"
    sequences, labels = [], []
    positions = {'start': 0, 'middle': (seq_len - len(motif)) // 2, 'end': seq_len - len(motif)}
    for label, pos in positions.items():
        for _ in range(n_per_class):
            prefix = "".join(np.random.choice(list(PROTEIN_ALPHABET), pos))
            suffix = "".join(np.random.choice(list(PROTEIN_ALPHABET), seq_len - len(prefix) - len(motif)))
            sequences.append(prefix + motif + suffix)
            labels.append(f"motif_{label}")
    return sequences, labels 