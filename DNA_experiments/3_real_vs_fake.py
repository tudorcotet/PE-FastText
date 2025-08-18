import argparse
import random
import numpy as np
import umap
import matplotlib.pyplot as plt
from gensim.models import FastText
from tokenizers import Tokenizer
import math

# --- Utility Functions ---

def read_fasta(filepath):
    """Reads sequences from a FASTA file."""
    sequences = []
    with open(filepath, 'r') as f:
        sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence.upper())
                sequence = ""
            else:
                sequence += line
        if sequence:
            sequences.append(sequence.upper())
    return sequences

def generate_random_dna(length):
    """Generates a random DNA sequence of a given length."""
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

# --- k-mer Model Functions ---

def train_kmer_fasttext_model(sequences, k, vector_size=128, window=5, min_count=1, epochs=5):
    """Trains a new FastText model on k-mers from the given sequences."""
    print(f"Training k-mer FastText model with k={k}, window={window}, epochs={epochs}...")
    sentences = []
    for seq in sequences:
        sentence = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        if sentence:
            sentences.append(sentence)
    
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1
    )
    return model

# --- BPE Model Functions ---

def train_bpe_fasttext_model(sequences, tokenizer, vector_size=512, window=5, min_count=1, epochs=5):
    """Trains a new FastText model on BPE tokens from the given sequences."""
    print(f"Training BPE FastText model with window={window}, epochs={epochs}...")
    
    tokenized_sequences = []
    for seq in sequences:
        encoding = tokenizer.encode(seq)
        tokenized_sequences.append(encoding.tokens)

    model = FastText(
        sentences=tokenized_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1
    )
    return model

# --- Positional Encoding Classes ---

class BasePositionalEncoding:
    """Abstract base class for positional encodings."""
    name: str = "base"
    def __init__(self, dim: int):
        self.dim = dim
    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        raise NotImplementedError

class SinusoidalEncoding(BasePositionalEncoding):
    """Classic sine/cosine encoding from *Attention Is All You Need*. """
    name = "sinusoid"
    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)[:, np.newaxis]
        d = np.arange(self.dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (d // 2)) / self.dim)
        angle_rads = positions * angle_rates
        sines = np.sin(angle_rads[:, 0::2])
        coses = np.cos(angle_rads[:, 1::2])
        return np.concatenate([sines, coses], axis=-1)

class RoPEEncoding(BasePositionalEncoding):
    """Rotary positional encoding."""
    name = "rope"
    def __init__(self, dim: int):
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even dimension")
        super().__init__(dim)
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)[:, np.newaxis]
        freqs = positions * self.inv_freq
        emb = np.zeros((len(positions), self.dim), dtype=np.float32)
        emb[:, 0::2] = np.cos(freqs)
        emb[:, 1::2] = np.sin(freqs)
        return emb

class ALiBiEncoding(BasePositionalEncoding):
    """Attention with Linear Biases simplified to a static slope vector."""
    name = "alibi"
    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)[:, np.newaxis]
        slopes = np.linspace(0, 1, self.dim)[np.newaxis, :]
        return positions * slopes

ENCODERS = {cls.name: cls for cls in [SinusoidalEncoding, RoPEEncoding, ALiBiEncoding]}

def build_positional_encoder(name: str, dim: int, **kwargs) -> BasePositionalEncoding:
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder: {name}")
    return ENCODERS[name](dim=dim, **kwargs)

# --- Embedding Generation ---

def create_embeddings_with_pe(model, sequences, model_type, tokenizer=None, k=None, pe_type='none', mode='mean'):
    """
    Creates a 'continuous' embedding where each dimension is a weighted average of all token embeddings.
    The weights are determined by a Gaussian function whose center (focus point) moves from the first token
    to the last token as the embedding dimension goes from the first to the last. The width of the
    Gaussian is smallest at the ends and largest in the middle to approximate a uniform average for the
    central dimensions.
    Optionally applies positional encoding before this aggregation.
    """
    d_model = model.vector_size
    all_embeddings = []

    for seq in sequences:
        seq = ''.join(c for c in seq if c in 'ATGC')
        
        token_embeddings = None
        num_tokens = 0

        if model_type == 'bpe':
            if not tokenizer:
                raise ValueError("Tokenizer must be provided for BPE model.")
            encoding = tokenizer.encode(seq)
            tokens = encoding.tokens
            if not tokens:
                all_embeddings.append(np.zeros(d_model))
                continue
            valid_tokens = [token for token in tokens if token in model.wv]
            if not valid_tokens:
                all_embeddings.append(np.zeros(d_model))
                continue
            token_embeddings = np.array([model.wv[token] for token in valid_tokens])

        elif model_type == 'kmer':
            if not k:
                raise ValueError("k must be provided for k-mer model.")
            if len(seq) < k:
                all_embeddings.append(np.zeros(d_model))
                continue
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            if not kmers:
                all_embeddings.append(np.zeros(d_model))
                continue
            valid_tokens = [kmer for kmer in kmers if kmer in model.wv]
            if not valid_tokens:
                all_embeddings.append(np.zeros(d_model))
                continue
            token_embeddings = np.array([model.wv[token] for token in valid_tokens])
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if token_embeddings is None or token_embeddings.shape[0] == 0:
            all_embeddings.append(np.zeros(d_model))
            continue
        
        num_tokens = token_embeddings.shape[0]

        if pe_type != 'none' and num_tokens > 0:
            positions = list(range(num_tokens))
            encoder = build_positional_encoder(name=pe_type, dim=d_model)
            pe = encoder(positions)
            token_embeddings += pe

        if num_tokens == 1:
            all_embeddings.append(token_embeddings[0])
            continue
        
        # --- Continuous Weighted Aggregation ---
        if mode == 'CVA':
            # Parameters for the weighting function
            sigma_min = 1.0
            sigma_max_factor = 2.0 # sigma_max will be num_tokens * sigma_max_factor
        
            j_coords = np.arange(d_model)
            i_coords = np.arange(num_tokens)

            focus_points = (num_tokens - 1) * j_coords / (d_model - 1)

            j_normalized = j_coords / (d_model - 1)
            parabola = 1 - (2 * j_normalized - 1)**2
            sigma_max = num_tokens * sigma_max_factor
            sigmas = sigma_min + (sigma_max - sigma_min) * parabola

            i_matrix = i_coords[:, np.newaxis]
            focus_matrix = focus_points[np.newaxis, :]
            sigma_matrix = sigmas[np.newaxis, :]

            weights = np.exp(-(i_matrix - focus_matrix)**2 / (2 * sigma_matrix**2))
            weights /= np.sum(weights, axis=0, keepdims=True)
        
        elif mode == 'mean': # effectively averages the tokens
            weights = np.ones((num_tokens, d_model)) / num_tokens

        else: # only CWA and mean are defined
            raise ValueError(f"Unknown mode: {mode}")

        final_embedding = np.sum(weights * token_embeddings, axis=0)
        all_embeddings.append(final_embedding)

    return np.array(all_embeddings)


# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate UMAPs of real vs. synthetic DNA sequences.")
    parser.add_argument("--model_type", type=str, default='kmer', choices=['kmer', 'bpe'], help="Type of model to use.")
    parser.add_argument("--fasta_path", type=str, default="./data/GCA_000001405.15_GRCh38_genomic.fna", help="Path to the genomic sequence corpus.")
    parser.add_argument("--train_fraction", type=float, default=0.1, help="Fraction of the corpus to use for training.")
    parser.add_argument("--num_sequences_plot", type=int, default=200, help="Number of sequences to use for the UMAP plot.")
    parser.add_argument("--sequence_length", type=int, default=2000, help="Length of sequences for training and evaluation.")
    parser.add_argument("--vector_size", type=int, default=128, help="Size of the vectors.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--window_size", type=int, default=5, help="Context window size for FastText.")
    parser.add_argument("--kmer_length", type=int, default=6, help="K-mer length for the k-mer model.")
    args = parser.parse_args()

    print("Loading and sampling sequences...")
    all_sequences = read_fasta(args.fasta_path)
    
    # Clean and truncate sequences
    all_sequences = [seq for seq in all_sequences if len(seq) >= args.sequence_length]
    all_sequences = [seq[:args.sequence_length] for seq in all_sequences]
    
    # Sample a subset for training
    num_train_sequences = int(len(all_sequences) * args.train_fraction)
    training_sequences = random.sample(all_sequences, num_train_sequences)
    print(f"Training on {len(training_sequences)} sequences (a {args.train_fraction*100:.2f}% fraction of the corpus).")

    # Train the specified model
    tokenizer = None
    if args.model_type == 'kmer':
        model = train_kmer_fasttext_model(
            training_sequences, 
            k=args.kmer_length, 
            vector_size=args.vector_size, 
            window=args.window_size, 
            epochs=args.epochs
        )
    else: # bpe
        print("Loading BPE tokenizer...")
        tokenizer = Tokenizer.from_file("hg38_tokenizer.json")
        model = train_bpe_fasttext_model(
            training_sequences, 
            tokenizer, 
            vector_size=args.vector_size, 
            window=args.window_size, 
            epochs=args.epochs
        )

    # --- Data Preparation for Plotting ---
    print(f"Generating {args.num_sequences_plot} real and synthetic sequences for plotting...")
    plot_sequences_real = random.sample(all_sequences, args.num_sequences_plot)
    plot_sequences_fake = [generate_random_dna(args.sequence_length) for _ in range(args.num_sequences_plot)]
    
    all_plot_sequences = plot_sequences_real + plot_sequences_fake
    labels = ['Real'] * args.num_sequences_plot + ['Synthetic'] * args.num_sequences_plot

    # --- Embedding Generation ---
    print("Generating embeddings for all 4 plot types...")
    pe_types_to_plot = ['none', 'sinusoid', 'rope', 'alibi']
    embedding_sets = {}
    for pe_type in pe_types_to_plot:
        print(f"  - Generating embeddings with PE: {pe_type}")
        embedding_sets[pe_type] = create_embeddings_with_pe(
            model, 
            all_plot_sequences, 
            args.model_type, 
            tokenizer=tokenizer, 
            k=args.kmer_length, 
            pe_type=pe_type
        )
    print("All embeddings generated.")


    # --- UMAP and Plotting ---
    print("Creating 2x2 UMAP plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    title_k_part = f", k={args.kmer_length}" if args.model_type == 'kmer' else ""
    fig.suptitle(f'UMAP of Real vs. Synthetic DNA Embeddings ({args.model_type.upper()} model{title_k_part})', fontsize=18)

    plot_titles = {
        'none': 'FastText (No PE)',
        'sinusoid': 'Sinusoidal PE',
        'rope': 'RoPE PE',
        'alibi': 'ALiBi PE'
    }
    color_map = {'Real': '#2E8B57', 'Synthetic': '#DC143C'}
    plot_colors = [color_map[label] for label in labels]
    axes_flat = axes.flatten()

    for i, pe_type in enumerate(pe_types_to_plot):
        ax = axes_flat[i]
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings = embedding_sets[pe_type]
        
        # Check for empty or all-zero embeddings which can cause UMAP errors
        if embeddings.shape[0] == 0 or np.all(embeddings == 0):
            print(f"Skipping UMAP for {pe_type} due to empty or zero embeddings.")
            ax.text(0.5, 0.5, 'No valid data to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(plot_titles[pe_type], fontsize=14)
            continue

        print(f"   - Reducing dimensionality for {plot_titles[pe_type]}...")
        embedding_2d = reducer.fit_transform(embeddings)
        
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=plot_colors, s=15, alpha=0.9)
        ax.set_title(plot_titles[pe_type], fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, linestyle='--', alpha=0.6)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=12) for k, v in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    output_filename = f'plots/umap_real_vs_fake_{args.model_type}_multi_pe.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"UMAP plot saved to {output_filename}")


if __name__ == "__main__":
    main()