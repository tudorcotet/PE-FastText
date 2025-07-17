import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
import umap
from fpdf import FPDF
import sys

# Add src to path to allow for local imports
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent / 'src'))
from pe_fasttext.utils import kmerize
from pe_fasttext.model import PEFastText
from plotting import plot_umap_comparison

# --- PDF Report Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sanity Checks Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_plot(self, image_path, title, interpretation):
        if not Path(image_path).exists():
            print(f"Warning: Plot not found at {image_path}, skipping.")
            return
        self.add_page()
        self.chapter_title(title)
        self.image(image_path, x=10, y=30, w=190)
        self.set_y(30 + 125)
        self.chapter_title("Interpretation")
        self.chapter_body(interpretation)

# --- Data Loading and Model Training ---
def load_sequences(dataset_name, split, sequence_field, max_sequences):
    """Load sequences from Hugging Face datasets."""
    print(f"Loading {max_sequences} sequences from {dataset_name}...")
    ds = load_dataset(dataset_name, split=split, streaming=True)
    sequences = []
    for row in tqdm(ds.take(max_sequences), total=max_sequences):
        sequences.append(row[sequence_field])
    return sequences

def train_model(sequences, k, vector_size, window, epochs):
    """Train a new FastText model."""
    print(f"Training FastText on {len(sequences)} sequences...")
    tokenized = [kmerize(s, k) for s in sequences]
    model = FastText(sentences=tokenized, vector_size=vector_size, window=window, min_count=1, epochs=epochs, sg=1, workers=4)
    return model

# --- Embedding and Benchmarking ---
def get_embeddings(model, sequences, k, pe_type='baseline', fusion='add'):
    """Generate embeddings for a list of sequences."""
    if pe_type == 'baseline':
        tokenized = [kmerize(s, k) for s in sequences]
        vectors = []
        for sent in tokenized:
            vecs = [model.wv[kmer] for kmer in sent if kmer in model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(model.vector_size))
        return np.vstack(vectors)
    else:
        pe_model = PEFastText(fasttext_model=model, pos_encoder=pe_type, fusion=fusion)
        return pe_model.embed(sequences, k=k, average_sequences=True)

def run_benchmark(model, benchmark_fn, benchmark_name, k, output_dir):
    """Run a UMAP benchmark for a given model and sequence generation function."""
    print(f"\n--- Running benchmark: {benchmark_name} ---")
    sequences, labels = benchmark_fn()
    
    pos_encoders = ['baseline', 'sinusoid', 'learned', 'rope', 'alibi', 'ft_alibi']
    embeds_dict = {}

    for pe in pos_encoders:
        print(f"Generating embeddings for {pe}...")
        embeds = get_embeddings(model, sequences, k, pe_type=pe)
        embeds_std = StandardScaler().fit_transform(embeds)
        embeds_dict[pe] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeds_std)
        
    plot_umap_comparison(
        embeds_dict,
        labels,
        output_dir / f"{benchmark_name.lower().replace(' ', '_')}.png",
        main_title=benchmark_name
    )

# --- Benchmark Definitions ---
def get_real_vs_random(n_seqs=100, k=8):
    """Benchmark: Real biological sequences vs. random sequences."""
    real_seqs = load_sequences('tattabio/OG_DNADataset', 'train', 'seq', n_seqs)
    random_seqs = ["".join(np.random.choice(list("ACGT"), len(s))) for s in real_seqs]
    labels = ['real'] * len(real_seqs) + ['random'] * len(random_seqs)
    return real_seqs + random_seqs, labels

def get_positional_motifs(n_per_class=20, seq_len=100):
    """Benchmark: Same motif at different positions."""
    motif = "GATTACA"
    sequences, labels = [], []
    positions = {'start': 0, 'middle': (seq_len - len(motif)) // 2, 'end': seq_len - len(motif)}
    for label, pos in positions.items():
        for _ in range(n_per_class):
            prefix = "".join(np.random.choice(list("ACGT"), pos))
            suffix = "".join(np.random.choice(list("ACGT"), seq_len - len(prefix) - len(motif)))
            sequences.append(prefix + motif + suffix)
            labels.append(f"{motif}_{label}")
    return sequences, labels

def get_variable_spacing(n_per_class=20, seq_len=100):
    """Benchmark: Two motifs with variable spacing."""
    motif1, motif2 = "TATAAA", "GATTACA"
    sequences, labels = [], []
    spacings = [5, 20, 50]
    for spacing in spacings:
        for _ in range(n_per_class):
            prefix_len = np.random.randint(5, 15)
            prefix = "".join(np.random.choice(list("ACGT"), prefix_len))
            middle = "".join(np.random.choice(list("ACGT"), spacing))
            total_len = len(prefix) + len(motif1) + len(middle) + len(motif2)
            suffix = "".join(np.random.choice(list("ACGT"), seq_len - total_len))
            sequences.append(prefix + motif1 + middle + motif2 + suffix)
            labels.append(f"spacing_{spacing}")
    return sequences, labels

# --- Main Execution ---
def main(args):
    FIGURES_DIR = Path("sanity_check/figures")
    FIGURES_DIR.mkdir(exist_ok=True)

    if args.model_path and Path(args.model_path).exists():
        print(f"Loading pre-trained model from {args.model_path}")
        model = FastText.load(args.model_path)
    else:
        train_seqs = load_sequences('tattabio/OG', 'train', 'IGS_seqs', 1000)
        flat_train_seqs = [s for sublist in train_seqs if sublist for s in sublist]
        model = train_model(flat_train_seqs, args.k, args.vector_size, args.window, args.epochs)

    benchmarks = [
        (get_real_vs_random, "Real vs. Random Sequences"),
        (get_positional_motifs, "Positional Motif Detection"),
        (get_variable_spacing, "Variable Motif Spacing"),
    ]
    for fn, name in benchmarks:
        run_benchmark(model, fn, name, args.k, FIGURES_DIR)

    # Generate PDF report
    pdf = PDF()
    plots_for_report = [
        (FIGURES_DIR / "real_vs._random_sequences.png", "Real vs. random sequences", "This test evaluates if the model can distinguish real DNA from random noise. A good model should form distinct clusters for the two categories."),
        (FIGURES_DIR / "positional_motif_detection.png", "Positional motif detection", "This tests if positional encoders can differentiate the same motif based on its location (start, middle, end). Models with effective PEs should show clear separation between these groups."),
        (FIGURES_DIR / "variable_motif_spacing.png", "Variable motif spacing", "This tests if models can understand the relative positioning of two motifs. Models with effective PEs should create distinct clusters based on the spacing between the motifs."),
    ]
    for path, title, interpretation in plots_for_report:
        pdf.add_plot(str(path), title, interpretation)
    pdf.output("sanity_checks_report.pdf", 'F')
    print("\nSanity check report saved to 'sanity_checks_report.pdf'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive sanity checks on embedding models.")
    parser.add_argument('--model_path', type=str, default=None, help='Optional path to a pre-trained FastText model.')
    parser.add_argument('--k', type=int, default=8, help='k-mer size for tokenization.')
    parser.add_argument('--vector_size', type=int, default=64, help='Embedding dimension for training.')
    parser.add_argument('--window', type=int, default=10, help='Context window size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    args = parser.parse_args()
    main(args) 