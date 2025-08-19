import os
from gensim.models import FastText
from tokenizers import Tokenizer
import argparse

# --- Paths ---
FASTA_PATH = "./data/GCA_000001405.15_GRCh38_genomic.fna"
KMER_MODEL_OUT = "./models/fasttext_kmer_hg38.bin"
BPE_MODEL_OUT = "./models/fasttext_bpe_hg38.bin"
TOKENIZER_PATH = "hg38_tokenizer.json"

# --- Default / Hard-coded Training Parameters ---
MIN_COUNT = 5
WORKERS = -1  # Use all available cores
SG = 1  # Skip-gram

# --- Modifyable Training Parameters ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train FastText models on FASTA using k-mers and BPE tokens.")
    # Paths
    parser.add_argument("--fasta-path", default=FASTA_PATH, help="Path to FASTA file")
    parser.add_argument("--kmer-model-out", default=KMER_MODEL_OUT, help="Output path for k-mer FastText model")
    parser.add_argument("--bpe-model-out", default=BPE_MODEL_OUT, help="Output path for BPE FastText model")
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH, help="Path to BPE tokenizer JSON")

    # Training params
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--k", type=int, default=6, help="k-mer size")
    parser.add_argument("--epochs_kmer", type=int, default=5)
    parser.add_argument("--epochs_bpe", type=int, default=15)
    parser.add_argument("--kmer-chunk-size", type=int, default=50000, help="Chunk size when forming k-mer sentences")
    parser.add_argument("--bpe-chunk-size", type=int, default=50000, help="Chunk size for BPE tokenization")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length per FASTA record (truncate if longer). Default: no limit")
    return parser.parse_args()


def kmer_sentence_stream(fasta_path, k, chunk_size=5000, max_seq_len=None):
    """Yields sentences of k-mers from a FASTA file."""
    with open(fasta_path, "r") as fh:
        seq = []
        for line in fh:
            line = line.strip().upper()
            if line.startswith(">"):
                if seq:
                    full_seq = "".join(seq)
                    if max_seq_len:
                        full_seq = full_seq[:max_seq_len]
                    for i in range(0, len(full_seq), chunk_size):
                        chunk = full_seq[i:i+chunk_size]
                        if len(chunk) >= k:
                            yield [chunk[j:j+k] for j in range(len(chunk) - k + 1)]
                    seq = []
            else:
                seq.append(line)
        if seq:
            full_seq = "".join(seq)
            if max_seq_len:
                full_seq = full_seq[:max_seq_len]
            for i in range(0, len(full_seq), chunk_size):
                chunk = full_seq[i:i+chunk_size]
                if len(chunk) >= k:
                    yield [chunk[j:j+k] for j in range(len(chunk) - k + 1)]


# Re-iterable wrapper for gensim (so multiple epochs/passes work)
class KmerCorpus:
    def __init__(self, fasta_path: str, k: int, chunk_size: int = 5000, max_seq_len: int | None = None):
        self.fasta_path = fasta_path
        self.k = k
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len

    def __iter__(self):
        return kmer_sentence_stream(self.fasta_path, self.k, chunk_size=self.chunk_size, max_seq_len=self.max_seq_len)


# --- BPE Training ---

def bpe_sentence_stream(fasta_path, tokenizer, chunk_size=5000, max_seq_len=None):
    """Yield BPE-tokenized sentences from FASTA."""
    with open(fasta_path, "r") as fh:
        seq = []
        for line in fh:
            line = line.strip().upper()
            if line.startswith(">"):
                if seq:
                    full_seq = "".join(seq)
                    if max_seq_len:
                        full_seq = full_seq[:max_seq_len]
                    for i in range(0, len(full_seq), chunk_size):
                        chunk = full_seq[i:i+chunk_size]
                        if len(chunk) > 10:
                            tokens = tokenizer.encode(chunk, add_special_tokens=False).tokens
                            if tokens:
                                yield tokens
                    seq = []
            else:
                seq.append(line)
        if seq:
            full_seq = "".join(seq)
            if max_seq_len:
                full_seq = full_seq[:max_seq_len]
            for i in range(0, len(full_seq), chunk_size):
                chunk = full_seq[i:i+chunk_size]
                if len(chunk) > 10:
                    tokens = tokenizer.encode(chunk, add_special_tokens=False).tokens
                    if tokens:
                        yield tokens


class BpeCorpus:
    def __init__(self, fasta_path: str, tokenizer_path: str, chunk_size: int = 5000, max_seq_len: int | None = None):
        self.fasta_path = fasta_path
        self.tokenizer_path = tokenizer_path
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        # Load once for efficiency; __iter__ will reuse it
        self._tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def __iter__(self):
        return bpe_sentence_stream(self.fasta_path, self._tokenizer, chunk_size=self.chunk_size, max_seq_len=self.max_seq_len)


def train_fasttext(sentences, vector_size, window, min_count, epochs, sg, workers):
    """Trains a FastText model. Ensures re-iterable corpus and valid workers count."""
    # Normalize workers: gensim expects a positive int
    if workers is None or workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    return FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=sg,
        workers=workers,
    )

def main():
    """Main function to train both models."""
    args = parse_args()

    # Ensure output directory exists
    for out_path in [args.kmer_model_out, args.bpe_model_out]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # --- Train k-mer Model ---
    print("--- Training k-mer FastText Model ---")
    print(f"Parameters: k={args.k}, vector_size={args.vector_size}, window={args.window}, epochs={args.epochs_kmer}")
    kmer_sentences = KmerCorpus(args.fasta_path, args.k, args.kmer_chunk_size, max_seq_len=args.max_seq_len)
    kmer_model = train_fasttext(
        kmer_sentences,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs_kmer,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
    )
    print(f"Saving k-mer model to {args.kmer_model_out}...")
    kmer_model.save(args.kmer_model_out)
    print("k-mer model training complete.")

    # --- Train BPE Model ---

    print("--- Training BPE FastText Model ---")
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
        print("Please run 1_train_BPE_tokenizer.py first.")
        return

    print(f"Parameters: vector_size={args.vector_size}, window={args.window}, epochs={args.epochs_bpe}")
    bpe_sentences = BpeCorpus(args.fasta_path, args.tokenizer_path, args.bpe_chunk_size, max_seq_len=args.max_seq_len)
    bpe_model = train_fasttext(
        bpe_sentences,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs_bpe,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS
    )
    print(f"Saving BPE model to {args.bpe_model_out}...")
    bpe_model.save(args.bpe_model_out)
    print("BPE model training complete.")


if __name__ == "__main__":
    main()
