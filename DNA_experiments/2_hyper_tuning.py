
import argparse
import random
import numpy as np
from gensim.models import FastText
from tokenizers import Tokenizer
from scipy.spatial.distance import cosine

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

def train_kmer_fasttext_model(sequences, k, vector_size=100, window=5, min_count=1, epochs=5):
    """Trains a new FastText model on k-mers from the given sequences."""
    print(f"Training k-mer FastText model with k={k}, window={window}...")
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

def create_kmer_embeddings(model, sequences, k):
    """Creates embeddings for sequences using a k-mer model."""
    d_model = model.vector_size
    all_embeddings = []
    for seq in sequences:
        seq = ''.join(c for c in seq if c in 'ATGC')
        if len(seq) < k:
            all_embeddings.append(np.zeros(d_model))
            continue
        
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        if not kmers:
            all_embeddings.append(np.zeros(d_model))
            continue

        valid_kmers = [kmer for kmer in kmers if kmer in model.wv]
        if not valid_kmers:
            all_embeddings.append(np.zeros(d_model))
            continue

        kmer_embeddings = np.array([model.wv[kmer] for kmer in valid_kmers])
        all_embeddings.append(np.mean(kmer_embeddings, axis=0))
    return np.array(all_embeddings)

def train_bpe_fasttext_model(sequences, tokenizer, vector_size=100, window=5, min_count=1, epochs=5):
    """Trains a new FastText model on BPE tokens from the given sequences."""
    print(f"Training BPE FastText model with window={window}...")
    
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

def create_bpe_embeddings(model, sequences, tokenizer):
    """Creates embeddings for sequences using a BPE model."""
    d_model = model.vector_size
    all_embeddings = []
    for seq in sequences:
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
        all_embeddings.append(np.mean(token_embeddings, axis=0))
    return np.array(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for k-mer and BPE models.")
    parser.add_argument("--fasta_path", type=str, default="./data/GCA_000001405.15_GRCh38_genomic.fna", help="Path to the genomic sequence corpus.")
    parser.add_argument("--num_sequences_eval", type=int, default=100, help="Number of sequences to use for evaluation.")
    parser.add_argument("--sequence_length", type=int, default=1000, help="Length of sequences for training and evaluation.")
    parser.add_argument("--vector_size", type=int, default=128, help="Size of the vectors.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    args = parser.parse_args()

    print("Loading and sampling sequences...")
    all_sequences = read_fasta(args.fasta_path)
    # Sample a subset for training to speed up the process
    training_sequences = all_sequences
    # clean sequences
    training_sequences = [seq for seq in training_sequences if len(seq) >= args.sequence_length]
    training_sequences = [seq[:args.sequence_length] for seq in training_sequences]


    # --- Hyperparameter Tuning for k-mer Model ---
    print("\n--- Starting Hyperparameter Tuning for k-mer Model ---")
    kmer_results = {}
    window_sizes_kmer = [5, 10, 20]
    kmer_dims = [3, 4, 5, 6, 7]
    epochs = [5, 10, 15]


    for window in window_sizes_kmer:
        for k in kmer_dims:
            for epoch in epochs:
                model = train_kmer_fasttext_model(training_sequences, k, vector_size=args.vector_size, window=window, epochs=epoch)

                print("Evaluating k-mer model...")
                eval_sequences_real = random.sample(all_sequences, args.num_sequences_eval)
                eval_sequences_real = [seq for seq in eval_sequences_real if len(seq) >= args.sequence_length]
                eval_sequences_real = [seq[:args.sequence_length] for seq in eval_sequences_real]
                
                eval_sequences_fake = [generate_random_dna(args.sequence_length) for _ in range(args.num_sequences_eval)]

                real_embeddings = create_kmer_embeddings(model, eval_sequences_real, k)
                fake_embeddings = create_kmer_embeddings(model, eval_sequences_fake, k)

                mean_real_embedding = np.mean(real_embeddings, axis=0)
                mean_fake_embedding = np.mean(fake_embeddings, axis=0)

                dist = cosine(mean_real_embedding, mean_fake_embedding)
                kmer_results[(window, k, epoch)] = dist
                print(f"Cosine distance for window={window}, k={k}, epoch={epoch}: {dist}")

    print("\n--- k-mer Model Hyperparameter Tuning Results ---")
    for (window, k, epoch), dist in kmer_results.items():
        print(f"Window: {window}, k-mer dim: {k}, Epochs: {epoch}, Cosine Distance: {dist}")

    # --- Hyperparameter Tuning for BPE Model ---
    print("\n--- Starting Hyperparameter Tuning for BPE Model ---")
    bpe_results = {}
    window_sizes_bpe = [5, 10, 20]

    print("Loading BPE tokenizer...")
    tokenizer = Tokenizer.from_file("hg38_tokenizer.json")

    for window in window_sizes_bpe:
        for epoch in epochs:
            print(f"\nTraining BPE model with window_size={window}, epochs={epoch}...")
            model = train_bpe_fasttext_model(training_sequences, tokenizer, vector_size=args.vector_size, window=window, epochs=epoch)

            print("Evaluating BPE model...")
            eval_sequences_real = random.sample(all_sequences, args.num_sequences_eval)
            eval_sequences_real = [seq for seq in eval_sequences_real if len(seq) >= args.sequence_length]
            eval_sequences_real = [seq[:args.sequence_length] for seq in eval_sequences_real]

            eval_sequences_fake = [generate_random_dna(args.sequence_length) for _ in range(args.num_sequences_eval)]

            real_embeddings = create_bpe_embeddings(model, eval_sequences_real, tokenizer)
            fake_embeddings = create_bpe_embeddings(model, eval_sequences_fake, tokenizer)

            mean_real_embedding = np.mean(real_embeddings, axis=0)
            mean_fake_embedding = np.mean(fake_embeddings, axis=0)

            dist = cosine(mean_real_embedding, mean_fake_embedding)
            bpe_results[(window, epoch)] = dist
            print(f"Cosine distance for window={window}, epoch={epoch}: {dist}")

    print("\n--- BPE Model Hyperparameter Tuning Results ---")
    for (window, epoch), dist in bpe_results.items():
        print(f"Window: {window}, Epochs: {epoch}, Cosine Distance: {dist}")

if __name__ == "__main__":
    main()
