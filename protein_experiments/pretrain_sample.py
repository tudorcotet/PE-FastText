#!/usr/bin/env python
"""Pre-train FastText on a sample of protein sequences."""

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from src.pretrain import pretrain_fasttext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_sample_sequences(dataset_name: str = "tattabio/OG_prot", max_sequences: int = 10000):
    """Load sample protein sequences from easier-to-access datasets."""
    
    logging.info(f"Loading sample sequences from {dataset_name}")
    
    if dataset_name == "tattabio/OG_prot":
        # This is a smaller, easier to download protein dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        sequences = []
        
        for i, example in enumerate(tqdm(dataset, desc="Loading sequences", total=max_sequences)):
            if i >= max_sequences:
                break
            sequence = example.get("sequence", "")
            if sequence and len(sequence) > 10:  # Filter very short sequences
                sequences.append(sequence)
        
    elif dataset_name == "InstaDeepAI/true-cds-protein-tasks":
        # Use the fluorescence task sequences as pre-training data
        dataset = load_dataset(dataset_name, name="fluorescence", split="train")
        sequences = dataset["sequence"][:max_sequences]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logging.info(f"Loaded {len(sequences)} sequences")
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Pre-train FastText on sample protein sequences')
    parser.add_argument('--output', type=Path, default=Path('models/sample_pretrained.bin'),
                       help='Output path for pre-trained model')
    parser.add_argument('--dataset', default='tattabio/OG_prot',
                       choices=['tattabio/OG_prot', 'InstaDeepAI/true-cds-protein-tasks'],
                       help='Dataset to use for pre-training')
    parser.add_argument('--tokenization', choices=['kmer', 'residue'], default='kmer',
                       help='Tokenization method')
    parser.add_argument('--k', type=int, default=6,
                       help='k-mer size (only for kmer tokenization)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--max-sequences', type=int, default=10000,
                       help='Maximum number of sequences')
    parser.add_argument('--window', type=int, default=5,
                       help='Context window size')
    parser.add_argument('--min-count', type=int, default=5,
                       help='Minimum word count')
    
    args = parser.parse_args()
    
    # Load sample sequences
    sequences = load_sample_sequences(args.dataset, args.max_sequences)
    
    # Tokenize
    if args.tokenization == "residue":
        logging.info("Using residue-level tokenization")
        tokenized = [list(seq) for seq in tqdm(sequences, desc="Tokenizing")]
    else:
        logging.info(f"Using {args.k}-mer tokenization")
        tokenized = []
        for seq in tqdm(sequences, desc="Tokenizing"):
            tokens = [seq[i:i+args.k] for i in range(len(seq) - args.k + 1)]
            if tokens:
                tokenized.append(tokens)
    
    logging.info(f"Training on {len(tokenized)} sequences")
    logging.info(f"Total tokens: {sum(len(seq) for seq in tokenized):,}")
    
    # Train using the same function but bypass the loader
    from pe_fasttext.fasttext_utils import train_fasttext
    
    model = train_fasttext(
        corpus_iter=tokenized,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        sg=1,  # Skip-gram
        epochs=args.epochs,
        workers=4
    )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    logging.info(f"Pre-trained model saved to {args.output}")


if __name__ == '__main__':
    main()