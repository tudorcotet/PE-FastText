#!/usr/bin/env python
"""Pre-train FastText on UniRef50."""

import argparse
import logging
from pathlib import Path

from src.pretrain import pretrain_fasttext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Pre-train FastText on UniRef50')
    parser.add_argument('--output', type=Path, default=Path('models/uniref50_pretrained_kmer5.bin'),
                       help='Output path for pre-trained model')
    parser.add_argument('--tokenization', choices=['kmer', 'residue'], default='kmer',
                       help='Tokenization method')
    parser.add_argument('--k', type=int, default=5,
                       help='k-mer size (only for kmer tokenization)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--train-split', type=float, default=0.01,
                       help='Fraction of UniRef50 to use (0.01 = 1%)')
    parser.add_argument('--max-sequences', type=int,
                       help='Maximum number of sequences')
    parser.add_argument('--window', type=int, default=5,
                       help='Context window size')
    parser.add_argument('--min-count', type=int, default=5,
                       help='Minimum word count')
    
    args = parser.parse_args()
    
    # Pre-train
    pretrain_fasttext(
        output_path=args.output,
        tokenization=args.tokenization,
        k=args.k,
        dim=args.dim,
        epochs=args.epochs,
        train_split=args.train_split,
        max_sequences=args.max_sequences,
        window=args.window,
        min_count=args.min_count
    )


if __name__ == '__main__':
    main()