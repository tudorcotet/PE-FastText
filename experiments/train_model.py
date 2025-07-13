"""Train FastText model from FASTA file with configurable parameters."""

import argparse
import os
from pathlib import Path

from pe_fasttext.config import TrainingConfig
from pe_fasttext.tokenization import fasta_stream
from pe_fasttext.fasttext_utils import train_fasttext

def corpus_iter(corpus_path: Path, config: TrainingConfig):
    """Yield k-mers from FASTA file."""
    print(f"Reading sequences from {corpus_path}...")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Could not find corpus file: {corpus_path}")
    
    # Use utils.CorpusIterator for consistency
    from pe_fasttext.utils import CorpusIterator
    return CorpusIterator(str(corpus_path), config.k, config.uppercase)

def main():
    parser = argparse.ArgumentParser(description="Train FastText on FASTA file")
    
    # Input/output arguments
    parser.add_argument("--fasta", default="data/uniref50.fasta", help="Input FASTA file")
    parser.add_argument("--output", default="fasttext_protein.bin", help="Output model path")
    
    # Training arguments
    parser.add_argument("--k", type=int, default=5, help="k-mer size")
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum k-mer count")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument("--sg", type=int, default=1, choices=[0, 1], help="1=skipgram, 0=CBOW")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        k=args.k,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        sg=args.sg,
        workers=args.workers
    )
    
    corpus_path = Path(args.fasta)
    output_path = Path(args.output)
    
    print("Starting FastText training...")
    print(f"Parameters: k={config.k}, dim={config.vector_size}, workers={config.workers}, epochs={config.epochs}")
    
    try:
        model = train_fasttext(
            corpus_iter(corpus_path, config),
            vector_size=config.vector_size,
            window=config.window,
            min_count=config.min_count,
            epochs=config.epochs,
            sg=config.sg,
            workers=config.workers
        )
        
        print(f"Training complete. Saving model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        
        if output_path.exists():
            print(f"Successfully saved model to {output_path}")
            print(f"Model file size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("Warning: Model file was not created!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 