"""Clean example of comparing embeddings with UMAP visualization.

This example demonstrates best practices:
- Using configuration objects
- No hardcoded values
- Reusing shared utilities
- Clean separation of concerns
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from gensim.models import Word2Vec, FastText

from pe_fasttext.config import TrainingConfig
from pe_fasttext.visualization import UMAPVisualizer
from pe_fasttext.utils import kmerize


def load_sequences(dataset_name: str, max_sequences: int, k: int):
    """Load sequences from HuggingFace dataset."""
    ds = load_dataset(dataset_name, streaming=True)['train']
    sequences = []
    labels = []
    
    for i, row in enumerate(ds):
        for seq in row.get('IGS_seqs', []):
            if len(seq) >= k:
                sequences.append(seq)
                labels.append(f"batch_{i % 5}")  # Group into 5 batches
                
                if len(sequences) >= max_sequences:
                    return sequences, labels
    
    return sequences, labels


def main():
    parser = argparse.ArgumentParser(description="Compare Word2Vec and FastText embeddings")
    
    # Data arguments
    parser.add_argument("--dataset", default="tattabio/OG", help="HuggingFace dataset")
    parser.add_argument("--max-sequences", type=int, default=100, help="Maximum sequences to use")
    
    # Model arguments
    parser.add_argument("--k", type=int, default=5, help="k-mer size")
    parser.add_argument("--vector-size", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    
    # Output arguments
    parser.add_argument("--output", default="plots/embedding_comparison.png", help="Output path")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading sequences from {args.dataset}...")
    sequences, labels = load_sequences(args.dataset, args.max_sequences, args.k)
    print(f"Loaded {len(sequences)} sequences")
    
    # Prepare training data
    sentences = [kmerize(seq, args.k) for seq in sequences]
    
    # Create training config
    config = TrainingConfig(
        k=args.k,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs,
        workers=1  # Single worker for reproducibility
    )
    
    # Train models
    print("Training Word2Vec...")
    w2v = Word2Vec(
        sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        epochs=config.epochs,
        workers=config.workers
    )
    
    print("Training FastText...")
    ft = FastText(
        sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        epochs=config.epochs,
        workers=config.workers
    )
    
    # Create visualizer and run comparison
    models = {"Word2Vec": w2v, "FastText": ft}
    visualizer = UMAPVisualizer(models, k=config.k)
    
    # Run visualization
    visualizer.run_benchmark(
        lambda: (sequences, labels),
        output_path=Path(args.output),
        title="Embedding Comparison"
    )
    
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()