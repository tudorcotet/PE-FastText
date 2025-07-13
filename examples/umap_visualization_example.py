"""Example of using the unified UMAP visualization module."""

from gensim.models import Word2Vec, FastText
from datasets import load_dataset
from pathlib import Path

from pe_fasttext.visualization import UMAPVisualizer
from pe_fasttext.utils import kmerize


def generate_og_sequences(k=5, max_sequences=100):
    """Generate sequences from OG dataset."""
    ds = load_dataset('tattabio/OG', streaming=True)['train']
    sequences = []
    labels = []
    
    for i, row in enumerate(ds):
        for seq in row['IGS_seqs']:
            if len(seq) >= k:
                sequences.append(seq)
                # Use sequence index as label for coloring
                labels.append(f"seq_{i % 5}")  # Group into 5 categories
                
                if len(sequences) >= max_sequences:
                    return sequences, labels
    
    return sequences, labels


def main():
    """Compare Word2Vec and FastText embeddings using UMAP."""
    
    # Parameters
    k = 5
    max_sequences = 100
    vector_size = 64
    
    # Generate sequences
    print("Generating sequences...")
    sequences, labels = generate_og_sequences(k, max_sequences)
    
    # Prepare training data (k-merized sequences)
    sentences = [kmerize(seq, k) for seq in sequences]
    
    # Train models
    print("Training Word2Vec...")
    w2v = Word2Vec(sentences, vector_size=vector_size, window=5, 
                   min_count=1, epochs=10, workers=1)
    
    print("Training FastText...")
    ft = FastText(sentences, vector_size=vector_size, window=5,
                  min_count=1, epochs=10, workers=1)
    
    # Create visualizer
    models = {"Word2Vec": w2v, "FastText": ft}
    visualizer = UMAPVisualizer(models, k=k)
    
    # Run benchmark with custom generator
    def sequence_generator():
        return sequences, labels
    
    visualizer.run_benchmark(
        sequence_generator,
        output_path=Path("plots/embeddings_comparison.png"),
        title="Word2Vec vs FastText"
    )


if __name__ == "__main__":
    main()