"""Train FastText model on HuggingFace biological sequence datasets."""

import argparse
import os
from pathlib import Path
from typing import Iterator, List

from datasets import load_dataset

from pe_fasttext.config import TrainingConfig, DatasetConfig
from pe_fasttext.fasttext_utils import train_fasttext
from pe_fasttext.utils import kmerize


class HuggingFaceCorpus:
    """Corpus iterator for HuggingFace datasets."""
    
    def __init__(self, dataset_config: DatasetConfig, training_config: TrainingConfig):
        self.dataset_config = dataset_config
        self.training_config = training_config

    def __iter__(self) -> Iterator[List[str]]:
        """Yield k-merized sequences from the dataset."""
        ds = load_dataset(
            self.dataset_config.name,
            split=self.dataset_config.split,
            streaming=self.dataset_config.streaming
        )
        
        count = 0
        for row in ds:
            sequences = row.get(self.dataset_config.field, [])
            if not isinstance(sequences, list):
                sequences = [sequences]
                
            for seq in sequences:
                if self.training_config.uppercase:
                    seq = seq.upper()
                    
                kmers = kmerize(seq, self.training_config.k)
                if kmers:
                    yield kmers
                    count += 1
                    if self.training_config.max_sequences and count >= self.training_config.max_sequences:
                        return

def train_model(dataset_config: DatasetConfig, training_config: TrainingConfig, output_path: Path):
    """Train FastText model with given configuration."""
    print(f"Training FastText on {dataset_config.name} ({dataset_config.field})")
    print(f"k={training_config.k}, max_sequences={training_config.max_sequences}")
    print(f"vector_size={training_config.vector_size}, epochs={training_config.epochs}")
    
    # Create corpus iterator
    corpus = HuggingFaceCorpus(dataset_config, training_config)
    
    # Train model
    model = train_fasttext(
        corpus_iter=corpus,
        vector_size=training_config.vector_size,
        window=training_config.window,
        min_count=training_config.min_count,
        epochs=training_config.epochs,
        sg=training_config.sg,
        workers=training_config.workers
    )
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Model saved: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train FastText on HuggingFace datasets")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="tattabio/OG", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--field", help="Field containing sequences (auto-detected if not specified)")
    
    # Training arguments
    parser.add_argument("--k", type=int, default=5, help="k-mer size")
    parser.add_argument("--vector-size", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum k-mer count")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--sg", type=int, default=1, choices=[0, 1], help="1=skipgram, 0=CBOW")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument("--max-sequences", type=int, help="Maximum sequences to use")
    parser.add_argument("--output", help="Output model path")
    
    args = parser.parse_args()
    
    # Create configurations
    dataset_config = DatasetConfig(
        name=args.dataset,
        split=args.split,
        field=args.field if args.field else "sequence"
    )
    
    training_config = TrainingConfig(
        k=args.k,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        sg=args.sg,
        workers=args.workers,
        max_sequences=args.max_sequences
    )
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        dataset_name_safe = args.dataset.replace('/', '_')
        output_path = Path(f"models/fasttext_{dataset_name_safe}_k{args.k}.bin")
    
    # Train model
    train_model(dataset_config, training_config, output_path)

if __name__ == "__main__":
    main() 