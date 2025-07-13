"""UniRef50 pre-training utilities."""

import logging
from pathlib import Path
from typing import List, Iterator, Optional
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from pe_fasttext.fasttext_utils import train_fasttext

logger = logging.getLogger(__name__)


class UniRef50Loader:
    """Load UniRef50 sequences for pre-training."""
    
    def __init__(self, train_split: float = 0.01, max_sequences: Optional[int] = None):
        """Initialize loader.
        
        Args:
            train_split: Fraction of UniRef50 to use (0.01 = 1%)
            max_sequences: Maximum number of sequences to load
        """
        self.train_split = train_split
        self.max_sequences = max_sequences
    
    def load_sequences(self, streaming: bool = True) -> Iterator[str]:
        """Load UniRef50 sequences.
        
        Args:
            streaming: If True, stream sequences to save memory
            
        Yields:
            Protein sequences
        """
        logger.info(f"Loading UniRef50 (split={self.train_split}, max={self.max_sequences})")
        
        # Load dataset
        if streaming:
            # For streaming, we can't use percentage splits directly
            # Instead, we'll sample based on train_split probability
            dataset = load_dataset(
                "agemagician/uniref50",
                split="train",
                streaming=True
            )
            
            # Use a simple sampling strategy
            count = 0
            skip_count = 0
            sample_rate = int(1.0 / self.train_split) if self.train_split < 1.0 else 1
            
            for idx, example in enumerate(dataset):
                if self.max_sequences and count >= self.max_sequences:
                    break
                
                # Sample based on train_split
                if idx % sample_rate == 0:
                    sequence = example.get("sequence", "")
                    if sequence:
                        yield sequence
                        count += 1
                else:
                    skip_count += 1
                    
                # Log progress periodically
                if (count + skip_count) % 10000 == 0:
                    total_processed = count + skip_count
                    logger.info(f"Progress: {count:,} sequences loaded, {skip_count:,} skipped (total: {total_processed:,})")
                    if self.max_sequences:
                        pct = (count / self.max_sequences) * 100
                        logger.info(f"  → {pct:.1f}% of target sequences collected")
        else:
            # For non-streaming, use percentage split
            split_pct = max(1, int(self.train_split * 100))
            dataset = load_dataset(
                "agemagician/uniref50",
                split=f"train[:{split_pct}%]"
            )
            
            count = 0
            for example in dataset:
                if self.max_sequences and count >= self.max_sequences:
                    break
                    
                sequence = example.get("sequence", "")
                if sequence:
                    yield sequence
                    count += 1
        
        logger.info(f"Loaded {count} sequences from UniRef50")


def pretrain_fasttext(
    output_path: Path,
    tokenization: str = "kmer",
    k: int = 6,
    dim: int = 128,
    epochs: int = 5,
    train_split: float = 0.01,
    max_sequences: Optional[int] = None,
    **kwargs
) -> Path:
    """Pre-train FastText on UniRef50.
    
    Args:
        output_path: Where to save the model
        tokenization: "kmer" or "residue"
        k: k-mer size (only for kmer tokenization)
        dim: Embedding dimension
        epochs: Number of training epochs
        train_split: Fraction of UniRef50 to use
        max_sequences: Maximum sequences to use
        **kwargs: Additional FastText parameters
        
    Returns:
        Path to saved model
    """
    if output_path.exists():
        logger.info(f"Pre-trained model already exists at {output_path}")
        return output_path
    
    logger.info("Pre-training FastText on UniRef50...")
    logger.info(f"Config: tokenization={tokenization}, k={k}, dim={dim}, epochs={epochs}")
    
    # Load sequences
    loader = UniRef50Loader(train_split=train_split, max_sequences=max_sequences)
    
    # For large datasets, use streaming tokenization
    if max_sequences and max_sequences > 50000:
        logger.info("Using streaming tokenization for large dataset")
        sequences = loader.load_sequences(streaming=True)
    else:
        sequences = list(loader.load_sequences(streaming=True))
    
    # Tokenize sequences
    if tokenization == "residue":
        logger.info("Using residue-level tokenization")
        if hasattr(sequences, '__iter__') and not isinstance(sequences, list):
            # Streaming sequences
            tokenized = []
            for seq in tqdm(sequences, desc="Tokenizing sequences"):
                tokenized.append(list(seq))
        else:
            tokenized = [list(seq) for seq in tqdm(sequences, desc="Tokenizing")]
    else:
        logger.info(f"Using {k}-mer tokenization")
        tokenized = []
        if hasattr(sequences, '__iter__') and not isinstance(sequences, list):
            # Streaming sequences
            for seq in tqdm(sequences, desc="Tokenizing sequences"):
                tokens = [seq[i:i+k] for i in range(len(seq) - k + 1)]
                if tokens:  # Skip sequences shorter than k
                    tokenized.append(tokens)
        else:
            for seq in tqdm(sequences, desc="Tokenizing"):
                tokens = [seq[i:i+k] for i in range(len(seq) - k + 1)]
                if tokens:  # Skip sequences shorter than k
                    tokenized.append(tokens)
    
    logger.info(f"Training on {len(tokenized)} sequences")
    logger.info(f"Total tokens: {sum(len(seq) for seq in tokenized):,}")
    
    # Default parameters
    params = {
        'vector_size': dim,
        'window': 5,
        'min_count': 5,  # Higher min_count for pre-training
        'sg': 1,  # Skip-gram
        'epochs': epochs,
        'workers': 4
    }
    params.update(kwargs)
    
    # Train model
    model = train_fasttext(corpus_iter=tokenized, **params)
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    logger.info(f"Pre-trained model saved to {output_path}")
    
    return output_path