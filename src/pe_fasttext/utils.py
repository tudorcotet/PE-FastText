"""Shared utilities for PE-FastText to avoid code duplication."""

from typing import List, Union, Optional, Callable
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import gzip
import shutil


def kmerize(seq: str, k: int) -> List[str]:
    """Extract k-mers from a sequence.
    
    Args:
        seq: Input sequence string
        k: k-mer size
        
    Returns:
        List of k-mers
    """
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def embed_sequences(
    model,
    sequences: List[str],
    k: int,
    aggregation: str = 'mean',
    progress_callback: Optional[Callable] = None
) -> np.ndarray:
    """Embed sequences using a trained model.
    
    Args:
        model: Trained FastText or PEFastText model
        sequences: List of sequences to embed
        k: k-mer size
        aggregation: How to aggregate k-mer embeddings ('mean' or 'sum')
        progress_callback: Optional callback for progress updates
        
    Returns:
        Array of embeddings, one per sequence
    """
    embeddings = []
    
    for i, seq in enumerate(sequences):
        kmers = kmerize(seq, k)
        
        if hasattr(model, 'embed_sequence'):
            # PEFastText model
            embedding = model.embed_sequence(seq)
        else:
            # Regular FastText model
            kmer_embeddings = []
            for kmer in kmers:
                if kmer in model.wv:
                    kmer_embeddings.append(model.wv[kmer])
            
            if kmer_embeddings:
                kmer_embeddings = np.array(kmer_embeddings)
                if aggregation == 'mean':
                    embedding = np.mean(kmer_embeddings, axis=0)
                elif aggregation == 'sum':
                    embedding = np.sum(kmer_embeddings, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
            else:
                # Return zero vector if no k-mers found
                embedding = np.zeros(model.wv.vector_size)
        
        embeddings.append(embedding)
        
        if progress_callback:
            progress_callback(i + 1, len(sequences))
    
    return np.array(embeddings)


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size
        
    Returns:
        Path to downloaded file
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path


def decompress_file(compressed_path: Path, output_path: Optional[Path] = None) -> Path:
    """Decompress a gzipped file.
    
    Args:
        compressed_path: Path to compressed file
        output_path: Optional output path (defaults to removing .gz extension)
        
    Returns:
        Path to decompressed file
    """
    if output_path is None:
        output_path = compressed_path.with_suffix('')
    
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path


class CorpusIterator:
    """Memory-efficient corpus iterator for large FASTA files."""
    
    def __init__(self, fasta_path: str, k: int, uppercase: bool = True):
        """Initialize corpus iterator.
        
        Args:
            fasta_path: Path to FASTA file
            k: k-mer size
            uppercase: Whether to convert sequences to uppercase
        """
        self.fasta_path = fasta_path
        self.k = k
        self.uppercase = uppercase
    
    def __iter__(self):
        """Iterate over k-merized sequences."""
        from .tokenization import fasta_stream
        
        for seq in fasta_stream(self.fasta_path):
            if self.uppercase:
                seq = seq.upper()
            kmers = kmerize(seq, self.k)
            if kmers:
                yield kmers