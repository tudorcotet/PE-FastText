"""Embedder implementations for protein sequences."""

from typing import Dict, Any
from .base import BaseEmbedder
from .fasttext import FastTextEmbedder

# Make ESM2 optional
try:
    from .esm2 import ESM2Embedder
    HAS_ESM2 = True
except ImportError:
    HAS_ESM2 = False


def create_embedder(config: Dict[str, Any]) -> BaseEmbedder:
    """Factory function to create embedders.
    
    Args:
        config: Embedder configuration with 'type' key
        
    Returns:
        Embedder instance
    """
    embedder_type = config.get('type', 'fasttext')
    
    if embedder_type == 'fasttext':
        return FastTextEmbedder(config)
    elif embedder_type == 'esm2':
        if not HAS_ESM2:
            raise ImportError("ESM2 not available. Install torch and transformers.")
        return ESM2Embedder(config)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


__all__ = ['BaseEmbedder', 'FastTextEmbedder', 'create_embedder']
if HAS_ESM2:
    __all__.append('ESM2Embedder')