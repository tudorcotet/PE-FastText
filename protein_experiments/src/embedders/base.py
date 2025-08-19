"""Base embedder interface for protein sequence embeddings."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedder with configuration.
        
        Args:
            config: Embedder-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def embed(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for sequences.
        
        Args:
            sequences: List of protein sequences
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings, shape (n_sequences, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass
    
    def __repr__(self) -> str:
        """String representation of embedder."""
        return f"{self.__class__.__name__}(config={self.config})"