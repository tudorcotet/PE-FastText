"""UMAP visualization utilities for PE-FastText embeddings."""

from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from pathlib import Path

from .utils import embed_sequences


class UMAPVisualizer:
    """Unified UMAP visualization for sequence embeddings."""
    
    def __init__(self, models: Dict[str, object], k: int = 5):
        """Initialize visualizer with models.
        
        Args:
            models: Dictionary of model_name -> model object
            k: k-mer size for tokenization
        """
        self.models = models
        self.k = k
        self.reducer = None
        
    def generate_sequences(self, generator_func: Callable) -> Tuple[List[str], List[str]]:
        """Generate sequences and labels using provided generator function.
        
        Args:
            generator_func: Function that returns (sequences, labels)
            
        Returns:
            Tuple of (sequences, labels)
        """
        return generator_func()
    
    def compute_embeddings(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all models.
        
        Args:
            sequences: List of sequences to embed
            
        Returns:
            Dictionary of model_name -> embeddings array
        """
        embeddings = {}
        
        for name, model in self.models.items():
            print(f"Computing embeddings with {name}...")
            embeddings[name] = embed_sequences(model, sequences, self.k)
            
        return embeddings
    
    def reduce_dimensions(self, embeddings: Dict[str, np.ndarray], 
                         n_components: int = 2,
                         use_pca_first: bool = True,
                         pca_components: int = 50) -> Dict[str, np.ndarray]:
        """Reduce embeddings to 2D using UMAP.
        
        Args:
            embeddings: Dictionary of model_name -> embeddings
            n_components: Final number of dimensions
            use_pca_first: Whether to apply PCA before UMAP
            pca_components: Number of PCA components
            
        Returns:
            Dictionary of model_name -> 2D embeddings
        """
        reduced = {}
        
        for name, emb in embeddings.items():
            print(f"Reducing dimensions for {name}...")
            
            # Apply PCA first if requested
            if use_pca_first and emb.shape[1] > pca_components:
                pca = PCA(n_components=pca_components)
                emb = pca.fit_transform(emb)
            
            # Apply UMAP
            self.reducer = UMAP(n_components=n_components, random_state=42)
            reduced[name] = self.reducer.fit_transform(emb)
            
        return reduced
    
    def plot_comparison(self, 
                       reduced_embeddings: Dict[str, np.ndarray],
                       labels: List[str],
                       output_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (15, 5),
                       title_prefix: str = ""):
        """Plot UMAP comparisons for all models.
        
        Args:
            reduced_embeddings: Dictionary of model_name -> 2D embeddings
            labels: Labels for coloring points
            output_path: Optional path to save figure
            figsize: Figure size
            title_prefix: Prefix for subplot titles
        """
        n_models = len(reduced_embeddings)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        # Get unique labels for consistent coloring
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for ax, (name, embeddings) in zip(axes, reduced_embeddings.items()):
            for label in unique_labels:
                mask = np.array(labels) == label
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                          c=[label_to_color[label]], label=label, alpha=0.6, s=50)
            
            ax.set_title(f"{title_prefix}{name}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            # Ensure plots directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_path}")
        else:
            plt.show()
            
    def run_benchmark(self,
                     generator_func: Callable,
                     output_path: Optional[Path] = None,
                     title: str = "Embedding Comparison"):
        """Run a complete benchmark with visualization.
        
        Args:
            generator_func: Function that generates (sequences, labels)
            output_path: Optional path to save figure
            title: Title prefix for plots
        """
        # Generate data
        sequences, labels = self.generate_sequences(generator_func)
        print(f"Generated {len(sequences)} sequences with {len(set(labels))} unique labels")
        
        # Compute embeddings
        embeddings = self.compute_embeddings(sequences)
        
        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings)
        
        # Plot
        self.plot_comparison(reduced, labels, output_path, title_prefix=title + " - ")