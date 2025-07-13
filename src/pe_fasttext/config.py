"""Configuration module for PE-FastText training parameters."""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class TrainingConfig:
    """Configuration for FastText training."""
    
    # Model parameters
    vector_size: int = 512
    window: int = 5
    min_count: int = 1
    epochs: int = 5
    sg: int = 1  # 1=skipgram, 0=CBOW
    
    # Training parameters
    workers: Optional[int] = None
    seed: int = 42
    
    # Data parameters
    k: int = 5  # k-mer size
    max_sequences: Optional[int] = None
    uppercase: bool = True
    
    def __post_init__(self):
        """Set default workers if not specified."""
        if self.workers is None:
            cpu_count = os.cpu_count()
            self.workers = max(1, (cpu_count - 1) if cpu_count else 1)


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    
    name: str
    split: str = "train"
    field: str = "sequence"  # Default field name
    streaming: bool = True
    
    # Common dataset configurations
    CONFIGS = {
        "tattabio/OG": {"field": "IGS_seqs"},
        "tattabio/OG_dna": {"field": "IGS_seqs"},
        "tattabio/OG_protein": {"field": "protein_seqs"},
        "uniref50": {"field": "sequence"},
        "mgnify_proteins": {"field": "sequence"},
    }
    
    def __post_init__(self):
        """Apply dataset-specific defaults."""
        if self.name in self.CONFIGS:
            config = self.CONFIGS[self.name]
            if self.field == "sequence" and "field" in config:
                self.field = config["field"]


@dataclass
class EmbeddingConfig:
    """Configuration for positional embeddings."""
    
    position_type: str = "sinusoidal"
    position_dim: Optional[int] = None
    max_sequence_length: int = 10000
    fusion_rule: str = "add"
    
    def __post_init__(self):
        """Validate configuration."""
        valid_position_types = {"sinusoidal", "learned", "rope", "alibi"}
        if self.position_type not in valid_position_types:
            raise ValueError(f"position_type must be one of {valid_position_types}")
        
        valid_fusion_rules = {"add", "concat"}
        if self.fusion_rule not in valid_fusion_rules:
            raise ValueError(f"fusion_rule must be one of {valid_fusion_rules}")