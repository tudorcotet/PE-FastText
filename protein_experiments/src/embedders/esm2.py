"""ESM2 embedder (optional, requires torch and transformers)."""

from typing import List, Dict, Any
import numpy as np
import logging

from src.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

# Make ESM2 optional
try:
    import torch
    from transformers import AutoTokenizer, EsmModel
    from tqdm import tqdm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("ESM2 embedder not available. Install torch and transformers.")


class ESM2Embedder(BaseEmbedder):
    """ESM2 protein language model embedder."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ESM2 embedder.
        
        Config keys:
            - model_name: HuggingFace model name (default: facebook/esm2_t6_8M_UR50D)
            - device: "cuda" or "cpu"
            - batch_size: Batch size for inference
            - layer: Which layer to extract (-1 for last)
            - pooling: "mean", "cls", or "max"
        """
        if not HAS_TORCH:
            raise ImportError("ESM2 requires torch and transformers. Install with: pip install torch transformers")
        
        super().__init__(config)
        self.model_name = config.get('model_name', 'facebook/esm2_t6_8M_UR50D')
        self.device = torch.device(config.get('device', 'cpu'))
        self.batch_size = config.get('batch_size', 32)
        self.layer = config.get('layer', -1)
        self.pooling = config.get('pooling', 'mean')
        
        # Load model and tokenizer
        logger.info(f"Loading ESM2 model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Cache embedding dimension
        self._embedding_dim = self.model.config.hidden_size
    
    def embed(self, sequences: List[str], show_progress: bool = True, average_sequences: bool = True) -> np.ndarray:
        """Generate embeddings for sequences."""
        all_embs = []
        
        # Process in batches
        n_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(sequences), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Generating ESM2 embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch_sequences = sequences[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract the correct layer's hidden states
                if self.layer == -1:
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs.hidden_states[self.layer]
                
                # If we need per-residue embeddings, we handle it here
                if not average_sequences:
                    # Remove special tokens (CLS, EOS) from each sequence's embeddings
                    for j, seq in enumerate(batch_sequences):
                        # Get the length of the original sequence without padding
                        seq_len = len(self.tokenizer.tokenize(seq))
                        # Extract embeddings, removing [CLS] and [SEP] tokens
                        all_embs.append(hidden_states[j, 1:seq_len+1].cpu().numpy())
                else:
                    # Pool embeddings to get one vector per sequence
                    batch_embeddings = self._pool(hidden_states, inputs["attention_mask"])
                    all_embs.extend(batch_embeddings.cpu().numpy())
        
        if not average_sequences:
            return all_embs # Return list of arrays with variable length

        return np.array(all_embs)
    
    def _pool(self, hidden_states, attention_mask):
        """Pool token embeddings to sequence embeddings."""
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            return sum_embeddings / sum_mask
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[mask == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim