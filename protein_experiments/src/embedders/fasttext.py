"""FastText embedder with optional positional encoding support."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

from src.embedders.base import BaseEmbedder
from pe_fasttext.model import PEFastText
from pe_fasttext.fasttext_utils import train_fasttext
from gensim.models import FastText

logger = logging.getLogger(__name__)


class FastTextEmbedder(BaseEmbedder):
    """FastText embedder with optional positional encoding."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FastText embedder.
        
        Config keys:
            - model_path: Path to save/load model
            - pretrained_path: Path to pre-trained model (optional)
            - fine_tune: Whether to fine-tune pre-trained model (default: True)
            - tokenization: "kmer" or "residue"
            - k: k-mer size (only for kmer tokenization)
            - dim: Embedding dimension
            - pos_encoder: Optional positional encoding type
            - fusion: "add" or "concat" (only with pos_encoder)
            - train_params: Dict with epochs, window, etc.
        """
        super().__init__(config)
        self.model_path = Path(config.get('model_path', 'fasttext_model.bin'))
        self.pretrained_path = config.get('pretrained_path')
        if self.pretrained_path:
            self.pretrained_path = Path(self.pretrained_path)
        self.fine_tune = config.get('fine_tune', True)
        self.tokenization = config.get('tokenization', 'kmer')
        self.k = config.get('k', 6)
        self.dim = config.get('dim', 128)
        self.pos_encoder = config.get('pos_encoder')
        self.fusion = config.get('fusion', 'add')
        self.train_params = config.get('train_params', {})
        
        self._model = None
        self._pe_model = None
    
    def _tokenize(self, sequences: List[str]) -> List[List[str]]:
        """Tokenize sequences based on tokenization method."""
        if self.tokenization == "residue":
            return [list(seq) for seq in sequences]
        else:  # kmer
            tokenized = []
            for seq in sequences:
                tokens = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
                tokenized.append(tokens)
            return tokenized
    
    def train(self, sequences: List[str]):
        """Train FastText model on sequences."""
        if self.model_path.exists():
            logger.info(f"Model already exists at {self.model_path}")
            return
        
        # Check if we should use pre-trained model
        if self.pretrained_path and self.pretrained_path.exists():
            if not self.fine_tune:
                # Just copy the pre-trained model
                logger.info(f"Using pre-trained model from {self.pretrained_path} (no fine-tuning)")
                import shutil
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.pretrained_path, self.model_path)
                # Also copy the vectors file if it exists
                vectors_src = Path(str(self.pretrained_path) + '.wv.vectors_ngrams.npy')
                if vectors_src.exists():
                    vectors_dst = Path(str(self.model_path) + '.wv.vectors_ngrams.npy')
                    shutil.copy2(vectors_src, vectors_dst)
                return
            else:
                # Load pre-trained model for fine-tuning
                logger.info(f"Loading pre-trained model from {self.pretrained_path} for fine-tuning")
                model = FastText.load(str(self.pretrained_path))
                
                # Prepare for fine-tuning
                logger.info(f"Fine-tuning on {len(sequences)} task sequences")
                tokenized = self._tokenize(sequences)
                
                # Update vocabulary with new tokens
                model.build_vocab(tokenized, update=True)
                
                # Fine-tune
                total_examples = len(tokenized)
                model.train(tokenized, total_examples=total_examples, 
                           epochs=self.train_params.get('epochs', 10))
        else:
            # Train from scratch
            logger.info(f"Training FastText model from scratch ({self.tokenization} tokenization)")
            tokenized = self._tokenize(sequences)
            
            # Default training parameters
            params = {
                'vector_size': self.dim,
                'window': 5,
                'min_count': 1,
                'sg': 1,  # Skip-gram
                'epochs': 10,
                'workers': 4
            }
            params.update(self.train_params)
            
            model = train_fasttext(corpus_iter=tokenized, **params)
        
        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(self.model_path))
        logger.info(f"Model saved to {self.model_path}")
    
    def _load_models(self):
        """Load FastText and optionally PE-FastText models."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}. Train first.")
            
            if self.pos_encoder:
                self._pe_model = PEFastText(
                    fasttext_path=str(self.model_path),
                    pos_encoder=self.pos_encoder,
                    fusion=self.fusion
                )
                self._model = self._pe_model.ft
            else:
                self._model = FastText.load(str(self.model_path))
    
    def embed(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for sequences."""
        self._load_models()
        
        embeddings = []
        iterator = tqdm(sequences, desc="Generating embeddings") if show_progress else sequences
        
        for seq in iterator:
            if self.tokenization == "residue":
                tokens = list(seq)
            else:
                tokens = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
            
            if len(tokens) == 0:
                # Handle empty sequences
                embeddings.append(np.zeros(self.embedding_dim))
                continue
            
            if self._pe_model:
                # Use PE-FastText
                positions = list(range(len(tokens)))
                emb = self._pe_model.embed_tokens(tokens, positions)
                embeddings.append(emb.mean(axis=0))
            else:
                # Regular FastText
                token_embs = []
                for token in tokens:
                    if token in self._model.wv:
                        token_embs.append(self._model.wv[token])
                
                if token_embs:
                    embeddings.append(np.mean(token_embs, axis=0))
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self._pe_model:
            return self._pe_model.dim_total
        else:
            return self.dim