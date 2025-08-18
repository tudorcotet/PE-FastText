"""Main PE-FastText model interface."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .fasttext_utils import load_fasttext
from .position_encodings import build_positional_encoder, BasePositionalEncoding


FusionRule = Literal["add", "concat"]


class PEFastText:
    """Positionally-enhanced FastText wrapper."""

    def __init__(
        self,
        fasttext_path: str,
        pos_encoder: str = "sinusoid",
        fusion: FusionRule = "add",
        pos_dim: int | None = None,
    ):
        # Load FastText model (gensim)
        self.ft = load_fasttext(fasttext_path)
        self.dim_sem = self.ft.vector_size

        # Build positional encoder
        if pos_dim is None:
            pos_dim = self.dim_sem  # keep same size if add, else can differ
        self.pos_enc: BasePositionalEncoding = build_positional_encoder(
            pos_encoder, pos_dim
        )
        self.fusion: FusionRule = fusion

        if fusion == "add" and pos_dim != self.dim_sem:
            raise ValueError("For 'add' fusion, pos_dim must equal semantics dim")
        self.dim_total = self.dim_sem if fusion == "add" else self.dim_sem + pos_dim

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def embed_tokens(self, tokens: list[str], positions: list[int]) -> np.ndarray:
        """Embed a *list* of tokens with their absolute positions.

        Parameters
        ----------
        tokens
            list of k-mer strings present in FastText vocab (otherwise OOV).
        positions
            Same length as tokens, absolute start index.
        """
        sem = np.vstack([self.ft.wv[token] for token in tokens]).astype(np.float32)
        pos = self.pos_enc(positions).astype(np.float32)
        if self.fusion == "add":
            out = sem + pos
        else:  # concat
            out = np.concatenate([sem, pos], axis=-1)
        return out

    def embed(self, sequences: list[str], k: int = 5, average_sequences: bool = False, show_progress: bool = False, tokenization: str = "kmer") -> list[np.ndarray] | np.ndarray:
        """Embed a list of sequences."""
        from .utils import kmerize
        from tqdm import tqdm

        all_embeddings = []
        iterator = tqdm(sequences, desc="Embedding sequences") if show_progress else sequences

        for seq in iterator:
            if tokenization == 'residue':
                tokens = list(seq)
            else:
                tokens = kmerize(seq, k)
            
            if not tokens:
                if average_sequences:
                    all_embeddings.append(np.zeros(self.dim_total))
                else:
                    all_embeddings.append(np.zeros((0, self.dim_total)))
                continue

            positions = list(range(len(tokens)))
            token_embs = self.embed_tokens(tokens, positions)
            
            if average_sequences:
                all_embeddings.append(token_embs.mean(axis=0))
            else:
                all_embeddings.append(token_embs)
        
        if average_sequences:
            return np.array(all_embeddings)
        return all_embeddings

    def embed_fasta(self, fasta_path: str, k: int = 5) -> np.ndarray:
        """Embed entire FASTA file into matrix (N tokens x dim).

        Warning: loads whole output in memory. Use stream_* utilities otherwise.
        """
        from .tokenization import fasta_stream

        tokens = []
        positions = []
        pos = 0
        for kmer in fasta_stream(fasta_path, k):
            tokens.append(kmer)
            positions.append(pos)
            pos += 1
        return self.embed_tokens(tokens, positions) 