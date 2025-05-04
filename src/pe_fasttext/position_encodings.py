"""Positional encodings for PE-FastText."""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

try:
    import torch
except ImportError:  # torch optional
    torch = None  # type: ignore


class BasePositionalEncoding:
    """Abstract base class for positional encodings."""

    name: str = "base"

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        raise NotImplementedError


class SinusoidalEncoding(BasePositionalEncoding):
    """Classic sine/cosine encoding from *Attention Is All You Need*.

    Dim must be even.
    """

    name = "sinusoid"

    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)[:, None]  # (N,1)
        d = np.arange(self.dim)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (d // 2)) / self.dim)
        angle_rads = positions * angle_rates
        sines = np.sin(angle_rads[:, 0::2])
        coses = np.cos(angle_rads[:, 1::2])
        return np.concatenate([sines, coses], axis=-1)


class LearnedPositionalEncoding(BasePositionalEncoding):
    """Simple learned embedding table (numpy version)."""

    name = "learned"

    def __init__(self, dim: int, max_len: int = 100_000):
        super().__init__(dim)
        self.max_len = max_len
        # random init
        self.table = np.random.randn(max_len, dim).astype(np.float32)

    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)
        if positions.max() >= self.max_len:
            # dynamically grow
            new_max = int(positions.max() * 1.1)
            pad = np.random.randn(new_max - self.max_len, self.dim).astype(np.float32)
            self.table = np.vstack([self.table, pad])
            self.max_len = new_max
        return self.table[positions]


class RoPEEncoding(BasePositionalEncoding):
    """Rotary positional encoding (extended to static vectors by flattening complex dims)."""

    name = "rope"

    def __init__(self, dim: int):
        if dim % 2 != 0:
            raise ValueError("RoPE requires even dimension")
        super().__init__(dim)
        self.half_dim = dim // 2
        # pre-compute inv freq
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, self.half_dim, 2) / self.half_dim))

    def _rope(self, pos: np.ndarray) -> np.ndarray:
        # This re-implements RoPE but outputs flattened real-valued vectors.
        # Adopted from https://github.com/lucidrains/rotary-embedding-torch
        freqs = np.einsum("i,j->ij", pos, self.inv_freq)  # (N, half_dim/2)
        emb = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
        return np.tile(emb, (1, 2))  # repeat to reach half_dim*2 = dim

    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)
        return self._rope(positions)


class ALiBiEncoding(BasePositionalEncoding):
    """Attention with Linear Biases simplified to static slope vector.*

    Not a faithful reproduction (requires attention). We approximate by a linear ramp.
    """

    name = "alibi"

    def __call__(self, positions: np.ndarray | list[int]) -> np.ndarray:
        positions = np.asarray(positions)[:, None]
        slopes = np.linspace(0, 1, self.dim)[None, :]
        return positions * slopes  # linear scaling


# map encoder names to their classes
ENCODERS = {  # type: ignore
    cls.name: cls
    for cls in [SinusoidalEncoding, LearnedPositionalEncoding, RoPEEncoding, ALiBiEncoding]
}


def build_positional_encoder(name: str, dim: int, **kwargs) -> BasePositionalEncoding:
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder: {name}")
    return ENCODERS[name](dim=dim, **kwargs) 