"""Tokenization utilities.

This module streams overlapping k-mers (subsequences) from FASTA / raw string.
It avoids materialising the full list in memory, yielding one k-mer at a time.

Usage
-----
>>> from pe_fasttext.tokenization import kmers
>>> for kmer in kmers("ACGTACGT", k=4):
...     print(kmer)
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import re

FASTA_HEADER_RE = re.compile(r"^>")

def kmers(seq: str | Sequence[str], k: int) -> Iterator[str]:
    """Yield all *overlapping* k-mers from *seq*.

    If *seq* is shorter than *k*, yields nothing.
    """
    if isinstance(seq, str):
        n = len(seq)
        for i in range(n - k + 1):
            yield seq[i : i + k]
    else:  # sequence of tokens
        for tokens in seq:
            yield from kmers(tokens, k)


def fasta_stream(path: str, k: int, *, upper: bool = True) -> Iterator[str]:
    """Stream k-mers from a FASTA file on disk.

    Parameters
    ----------
    path
        Path to FASTA file.
    k
        k-mer length.
    upper
        Convert sequences to uppercase (default).
    """
    with open(path, "r") as fh:
        buf: list[str] = []
        for line in fh:
            if FASTA_HEADER_RE.match(line):
                # Flush previous sequence buffer.
                if buf:
                    seq = "".join(buf)
                    if upper:
                        seq = seq.upper()
                    yield from kmers(seq, k)
                    buf.clear()
            else:
                buf.append(line.strip())
        # flush last record
        if buf:
            seq = "".join(buf)
            if upper:
                seq = seq.upper()
            yield from kmers(seq, k) 