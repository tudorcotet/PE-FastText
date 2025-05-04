"""Command-line interface for PE-FastText."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

import typer
import numpy as np
from rich import print

from .dataset import download_dataset, kmers_from_dataset
from .fasttext_utils import train_fasttext, load_fasttext
from .tokenization import fasta_stream
from .model import PEFastText
from .position_encodings import ENCODERS

app = typer.Typer(add_help_option=True)


@app.command("download")
def download_datasets_cli(
    corpus: List[str] = typer.Option(..., help="Dataset name(s) to fetch"),
    output: str = typer.Option("data", help="Destination directory"),
):
    """Download raw FASTA corpora."""
    for name in corpus:
        path = download_dataset(name, output)
        print(f"Downloaded {name} -> {path}")


@app.command("train")
def train_fasttext_cli(
    corpus: str = typer.Option(..., help="Path to FASTA corpus"),
    kmer: List[int] = typer.Option([5], help="k-mer sizes"),
    dim: int = typer.Option(512, help="Embedding dimensionality"),
    workers: int = typer.Option(0, help="Workers (0 = all cores)"),
    epochs: int = typer.Option(5, help="Number of epochs"),
    output: str = typer.Option("fasttext.bin", help="Output path"),
    fmt: Literal["bin", "vec"] = typer.Option("bin", "--format", "-f", help="Output format: bin or vec"),
):
    """Train FastText model from FASTA."""
    if workers == 0:
        workers = os.cpu_count() or 1
    # Generate list of tokens (k-mers) per FASTA record as a "sentence"
    def corpus_iter():
        for k in kmer:
            tokens = list(fasta_stream(corpus, k))
            yield tokens

    model = train_fasttext(corpus_iter(), vector_size=dim, workers=workers, epochs=epochs)
    if fmt == "bin":
        model.save(output)
    else:
        model.wv.save_word2vec_format(output, binary=False)
    print(f"Saved FastText model -> {output} (format={fmt})")


@app.command("embed")
def embed_cli(
    model: str = typer.Option(..., help="Path to FastText .bin"),
    pos_encoder: str = typer.Option("sinusoid", help="Positional encoder name"),
    fusion: str = typer.Option("add"),
    fasta: str = typer.Option(..., help="FASTA to embed"),
    index_out: str = typer.Option("embeddings.npy"),
):
    """Embed a FASTA and save a NumPy matrix."""
    peft = PEFastText(fasttext_path=model, pos_encoder=pos_encoder, fusion=fusion)
    mat = peft.embed_fasta(fasta)
    np.save(index_out, mat)
    print(f"Saved embeddings -> {index_out}")


@app.command("benchmark")
def benchmark_cli():
    from .benchmark import run_benchmarks
    run_benchmarks()


@app.command("ablation")
def ablation_cli():
    from .ablation import run_ablation
    run_ablation()


if __name__ == "__main__":
    app() 