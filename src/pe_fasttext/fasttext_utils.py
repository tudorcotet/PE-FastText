"""Utilities around gensim FastText for biological sequences."""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Sequence

from gensim.models.fasttext import FastText  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore


class LossLogger(CallbackAny2Vec):
    """Logs loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0.0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f"[fasttext] epoch {self.epoch}: loss={loss - self.loss_previous_step:.2f}")
        self.loss_previous_step = loss
        self.epoch += 1


def train_fasttext(
    corpus_iter: Iterable[Sequence[str]],
    vector_size: int = 512,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 5,
    sg: int = 1,
    workers: int | None = None,
    **kwargs,
) -> FastText:
    """Train a gensim FastText model from an *iterator* of token lists.

    Parameters
    ----------
    corpus_iter
        Iterable over lists of tokens (k-mers).
    vector_size
        Embedding dimensionality.
    window
        Context window.
    """
    if workers is None:
        workers = max(mp.cpu_count() - 1, 1)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        **kwargs,
    )
    # build vocab in streaming way: need a first pass
    model.build_vocab(corpus_iter)
    model.train(
        corpus_iter,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[LossLogger()],
    )
    return model


def load_fasttext(path: str | Path) -> FastText:
    """Load a model from disk (binary format)."""
    return FastText.load(str(path)) 