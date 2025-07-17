"""Utilities around gensim FastText for biological sequences."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence
from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing as mp

logger = logging.getLogger(__name__)


class LossLogger(CallbackAny2Vec):
    """Logs loss at the end of each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logger.info(f"Epoch {self.epoch}, loss: {loss}")
        self.epoch += 1


def load_fasttext(path: str) -> FastText:
    """Load a saved gensim FastText model."""
    return FastText.load(path)


def train_fasttext(corpus_iter: Iterable[Sequence[str]], **kwargs) -> FastText:
    """Train a gensim FastText model from an *iterator* of token lists.

    Parameters
    ----------
    corpus_iter
        Iterable over lists of tokens (k-mers).
    kwargs
        Other kwargs to pass to `gensim.models.fasttext.FastText`.
        For example: `vector_size`, `window`, `min_count`, `epochs`.
        The `seed` parameter is crucial for reproducibility.
    """
    if "workers" not in kwargs:
        kwargs["workers"] = max(mp.cpu_count() - 1, 1)

    # Ensure seed is set for reproducibility
    if "seed" not in kwargs:
        kwargs["seed"] = 42

    logger.info(f"Training FastText with parameters: {kwargs}")
    
    # Separate model instantiation from training
    training_params = kwargs.copy()
    epochs = training_params.pop("epochs", 10) # Remove epochs for constructor
    
    model = FastText(**training_params)
    
    # Build vocab and train
    model.build_vocab(corpus_iter=corpus_iter)
    model.train(
        corpus_iter=corpus_iter,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[LossLogger()],
    )
    return model 