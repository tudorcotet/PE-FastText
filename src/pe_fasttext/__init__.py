"""PE-FastText: Positionally-enhanced FastText embeddings for biological sequences."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pe-fasttext")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from .model import PEFastText  # noqa: F401 