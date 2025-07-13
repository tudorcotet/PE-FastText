"""Dataset download + preprocessing utilities for PE-FastText."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypedDict

from .tokenization import fasta_stream

class DatasetInfo(TypedDict):
    url: str
    size: int

# URLs or FTP roots for datasets (simplified subset)
DATASETS: dict[str, DatasetInfo] = {
    "uniref50": {
        "url": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
        "size": 20_000_000_000,  # bytes ~20 GB compressed
    },
    "mgnify_protein": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_protein_catalogue/2022/psc.fasta.gz",
        "size": 120_000_000_000,
    },
    # add more as needed
}


CHUNK = 1 << 20  # 1 MiB


def _download_file(url: str, dst: Path, *, show_progress: bool = True):
    """Download file using shared utility."""
    from .utils import download_file
    if dst.exists():
        return dst
    return download_file(url, dst)


def download_dataset(name: str, out_dir: str | Path = "data") -> Path:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    info = DATASETS[name]
    url = info["url"]
    fname = url.split("/")[-1]
    out_dir = Path(out_dir)
    dst = out_dir / fname
    _download_file(url, dst)
    # decompress if gz
    if dst.suffix == ".gz":
        decompressed = dst.with_suffix("")  # remove .gz
        if not decompressed.exists():
            from .utils import decompress_file
            decompress_file(dst, decompressed)
        return decompressed
    return dst


# ------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------

def kmers_from_dataset(name: str, k: int, out_dir: str | Path = "data") -> Iterator[str]:
    """Stream k-mers from a (downloaded) dataset."""
    fasta_path = download_dataset(name, out_dir)
    return fasta_stream(str(fasta_path), k) 