import os
import gzip
from pathlib import Path
from typing import Iterator, Dict, Optional, Union, List
from datasets import GeneratorBasedBuilder, DatasetInfo, Features, Value, SplitGenerator, Split, BuilderConfig
from Bio import SeqIO
import argparse
import csv
import json

# -----------------------------
# Metadata loading utilities
# -----------------------------
def load_metadata(metadata_path: Optional[str], fmt: str = "auto") -> Dict[str, dict]:
    """
    Load metadata mapping from a CSV, TSV, or JSON file.
    Returns a dict mapping sequence IDs to metadata dicts.
    """
    if not metadata_path:
        return {}
    metadata = {}
    # Auto-detect format if not specified
    if fmt == "auto":
        if metadata_path.endswith(".json"):
            fmt = "json"
        elif metadata_path.endswith(".csv"):
            fmt = "csv"
        elif metadata_path.endswith(".tsv"):
            fmt = "tsv"
        else:
            raise ValueError("Unknown metadata file format. Please specify --metadata_format.")
    if fmt == "json":
        with open(metadata_path) as f:
            data = json.load(f)
            # Accept either {id: {...}} or [{id:..., ...}, ...]
            if isinstance(data, dict):
                metadata = data
            elif isinstance(data, list):
                for entry in data:
                    seq_id = entry.get("id")
                    if seq_id:
                        metadata[seq_id] = entry
    elif fmt in ("csv", "tsv"):
        delimiter = "," if fmt == "csv" else "\t"
        with open(metadata_path) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                seq_id = row.get("id")
                if seq_id:
                    metadata[seq_id] = row
    else:
        raise ValueError(f"Unsupported metadata format: {fmt}")
    return metadata

# -----------------------------
# FASTA streaming utilities
# -----------------------------
def stream_fasta_records(
    fasta_path: Union[str, Path],
    metadata: Optional[Dict[str, dict]] = None
) -> Iterator[Dict[str, str]]:
    """
    Stream records from a (possibly gzipped) FASTA file.
    Yields dicts with 'id', 'sequence', 'description', and optional 'metadata'.
    """
    fasta_path = Path(fasta_path)
    open_func = gzip.open if fasta_path.suffix == ".gz" else open
    mode = "rt"
    with open_func(fasta_path, mode) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            entry = {
                "id": seq_id,
                "sequence": str(record.seq),
                "description": record.description,
            }
            if metadata and seq_id in metadata:
                entry["metadata"] = metadata[seq_id]
            else:
                entry["metadata"] = None
            yield entry

def stream_fasta_directory(
    directory: Union[str, Path],
    metadata: Optional[Dict[str, dict]] = None
) -> Iterator[Dict[str, str]]:
    """
    Stream all FASTA records from all .fasta/.fa/.fasta.gz/.fa.gz files in a directory.
    """
    directory = Path(directory)
    for file in directory.iterdir():
        if file.suffix in [".fa", ".fasta"] or file.suffixes[-2:] in [[".fa", ".gz"], [".fasta", ".gz"]]:
            yield from stream_fasta_records(file, metadata)

# -----------------------------
# BuilderConfig for dataset variants
# -----------------------------
class FastaDatasetConfig(BuilderConfig):
    """
    Configuration for different FASTA dataset variants.
    Allows specifying metadata, label schemes, or parsing options.
    """
    def __init__(
        self,
        name: str,
        description: str = "",
        metadata_path: Optional[str] = None,
        metadata_format: str = "auto",
        **kwargs
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.metadata_path = metadata_path
        self.metadata_format = metadata_format

# -----------------------------
# Main HuggingFace dataset class
# -----------------------------
class FastaDataset(GeneratorBasedBuilder):
    """
    Generic HuggingFace datasets generator for FASTA files or directories, with optional metadata and flexible configs.
    """
    VERSION = "1.0.0"
    BUILDER_CONFIG_CLASS = FastaDatasetConfig
    BUILDER_CONFIGS = [
        FastaDatasetConfig(
            name="default",
            description="Default: generic FASTA parsing, no extra metadata."
        ),
        # Add more configs here for different variants if needed
        # Example:
        # FastaDatasetConfig(
        #     name="with_labels",
        #     description="FASTA with per-sequence labels from a CSV file.",
        #     metadata_path="labels.csv",
        #     metadata_format="csv"
        # ),
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="Generic FASTA dataset (DNA or protein, unmodified), with optional metadata",
            features=Features({
                "id": Value("string"),
                "sequence": Value("string"),
                "description": Value("string"),
                "metadata": Value("string"),  # Store as JSON string for flexibility
            }),
            supervised_keys=None,
            homepage=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        data_path = Path(dl_manager.manual_dir)
        # Use config to get metadata path/format
        metadata_path = self.config.metadata_path
        metadata_format = self.config.metadata_format
        metadata = load_metadata(metadata_path, metadata_format) if metadata_path else None
        # If it's a directory, treat as directory, else as file
        if data_path.is_dir():
            files = [str(f) for f in data_path.iterdir() if f.is_file()]
        else:
            files = [str(data_path)]
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"files": files, "metadata": metadata},
            )
        ]

    def _generate_examples(self, files: List[str], metadata: Optional[Dict[str, dict]] = None):
        idx = 0
        for file in files:
            for record in stream_fasta_records(file, metadata):
                # Store metadata as JSON string for HuggingFace compatibility
                if record["metadata"] is not None:
                    record["metadata"] = json.dumps(record["metadata"])
                yield idx, record
                idx += 1

# -----------------------------
# CLI for local preview/testing
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stream and convert FASTA to HuggingFace dataset format, with optional metadata and config variants."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to FASTA file or directory of FASTA files.")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata file (CSV, TSV, or JSON) with 'id' column.")
    parser.add_argument("--metadata_format", type=str, default="auto", choices=["auto", "csv", "tsv", "json"], help="Format of metadata file.")
    parser.add_argument("--max_records", type=int, default=10, help="Max records to print (for preview).")
    parser.add_argument("--config", type=str, default="default", help="Dataset config variant to use.")
    args = parser.parse_args()

    # Load metadata if provided
    metadata = load_metadata(args.metadata, args.metadata_format) if args.metadata else None
    input_path = Path(args.input)
    if input_path.is_dir():
        generator = stream_fasta_directory(input_path, metadata)
    else:
        generator = stream_fasta_records(input_path, metadata)

    print(f"\nPreviewing up to {args.max_records} records from {args.input} (config: {args.config}):\n")
    for i, record in enumerate(generator):
        print(json.dumps(record, indent=2))
        if i + 1 >= args.max_records:
            break
    print("\nDone.")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
