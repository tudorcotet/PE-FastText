import os
import gzip
import shutil
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile

class DatasetInfo:
    def __init__(self, name, urls, size_gb=None, description=None):
        self.name = name
        self.urls = urls if isinstance(urls, list) else [urls]
        self.size_gb = size_gb
        self.description = description

DATASETS = {
    "uniref50": DatasetInfo(
        name="UniRef50",
        urls=[
            "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
            "https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
        ],
        size_gb=20,
        description="UniRef50 protein sequence database"
    ),
    "grch38": DatasetInfo(
        name="GRCh38 Reference Genome",
        urls=[
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/GCA_000001405.15_GRCh38_genomic.fna.gz"
        ],
        size_gb=3,
        description="GRCh38 reference genome"
    ),
    "encode_regulatory": DatasetInfo(
        name="ENCODE Regulatory Elements",
        urls=[
            "https://www.encodeproject.org/files/ENCFF001TDO/@@download/ENCFF001TDO.bed.gz",  # Candidate regulatory elements
            "https://www.encodeproject.org/files/ENCFF001TDP/@@download/ENCFF001TDP.bed.gz",  # Candidate enhancers
            "https://www.encodeproject.org/files/ENCFF001TDQ/@@download/ENCFF001TDQ.bed.gz"   # Candidate promoters
        ],
        size_gb=1,
        description="ENCODE regulatory elements including enhancers and promoters"
    )
    # TODO: Add more biological sequence datasets here
    # Examples of datasets that could be added:
    # - MGnify protein catalog (requires significant storage ~120GB)
    # - CAMI II simulated metagenomes
    # - Custom protein/DNA sequence collections
    # - Specialized genome assemblies
}

def download_file(url: str, output_path: Path, chunk_size: int = 1024*1024) -> bool:
    """
    Download a file with progress bar.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False

def decompress_file(input_path: Path, output_path: Path = None) -> Path:
    """
    Decompress a file (supports .gz and .tar.gz).
    Returns path to decompressed file.
    """
    print(f"\nDecompressing {input_path}...")
    
    if input_path.suffix == '.gz' and input_path.suffixes[-2:] != ['.tar', '.gz']:
        # Regular gzip file
        output_path = output_path or input_path.with_suffix('')
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    elif input_path.suffixes[-2:] == ['.tar', '.gz']:
        # Tar.gz file
        output_path = output_path or input_path.parent
        with tarfile.open(input_path, 'r:gz') as tar:
            tar.extractall(path=output_path)
            # Get the extracted directory name
            output_path = output_path / Path(tar.getnames()[0]).parts[0]
    
    print(f"Successfully decompressed to {output_path}")
    return output_path

def download_dataset(name: str, output_dir: str = "data", chunk_size: int = 1024*1024) -> list[Path]:
    """
    Download and decompress a dataset.
    Returns list of paths to decompressed files.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {', '.join(DATASETS.keys())}")
    
    dataset = DATASETS[name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset.name}")
    print(f"Description: {dataset.description}")
    if dataset.size_gb:
        print(f"Approximate size: {dataset.size_gb} GB")
    
    decompressed_files = []
    
    for url in dataset.urls:
        filename = Path(url).name
        output_path = output_dir / filename
        final_path = output_path.with_suffix('') if output_path.suffix == '.gz' else output_dir
        
        # Skip if final file exists
        if final_path.exists():
            print(f"File already exists: {final_path}")
            decompressed_files.append(final_path)
            continue
        
        # Try downloading
        success = False
        success = download_file(url, output_path, chunk_size)
        
        if not success:
            print(f"Failed to download {filename}")
            continue
        
        # Decompress
        try:
            decompressed = decompress_file(output_path)
            decompressed_files.append(decompressed)
            # Remove compressed file to save space
            output_path.unlink()
            print(f"Removed compressed file: {output_path}")
        except Exception as e:
            print(f"Error during decompression: {e}")
            if output_path.exists():
                output_path.unlink()
            continue
    
    if not decompressed_files:
        raise Exception(f"Failed to download {name} from any source")
    
    return decompressed_files

def list_datasets():
    """Print information about available datasets."""
    print("\nAvailable datasets:")
    print("-" * 80)
    for key, dataset in DATASETS.items():
        print(f"\nName: {dataset.name} (key: {key})")
        print(f"Description: {dataset.description}")
        if dataset.size_gb:
            print(f"Approximate size: {dataset.size_gb} GB")
    print("\nUsage: python download.py --dataset <dataset_key>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download biological sequence datasets")
    parser.add_argument("--dataset", type=str, help="Dataset to download")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.dataset:
        try:
            output_paths = download_dataset(args.dataset, args.output)
            print(f"\nDownload completed successfully!")
            print("Files downloaded:")
            for path in output_paths:
                print(f"- {path}")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        parser.print_help()
