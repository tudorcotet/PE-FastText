import os
import gzip
import shutil
import argparse
from pathlib import Path
from typing import Optional # Import Optional
import requests
from tqdm import tqdm # type: ignore
import tarfile

class DatasetInfo:
    def __init__(self, name, urls=None, size_gb=None, description=None, study_accession=None, result_group=None):
        self.name = name
        self.urls = urls if urls is not None else []
        self.size_gb = size_gb
        self.description = description
        self.study_accession = study_accession
        self.result_group = result_group

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
    ),
    "mgnify_clusters": DatasetInfo(
        name="MGnify Protein Clusters (v2024_04)",
        urls=[
            "http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/mgy_clusters.fa.gz"
        ],
        size_gb=74, # Compressed size, uncompressed will be larger
        description="Non-redundant protein cluster representative sequences from MGnify"
    ),
    "mgnify_tara_oceans_seq": DatasetInfo(
        name="MGnify Tara Oceans Sequence Data (MGYS00000410)",
        study_accession="MGYS00000410",
        result_group="sequence_data",
        description="Sequence data from the Tara Oceans prokaryotic study MGYS00000410 downloaded via mg-toolkit."
    )
    # TODO: Add more biological sequence datasets here
    # Examples of datasets that could be added:
    # - MGnify protein catalog (requires significant storage ~120GB)
    # - CAMI II simulated metagenomes
    # - Custom protein/DNA sequence collections
    # - Specialized genome assemblies
}

# Try importing mg_toolkit components, handle ImportError if not installed
MG_TOOLKIT_AVAILABLE = False
try:
    from mg_toolkit.bulk_download import BulkDownloader # type: ignore
    # Import other necessary components used by BulkDownloader based on source code scan
    # Based on the source code, BulkDownloader uses these directly or indirectly
    from mg_toolkit.constants import API_BASE, MG_ANALYSES_BASE_URL, MG_ANALYSES_DOWNLOADS_URL # type: ignore
    from mg_toolkit.exceptions import FailToGetException # type: ignore
    from requests import Session
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry # type: ignore
    # tqdm is already imported
    MG_TOOLKIT_AVAILABLE = True
except ImportError:
    print("Warning: mg_toolkit not found. Cannot use MGnify download functionality.")

def download_file(url: str, output_path: Path, chunk_size: int = 1024*1024) -> bool:
    """
    Download a file with progress bar.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"Downloading from {url}...")
        # Use the requests library directly
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

def decompress_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Decompress a file (supports .gz and .tar.gz).
    Returns path to decompressed file.
    """
    print(f"\nDecompressing {input_path}...")

    if input_path.suffix == '.gz' and input_path.suffixes[-2:] != ['.tar', '.gz']:
        # Regular gzip file
        # Ensure output_path is a Path object
        output_path = output_path if output_path is not None else input_path.with_suffix('')
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    elif input_path.suffixes[-2:] == ['.tar', '.gz']:
        # Tar.gz file
        # Ensure output_path is a Path object
        output_path = output_path if output_path is not None else input_path.parent
        with tarfile.open(input_path, 'r:gz') as tar:
            # Extract all contents to the output directory
            tar.extractall(path=output_path)
            # Assuming the first extracted item is the main directory/file
            # This might need adjustment based on actual tar file structure
            extracted_items = tar.getnames()
            if extracted_items:
                 # Return the path to the extracted directory or file
                 output_path = output_path / extracted_items[0].split('/')[0]
            else:
                 # If tar is empty or structure is unexpected, return the output_path
                 print(f"Warning: Could not determine main extracted path from {input_path}.")

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
    # Ensure output_dir is a Path object from the start
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {dataset.name}")
    print(f"Description: {dataset.description}")
    if dataset.size_gb:
        print(f"Approximate size: {dataset.size_gb} GB")

    decompressed_files = []

    if dataset.study_accession and dataset.result_group:
        # Use the mg_toolkit download function for MGnify datasets
        # Pass the Path object to the function
        try:
            downloaded = download_mgnify_study_data(dataset.study_accession, dataset.result_group, output_dir_path)
            decompressed_files.extend(downloaded)
        except Exception as e:
            print(f"Failed to download MGnify data using mg_toolkit: {e}")
            # Depending on requirements, you might want to raise the exception
            # raise e

    elif dataset.urls:
        # Use the standard file download for URL-based datasets
        for url in dataset.urls:
            filename = Path(url).name
            # Use Path objects for joining
            output_path = output_dir_path / filename

            # Determine the expected final path after potential decompression
            # This is a simplification; decompression might result in directories
            if output_path.suffix == '.gz':
                 # Assuming gzip decompresses to file without .gz
                 final_path_before_decompress = output_path
                 expected_final_path = output_path.with_suffix('')
            elif output_path.suffixes[-2:] == ['.tar', '.gz']:
                 # Assuming tar.gz extracts to a directory named after the tarball
                 final_path_before_decompress = output_path
                 # This is a placeholder; actual extracted path needs to be confirmed
                 expected_final_path = output_dir_path / output_path.with_suffix('').with_suffix('').name
            else:
                 final_path_before_decompress = output_path
                 expected_final_path = output_path # No decompression expected

            # Skip if expected final file/directory exists
            if expected_final_path.exists():
                print(f"Expected final file/directory already exists: {expected_final_path}")
                # Add the expected final path to the list of downloaded files
                # This might not be perfectly accurate if tar.gz extracts multiple files
                decompressed_files.append(expected_final_path)
                continue

            # Try downloading
            success = download_file(url, output_path, chunk_size)

            if not success:
                print(f"Failed to download {filename}")
                continue

            # Decompress if necessary
            if output_path.suffix == '.gz' or output_path.suffixes[-2:] == ['.tar', '.gz']:
                try:
                    # Pass the output_dir_path to decompress_file for tar.gz extraction
                    decompressed = decompress_file(output_path, output_path=output_dir_path if output_path.suffixes[-2:] == ['.tar', '.gz'] else None)
                    # Add the decompressed file/directory path to the list
                    decompressed_files.append(decompressed)
                    # Remove compressed file to save space
                    output_path.unlink(missing_ok=True)
                    print(f"Removed compressed file: {output_path}")
                except Exception as e:
                    print(f"Error during decompression: {e}")
                    if output_path.exists():
                        output_path.unlink(missing_ok=True)
                    continue
            else:
                 # If no decompression, the downloaded file is the final file
                 decompressed_files.append(output_path)

    if not decompressed_files:
        # Only raise an error if it was a URL-based download and no files were downloaded
        # For mg_toolkit, the error is raised within download_mgnify_study_data
        if dataset.urls:
             raise Exception(f"Failed to download {name} from any source")
        # If it was an mg_toolkit download and no files were found after running, it might be an issue
        # but the error from run() would be more informative.

    return decompressed_files

def download_mgnify_study_data(study_accession: str, result_group: str, output_dir: Path) -> list[Path]:
    """
    Download sequence data for a MGnify study using the mg_toolkit Python API.
    """
    if not MG_TOOLKIT_AVAILABLE:
        print("mg_toolkit is not installed, cannot download MGnify data.")
        return [] # Or raise an error

    print(f"Downloading {result_group} for MGnify study {study_accession} to {output_dir}")

    # BulkDownloader expects arguments similar to the command line tool
    # We need to create a simple object that mimics the command-line arguments object
    class DownloadArgs:
        def __init__(self, accession, output_path, pipeline=None, result_group=None):
            self.accession = accession
            # Ensure output_path is a string as expected by BulkDownloader
            self.output_path = str(output_path)
            self.pipeline = pipeline
            self.result_group = result_group

    # Pass output_dir as a Path object, convert to string inside DownloadArgs
    args = DownloadArgs(study_accession, output_dir, result_group=result_group)

    downloader = BulkDownloader(
        project_id=args.accession,
        output_path=args.output_path,
        version=args.pipeline, # None for all versions
        result_group=args.result_group
    )

    # The run method doesn't seem to return a list of downloaded files based on a quick scan.
    # It likely downloads files into the output directory structure it creates.
    # We might need to list the files in the output directory after running.
    try:
        # The run method itself prints progress, so no extra tqdm needed here
        downloader.run()
    except Exception as e:
        print(f"Error during mg_toolkit download: {e}")
        raise # Re-raise the exception

    # After running, list the files that were downloaded into the expected structure
    # The structure seems to be output_dir / project_id / pipeline_version / subdir_folder_name / file_name
    # For sequence_data, subdir_folder_name is likely 'sequence_data'.
    # Since pipeline_version is None, it might create subdirectories for each version downloaded.
    # Let's list all files within the project_id directory.
    # Construct the path using Path objects for correct joining
    study_output_dir = output_dir / study_accession
    downloaded_files = []
    if study_output_dir.exists() and study_output_dir.is_dir():
        # Walk through the directory to find all downloaded files
        for root, _, files in os.walk(study_output_dir):
            for file in files:
                downloaded_files.append(Path(root) / file)

    return downloaded_files

def list_datasets():
    """Print information about available datasets."""
    print("\nAvailable datasets:")
    print("-" * 80)
    for key, dataset in DATASETS.items():
        print(f"\nName: {dataset.name} (key: {key})")
        print(f"Description: {dataset.description}")
        if dataset.size_gb:
            print(f"Approximate size: {dataset.size_gb} GB")
        if dataset.study_accession:
             print(f"MGnify Study Accession: {dataset.study_accession}")
        if dataset.result_group:
             print(f"MGnify Result Group: {dataset.result_group}")
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
            # Pass the output directory as a string to download_dataset
            output_paths = download_dataset(args.dataset, args.output)
            print(f"\nDownload completed successfully!")
            print("Files downloaded:")
            for path in output_paths:
                print(f"- {path}")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        parser.print_help()
