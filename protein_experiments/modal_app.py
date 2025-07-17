"""Modal app for distributed pre-training with dataset management."""

import modal
import time
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import requests
from tqdm import tqdm

# Create Modal app
app = modal.App("pe-fasttext-complete")

# Create volumes
dataset_volume = modal.Volume.from_name("pe-fasttext-datasets", create_if_missing=True)
model_volume = modal.Volume.from_name("pe-fasttext-models", create_if_missing=True)
results_volume = modal.Volume.from_name("pe-fasttext-results6", create_if_missing=True)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "numpy==1.26.4",
        "scikit-learn==1.6.1", 
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "datasets==2.20.0",
        "gensim>=4.0.0",
        "pandas>=2.0.0",
        "requests>=2.32.2",
        "biopython>=1.80",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pyarrow>=14.0.0",
        "xgboost>=1.5.0",
        "typing-extensions",
    )
    # Add PE_FastText source files
    .copy_local_file("../src/pe_fasttext/__init__.py", "/pe_fasttext/__init__.py")
    .copy_local_file("../src/pe_fasttext/model.py", "/pe_fasttext/model.py")
    .copy_local_file("../src/pe_fasttext/tokenization.py", "/pe_fasttext/tokenization.py")
    .copy_local_file("../src/pe_fasttext/position_encodings.py", "/pe_fasttext/position_encodings.py")
    .copy_local_file("../src/pe_fasttext/fasttext_utils.py", "/pe_fasttext/fasttext_utils.py")
    .copy_local_file("../src/pe_fasttext/utils.py", "/pe_fasttext/utils.py")
    # Add protein experiments files
    .copy_local_file("src/__init__.py", "/protein_experiments/src/__init__.py")
    .copy_local_file("src/data.py", "/protein_experiments/src/data.py")
    .copy_local_file("src/embedders/__init__.py", "/protein_experiments/src/embedders/__init__.py")
    .copy_local_file("src/embedders/base.py", "/protein_experiments/src/embedders/base.py")
    .copy_local_file("src/embedders/fasttext.py", "/protein_experiments/src/embedders/fasttext.py")
    .copy_local_file("src/embedders/esm2.py", "/protein_experiments/src/embedders/esm2.py")
    .copy_local_file("src/experiment.py", "/protein_experiments/src/experiment.py")
)


def download_file(url: str, output_path: Path, chunk_size: int = 1024*1024) -> bool:
    """Download a file with progress bar."""
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


def decompress_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Decompress a gzip file."""
    print(f"\nDecompressing {input_path}...")
    
    if output_path is None:
        output_path = input_path.with_suffix('')

    if input_path.suffix == '.gz':
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    print(f"Successfully decompressed to {output_path}")
    return output_path


def ensure_uniref50_downloaded(dataset_dir: Path) -> Path:
    """Ensure UniRef50 is downloaded and return path to FASTA file."""
    final_path = dataset_dir / "uniref50.fasta"
    
    # Check if already downloaded
    if final_path.exists():
        print(f"UniRef50 already exists at {final_path}")
        print(f"File size: {final_path.stat().st_size / 1e9:.1f} GB")
        return final_path
    
    # URLs for UniRef50
    urls = [
        "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
        "https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    ]
    
    compressed_path = dataset_dir / "uniref50.fasta.gz"
    
    # Try downloading from each URL
    for url in urls:
        if download_file(url, compressed_path):
            break
    else:
        raise Exception("Failed to download UniRef50 from any source")
    
    # Decompress
    decompress_file(compressed_path, final_path)
    
    # Remove compressed file to save space
    compressed_path.unlink()
    print(f"Removed compressed file: {compressed_path}")
    
    print(f"UniRef50 ready at {final_path}")
    print(f"File size: {final_path.stat().st_size / 1e9:.1f} GB")
    
    return final_path


@app.function(
    image=image,
    timeout=14400,  # 4 hours
    volumes={"/datasets": dataset_volume},
)
def decompress_uniref50():
    """Decompress UniRef50."""
    dataset_dir = Path("/datasets")
    compressed = dataset_dir / "uniref50.fasta.gz"
    decompressed = dataset_dir / "uniref50.fasta"
    
    if not compressed.exists():
        raise FileNotFoundError(f"No compressed file at {compressed}")
    
    print(f"Decompressing {compressed.stat().st_size / 1e9:.1f}GB file...")
    start = time.time()
    
    with gzip.open(compressed, 'rb') as f_in:
        with open(decompressed, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=16*1024*1024)
    
    print(f"Done in {(time.time() - start)/60:.1f}min, {decompressed.stat().st_size / 1e9:.1f}GB")
    dataset_volume.commit()
    return str(decompressed)


@app.function(
    image=image,
    timeout=7200,  # 2 hours for download
    volumes={"/datasets": dataset_volume},
)
def download_uniref50():
    """Download and prepare UniRef50 dataset."""
    dataset_dir = Path("/datasets")
    dataset_dir.mkdir(exist_ok=True)
    
    final_path = ensure_uniref50_downloaded(dataset_dir)
    
    # Commit to volume
    dataset_volume.commit()
    
    return str(final_path)


def stream_fasta_sequences(fasta_path: Path, max_sequences: Optional[int] = None, 
                          sample_rate: int = 1):
    """Stream sequences from a FASTA file."""
    count = 0
    current_seq: List[str] = []
    
    with open(fasta_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            
            if line.startswith('>'):
                # Process previous sequence if exists
                if current_seq and idx % sample_rate == 0:
                    sequence = ''.join(current_seq)
                    if sequence:
                        yield sequence
                        count += 1
                        if max_sequences and count >= max_sequences:
                            return
                
                # Start new sequence
                current_seq = []
            else:
                # Add to current sequence
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_seq and count < (max_sequences or float('inf')):
            sequence = ''.join(current_seq)
            if sequence:
                yield sequence


@app.function(
    image=image,
    gpu="T4",
    timeout=86400,  # 24 hours
    volumes={
        "/datasets": dataset_volume,
        "/models": model_volume,
    },
)
def pretrain_uniref50_fasta(
    output_name: str = "uniref50_pretrained_10pct_kmer.bin",
    train_split: float = 0.1,
    max_sequences: int = 100000000000000000,
    tokenization: str = "kmer",
    k: int = 5,
    dim: int = 128,
    epochs: int = 10,
    window: int = 5,
    min_count: int = 5,
):
    """Pre-train FastText on UniRef50 from FASTA file."""
    import sys
    sys.path.insert(0, "/")
    sys.path.insert(0, "/protein_experiments")
    
    from pe_fasttext.fasttext_utils import train_fasttext
    from pathlib import Path
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # First ensure UniRef50 is downloaded
    dataset_dir = Path("/datasets")
    dataset_dir.mkdir(exist_ok=True)
    fasta_path = ensure_uniref50_downloaded(dataset_dir)
    
    # Commit dataset volume if we downloaded
    if not fasta_path.exists():
        dataset_volume.commit()
    
    output_path = Path(f"/models/{output_name}")
    
    if output_path.exists():
        print(f"Pre-trained model already exists at {output_path}")
        return str(output_path)
    
    print(f"Pre-training on {train_split*100}% of UniRef50")
    print(f"Max sequences: {max_sequences:,}")
    print(f"Config: tokenization={tokenization}, k={k}, dim={dim}, epochs={epochs}")
    
    start_time = time.time()
    
    # Calculate sampling rate based on train_split
    sample_rate = int(1.0 / train_split) if train_split < 1.0 else 1
    
    # Stream and tokenize sequences
    sequences = stream_fasta_sequences(fasta_path, max_sequences, sample_rate)
    
    # Tokenize sequences
    tokenized = []
    
    if tokenization == "residue":
        print("Using residue-level tokenization")
        for seq in tqdm(sequences, desc="Tokenizing sequences", total=max_sequences):
            tokenized.append(list(seq))
    else:
        print(f"Using {k}-mer tokenization")
        for seq in tqdm(sequences, desc="Tokenizing sequences", total=max_sequences):
            tokens = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            if tokens:  # Skip sequences shorter than k
                tokenized.append(tokens)
    
    print(f"Training on {len(tokenized)} sequences")
    print(f"Total tokens: {sum(len(seq) for seq in tokenized):,}")
    
    # Create epoch callback to save model after each epoch
    from gensim.models.callbacks import CallbackAny2Vec
    
    class EpochSaver(CallbackAny2Vec):
        def __init__(self, output_path):
            self.output_path = Path(output_path)
            self.epoch = 0
            
        def on_epoch_end(self, model):
            self.epoch += 1
            epoch_path = self.output_path.with_suffix(f'.epoch{self.epoch}.bin')
            model.save(str(epoch_path))
            print(f"Saved epoch {self.epoch} model to {epoch_path}")
            model_volume.commit()
    
    # Train model
    from pe_fasttext.fasttext_utils import LossLogger
    from gensim.models.fasttext import FastText
    
    model = FastText(
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=4
    )
    
    model.build_vocab(tokenized)
    model.train(
        tokenized,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[LossLogger(), EpochSaver(output_path)],
    )
    
    # Save final model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Pre-trained model saved to {output_path}")
    
    # Commit to volume
    model_volume.commit()
    
    elapsed = time.time() - start_time
    print(f"Pre-training completed in {elapsed/3600:.1f} hours")
    
    return {
        "model_path": str(output_path),
        "time_hours": elapsed / 3600,
        "sequences": len(tokenized),
    }


def _run_experiment_logic(config: Dict[str, Any]) -> Dict[str, Any]:
    """Logic for running a single experiment."""
    import sys
    sys.path.insert(0, "/")
    sys.path.insert(0, "/protein_experiments")
    
    from src.experiment import Experiment
    import json
    from pathlib import Path
    import traceback
    import torch

    try:
        # Dynamically set device for ESM2
        if config['embedder']['type'] == 'esm2':
            if torch.cuda.is_available():
                config['embedder']['device'] = 'cuda'
            else:
                config['embedder']['device'] = 'cpu'
                print("WARNING: No GPU available for ESM2, using CPU. This will be slow.")

        # Update embedder model path before initializing Experiment
        if "pretrained_path" in config.get("embedder", {}):
            pretrained_name = Path(config['embedder']['pretrained_path']).name
            config["embedder"]["model_path"] = f"/models/{pretrained_name}"
        else:
            # For training from scratch, set a unique path in the results volume
            config["embedder"]["model_path"] = f"/results/runs/{config['task']}_{config['embedder']['type']}_{config.get('id', 'exp')}/model.bin"

        # Update paths to use Modal volumes
        if "pretrained_path" in config.get("embedder", {}):
            pretrained_name = Path(config['embedder']['pretrained_path']).name
            config["embedder"]["pretrained_path"] = f"/models/{pretrained_name}"
            
            # Check if pretrained model exists
            pretrained_path = Path(config["embedder"]["pretrained_path"])
            if not pretrained_path.exists():
                print(f"WARNING: Pretrained model not found at {pretrained_path}")
                print("Will attempt to train from scratch...")
                # Remove pretrained_path to train from scratch
                del config["embedder"]["pretrained_path"]
        
        # Set output directory with task name
        config["output_dir"] = f"/results/runs/{config['task']}_{config['embedder']['type']}_{config.get('id', 'exp')}"
        
        print(f"Running experiment: {config['task']} with {config['embedder']['type']}")
        
        experiment = Experiment(config, dataset_volume=dataset_volume)
        results = experiment.run()
        
        # Create results directory structure
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config for reference
        config_file = output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save a summary file at the root for easy access
        summary_dir = Path("/results/summary")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / f"summary_{config['task']}_{config['embedder']['type']}_{config.get('id', 'exp')}.json"
        summary = {
            "config": config,
            "results": results,
            "output_dir": str(output_dir)
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}")
        print(f"Summary saved to {summary_file}")
        
        # Commit results
        results_volume.commit()
        
        return results
    except Exception as e:
        print(f"--- EXPERIMENT FAILED ---")
        print(f"Config ID: {config.get('id', 'N/A')}")
        print(f"Task: {config.get('task', 'N/A')}, Embedder: {config.get('embedder', {}).get('type', 'N/A')}")
        print(f"Error: {e}")
        
        error_result = {
            "config": config,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        # Save error summary to results volume for debugging
        output_dir = Path(f"/results/runs/{config['task']}_{config['embedder']['type']}_{config.get('id', 'exp')}_ERROR")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = output_dir / "error.json"
        with open(error_file, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        results_volume.commit()
        print(f"Saved error details to {error_file}")

        return error_result


@app.function(
    image=image,
    timeout=86400,
    volumes={
        "/models": model_volume,
        "/results": results_volume,
        "/datasets": dataset_volume,
    },
)
def run_experiment_cpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment on a CPU."""
    return _run_experiment_logic(config)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize experiment results."""
    summary: Dict[str, Any] = {
        "total_experiments": len(results),
        "by_task": {},
        "by_embedder": {},
        "best_performers": {}
    }
    
    for result in results:
        if "error" in result:
            continue
            
        task = result["task"]
        embedder = result["embedder"]["type"]
        
        # Track by task
        if task not in summary["by_task"]:
            summary["by_task"][task] = []
        summary["by_task"][task].append({
            "embedder": embedder,
            "r2": result.get("r2", result.get("accuracy", 0)),
            "time": result["embedding_time"]
        })
        
        # Track by embedder
        if embedder not in summary["by_embedder"]:
            summary["by_embedder"][embedder] = []
        summary["by_embedder"][embedder].append({
            "task": task,
            "r2": result.get("r2", result.get("accuracy", 0))
        })
    
    # Find best performers per task
    for task, results in summary["by_task"].items():
        best = max(results, key=lambda x: x["r2"])
        summary["best_performers"][task] = best
    
    return summary


def generate_all_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all experiment configurations for protein experiments."""
    
    # --- Residue-based models ---
    residue_pretrained_model = "uniref50_pretrained_full_residue.bin"
    kmer_pretrained_model = "uniref50_pretrained_10pct_kmer.epoch6.bin"
    
    # All parameter combinations
    fine_tuning = [True, False]
    pos_encoders = ["sinusoid", "learned", "rope", "alibi", "ft_alibi", None]  # None for baseline
    fusions = ["add", "concatenate"]
    predictors = ["rf", "mlp", "xgboost"]
    tasks = ["fluorescence", "stability", "mpp", "beta_lactamase_complete", "ssp", "deeploc"]
    seeds = [42, 123, 456, 789, 1011]
    
    configs = []
    config_id = 0
    
    # FastText experiments (residue-based)
    for fine_tune in fine_tuning:
        for pos_encoder in pos_encoders:
            # Skip fusion variations for baseline (no pos_encoder)
            fusion_options = [None] if pos_encoder is None else fusions
            
            for fusion in fusion_options:
                for predictor_type in predictors:
                    for task in tasks:
                        for seed in seeds:
                            config_id += 1
                            
                            embedder_config = {
                                "type": "fasttext",
                                "tokenization": "residue",
                                "dim": 128,
                                "pretrained_path": residue_pretrained_model,
                                "fine_tune": fine_tune,
                                "save_model": False,
                                "train_params": {"epochs": 5} if fine_tune else {}
                            }
                            
                            # Add positional encoding if specified
                            if pos_encoder:
                                embedder_config["pos_encoder"] = pos_encoder
                                embedder_config["fusion"] = fusion
                            
                            config = {
                                "id": config_id,
                                "task": task,
                                "embedder": embedder_config,
                                "predictor": {
                                    "type": predictor_type,
                                    "seed": seed
                                }
                            }
                            
                            configs.append(config)

    # FastText experiments (k-mer based)
    for fine_tune in fine_tuning:
        for pos_encoder in pos_encoders:
            # Skip fusion variations for baseline (no pos_encoder)
            fusion_options = [None] if pos_encoder is None else fusions
            
            for fusion in fusion_options:
                for predictor_type in predictors:
                    for task in tasks:
                        # SSP is a per-residue task and not compatible with k-mer tokenization
                        if task == 'ssp':
                            continue

                        for seed in seeds:
                            config_id += 1
                            
                            embedder_config = {
                                "type": "fasttext",
                                "tokenization": "kmer",
                                "k": 5,
                                "dim": 128,
                                "pretrained_path": kmer_pretrained_model,
                                "fine_tune": fine_tune,
                                "save_model": False,
                                "train_params": {"epochs": 5} if fine_tune else {}
                            }
                            
                            # Add positional encoding if specified
                            if pos_encoder:
                                embedder_config["pos_encoder"] = pos_encoder
                                embedder_config["fusion"] = fusion
                            
                            config = {
                                "id": config_id,
                                "task": task,
                                "embedder": embedder_config,
                                "predictor": {
                                    "type": predictor_type,
                                    "seed": seed
                                }
                            }
                            
                            configs.append(config)
    
    # ESM2 experiments (no positional encoding variations)
    esm_models = [
        "facebook/esm2_t6_8M_UR50D",    # 8M parameters
    ]
    
    for esm_model in esm_models:
        for predictor_type in predictors:
            for task in tasks:
                for seed in seeds:
                    config_id += 1
                    
                    config = {
                        "id": config_id,
                        "task": task,
                        "embedder": {
                            "type": "esm2",
                            "model_name": esm_model,
                            "device": "cuda",  # Use GPU if available: "cuda"
                            "batch_size": 32,
                            "max_length": 1024  # ESM2 max length
                        },
                        "predictor": {
                            "type": predictor_type,
                            "seed": seed
                        }
                    }
                    
                    configs.append(config)
    
    return configs


@app.function(
    image=image,
    timeout=86400,
    volumes={"/models": model_volume, "/results": results_volume},
)
def run_ssp_experiments():
    """Run all SSP experiments on CPU."""
    configs = generate_all_experiment_configs()
    ssp_configs = [c for c in configs if c['task'] == 'ssp']
    
    print(f"Running {len(ssp_configs)} SSP experiments in parallel on CPU...")
    results = list(run_experiment_cpu.map(ssp_configs))
    
    summary = summarize_results(results)
    with open("/results/ssp_results.json", 'w') as f:
        json.dump({"experiments": results, "summary": summary}, f, indent=2)
    
    results_volume.commit()
    print(f"Completed {len(results)} SSP experiments!")
    return summary


@app.function(
    image=image,
    timeout=86400,
    volumes={"/models": model_volume, "/results": results_volume},
)
def run_deeploc_experiments():
    """Run all DeepLoc experiments on CPU."""
    configs = generate_all_experiment_configs()
    deeploc_configs = [c for c in configs if c['task'] == 'deeploc']
    
    print(f"Running {len(deeploc_configs)} DeepLoc experiments in parallel on CPU...")
    results = list(run_experiment_cpu.map(deeploc_configs))
    
    summary = summarize_results(results)
    with open("/results/deeploc_results.json", 'w') as f:
        json.dump({"experiments": results, "summary": summary}, f, indent=2)
    
    results_volume.commit()
    print(f"Completed {len(results)} DeepLoc experiments!")
    return summary


@app.function(
    image=image,
    timeout=86400,
    volumes={"/models": model_volume, "/results": results_volume},
)
def run_esm2_experiments():
    """Run all ESM2 experiments on CPU."""
    configs = generate_all_experiment_configs()
    esm2_configs = [c for c in configs if c['embedder']['type'] == 'esm2']
    
    print(f"Running {len(esm2_configs)} ESM2 experiments in parallel on CPU...")
    results = list(run_experiment_cpu.map(esm2_configs))
    
    summary = summarize_results(results)
    with open("/results/esm2_results.json", 'w') as f:
        json.dump({"experiments": results, "summary": summary}, f, indent=2)
    
    results_volume.commit()
    print(f"Completed {len(results)} ESM2 experiments!")
    return summary
    

@app.function(
    image=image,
    timeout=86400,  # 24 hours
    volumes={"/models": model_volume, "/results": results_volume},
)
def run_all_experiments():
    """
    Run all experiments, but intelligently skip those that have already
    completed successfully.
    """
    from pathlib import Path

    # 1. Identify already completed experiments by checking directory names
    results_run_dir = Path("/results/runs")
    completed_runs = set()
    if results_run_dir.exists():
        for run_dir in results_run_dir.iterdir():
            # A successful run is a directory that doesn't end with _ERROR
            if run_dir.is_dir() and not run_dir.name.endswith("_ERROR"):
                completed_runs.add(run_dir.name)

    print(f"Found {len(completed_runs)} successfully completed experiments. These will be skipped.")

    # 2. Filter configurations to find experiments that need to be run
    all_configs = generate_all_experiment_configs()
    configs_to_run = []
    for config in all_configs:
        # Recreate the expected output directory name for each config
        run_name = f"{config['task']}_{config['embedder']['type']}_{config.get('id', 'exp')}"
        if run_name not in completed_runs:
            configs_to_run.append(config)
    
    total_possible = len(all_configs)
    print(f"Total experiments to run now: {len(configs_to_run)} out of {total_possible} total possible experiments.")

    if not configs_to_run:
        print("All experiments have already completed successfully. Nothing to do.")
        return {"status": "completed", "message": "All experiments finished."}
    
    print(f"Running all {len(configs_to_run)} experiments on CPU.")
    
    # Run all experiments on CPU
    results = list(run_experiment_cpu.map(configs_to_run))
    
    summary = summarize_results(results)
    
    print("Note: /results/all_results.json will be overwritten with the summary of this run's experiments.")
    with open("/results/all_results.json", 'w') as f:
        json.dump({"experiments": results, "summary": summary}, f, indent=2)
    
    results_volume.commit()
    print(f"Completed {len(results)} experiments!")
    return summary


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=86400

)
def create_summary_parquet():
    """Create a single Parquet file from all summary JSONs."""
    import pandas as pd
    from pathlib import Path
    import json

    summary_dir = Path("/results/summary")
    if not summary_dir.exists():
        print("No summary directory found. Nothing to do.")
        return

    all_results = []
    summary_files = list(summary_dir.glob("summary_*.json"))
    print(f"Found {len(summary_files)} summary files to process.")

    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)

            # Skip failed experiments
            if "error" in data:
                continue

            config = data.get("config", {})
            results = data.get("results", {})
            embedder_config = config.get("embedder", {})
            predictor_config = config.get("predictor", {})
            val_metrics = results.get("val_metrics", {})

            # Common fields for all rows generated from this summary file
            base_result = {
                "id": config.get("id"),
                "task": config.get("task"),
                "embedder_type": embedder_config.get("type"),
                "tokenization": embedder_config.get("tokenization"),
                "k": embedder_config.get("k"),
                "pos_encoder": embedder_config.get("pos_encoder"),
                "fusion": embedder_config.get("fusion"),
                "fine_tune": embedder_config.get("fine_tune"),
                "predictor_type": predictor_config.get("type"),
                "seed": predictor_config.get("seed"),
                "val_r2": val_metrics.get("r2"),
                "val_mse": val_metrics.get("mse"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_f1": val_metrics.get("f1"),
                "embedding_time": results.get("embedding_time"),
                "train_time": results.get("train_time"),
            }

            test_metrics = results.get("test_metrics")
            
            # If test_metrics is a dictionary, iterate over each test split (e.g., for SSP)
            if isinstance(test_metrics, dict) and test_metrics:
                for split_name, metrics in test_metrics.items():
                    flat_result = base_result.copy()
                    flat_result.update({
                        "test_split": split_name,
                        "test_r2": metrics.get("r2"),
                        "test_mse": metrics.get("mse"),
                        "test_accuracy": metrics.get("accuracy"),
                        "test_f1": metrics.get("f1"),
                    })
                    all_results.append(flat_result)
            else:
                # Fallback for old format or tasks with a single test set
                flat_result = base_result.copy()
                flat_result.update({
                    "test_split": "test",  # Assume 'test' if not otherwise specified
                    "test_r2": results.get("r2"),
                    "test_mse": results.get("mse"),
                    "test_accuracy": results.get("accuracy"),
                    "test_f1": results.get("f1"),
                })
                all_results.append(flat_result)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Could not process {summary_file}: {e}")

    if not all_results:
        print("No valid results found to create Parquet file.")
        return

    df = pd.DataFrame(all_results)
    
    # Define output path
    output_path = Path("/results/results_summary.parquet")
    df.to_parquet(output_path, index=False)
    
    results_volume.commit()
    print(f"Successfully created summary Parquet file at {output_path}")


@app.function(
    image=image,
    volumes={
        "/datasets": dataset_volume,
        "/models": model_volume,
        "/results": results_volume,
    },
)
def status():
    """Check status of volumes and files."""
    print("=== PE-FastText Modal Status ===\n")
    
    # Check datasets
    print("Datasets Volume:")
    dataset_dir = Path("/datasets")
    if dataset_dir.exists():
        for file in dataset_dir.iterdir():
            size_gb = file.stat().st_size / 1e9 if file.is_file() else 0
            print(f"  - {file.name}: {size_gb:.1f} GB")
    else:
        print("  - Empty")
    
    # Check models
    print("\nModels Volume:")
    model_dir = Path("/models")
    if model_dir.exists():
        for file in model_dir.iterdir():
            size_mb = file.stat().st_size / 1e6 if file.is_file() else 0
            print(f"  - {file.name}: {size_mb:.1f} MB")
    else:
        print("  - Empty")
    
    # Check results
    print("\nResults Volume:")
    results_dir = Path("/results")
    if results_dir.exists():
        for file in results_dir.iterdir():
            print(f"  - {file.name}")
    else:
        print("  - Empty")
    
    return {
        "datasets": list(dataset_dir.iterdir()) if dataset_dir.exists() else [],
        "models": list(model_dir.iterdir()) if model_dir.exists() else [],
        "results": list(results_dir.iterdir()) if results_dir.exists() else [],
    }


@app.local_entrypoint()
def create_summary():
    """Local entrypoint to trigger the Parquet summary creation."""
    print("Triggering summary creation...")
    create_summary_parquet.remote()
    print("Summary creation job sent. Check the Modal UI for progress.")


if __name__ == "__main__":
    # Example usage:
    # modal run modal_app.py::run_all_experiments
    # modal run modal_app.py::run_ssp_experiments
    # modal run modal_app.py::run_esm2_experiments
    # modal run modal_app.py::status
    pass