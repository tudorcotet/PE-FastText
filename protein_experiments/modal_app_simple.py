"""Modal app for distributed pre-training and experiments."""

import modal
import time
from pathlib import Path
from typing import Dict, List, Any
import json

# Create Modal app
app = modal.App("pe-fasttext-experiments")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0", 
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "datasets>=2.0.0",
        "gensim>=4.0.0",
        "pandas>=2.0.0",
    )
    # Copy necessary Python files directly
    .copy_local_file("../src/pe_fasttext/__init__.py", "/pe_fasttext/__init__.py")
    .copy_local_file("../src/pe_fasttext/model.py", "/pe_fasttext/model.py")
    .copy_local_file("../src/pe_fasttext/tokenization.py", "/pe_fasttext/tokenization.py")
    .copy_local_file("../src/pe_fasttext/position_encodings.py", "/pe_fasttext/position_encodings.py")
    .copy_local_file("../src/pe_fasttext/fasttext_utils.py", "/pe_fasttext/fasttext_utils.py")
    .copy_local_file("../src/pe_fasttext/utils.py", "/pe_fasttext/utils.py")
    .copy_local_file("../src/pe_fasttext/config.py", "/pe_fasttext/config.py")
    # Copy protein experiments files
    .copy_local_file("src/__init__.py", "/protein_experiments/src/__init__.py")
    .copy_local_file("src/data.py", "/protein_experiments/src/data.py")
    .copy_local_file("src/embedders/__init__.py", "/protein_experiments/src/embedders/__init__.py")
    .copy_local_file("src/embedders/base.py", "/protein_experiments/src/embedders/base.py")
    .copy_local_file("src/embedders/fasttext.py", "/protein_experiments/src/embedders/fasttext.py")
    .copy_local_file("src/embedders/esm2.py", "/protein_experiments/src/embedders/esm2.py")
    .copy_local_file("src/experiment.py", "/protein_experiments/src/experiment.py")
    .copy_local_file("src/pretrain.py", "/protein_experiments/src/pretrain.py")
)

# Create volumes for model storage
model_volume = modal.Volume.from_name("pe-fasttext-models", create_if_missing=True)
results_volume = modal.Volume.from_name("pe-fasttext-results", create_if_missing=True)


@app.function(
    image=image,
    gpu=None,  # Add GPU if needed: gpu="T4"
    timeout=86400,  # 24 hours
    volumes={
        "/models": model_volume,
        "/results": results_volume,
    },
)
def pretrain_uniref50(
    output_name: str = "uniref50_pretrained.bin",
    train_split: float = 0.01,
    max_sequences: int = 500000,
    tokenization: str = "kmer",
    k: int = 5,
    dim: int = 128,
    epochs: int = 10,
):
    """Pre-train FastText on UniRef50."""
    import sys
    sys.path.insert(0, "/")
    sys.path.insert(0, "/protein_experiments")
    
    from src.pretrain import pretrain_fasttext
    from pathlib import Path
    
    output_path = Path(f"/models/{output_name}")
    
    print(f"Pre-training on {train_split*100}% of UniRef50")
    print(f"Max sequences: {max_sequences:,}")
    
    start_time = time.time()
    
    model_path = pretrain_fasttext(
        output_path=output_path,
        tokenization=tokenization,
        k=k,
        dim=dim,
        epochs=epochs,
        train_split=train_split,
        max_sequences=max_sequences,
        window=5,
        min_count=5,
    )
    
    elapsed = time.time() - start_time
    print(f"Pre-training completed in {elapsed/3600:.1f} hours")
    
    # Commit to volume
    model_volume.commit()
    
    return {
        "model_path": str(model_path),
        "time_hours": elapsed / 3600,
        "sequences": max_sequences,
    }


@app.function(
    image=image,
    timeout=3600,  # 1 hour per experiment
    volumes={
        "/models": model_volume,
        "/results": results_volume,
    },
)
def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment."""
    import sys
    sys.path.insert(0, "/")
    sys.path.insert(0, "/protein_experiments")
    
    from src.experiment import Experiment
    import json
    from pathlib import Path
    
    # Update paths to use Modal volumes
    if "pretrained_path" in config.get("embedder", {}):
        config["embedder"]["pretrained_path"] = f"/models/{Path(config['embedder']['pretrained_path']).name}"
    
    config["output_dir"] = "/results"
    
    print(f"Running experiment: {config['task']} with {config['embedder']['type']}")
    
    experiment = Experiment(config)
    results = experiment.run()
    
    # Save results
    result_file = f"/results/{config['task']}_{config['embedder']['type']}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Commit results
    results_volume.commit()
    
    return results


@app.function(
    image=image,
    timeout=10800,  # 3 hours
    volumes={
        "/models": model_volume,
        "/results": results_volume,
    },
)
def run_experiment_batch(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run multiple experiments in parallel."""
    # Use Modal's parallel map
    results = list(run_experiment.map(configs))
    
    # Save combined results
    all_results = {
        "experiments": results,
        "summary": summarize_results(results)
    }
    
    with open("/results/batch_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    results_volume.commit()
    
    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize experiment results."""
    summary = {
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


# Standalone functions for direct invocation


def generate_comparison_configs() -> List[Dict[str, Any]]:
    """Generate configs for comparison experiments."""
    tasks = ["fluorescence", "stability", "mpp", "beta_lactamase_complete"]
    
    embedder_configs = [
        {"type": "fasttext", "tokenization": "kmer", "k": 6, "dim": 128},
        {"type": "fasttext", "tokenization": "kmer", "k": 6, "dim": 128, 
         "pos_encoder": "sinusoid", "fusion": "add"},
        {"type": "fasttext", "tokenization": "kmer", "k": 6, "dim": 128,
         "pos_encoder": "rope", "fusion": "add"},
        {"type": "fasttext", "tokenization": "residue", "dim": 128},
    ]
    
    configs = []
    for task in tasks:
        for embedder in embedder_configs:
            config = {
                "task": task,
                "embedder": embedder.copy(),
                "predictor": {"type": "rf", "n_estimators": 100, "max_depth": 20},
                "data": {"train_size": 0.5, "val_split": 0.1, "max_length": 500},
            }
            
            # Add model paths
            pos_enc = embedder.get("pos_encoder", "none")
            tokenization = embedder.get("tokenization", "kmer")
            config["embedder"]["model_path"] = f"models/{task}_{tokenization}_{pos_enc}.bin"
            
            configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Example usage:
    # modal run modal_app_simple.py::pretrain_uniref50
    # modal run modal_app_simple.py::run_experiment_batch
    pass