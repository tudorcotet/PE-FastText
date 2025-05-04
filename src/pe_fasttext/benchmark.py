"""Unsupervised and supervised benchmark runner for PE-FastText."""
import os
import yaml  # type: ignore


def run_benchmarks(config_path: str = "config/default.yaml", out_dir: str = "results") -> None:
    """Run suite of benchmarks as defined in a YAML config."""
    print(f"[benchmark] Loading config: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    # TODO: implement unsupervised and supervised benchmarks
    os.makedirs(out_dir, exist_ok=True)
    print(f"[benchmark] Output directory: {out_dir}")
    # ... add tasks from config
    print(f"[benchmark] (stub) Benchmarks completed.") 