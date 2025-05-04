"""Ablation grid runner for PE-FastText."""
import os
import yaml  # type: ignore


def run_ablation(config_path: str = "config/ablation.yaml", out_dir: str = "results") -> None:
    """Run ablation experiments as defined in a YAML config."""
    print(f"[ablation] Loading config: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    os.makedirs(out_dir, exist_ok=True)
    # TODO: implement ablation logic
    print(f"[ablation] Output directory: {out_dir}")
    print(f"[ablation] (stub) Ablation completed.") 