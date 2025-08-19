#!/usr/bin/env python
"""Main entry point for running protein embedding experiments."""

import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.experiment import Experiment, ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_default_embedder_configs():
    """Get default embedder configurations for comparison."""
    return [
        # FastText variants with k-mer tokenization
        {'type': 'fasttext', 'model_path': 'models/fasttext_kmer.bin', 
         'tokenization': 'kmer', 'k': 6, 'dim': 128},
        
        {'type': 'fasttext', 'model_path': 'models/fasttext_kmer_sin.bin',
         'tokenization': 'kmer', 'k': 6, 'dim': 128, 
         'pos_encoder': 'sinusoid', 'fusion': 'add'},
        
        {'type': 'fasttext', 'model_path': 'models/fasttext_kmer_rope.bin',
         'tokenization': 'kmer', 'k': 6, 'dim': 128,
         'pos_encoder': 'rope', 'fusion': 'add'},
        
        # FastText variants with residue tokenization
        {'type': 'fasttext', 'model_path': 'models/fasttext_residue.bin',
         'tokenization': 'residue', 'dim': 128},
        
        {'type': 'fasttext', 'model_path': 'models/fasttext_residue_sin.bin',
         'tokenization': 'residue', 'dim': 128,
         'pos_encoder': 'sinusoid', 'fusion': 'add'},
        
        # ESM2 (if available)
        {'type': 'esm2', 'model_name': 'facebook/esm2_t6_8M_UR50D',
         'device': 'cpu', 'batch_size': 32}
    ]


def main():
    parser = argparse.ArgumentParser(description='Run protein embedding experiments')
    
    # Modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=Path, 
                           help='Run from config file')
    mode_group.add_argument('--compare', action='store_true',
                           help='Run comparison across all tasks and embedders')
    mode_group.add_argument('--single', action='store_true',
                           help='Run single experiment with CLI args')
    
    # Single experiment args
    parser.add_argument('--task', default='fluorescence',
                       choices=['fluorescence', 'stability', 'mpp', 
                               'beta_lactamase_complete', 'ssp'],
                       help='Task to run')
    parser.add_argument('--embedder', default='fasttext',
                       choices=['fasttext', 'esm2'],
                       help='Embedder type')
    parser.add_argument('--pos-encoder', 
                       choices=['sinusoid', 'learned', 'rope', 'alibi', 'ft_alibi'],
                       help='Position encoder type (FastText only)')
    parser.add_argument('--tokenization', default='kmer',
                       choices=['kmer', 'residue'],
                       help='Tokenization method (FastText only)')
    parser.add_argument('--predictor', default='rf',
                       choices=['rf', 'linear'],
                       help='Predictor type')
    
    # Data args
    parser.add_argument('--train-size', type=float, default=0.5,
                       help='Fraction of training data to use')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Fraction of train for validation')
    parser.add_argument('--max-length', type=int,
                       help='Maximum sequence length')
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Output directory')
    parser.add_argument('--save-results', type=Path,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.config:
        # Run from config file
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        
        experiment = Experiment(config)
        results = experiment.run()
        
    elif args.compare:
        # Run comparison
        logger.info("Running comparison experiments")
        
        base_config = {
            'predictor': {'type': 'rf', 'n_estimators': 100},
            'data': {
                'train_size': args.train_size,
                'val_split': args.val_split,
                'max_length': args.max_length
            },
            'output_dir': args.output_dir
        }
        
        runner = ExperimentRunner(base_config)
        
        # Get tasks and embedder configs
        tasks = ['fluorescence', 'stability', 'mpp', 'beta_lactamase_complete']
        embedder_configs = get_default_embedder_configs()
        
        # Run comparison
        results = runner.run_comparison(tasks, embedder_configs)
        runner.summarize_results()
        
    else:
        # Run single experiment
        logger.info(f"Running single experiment: {args.task} with {args.embedder}")
        
        # Build embedder config
        embedder_config = {'type': args.embedder}
        
        if args.embedder == 'fasttext':
            embedder_config.update({
                'model_path': f'models/{args.task}_{args.embedder}_{args.tokenization}.bin',
                'tokenization': args.tokenization,
                'k': 6 if args.tokenization == 'kmer' else None,
                'dim': 128,
                'train_params': {'epochs': 10, 'window': 5}
            })
            
            if args.pos_encoder:
                embedder_config.update({
                    'pos_encoder': args.pos_encoder,
                    'fusion': 'add'
                })
        
        # Build full config
        config = {
            'task': args.task,
            'embedder': embedder_config,
            'predictor': {'type': args.predictor},
            'data': {
                'task': args.task,
                'train_size': args.train_size,
                'val_split': args.val_split,
                'max_length': args.max_length
            },
            'output_dir': args.output_dir
        }
        
        experiment = Experiment(config)
        results = experiment.run()
    
    # Save results if requested
    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.save_results}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())