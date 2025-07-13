#!/usr/bin/env python
"""Simple wrapper to run Modal tasks."""

import argparse
import subprocess
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run experiments on Modal')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Pre-train command
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-train on UniRef50')
    pretrain_parser.add_argument('--split', type=float, default=0.01,
                                help='Fraction of UniRef50 (default: 0.01 = 1%)')
    pretrain_parser.add_argument('--sequences', type=int,
                                help='Max sequences (overrides split)')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run experiments')
    exp_parser.add_argument('--config', type=Path,
                           help='Config file for single experiment')
    exp_parser.add_argument('--compare', action='store_true',
                           help='Run full comparison')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy Modal app')
    
    args = parser.parse_args()
    
    if args.command == 'pretrain':
        print(f"🚀 Starting pre-training on {args.split*100}% of UniRef50...")
        cmd = [
            "modal", "run", "--detach",
            "modal_app.py::pretrain_uniref50"
        ]
        subprocess.run(cmd)
        
    elif args.command == 'experiment':
        if args.config:
            print(f"🧪 Running experiment from {args.config}")
            cmd = [
                "modal", "run", "modal_app.py",
                "--mode", "experiment", 
                "--config-file", str(args.config)
            ]
        else:
            print("🔬 Running comparison experiments...")
            cmd = ["modal", "run", "modal_app.py", "--mode", "experiment"]
        subprocess.run(cmd)
        
    elif args.command == 'deploy':
        print("📦 Deploying Modal app...")
        subprocess.run(["modal", "deploy", "modal_app.py"])
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()