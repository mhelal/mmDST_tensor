#!/usr/bin/env python3
"""
run_all.py - Run all experiments with 5 seeds for reproducibility
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

SEEDS = [42, 123, 456, 789, 101112]

def run_experiment(script, seed, output_dir):
    """Run a single experiment with given seed."""
    cmd = f"python {script} --seed {seed} --output {output_dir}/seed_{seed}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    experiments = [
        ("scaling", "scaling/run_scaling.py"),
        ("ablation", "ablation/band_sensitivity.py"),
        ("validation_global", "validation/balibase_ablation.py"),
        ("validation_local", "validation/local_ablation.py"),
    ]
    
    all_results = {}
    
    for exp_name, script in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name} with {len(SEEDS)} seeds")
        print('='*60)
        
        exp_dir = results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        for seed in SEEDS:
            print(f"\nSeed: {seed}")
            result = run_experiment(script, seed, exp_dir)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
    
    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()
