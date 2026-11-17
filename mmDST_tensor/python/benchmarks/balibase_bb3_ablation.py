#!/usr/bin/env python3
"""
balibase_bb3_ablation.py - BAliBASE 3.0 ablation from bb3_release
"""

import sys
import csv
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/mnt/d/mmDst/code')
from core import fill

def parse_fasta_alignment(aln_file):
    """Parse FASTA alignment file."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(aln_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
    
    if current_id:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences

def compute_sp_score(rows):
    """Compute SP score for an alignment."""
    if not rows:
        return 0
    
    k = len(rows)
    L = len(rows[0])
    score = 0
    
    for col in range(L):
        for i in range(k):
            for j in range(i+1, k):
                a, b = rows[i][col], rows[j][col]
                if a == '-' or b == '-':
                    score += -6
                elif a == b:
                    score += 3
                else:
                    score += -2
    return score

def extract_sequences(rows):
    """Extract ungapped sequences from alignment rows."""
    return [''.join([c for c in row if c != '-']) for row in rows]

def get_sequence_lengths(rows):
    """Get lengths of ungapped sequences."""
    return [len(''.join([c for c in row if c != '-'])) for row in rows]

def run_balibase_experiment(aln_file, max_k=4, max_L=30, description=""):
    """Run ablation on a single BAliBASE case."""
    
    try:
        rows_dict = parse_fasta_alignment(aln_file)
    except Exception as e:
        return None
    
    names = list(rows_dict.keys())
    
    if len(names) < 2:
        return None
    
    # Limit sequences
    names = names[:max_k]
    ref_rows = [rows_dict[n] for n in names]
    
    # Truncate to max_L
    if max_L:
        ref_rows = [r[:max_L] for r in ref_rows]
    
    # Get true SP score
    true_sp = compute_sp_score(ref_rows)
    
    # Extract ungapped sequences
    seqs = extract_sequences(ref_rows)
    seq_lengths = [len(s) for s in seqs]
    
    results = {
        'case': Path(aln_file).stem,
        'k': len(names),
        'L': max_L,
        'true_sp': true_sp,
        'seq_lengths': seq_lengths,
        'description': description,
    }
    
    # Exact fill
    try:
        start = time.perf_counter()
        r = fill(seqs, mode="global", band=None, semiring="maxplus")
        results['exact_score'] = r['score']
        results['exact_time'] = time.perf_counter() - start
        results['exact_cells'] = r['cells']
    except Exception as e:
        results['exact_score'] = None
        results['exact_error'] = str(e)
    
    # Banded fills
    for eps in [1, 2, 3, 5]:
        try:
            start = time.perf_counter()
            r = fill(seqs, mode="global", band=eps, semiring="maxplus", part_size=1)
            results[f'band{eps}_score'] = r['score']
            results[f'band{eps}_time'] = time.perf_counter() - start
            results[f'band{eps}_cells'] = r['cells']
            if results['exact_score'] is not None:
                results[f'band{eps}_gap'] = r['score'] - results['exact_score']
                results[f'band{eps}_pct'] = (r['score'] / results['exact_score'] * 100) if results['exact_score'] != 0 else None
        except Exception as e:
            results[f'band{eps}_score'] = None
    
    return results

def main():
    # Find BAliBASE files
    balibase_dir = Path("/mnt/d/mmDst/code/data/Balibase/bb3_release")
    
    if not balibase_dir.exists():
        print(f"Directory not found: {balibase_dir}")
        return
    
    # Find all .in_tfa files (unaligned sequences) in RV subdirectories
    aln_files = []
    for rv_dir in balibase_dir.glob("RV*"):
        if rv_dir.is_dir():
            # Look for .in_tfa files (unaligned sequences)
            tfa_files = list(rv_dir.glob("*.in_tfa"))
            aln_files.extend(tfa_files)
    
    # Also look for .msf files (reference alignments)
    msf_files = []
    for rv_dir in balibase_dir.glob("RV*"):
        if rv_dir.is_dir():
            msf_files.extend(rv_dir.glob("*.msf"))
    
    print(f"Found {len(aln_files)} unaligned sequence files (.in_tfa)")
    print(f"Found {len(msf_files)} reference alignment files (.msf)")
    
    if not aln_files:
        print("No .in_tfa files found!")
        print("Looking for other formats...")
        # Try .tfa and .fa
        for rv_dir in balibase_dir.glob("RV*"):
            if rv_dir.is_dir():
                aln_files.extend(rv_dir.glob("*.tfa"))
                aln_files.extend(rv_dir.glob("*.fa"))
        print(f"Found {len(aln_files)} files")
    
    if not aln_files:
        print("No alignment files found!")
        print("Please check the BAliBASE directory structure.")
        return
    
    # Limit files for testing
    aln_files = aln_files[:30]  # Use first 30 files
    
    # Configurations to test
    configs = [
        (3, 20, "k3_L20"),
        (3, 30, "k3_L30"),
        (3, 50, "k3_L50"),
        (4, 20, "k4_L20"),
        (4, 30, "k4_L30"),
    ]
    
    all_results = []
    
    for max_k, max_L, label in configs:
        print(f"\nTesting {label} on {len(aln_files)} files...")
        
        for aln_file in tqdm(aln_files):
            # Get the RV set name
            rv_set = aln_file.parent.name
            
            result = run_balibase_experiment(
                aln_file, 
                max_k=max_k, 
                max_L=max_L,
                description=f"{rv_set}_{label}"
            )
            
            if result:
                result['rv_set'] = rv_set
                all_results.append(result)
    
    # Save results
    output_file = Path("/mnt/d/mmDst/code/data/balibase_bb3_results.csv")
    if all_results:
        with open(output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to {output_file}")
        print(f"Total cases: {len(all_results)}")
        
        # Summary statistics
        print("\n" + "="*70)
        print("BALIBASE BB3 ABLATION SUMMARY")
        print("="*70)
        
        for config in set(r['description'] for r in all_results):
            config_results = [r for r in all_results if r['description'] == config]
            print(f"\n{config}: {len(config_results)} cases")
            
            exact_ok = sum(1 for r in config_results if r.get('exact_score') is not None)
            print(f"  Exact: {exact_ok}/{len(config_results)} succeeded")
            
            if exact_ok > 0:
                for eps in [1, 2, 3, 5]:
                    valid = [r for r in config_results 
                            if r.get(f'band{eps}_score') is not None 
                            and r.get('exact_score') is not None]
                    if valid:
                        exact_avg = np.mean([r['exact_score'] for r in valid])
                        band_avg = np.mean([r[f'band{eps}_score'] for r in valid])
                        gap_avg = np.mean([r[f'band{eps}_gap'] for r in valid])
                        cells_avg = np.mean([r[f'band{eps}_cells'] for r in valid])
                        pct_avg = np.mean([r[f'band{eps}_pct'] for r in valid if r.get(f'band{eps}_pct') is not None])
                        
                        print(f"  Band eps={eps}: {len(valid)} valid, "
                              f"score={band_avg:.1f} ({pct_avg:.1f}%), "
                              f"gap={gap_avg:.1f}, cells={cells_avg:.0f}")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
