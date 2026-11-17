#!/usr/bin/env python3
"""
local_ablation_cli.py - Local alignment ablation with JASPAR ground truth
Command-line interface for running local alignment experiments
"""

import sys
import csv
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, '/mnt/d/mmDst/code')
from core import fill
from local_align import local_align

def load_ground_truth(gt_dir, max_files=None):
    """Load ground truth TSV files."""
    ground_truth = defaultdict(list)
    gt_dir = Path(gt_dir)
    
    # Look for both formats
    tsv_files = list(gt_dir.glob("*_logodds.tsv"))
    tsv_files.extend(list(gt_dir.glob("JASPAR_*.tsv")))
    
    # Limit files if specified
    if max_files:
        tsv_files = tsv_files[:max_files]
    
    print(f"Loading {len(tsv_files)} ground truth files...")
    
    for tsv_file in tqdm(tsv_files, desc="Loading GT files"):
        try:
            with open(tsv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    seq = row['sequence_name']
                    ground_truth[seq].append({
                        'start': int(row['start']),
                        'stop': int(row['stop']),
                        'strand': row['strand'],
                        'score': float(row['score']),
                        'pvalue': float(row['p-value']),
                        'motif': row['motif_id']
                    })
        except Exception as e:
            print(f"Error loading {tsv_file.name}: {e}")
    
    return dict(ground_truth)

def load_sequences(fasta_file, max_seqs=None, truncate=None):
    """Load sequences from FASTA file."""
    sequences = {}
    
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    seq = ''.join(current_seq)
                    if truncate:
                        seq = seq[:truncate]
                    sequences[current_id] = seq
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id:
            seq = ''.join(current_seq)
            if truncate:
                seq = seq[:truncate]
            sequences[current_id] = seq
    
    # Limit sequences if specified
    if max_seqs:
        items = list(sequences.items())[:max_seqs]
        sequences = dict(items)
    
    return sequences

def evaluate_local_accuracy(seqs, ground_truth, seq_names=None, band=None, verbose=False):
    """Evaluate local alignment against ground truth."""
    
    if seq_names is None:
        seq_names = [f"seq_{i}" for i in range(len(seqs))]
    
    # Run local alignment
    try:
        start = time.perf_counter()
        r = fill(seqs, mode="local", band=band, semiring="maxplus")
        elapsed = time.perf_counter() - start
        
        rows, score, spans = r['rows'], r['score'], r['spans']
        cells = r['cells']
        
        # Calculate position recovery
        total_TP = 0
        total_FP = 0
        total_FN = 0
        
        for i, (seq, span, name) in enumerate(zip(seqs, spans, seq_names)):
            if span[0] is None:
                continue
            
            pred_start, pred_stop = span[0], span[1]
            pred_positions = set(range(pred_start, pred_stop + 1))
            
            # Find true motif positions for this sequence
            true_sites = ground_truth.get(name, [])
            
            for site in true_sites:
                # Check if motif is within sequence length
                if site['stop'] > len(seq):
                    continue
                true_positions = set(range(site['start'], min(site['stop'], len(seq))))
                TP = len(pred_positions & true_positions)
                FP = len(pred_positions - true_positions)
                FN = len(true_positions - pred_positions)
                
                total_TP += TP
                total_FP += FP
                total_FN += FN
        
        # Calculate metrics
        nSn = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        nPPV = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        nPC = total_TP / (total_TP + total_FN + total_FP) if (total_TP + total_FN + total_FP) > 0 else 0
        F1 = 2 * nSn * nPPV / (nSn + nPPV) if (nSn + nPPV) > 0 else 0
        
        return {
            'score': score,
            'cells': cells,
            'time': elapsed,
            'nSn': nSn,
            'nPPV': nPPV,
            'nPC': nPC,
            'F1': F1,
            'TP': total_TP,
            'FP': total_FP,
            'FN': total_FN
        }
    
    except Exception as e:
        print(f"Error in local alignment: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Local alignment ablation with JASPAR ground truth')
    parser.add_argument('--gt-dir', type=str, 
                        default='/mnt/d/mmDst/code/data/fimo_ground_truth',
                        help='Directory containing ground truth TSV files')
    parser.add_argument('--fasta', type=str,
                        default='/mnt/d/mmDst/code/data/motif_databases/upstream1000.fa',
                        help='FASTA file with sequences')
    parser.add_argument('--output', type=str,
                        default='../../experiments/results/local_ablation_results.csv',
                        help='Output CSV file')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of sequences to align')
    parser.add_argument('--lengths', type=int, nargs='+', default=[20, 30, 50, 100],
                        help='Sequence lengths to test')
    parser.add_argument('--bands', type=int, nargs='+', default=[None, 1, 2, 3, 5],
                        help='Band widths to test (None for exact)')
    parser.add_argument('--max-gt-files', type=int, default=10,
                        help='Maximum number of ground truth files to load')
    parser.add_argument('--max-seqs', type=int, default=100,
                        help='Maximum number of sequences to load')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of random sequence subsets to test')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"Loading ground truth from {args.gt_dir}")
    ground_truth = load_ground_truth(args.gt_dir, max_files=args.max_gt_files)
    print(f"Loaded ground truth for {len(ground_truth)} sequences")
    
    # Load sequences
    print(f"Loading sequences from {args.fasta}")
    sequences = load_sequences(args.fasta, max_seqs=args.max_seqs)
    print(f"Loaded {len(sequences)} sequences")
    
    # Find sequences with ground truth
    seqs_with_gt = [s for s in sequences.keys() if s in ground_truth]
    print(f"Sequences with ground truth: {len(seqs_with_gt)}")
    
    if len(seqs_with_gt) < args.k:
        print(f"Warning: Only {len(seqs_with_gt)} sequences have ground truth (need {args.k})")
        args.k = len(seqs_with_gt)
    
    # Run experiments
    all_results = []
    np.random.seed(42)
    
    for L in args.lengths:
        for band in args.bands:
            for trial in range(args.trials):
                # Sample sequences
                sampled = np.random.choice(seqs_with_gt, size=min(args.k, len(seqs_with_gt)), replace=False)
                
                # Get sequences and names
                test_seqs = [sequences[name][:L] for name in sampled]
                seq_names = list(sampled)
                
                band_label = "exact" if band is None else f"eps{band}"
                
                if args.verbose:
                    print(f"\nTesting L={L}, band={band_label}, trial={trial+1}")
                
                # Evaluate
                metrics = evaluate_local_accuracy(
                    test_seqs, ground_truth, seq_names, 
                    band=band, verbose=args.verbose
                )
                
                if metrics:
                    result = {
                        'L': L,
                        'band': band_label,
                        'trial': trial + 1,
                        'k': len(test_seqs),
                        'score': metrics['score'],
                        'cells': metrics['cells'],
                        'time': metrics['time'],
                        'nSn': metrics['nSn'],
                        'nPPV': metrics['nPPV'],
                        'nPC': metrics['nPC'],
                        'F1': metrics['F1'],
                        'TP': metrics['TP'],
                        'FP': metrics['FP'],
                        'FN': metrics['FN']
                    }
                    all_results.append(result)
                    
                    if args.verbose:
                        print(f"  F1={metrics['F1']:.3f}, nSn={metrics['nSn']:.3f}, nPPV={metrics['nPPV']:.3f}")
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if all_results:
        with open(output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to {output_file}")
        print(f"Total cases: {len(all_results)}")
        
        # Summary
        print("\n" + "="*80)
        print("LOCAL ALIGNMENT ABLATION SUMMARY")
        print("="*80)
        
        # Group by L and band
        for L in args.lengths:
            print(f"\nLength L={L}:")
            for band in args.bands:
                band_label = "exact" if band is None else f"eps{band}"
                valid = [r for r in all_results if r['L'] == L and r['band'] == band_label]
                if valid:
                    avg_F1 = np.mean([r['F1'] for r in valid])
                    avg_nSn = np.mean([r['nSn'] for r in valid])
                    avg_nPPV = np.mean([r['nPPV'] for r in valid])
                    avg_cells = np.mean([r['cells'] for r in valid])
                    print(f"  {band_label}: F1={avg_F1:.3f}, nSn={avg_nSn:.3f}, nPPV={avg_nPPV:.3f}, cells={avg_cells:.0f}")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
