#!/usr/bin/env python3
"""
run_jaspar_scan.py - Scan JASPAR vertebrate motifs against upstream1000.fa
"""

import os
import sys
import csv
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add the parent directory to path so we can import scan_motifs
sys.path.insert(0, str(Path(__file__).parent))

from scan_motifs import parse_meme_file, parse_fasta, scan_sequence

def scan_jaspar_against_fasta():
    """Scan all JASPAR motifs against upstream1000.fa sequences."""
    
    # Paths
    jaspar_dir = Path("/mnt/d/mmDst/code/data/jaspar_vertebrates_memes")
    fasta_file = Path("/mnt/d/mmDst/code/data/motif_databases/upstream1000.fa")
    output_dir = Path("/mnt/d/mmDst/code/data/fimo_ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not jaspar_dir.exists():
        print(f"ERROR: JASPAR directory not found: {jaspar_dir}")
        return
    
    if not fasta_file.exists():
        print(f"ERROR: FASTA file not found: {fasta_file}")
        return
    
    # Find all .meme files in JASPAR directory (including subdirectories)
    meme_files = list(jaspar_dir.rglob("*.meme"))
    print(f"Found {len(meme_files)} motif files in {jaspar_dir}")
    
    if not meme_files:
        # Try looking for .pfm or .txt files
        pfm_files = list(jaspar_dir.rglob("*.pfm"))
        txt_files = list(jaspar_dir.rglob("*.txt"))
        print(f"Found {len(pfm_files)} PFM files, {len(txt_files)} TXT files")
        
        # If only PFM files, we need to handle them
        if pfm_files and not meme_files:
            print("Found PFM files - parsing as JASPAR format...")
            meme_files = pfm_files
    
    if not meme_files:
        print("No motif files found. Please check the directory.")
        return
    
    # Load sequences
    print(f"Loading sequences from {fasta_file}")
    sequences = parse_fasta(fasta_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Limit sequences for testing (use first 1000 sequences to keep it manageable)
    # For full run, remove this limit
    seq_items = list(sequences.items())[:1000]
    print(f"Using {len(seq_items)} sequences (limit to 1000 for testing)")
    
    # Process each motif file
    total_motifs = 0
    total_hits = 0
    all_results = []
    
    for motif_file in tqdm(meme_files, desc="Processing motifs"):
        try:
            motifs = parse_meme_file(motif_file)
        except Exception as e:
            print(f"\nError parsing {motif_file.name}: {e}")
            continue
        
        if not motifs:
            continue
        
        total_motifs += len(motifs)
        
        # Create a flat output name
        flat_name = f"JASPAR_{motif_file.stem}.tsv"
        output_path = output_dir / flat_name
        
        hits_this_file = 0
        
        with open(output_path, 'w') as out_f:
            writer = csv.writer(out_f, delimiter='\t')
            writer.writerow(['motif_id', 'sequence_name', 'start', 'stop', 
                           'strand', 'score', 'p-value'])
            
            for motif in motifs:
                pwm = motif.get('pwm')
                if pwm is None or pwm.size == 0:
                    continue
                
                motif_name = motif.get('name', motif_file.stem)
                
                for seq_id, seq in seq_items:
                    # Skip empty sequences
                    if not seq:
                        continue
                    
                    # Forward strand
                    for hit in scan_sequence(seq, pwm, threshold=1e-5):
                        writer.writerow([motif_name, seq_id, *hit])
                        hits_this_file += 1
                        all_results.append({
                            'motif': motif_name,
                            'sequence': seq_id,
                            'start': hit[0],
                            'stop': hit[1],
                            'strand': hit[2],
                            'score': hit[3],
                            'pvalue': hit[4]
                        })
                    
                    # Reverse strand
                    for hit in scan_sequence(seq, pwm, threshold=1e-5, reverse=True):
                        writer.writerow([motif_name, seq_id, *hit])
                        hits_this_file += 1
                        all_results.append({
                            'motif': motif_name,
                            'sequence': seq_id,
                            'start': hit[0],
                            'stop': hit[1],
                            'strand': hit[2],
                            'score': hit[3],
                            'pvalue': hit[4]
                        })
        
        total_hits += hits_this_file
        if hits_this_file > 0:
            print(f"\n  {motif_file.name}: {len(motifs)} motifs, {hits_this_file} hits")
    
    print(f"\n✓ Processed {total_motifs} motifs, {total_hits} total hits")
    print(f"✓ Output directory: {output_dir}")
    
    # Create a summary file with all results
    summary_path = output_dir / "all_jaspar_hits.tsv"
    with open(summary_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['motif', 'sequence', 'start', 'stop', 
                                               'strand', 'score', 'pvalue'], delimiter='\t')
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✓ Summary file: {summary_path} ({len(all_results)} total hits)")
    
    return all_results

if __name__ == "__main__":
    print("=" * 60)
    print("JASPAR Motif Scanner")
    print("=" * 60)
    
    start_time = time.time()
    results = scan_jaspar_against_fasta()
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Generated {len(results) if results else 0} motif hits")
