#!/usr/bin/env python3
"""
scan_motifs.py - Multi-format motif scanner
Handles .meme, .pfm, .jaspar, .txt, .matrix formats
"""

import os
import re
import gzip
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv
import argparse
from tqdm import tqdm

# ============================================================================
# MULTI-FORMAT PARSER
# ============================================================================

def detect_format(filepath):
    """Detect motif file format by content."""
    with open(filepath, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(5)]
    
    content = ' '.join(first_lines).upper()
    
    if 'MEME' in content and 'MOTIF' in content:
        return 'meme'
    elif 'JASPAR' in content:
        return 'jaspar'
    elif all(c in 'ACGT0123456789. \t-' for c in content.replace('>', '').replace('#', '')):
        # Check if it looks like a PFM (matrix with 4 rows)
        return 'pfm'
    else:
        return 'unknown'

def parse_meme_file(filepath):
    """Parse MEME format."""
    motifs = []
    current = None
    pwm_lines = []
    reading_pwm = False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('MOTIF'):
            if current and pwm_lines:
                current['pwm'] = _parse_pwm_lines(pwm_lines)
                motifs.append(current)
            
            parts = line.split()
            current = {
                'name': parts[1] if len(parts) > 1 else 'unknown',
                'alphabet': 'ACGT',
                'strands': '+-',
                'pwm': None
            }
            pwm_lines = []
            reading_pwm = False
        
        elif line and (line[0].isdigit() or line[0] == '-'):
            reading_pwm = True
            pwm_lines.append(line)
        
        elif reading_pwm and not line:
            if current and pwm_lines:
                current['pwm'] = _parse_pwm_lines(pwm_lines)
                motifs.append(current)
                current = None
                pwm_lines = []
                reading_pwm = False
    
    if current and pwm_lines:
        current['pwm'] = _parse_pwm_lines(pwm_lines)
        motifs.append(current)
    
    return motifs

def parse_jaspar_file(filepath):
    """Parse JASPAR format."""
    motifs = []
    current = None
    pwm_lines = []
    reading_pwm = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('>'):
                if current and pwm_lines:
                    current['pwm'] = _parse_pwm_lines(pwm_lines)
                    motifs.append(current)
                current = {
                    'name': line[1:].split()[0],
                    'alphabet': 'ACGT',
                    'strands': '+-',
                    'pwm': None
                }
                pwm_lines = []
                reading_pwm = False
            elif re.match(r'^[ACGT\s]+$', line, re.I):
                # JASPAR matrix
                parts = line.split()
                if len(parts) >= 4:
                    pwm_lines.append(line)
                    reading_pwm = True
    
    if current and pwm_lines:
        current['pwm'] = _parse_pwm_lines(pwm_lines)
        motifs.append(current)
    
    return motifs

def parse_pfm_file(filepath):
    """Parse PFM (Position Frequency Matrix) format."""
    motifs = []
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by > or by blank lines
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        # Check if it's a PFM (4 rows: A, C, G, T)
        pwm = []
        name = Path(filepath).stem
        
        for line in lines:
            if line.startswith('>'):
                name = line[1:].strip()
                continue
            if line.startswith('#'):
                continue
            
            nums = re.findall(r'[-+]?\d*\.?\d+', line)
            if nums:
                pwm.append([float(x) for x in nums])
        
        if len(pwm) >= 4:
            # Transpose to (4, L)
            pwm_matrix = np.array(pwm).T
            # Convert counts to probabilities
            pwm_matrix = pwm_matrix / pwm_matrix.sum(axis=0, keepdims=True)
            motifs.append({
                'name': name,
                'alphabet': 'ACGT',
                'strands': '+-',
                'pwm': pwm_matrix
            })
    
    return motifs

def parse_transfac_file(filepath):
    """Parse TRANSFAC format."""
    motifs = []
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by // line
    blocks = content.split('//')
    
    for block in blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        name = None
        pwm_data = []
        reading_matrix = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('ID'):
                name = line.split()[1] if len(line.split()) > 1 else 'unknown'
            elif line.startswith('P0') or line.startswith('P1'):
                reading_matrix = True
                parts = line.split()[1:]
                if parts:
                    pwm_data.append([float(x) for x in parts])
            elif reading_matrix and line.startswith('P') and not line.startswith('P0'):
                parts = line.split()[1:]
                if parts:
                    pwm_data.append([float(x) for x in parts])
        
        if pwm_data:
            # TRANSFAC rows are often (A, C, G, T)
            pwm_matrix = np.array(pwm_data).T
            pwm_matrix = pwm_matrix / pwm_matrix.sum(axis=0, keepdims=True)
            motifs.append({
                'name': name or 'unknown',
                'alphabet': 'ACGT',
                'strands': '+-',
                'pwm': pwm_matrix
            })
    
    return motifs

def _parse_pwm_lines(lines):
    """Parse PWM lines into numpy array."""
    pwm_rows = []
    for line in lines:
        parts = line.split()
        # Try to extract numbers
        nums = [float(x) for x in parts if re.match(r'[-+]?\d*\.?\d+', x)]
        if len(nums) >= 4:
            pwm_rows.append(nums[:4])
    
    if not pwm_rows:
        return np.zeros((4, 1))
    
    pwm = np.array(pwm_rows).T
    # Normalize to probabilities
    col_sums = pwm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    pwm = pwm / col_sums
    return pwm

def detect_and_parse(filepath):
    """Detect format and parse accordingly."""
    ext = filepath.suffix.lower()
    
    if ext == '.meme':
        return parse_meme_file(filepath)
    elif ext == '.jaspar':
        return parse_jaspar_file(filepath)
    elif ext == '.pfm':
        return parse_pfm_file(filepath)
    elif ext == '.txt' or ext == '.matrix':
        # Try both
        try:
            return parse_meme_file(filepath)
        except:
            try:
                return parse_pfm_file(filepath)
            except:
                return parse_transfac_file(filepath)
    else:
        # Try all parsers
        for parser in [parse_meme_file, parse_jaspar_file, parse_pfm_file, parse_transfac_file]:
            try:
                motifs = parser(filepath)
                if motifs:
                    return motifs
            except:
                continue
        return []

# ============================================================================
# REST OF THE CODE (parse_fasta, scan_sequence, etc.)
# ============================================================================

def parse_fasta(filepath):
    """Parse FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []
    
    try:
        if str(filepath).endswith('.gz'):
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'
        
        with opener(filepath, mode) as f:
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
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}
    
    return sequences

def scan_sequence(seq, pwm, threshold=1e-5, reverse=False):
    """Scan one sequence for motif matches."""
    if not seq or pwm is None or pwm.size == 0:
        return []
    
    if reverse:
        trans = str.maketrans('ACGT', 'TGCA')
        seq = seq.translate(trans)[::-1]
    
    nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = pwm.shape[1]
    matches = []
    
    if L > len(seq) or L == 0:
        return matches
    
    for pos in range(len(seq) - L + 1):
        window = seq[pos:pos+L]
        score = 0.0
        
        for j, nt in enumerate(window):
            idx = nucleotide_to_idx.get(nt)
            if idx is None:
                score += np.mean(pwm[:, j])
            else:
                score += pwm[idx, j]
        
        # Simple threshold using score
        # For p-value, approximate
        if score > -5:  # Heuristic threshold
            pval = _score_to_pvalue(score)
            if pval < threshold:
                matches.append((pos, pos + L, '-' if reverse else '+', float(score), float(pval)))
    
    return matches

def _score_to_pvalue(score, mean=-2.0, std=1.5):
    """Convert score to p-value."""
    from math import erf, sqrt
    z = (score - mean) / std
    z = max(min(z, 10), -10)
    pval = 0.5 * (1 + erf(z / sqrt(2)))
    return max(1e-20, min(1.0, pval))

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_ground_truth(motif_dir, fasta_file, output_dir, threshold=1e-5):
    """Main pipeline."""
    motif_dir = Path(motif_dir)
    fasta_file = Path(fasta_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading sequences from {fasta_file}")
    sequences = parse_fasta(fasta_file)
    
    if not sequences:
        print(f"ERROR: No sequences loaded")
        return
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Find ALL motif files
    motif_extensions = ['.meme', '.jaspar', '.pfm', '.txt', '.matrix', '.transfac']
    motif_files = []
    for ext in motif_extensions:
        motif_files.extend(motif_dir.rglob(f'*{ext}'))
        motif_files.extend(motif_dir.rglob(f'*{ext.upper()}'))
    
    # Remove duplicates
    motif_files = list(set(motif_files))
    
    print(f"Found {len(motif_files)} motif files")
    
    if not motif_files:
        print("WARNING: No motif files found. Checking directory contents...")
        print(f"Contents of {motif_dir}:")
        for item in motif_dir.iterdir():
            if item.is_dir():
                print(f"  [DIR] {item.name}")
                for sub in list(item.iterdir())[:5]:
                    print(f"    {sub.name}")
            else:
                print(f"  [FILE] {item.name}")
        return
    
    total_motifs = 0
    total_hits = 0
    
    for motif_file in tqdm(motif_files, desc="Processing motif files"):
        try:
            motifs = detect_and_parse(motif_file)
        except Exception as e:
            print(f"\nError parsing {motif_file.name}: {e}")
            continue
        
        if not motifs:
            continue
        
        total_motifs += len(motifs)
        
        # Output file
        rel_path = motif_file.relative_to(motif_dir)
        flat_name = "_".join(rel_path.with_suffix('.tsv').parts)
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
                
                motif_name = motif.get('name', 'unknown')
                
                # Sample sequences to avoid memory issues
                for seq_id, seq in list(sequences.items())[:1000]:  # Limit for testing
                    for hit in scan_sequence(seq, pwm, threshold):
                        writer.writerow([motif_name, seq_id, *hit])
                        hits_this_file += 1
                    
                    for hit in scan_sequence(seq, pwm, threshold, reverse=True):
                        writer.writerow([motif_name, seq_id, *hit])
                        hits_this_file += 1
        
        total_hits += hits_this_file
        print(f"\n  {motif_file.name}: {len(motifs)} motifs, {hits_this_file} hits")
    
    print(f"\n✓ Processed {total_motifs} motifs, {total_hits} total hits")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-format motif scanner')
    parser.add_argument('--motif-dir', default='motif_databases')
    parser.add_argument('--fasta', default='motif_databases/upstream1000.fa')
    parser.add_argument('--output-dir', default='fimo_ground_truth')
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--limit', type=int, default=1000, help='Limit sequences for testing')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    motif_dir = script_dir / args.motif_dir
    fasta_file = script_dir / args.fasta
    
    if not fasta_file.exists():
        fasta_gz = fasta_file.with_suffix('.fa.gz')
        if fasta_gz.exists():
            fasta_file = fasta_gz
    
    generate_ground_truth(motif_dir, fasta_file, args.output_dir, args.threshold)

if __name__ == "__main__":
    main()
