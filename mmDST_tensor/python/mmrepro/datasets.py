"""Minimal BAliBASE and JASPAR readers used by the reproducibility runner."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np


def read_fasta(path):
    d={}; name=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith('>'):
            name=line[1:].split()[0]; d[name]=[]
        elif name is not None: d[name].append(line)
    return {k:''.join(v).upper() for k,v in d.items()}


def read_msf(path):
    lines=open(path).read().splitlines(); start=0
    for i,l in enumerate(lines):
        if l.strip()=='//': start=i+1; break
    rows={}
    for l in lines[start:]:
        if not l.strip(): continue
        p=l.split(None,1)
        if len(p)!=2: continue
        name,rest=p; seq=rest.replace(' ','').replace('.','-').replace('~','-')
        rows.setdefault(name,[]).append(seq.upper())
    return {k:''.join(v) for k,v in rows.items()}


def crop_reference(ref, names, L):
    """Keep reference columns until each selected row has consumed <=L residues;
    drop columns containing residues beyond the truncated input in any row."""
    names=[n for n in names if n in ref]
    if len(names)<2:return None
    pos={n:0 for n in names}; cols=[]; width=len(ref[names[0]])
    for j in range(width):
        keep=True; col={}
        for n in names:
            ch=ref[n][j]; col[n]=ch
            if ch!='-' and pos[n]>=L: keep=False
        if keep: cols.append(col)
        for n in names:
            if ref[n][j]!='-': pos[n]+=1
    return {n:''.join(c[n] for c in cols) for n in names}


def _pairs(rows):
    names=list(rows); pos={n:0 for n in names}; out=set(); L=len(rows[names[0]])
    for j in range(L):
        present=[]
        for n in names:
            ch=rows[n][j]
            if ch!='-': present.append((n,pos[n])); pos[n]+= ch!='-'
        for a in range(len(present)):
            for b in range(a+1,len(present)):
                x,y=present[a],present[b]; out.add(tuple(sorted((x,y))))
    return out


def reference_sp(our_rows, ref_rows):
    if not ref_rows:return float('nan')
    common=[n for n in our_rows if n in ref_rows]
    if len(common)<2:return float('nan')
    o={n:our_rows[n] for n in common}; r={n:ref_rows[n] for n in common}
    po,pr=_pairs(o),_pairs(r); return len(po&pr)/len(pr) if pr else float('nan')


def read_jaspar_pfms(path):
    """Parse common JASPAR PFM text format and return motif_id->consensus DNA."""
    motifs={}; mid=None; rows={}
    def flush():
        nonlocal rows,mid
        if mid and all(b in rows for b in 'ACGT'):
            M=np.vstack([rows[b] for b in 'ACGT']); consensus=''.join('ACGT'[i] for i in np.argmax(M,axis=0)); motifs[mid]=consensus
        rows={}
    for raw in open(path):
        line=raw.strip()
        if not line: continue
        if line.startswith('>'):
            flush(); mid=line[1:].split()[0]; continue
        m=re.match(r'^([ACGT])\s*\[?\s*(.*?)\s*\]?$',line,re.I)
        if m:
            vals=[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',m.group(2))]
            rows[m.group(1).upper()]=vals
    flush(); return motifs


def read_scanner_hits(hit_dir, positive_strand_only=True):
    """Read delimited scanner hits with explicit required columns.

    This reader is a compatibility implementation, not an original recovered
    scanner parser. Expected columns: motif_id, sequence_name, start, stop;
    optional strand. Coordinates are passed through unchanged.
    """
    import csv
    from pathlib import Path
    path=Path(hit_dir)
    files=sorted(p for p in path.rglob('*') if p.suffix.lower() in ('.csv','.tsv')) if path.is_dir() else [path]
    if not files: raise FileNotFoundError(f'No CSV/TSV scanner-hit files found: {hit_dir}')
    hits=[]
    for f in files:
        with f.open(newline='') as stream:
            reader=csv.DictReader(stream,delimiter='\t' if f.suffix.lower()=='.tsv' else ',')
            required={'motif_id','sequence_name','start','stop'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{f}: required scanner columns {sorted(required)}; found {reader.fieldnames}')
            for row in reader:
                if positive_strand_only and row.get('strand','+').strip() not in ('+','1','forward','Forward'):
                    continue
                row['start']=int(row['start']); row['stop']=int(row['stop'])
                hits.append(row)
    return hits
