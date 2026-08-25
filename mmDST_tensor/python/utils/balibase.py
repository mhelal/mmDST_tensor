"""
balibase.py -- load BAliBASE cases (TFA unaligned input + MSF reference
alignment) and provide reference-based scoring.

BAliBASE ships:
  * <id>.tfa : unaligned sequences (FASTA)          -> the INPUT to an aligner
  * <id>.msf : the reference alignment (GCG/MSF)    -> the GROUND-TRUTH columns

IMPORTANT distinction (read this before trusting any "score"):
  BAliBASE references are STRUCTURAL alignments.  They do NOT come with a
  sum-of-pairs score under our linear (+3/-2/-6) scheme.  So there are two
  different quantities and we never conflate them:

    (A) SP score under OUR scoring of OUR alignment  -- what the DP optimises.
    (B) agreement with the reference (SP / TC columns) -- how "correct" we are.

  The classic BAliBASE metrics are:
    SP  (sum-of-pairs): fraction of residue pairs aligned in the reference that
        our alignment also aligns.
    TC  (total column): fraction of reference columns reproduced exactly.
  These are the accuracy numbers; the DP objective is a separate axis.

This module gives: load_case(), and reference_sp_tc(our_alignment, ref_alignment).
"""

from __future__ import annotations
import os
import re
import glob


def read_tfa(path):
    """Return dict name->ungapped sequence (uppercase)."""
    seqs = {}
    name = None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name is not None:
            seqs[name].append(line.strip())
    return {k: "".join(v).upper().replace(" ", "") for k, v in seqs.items()}


def read_msf(path):
    """Parse a GCG/MSF reference alignment -> dict name->aligned row.

    MSF uses '.' and '~' for gaps and lays sequences out in blocks after a
    line containing '//'. Residues are grouped in 10-char columns separated by
    spaces; we strip the spaces and keep gap characters as '-'.
    """
    lines = open(path).read().splitlines()
    # find the '//' separator
    start = 0
    for i, l in enumerate(lines):
        if l.strip() == "//":
            start = i + 1
            break
    rows = {}
    for l in lines[start:]:
        if not l.strip():
            continue
        parts = l.split(None, 1)
        if len(parts) != 2:
            continue
        name, rest = parts[0], parts[1]
        seq = rest.replace(" ", "").replace(".", "-").replace("~", "-")
        rows.setdefault(name, []).append(seq)
    return {k: "".join(v).upper() for k, v in rows.items()}


def load_case(tfa_path, msf_path=None):
    """Load one BAliBASE case. Returns (names, seqs, ref_alignment_or_None)."""
    seqs = read_tfa(tfa_path)
    ref = None
    if msf_path and os.path.exists(msf_path):
        ref = read_msf(msf_path)
    names = list(seqs.keys())
    return names, seqs, ref


def find_cases(root, refset="RV11"):
    """Yield (id, tfa_path, msf_path) for a BAliBASE reference subset."""
    base = os.path.join(root, refset)
    for tfa in sorted(glob.glob(os.path.join(base, "*.tfa"))):
        cid = os.path.splitext(os.path.basename(tfa))[0]
        msf = os.path.join(base, cid + ".msf")
        yield cid, tfa, (msf if os.path.exists(msf) else None)


# ------------------------------------------------------------------ metrics
def _pairs_in_alignment(rows):
    """Set of aligned residue pairs. Each row is a gapped string; residue
    positions are numbered per sequence. A column contributes pairs between
    every two non-gap residues. Returns set of (seq_i, pos_i, seq_j, pos_j)."""
    names = list(rows.keys())
    # per-sequence running residue index at each column
    L = len(next(iter(rows.values())))
    pos = {n: 0 for n in names}
    pairs = set()
    for col in range(L):
        present = []
        for n in names:
            c = rows[n][col]
            if c != "-":
                present.append((n, pos[n]))
                pos[n] += 1
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                (ni, pi), (nj, pj) = present[a], present[b]
                key = tuple(sorted([(ni, pi), (nj, pj)]))
                pairs.add(key)
    return pairs


def reference_sp_tc(our_rows, ref_rows):
    """Compute BAliBASE-style SP and TC of our alignment vs the reference.

    our_rows, ref_rows : dict name->gapped string, over the SAME set of names.
    SP = |pairs(ours) ∩ pairs(ref)| / |pairs(ref)|
    TC = fraction of reference columns exactly reproduced in ours.
    Returns (SP, TC).
    """
    common = set(our_rows) & set(ref_rows)
    our = {k: our_rows[k] for k in common}
    ref = {k: ref_rows[k] for k in common}
    if len(common) < 2:
        return float("nan"), float("nan")
    po = _pairs_in_alignment(our)
    pr = _pairs_in_alignment(ref)
    sp = (len(po & pr) / len(pr)) if pr else float("nan")

    # TC: compare column-sets as multisets of (name->residue-index) signatures
    def column_signatures(rows):
        names = list(rows.keys())
        L = len(next(iter(rows.values())))
        pos = {n: 0 for n in names}
        sigs = []
        for col in range(L):
            sig = []
            for n in sorted(names):
                c = rows[n][col]
                if c != "-":
                    sig.append((n, pos[n]))
                    pos[n] += 1
            sigs.append(tuple(sig))
        return sigs

    so = column_signatures(our)
    sr = column_signatures(ref)
    from collections import Counter
    co = Counter(s for s in so if s)
    cr = Counter(s for s in sr if s)
    matched = sum((co & cr).values())
    tc = matched / max(1, sum(1 for s in sr if s))
    return sp, tc


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for cid, tfa, msf in find_cases(root):
        names, seqs, ref = load_case(tfa, msf)
        lens = [len(s) for s in seqs.values()]
        print(f"{cid}: k={len(names)} lens={min(lens)}-{max(lens)} "
              f"ref={'yes' if ref else 'no'}")
