"""
local_align.py -- LOCAL (Smith-Waterman) API. Thin shim over core.py (mode=local).
Retains smith_waterman_2d for independent validation.
"""
from __future__ import annotations
import numpy as np
from core import fill, encode, sub_score, GAP, GAPI, _LUT

MATCH, MIS = 3, -2


def local_fill(seqs):
    r = fill(seqs, mode="local", semiring="maxplus")
    return r["flat"], r["prev"], r["anchor"], r["score"], r["geo"].shape


def local_align(seqs):
    r = fill(seqs, mode="local", semiring="maxplus")
    return r["rows"], r["score"], r["spans"]


def smith_waterman_2d(a, b, match=MATCH, mis=MIS, gap=GAP):
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), dtype=int); best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = H[i-1, j-1] + (match if a[i-1] == b[j-1] else mis)
            H[i, j] = max(0, d, H[i-1, j] + gap, H[i, j-1] + gap)
            best = max(best, H[i, j])
    return best


if __name__ == "__main__":
    import random
    random.seed(0); bases = "ACGT"; d = 0
    for _ in range(200):
        a = "".join(random.choice(bases) for _ in range(random.randint(3, 9)))
        b = "".join(random.choice(bases) for _ in range(random.randint(3, 9)))
        _, _, _, best, _ = local_fill([a, b])
        if best != smith_waterman_2d(a, b): d += 1
    print(f"local (shim over core) vs Smith-Waterman: {200-d}/200")
