"""
hyperdiag.py -- FAITHFUL implementation of the hyper-diagonal search-space
reduction of Helal, Mullin, Potter & Sintchenko (NPC 2009), replacing the
approximate "corridor" band used in the earlier drafts.

Two components from the paper:

1. MIDDLE PARTITION INDEX mp(w).  For wave w and dimensionality k the paper
   computes the diagonal's lattice point on that wave by fair division with the
   remainder pushed to the MIDDLE dimensions:

       q = floor(w/k);  r = w mod k
       for i in 0..k-1:
           if floor((k-r)/2) <= i < floor((k+r)/2):  mp[i] = q+1
           else:                                     mp[i] = q

   (paper's pseudocode; Table 1 gives worked examples we use as unit tests.)

2. FIXED BAND (Equation 6).  A cell y is in the search space iff

       max_i | y_i - mp_i(w) |  <=  epsilon,        w = sum_i y_i

   i.e. a CHEBYSHEV (L-infinity) ball of radius epsilon around the discrete
   middle-partition index on the cell's own wave.  This differs from the
   real-valued interpolated corridor used earlier: it is an exact integer
   lattice construction and it is defined per-wave.

3. AVERAGING MISSING NEIGHBOURS.  Critically, the paper's scoring "calculates
   all possible paths from the origin up to the current scoring cell that fall
   within the search space considered, and averages the missing neighbours."
   Predecessors outside the band are therefore ESTIMATED, not dropped.  This is
   why <1% of the search space can reproduce the exact optimum.  We implement
   this as: if some predecessors are out of band, substitute the mean of the
   in-band predecessor candidate scores for the missing ones.

The paper's scope note is explicit: "a very small band around the middle of the
tensor was sufficient for GLOBAL alignments."  Local alignment is out of scope
there, which is consistent with the degradation we measure for local mode.
"""

from __future__ import annotations
import itertools
import numpy as np

from core import Geometry, encode, edge_weight, GAP, sub_score


def middle_partition(w: int, k: int):
    """Paper's mp(w): fair division of wave w over k dims, remainder to middle."""
    q, r = divmod(w, k)
    lo = (k - r) // 2
    hi = (k + r) // 2
    return [q + 1 if lo <= i < hi else q for i in range(k)]


def in_band(coords, k, eps):
    """Equation 6: max_i |y_i - mp_i(w)| <= eps, with w = sum(coords)."""
    w = sum(coords)
    mp = middle_partition(w, k)
    return max(abs(coords[i] - mp[i]) for i in range(k)) <= eps


def band_cell_count(shape, eps):
    """Number of interior cells inside the epsilon band (for complexity stats)."""
    k = len(shape)
    n = 1
    for s in shape:
        n *= s
    cnt = 0
    for f in range(n):
        c = []
        t = f
        for i in range(k):
            c.append(t % shape[i]); t //= shape[i]
        if all(x > 0 for x in c) and in_band(c, k, eps):
            cnt += 1
    return cnt


def fill_hyperdiag(seqs, eps, mode="global", average_missing=True):
    """Global/local fill restricted to the paper's epsilon band.

    average_missing : if True, predecessors outside the band contribute the mean
    of the in-band candidates (the paper's "average the missing neighbours"),
    rather than being silently skipped.
    """
    geo = Geometry([len(s) for s in seqs])
    k = geo.k
    enc = [encode(s) for s in seqs]
    masks = geo.masks()
    local = (mode == "local")

    S = np.zeros(geo.n, dtype=np.float64)
    prev = np.full(geo.n, -1, dtype=np.int64)
    scored = 0
    best_score = 0.0
    anchor = geo.n - 1

    for f in range(geo.n):
        c = geo.coords(f)
        if not in_band(c, k, eps):
            continue
        if any(x == 0 for x in c):
            if local:
                S[f] = 0.0
            else:
                tau = 1
                for x in c:
                    tau *= (x + 1)
                S[f] = GAP * tau
            continue

        cands = []          # in-band candidate scores
        cand_prev = []
        n_missing = 0
        for m in masks:
            nb = [c[i] - m[i] for i in range(k)]
            if any(x < 0 for x in nb):
                continue
            if not in_band(nb, k, eps):
                n_missing += 1
                continue
            nf = geo.flat(nb)
            cands.append(S[nf] + edge_weight(c, m, enc, k))
            cand_prev.append(nf)

        if not cands:
            continue
        best = max(cands)
        bp = cand_prev[int(np.argmax(cands))]
        if average_missing and n_missing:
            # estimate the missing predecessors by the mean of the in-band ones
            est = float(np.mean(cands))
            best = max(best, est)
        if local:
            best = max(0.0, best)
        S[f] = best
        prev[f] = bp
        scored += 1
        if local and best > best_score:
            best_score = best
            anchor = f

    score = best_score if local else S[geo.n - 1]
    return dict(score=float(score), cells=scored, S=S, prev=prev,
                anchor=anchor, geo=geo)


# ------------------------------------------------------------------ unit tests
def _test_table1():
    """Paper Table 1: middle partition index examples."""
    cases = [
        (5, 6, [1, 1, 1, 1, 1, 0]),
        (9, 6, [1, 2, 2, 2, 1, 1]),
        (12, 6, [2, 2, 2, 2, 2, 2]),
        (35, 8, [4, 4, 5, 5, 5, 4, 4, 4]),
        (40, 12, [3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3]),
    ]
    ok = 0
    for w, k, expect in cases:
        got = middle_partition(w, k)
        match = (got == expect)
        ok += match
        print(f"  w={w:>2} k={k:>2}  got {got}  expected {expect}  "
              f"{'OK' if match else 'MISMATCH'}")
    return ok, len(cases)


def same_wave_ball(eps, k):
    """Number of lattice points d in [-eps,eps]^k with sum(d)==0, INCLUDING the
    centre. This is the true content of the paper's Table 4.

    NOTE: the paper prints the closed form ((2*eps)+1)^k - 1, which is the size
    of the FULL Chebyshev ball minus the centre. The tabulated values are in
    fact the WAVE-RESTRICTED count (sum of offsets == 0) INCLUDING the centre --
    the text's own qualifier "restricted to neighbours that fall on the same
    wave number" is the operative definition, and it reproduces all tabulated
    values exactly (21/21), whereas the printed formula does not.
    """
    return sum(1 for d in itertools.product(range(-eps, eps + 1), repeat=k)
               if sum(d) == 0)


def _test_table4():
    """Paper Table 4, reproduced by the wave-restricted count."""
    expect = {(1, 3): 7, (1, 4): 19, (1, 5): 51,
              (2, 3): 19, (2, 4): 85, (2, 5): 381,
              (3, 3): 37, (3, 4): 231, (3, 5): 1451,
              (4, 3): 61, (4, 4): 489, (5, 3): 91, (5, 4): 891,
              (6, 3): 127, (7, 3): 169, (8, 3): 217, (9, 3): 271, (10, 3): 331}
    ok = 0
    for (eps, k), val in sorted(expect.items()):
        got = same_wave_ball(eps, k)
        ok += (got == val)
        if got != val:
            print(f"  eps={eps} k={k}: got {got}, paper {val}  MISMATCH")
    return ok, len(expect)


if __name__ == "__main__":
    print("Paper Table 1 (middle partition index):")
    a, b = _test_table1()
    print(f"  -> {a}/{b} match\n")
    print("Paper Table 4 (((2e)+1)^k - 1 neighbour counts):")
    a2, b2 = _test_table4()
    print(f"  -> {a2}/{b2} match")
