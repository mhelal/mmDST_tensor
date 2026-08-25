"""
waves.py -- wave (anti-diagonal) utilities. Thin shim: the wave geometry now
lives in core.Geometry; this re-exports for any importer and provides the
fill_by_waves reference used in earlier validation.
"""
from __future__ import annotations
import numpy as np
from core import Geometry, encode, fill, GAP, MATCH, MIS


def fill_by_waves(lengths, seqs_enc):
    """Back-compat: return (S_flat, prev, geo) for a global maxplus fill.
    seqs_enc is accepted for signature compatibility; sequences are rebuilt."""
    # reconstruct sequences from encodings
    from core import ALPHA
    seqs = ["".join(ALPHA[int(x)] for x in e) for e in seqs_enc]
    r = fill(seqs, mode="global", semiring="maxplus")
    return r["flat"], r["prev"], r["geo"]


if __name__ == "__main__":
    from core import encode
    s, p, g = fill_by_waves([4, 3], [encode("AATC"), encode("ATC")])
    print("sink", int(s[g.n - 1]))


# --- back-compat for msa_cupy/test_msa_cupy ---
GAPI = None
def init_border(geo, S):
    """Global border init on a flat array, matching core."""
    for f in range(geo.n):
        c = geo.coords(f)
        if any(x == 0 for x in c):
            t = 1
            for x in c:
                t *= (x + 1)
            S[f] = GAP * t
