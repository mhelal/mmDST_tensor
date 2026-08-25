"""
banded.py -- exact and hyper-diagonal-banded fills. Thin wrappers over
core.fill(); no recurrence here.

CORRECTED (NPC-2009 faithful): the band is now the Chebyshev ball around the
discrete middle-partition index mp(w), applied at partition granularity, with
out-of-band predecessors imputed by averaging in-band candidates. See core.py
Geometry.middle_partition_for / Geometry.in_band.
"""
from __future__ import annotations
import sys
from core import fill, encode, sub_score, GAP, Geometry  # re-exports


def _run(seqs, band, mode, part_size=1, average_missing=True):
    r = fill(seqs, mode=mode, band=band, semiring="maxplus",
             part_size=part_size, average_missing=average_missing)
    tbytes = r["flat"].nbytes * 2 if band is None else sys.getsizeof(r["flat"])
    return r["score"], r["rows"], r["cells"], tbytes


def full_fill(seqs, mode="global"):
    """Exact fill. Returns (score, rows, cells_scored, tensor_bytes)."""
    return _run(seqs, None, mode)


def banded_fill(seqs, eps, mode="global", part_size=1, average_missing=True):
    """Hyper-diagonal banded fill (NPC-2009 Eq. 6). eps is the band radius."""
    return _run(seqs, eps, mode, part_size, average_missing)


if __name__ == "__main__":
    print("global exact:", full_fill(["AATC", "ATC", "AC"])[:3])
    print("global eps=2:", banded_fill(["AATC", "ATC", "AC"], 2)[:3])
    print("local  exact:", full_fill(["AATC", "ATC", "AC"], mode="local")[:3])
