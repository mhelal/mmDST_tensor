"""
moa_msa_numpy.py -- Stage-1 global oracle. Now a THIN SHIM over core.py so the
recurrence exists in exactly one place. The MoAMSA class API is preserved for
any code/tests that import it.
"""
from __future__ import annotations
import numpy as np
from core import fill, Geometry, encode, sub_score, GAP


class MoAMSA:
    def __init__(self, sequences, stype=1):
        self.seqs = list(sequences)
        self.k = len(self.seqs)
        self._r = None
        self.shape = tuple(len(s) + 1 for s in self.seqs)

    def fill(self):
        self._r = fill(self.seqs, mode="global", semiring="maxplus")
        # expose an ndarray S with the SAME (F-order) layout as before
        self.S = self._r["flat"].reshape(self.shape, order="F")
        self.prev = self._r["prev"]
        return self.S

    def align(self):
        if self._r is None:
            self.fill()
        return self._r["rows"], self._r["score"]


def align(sequences, stype=1):
    r = fill(list(sequences), mode="global")
    return r["rows"], r["score"]


if __name__ == "__main__":
    print(align(["AATC", "ATC", "AC"]))


# --- back-compat helpers re-exported from core semantics ---
def tau(shape):
    r = 1
    for s in shape:
        r *= int(s)
    return r


def gap_score(stype=1):
    return GAP
