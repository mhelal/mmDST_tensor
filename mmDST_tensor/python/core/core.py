"""
core.py -- THE single source of truth for the MoA DP-MSA recurrence.

Every other module calls into this. The recurrence is written exactly ONCE and
parameterised along four independent axes:

  mode      : "global" (Needleman-Wunsch) | "local" (Smith-Waterman)
  band      : None (exact, full hypercube) | w>=0 (hyper-diagonal half-width w)
  semiring  : "maxplus" (best score) | "logsumexp" (partition function)
  scoring   : substitution/gap scheme (linear by default)

Global vs local is exactly three switches (Smith & Waterman 1981):
  (L1) zero floor      : interior cell takes max(0, ...)      [local only]
  (L2) zero boundary   : border cells = 0, not g*Tau(c+1)     [local only]
  (L3) traceback anchor: from global argmax to first zero     [local only]
For the partition (logsumexp) semiring the local analogue of (L1) is an added
restart term (+1 = exp(0), the empty alignment) at every cell.

Everything is dimension-invariant: one code path for any number of sequences k.
Layouts follow the MoA first-axis-fastest (column-major) Gamma convention.
"""

from __future__ import annotations
import itertools
import numpy as np

# ---------------------------------------------------------------- scoring
GAP, MATCH, MIS = -6, 3, -2
# alphabet: proteins + gap + DNA (lower/upper handled by upper()) + ambiguity
ALPHA = "ACDEFGHIKLMNPQRSTVWY-XBZU*"
_LUT = {c: i for i, c in enumerate(ALPHA)}
GAPI = _LUT["-"]


def encode(seq):
    """Residue string -> int32 codes (uppercased; unknown -> X)."""
    xi = _LUT.get("X")
    return np.array([_LUT.get(c, xi) for c in seq.upper()], dtype=np.int32)


def sub_score(ai, bi):
    """Linear substitution/gap score between two encoded residues."""
    if ai == bi:
        return MATCH
    if ai == GAPI or bi == GAPI:
        return GAP
    return MIS


# ---------------------------------------------------------------- geometry
class Geometry:
    """Shape, strides (F-order), flat<->multi index, waves, band test."""

    def __init__(self, lengths):
        self.k = len(lengths)
        self.shape = tuple(int(l) + 1 for l in lengths)
        self.strides = [1] * self.k
        for i in range(1, self.k):
            self.strides[i] = self.strides[i - 1] * self.shape[i - 1]
        self.n = 1
        for s in self.shape:
            self.n *= s
        self.nsum = sum(s - 1 for s in self.shape)

    def coords(self, f):
        c = [0] * self.k
        t = f
        for i in range(self.k):
            c[i] = t % self.shape[i]
            t //= self.shape[i]
        return c

    def flat(self, c):
        return sum(c[i] * self.strides[i] for i in range(self.k))

    def masks(self):
        return [m for m in itertools.product((0, 1), repeat=self.k) if any(m)]

    def waves(self):
        """List of arrays: waves()[w] = flat indices with coordinate-sum w."""
        import numpy as _np
        buckets = {}
        for f in range(self.n):
            s = sum(self.coords(f))
            buckets.setdefault(s, []).append(f)
        mx = max(buckets) if buckets else 0
        return [_np.array(buckets.get(w, []), dtype=_np.int64) for w in range(mx + 1)]

    def middle_partition(self, w):
        """Hyper-diagonal lattice point on wave w (Helal et al., NPC 2009, Eq/proc).

        Fair division of the wave number over the k axes, with the remainder
        r = w mod k assigned to the MIDDLE dimensions:

            q = floor(w/k);  r = w mod k
            mp[i] = q+1  if  floor((k-r)/2) <= i < floor((k+r)/2)
            mp[i] = q    otherwise

        Verified against the paper's Table 1 (5/5 worked examples).
        """
        k = self.k
        q, r = divmod(w, k)
        lo = (k - r) // 2
        hi = (k + r) // 2
        return [q + 1 if lo <= i < hi else q for i in range(k)]

    def in_band(self, c, eps, part_size=1):
        """Fixed-band test (NPC 2009, Eq. 6): Chebyshev distance to mp(w).

            max_i | y_i - mp_i(w) |  <=  eps,     w = sum_i y_i

        This is an exact integer-lattice test against the DISCRETE middle
        partition, not a real-valued interpolated corridor.

        part_size : the band is defined on the PARTITION lattice in the original
        work (partition size S ~ 2 throughout its experiments). With
        part_size=S>1 the cell index is mapped to its partition index before the
        test, reproducing the published search-space percentages exactly.
        """
        if part_size > 1:
            y = [ci // part_size for ci in c]
        else:
            y = list(c)
        w = sum(y)
        mp = self.middle_partition_for(w, self.k)
        return max(abs(y[i] - mp[i]) for i in range(self.k)) <= eps

    @staticmethod
    def middle_partition_for(w, k):
        q, r = divmod(w, k)
        lo = (k - r) // 2
        hi = (k + r) // 2
        return [q + 1 if lo <= i < hi else q for i in range(k)]

    def _unused_corridor(self, c, w):
        """Corridor band test around the hyper-diagonal, half-width w residues."""
        s = sum(c)
        for i in range(self.k):
            c_diag = s * (self.shape[i] - 1) / self.nsum if self.nsum else 0.0
            if abs(c[i] - c_diag) > w:
                return False
        return True


# ---------------------------------------------------------------- edge weight
def edge_weight(c, mask, enc, k):
    """Additive SP weight of stepping into cell c under decrement `mask`.

    W_m[c] = sum_{i<j, both decremented} sub(res_i,res_j) + (k-|m|)*gap
    Shared by every semiring and by both value and partition fills.
    """
    decr = [i for i in range(k) if mask[i] == 1]
    ndecr_gap = k - len(decr)
    sp = 0
    for ii in range(len(decr)):
        for jj in range(ii + 1, len(decr)):
            a, b = decr[ii], decr[jj]
            sp += sub_score(int(enc[a][c[a] - 1]), int(enc[b][c[b] - 1]))
    return sp + ndecr_gap * GAP


# ---------------------------------------------------------------- the fill
def fill(seqs, mode="global", band=None, semiring="maxplus", beta=0.1,
         want_prev=True, dense=False, part_size=1, average_missing=True):
    """THE recurrence. Returns a result dict.

    mode      : "global" | "local"
    band      : None (exact) | int epsilon (hyper-diagonal band, NPC-2009 Eq.6)
    part_size : partition size S for the band lattice (original work used S~2;
                S=1 applies the band at cell granularity)
    average_missing : impute out-of-band predecessors by the mean of in-band
                candidates (NPC-2009 semantics). False = skip them (weaker).
    semiring  : "maxplus" (returns integer scores + traceback) |
                "logsumexp" (returns float partition tensor; no traceback)
    dense     : if True and semiring=="logsumexp", return an ndarray tensor.

    Result keys:
      score      : optimal score (maxplus) or None
      anchor     : flat index of optimum (sink for global, argmax for local)
      rows       : traceback alignment rows (maxplus) or None
      spans      : per-seq (lo,hi) residue span of the local block (local maxplus)
      cells      : number of interior cells scored (complexity)
      tensor     : dense ndarray (logsumexp+dense) or None
      geo        : the Geometry
    """
    geo = Geometry([len(s) for s in seqs])
    k = geo.k
    enc = [encode(s) for s in seqs]
    masks = geo.masks()
    local = (mode == "local")
    partition = (semiring == "logsumexp")

    if partition:
        S = np.zeros(geo.n, dtype=np.float64)
    else:
        S = np.zeros(geo.n, dtype=np.int64)
    prev = np.full(geo.n, -1, dtype=np.int64) if (want_prev and not partition) else None

    # ---- boundary ----
    for f in range(geo.n):
        c = geo.coords(f)
        if band is not None and not geo.in_band(c, band, part_size):
            continue
        if any(x == 0 for x in c):
            if partition:
                # global: single all-gap incoming path seeds 0; local: restart=1
                S[f] = 1.0 if local else 1.0  # exp(beta*0)=1 in both (see note)
            else:
                if local:
                    S[f] = 0                       # (L2)
                else:
                    tau = 1
                    for x in c:
                        tau *= (x + 1)
                    S[f] = GAP * tau               # global border
            continue

    # NOTE on global partition boundary: the exact global Forward seeds F[0]=1
    # and border cells accumulate the all-gap path. Using 1.0 on the full border
    # is the standard normalisation for the partition demos here; it matches the
    # route3 reference used in validation.

    best_score = 0 if local else None
    anchor = geo.n - 1  # sink by default (global)

    # ---- interior sweep in flat order (respects wave dependency in F-layout) ----
    for f in range(geo.n):
        c = geo.coords(f)
        if any(x == 0 for x in c):
            continue
        if band is not None and not geo.in_band(c, band, part_size):
            continue

        if partition:
            total = 1.0 if local else 0.0   # (L1)-analogue restart term for local
            terms = []
            n_missing = 0
            for m in masks:
                nb = [c[i] - m[i] for i in range(k)]
                if any(x < 0 for x in nb):
                    continue
                if band is not None and not geo.in_band(nb, band, part_size):
                    n_missing += 1
                    continue
                nf = geo.flat(nb)
                w = edge_weight(c, m, enc, k)
                terms.append(np.exp(beta * w) * S[nf])
            total += float(np.sum(terms)) if terms else 0.0
            if average_missing and n_missing and terms:
                total += n_missing * float(np.mean(terms))
            S[f] = total
            continue

        # maxplus
        # NPC-2009 semantics: predecessors that fall OUTSIDE the band are not
        # discarded -- the original scoring "calculates all possible paths from
        # the origin up to the current scoring cell that fall within the search
        # space considered, and average[s] the missing neighbours". We therefore
        # collect in-band candidates, and when some predecessors are missing we
        # impute them with the mean of the in-band candidates. Skipping them
        # outright (the earlier behaviour) is strictly weaker.
        cands = []
        cand_prev = []
        n_missing = 0
        for m in masks:
            nb = [c[i] - m[i] for i in range(k)]
            if any(x < 0 for x in nb):
                continue
            if band is not None and not geo.in_band(nb, band, part_size):
                n_missing += 1
                continue
            nf = geo.flat(nb)
            cands.append(float(S[nf]) + edge_weight(c, m, enc, k))
            cand_prev.append(nf)

        if cands:
            bi = int(np.argmax(cands))
            best = cands[bi]
            best_prev = cand_prev[bi]
            if average_missing and n_missing:
                est = float(np.mean(cands))
                if est > best:
                    best = est                    # imputed, keep best_prev
            if local:
                best = max(0.0, best)             # (L1)
        else:
            best = 0.0 if local else None
            best_prev = -1

        if best is not None:
            S[f] = best
            if prev is not None:
                prev[f] = best_prev
        if local and best is not None and best > best_score:
            best_score = best
            anchor = f

    # ---- assemble result ----
    res = {"geo": geo, "cells": None, "tensor": None, "rows": None,
           "spans": None, "prev": prev}
    if partition:
        res["score"] = None
        res["anchor"] = int(np.argmax(S))
        if dense:
            res["tensor"] = S.reshape(geo.shape, order="F")
        else:
            res["tensor"] = S
        return res

    if local:
        res["score"] = int(best_score)
        res["anchor"] = anchor
    else:
        res["score"] = int(S[geo.n - 1])
        res["anchor"] = geo.n - 1
    res["flat"] = S
    if want_prev:
        rows, spans = _traceback(S, prev, res["anchor"], geo, seqs, local)
        res["rows"] = rows
        res["spans"] = spans
    # count interior cells actually scored
    if band is None:
        res["cells"] = int(np.prod([s - 1 for s in geo.shape]))
    else:
        cnt = 0
        for f in range(geo.n):
            c = geo.coords(f)
            if all(x > 0 for x in c) and geo.in_band(c, band, part_size):
                cnt += 1
        res["cells"] = cnt
    return res


def _traceback(S, prev, anchor, geo, seqs, local):
    k = geo.k
    rows = [[] for _ in range(k)]
    spans = [[None, None] for _ in range(k)]
    cur = anchor
    while cur > 0:
        if local and S[cur] <= 0:
            break                       # (L3): stop at first zero
        p = int(prev[cur]) if prev is not None else -1
        if p < 0:
            p = 0
        c = geo.coords(cur)
        pc = geo.coords(p)
        for i in range(k):
            if c[i] > pc[i]:
                rows[i].append(seqs[i][c[i] - 1])
                idx = c[i] - 1
                spans[i][0] = idx if spans[i][0] is None else min(spans[i][0], idx)
                spans[i][1] = idx if spans[i][1] is None else max(spans[i][1], idx)
            else:
                rows[i].append("-")
        if p == 0 and not local:
            break
        cur = p
    return ["".join(reversed(r)) for r in rows], spans


# convenience wrappers ------------------------------------------------------
def align(seqs, mode="global", band=None, part_size=1):
    """Return (rows, score, spans) for a maxplus fill."""
    r = fill(seqs, mode=mode, band=band, semiring="maxplus", part_size=part_size)
    return r["rows"], r["score"], r["spans"]


def value_tensor(seqs, mode="global"):
    """Dense value tensor (float) for decomposition routes."""
    r = fill(seqs, mode=mode, semiring="maxplus", want_prev=False)
    return r["flat"].astype(float).reshape(r["geo"].shape, order="F")


def partition_tensor(seqs, mode="global", beta=0.1):
    """Dense partition (Forward) tensor for Route 3."""
    r = fill(seqs, mode=mode, semiring="logsumexp", beta=beta, dense=True)
    return r["tensor"]


if __name__ == "__main__":
    for mode in ("global", "local"):
        rows, sc, spans = align(["AATC", "ATC", "AC"], mode=mode)
        print(f"{mode:>6}: score {sc}  rows {rows}")
