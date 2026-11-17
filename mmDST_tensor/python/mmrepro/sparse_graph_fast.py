"""Fast sparse-graph construction for banded k-way MSA.

Drop-in companion to sparse_dp.py. It preserves SparseGraph semantics but
replaces Python tuple-dictionary predecessor discovery with mixed-radix integer
keys + vectorised NumPy search. This avoids one Python dict lookup for every
(state, transition-mask) pair.
"""
from __future__ import annotations
import math
import numpy as np
from .sparse_dp import (
    SparseGraph, generate_band_cells, generate_full_cells, encode,
    GAP, MATCH, MIS, GAPI,
)


def _mixed_radix_strides(lengths):
    strides = np.ones(len(lengths), dtype=np.int64)
    acc = 1
    lim = np.iinfo(np.int64).max
    for i, L in enumerate(lengths):
        strides[i] = acc
        dim = int(L) + 1
        if acc > lim // dim:
            raise OverflowError("Cartesian coordinate key exceeds int64")
        acc *= dim
    return strides


def _keys(coords, strides):
    return coords.astype(np.int64, copy=False) @ strides


def _mask_bits(k):
    masks = []
    for code in range(1, 1 << k):
        active = np.fromiter((i for i in range(k) if code & (1 << i)), dtype=np.int16)
        masks.append(active)
    return masks


def _vector_edge_weight(coords, dst, active, enc, k):
    w = np.full(dst.size, (k - len(active)) * GAP, dtype=np.int32)
    if dst.size == 0:
        return w
    for aa in range(len(active)):
        i = int(active[aa])
        ai = enc[i][coords[dst, i] - 1]
        for bb in range(aa + 1, len(active)):
            j = int(active[bb])
            bj = enc[j][coords[dst, j] - 1]
            s = np.where(ai == bj, MATCH, np.where((ai == GAPI) | (bj == GAPI), GAP, MIS))
            w += s.astype(np.int32, copy=False)
    return w


def build_sparse_graph_fast(seqs, band, part_size=1, deltas=None):
    """Build SparseGraph using vectorised predecessor lookup.

    Graph generation remains support-only: no dense DP tensor is allocated.
    The two-pass edge builder first counts exact degrees, allocates CSR once,
    then fills predecessor ids and int32 edge weights in place.
    """
    lengths = tuple(map(len, seqs))
    k = len(seqs)
    if band is None:
        coords_list, wave_offsets = generate_full_cells(lengths)
    else:
        coords_list, wave_offsets = generate_band_cells(lengths, int(band), int(part_size), deltas)

    coords = np.asarray(coords_list, dtype=np.int32).reshape((-1, k))
    n = len(coords)
    if n == 0:
        return SparseGraph(coords, wave_offsets, np.zeros(1, np.int64),
                           np.empty(0, np.int64), np.empty(0, np.int32), lengths)

    strides = _mixed_radix_strides(lengths)
    key = _keys(coords, strides)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    masks = _mask_bits(k)

    # Pass 1: count only actual predecessor edges.
    deg = np.zeros(n, dtype=np.int32)
    for active in masks:
        valid = np.all(coords[:, active] > 0, axis=1)
        dst = np.flatnonzero(valid)
        if dst.size == 0:
            continue
        delta_key = int(strides[active].sum())
        pred_key = key[dst] - delta_key
        pos = np.searchsorted(skey, pred_key)
        ok = pos < n
        if np.any(ok):
            pos2 = pos[ok]
            ok_idx = np.flatnonzero(ok)
            found = skey[pos2] == pred_key[ok_idx]
            if np.any(found):
                deg[dst[ok_idx[found]]] += 1

    ptr = np.empty(n + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(deg, dtype=np.int64, out=ptr[1:])
    nedges = int(ptr[-1])
    src = np.empty(nedges, dtype=np.int64)
    ew = np.empty(nedges, dtype=np.int32)
    cursor = ptr[:-1].copy()
    enc = [encode(s) for s in seqs]

    # Pass 2: resolve sources and fill exact CSR allocation.
    for active in masks:
        valid = np.all(coords[:, active] > 0, axis=1)
        dst = np.flatnonzero(valid)
        if dst.size == 0:
            continue
        delta_key = int(strides[active].sum())
        pred_key = key[dst] - delta_key
        pos = np.searchsorted(skey, pred_key)
        ok = pos < n
        if not np.any(ok):
            continue
        oi = np.flatnonzero(ok)
        pos2 = pos[ok]
        found = skey[pos2] == pred_key[oi]
        if not np.any(found):
            continue
        dst2 = dst[oi[found]]
        src2 = order[pos2[found]].astype(np.int64, copy=False)
        write = cursor[dst2]
        src[write] = src2
        ew[write] = _vector_edge_weight(coords, dst2, active, enc, k)
        cursor[dst2] += 1

    return SparseGraph(coords, np.asarray(wave_offsets, np.int64), ptr, src, ew, lengths)
