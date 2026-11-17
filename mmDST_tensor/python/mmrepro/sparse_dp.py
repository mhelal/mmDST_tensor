"""Sparse, wave-generated k-way MSA dynamic programming.

This module restores the key property of the original C implementation:
no full Cartesian DP tensor is allocated or swept when a band is active.
Admissible partition/cell coordinates are generated directly by wave.

Two storage layers are used:
  * SparseGraph: compact coordinate array + predecessor edge lists, only for
    generated states. This is reusable by CPU and CUDA backends.
  * scores/traceback: one scalar per generated state, never product(shape).

The recurrence is dimension-invariant and supports global/local max-plus and
partition/log-sum-exp routes.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterable, Sequence, Optional
import numpy as np

GAP, MATCH, MIS = -6, 3, -2
ALPHA = "ACDEFGHIKLMNPQRSTVWY-XBZU*"
_LUT = {c: i for i, c in enumerate(ALPHA)}
GAPI = _LUT["-"]


def encode(seq: str) -> np.ndarray:
    xi = _LUT["X"]
    return np.asarray([_LUT.get(c.upper(), xi) for c in seq], dtype=np.int16)


def sub_score(a: int, b: int) -> int:
    if a == b:
        return MATCH
    if a == GAPI or b == GAPI:
        return GAP
    return MIS


def middle_partition(w: int, k: int) -> tuple[int, ...]:
    q, r = divmod(int(w), int(k))
    lo = (k - r) // 2
    hi = (k + r) // 2
    return tuple(q + 1 if lo <= i < hi else q for i in range(k))


def _zero_sum_offsets(k: int, eps: int) -> Iterable[tuple[int, ...]]:
    """Generate offsets in [-eps,eps]^k with sum zero without k-D filtering."""
    cur = [0] * k
    def rec(i: int, remaining: int):
        if i == k - 1:
            if -eps <= remaining <= eps:
                cur[i] = remaining
                yield tuple(cur)
            return
        lo = max(-eps, remaining - eps * (k - i - 1))
        hi = min(eps, remaining + eps * (k - i - 1))
        for x in range(lo, hi + 1):
            cur[i] = x
            yield from rec(i + 1, remaining - x)
    yield from rec(0, 0)


def _partition_shape(lengths: Sequence[int], part_size: int) -> tuple[int, ...]:
    # DP coordinates are 0..L inclusive. Partition indices cover these cells.
    return tuple((int(L) // part_size) + 1 for L in lengths)


def generate_partition_wave(lengths: Sequence[int], wave: int, eps: int,
                            part_size: int = 1,
                            delta: Optional[Sequence[int]] = None) -> list[tuple[int, ...]]:
    """Generate only admissible partition indices on one partition wave.

    delta is a translation in partition-index units. delta=0 is the original
    geometric hyper-diagonal; seed-anchored local bands supply nonzero delta.
    """
    k = len(lengths)
    pshape = _partition_shape(lengths, part_size)
    d = tuple(0 for _ in range(k)) if delta is None else tuple(map(int, delta))
    mp = middle_partition(wave, k)
    out = []
    # Any translated coordinate p on wave has p = mp + d + z. Because d need
    # not sum to zero, z must compensate to keep sum(p)==wave.
    required = -sum(d)
    cur = [0] * k
    def rec(i: int, remaining: int):
        if i == k - 1:
            if -eps <= remaining <= eps:
                cur[i] = remaining
                p = tuple(mp[j] + d[j] + cur[j] for j in range(k))
                if sum(p) == wave and all(0 <= p[j] < pshape[j] for j in range(k)):
                    out.append(p)
            return
        lo = max(-eps, remaining - eps * (k-i-1))
        hi = min(eps, remaining + eps * (k-i-1))
        for x in range(lo, hi+1):
            cur[i] = x
            rec(i+1, remaining-x)
    rec(0, required)
    return sorted(set(out))


def cells_in_partition(p: Sequence[int], lengths: Sequence[int], part_size: int) -> Iterable[tuple[int, ...]]:
    ranges = []
    for pi, L in zip(p, lengths):
        lo = int(pi) * part_size
        hi = min(lo + part_size, int(L) + 1)
        if lo > L:
            return
        ranges.append(range(lo, hi))
    yield from product(*ranges)


def generate_band_cells(lengths: Sequence[int], eps: int, part_size: int = 1,
                        deltas: Optional[Sequence[Sequence[int]]] = None,
                        include_zero: bool = True) -> tuple[list[tuple[int,...]], np.ndarray]:
    """Generate the union of one or more shifted bands, ordered by DP wave.

    Returns coordinates and wave offsets. No Cartesian tensor is enumerated.
    """
    k = len(lengths)
    if deltas is None:
        deltas = [tuple(0 for _ in range(k))]
    dd = [tuple(map(int, x)) for x in deltas]
    if include_zero and tuple(0 for _ in range(k)) not in dd:
        dd = [tuple(0 for _ in range(k))] + dd

    pshape = _partition_shape(lengths, part_size)
    max_pwave = sum(s - 1 for s in pshape)
    by_wave: dict[int, set[tuple[int,...]]] = {}
    for pw in range(max_pwave + 1):
        parts: set[tuple[int,...]] = set()
        for delta in dd:
            parts.update(generate_partition_wave(lengths, pw, eps, part_size, delta))
        for p in parts:
            for c in cells_in_partition(p, lengths, part_size):
                by_wave.setdefault(sum(c), set()).add(c)

    max_wave = sum(lengths)
    coords: list[tuple[int,...]] = []
    offsets = [0]
    for w in range(max_wave + 1):
        cw = sorted(by_wave.get(w, ()))
        coords.extend(cw)
        offsets.append(len(coords))
    return coords, np.asarray(offsets, dtype=np.int64)


def generate_full_cells(lengths: Sequence[int]) -> tuple[list[tuple[int,...]], np.ndarray]:
    """Exact ground truth generator by wave. This enumerates all states but still
    avoids allocating a dense tensor; use only for tractable exact controls."""
    k = len(lengths)
    shape = tuple(L+1 for L in lengths)
    by_wave = [[] for _ in range(sum(lengths)+1)]
    for c in product(*[range(s) for s in shape]):
        by_wave[sum(c)].append(c)
    coords=[]; offsets=[0]
    for cw in by_wave:
        coords.extend(cw); offsets.append(len(coords))
    return coords, np.asarray(offsets, dtype=np.int64)


def _masks(k: int) -> list[tuple[int,...]]:
    return [m for m in product((0,1), repeat=k) if any(m)]


def edge_weight(c: Sequence[int], mask: Sequence[int], enc: Sequence[np.ndarray]) -> int:
    k = len(c)
    decr = [i for i in range(k) if mask[i]]
    s = (k - len(decr)) * GAP
    for x in range(len(decr)):
        for y in range(x+1, len(decr)):
            a,b=decr[x],decr[y]
            s += sub_score(int(enc[a][c[a]-1]), int(enc[b][c[b]-1]))
    return int(s)


@dataclass
class SparseGraph:
    coords: np.ndarray          # [n,k] int32
    wave_offsets: np.ndarray    # [max_wave+2]
    edge_ptr: np.ndarray        # [n+1]
    edge_src: np.ndarray        # predecessor state ids
    edge_w: np.ndarray          # additive score per edge
    lengths: tuple[int,...]

    @property
    def n_states(self) -> int: return int(self.coords.shape[0])
    @property
    def n_edges(self) -> int: return int(self.edge_src.size)
    @property
    def dense_cells(self) -> int: return int(prod(L+1 for L in self.lengths))


def build_sparse_graph(seqs: Sequence[str], band: Optional[int], part_size: int = 1,
                       deltas: Optional[Sequence[Sequence[int]]] = None) -> SparseGraph:
    lengths = tuple(map(len, seqs)); k=len(seqs)
    if band is None:
        coords_list, wave_offsets = generate_full_cells(lengths)
    else:
        coords_list, wave_offsets = generate_band_cells(lengths, band, part_size, deltas)
    coords = np.asarray(coords_list, dtype=np.int32).reshape((-1,k))
    id_of = {tuple(map(int,c)): i for i,c in enumerate(coords_list)}
    enc=[encode(s) for s in seqs]; masks=_masks(k)
    ptr=[0]; src=[]; ew=[]
    for c in coords_list:
        for m in masks:
            nb=tuple(c[i]-m[i] for i in range(k))
            if any(x<0 for x in nb): continue
            j=id_of.get(nb)
            if j is None: continue
            src.append(j); ew.append(edge_weight(c,m,enc))
        ptr.append(len(src))
    return SparseGraph(coords, wave_offsets, np.asarray(ptr,np.int64),
                       np.asarray(src,np.int64), np.asarray(ew,np.float64), lengths)


def _global_boundary_score(c: Sequence[int]) -> float:
    tau=1
    for x in c: tau *= (int(x)+1)
    return float(GAP*tau)


def run_cpu(graph: SparseGraph, seqs: Sequence[str], mode="global", semiring="maxplus",
            beta=0.1, want_traceback=True) -> dict:
    """Sequential sparse graph evaluation, used as the reference CPU backend."""
    n=graph.n_states; local=(mode=="local"); partition=(semiring=="logsumexp")
    score=np.zeros(n, dtype=np.float64)
    prev=np.full(n,-1,dtype=np.int64) if want_traceback and not partition else None
    best=0.0; anchor=-1
    coord_to_id={tuple(map(int,c)):i for i,c in enumerate(graph.coords.tolist())}
    for i in range(n):
        c=graph.coords[i]
        if np.any(c==0):
            score[i]=1.0 if partition else (0.0 if local else _global_boundary_score(c))
            if local and not partition and score[i]>=best: anchor=i
            continue
        a,b=int(graph.edge_ptr[i]),int(graph.edge_ptr[i+1])
        if partition:
            total=1.0 if local else 0.0
            if b>a:
                total += float(np.sum(np.exp(beta*graph.edge_w[a:b])*score[graph.edge_src[a:b]]))
            score[i]=total
        else:
            if b>a:
                vals=score[graph.edge_src[a:b]]+graph.edge_w[a:b]
                q=int(np.argmax(vals)); v=float(vals[q]); p=int(graph.edge_src[a+q])
                if local and v<0: v=0.0; p=-1
            else:
                v=0.0 if local else -np.inf; p=-1
            score[i]=v
            if prev is not None: prev[i]=p
            if local and v>best: best=v; anchor=i
    sink=coord_to_id.get(tuple(graph.lengths), -1)
    if partition:
        functional=float(np.log(max(score[sink if mode=="global" else int(np.argmax(score))], 1e-300)))
        return dict(functional=functional, values=score, anchor=(sink if mode=="global" else int(np.argmax(score))),
                    cells=n, edges=graph.n_edges, graph=graph)
    if mode=="global": anchor=sink; best=float(score[sink]) if sink>=0 else -np.inf
    rows, spans = traceback_sparse(graph, seqs, score, prev, anchor, local) if want_traceback else (None,None)
    return dict(score=best, values=score, prev=prev, anchor=anchor, rows=rows, spans=spans,
                cells=n, edges=graph.n_edges, graph=graph)


def traceback_sparse(graph, seqs, values, prev, anchor, local):
    k=len(seqs); rows=[[] for _ in range(k)]; spans=[[None,None] for _ in range(k)]
    cur=int(anchor)
    while cur>=0:
        if local and values[cur]<=0: break
        p=int(prev[cur]) if prev is not None else -1
        c=graph.coords[cur]
        pc=np.zeros(k,dtype=int) if p<0 else graph.coords[p]
        for i in range(k):
            if c[i]>pc[i]:
                rows[i].append(seqs[i][int(c[i])-1]); idx=int(c[i])-1
                spans[i][0]=idx if spans[i][0] is None else min(spans[i][0],idx)
                spans[i][1]=idx if spans[i][1] is None else max(spans[i][1],idx)
            else: rows[i].append('-')
        if p<0: break
        cur=p
    return ["".join(reversed(r)) for r in rows], spans


def dense_value_tensor(seqs, mode="global"):
    """Exact dense tensor only for tractable decomposition experiments."""
    g=build_sparse_graph(seqs, band=None); r=run_cpu(g,seqs,mode,"maxplus",want_traceback=False)
    T=np.zeros(tuple(L+1 for L in g.lengths),dtype=float)
    for c,v in zip(g.coords,r['values']): T[tuple(c)]=v
    return T


def dense_partition_tensor(seqs, mode="global", beta=0.1):
    g=build_sparse_graph(seqs, band=None); r=run_cpu(g,seqs,mode,"logsumexp",beta,want_traceback=False)
    T=np.zeros(tuple(L+1 for L in g.lengths),dtype=float)
    for c,v in zip(g.coords,r['values']): T[tuple(c)]=v
    return T
