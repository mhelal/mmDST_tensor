"""
decomp.py -- minimal, dependency-free tensor decompositions over the real field.

Implements the classical field-based decompositions we need for Routes 2 and 3,
using only numpy.linalg.  These are standard algorithms; kept short and audited
rather than pulled from a library so the field-vs-semiring distinction stays
explicit (these do NOT apply to the tropical DP tensor -- that is the whole point
of the comparison).

  tt_decompose / tt_reconstruct : Tensor-Train (TT-SVD, Oseledets 2011)
  tucker_hosvd / tucker_reconstruct : Tucker via truncated HOSVD
  cp_als / cp_reconstruct : CP / PARAFAC via alternating least squares

All operate on real ndarrays.  Error is reported as relative Frobenius norm.
"""

from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ TT-SVD
def tt_decompose(T, ranks=None, eps=None):
    """TT-SVD.  Returns list of cores G_i with shapes (r_{i-1}, n_i, r_i).

    ranks : optional list of internal bond dims (len = ndim-1) to cap at.
    eps   : optional relative truncation tolerance (per-mode).
    """
    T = np.asarray(T, dtype=float)
    shape = T.shape
    d = T.ndim
    cores = []
    r_prev = 1
    C = T.reshape(shape[0], -1)
    norm = np.linalg.norm(T)
    for k in range(d - 1):
        C = C.reshape(r_prev * shape[k], -1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        # choose rank
        r = len(S)
        if ranks is not None:
            r = min(r, ranks[k])
        if eps is not None and norm > 0:
            # keep singular values until tail energy < eps
            tail = np.sqrt(np.cumsum(S[::-1] ** 2)[::-1])
            keep = np.searchsorted(-tail, -(eps * norm / np.sqrt(d - 1)))
            r = min(r, max(1, keep))
        r = max(1, r)
        U = U[:, :r]; S = S[:r]; Vt = Vt[:r, :]
        cores.append(U.reshape(r_prev, shape[k], r))
        C = np.diag(S) @ Vt
        r_prev = r
    cores.append(C.reshape(r_prev, shape[d - 1], 1))
    return cores


def tt_reconstruct(cores):
    T = cores[0]  # (1, n0, r0)
    for G in cores[1:]:
        # contract last bond of T with first bond of G
        T = np.tensordot(T, G, axes=([T.ndim - 1], [0]))
    return T.reshape([G.shape[1] for G in [cores[0]] + cores[1:]] if False
                     else [c.shape[1] for c in cores])


def tt_ranks(cores):
    return [c.shape[2] for c in cores[:-1]]


# --------------------------------------------------------------- Tucker/HOSVD
def _unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def tucker_hosvd(T, ranks):
    """Truncated HOSVD.  Returns (core, factors)."""
    T = np.asarray(T, dtype=float)
    factors = []
    for mode in range(T.ndim):
        U, _, _ = np.linalg.svd(_unfold(T, mode), full_matrices=False)
        factors.append(U[:, :ranks[mode]])
    core = T.copy()
    for mode, U in enumerate(factors):
        core = np.tensordot(U.T, core, axes=([1], [mode]))
        core = np.moveaxis(core, 0, mode)
    return core, factors


def tucker_reconstruct(core, factors):
    T = core
    for mode, U in enumerate(factors):
        T = np.tensordot(U, T, axes=([1], [mode]))
        T = np.moveaxis(T, 0, mode)
    return T


# ------------------------------------------------------------------ CP / ALS
def cp_als(T, rank, iters=200, seed=0):
    """CP decomposition via ALS.  Returns (weights, factors)."""
    T = np.asarray(T, dtype=float)
    rng = np.random.default_rng(seed)
    d = T.ndim
    factors = [rng.standard_normal((T.shape[m], rank)) for m in range(d)]
    for _ in range(iters):
        for mode in range(d):
            # Khatri-Rao of all other factors
            V = np.ones((rank, rank))
            for m in range(d):
                if m != mode:
                    V *= factors[m].T @ factors[m]
            kr = _khatri_rao([factors[m] for m in range(d) if m != mode])
            unf = _unfold(T, mode)
            factors[mode] = unf @ kr @ np.linalg.pinv(V)
    # normalise columns, fold norms into weights
    weights = np.ones(rank)
    for mode in range(d):
        nrm = np.linalg.norm(factors[mode], axis=0)
        nrm[nrm == 0] = 1
        factors[mode] /= nrm
        weights *= nrm
    return weights, factors


def _khatri_rao(mats):
    # column-wise Khatri-Rao (matching cp_als unfolding order)
    R = mats[0].shape[1]
    out = mats[0]
    for M in mats[1:]:
        out = (out[:, None, :] * M[None, :, :]).reshape(-1, R)
    return out


def cp_reconstruct(weights, factors):
    d = len(factors)
    shape = [f.shape[0] for f in factors]
    R = len(weights)
    T = np.zeros(shape)
    for r in range(R):
        term = weights[r]
        vecs = [factors[m][:, r] for m in range(d)]
        outer = vecs[0]
        for v in vecs[1:]:
            outer = np.multiply.outer(outer, v)
        T += term * outer
    return T


def rel_error(T, That):
    T = np.asarray(T, dtype=float)
    n = np.linalg.norm(T)
    return np.linalg.norm(T - That) / (n if n > 0 else 1.0)


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # exact low-rank TT tensor should reconstruct near-perfectly at right ranks
    T = rng.standard_normal((4, 5, 3, 4))
    cores = tt_decompose(T)
    print("TT full-rank rel err:", rel_error(T, tt_reconstruct(cores)))
    core, facs = tucker_hosvd(T, [4, 5, 3, 4])
    print("Tucker full-rank rel err:", rel_error(T, tucker_reconstruct(core, facs)))
    w, f = cp_als(T, rank=20, iters=100)
    print("CP rank-20 rel err:", rel_error(T, cp_reconstruct(w, f)))
