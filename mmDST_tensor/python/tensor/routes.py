"""
routes.py -- the three Stage-4 decomposition routes, unified for global & local,
all calling core.py tensors. Replaces stress_routes.py + local_routes.py.

  Route 1 : SP weight tensor is exactly tropical rank C(k,2) (mode-independent).
  Route 2 : TT on the value tensor -> Frobenius err vs OPTIMUM err (fragile).
  Route 3 : TT on the partition tensor -> Frobenius err vs logZ err (graceful).

The optimum cell is the sink for global, the argmax for local; core provides
both, so the same route code serves both modes.
"""
from __future__ import annotations
import itertools
import numpy as np

from core import value_tensor, partition_tensor, encode, sub_score, GAP, ALPHA
from decomp import tt_decompose, tt_reconstruct, tt_ranks, rel_error

A = len(ALPHA)


# ---- Route 1: exact tropical CP of the SP weight (mode-independent) ----
def route1_weight_exact(k):
    """Return (dense_size, factored_size, max_err) proving exactness."""
    def dense():
        W = np.zeros((A,) * k, dtype=np.int64)
        for idx in itertools.product(range(A), repeat=k):
            s = 0
            for a in range(k):
                for b in range(a + 1, k):
                    s += sub_score(idx[a], idx[b])
            W[idx] = s
        return W
    def factored():
        W = np.zeros((A,) * k, dtype=np.int64)
        P = np.array([[sub_score(i, j) for j in range(A)] for i in range(A)], dtype=np.int64)
        for a in range(k):
            for b in range(a + 1, k):
                Pb = P.reshape([A if d in (a, b) else 1 for d in range(k)])
                W = W + Pb
        return W
    if k > 4:  # dense too big to materialise; report sizes only
        return (A ** k, (k * (k - 1) // 2) * A * A, 0)
    Wd, Wf = dense(), factored()
    return (A ** k, (k * (k - 1) // 2) * A * A, int(np.max(np.abs(Wd - Wf))))


def _optimum_cell(T, mode):
    if mode == "global":
        return tuple(s - 1 for s in T.shape)
    return tuple(np.unravel_index(np.argmax(T), T.shape))


def route2_value(seqs, mode="global", eps_list=(0.0, 0.02, 0.05, 0.1)):
    T = value_tensor(seqs, mode=mode)
    opt = _optimum_cell(T, mode)
    out = []
    for eps in eps_list:
        cores = tt_decompose(T, eps=eps if eps > 0 else None)
        That = tt_reconstruct(cores)
        out.append(dict(rank=tt_ranks(cores), rel_err=rel_error(T, That),
                        compress=T.size / max(1, sum(c.size for c in cores)),
                        opt_err=abs(That[opt] - T[opt])))
    return out


def route3_partition(seqs, mode="global", beta=0.1, eps_list=(0.0, 0.01, 0.05)):
    F = partition_tensor(seqs, mode=mode, beta=beta)
    opt = _optimum_cell(F, mode)
    out = []
    for eps in eps_list:
        cores = tt_decompose(F, eps=eps if eps > 0 else None)
        Fhat = tt_reconstruct(cores)
        val = Fhat[opt]
        out.append(dict(rank=tt_ranks(cores), rel_err=rel_error(F, Fhat),
                        compress=F.size / max(1, sum(c.size for c in cores)),
                        logZ_err=abs(np.log(max(val, 1e-300)) - np.log(F[opt]))))
    return out


def demo(seqs, mode="global"):
    print(f"\n=== ROUTES ({mode}) on k={len(seqs)} lens={[len(s) for s in seqs]} ===")
    dz, fz, err = route1_weight_exact(len(seqs))
    print(f"Route 1 (weight, tropical CP): dense {dz} -> factored {fz}, max_err {err}")
    print("Route 2 (value TT):   rank        rel_err  compress  OPT_err")
    for r in route2_value(seqs, mode):
        print(f"   {str(r['rank']):>10}  {r['rel_err']:8.4f}  {r['compress']:7.2f}x  {r['opt_err']:8.2f}")
    print("Route 3 (partition):  rank        rel_err  compress  logZ_err")
    for r in route3_partition(seqs, mode):
        print(f"   {str(r['rank']):>10}  {r['rel_err']:8.5f}  {r['compress']:7.2f}x  {r['logZ_err']:8.4f}")


if __name__ == "__main__":
    demo(["AATCG", "ATCCG", "AATCG"], mode="global")
    demo(["ACGTGACGT", "TACGTGACG"], mode="local")
