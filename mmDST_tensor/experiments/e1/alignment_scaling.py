#!/usr/bin/env python3
"""
alignment_scaling.py -- REWRITTEN scaling experiment.

Two changes from the original tensor-scaling study (which used identical
"ACGT"*3 repeats and reported a size-growth "compression factor"):

  1. ALIGNMENT SCALING, not tensor scaling. We vary sequence LENGTH and
     DIVERGENCE (mutation rate between related sequences), and report how the
     decomposition fidelity of the *partition tensor* behaves -- i.e. how well
     the alignment's soft-score structure compresses as the alignment problem
     gets harder, not merely as the array gets bigger.

  2. REALISTIC SEQUENCES. Identical repeats make the tensor trivially low-rank
     and flatter the method. We generate a common ancestor and mutate each
     sequence independently at a controlled rate, so the tensor reflects a real
     (divergent) alignment problem.

We also test three concrete levers for IMPROVING reconstruction error, so the
paper can report an honest improvement rather than a fixed number:

  (A) beta (inverse temperature): smaller beta -> smoother partition field ->
      lower TT-rank needed. We sweep beta and report the error/rank frontier.
  (B) log-domain decomposition: decompose log F instead of F, which tames the
      dynamic range that inflates Frobenius error on the raw partition tensor.
  (C) rank budget: the honest accuracy/compression trade, reported as a curve
      rather than a single cherry-picked point.

Outputs CSVs the paper's tables/figures are built from.
"""

from __future__ import annotations
import csv
import json
import random
import numpy as np
from pathlib import Path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/tensor"))
from core import fill
from decomp import tt_decompose, tt_reconstruct, rel_error

BASES = "ACGT"
OUT = Path(__file__).parent / "results_alignment"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- sequences
def ancestor(length, rng):
    return "".join(rng.choice(BASES) for _ in range(length))


def mutate(seq, rate, rng):
    return "".join(rng.choice([b for b in BASES if b != c]) if rng.random() < rate
                    else c for c in seq)


def divergent_set(k, length, rate, rng):
    """A common ancestor mutated independently k times -> a realistic,
    non-trivial alignment problem (unlike identical repeats)."""
    anc = ancestor(length, rng)
    return [mutate(anc, rate, rng) for _ in range(k)]


# ---------------------------------------------------------------- tensors
def partition_tensor(seqs, beta):
    r = fill(seqs, mode="global", semiring="logsumexp", beta=beta, dense=True)
    return r["tensor"]


def value_tensor(seqs):
    r = fill(seqs, mode="global", semiring="maxplus", want_prev=False)
    return r["flat"].astype(float).reshape(r["geo"].shape, order="F")


# ---------------------------------------------------------------- fidelity
def tt_fidelity(T, ranks, log_domain=False):
    """Decompose T (optionally in log domain) at given TT ranks; return
    (rel_error, compression, sink_error)."""
    sink = tuple(s - 1 for s in T.shape)
    if log_domain:
        # shift so all entries positive, decompose log, exponentiate back
        base = np.log(np.clip(T, 1e-300, None))
        cores = tt_decompose(base, ranks=ranks)
        rec = tt_reconstruct(cores)
        That = np.exp(rec)
        params = sum(c.size for c in cores)
    else:
        cores = tt_decompose(T, ranks=ranks)
        That = tt_reconstruct(cores)
        params = sum(c.size for c in cores)
    err = rel_error(T, That)
    comp = T.size / max(1, params)
    # sink fidelity in the meaningful (log) domain for the partition tensor
    lz_true = np.log(max(T[sink], 1e-300))
    lz_hat = np.log(max(That[sink], 1e-300))
    return err, comp, abs(lz_hat - lz_true)


# ---------------------------------------------------------------- experiments
def exp_divergence(rng):
    """Alignment-scaling: fix k and length, vary DIVERGENCE. Report how
    partition-tensor compressibility degrades as sequences diverge."""
    rows = []
    k, length = 3, 12
    for rate in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
        # average over several realisations for stability
        errs, comps, lzs = [], [], []
        for _ in range(5):
            seqs = divergent_set(k, length, rate, rng)
            F = partition_tensor(seqs, beta=0.1)
            e, c, lz = tt_fidelity(F, ranks=[3] * (k - 1))
            errs.append(e); comps.append(c); lzs.append(lz)
        rows.append(dict(divergence=rate, rel_error=np.mean(errs),
                         compression=np.mean(comps), logZ_error=np.mean(lzs)))
    _write(rows, "divergence.csv",
           ["divergence", "rel_error", "compression", "logZ_error"])
    return rows


def exp_beta(rng):
    """Lever A: sweep beta. Smaller beta -> smoother field -> better fidelity
    at fixed rank. Report the frontier."""
    rows = []
    seqs = divergent_set(3, 12, 0.15, rng)
    for beta in [0.02, 0.05, 0.1, 0.2, 0.4]:
        F = partition_tensor(seqs, beta=beta)
        e, c, lz = tt_fidelity(F, ranks=[3, 3])
        rows.append(dict(beta=beta, rel_error=e, compression=c, logZ_error=lz))
    _write(rows, "beta_sweep.csv", ["beta", "rel_error", "compression", "logZ_error"])
    return rows


def exp_logdomain(rng):
    """Lever B: raw vs log-domain decomposition of the partition tensor."""
    rows = []
    seqs = divergent_set(3, 12, 0.15, rng)
    F = partition_tensor(seqs, beta=0.1)
    for name, logd in [("raw", False), ("log-domain", True)]:
        e, c, lz = tt_fidelity(F, ranks=[3, 3], log_domain=logd)
        rows.append(dict(method=name, rel_error=e, compression=c, logZ_error=lz))
    _write(rows, "logdomain.csv", ["method", "rel_error", "compression", "logZ_error"])
    return rows


def exp_rank_budget(rng):
    """Lever C: the honest accuracy/compression curve at fixed problem."""
    rows = []
    seqs = divergent_set(3, 14, 0.15, rng)
    F = partition_tensor(seqs, beta=0.1)
    for r in [1, 2, 3, 4, 5, 6]:
        e, c, lz = tt_fidelity(F, ranks=[r, r])
        rows.append(dict(rank=r, rel_error=e, compression=c, logZ_error=lz))
    _write(rows, "rank_budget.csv", ["rank", "rel_error", "compression", "logZ_error"])
    return rows


def exp_value_vs_partition(rng):
    """The headline contrast, on realistic sequences: value tensor breaks,
    partition tensor holds -- reported at matched compression."""
    rows = []
    for k in [2, 3, 4]:
        seqs = divergent_set(k, 10, 0.15, rng)
        V = value_tensor(seqs)
        F = partition_tensor(seqs, beta=0.1)
        vsink = tuple(s - 1 for s in V.shape)
        # value tensor: optimum error
        cores = tt_decompose(V, ranks=[3] * (k - 1))
        Vhat = tt_reconstruct(cores)
        v_opt_err = abs(Vhat[vsink] - V[vsink])
        v_relerr = rel_error(V, Vhat)
        v_comp = V.size / sum(c.size for c in cores)
        # partition tensor: logZ error
        fe, fc, flz = tt_fidelity(F, ranks=[3] * (k - 1))
        rows.append(dict(k=k, value_relerr=round(v_relerr, 4),
                         value_opt_error=round(v_opt_err, 2), value_comp=round(v_comp, 1),
                         partition_relerr=round(fe, 4), partition_logZ_error=round(flz, 6),
                         partition_comp=round(fc, 1)))
    _write(rows, "value_vs_partition.csv",
           ["k", "value_relerr", "value_opt_error", "value_comp",
            "partition_relerr", "partition_logZ_error", "partition_comp"])
    return rows


def _write(rows, name, fields):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    rng = random.Random(42)
    print("== Alignment-scaling experiments (realistic divergent sequences) ==\n")

    print("[1] Value vs partition on realistic sequences:")
    for r in exp_value_vs_partition(rng):
        print(f"  k={r['k']}: value opt_err={r['value_opt_error']:>7} "
              f"(rel {r['value_relerr']}) | partition logZ_err={r['partition_logZ_error']:.2e}")

    print("\n[2] Divergence sweep (partition tensor, fixed rank 3):")
    for r in exp_divergence(rng):
        print(f"  div={r['divergence']:.2f}: rel_err={r['rel_error']:.4f} "
              f"logZ_err={r['logZ_error']:.2e} comp={r['compression']:.1f}x")

    print("\n[3] Lever A -- beta sweep (fidelity vs smoothness):")
    for r in exp_beta(rng):
        print(f"  beta={r['beta']:.2f}: rel_err={r['rel_error']:.4f} logZ_err={r['logZ_error']:.2e}")

    print("\n[4] Lever B -- raw vs log-domain decomposition:")
    for r in exp_logdomain(rng):
        print(f"  {r['method']:>10}: rel_err={r['rel_error']:.4f} logZ_err={r['logZ_error']:.2e}")

    print("\n[5] Lever C -- rank/accuracy frontier:")
    for r in exp_rank_budget(rng):
        print(f"  rank={r['rank']}: rel_err={r['rel_error']:.4f} "
              f"logZ_err={r['logZ_error']:.2e} comp={r['compression']:.1f}x")

    print(f"\nCSVs written to {OUT}/")


if __name__ == "__main__":
    main()
