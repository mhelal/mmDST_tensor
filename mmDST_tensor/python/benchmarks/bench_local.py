"""
bench_local.py -- LOCAL benchmark: motif recovery on ground truth, complexity,
band vs exact, and (via routes.py) reconstruction error. Calls core.py.

The LOCAL half of the unified benchmarking: SAME core as bench_global, different
dataset (implanted-motif ground truth) and metrics (nSn/nPPV/nPC/F1 position
recovery, plus complexity and local-tensor reconstruction error).
"""
from __future__ import annotations
import argparse, random, time, tracemalloc
import numpy as np
from core import fill
from banded import full_fill, banded_fill
from routes import route2_value, route3_partition

BASES = "ACGT"


def rseq(n, rng): return "".join(rng.choice(BASES) for _ in range(n))
def mutate(m, rate, rng):
    return "".join(rng.choice([b for b in BASES if b != c]) if rng.random() < rate else c for c in m)


def make_case(k, bg, motif, mr, rng):
    seqs, truth = [], []
    for _ in range(k):
        b = rseq(bg, rng); pos = rng.randint(0, bg); mm = mutate(motif, mr, rng)
        seqs.append(b[:pos] + mm + b[pos:]); truth.append(set(range(pos, pos + len(mm))))
    return seqs, truth


def predicted(seqs, band=None, part_size=1):
    r = fill(seqs, mode="local", band=band, semiring="maxplus", part_size=part_size)
    pred = []
    for lo, hi in r["spans"]:
        pred.append(set() if lo is None else set(range(lo, hi + 1)))
    return pred, r["score"], r["cells"]


def recovery(truth, pred):
    TP = sum(len(t & p) for t, p in zip(truth, pred))
    FP = sum(len(p - t) for t, p in zip(truth, pred))
    FN = sum(len(t - p) for t, p in zip(truth, pred))
    nSn = TP / (TP + FN) if TP + FN else float("nan")
    nPPV = TP / (TP + FP) if TP + FP else float("nan")
    nPC = TP / (TP + FN + FP) if TP + FN + FP else float("nan")
    f1 = 2 * nSn * nPPV / (nSn + nPPV) if nSn and nPPV else float("nan")
    return nSn, nPPV, nPC, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", default="ACGTGACGTGCA")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bg", type=int, default=14)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--band", type=int, default=None, help="epsilon (NPC-2009 band radius); local is OUT OF SCOPE of the original method")
    ap.add_argument("--routes", action="store_true")
    args = ap.parse_args()
    rng = random.Random(1)
    print(f"LOCAL benchmark: motif '{args.motif}' k={args.k} bg={args.bg} "
          f"{args.trials} trials/rate\n")
    print(f"{'mut':>5} {'nSn':>6} {'nPPV':>6} {'nPC':>6} {'F1':>6} {'score':>7} {'sec':>7} {'cells':>7}")
    for mr in (0.0, 0.1, 0.2, 0.3):
        agg = {"nSn": [], "nPPV": [], "nPC": [], "F1": [], "sc": [], "sec": [], "cells": []}
        for _ in range(args.trials):
            seqs, truth = make_case(args.k, args.bg, args.motif, mr, rng)
            t0 = time.perf_counter()
            pred, sc, cells = predicted(seqs, band=args.band)
            agg["sec"].append(time.perf_counter() - t0)
            nSn, nPPV, nPC, f1 = recovery(truth, pred)
            for kk, v in zip(("nSn", "nPPV", "nPC", "F1"), (nSn, nPPV, nPC, f1)):
                if not np.isnan(v): agg[kk].append(v)
            agg["sc"].append(sc); agg["cells"].append(cells)
        m = lambda x: round(np.mean(x), 3) if x else float("nan")
        print(f"{mr:>5} {m(agg['nSn']):>6} {m(agg['nPPV']):>6} {m(agg['nPC']):>6} "
              f"{m(agg['F1']):>6} {m(agg['sc']):>7} {m(agg['sec']):>7} {m(agg['cells']):>7.0f}")

    if args.routes:
        seqs = ["ACGTGACGT", "TACGTGACG"]
        print("\nRECONSTRUCTION (local tensors):")
        print(" Route2 value:", route2_value(seqs, "local")[2])
        print(" Route3 partition:", route3_partition(seqs, "local")[1])


if __name__ == "__main__":
    main()
