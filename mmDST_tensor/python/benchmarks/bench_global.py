"""
bench_global.py -- GLOBAL benchmark on BAliBASE: score, complexity (time/memory),
band vs exact, and (via routes.py) reconstruction error. Calls core.py.

This is the GLOBAL half of the unified benchmarking: same core, BAliBASE data,
global-appropriate metrics (SP score, cells/time/memory, band accuracy gap,
reconstruction error on the global tensors).
"""
from __future__ import annotations
import argparse, csv, os, time, tracemalloc
import numpy as np
import balibase as bb
from banded import full_fill, banded_fill
from routes import route2_value, route3_partition

CELL_BUDGET = 200_000


def cube(lengths):
    n = 1
    for L in lengths: n *= (L + 1)
    return n


def measure(fn, *a):
    tracemalloc.start(); t0 = time.perf_counter()
    r = fn(*a); wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return r, wall, peak


PART_SIZE = 1
NO_AVG = False


def run_case(names, seqs, ref, band_w, budget=CELL_BUDGET):
    sl = [seqs[n] for n in names]; lengths = [len(s) for s in sl]
    row = {"k": len(names), "min_len": min(lengths), "max_len": max(lengths),
           "cube_cells": cube(lengths)}
    if row["cube_cells"] <= budget:
        (res, wall, mem) = measure(full_fill, sl, "global")
        sc, rows, cells, tb = res
        row.update(exact_score=sc, exact_sec=round(wall, 4),
                   exact_mem_mb=round(mem / 1e6, 2), exact_cells=cells)
        if ref:
            our = {names[i]: rows[i] for i in range(len(names))}
            sp, tc = bb.reference_sp_tc(our, ref)
            row.update(ref_SP=round(sp, 3), ref_TC=round(tc, 3))
    else:
        row.update(exact_score="SKIP", exact_sec="", exact_mem_mb="", exact_cells="")
    (res, wall, mem) = measure(banded_fill, sl, band_w, "global", PART_SIZE, not NO_AVG)
    bsc, brows, bcells, btb = res
    row.update(band_score=bsc, band_sec=round(wall, 4),
               band_mem_mb=round(mem / 1e6, 2), band_cells=bcells)
    if isinstance(row.get("exact_score"), int):
        row["band_gap"] = bsc - row["exact_score"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/claude/mmDST/mmDST-master/data/bbase")
    ap.add_argument("--refset", default="RV11")
    ap.add_argument("--case", default="BB11001")
    ap.add_argument("--band", type=int, default=2, help="epsilon (NPC-2009 Eq.6 band radius)")
    ap.add_argument("--part-size", type=int, default=1, help="partition size S for the band lattice")
    ap.add_argument("--no-average", action="store_true", help="skip out-of-band preds instead of averaging")
    ap.add_argument("--routes", action="store_true")
    ap.add_argument("--out", default="global_results.csv")
    args = ap.parse_args()
    tfa = os.path.join(args.data, args.refset, args.case + ".tfa")
    msf = os.path.join(args.data, args.refset, args.case + ".msf")
    names, seqs, ref = bb.load_case(tfa, msf)
    global PART_SIZE, NO_AVG
    PART_SIZE = args.part_size
    NO_AVG = args.no_average
    print(f"GLOBAL benchmark {args.case}: k={len(names)}  eps={args.band} S={PART_SIZE} "
          f"avg_missing={not NO_AVG}")

    def trunc(nm, L): return {n: seqs[n][:L] for n in nm}
    rows = []
    # length sweep k=2
    for L in [5, 10, 20, 30, 50, 75, 100, 150, 200]:
        r = run_case(names[:2], trunc(names[:2], L), None, args.band)
        r["sweep"] = "length"; r["L"] = L; rows.append(r)
    # k sweep
    for kk in range(2, min(len(names), 6) + 1):
        r = run_case(names[:kk], trunc(names[:kk], 8), None, args.band)
        r["sweep"] = "k"; rows.append(r)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    print("\nLENGTH sweep (k=2): L, cube, exact_sec, exact_cells, band_cells, band_sec")
    for r in rows:
        if r["sweep"] == "length":
            print(f"  L={r['L']:>3} cube={r['cube_cells']:>7} "
                  f"exact={r['exact_sec']}s/{r['exact_cells']}c "
                  f"band={r['band_sec']}s/{r['band_cells']}c")
    print("\nK sweep: k, cube, exact(score/sec), band(score/cells), gap")
    for r in rows:
        if r["sweep"] == "k":
            print(f"  k={r['k']} cube={r['cube_cells']:>8} "
                  f"exact={r.get('exact_score')}/{r.get('exact_sec')}s "
                  f"band={r.get('band_score')}/{r.get('band_cells')}c "
                  f"gap={r.get('band_gap','')}")

    if args.routes:
        chosen = [seqs[n][:6] for n in names[:3]]
        print("\nRECONSTRUCTION (global tensors):")
        print(" Route2 value:", route2_value(chosen, "global")[2])
        print(" Route3 partition:", route3_partition(chosen, "global")[1])
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
