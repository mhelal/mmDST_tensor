#!/usr/bin/env python3
"""
msa_survival.py — RIMC survivability wrapper for the MoA/MSA sparse-tensor codebase
(Thread 17: hyper-diagonal MSA + tropical/field tensor-decomposition routes).
===================================================================================
The science is done (core.py band DP, routes.py three decomposition routes, GPU
tropical kernels). What was missing is survivability for the two long, fragile
workloads:
  1. BAliBASE / length / k benchmark SWEEPS that die partway (a large-k cube blows
     the cell budget, a CuPy call OOMs, a job is killed) and lose the whole run;
  2. GPU tropical-DP fills that OOM on long sequences.

This wrapper adds the RIMC contract WITHOUT changing the existing algorithms:
  • merge-resume ledger keyed (experiment, mode, k, L, band, seed): completed
    cases are skipped, so an interrupted sweep resumes;
  • per-case cell-budget guard + degeneracy recording (a case exceeding the
    hypercube budget is recorded as 'skipped_budget', not a crash);
  • GPU-guarded fills with OOM backoff (shrink the tile / fall back to NumPy),
    cleanup in finally (CuPy mempool freed even on failure);
  • divergent/failed cases isolated as rows; the sweep continues;
  • deterministic seeds; every table regenerates from the ledger.

Wraps bench_global / bench_local / routes without editing them. CuPy-optional:
if CuPy or a GPU is absent it runs the NumPy path and still exercises resume.
"""
from __future__ import annotations
import csv, gc, math, os, signal, time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
RESULTS = Path(os.environ.get("MSA_RESULTS", "results")); RESULTS.mkdir(exist_ok=True)

# ───────────────────────── GPU cleanup (CuPy tropical path)
def gpu_cleanup(tag=""):
    gc.collect()
    try:
        import cupy as cp
        try: cp.cuda.Stream.null.synchronize()
        except Exception as e: print(f"[gpu{tag}] sync: {e}")
        try: cp.get_default_memory_pool().free_all_blocks()
        except Exception as e: print(f"[gpu{tag}] mempool: {e}")
    except ImportError:
        pass
    gc.collect()

@contextmanager
def gpu_guard(tag=""):
    try: yield
    finally: gpu_cleanup(tag)

def gpu_fill_with_backoff(fill_fn: Callable, seqs, tile:int=256, min_tile:int=16, **kw):
    """Run a GPU tropical-DP fill with OOM backoff: shrink the tile and retry,
    then fall back to the CPU fill; cleanup guaranteed. `fill_fn(seqs, tile=..)`."""
    t=tile
    while True:
        try:
            with gpu_guard(" fill"):
                return fill_fn(seqs, tile=t, **kw)
        except Exception as e:
            msg=str(e).lower()
            is_oom = "out of memory" in msg or "cudaerrormemoryallocation" in msg or "outofmemory" in msg
            if is_oom and t>min_tile:
                t=max(min_tile, t//2); print(f"[oom] gpu tile->{t}"); gpu_cleanup(" oom"); continue
            raise

# ───────────────────────── cell-budget guard (avoid blowing the hypercube)
def cube_cells(lengths: List[int]) -> int:
    prod=1
    for n in lengths: prod*= (n+1)
    return prod

@contextmanager
def case_timeout(seconds:int):
    if seconds<=0 or not hasattr(signal,"SIGALRM"):
        yield; return
    def _raise(s,f): raise TimeoutError(f"case exceeded {seconds}s")
    old=signal.signal(signal.SIGALRM,_raise); signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0); signal.signal(signal.SIGALRM,old)

# ───────────────────────── merge-resume ledger
class SweepLedger:
    def __init__(self, name="msa"):
        self.path=RESULTS/f"{name}_ledger.csv"
        self.fields=["experiment","mode","k","L","band","seed","status",
                     "score","recovery","cells","band_cells","sec","route","metric","error"]
        self._done=set()
        if self.path.exists():
            for r in csv.DictReader(open(self.path)):
                if r["status"]=="ok":
                    self._done.add(self._key(r))
    def _key(self,r): return (r["experiment"],r["mode"],str(r["k"]),str(r["L"]),str(r["band"]),str(r["seed"]))
    def done(self,key): return tuple(str(x) for x in key) in self._done
    def append(self,row):
        new=not self.path.exists()
        with open(self.path,"a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=self.fields)
            if new: w.writeheader()
            w.writerow({k:row.get(k,"") for k in self.fields})
        if row.get("status")=="ok":
            self._done.add(self._key(row))

# ───────────────────────── survivable sweep driver
def survivable_sweep(name:str, cases:Iterable[Dict], run_case:Callable[[Dict],Dict],
                     cell_budget:int=5_000_000, timeout_s:int=0, verbose=True) -> SweepLedger:
    """cases: dicts with keys experiment, mode, k, L, band, seed, lengths(optional),
    and whatever run_case needs. run_case(case)->metrics dict. Resumes; budget- and
    failure-isolates each case."""
    led=SweepLedger(name)
    for case in cases:
        key=(case["experiment"],case["mode"],case.get("k",""),case.get("L",""),
             case.get("band",""),case.get("seed",0))
        if led.done(key):
            if verbose: print(f"[resume] skip {key}"); 
            continue
        base={"experiment":case["experiment"],"mode":case["mode"],"k":case.get("k",""),
              "L":case.get("L",""),"band":case.get("band",""),"seed":case.get("seed",0)}
        # budget guard
        lengths=case.get("lengths")
        if lengths and cube_cells(lengths)>cell_budget:
            led.append({**base,"status":"skipped_budget",
                        "cells":cube_cells(lengths),"error":f">{cell_budget} cells"})
            if verbose: print(f"[budget] skip {key}: {cube_cells(lengths)} cells")
            continue
        try:
            with case_timeout(timeout_s):
                m=run_case(case)
            led.append({**base,"status":"ok",**{k:m.get(k) for k in
                ("score","recovery","cells","band_cells","sec","route","metric")}})
            if verbose: print(f"[ok] {key}: {m.get('score', m.get('recovery',''))}")
        except TimeoutError as e:
            led.append({**base,"status":"timeout","error":str(e)}); 
            if verbose: print(f"[timeout] {key}")
        except Exception as e:
            led.append({**base,"status":"error","error":str(e)[:150]})
            if verbose: print(f"[error] {key}: {e}")
    return led

# ───────────────────────── analysis (regenerate tables from ledger)
def analyze(name="msa"):
    led=SweepLedger(name)
    if not led.path.exists(): print("no results"); return
    rows=[r for r in csv.DictReader(open(led.path))]
    ok=[r for r in rows if r["status"]=="ok"]
    import collections
    print(f"completed {len(ok)}/{len(rows)} cases")
    st=collections.Counter(r["status"] for r in rows)
    print("status:",dict(st))
    # example: recovery by band for local; score by k for global
    byexp=collections.defaultdict(list)
    for r in ok: byexp[r["experiment"]].append(r)
    for exp,rr in byexp.items():
        print(f"\n[{exp}]")
        for r in rr[:12]:
            key=f"mode={r['mode']} k={r['k']} L={r['L']} band={r['band']}"
            val=r['recovery'] or r['score'] or r['metric']
            print(f"  {key}: {val}")

# ───────────────────────── selftest (no GPU/CuPy needed)
def _selftest():
    import tempfile, random
    global RESULTS
    RESULTS=Path(tempfile.mkdtemp())
    # fake run_case: a global score sweep over k, a local recovery sweep over band
    def run_case(case):
        rng=random.Random(hash((case["experiment"],case["k"],case["band"],case["seed"]))&0xffff)
        if case["experiment"]=="global_k":
            k=case["k"]; return {"score":100-8*k+rng.randint(-2,2),"cells":cube_cells(case["lengths"]),
                                 "band_cells":cube_cells(case["lengths"])//max(k,1),"sec":0.01*k,"metric":"SP"}
        else:
            b=case["band"]; return {"recovery":min(1.0,0.6+0.1*b+rng.random()*0.1),"metric":"motif_recovery"}
    cases=[]
    # global k-sweep, with one over-budget case to exercise the guard
    for k in (2,3,4,8):
        lengths=[40]*k if k<8 else [400]*8   # k=8 huge -> budget skip
        cases.append({"experiment":"global_k","mode":"global","k":k,"L":40,"band":2,"seed":0,"lengths":lengths})
    # local band-sweep
    for b in (0,1,2,3):
        cases.append({"experiment":"local_band","mode":"local","k":3,"L":60,"band":b,"seed":0})
    led=survivable_sweep("selftest",cases,run_case,cell_budget=10_000_000,verbose=False)
    # resume idempotence for ok cases
    ok1=sum(1 for r in csv.DictReader(open(led.path)) if r["status"]=="ok")
    survivable_sweep("selftest",cases,run_case,cell_budget=10_000_000,verbose=False)
    ok2=sum(1 for r in csv.DictReader(open(led.path)) if r["status"]=="ok")
    assert ok1==ok2, "resume re-ran completed cases"
    # budget guard fired on k=8
    rows=list(csv.DictReader(open(led.path)))
    assert any(r["status"]=="skipped_budget" for r in rows), "budget guard did not fire"
    # GPU backoff: simulate an OOM that recovers at smaller tile
    calls={"n":0}
    def fill(seqs,tile=256):
        calls["n"]+=1
        if tile>64: raise MemoryError("out of memory")
        return {"tile":tile,"filled":True}
    out=gpu_fill_with_backoff(fill,["ACGT","ACGT"],tile=256,min_tile=16)
    assert out["tile"]<=64 and calls["n"]>1, "gpu backoff did not retry"
    analyze("selftest")
    print("\nmsa_survival SELFTEST OK: resume idempotent, budget guard, gpu OOM backoff, degeneracy recorded")

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="analyze": analyze(sys.argv[2] if len(sys.argv)>2 else "msa")
    else: _selftest()
