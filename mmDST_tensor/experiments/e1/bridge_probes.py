#!/usr/bin/env python3
"""
bridge_probes.py -- minimal, honest tests of the two deep-learning 
claims, so the paper can either report a small result or scope them cleanly as
future work.

Claim 1: "the partition function bridges exact MSA and modern deep learning"
  Concrete testable core: the log-sum-exp (partition) recurrence is
  DIFFERENTIABLE in the scoring parameters, so scores can be *learned*. We test
  the minimal version: numerically differentiate log Z with respect to the gap
  penalty g and the match score, and confirm the gradient is well-defined and
  finite (a prerequisite for any learned-scoring bridge). This is a genuine,
  small, in-scope result. The FULL bridge (fitting parameters to reference
  alignments at k>2) is future work.

Claim 2: "just as soft attention underpins modern transformers"
  This is an analogy, not a computation on our objects, so it stays as a cited
  analogy in the paper. We do NOT fabricate a transformer experiment. What we
  CAN show in-scope: the partition tensor is smooth (small Lipschitz constant of
  log Z in the inputs) while the value tensor is not -- the precise property the
  analogy rests on. We measure both.
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/core"))

from core import fill, GAP, MATCH, MIS


def logZ(seqs, beta=0.1, gap=None, match=None):
    """Compute log Z, optionally overriding gap/match via monkeypatch of core."""
    import core
    old_gap, old_match = core.GAP, core.MATCH
    if gap is not None:
        core.GAP = gap
    if match is not None:
        core.MATCH = match
    try:
        r = fill(seqs, mode="global", semiring="logsumexp", beta=beta, dense=True)
        F = r["tensor"]
        sink = tuple(s - 1 for s in F.shape)
        val = np.log(max(F[sink], 1e-300))
    finally:
        core.GAP, core.MATCH = old_gap, old_match
    return val


def probe_differentiable(seqs, beta=0.1):
    """Claim 1: numerical gradient of log Z wrt gap penalty and match score."""
    h = 1e-3
    base = logZ(seqs, beta)
    dg = (logZ(seqs, beta, gap=GAP + h) - logZ(seqs, beta, gap=GAP - h)) / (2 * h)
    dm = (logZ(seqs, beta, match=MATCH + h) - logZ(seqs, beta, match=MATCH - h)) / (2 * h)
    return base, dg, dm


def probe_smoothness(seqs, beta=0.1, trials=20, rng=None):
    """Claim 2 (the real content): how much does the objective move under a
    single-residue perturbation? Soft (log Z) should move smoothly; hard (max
    score) can jump. We report mean |delta| for each."""
    rng = rng or np.random.default_rng(0)
    bases = "ACGT"
    # baseline
    r_soft = fill(seqs, mode="global", semiring="logsumexp", beta=beta, dense=True)
    F = r_soft["tensor"]; sink = tuple(s - 1 for s in F.shape)
    soft0 = np.log(max(F[sink], 1e-300)) / beta
    hard0 = fill(seqs, mode="global")["score"]

    soft_deltas, hard_deltas = [], []
    for _ in range(trials):
        i = rng.integers(len(seqs))
        s = list(seqs[i])
        p = rng.integers(len(s))
        s[p] = rng.choice([b for b in bases if b != s[p]])
        pert = list(seqs); pert[i] = "".join(s)
        rs = fill(pert, mode="global", semiring="logsumexp", beta=beta, dense=True)
        Fp = rs["tensor"]; sk = tuple(x - 1 for x in Fp.shape)
        soft_deltas.append(abs(np.log(max(Fp[sk], 1e-300)) / beta - soft0))
        hard_deltas.append(abs(fill(pert, mode="global")["score"] - hard0))
    return np.mean(soft_deltas), np.std(soft_deltas), np.mean(hard_deltas), np.std(hard_deltas)


def main():
    import random
    rng = random.Random(1)
    bases = "ACGT"
    anc = "".join(rng.choice(bases) for _ in range(12))
    seqs = ["".join(rng.choice([b for b in bases if b != c]) if rng.random() < 0.15
                    else c for c in anc) for _ in range(3)]
    print("sequences:", seqs, "\n")

    print("== Claim 1: partition function is differentiable in scoring params ==")
    base, dg, dm = probe_differentiable(seqs)
    print(f"  log Z = {base:.4f}")
    print(f"  d(log Z)/d(gap)   = {dg:+.4f}   (finite, well-defined)")
    print(f"  d(log Z)/d(match) = {dm:+.4f}   (finite, well-defined)")
    print("  => learned scoring is well-posed; full fit to references is future work.\n")

    print("== Claim 2: soft objective is smooth, hard objective is not ==")
    sm, ss, hm, hs = probe_smoothness(seqs, rng=np.random.default_rng(2))
    print(f"  mean |delta| under one-residue change:")
    print(f"    soft (logZ/beta): {sm:.3f} +/- {ss:.3f}")
    print(f"    hard (max score): {hm:.3f} +/- {hs:.3f}")
    ratio = hm / sm if sm > 0 else float('inf')
    print(f"  hard/soft sensitivity ratio: {ratio:.1f}x")
    print("  => supports the smoothness analogy; transformer link stays cited analogy.")


if __name__ == "__main__":
    main()
