#!/usr/bin/env python3
"""
tensor_ring.py
GPU-native Tensor-Ring decomposition for MSA tensors.

Key fixes/improvements over the original implementation:
  * Keeps all ALS calculations on the GPU.
  * No CuPy <-> NumPy mixing during decomposition/error evaluation.
  * Uses exact local least-squares updates (CuPy lstsq).
  * Uses an SVD-optimal solution for the d=2 matrix case.
  * Uses multiple random restarts for d>2 and keeps the best solution.
  * Normalizes cores after updates to improve numerical stability.
  * Transfers data back to CPU only for final reporting/JSON serialization.

The Tensor-Ring problem for d>2 is non-convex, so "optimal" means the
best solution found by the configured multi-start ALS procedure; a
global optimum cannot in general be guaranteed by ALS.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cupy as cp
    HAS_GPU = bool(cp.cuda.is_available())
except Exception:
    cp = None
    HAS_GPU = False

# Project modules.
sys.path.insert(0, "/mnt/d/mmDst/code")
from core import fill


class TensorRing:
    def __init__(
        self,
        use_gpu: bool = True,
        verbose: bool = True,
        tol: float = 1e-8,
        ridge: float = 1e-10,
        seed: int = 1234,
    ):
        self.use_gpu = bool(use_gpu and HAS_GPU)
        self.verbose = verbose
        self.tol = tol
        self.ridge = ridge
        self.seed = seed

        self.xp = cp if self.use_gpu else np
        self.cores = None
        self.shape = None
        self.bond_dim = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def _to_device(self, x):
        if self.use_gpu:
            return cp.asarray(x)
        return np.asarray(x)

    def _to_cpu(self, x):
        if self.use_gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    def _sync(self):
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

    def _random_core(self, shape):
        if self.use_gpu:
            # CuPy's global RNG is fine here; the experiment is reproducible
            # after seeding below.
            return (cp.random.standard_normal(shape).astype(cp.float64) * 0.1)
        return np.random.default_rng(self.seed).standard_normal(shape).astype(np.float64) * 0.1

    # ------------------------------------------------------------------
    # Core contractions
    # ------------------------------------------------------------------
    def reconstruct_gpu(self, cores):
        """Reconstruct entirely on the current device."""
        cores = [self._to_device(c) for c in cores]
        d = len(cores)

        if d == 1:
            # A one-core ring requires a trace of the two bond indices.
            return self.xp.trace(cores[0], axis1=0, axis2=2)

        result = cores[0]
        for core in cores[1:]:
            result = self.xp.tensordot(result, core, axes=([-1], [0]))

        # result = (r, I0, ..., Id-1, r)
        return self.xp.trace(result, axis1=0, axis2=-1)

    def relative_error_gpu(self, tensor, tensor_hat):
        diff = tensor - tensor_hat
        denom = self.xp.linalg.norm(tensor)
        return float((self.xp.linalg.norm(diff) / self.xp.maximum(denom, 1e-30)).item())

    # ------------------------------------------------------------------
    # d=2: globally optimal matrix solution
    # ------------------------------------------------------------------
    def _decompose_matrix_optimal(self, tensor, bond_dim):
        """
        For d=2 a Tensor Ring with bond dimension r can represent a
        matrix factorization of rank at most r^2.

        Therefore the best rank-r^2 approximation is obtained directly
        by truncated SVD, rather than non-convex ALS.
        """
        m, n = tensor.shape
        r = bond_dim
        max_rank = r * r

        U, S, Vh = self.xp.linalg.svd(tensor, full_matrices=False)
        q = min(max_rank, int(S.shape[0]))

        # Keep the best rank-r^2 approximation.
        sroot = self.xp.sqrt(self.xp.maximum(S[:q], 0))
        A = U[:, :q] * sroot[None, :]
        B = sroot[:, None] * Vh[:q, :]

        # Embed q <= r^2 into exactly r^2 bond-pair channels.
        A_full = self.xp.zeros((m, r * r), dtype=tensor.dtype)
        B_full = self.xp.zeros((r * r, n), dtype=tensor.dtype)
        A_full[:, :q] = A
        B_full[:q, :] = B

        # Core 0: G0[a, i, b], pair index = a*r+b.
        G0 = A_full.reshape(m, r, r).transpose(1, 0, 2)

        # Core 1: G1[b, j, a], pair index = b*r+a.
        G1 = B_full.reshape(r, r, n).transpose(0, 2, 1)

        return [G0, G1]

    # ------------------------------------------------------------------
    # d>2: exact local least-squares TR-ALS
    # ------------------------------------------------------------------
    def _environment(self, cores, n):
        """
        Contract every core except n.

        Returned tensor has dimensions:
            (r, I[n+1], ..., I[n-1], r)

        The first and last dimensions correspond to the two bond indices
        surrounding core n.
        """
        d = len(cores)
        env = cores[(n + 1) % d]

        for step in range(1, d - 1):
            idx = (n + 1 + step) % d
            env = self.xp.tensordot(env, cores[idx], axes=([-1], [0]))

        return env

    def _update_core(self, cores, tensor, n):
        d = len(self.shape)
        r = self.bond_dim
        In = self.shape[n]

        env = self._environment(cores, n)

        # env currently:
        # (r_left, I[n+1], ..., I[n-1], r_right)
        env = env.reshape(r, -1, r)
        env = self.xp.transpose(env, (1, 0, 2))
        E = env.reshape(-1, r * r)

        # Put tensor physical dimensions in the same cyclic order:
        # n+1, ..., d-1, 0, ..., n-1, n
        perm = [(n + 1 + i) % d for i in range(d)]
        T = self.xp.transpose(tensor, perm)
        T = T.reshape(-1, In)

        # Solve E @ X ~= T, where X has shape (r^2, In).
        #
        # CuPy lstsq is executed on the GPU.  The ridge term improves
        # stability when the environment is nearly rank deficient.
        if self.ridge > 0:
            gram = E.T @ E
            rhs = E.T @ T
            gram = gram + self.ridge * self.xp.eye(r * r, dtype=E.dtype)
            X = self.xp.linalg.solve(gram, rhs)
        else:
            X = self.xp.linalg.lstsq(E, T, rcond=None)[0]

        # X[a*r+b, i] -> G[a, i, b]
        return X.reshape(r, r, In).transpose(0, 2, 1)

    def _normalize_cores(self, cores):
        """
        Preserve the represented tensor while controlling core magnitudes.
        A core's Frobenius norm is absorbed into the following core.
        """
        d = len(cores)
        for n in range(d - 1):
            norm = self.xp.linalg.norm(cores[n])
            if float(norm.item()) > 0:
                cores[n] = cores[n] / norm
                cores[n + 1] = cores[n + 1] * norm
        return cores

    def _als(self, tensor, bond_dim, max_iter):
        d = tensor.ndim
        r = bond_dim

        cores = [
            self._random_core((r, self.shape[n], r))
            for n in range(d)
        ]

        # A slightly more balanced initialization.
        cores = self._normalize_cores(cores)

        previous = np.inf
        best_cores = None
        best_error = np.inf

        for it in range(max_iter):
            for n in range(d):
                cores[n] = self._update_core(cores, tensor, n)

            cores = self._normalize_cores(cores)

            tensor_hat = self.reconstruct_gpu(cores)
            error = self.relative_error_gpu(tensor, tensor_hat)

            if error < best_error:
                best_error = error
                best_cores = [c.copy() for c in cores]

            if self.verbose and (it == 0 or (it + 1) % 5 == 0 or it == max_iter - 1):
                print(f"  Iter {it + 1:3d}: error = {error:.10f}")

            # Relative improvement convergence test.
            if abs(previous - error) <= self.tol * max(1.0, previous):
                if self.verbose:
                    print(f"  Converged at iteration {it + 1}")
                break

            previous = error

        return best_cores, best_error

    # ------------------------------------------------------------------
    # Public decomposition
    # ------------------------------------------------------------------
    def decompose(
        self,
        tensor: np.ndarray,
        bond_dim: int = 2,
        max_iter: int = 50,
        n_restarts: int = 4,
    ):
        tensor = self._to_device(tensor).astype(self.xp.float64, copy=False)
        self.shape = tensor.shape
        self.bond_dim = bond_dim

        if tensor.ndim == 2:
            cores = self._decompose_matrix_optimal(tensor, bond_dim)
            err = self.relative_error_gpu(
                tensor, self.reconstruct_gpu(cores)
            )
            if self.verbose:
                print(f"  Exact SVD-optimal d=2 TR error = {err:.10f}")
            self.cores = [self._to_cpu(c) for c in cores]
            return self.cores

        global_best = None
        global_error = np.inf

        # Reproducible independent restarts.
        if self.use_gpu:
            cp.random.seed(self.seed)
        else:
            np.random.seed(self.seed)

        for restart in range(n_restarts):
            if self.verbose and n_restarts > 1:
                print(f"  Restart {restart + 1}/{n_restarts}")

            cores, error = self._als(tensor, bond_dim, max_iter)

            if error < global_error:
                global_error = error
                global_best = [c.copy() for c in cores]

        self.cores = [self._to_cpu(c) for c in global_best]
        return self.cores

    # ------------------------------------------------------------------
    # Compression / metrics
    # ------------------------------------------------------------------
    def compress(
        self,
        tensor: np.ndarray,
        bond_dim: int = 2,
        max_iter: int = 50,
        n_restarts: int = 4,
    ) -> Dict:
        start = time.perf_counter()

        tensor_gpu = self._to_device(tensor).astype(self.xp.float64, copy=False)

        cores_cpu = self.decompose(
            tensor_gpu,
            bond_dim=bond_dim,
            max_iter=max_iter,
            n_restarts=n_restarts,
        )

        # Final reconstruction on GPU for accurate device-side metric.
        cores_gpu = [self._to_device(c) for c in cores_cpu]
        tensor_hat_gpu = self.reconstruct_gpu(cores_gpu)

        rel_err = self.relative_error_gpu(tensor_gpu, tensor_hat_gpu)

        # One scalar transfer at the end is fine.
        tensor_hat = self._to_cpu(tensor_hat_gpu)

        # Log-Z error.
        logZ_error = None
        if tensor.ndim > 1:
            sink = tuple(s - 1 for s in tensor.shape)
            true_value = float(np.asarray(tensor)[sink])
            hat_value = float(tensor_hat[sink])

            true_value = max(true_value, 1e-300)
            hat_value = max(hat_value, 1e-300)
            logZ_error = abs(np.log(hat_value) - np.log(true_value))

        compressed_size = sum(int(c.size) for c in cores_cpu)
        original_size = int(np.prod(tensor.shape))
        compression = original_size / compressed_size

        self._sync()
        elapsed = time.perf_counter() - start

        return {
            "cores": cores_cpu,
            "bond_dim": bond_dim,
            "compression": compression,
            "rel_error": rel_err,
            "logZ_error": logZ_error,
            "time": elapsed,
            "compressed_size": compressed_size,
            "original_size": original_size,
        }


# ============================================================================
# Experiments
# ============================================================================

def run_experiments():
    print("=" * 70)
    print("GPU TENSOR-RING EXPERIMENTS")
    print("=" * 70)

    results = {}

    for k in [2, 3, 4]:
        print(f"\nTesting k={k}...")

        seqs = ["ACGT" * 3 for _ in range(k)]
        r = fill(
            seqs,
            mode="global",
            semiring="logsumexp",
            beta=0.1,
            dense=True,
        )
        F = np.asarray(r["tensor"], dtype=np.float64)

        print(f"  Tensor shape: {F.shape}, size: {F.size:,}")

        tr = TensorRing(
            use_gpu=HAS_GPU,
            verbose=True,
            tol=1e-8,
            ridge=1e-10,
            seed=1234,
        )

        result = tr.compress(
            F,
            bond_dim=2,
            max_iter=50,
            n_restarts=4,
        )

        print(
            f"  TR: compression={result['compression']:.2f}x, "
            f"error={result['rel_error']:.10f}"
        )
        print(
            f"      time={result['time']:.3f}s, "
            f"logZ_err={result['logZ_error']:.10f}"
        )

        results[f"k{k}"] = {
            "k": k,
            "shape": list(F.shape),
            "compression": result["compression"],
            "error": result["rel_error"],
            "logZ_error": result["logZ_error"],
            "time": result["time"],
            "size": int(F.size),
        }

    output_file = Path("/mnt/d/mmDst/code/data/tr_gpu_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(
        f"{'k':<4} {'Compression':<15} "
        f"{'Error':<16} {'logZ Error':<16} {'Time (s)':<10}"
    )
    print("-" * 70)

    for _, r in results.items():
        print(
            f"{r['k']:<4} "
            f"{r['compression']:<15.2f} "
            f"{r['error']:<16.10f} "
            f"{r['logZ_error']:<16.10f} "
            f"{r['time']:<10.3f}"
        )

    return results


def main():
    print("GPU TENSOR-RING DECOMPOSITION FOR MSA")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU available: {HAS_GPU}")

    if HAS_GPU:
        try:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            print(f"GPU: {name}")
            print(f"GPU memory: {dev.mem_info[1] / 1024**3:.2f} GB total")
        except Exception as e:
            print(f"GPU information unavailable: {e}")

    print("=" * 70)
    run_experiments()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
