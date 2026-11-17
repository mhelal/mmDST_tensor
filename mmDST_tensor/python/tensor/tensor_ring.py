#!/usr/bin/env python3
"""
tensor_ring.py - Complete Tensor-Ring (TR) Decomposition for MSA
Implements the TR-ALS algorithm with GPU acceleration for MSA tensors.

Based on:
- Zhao et al. 2016: "Tensor ring decomposition"
- TR-ALS algorithm with circular bond dimensions
- GPU acceleration via CuPy
"""

import numpy as np
import cupy as cp
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from datetime import datetime

# Import core MSA functionality
import sys
sys.path.insert(0, '/mnt/d/mmDst/code')
from core import fill
from decomp import rel_error

class TensorRing:
    """
    Tensor-Ring (TR) decomposition with GPU acceleration.
    
    TR represents a tensor as a circular chain of 3D cores:
    T(i1,...,iN) = Tr(G1(i1) G2(i2) ... GN(iN))
    
    where Gk(ik) is a matrix of size (R_{k-1} x R_k) with R_0 = R_N.
    """
    
    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        """
        Initialize TR decomposition.
        
        Args:
            use_gpu: Whether to use GPU acceleration via CuPy
            verbose: Print progress information
        """
        self.use_gpu = use_gpu and self._check_gpu()
        self.verbose = verbose
        self.xp = cp if self.use_gpu else np
        self.cores = None
        self.bond_dims = None
        self.shape = None
        
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import cupy as cp
            cp.cuda.is_available()
            return True
        except:
            return False
    
    def _to_device(self, arr):
        """Move array to GPU if available."""
        if self.use_gpu:
            return cp.asarray(arr)
        return np.asarray(arr)
    
    def _to_cpu(self, arr):
        """Move array from GPU to CPU."""
        if self.use_gpu and hasattr(arr, 'get'):
            return arr.get()
        return arr
    
    def _random_core(self, shape):
        """Create random core with Xavier initialization."""
        if self.use_gpu:
            return cp.random.randn(*shape).astype(cp.float64) * np.sqrt(2.0 / (shape[0] + shape[2]))
        return np.random.randn(*shape).astype(np.float64) * np.sqrt(2.0 / (shape[0] + shape[2]))
    
    def decompose(self, tensor: np.ndarray, bond_dim: int = 2, 
                  max_iter: int = 50, tol: float = 1e-6) -> List[np.ndarray]:
        """
        Decompose tensor using TR-ALS algorithm.
        
        Args:
            tensor: Input tensor
            bond_dim: Bond dimension r (circular)
            max_iter: Maximum ALS iterations
            tol: Convergence tolerance
            
        Returns:
            List of TR cores
        """
        tensor = self._to_device(tensor)
        self.shape = tensor.shape
        d = len(self.shape)
        
        # Initialize TR cores
        # For TR, we need R_0 = R_N = bond_dim
        self.bond_dims = [bond_dim] * (d + 1)  # Circular: R_0 = R_d = bond_dim
        
        # Initialize cores randomly
        cores = []
        for n in range(d):
            core_shape = (bond_dim, self.shape[n], bond_dim)
            core = self._random_core(core_shape)
            cores.append(core)
        
        # TR-ALS iteration
        tensor_norm = self.xp.linalg.norm(tensor)
        prev_error = float('inf')
        
        for it in range(max_iter):
            # For each core, fix others and solve least squares
            for n in range(d):
                # Unfold tensor and compute Gram matrices
                unfolded, rank = self._unfold_for_core(tensor, cores, n)
                
                # Solve least squares using SVD
                U, S, Vt = self.xp.linalg.svd(unfolded, full_matrices=False)
                r = min(bond_dim, len(S))
                
                # Truncate to bond dimension
                U = U[:, :r]
                S = S[:r]
                Vt = Vt[:r, :]
                
                # Update core
                core = self._fold_core(U, S, Vt, cores, n, rank)
                cores[n] = core
            
            # Check convergence
            tensor_hat = self.reconstruct(cores)
            error = self.xp.linalg.norm(tensor - tensor_hat) / tensor_norm
            
            if self.verbose:
                print(f"  Iteration {it+1}: error = {error:.6f}")
            
            if abs(error - prev_error) < tol:
                break
            prev_error = error
        
        self.cores = [self._to_cpu(c) for c in cores]
        return self.cores
    
    def _unfold_for_core(self, tensor, cores, n):
        """
        Unfold tensor for core n in TR-ALS.
        
        This implements the TR unfolding that preserves circular structure.
        """
        d = len(self.shape)
        bond_dim = self.bond_dims[0]
        
        # Build left and right contractions
        left_cores = cores[:n]
        right_cores = cores[n+1:]
        
        # Contract left side
        if left_cores:
            # Need to handle circular property for TR
            left = self._contract_cores(left_cores, circular_start=True)
        else:
            left = self.xp.eye(bond_dim).reshape(1, bond_dim, bond_dim)
        
        # Contract right side
        if right_cores:
            right = self._contract_cores(right_cores, circular_end=True)
        else:
            right = self.xp.eye(bond_dim).reshape(1, bond_dim, bond_dim)
        
        # Contract left and right to get the unfolding
        # For TR, the left and right contractions are merged through the circular bond
        if left_cores and right_cores:
            # Contract left and right through the circular bond
            # This is the key difference from TT: the circular bond dimension is maintained
            left_right = self.xp.einsum('iab,jbc->iac', left, right)
            # Reshape to combine dimensions
            unfolded = self.xp.einsum('iab,ibc->iac', left_right, tensor)
        else:
            unfolded = tensor
        
        # Reshape to 2D for SVD
        unfolded = unfolded.reshape(-1, unfolded.shape[-1])
        
        return unfolded, bond_dim
    
    def _contract_cores(self, cores, circular_start=False, circular_end=False):
        """
        Contract a sequence of cores, optionally with circular connections.
        """
        if not cores:
            return None
        
        result = cores[0]
        for core in cores[1:]:
            result = self.xp.tensordot(result, core, axes=([-1], [0]))
        
        if circular_start:
            # Reshape to handle circular bond
            result = result.reshape(-1, result.shape[-2], result.shape[-1])
        
        if circular_end:
            # Reshape to handle circular bond
            result = result.reshape(result.shape[0], result.shape[1], -1)
        
        return result
    
    def _fold_core(self, U, S, Vt, cores, n, rank):
        """
        Fold SVD result back into TR core shape.
        """
        # The core shape depends on whether n is the first or last core
        bond_dim = self.bond_dims[0]
        
        # The shape should be (bond_dim, self.shape[n], bond_dim)
        core = U @ self.xp.diag(S) @ Vt
        core = core.reshape(bond_dim, self.shape[n], bond_dim)
        
        return core
    
    def reconstruct(self, cores: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Reconstruct tensor from TR cores.
        
        Args:
            cores: List of TR cores (uses stored cores if None)
            
        Returns:
            Reconstructed tensor
        """
        if cores is None:
            cores = self.cores
        
        if cores is None:
            raise ValueError("No cores available. Run decompose() first.")
        
        # Move to device if needed
        cores = [self._to_device(c) for c in cores]
        d = len(cores)
        shape = [c.shape[1] for c in cores]
        
        # Contract all cores with trace
        # For TR: T = Tr(G1 * G2 * ... * Gd)
        # The trace operation handles the circular bond
        
        # Contract cores sequentially
        result = cores[0]
        for core in cores[1:]:
            # Contract last dimension of result with first dimension of core
            result = self.xp.tensordot(result, core, axes=([-1], [0]))
        
        # Take the trace to handle circular bond
        # For TR, the first and last dimensions are the bond dimensions
        # We need to trace over them
        if d > 1:
            # Reshape to combine dimensions
            result = result.reshape(-1, self.bond_dims[0], self.bond_dims[0])
            # Trace over the bond dimensions
            result = self.xp.trace(result, axis1=1, axis2=2)
        else:
            # For d=1, TR = Tr(G) = trace of the core
            result = self.xp.trace(result, axis1=0, axis2=2)
        
        # Reshape to the original tensor shape
        tensor = result.reshape(shape)
        
        return self._to_cpu(tensor)
    
    def compress(self, tensor: np.ndarray, bond_dim: int = 2,
                 max_iter: int = 30) -> Dict:
        """
        Compress tensor using TR decomposition with performance metrics.
        
        Args:
            tensor: Input tensor
            bond_dim: Bond dimension
            max_iter: Maximum ALS iterations
            
        Returns:
            Dictionary with compression results
        """
        start_time = time.perf_counter()
        
        # Decompose
        cores = self.decompose(tensor, bond_dim, max_iter)
        
        # Measure compression
        decomp_time = time.perf_counter() - start_time
        
        # Reconstruct
        tensor_hat = self.reconstruct(cores)
        
        # Calculate metrics
        compressed_size = sum(c.size for c in cores)
        original_size = tensor.size
        compression = original_size / compressed_size
        
        # Error metrics
        err = rel_error(tensor, tensor_hat)
        
        # Get bond dimensions
        bond_dims = [c.shape[0] for c in cores] + [cores[-1].shape[2]]
        
        return {
            'cores': cores,
            'bond_dims': bond_dims,
            'compression': compression,
            'rel_error': err,
            'time': decomp_time,
            'compressed_size': compressed_size,
            'original_size': original_size,
            'method': 'Tensor-Ring (TR)'
        }


def tr_decompose(T, bond_dim=2, max_iter=30):
    """
    Convenience function for TR decomposition.
    
    Args:
        T: Input tensor
        bond_dim: Bond dimension
        max_iter: Maximum ALS iterations
        
    Returns:
        List of TR cores
    """
    tr = TensorRing(use_gpu=True)
    return tr.decompose(T, bond_dim, max_iter)


def tr_reconstruct(cores):
    """
    Reconstruct tensor from TR cores.
    
    Args:
        cores: List of TR cores
        
    Returns:
        Reconstructed tensor
    """
    tr = TensorRing(use_gpu=False)
    return tr.reconstruct(cores)


# ============================================================================
# Experiments for MSA
# ============================================================================

def experiment_tr_vs_tt():
    """
    Compare TR vs TT decomposition on MSA partition tensors.
    """
    print("\n" + "="*70)
    print("TENSOR-RING VS TENSOR-TRAIN COMPARISON")
    print("="*70)
    
    results = {}
    
    for k in [2, 3, 4, 5, 6]:
        print(f"\nTesting k={k}...")
        
        # Generate synthetic MSA partition tensor
        seqs = ["ACGT" * 5 for _ in range(k)]  # L=20
        r = fill(seqs, mode="global", semiring="logsumexp", beta=0.1, dense=True)
        F = r['tensor']
        
        print(f"  Tensor shape: {F.shape}, size: {F.size:,}")
        
        # Test different bond dimensions
        for bond_dim in [2, 3]:
            print(f"\n  Bond dimension: {bond_dim}")
            
            # TR decomposition
            tr = TensorRing(use_gpu=True)
            tr_result = tr.compress(F, bond_dim=bond_dim, max_iter=30)
            
            print(f"    TR: compression={tr_result['compression']:.1f}x, error={tr_result['rel_error']:.6f}, time={tr_result['time']:.3f}s")
            
            # TT decomposition for comparison
            from decomp import tt_decompose, tt_reconstruct
            try:
                start = time.perf_counter()
                cores_tt = tt_decompose(F, ranks=[bond_dim] * (k-1))
                tt_time = time.perf_counter() - start
                F_hat_tt = tt_reconstruct(cores_tt)
                tt_err = rel_error(F, F_hat_tt)
                tt_comp = F.size / sum(c.size for c in cores_tt)
                
                print(f"    TT: compression={tt_comp:.1f}x, error={tt_err:.6f}, time={tt_time:.3f}s")
                
                # Store for comparison
                key = f"k{k}_r{bond_dim}"
                results[key] = {
                    'k': k,
                    'bond_dim': bond_dim,
                    'tr_compression': tr_result['compression'],
                    'tr_error': tr_result['rel_error'],
                    'tr_time': tr_result['time'],
                    'tt_compression': tt_comp,
                    'tt_error': tt_err,
                    'tt_time': tt_time
                }
            except Exception as e:
                print(f"    TT: failed - {e}")
                results[f"k{k}_r{bond_dim}"] = {
                    'k': k,
                    'bond_dim': bond_dim,
                    'tr_compression': tr_result['compression'],
                    'tr_error': tr_result['rel_error'],
                    'tr_time': tr_result['time'],
                    'tt_compression': None,
                    'tt_error': None,
                    'tt_time': None,
                    'tt_failed': str(e)
                }
    
    # Save results
    output_file = Path("/mnt/d/mmDst/code/data/tr_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "-"*70)
    print("SUMMARY: TR vs TT Comparison")
    print("-"*70)
    print(f"{'k':<4} {'r':<4} {'TR Comp':<12} {'TR Error':<12} {'TT Comp':<12} {'TT Error':<12}")
    print("-"*70)
    
    for key, r in results.items():
        if r.get('tt_compression'):
            print(f"{r['k']:<4} {r['bond_dim']:<4} {r['tr_compression']:<12.1f} {r['tr_error']:<12.6f} {r['tt_compression']:<12.1f} {r['tt_error']:<12.6f}")
        else:
            print(f"{r['k']:<4} {r['bond_dim']:<4} {r['tr_compression']:<12.1f} {r['tr_error']:<12.6f} {'N/A':<12} {'N/A':<12}")
    
    return results


def experiment_tr_scaling():
    """
    Test TR scaling with k and bond dimension.
    """
    print("\n" + "="*70)
    print("TENSOR-RING SCALING EXPERIMENT")
    print("="*70)
    
    results = []
    
    for k in [2, 3, 4, 5, 6]:
        for bond_dim in [2, 3, 4]:
            print(f"\nTesting k={k}, r={bond_dim}")
            
            # Generate synthetic MSA partition tensor
            seqs = ["ACGT" * 5 for _ in range(k)]
            try:
                r = fill(seqs, mode="global", semiring="logsumexp", beta=0.1, dense=True)
                F = r['tensor']
            except Exception as e:
                print(f"  Failed to create tensor: {e}")
                continue
            
            print(f"  Tensor shape: {F.shape}, size: {F.size:,}")
            
            tr = TensorRing(use_gpu=True)
            try:
                result = tr.compress(F, bond_dim=bond_dim, max_iter=30)
                
                results.append({
                    'k': k,
                    'bond_dim': bond_dim,
                    'compression': result['compression'],
                    'error': result['rel_error'],
                    'time': result['time'],
                    'size': F.size
                })
                
                print(f"  Compression: {result['compression']:.1f}x, error: {result['rel_error']:.6f}, time: {result['time']:.3f}s")
                
            except Exception as e:
                print(f"  Failed: {e}")
                results.append({
                    'k': k,
                    'bond_dim': bond_dim,
                    'compression': None,
                    'error': None,
                    'time': None,
                    'size': F.size,
                    'failed': str(e)
                })
    
    # Save results
    output_file = Path("/mnt/d/mmDst/code/data/tr_scaling_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


def experiment_tr_balibase():
    """
    Test TR on BAliBASE global alignment.
    """
    print("\n" + "="*70)
    print("TENSOR-RING ON BALIBASE")
    print("="*70)
    
    from balibase import read_msf, read_tfa, load_case, reference_sp_tc
    
    # Find BAliBASE files
    balibase_dir = Path("/mnt/d/mmDst/code/data/Balibase/bb3_release")
    
    if not balibase_dir.exists():
        print("BAliBASE not found. Run data download first.")
        return
    
    results = []
    
    # Test on a few RV11 cases
    rv11_dir = balibase_dir / "RV11"
    tfa_files = list(rv11_dir.glob("*.tfa"))[:5]  # First 5 cases
    
    for tfa_file in tfa_files:
        case_name = tfa_file.stem
        msf_file = rv11_dir / f"{case_name}.msf"
        
        if not msf_file.exists():
            continue
        
        print(f"\nTesting case: {case_name}")
        
        # Load sequences
        seqs_dict = read_tfa(tfa_file)
        names = list(seqs_dict.keys())[:3]  # k=3
        seqs = [seqs_dict[n][:20] for n in names]  # Truncate to L=20
        
        # Get partition tensor
        r = fill(seqs, mode="global", semiring="logsumexp", beta=0.1, dense=True)
        F = r['tensor']
        
        print(f"  Tensor shape: {F.shape}, size: {F.size:,}")
        
        # TR decomposition
        tr = TensorRing(use_gpu=True)
        
        for bond_dim in [2, 3]:
            try:
                result = tr.compress(F, bond_dim=bond_dim, max_iter=20)
                
                # Get exact score for comparison
                r_exact = fill(seqs, mode="global", band=None, semiring="maxplus")
                exact_score = r_exact['score']
                
                # Reconstruct and get score
                F_hat = tr.reconstruct()
                
                # Get score from reconstructed partition tensor
                # For k=3, the sink is at (len1, len2, len3)
                sink = tuple(s-1 for s in F.shape)
                logZ_hat = np.log(max(F_hat[sink], 1e-300))
                logZ_true = np.log(max(F[sink], 1e-300))
                logZ_err = abs(logZ_hat - logZ_true)
                
                results.append({
                    'case': case_name,
                    'bond_dim': bond_dim,
                    'compression': result['compression'],
                    'error': result['rel_error'],
                    'time': result['time'],
                    'logZ_error': logZ_err,
                    'exact_score': exact_score
                })
                
                print(f"  r={bond_dim}: compression={result['compression']:.1f}x, logZ_err={logZ_err:.6f}")
                
            except Exception as e:
                print(f"  r={bond_dim}: failed - {e}")
    
    # Save results
    output_file = Path("/mnt/d/mmDst/code/data/tr_balibase_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


def main():
    """Run all TR experiments."""
    print("="*70)
    print("TENSOR-RING DECOMPOSITION FOR MSA")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    tr = TensorRing(use_gpu=True)
    print(f"GPU available: {tr.use_gpu}")
    
    # Run experiments
    experiments = [
        ("TR vs TT Comparison", experiment_tr_vs_tt),
        ("TR Scaling Experiment", experiment_tr_scaling),
        ("TR on BAliBASE", experiment_tr_balibase),
    ]
    
    for name, func in experiments:
        print("\n" + "="*70)
        print(f"Starting: {name}")
        print("="*70)
        
        try:
            func()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL TR EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Results saved to: /mnt/d/mmDst/code/data/tr_*.json")


if __name__ == "__main__":
    main()
