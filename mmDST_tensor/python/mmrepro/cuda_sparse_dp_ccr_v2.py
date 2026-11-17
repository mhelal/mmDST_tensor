"""CCR CUDA backend v2: pinned H2D, resident CSR, CUDA Graph replay.

CUDA Graphs are optional and fall back to ordinary stream launches if capture
is unavailable in the installed CuPy/CUDA combination.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from .sparse_dp import SparseGraph


def cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_KERNEL = r'''
extern "C" __global__
void wave_maxplus_i64(
    const long long start, const long long end, const int k, const int local,
    const int* __restrict__ coords,
    const long long* __restrict__ ptr,
    const long long* __restrict__ src,
    const int* __restrict__ ew,
    long long* __restrict__ val)
{
    const long long i = start + (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= end) return;
    bool boundary = false;
    bool origin = true;
    long long tau = 1;
    #pragma unroll 1
    for (int d=0; d<k; ++d) {
        const int x = coords[i*k+d];
        boundary |= (x == 0);
        origin &= (x == 0);
        tau *= (long long)(x + 1);
    }
    // The DP identity state is V[0,...,0] = 0.
    if (origin) {
        val[i] = 0LL;
        return;
    }
    if (boundary) {
        val[i] = local ? 0LL : -6LL * tau;
        return;
    }
    const long long NEG = -0x3fffffffffffffffLL;
    long long best = NEG;
    for (long long e=ptr[i]; e<ptr[i+1]; ++e) {
        const long long cand = val[src[e]] + (long long)ew[e];
        if (cand > best) best = cand;
    }
    if (local && best < 0LL) best = 0LL;
    val[i] = best;
}
'''


def _pinned_copy(a, dtype):
    import cupy as cp
    x = np.ascontiguousarray(a, dtype=dtype)
    mem = cp.cuda.alloc_pinned_memory(x.nbytes)
    p = np.frombuffer(mem, dtype=x.dtype, count=x.size).reshape(x.shape)
    p[...] = x
    return mem, p


@dataclass
class DeviceSparseGraphV2:
    graph: SparseGraph
    coords: object
    ptr: object
    src: object
    ew: object
    val: object
    stream: object
    kernel: object
    h2d_sec: float
    h2d_bytes: int
    pinned_refs: list = field(default_factory=list)
    graph_cache: dict = field(default_factory=dict)

    @classmethod
    def from_host(cls, graph):
        import cupy as cp
        if not cuda_available():
            raise RuntimeError("CUDA/CuPy unavailable")

        # CuPy already pools device allocations; use a dedicated non-blocking
        # stream and pinned host staging so H2D copies can be asynchronous.
        stream = cp.cuda.Stream(non_blocking=True)
        refs = []
        hm, hc = _pinned_copy(graph.coords.ravel(), np.int32); refs.append(hm)
        hm, hp = _pinned_copy(graph.edge_ptr, np.int64); refs.append(hm)
        hm, hs = _pinned_copy(graph.edge_src, np.int64); refs.append(hm)
        hm, he = _pinned_copy(graph.edge_w, np.int32); refs.append(hm)
        nbytes = hc.nbytes + hp.nbytes + hs.nbytes + he.nbytes

        with stream:
            dc = cp.empty(hc.shape, dtype=cp.int32)
            dp = cp.empty(hp.shape, dtype=cp.int64)
            ds = cp.empty(hs.shape, dtype=cp.int64)
            de = cp.empty(he.shape, dtype=cp.int32)
            dv = cp.empty(graph.n_states, dtype=cp.int64)
            t0 = time.perf_counter()
            dc.set(hc, stream=stream); dp.set(hp, stream=stream)
            ds.set(hs, stream=stream); de.set(he, stream=stream)
        stream.synchronize()
        h2d = time.perf_counter() - t0
        ker = cp.RawKernel(_KERNEL, "wave_maxplus_i64", backend="nvrtc")
        return cls(graph, dc, dp, ds, de, dv, stream, ker, h2d, nbytes, refs, {})

    def _launch_waves(self, mode, block_size):
        local = np.int32(mode == "local")
        k = np.int32(self.graph.coords.shape[1])
        widths = np.diff(self.graph.wave_offsets)
        for w, n0 in enumerate(widths):
            n = int(n0)
            if n <= 0: continue
            a = int(self.graph.wave_offsets[w]); b = int(self.graph.wave_offsets[w+1])
            blocks = (n + block_size - 1) // block_size
            self.kernel((blocks,), (block_size,),
                        (np.int64(a), np.int64(b), k, local,
                         self.coords, self.ptr, self.src, self.ew, self.val),
                        stream=self.stream)

    def capture(self, mode, block_size):
        key = (mode, int(block_size))
        if key in self.graph_cache:
            return self.graph_cache[key]
        try:
            with self.stream:
                self.stream.begin_capture()
                self._launch_waves(mode, block_size)
                cg = self.stream.end_capture()
            # Upload removes some first-launch setup from the timed repeats.
            if hasattr(cg, "upload"):
                cg.upload(stream=self.stream)
                self.stream.synchronize()
            self.graph_cache[key] = cg
            return cg
        except Exception:
            # Safe fallback for older CuPy/CUDA or capture restrictions.
            self.graph_cache[key] = None
            return None

    def execute(self, mode="global", block_size=256, use_cuda_graph=True, fetch_score=False):
        import cupy as cp
        cg = self.capture(mode, block_size) if use_cuda_graph else None
        start = cp.cuda.Event(); stop = cp.cuda.Event()
        start.record(self.stream)
        if cg is not None:
            cg.launch(stream=self.stream)
        else:
            with self.stream:
                self._launch_waves(mode, block_size)
        stop.record(self.stream); stop.synchronize()
        sec = cp.cuda.get_elapsed_time(start, stop) / 1000.0
        result = {"kernel_sec": float(sec), "cuda_graph": int(cg is not None),
                  "block_size": int(block_size)}
        if fetch_score:
            if mode == "global":
                sink = np.where(np.all(self.graph.coords == np.asarray(self.graph.lengths), axis=1))[0]
                anchor = int(sink[0]) if sink.size else -1
                result["score"] = int(self.val[anchor].get(stream=self.stream)) if anchor >= 0 else None
            else:
                # Score retrieval is outside the timed region.
                result["score"] = int(cp.max(self.val).get(stream=self.stream))
        return result


def benchmark_resident_v2(graph, mode="global", block_sizes=(128,256,512), repeats=10,
                           warmup=2, use_cuda_graph=True, autotune=False):
    dev = DeviceSparseGraphV2.from_host(graph)
    widths = np.diff(graph.wave_offsets); nz = widths[widths > 0]
    common = dict(h2d_sec=dev.h2d_sec, h2d_bytes=dev.h2d_bytes,
                  launches=int(np.count_nonzero(widths)),
                  max_wave_width=int(widths.max()) if len(widths) else 0,
                  mean_nonempty_wave_width=float(nz.mean()) if len(nz) else 0.0)

    candidates = list(block_sizes)
    if autotune and len(candidates) > 1:
        pilot = []
        for bs in candidates:
            for _ in range(max(1, warmup)):
                dev.execute(mode, bs, use_cuda_graph, False)
            times = [dev.execute(mode, bs, use_cuda_graph, False)["kernel_sec"] for _ in range(3)]
            pilot.append((float(np.median(times)), bs))
        candidates = [min(pilot)[1]]

    rows = []
    # Validate/fetch the score once per chosen block size, not every repeat.
    for bs in candidates:
        for _ in range(warmup): dev.execute(mode, bs, use_cuda_graph, False)
        score = dev.execute(mode, bs, use_cuda_graph, True).get("score")
        for rep in range(repeats):
            r = dev.execute(mode, bs, use_cuda_graph, False)
            r.update(common); r.update(repeat=rep, score=score)
            rows.append(r)
    return rows
