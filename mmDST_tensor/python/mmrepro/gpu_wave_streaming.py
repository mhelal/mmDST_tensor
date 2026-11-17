"""GPU-resident wave-streaming k-way MSA Forward DP.

Primary invariants
------------------
* Never materialise the dense (L+1)^k tensor.
* Never materialise a whole sparse CSR graph.
* Generate only states belonging to admissible partitions of the current wave.
* Keep only the previous k waves, because a k-way MSA transition decreases
  the wave index by popcount(mask) in [1,k].
* Discover predecessors arithmetically from mixed-radix keys and score them
  immediately; edges are counted for CCR but are never stored globally.
* Consecutive tiny waves are scored inside ONE CUDA block/kernel with
  __syncthreads barriers, eliminating one-launch-per-wave overhead at the two
  ends of the wavefront.  Wide waves use a normal many-block kernel and saturate
  the GPU; optional tiling is used only when it is required by a user cap.
* Retain a bounded bottom-k (hash-priority) observation reservoir on device for
  downstream Tensor-Ring fitting.

The host is a control plane only.  Sequence bytes, wave keys/values, CCR
counters and TR observations stay in VRAM until final scalar/result export.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Sequence

import numpy as np

GAP, MATCH, MIS = -6, 3, -2
MAX_K = 10


def cuda_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def mixed_radix_strides(lengths: Sequence[int]) -> np.ndarray:
    out = np.ones(len(lengths), dtype=np.int64)
    acc = 1
    lim = np.iinfo(np.int64).max
    for i, L in enumerate(lengths):
        out[i] = acc
        dim = int(L) + 1
        if acc > lim // dim:
            raise OverflowError("mixed-radix coordinate key exceeds int64")
        acc *= dim
    return out


def _part_shape(lengths: Sequence[int], P: int) -> np.ndarray:
    return np.asarray([(int(L) // int(P)) + 1 for L in lengths], dtype=np.int32)


def _part_strides(pshape: Sequence[int]) -> np.ndarray:
    s = np.ones(len(pshape), dtype=np.int64)
    acc = 1
    for i, n in enumerate(pshape):
        s[i] = acc
        acc *= int(n)
    return s


def _mask_tables(k: int, strides: np.ndarray):
    n = 1 << k
    d = np.zeros(n, dtype=np.int64)
    p = np.zeros(n, dtype=np.int32)
    for m in range(1, n):
        bits = [i for i in range(k) if (m >> i) & 1]
        d[m] = int(strides[bits].sum())
        p[m] = len(bits)
    return d, p


def _normalise_deltas(k: int, deltas: Sequence[Sequence[int]] | None):
    zero = tuple(0 for _ in range(k))
    dd = [zero]
    if deltas:
        for x in deltas:
            t = tuple(int(v) for v in x)
            if t not in dd:
                dd.append(t)
    return np.asarray(dd, dtype=np.int32)


_PARTITION_KERNEL = r'''
extern "C" __global__
void emit_partitions(
    const int k, const int eps, const int base,
    const int npw, const int* __restrict__ pws,
    const int nd, const int* __restrict__ deltas,
    const int* __restrict__ pshape,
    const long long* __restrict__ pstrides,
    const long long ncode,
    long long* __restrict__ out)
{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = (long long)npw * nd * ncode;
    if (t >= total) return;
    long long q = t;
    const long long code0 = q % ncode; q /= ncode;
    const int di = (int)(q % nd); q /= nd;
    const int pw = pws[(int)q];

    int mp[10];
    const int qq = pw / k, rr = pw - qq*k;
    const int lo = (k - rr) / 2, hi = (k + rr) / 2;
    for (int i=0;i<k;i++) mp[i] = qq + ((lo <= i && i < hi) ? 1 : 0);

    int z[10];
    long long code = code0;
    int sumz = 0;
    for (int i=0;i<k-1;i++) {
        const int digit = (int)(code % base); code /= base;
        z[i] = digit - eps; sumz += z[i];
    }
    int sumd = 0;
    for (int i=0;i<k;i++) sumd += deltas[di*k+i];
    z[k-1] = -sumd - sumz;
    if (z[k-1] < -eps || z[k-1] > eps) { out[t] = -1; return; }

    long long pk = 0;
    int sump = 0;
    for (int i=0;i<k;i++) {
        const int p = mp[i] + deltas[di*k+i] + z[i];
        if (p < 0 || p >= pshape[i]) { out[t] = -1; return; }
        sump += p;
        pk += (long long)p * pstrides[i];
    }
    out[t] = (sump == pw) ? pk : -1;
}
'''

_CELL_KERNEL = r'''
extern "C" __global__
void emit_cells_for_wave(
    const int k, const int P, const int wave,
    const long long nparts,
    const long long* __restrict__ partkeys,
    const int* __restrict__ pshape,
    const long long* __restrict__ pstrides,
    const int* __restrict__ lengths,
    const long long* __restrict__ strides,
    const long long local_count,
    long long* __restrict__ out)
{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = nparts * local_count;
    if (t >= total) return;
    const long long pi = t / local_count;
    long long lc = t - pi*local_count;
    long long pk = partkeys[pi];
    long long key = 0;
    int sw = 0;
    for (int d=0; d<k; ++d) {
        const int p = (int)((pk / pstrides[d]) % (long long)pshape[d]);
        const int r = (int)(lc % P); lc /= P;
        const int c = p*P + r;
        if (c > lengths[d]) { out[t] = -1; return; }
        sw += c;
        key += (long long)c * strides[d];
    }
    out[t] = (sw == wave) ? key : -1;
}
'''

_FULL_WAVE_KERNEL = r'''
extern "C" __global__
void emit_full_wave(
    const int k, const int wave,
    const int* __restrict__ lengths,
    const long long* __restrict__ strides,
    const long long ncombo,
    long long* __restrict__ out)
{
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ncombo) return;
    long long q=t, key=0;
    int sum=0;
    for (int d=0; d<k-1; ++d) {
        const int dim=lengths[d]+1;
        const int c=(int)(q % dim); q/=dim;
        sum += c; key += (long long)c*strides[d];
    }
    const int last=wave-sum;
    if (last < 0 || last > lengths[k-1]) { out[t]=-1; return; }
    key += (long long)last*strides[k-1];
    out[t]=key;
}
'''

_SCORE_COMMON = r'''
// NVRTC-portable: do not include host libc math.h. CUDA device math
// functions (exp/log) are NVRTC built-ins; use an explicit sentinel.
#define MM_NEG_INF (-1.7976931348623157e+308)
__device__ __forceinline__ long long lower_bound_ll(const long long* a, long long n, long long x) {
    long long lo=0, hi=n;
    while (lo<hi) { const long long m=(lo+hi)>>1; if (a[m] < x) lo=m+1; else hi=m; }
    return lo;
}
__device__ __forceinline__ double logadd(double a, double b) {
    if (a <= MM_NEG_INF) return b;
    if (b <= MM_NEG_INF) return a;
    const double m = a>b ? a:b;
    const double d = a>b ? (a-b) : (b-a);
    return m + log(1.0 + exp(-d));
}
__device__ __forceinline__ long long warp_sum_ll(long long x) {
    for (int off=16; off>0; off>>=1) x += __shfl_down_sync(0xffffffff, x, off);
    return x;
}
__device__ __forceinline__ void decode_key(long long key, int k, const int* lengths,
                                            const long long* strides, int* c) {
    #pragma unroll 1
    for (int d=0; d<k; ++d) c[d]=(int)((key/strides[d]) % (long long)(lengths[d]+1));
}
__device__ __forceinline__ int edge_weight_mask(
    int mask, int pop, int k, const int* c,
    const unsigned char* seq, const int* soff)
{
    int w=(k-pop)*(-6);
    #pragma unroll 1
    for (int i=0;i<k;i++) if (mask&(1<<i)) {
        const unsigned char a=seq[soff[i]+c[i]-1];
        for (int j=i+1;j<k;j++) if (mask&(1<<j)) {
            const unsigned char b=seq[soff[j]+c[j]-1];
            w += (a==b)?3:((a==45 || b==45)?-6:-2);
        }
    }
    return w;
}
'''

_SCORE_WAVE_KERNEL = _SCORE_COMMON + r'''
extern "C" __global__
void score_wave_forward(
    const long long n,
    const int k, const int P, const int local,
    const double beta,
    const long long* __restrict__ keys,
    double* __restrict__ vals,
    const unsigned long long* __restrict__ hkeyptr,
    const unsigned long long* __restrict__ hvalptr,
    const long long* __restrict__ hn,
    const int* __restrict__ lengths,
    const long long* __restrict__ strides,
    const long long* __restrict__ mdelta,
    const int* __restrict__ mpop,
    const unsigned char* __restrict__ seq,
    const int* __restrict__ soff,
    unsigned long long* __restrict__ counters)
{
    long long accepted_local=0, cross_local=0;
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    const long long step=(long long)blockDim.x*gridDim.x;
    for (long long ii=tid; ii<n; ii+=step) {
        const long long key=keys[ii];
        int c[10]; decode_key(key,k,lengths,strides,c);
        bool origin=true; int zero_mask=0;
        for (int d=0;d<k;d++) { origin &= (c[d]==0); if (c[d]==0) zero_mask|=(1<<d); }
        if (origin) { vals[ii]=0.0; continue; }
        double lse=MM_NEG_INF;
        const int nm=1<<k;
        for (int m=1;m<nm;m++) {
            if (m & zero_mask) continue;
            const int lag=mpop[m];
            const long long ns=hn[lag]; if (ns<=0) continue;
            const long long* sk=(const long long*)(hkeyptr[lag]);
            const double* sv=(const double*)(hvalptr[lag]);
            const long long pk=key-md[0];
        }
    }
}
'''
# The source above is intentionally assembled below with a replacement. NVRTC
# diagnostics are clearer when the common helper text remains separate.
_SCORE_WAVE_KERNEL = _SCORE_COMMON + r'''
extern "C" __global__
void score_wave_forward(
    const long long n, const int k, const int P, const int local, const double beta,
    const long long* __restrict__ keys, double* __restrict__ vals,
    const unsigned long long* __restrict__ hkeyptr,
    const unsigned long long* __restrict__ hvalptr,
    const long long* __restrict__ hn,
    const int* __restrict__ lengths, const long long* __restrict__ strides,
    const long long* __restrict__ mdelta, const int* __restrict__ mpop,
    const unsigned char* __restrict__ seq, const int* __restrict__ soff,
    unsigned long long* __restrict__ counters)
{
    long long accepted_local=0, cross_local=0;
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    const long long step=(long long)blockDim.x*gridDim.x;
    for (long long ii=tid; ii<n; ii+=step) {
        const long long key=keys[ii];
        int c[10]; decode_key(key,k,lengths,strides,c);
        bool origin=true; int zero_mask=0;
        for (int d=0;d<k;d++) { origin &= (c[d]==0); if (c[d]==0) zero_mask|=(1<<d); }
        if (origin) { vals[ii]=0.0; continue; }
        double lse=MM_NEG_INF;
        const int nm=1<<k;
        for (int m=1;m<nm;m++) {
            if (m & zero_mask) continue;
            const int lag=mpop[m];
            const long long ns=hn[lag]; if (ns<=0) continue;
            const long long* sk=(const long long*)(hkeyptr[lag]);
            const double* sv=(const double*)(hvalptr[lag]);
            const long long pk=key-mdelta[m];
            const long long pos=lower_bound_ll(sk,ns,pk);
            if (pos>=ns || sk[pos]!=pk) continue;
            const int ew=edge_weight_mask(m,lag,k,c,seq,soff);
            lse=logadd(lse,sv[pos]+beta*(double)ew);
            accepted_local++;
            bool cross=false;
            if (P>1) for (int d=0;d<k;d++) if ((m&(1<<d)) && (c[d]%P==0)) { cross=true; break; }
            if (P==1) cross=true;
            if (cross) cross_local++;
        }
        vals[ii]=local ? logadd(0.0,lse) : lse;
    }
    accepted_local=warp_sum_ll(accepted_local); cross_local=warp_sum_ll(cross_local);
    if ((threadIdx.x & 31)==0) {
        atomicAdd((unsigned long long*)&counters[0],(unsigned long long)accepted_local);
        atomicAdd((unsigned long long*)&counters[1],(unsigned long long)cross_local);
    }
}
'''

_SCORE_BUNDLE_KERNEL = _SCORE_COMMON + r'''
extern "C" __global__
void score_small_bundle_forward(
    const int B, const int k, const int P, const int local, const double beta,
    const long long* __restrict__ keys, double* __restrict__ vals,
    const long long* __restrict__ boff,
    const unsigned long long* __restrict__ old_kptr,
    const unsigned long long* __restrict__ old_vptr,
    const long long* __restrict__ old_n,
    const int* __restrict__ lengths, const long long* __restrict__ strides,
    const long long* __restrict__ mdelta, const int* __restrict__ mpop,
    const unsigned char* __restrict__ seq, const int* __restrict__ soff,
    unsigned long long* __restrict__ counters)
{
    // Exactly one block.  Consecutive dependent waves are separated by a block
    // barrier, so this removes host/kernel launch latency without violating DP.
    for (int j=0;j<B;j++) {
        long long accepted_local=0, cross_local=0;
        const long long a=boff[j], b=boff[j+1];
        for (long long ii=a+threadIdx.x; ii<b; ii+=blockDim.x) {
            const long long key=keys[ii];
            int c[10]; decode_key(key,k,lengths,strides,c);
            bool origin=true; int zero_mask=0;
            for (int d=0;d<k;d++) { origin &= (c[d]==0); if (c[d]==0) zero_mask|=(1<<d); }
            if (origin) { vals[ii]=0.0; continue; }
            double lse=MM_NEG_INF;
            const int nm=1<<k;
            for (int m=1;m<nm;m++) {
                if (m & zero_mask) continue;
                const int lag=mpop[m];
                const long long srcj=(long long)j-lag;
                const long long* sk; const double* sv; long long ns;
                if (srcj>=0) {
                    sk=keys+boff[srcj]; sv=vals+boff[srcj]; ns=boff[srcj+1]-boff[srcj];
                } else {
                    const int oldlag=(int)(-srcj);
                    if (oldlag>k) continue;
                    ns=old_n[oldlag]; if (ns<=0) continue;
                    sk=(const long long*)(old_kptr[oldlag]);
                    sv=(const double*)(old_vptr[oldlag]);
                }
                const long long pk=key-mdelta[m];
                const long long pos=lower_bound_ll(sk,ns,pk);
                if (pos>=ns || sk[pos]!=pk) continue;
                const int ew=edge_weight_mask(m,lag,k,c,seq,soff);
                lse=logadd(lse,sv[pos]+beta*(double)ew);
                accepted_local++;
                bool cross=false;
                if (P>1) for (int d=0;d<k;d++) if ((m&(1<<d)) && (c[d]%P==0)) { cross=true; break; }
                if (P==1) cross=true;
                if (cross) cross_local++;
            }
            vals[ii]=local ? logadd(0.0,lse) : lse;
        }
        accepted_local=warp_sum_ll(accepted_local); cross_local=warp_sum_ll(cross_local);
        if ((threadIdx.x&31)==0) {
            atomicAdd((unsigned long long*)&counters[2*j],(unsigned long long)accepted_local);
            atomicAdd((unsigned long long*)&counters[2*j+1],(unsigned long long)cross_local);
        }
        __syncthreads();
    }
}
'''


@dataclass
class Wave:
    index: int
    keys: object
    vals: object

    @property
    def n(self):
        return int(self.keys.size)


class DeviceReservoir:
    """Bounded bottom-k hash reservoir, entirely device resident."""
    def __init__(self, capacity: int, seed: int):
        import cupy as cp
        self.cp=cp; self.capacity=max(0,int(capacity)); self.seed=np.uint64(seed)
        self.keys=cp.empty(0,dtype=cp.int64)
        self.logv=cp.empty(0,dtype=cp.float64)
        self.prio=cp.empty(0,dtype=cp.uint64)

    def _priority(self, keys):
        cp=self.cp
        x=keys.astype(cp.uint64,copy=False) + cp.uint64(self.seed) + cp.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>30))*cp.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>27))*cp.uint64(0x94D049BB133111EB)
        return x^(x>>31)

    def update(self, keys, logv):
        cp=self.cp
        if self.capacity<=0 or keys.size==0: return
        p=self._priority(keys)
        kk=cp.concatenate((self.keys,keys)); vv=cp.concatenate((self.logv,logv)); pp=cp.concatenate((self.prio,p))
        if kk.size>self.capacity:
            ids=cp.argpartition(pp,self.capacity-1)[:self.capacity]
            kk=kk[ids]; vv=vv[ids]; pp=pp[ids]
        self.keys,self.logv,self.prio=kk,vv,pp

    def ensure(self,key:int,logv:float):
        cp=self.cp
        if self.capacity<=0: return
        hit=cp.any(self.keys==cp.int64(key)) if self.keys.size else cp.asarray(False)
        if bool(hit.item()): return
        k=cp.asarray([key],dtype=cp.int64); v=cp.asarray([logv],dtype=cp.float64); p=cp.asarray([0],dtype=cp.uint64)
        kk=cp.concatenate((k,self.keys)); vv=cp.concatenate((v,self.logv)); pp=cp.concatenate((p,self.prio))
        self.keys,self.logv,self.prio=kk[:self.capacity],vv[:self.capacity],pp[:self.capacity]


def decode_keys_device(keys, lengths: Sequence[int], strides=None):
    import cupy as cp
    if strides is None: strides=mixed_radix_strides(lengths)
    dstr=cp.asarray(strides,dtype=cp.int64)
    dims=cp.asarray([int(L)+1 for L in lengths],dtype=cp.int64)
    return ((keys[:,None]//dstr[None,:]) % dims[None,:]).astype(cp.int64)


class GpuWaveStreamingForward:
    def __init__(self, seqs: Sequence[str], P=1, eps=0, deltas=None, beta=0.1,
                 mode='global', block_size=256, small_wave_threshold=1024,
                 small_bundle_max_waves=16, reservoir_size=1_000_000,
                 seed=12345, vram_fraction=0.72, hard_vram_gb=11.5,
                 max_blocks_per_sm=8, profile_waves=False):
        if not cuda_available(): raise RuntimeError('CUDA/CuPy unavailable')
        import cupy as cp
        self.cp=cp; self.seqs=list(seqs); self.lengths=tuple(map(len,seqs)); self.k=len(seqs)
        if not (1 <= self.k <= MAX_K): raise ValueError(f'k must be 1..{MAX_K}')
        self.P=int(P); self.eps=None if eps is None else int(eps); self.beta=float(beta); self.mode=mode
        self.block=int(block_size); self.small=int(small_wave_threshold); self.bundle_max=int(small_bundle_max_waves)
        self.vram_fraction=float(vram_fraction); self.hard_vram=int(float(hard_vram_gb)*(1024**3)); self.max_blocks_per_sm=int(max_blocks_per_sm)
        self.profile=bool(profile_waves); self.maxwave=sum(self.lengths)
        self.strides=mixed_radix_strides(self.lengths); self.pshape=_part_shape(self.lengths,self.P); self.pstrides=_part_strides(self.pshape)
        md,mp=_mask_tables(self.k,self.strides)
        self.dlen=cp.asarray(self.lengths,dtype=cp.int32); self.dstr=cp.asarray(self.strides,dtype=cp.int64)
        self.dpshape=cp.asarray(self.pshape,dtype=cp.int32); self.dpstr=cp.asarray(self.pstrides,dtype=cp.int64)
        self.dmd=cp.asarray(md,dtype=cp.int64); self.dmp=cp.asarray(mp,dtype=cp.int32)
        dd=_normalise_deltas(self.k,deltas); self.deltas_host=dd; self.dd=cp.asarray(dd,dtype=cp.int32)
        # Pack sequence bytes once.  No per-wave H2D sequence traffic.
        offsets=[0]; chunks=[]
        for s in self.seqs:
            b=np.frombuffer(s.encode('ascii'),dtype=np.uint8).copy(); chunks.append(b); offsets.append(offsets[-1]+len(b))
        packed=np.concatenate(chunks) if chunks else np.empty(0,np.uint8)
        self.dseq=cp.asarray(packed,dtype=cp.uint8); self.dsoff=cp.asarray(offsets[:-1],dtype=cp.int32)
        self.k_part=cp.RawKernel(_PARTITION_KERNEL,'emit_partitions',backend='nvrtc')
        self.k_cell=cp.RawKernel(_CELL_KERNEL,'emit_cells_for_wave',backend='nvrtc')
        self.k_full=cp.RawKernel(_FULL_WAVE_KERNEL,'emit_full_wave',backend='nvrtc')
        self.k_score=cp.RawKernel(_SCORE_WAVE_KERNEL,'score_wave_forward',backend='nvrtc')
        self.k_bundle=cp.RawKernel(_SCORE_BUNDLE_KERNEL,'score_small_bundle_forward',backend='nvrtc')
        self.reservoir=DeviceReservoir(reservoir_size,seed)
        self.seed=seed
        self.history={}
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.sms=int(props.get('multiProcessorCount',props.get(b'multiProcessorCount',1)))
        self.geom_stream=cp.cuda.Stream(non_blocking=True); self.compute_stream=cp.cuda.Stream(non_blocking=True)
        self.stats=dict(states=0,accepted_edges=0,cross_edges=0,score_launches=0,small_bundle_launches=0,
                        bundled_waves=0,single_wave_launches=0,geometry_sec=0.0,score_sec=0.0,
                        peak_active_bytes=0,peak_wave_states=0,waves_nonempty=0,max_bundle_size=0)
        self.wave_rows=[]
        self.global_max=-math.inf; self.global_max_key=0

    def _vram_guard(self, need_bytes:int, label:str):
        cp=self.cp
        free,total=cp.cuda.runtime.memGetInfo(); cap=min(self.hard_vram,int(total*self.vram_fraction))
        pool=cp.get_default_memory_pool(); active=int(pool.used_bytes())
        if active+int(need_bytes) > cap or int(need_bytes) > int(free*0.85):
            # Cached blocks can make runtime free memory look lower than actual reusable memory.
            pool.free_all_blocks(); free,_=cp.cuda.runtime.memGetInfo(); active=int(pool.used_bytes())
        if active+int(need_bytes) > cap or int(need_bytes) > int(free*0.90):
            raise MemoryError(f'GPU resource guard at {label}: need~{need_bytes/2**30:.2f} GiB, active={active/2**30:.2f} GiB, free={free/2**30:.2f} GiB, cap={cap/2**30:.2f} GiB')

    def _candidate_partition_waves(self,w:int):
        if self.P<=1: return [w]
        lo=max(0,math.ceil((w-self.k*(self.P-1))/self.P)); hi=min(int(sum(self.pshape-1)),w//self.P)
        return list(range(lo,hi+1)) if lo<=hi else []

    def _generate_full_wave(self,w:int,stream):
        cp=self.cp
        ncombo=1
        for L in self.lengths[:-1]: ncombo*=int(L)+1
        self._vram_guard(ncombo*8,'full-wave candidates')
        out=cp.empty(ncombo,dtype=cp.int64)
        blocks=(ncombo+self.block-1)//self.block
        self.k_full((blocks,),(self.block,),(np.int32(self.k),np.int32(w),self.dlen,self.dstr,np.int64(ncombo),out),stream=stream)
        keys=out[out>=0]
        return cp.sort(keys)

    def generate_wave(self,w:int,stream=None):
        cp=self.cp; stream=stream or self.geom_stream
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(stream)
        with stream:
            if self.eps is None:
                keys=self._generate_full_wave(w,stream)
            else:
                pws=self._candidate_partition_waves(w)
                if not pws:
                    keys=cp.empty(0,dtype=cp.int64)
                else:
                    dpws=cp.asarray(pws,dtype=cp.int32)
                    base=2*self.eps+1; ncode=base**max(0,self.k-1); nd=int(self.dd.shape[0]); total=len(pws)*nd*ncode
                    self._vram_guard(total*8,'partition candidates')
                    cand=cp.empty(total,dtype=cp.int64)
                    blocks=(total+self.block-1)//self.block
                    self.k_part((blocks,),(self.block,),(np.int32(self.k),np.int32(self.eps),np.int32(base),np.int32(len(pws)),dpws,
                                np.int32(nd),self.dd,self.dpshape,self.dpstr,np.int64(ncode),cand),stream=stream)
                    parts=cp.unique(cand[cand>=0])
                    local_count=self.P**self.k; ncell=int(parts.size)*local_count
                    self._vram_guard(ncell*8,'wave cell candidates')
                    raw=cp.empty(ncell,dtype=cp.int64)
                    blocks=(ncell+self.block-1)//self.block if ncell else 0
                    if blocks:
                        self.k_cell((blocks,),(self.block,),(np.int32(self.k),np.int32(self.P),np.int32(w),np.int64(parts.size),parts,
                                    self.dpshape,self.dpstr,self.dlen,self.dstr,np.int64(local_count),raw),stream=stream)
                        keys=cp.sort(raw[raw>=0])
                    else: keys=cp.empty(0,dtype=cp.int64)
        e1.record(stream); e1.synchronize(); self.stats['geometry_sec']+=float(cp.cuda.get_elapsed_time(e0,e1))/1000.0
        return keys

    def _history_ptrs(self,base_wave:int):
        cp=self.cp
        kp=np.zeros(self.k+1,dtype=np.uint64); vp=np.zeros(self.k+1,dtype=np.uint64); nn=np.zeros(self.k+1,dtype=np.int64)
        for lag in range(1,self.k+1):
            x=self.history.get(base_wave-lag)
            if x is not None:
                kp[lag]=np.uint64(x.keys.data.ptr); vp[lag]=np.uint64(x.vals.data.ptr); nn[lag]=x.n
        return cp.asarray(kp),cp.asarray(vp),cp.asarray(nn)

    def _record_wave(self,w,keys,vals,accepted,cross,score_sec,launch_kind):
        cp=self.cp; n=int(keys.size)
        self.stats['states']+=n; self.stats['accepted_edges']+=int(accepted); self.stats['cross_edges']+=int(cross)
        self.stats['peak_wave_states']=max(self.stats['peak_wave_states'],n); self.stats['waves_nonempty']+=int(n>0)
        if n:
            # Device-side observation retention before the wave can expire.
            self.reservoir.update(keys,vals)
            # This scalar synchronization is profiling/control only; no arrays move to host.
            j=int(cp.argmax(vals).item()); mv=float(vals[j].item())
            if mv>self.global_max: self.global_max=mv; self.global_max_key=int(keys[j].item())
        active=sum(int(x.keys.nbytes+x.vals.nbytes) for x in self.history.values()) + int(keys.nbytes+vals.nbytes)
        self.stats['peak_active_bytes']=max(self.stats['peak_active_bytes'],active)
        if self.profile:
            self.wave_rows.append(dict(wave=w,states=n,accepted_edges=int(accepted),cross_edges=int(cross),kernel_sec=float(score_sec),launch_kind=launch_kind))

    def _expire(self,current_wave:int):
        # Wave current-k can still be used by current; after current is complete it cannot be used again.
        old=current_wave-self.k
        if old in self.history: del self.history[old]

    def launch_single(self,w:int,keys):
        """Launch one wide wave without synchronising, enabling geometry prefetch.

        All launch-side temporaries are created on compute_stream.  This is required
        because compute_stream is non-blocking: constructing pointer tables/counters
        on the default stream and immediately dereferencing them from compute_stream
        is a real cross-stream race and can surface as cudaErrorIllegalAddress.
        """
        cp=self.cp
        with self.compute_stream:
            vals=cp.empty(keys.size,dtype=cp.float64)
            kp,vp,nn=self._history_ptrs(w)
            counters=cp.zeros(2,dtype=cp.uint64)
            e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(self.compute_stream)
            if keys.size:
                blocks=min((int(keys.size)+self.block-1)//self.block,max(1,self.sms*self.max_blocks_per_sm))
                self.k_score((blocks,),(self.block,),(np.int64(keys.size),np.int32(self.k),np.int32(self.P),np.int32(self.mode=='local'),np.float64(self.beta),
                             keys,vals,kp,vp,nn,self.dlen,self.dstr,self.dmd,self.dmp,self.dseq,self.dsoff,counters),stream=self.compute_stream)
            e1.record(self.compute_stream)
        return (w,keys,vals,counters,e0,e1,kp,vp,nn)

    def finish_single(self,token):
        cp=self.cp; w,keys,vals,counters,e0,e1,kp,vp,nn=token
        e1.synchronize(); sec=float(cp.cuda.get_elapsed_time(e0,e1))/1000.0
        cc=cp.asnumpy(counters); self.stats['score_sec']+=sec; self.stats['score_launches']+=int(keys.size>0); self.stats['single_wave_launches']+=int(keys.size>0)
        wave=Wave(w,keys,vals); self._record_wave(w,keys,vals,int(cc[0]),int(cc[1]),sec,'single'); self.history[w]=wave; self._expire(w)
        return wave

    def score_single(self,w:int,keys):
        return self.finish_single(self.launch_single(w,keys))

    def score_bundle(self,base_w:int,key_list):
        cp=self.cp; B=len(key_list)
        if B==1: return [self.score_single(base_w,key_list[0])]
        sizes=[int(x.size) for x in key_list]; off=np.zeros(B+1,dtype=np.int64); np.cumsum(sizes,out=off[1:])
        # Keep concatenation, offsets, pointer tables, counters and the kernel in one
        # non-blocking stream so the kernel cannot observe uninitialised pointer data.
        with self.compute_stream:
            keys=cp.concatenate(key_list) if sum(sizes) else cp.empty(0,dtype=cp.int64)
            vals=cp.empty(keys.size,dtype=cp.float64)
            doff=cp.asarray(off)
            kp,vp,nn=self._history_ptrs(base_w)
            counters=cp.zeros(2*B,dtype=cp.uint64)
            e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(self.compute_stream)
            if keys.size:
                self.k_bundle((1,),(self.block,),(np.int32(B),np.int32(self.k),np.int32(self.P),np.int32(self.mode=='local'),np.float64(self.beta),keys,vals,doff,
                              kp,vp,nn,self.dlen,self.dstr,self.dmd,self.dmp,self.dseq,self.dsoff,counters),stream=self.compute_stream)
            e1.record(self.compute_stream)
        e1.synchronize(); sec=float(cp.cuda.get_elapsed_time(e0,e1))/1000.0
        cc=cp.asnumpy(counters); self.stats['score_sec']+=sec; self.stats['score_launches']+=int(keys.size>0); self.stats['small_bundle_launches']+=int(keys.size>0)
        self.stats['bundled_waves']+=B; self.stats['max_bundle_size']=max(self.stats['max_bundle_size'],B)
        waves=[]
        for j in range(B):
            kv=keys[off[j]:off[j+1]]; vv=vals[off[j]:off[j+1]]; ww=base_w+j
            a=int(cc[2*j]); x=int(cc[2*j+1]); share=sec*(sizes[j]/max(1,sum(sizes)))
            wave=Wave(ww,kv,vv); self._record_wave(ww,kv,vv,a,x,share,'small_bundle'); self.history[ww]=wave; self._expire(ww); waves.append(wave)
        return waves

    def run(self):
        cp=self.cp; t0=time.perf_counter(); w=0; pending=None
        sink_key=int(sum(int(L)*int(st) for L,st in zip(self.lengths,self.strides))); sink_log=-math.inf
        while w<=self.maxwave:
            if pending is not None and pending[0]==w:
                keys=pending[1]; pending=None
            else:
                keys=self.generate_wave(w)
            if int(keys.size)<=self.small:
                # Fuse consecutive tiny dependent waves into one block/kernel.
                key_list=[keys]; j=w+1
                while j<=self.maxwave and len(key_list)<self.bundle_max:
                    nxt=self.generate_wave(j)
                    if int(nxt.size)>self.small:
                        pending=(j,nxt); break
                    key_list.append(nxt); j+=1
                waves=self.score_bundle(w,key_list)
                for x in waves:
                    if self.mode=='global' and x.index==self.maxwave and x.n:
                        # At w=sum(lengths), the only valid Cartesian coordinate is
                        # the sink itself.  No binary/searchsorted lookup is needed.
                        if x.n != 1 or int(x.keys[0].item()) != sink_key:
                            raise RuntimeError(f'final global wave is not the unique sink: n={x.n}')
                        sink_log=float(x.vals[0].item())
                w += len(key_list)
            else:
                # Wide-wave scoring runs in the compute stream while geometry for
                # the next wave is generated independently in the geometry stream.
                token=self.launch_single(w,keys)
                if w<self.maxwave:
                    nxt=self.generate_wave(w+1,self.geom_stream)
                    pending=(w+1,nxt)
                x=self.finish_single(token)
                if self.mode=='global' and w==self.maxwave and x.n:
                    if x.n != 1 or int(x.keys[0].item()) != sink_key:
                        raise RuntimeError(f'final global wave is not the unique sink: n={x.n}')
                    sink_log=float(x.vals[0].item())
                w+=1
        cp.cuda.Stream.null.synchronize(); wall=time.perf_counter()-t0
        functional=sink_log if self.mode=='global' else self.global_max
        fkey=sink_key if self.mode=='global' else self.global_max_key
        if math.isfinite(functional): self.reservoir.ensure(fkey,functional)
        a=self.stats['accepted_edges']; x=self.stats['cross_edges']
        self.stats['ccr_edge']=float(a/x) if x else math.inf
        self.stats['cross_edge_fraction']=float(x/a) if a else 0.0
        self.stats['wall_sec']=wall; self.stats['functional']=functional; self.stats['functional_key']=fkey
        self.stats['naive_wave_launches']=self.stats['waves_nonempty']
        self.stats['launch_reduction_fraction']=1.0-self.stats['score_launches']/max(1,self.stats['waves_nonempty'])
        self.stats['observation_count']=int(self.reservoir.keys.size)
        return dict(stats=dict(self.stats),wave_rows=list(self.wave_rows),reservoir=self.reservoir,
                    lengths=self.lengths,strides=self.strides,sink_key=sink_key)
