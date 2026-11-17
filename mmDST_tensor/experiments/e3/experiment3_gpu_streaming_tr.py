#!/usr/bin/env python3
"""Experiment 3: GPU-resident streamed CCR + Forward + Tensor Ring.

This is the primary optimized route.  It replaces whole sparse-graph CSR
materialisation by just-in-time GPU wave generation and direct predecessor
lookup.  The only persistent DP state is a circular window of the previous k
waves plus a bounded GPU observation reservoir for TR fitting.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,signal,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; PY=ROOT/'python'
if str(PY) not in sys.path: sys.path.insert(0,str(PY))
from mmrepro.gpu_wave_streaming import (GpuWaveStreamingForward,cuda_available,
                                        decode_keys_device,mixed_radix_strides)
from mmrepro.device_tr_stream import fit_tr_from_cupy,tr_parameter_count,torch_available
try:
    from mmrepro.seeds import shared_qmer_anchors
except Exception:
    shared_qmer_anchors=None

DNA='ACGT'; STOP=False

def _sigint(*_):
    global STOP; STOP=True; print('\n[checkpoint] stop requested; current configuration will finish safely',flush=True)

def ints(s): return [int(x) for x in str(s).split(',') if x.strip()]
def floats(s): return [float(x) for x in str(s).split(',') if x.strip()]
def _symround(x,d): return int(math.floor(x/d+0.5)) if x>=0 else -int(math.floor(-x/d+0.5))

def mutate(s,p,rng):
    out=[]
    for ch in s:
        out.append(rng.choice([x for x in DNA if x!=ch]) if rng.random()<p else ch)
    return ''.join(out)

def synthetic_global(k,L,div,rng):
    anc=''.join(rng.choice(DNA) for _ in range(L))
    return [mutate(anc,div,rng) for _ in range(k)]

def synthetic_local(k,motif_len,flank_len,div,rng):
    motif=''.join(rng.choice(DNA) for _ in range(motif_len)); seqs=[]
    bg=list('EFHIKLMNPQRSTVWY')
    for i in range(k):
        m=mutate(motif,div,rng); left=rng.randrange(flank_len+1); ch=bg[i%len(bg)]
        seqs.append(ch*left+m+ch*(flank_len-left))
    return seqs

def qdeltas(seqs,q,P,max_anchors=8):
    if shared_qmer_anchors is None: return [tuple(0 for _ in seqs)],0,0
    aa=shared_qmer_anchors(seqs,q=q,max_anchors=max_anchors); out=[]; seen=set()
    for x in aa:
        d=tuple(_symround(int(v),P) for v in x[3])
        if d not in seen: seen.add(d); out.append(d)
    return out,len(aa),len(out)

def seed_for(master,*parts):
    h=hashlib.sha256(('|'.join(map(str,parts))).encode()).digest()
    return int(master)+(int.from_bytes(h[:4],'little') & 0x7fffffff)

class Journal:
    def __init__(self,out,protocol):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.path=self.out/'e3_stream_checkpoint.jsonl'; self.wave=self.out/'e3_stream_waves.csv'
        self.fp=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
        meta=self.out/'e3_stream_protocol.json'
        if meta.exists() and self.path.exists():
            old=json.loads(meta.read_text())
            if old.get('fingerprint')!=self.fp: raise SystemExit('Protocol differs: choose a new --out directory.')
        meta.write_text(json.dumps({'fingerprint':self.fp,'protocol':protocol},indent=2))
        self.done=set(); self.rows=[]
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r=json.loads(line); self.rows.append(r); self.done.add(r['key'])
                except Exception: pass
            print(f'[checkpoint] resumed {len(self.done)} result rows',flush=True)
    def add(self,row):
        row=dict(row); row['fingerprint']=self.fp
        with self.path.open('a') as f:
            f.write(json.dumps(row,separators=(',',':'),allow_nan=True)+'\n'); f.flush(); os.fsync(f.fileno())
        self.rows.append(row); self.done.add(row['key'])
    def waves(self,graph_key,rows):
        if not rows: return
        fields=['graph_key','wave','states','accepted_edges','cross_edges','kernel_sec','launch_kind']
        exists=self.wave.exists()
        with self.wave.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if not exists: w.writeheader()
            for r in rows: w.writerow({'graph_key':graph_key,**r})
    def consolidate(self):
        if not self.rows: return
        fields=[]
        for r in self.rows:
            for k in r:
                if k not in fields: fields.append(k)
        tmp=self.out/'e3_stream_raw.csv.tmp'; dst=self.out/'e3_stream_raw.csv'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(self.rows)
        os.replace(tmp,dst)


def gpu_info():
    import cupy as cp
    p=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    def g(name,default=None): return p.get(name,p.get(name.encode(),default))
    free,total=cp.cuda.runtime.memGetInfo()
    return dict(name=(g('name',b'?').decode() if isinstance(g('name'),bytes) else str(g('name'))),
                total_vram_bytes=int(total),free_vram_bytes=int(free),sm_count=int(g('multiProcessorCount',0)),
                cc=f"{g('major',0)}.{g('minor',0)}")


def self_test(args):
    """Small exact support/value check against the existing CPU implementation."""
    if args.skip_self_test: return
    try:
        import cupy as cp
        from mmrepro.sparse_dp import generate_band_cells
        from mmrepro.sparse_graph_fast import build_sparse_graph_fast
    except Exception as e:
        print('[self-test] CPU reference modules unavailable; geometry test skipped:',e,flush=True); return
    seqs=['ACGT','ACGT','ACGT']; P=2; eps=1
    eng=GpuWaveStreamingForward(seqs,P=P,eps=eps,beta=0.1,mode='global',small_wave_threshold=64,
                                small_bundle_max_waves=8,reservoir_size=256,hard_vram_gb=min(2,args.hard_vram_gb),profile_waves=False)
    got=[]
    for w in range(sum(map(len,seqs))+1): got.extend(cp.asnumpy(eng.generate_wave(w)).tolist())
    coords,_=generate_band_cells(tuple(map(len,seqs)),eps,P,None)
    st=mixed_radix_strides(tuple(map(len,seqs))); want=sorted(sum(int(c[i])*int(st[i]) for i in range(3)) for c in coords)
    if sorted(got)!=want: raise RuntimeError('GPU streamed geometry does not match CPU band geometry')
    out=eng.run(); g=build_sparse_graph_fast(seqs,eps,P,None)
    # Independent CPU log-domain Forward recurrence over the small reference CSR.
    # This avoids depending on a separate cuda_forward_log module and keeps the
    # startup correctness test self-contained.
    n=int(g.coords.shape[0]); vals=np.full(n,-np.inf,dtype=np.float64)
    for i in range(n):
        if np.all(g.coords[i] == 0):
            vals[i]=0.0
            continue
        a=int(g.edge_ptr[i]); b=int(g.edge_ptr[i+1])
        if b>a:
            z=vals[g.edge_src[a:b]] + 0.1*g.edge_w[a:b].astype(np.float64)
            m=float(np.max(z))
            lse=m + math.log(float(np.exp(z-m).sum())) if math.isfinite(m) else -math.inf
        else:
            lse=-math.inf
        vals[i]=np.logaddexp(0.0,lse) if False else lse
    sink=np.asarray(tuple(map(len,seqs)),dtype=np.int32)
    hit=np.flatnonzero(np.all(g.coords == sink[None,:],axis=1))
    if hit.size != 1: raise RuntimeError('CPU self-test graph is missing/duplicating the global sink')
    ref_logz=float(vals[int(hit[0])])
    if not math.isclose(out['stats']['functional'],ref_logz,rel_tol=1e-9,abs_tol=1e-9):
        raise RuntimeError(f"GPU streamed Forward mismatch: {out['stats']['functional']} vs {ref_logz}")
    print('[self-test] GPU geometry and Forward match CPU reference',flush=True)


def run_one(J,dataset,case,seqs,mode,k,P,eps,q,div,args,exact=None):
    import cupy as cp
    graph_key=f'{dataset}|{case}|{mode}|k{k}|P{P}|e{eps}|q{q}|d{div}'
    rank_keys=[graph_key+f'|r{r}' for r in args.ranks]
    if all(x in J.done for x in rank_keys): return
    deltas=[]; ncellanchors=npartanchors=0
    if mode=='local': deltas,ncellanchors,npartanchors=qdeltas(seqs,q,P,args.max_anchors)
    print('[stream]',graph_key,flush=True)
    cp.get_default_memory_pool().free_all_blocks()
    try:
        eng=GpuWaveStreamingForward(seqs,P=P,eps=eps,deltas=deltas,beta=args.beta,mode=mode,
            block_size=args.block_size,small_wave_threshold=args.small_wave_threshold,
            small_bundle_max_waves=args.small_bundle_max_waves,reservoir_size=args.reservoir_size,
            seed=seed_for(args.seed,graph_key),vram_fraction=args.vram_fraction,
            hard_vram_gb=args.hard_vram_gb,max_blocks_per_sm=args.max_blocks_per_sm,
            profile_waves=args.profile_waves)
        rout=eng.run(); st=rout['stats']
        if args.profile_waves: J.waves(graph_key,rout['wave_rows'])
        # Convert only the bounded reservoir through zero-copy DLPack.
        res=rout['reservoir']; coords_cp=decode_keys_device(res.keys,rout['lengths'],rout['strides'])
        finite=cp.isfinite(res.logv); coords_cp=coords_cp[finite]; logv_cp=res.logv[finite]
        shift=float(cp.max(logv_cp).item()) if logv_cp.size else 0.0
        target_cp=cp.exp(cp.clip(logv_cp-shift,-80.0,0.0)).astype(cp.float32)
        fkey=int(st['functional_key']); fcoord_cp=decode_keys_device(cp.asarray([fkey],dtype=cp.int64),rout['lengths'],rout['strides'])
        fscaled=math.exp(max(-80.0,min(0.0,float(st['functional'])-shift))) if math.isfinite(st['functional']) else None
        base=dict(graph_key=graph_key,dataset=dataset,case=case,mode=mode,k=k,P=P,eps=eps,q=q,divergence=div,
                  lengths=';'.join(map(str,map(len,seqs))),dense_cells=int(math.prod(len(s)+1 for s in seqs)),
                  n_cell_anchors=ncellanchors,n_partition_anchors=npartanchors,band_functional=float(st['functional']),
                  functional_kind=('logZ_sink' if mode=='global' else 'local_max_logF'),
                  states=int(st['states']),accepted_edges=int(st['accepted_edges']),cross_edges=int(st['cross_edges']),
                  ccr_edge=float(st['ccr_edge']),cross_edge_fraction=float(st['cross_edge_fraction']),
                  geometry_sec=float(st['geometry_sec']),score_sec=float(st['score_sec']),stream_wall_sec=float(st['wall_sec']),
                  peak_wave_states=int(st['peak_wave_states']),peak_active_bytes=int(st['peak_active_bytes']),
                  waves_nonempty=int(st['waves_nonempty']),naive_wave_launches=int(st['naive_wave_launches']),
                  score_launches=int(st['score_launches']),launch_reduction_fraction=float(st['launch_reduction_fraction']),
                  small_bundle_launches=int(st['small_bundle_launches']),bundled_waves=int(st['bundled_waves']),
                  single_wave_launches=int(st['single_wave_launches']),max_bundle_size=int(st['max_bundle_size']),
                  tr_observations=int(coords_cp.shape[0]),tr_shift=shift,status_stream='OK',
                  exact_functional=(None if exact is None else float(exact['functional'])),
                  exact_states=(None if exact is None else int(exact['states'])),
                  band_functional_error=(None if exact is None else abs(float(st['functional'])-float(exact['functional']))))
        for r,key in zip(args.ranks,rank_keys):
            if key in J.done: continue
            try:
                tr=fit_tr_from_cupy(coords_cp,target_cp,tuple(len(s)+1 for s in seqs),rank=r,epochs=args.epochs,
                    batch_size=args.batch_size,lr=args.lr,weight_decay=args.weight_decay,val_fraction=args.val_fraction,
                    seed=seed_for(args.seed,key),functional_coord_cp=fcoord_cp,functional_target=fscaled,
                    functional_weight=args.functional_weight,dtype=args.tr_dtype,early_stop_patience=args.early_stop_patience)
                pred=float(tr['functional_pred_scaled'])
                trfunc=(shift+math.log(pred)) if pred>0 and math.isfinite(pred) else math.nan
                dfunc=abs(trfunc-float(st['functional'])) if math.isfinite(trfunc) and math.isfinite(float(st['functional'])) else math.inf
                row={**base,'key':key,'status':'OK','rank':r,'tr_params':int(tr['params']),
                     'tr_compression_vs_dense':base['dense_cells']/max(1,int(tr['params'])),
                     'tr_compression_vs_generated':base['states']/max(1,int(tr['params'])),
                     'tr_val_rel_error':float(tr['val_rel_error']),'tr_fit_sec':float(tr['fit_sec']),
                     'tr_epochs':int(tr['epochs_run']),'tr_functional':trfunc,'tr_functional_error':dfunc,
                     'tr_peak_torch_bytes':int(tr['peak_torch_bytes'])}
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception: pass
                    row={**base,'key':key,'status':'TR_OOM','rank':r,'error':str(e)[:300]}
                else: raise
            J.add(row); print('[saved]',key,row['status'],flush=True)
    except MemoryError as e:
        for r,key in zip(args.ranks,rank_keys):
            if key not in J.done:
                J.add(dict(key=key,graph_key=graph_key,status='RESOURCE_LIMIT_GPU',dataset=dataset,case=case,mode=mode,k=k,P=P,eps=eps,q=q,divergence=div,rank=r,error=str(e)))
        print('[resource-limit]',graph_key,e,flush=True)
    except cp.cuda.runtime.CUDARuntimeError as e:
        # Illegal-address errors poison the CUDA context; continuing in the same
        # process only produces a cascade of misleading deallocation failures.
        print('[fatal-cuda]',graph_key,e,flush=True)
        print('[fatal-cuda] checkpoint is intact; restart the same command to resume.',flush=True)
        raise
    finally:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception: pass


def exact_control(seqs,mode,args):
    dense=int(math.prod(len(s)+1 for s in seqs))
    if dense>args.max_full_states: return None
    import cupy as cp
    print(f'[exact-stream] {mode} dense-state count={dense:,} (wave streamed, no dense tensor)',flush=True)
    eng=GpuWaveStreamingForward(seqs,P=1,eps=None,beta=args.beta,mode=mode,block_size=args.block_size,
        small_wave_threshold=args.small_wave_threshold,small_bundle_max_waves=args.small_bundle_max_waves,
        reservoir_size=0,seed=args.seed,vram_fraction=args.vram_fraction,hard_vram_gb=args.hard_vram_gb,
        max_blocks_per_sm=args.max_blocks_per_sm,profile_waves=False)
    out=eng.run(); cp.get_default_memory_pool().free_all_blocks()
    return {'functional':float(out['stats']['functional']),'states':int(out['stats']['states']),
            'accepted_edges':int(out['stats']['accepted_edges']),'wall_sec':float(out['stats']['wall_sec'])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',default='synthetic-global,synthetic-local')
    ap.add_argument('--k',type=ints,default=ints('3,4,5,6'))
    ap.add_argument('--part-sizes',type=ints,default=ints('1,2,4'))
    ap.add_argument('--eps',type=ints,default=ints('0,1'))
    ap.add_argument('--q',type=ints,default=ints('3'))
    ap.add_argument('--divergence',type=floats,default=floats('0,0.2,0.4'))
    ap.add_argument('--ranks',type=ints,default=ints('2,4,8'))
    ap.add_argument('--trials',type=int,default=3); ap.add_argument('--global-L',type=int,default=12)
    ap.add_argument('--motif-len',type=int,default=9); ap.add_argument('--flank-len',type=int,default=12)
    ap.add_argument('--beta',type=float,default=0.1); ap.add_argument('--seed',type=int,default=12345)
    ap.add_argument('--block-size',type=int,default=256)
    ap.add_argument('--small-wave-threshold',type=int,default=1024,
                    help='Consecutive waves no wider than this are fused into a one-block multi-wave kernel.')
    ap.add_argument('--small-bundle-max-waves',type=int,default=16)
    ap.add_argument('--max-blocks-per-sm',type=int,default=8)
    ap.add_argument('--reservoir-size',type=int,default=1_000_000)
    ap.add_argument('--hard-vram-gb',type=float,default=11.5)
    ap.add_argument('--vram-fraction',type=float,default=0.72)
    ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--batch-size',type=int,default=65536)
    ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--weight-decay',type=float,default=0.0)
    ap.add_argument('--val-fraction',type=float,default=0.1); ap.add_argument('--functional-weight',type=float,default=20.0)
    ap.add_argument('--early-stop-patience',type=int,default=8); ap.add_argument('--tr-dtype',choices=['float32','float64'],default='float32')
    ap.add_argument('--max-anchors',type=int,default=8); ap.add_argument('--max-full-states',type=int,default=250000)
    ap.add_argument('--profile-waves',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--skip-self-test',action='store_true'); ap.add_argument('--out',default='experiments/e3_stream/main')
    args=ap.parse_args()
    if not cuda_available(): raise SystemExit('CuPy CUDA is required for the optimized streaming experiment.')
    if not torch_available(): raise SystemExit('PyTorch with CUDA is required for Tensor-Ring fitting.')
    protocol=vars(args).copy(); protocol['hardware']=gpu_info(); protocol['architecture']='gpu_wave_stream_no_dense_no_global_csr_v1'
    print('GPU:',protocol['hardware'],flush=True)
    self_test(args)
    J=Journal(args.out,protocol); signal.signal(signal.SIGINT,_sigint)
    datasets=[x.strip() for x in args.datasets.split(',') if x.strip()]
    for dataset in datasets:
        mode='global' if dataset=='synthetic-global' else 'local'
        for k in args.k:
            for div in args.divergence:
                for trial in range(args.trials):
                    rng=random.Random(seed_for(args.seed,dataset,k,div,trial))
                    seqs=(synthetic_global(k,args.global_L,div,rng) if mode=='global'
                          else synthetic_local(k,args.motif_len,args.flank_len,div,rng))
                    case=f'k{k}_d{div:g}_trial{trial}'
                    exact=exact_control(seqs,mode,args)
                    for P in args.part_sizes:
                        for eps in args.eps:
                            qs=[None] if mode=='global' else args.q
                            for q in qs:
                                run_one(J,dataset,case,seqs,mode,k,P,eps,q,div,args,exact=exact)
                                J.consolidate()
                                if STOP:
                                    print('[done] stopped safely; resume with identical command',flush=True); return
    J.consolidate(); print('[done] experiment complete',flush=True)

if __name__=='__main__': main()
