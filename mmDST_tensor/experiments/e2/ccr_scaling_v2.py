#!/usr/bin/env python3
"""Checkpointable/resumable CCR sweep with fast graph builder and CUDA Graphs.

Safe interruption:
  * one JSON record is fsync'd after each completed (dataset,case,P,eps,q)
    configuration;
  * Ctrl-C consolidates all completed records to CSV before exiting;
  * --resume (default) skips completed configurations;
  * --fresh refuses to reuse the checkpoint and starts a new one.

Adaptive mode:
  epsilon expands in ascending order and stops for a fixed (case,P,q) at the
  first score-preserving configuration.  Only that necessary prefix is built.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, signal, statistics, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from mmrepro.sparse_dp import MATCH, run_cpu
from mmrepro.sparse_graph_fast import build_sparse_graph_fast
from mmrepro.seeds import shared_qmer_anchors
from mmrepro.cuda_sparse_dp_ccr_v2 import cuda_available, benchmark_resident_v2
from mmrepro.datasets import read_fasta, read_scanner_hits

DNA='ACGT'; LOCAL_FLANK_SYMBOLS=list('EFHIKLMNPQRSTVWY')
STOP_REQUESTED=False


def _sigint(signum, frame):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    STOP_REQUESTED=True
    print('\n[checkpoint] stop requested: finishing current configuration, then exiting safely...', flush=True)


def choose2(k): return k*(k-1)//2

def synthetic_global(k,L,rng):
    s=''.join(rng.choice(DNA) for _ in range(L))
    return [s]*k, MATCH*L*choose2(k), {'kind':'certified_identical_global'}

def synthetic_local(k,motif_len,flank_len,rng):
    if k>len(LOCAL_FLANK_SYMBOLS): raise ValueError('k too high for local construction')
    motif=''.join(rng.choice(DNA) for _ in range(motif_len)); seqs=[]
    for i in range(k):
        left=rng.randrange(flank_len+1); ch=LOCAL_FLANK_SYMBOLS[i]
        seqs.append(ch*left+motif+ch*(flank_len-left))
    return seqs, MATCH*motif_len*choose2(k), {'kind':'constructed_local_target'}

def symmetric_round_div(x,d):
    return int(math.floor(x/d+0.5)) if x>=0 else -int(math.floor((-x)/d+0.5))

def qmer_partition_deltas(seqs,q,P,max_anchors=8):
    a=shared_qmer_anchors(seqs,q=q,max_anchors=max_anchors)
    out=[]; seen=set()
    for x in a:
        d=tuple(symmetric_round_div(int(v),P) for v in x[3])
        if d not in seen: seen.add(d); out.append(d)
    return out,len(a),len(out)

def graph_bytes(g):
    return g.coords.nbytes+g.wave_offsets.nbytes+g.edge_ptr.nbytes+g.edge_src.nbytes+g.edge_w.nbytes


def ccr_metrics_fast(g,P,detailed_halo=False):
    """Low-overhead CCR metrics.

    Default uses cross-partition predecessor edges as communication units.  The
    optional detailed halo metric additionally deduplicates (destination
    partition, source state) and is therefore more memory-intensive.
    """
    n=g.n_states; deg=np.diff(g.edge_ptr)
    if n==0 or g.n_edges==0:
        return dict(n_partitions=0,cross_edges=0,internal_edges=0,ccr_edge=math.inf,
                    cross_edge_fraction=0.0,halo_states='',ccr_cells_halo='',ccr_edges_halo='')
    pcoords=(g.coords//int(P)).astype(np.int64,copy=False)
    # compact partition id via np.unique(axis=0)
    _, pinv=np.unique(pcoords,axis=0,return_inverse=True)
    dst=np.repeat(np.arange(n,dtype=np.int64),deg)
    cross=(pinv[dst] != pinv[g.edge_src])
    ncross=int(np.count_nonzero(cross)); internal=int(g.n_edges-ncross)
    out=dict(n_partitions=int(pinv.max()+1),cross_edges=ncross,internal_edges=internal,
             ccr_edge=(g.n_edges/ncross if ncross else math.inf),
             cross_edge_fraction=(ncross/g.n_edges if g.n_edges else 0.0),
             halo_states='',ccr_cells_halo='',ccr_edges_halo='')
    if detailed_halo and ncross:
        dc=dst[cross]; sc=g.edge_src[cross]
        # Unique source value per destination partition: exact algorithmic halo.
        pair=pinv[dc].astype(np.uint64)*np.uint64(max(1,n))+sc.astype(np.uint64)
        halo=int(np.unique(pair).size)
        out.update(halo_states=halo,ccr_cells_halo=n/halo,ccr_edges_halo=g.n_edges/halo)
    return out


def exact_reference_if_feasible(seqs,mode,max_full_states):
    dense=math.prod(len(s)+1 for s in seqs)
    if dense>max_full_states: return None
    t=time.perf_counter(); g=build_sparse_graph_fast(seqs,None)
    r=run_cpu(g,seqs,mode=mode,semiring='maxplus',want_traceback=False)
    return dict(score=float(r['score']),reference_kind='unrestricted_sparse_dp',reference_certified=1,
                reference_cells=g.n_states,reference_sec=time.perf_counter()-t)


def config_key(dataset,case,mode,k,P,eps,q):
    return '|'.join(map(str,(dataset,case,mode,k,P,eps,'' if q is None else q)))

class Journal:
    def __init__(self,out,protocol,resume=True,fresh=False):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.path=self.out/'ccr_checkpoint.jsonl'; self.meta=self.out/'protocol.json'
        self.protocol=protocol
        self.fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
        if fresh and self.path.exists():
            stamp=time.strftime('%Y%m%d_%H%M%S'); self.path.rename(self.out/f'ccr_checkpoint_{stamp}.jsonl')
        if self.meta.exists() and self.path.exists() and resume:
            old=json.loads(self.meta.read_text())
            if old.get('fingerprint') != self.fingerprint:
                raise SystemExit('Existing checkpoint protocol differs. Use a new --out or --fresh.')
        self.meta.write_text(json.dumps({'fingerprint':self.fingerprint,'protocol':protocol},indent=2))
        self.done=set(); self.records=[]
        if resume and self.path.exists(): self._load()
    def _load(self):
        with self.path.open() as f:
            for line in f:
                try:
                    rec=json.loads(line); self.records.append(rec); self.done.add(rec['config_key'])
                except json.JSONDecodeError:
                    print('[checkpoint] ignored incomplete final JSONL line')
        print(f'[checkpoint] resumed {len(self.done)} completed configurations',flush=True)
    def has(self,key): return key in self.done
    def append(self,rec):
        rec=dict(rec); rec['protocol_fingerprint']=self.fingerprint
        line=json.dumps(rec,separators=(',',':'),allow_nan=True)
        with self.path.open('a') as f:
            f.write(line+'\n'); f.flush(); os.fsync(f.fileno())
        self.records.append(rec); self.done.add(rec['config_key'])
    def csv(self):
        rows=[]
        for rec in self.records:
            base={k:v for k,v in rec.items() if k!='gpu_rows'}
            gr=rec.get('gpu_rows') or [{}]
            for x in gr:
                y=dict(base); y.update(x); rows.append(y)
        if not rows: return
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
        tmp=self.out/'ccr_raw.csv.tmp'; final=self.out/'ccr_raw.csv'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        os.replace(tmp,final)
        print('[checkpoint] wrote',final,flush=True)


def run_family(J,dataset,case,seqs,mode,reference,k,Pvals,Evals,Qvals,Bvals,args):
    global STOP_REQUESTED
    ref=None if reference is None else float(reference['score'])
    for P in Pvals:
        qs=Qvals if mode=='local' else [None]
        for q in qs:
            # Resume adaptive state: if an earlier completed eps recovered, skip
            # the rest of this branch immediately.
            recovered_already=False
            for eps in Evals:
                key=config_key(dataset,case,mode,k,P,eps,q)
                if J.has(key):
                    old=next((r for r in reversed(J.records) if r['config_key']==key),None)
                    if args.adaptive and old and int(old.get('recovered',0) or 0)==1:
                        recovered_already=True; break
                    continue
                if recovered_already: break
                if STOP_REQUESTED: return

                if mode=='local' and q is not None:
                    deltas,na,np_=qmer_partition_deltas(seqs,q,P)
                    if not deltas: deltas=[tuple([0]*k)]
                else: deltas=None; na=np_=0

                print(f'[build] {dataset} {case} k={k} P={P} eps={eps} q={q}',flush=True)
                t=time.perf_counter(); g=build_sparse_graph_fast(seqs,eps,P,deltas); build=time.perf_counter()-t

                # GPU is the adaptive score engine. It avoids a duplicate CPU DP
                # pass for every candidate. CPU timing can be requested separately.
                gpu_rows=[]; measured_score=None
                if cuda_available():
                    gpu_rows=benchmark_resident_v2(g,mode=mode,block_sizes=Bvals,
                        repeats=args.gpu_repeats,warmup=args.gpu_warmup,
                        use_cuda_graph=(not args.no_cuda_graph),autotune=args.autotune_block)
                    if gpu_rows: measured_score=float(gpu_rows[0]['score'])
                else:
                    rr=run_cpu(g,seqs,mode=mode,semiring='maxplus',want_traceback=False); measured_score=float(rr['score'])

                cpu_sec=''; cpu_equal=''
                if args.cpu_timing:
                    t=time.perf_counter(); rr=run_cpu(g,seqs,mode=mode,semiring='maxplus',want_traceback=False)
                    cpu_sec=time.perf_counter()-t; cpu_equal=int(abs(float(rr['score'])-float(measured_score))<1e-9)

                recovered=(ref is not None and measured_score is not None and abs(measured_score-ref)<1e-9)
                ccr=ccr_metrics_fast(g,P,detailed_halo=args.detailed_halo)
                rec=dict(config_key=key,dataset=dataset,case=case,mode=mode,k=k,
                    lengths=';'.join(map(str,map(len,seqs))),dense_cells=g.dense_cells,
                    part_size=P,eps=eps,q=('' if q is None else q),effective_cell_halfwidth=P*eps,
                    n_cell_anchors=na,n_partition_anchors=np_,score=measured_score,
                    reference_score=('' if ref is None else ref),recovered=('' if ref is None else int(recovered)),
                    reference_kind=('' if reference is None else reference['reference_kind']),
                    reference_certified=('' if reference is None else reference['reference_certified']),
                    cells=g.n_states,cell_fraction=g.n_states/g.dense_cells,edges=g.n_edges,
                    graph_bytes=graph_bytes(g),graph_build_sec=build,cpu_sec=cpu_sec,cpu_equal_gpu=cpu_equal,
                    **ccr,gpu_rows=gpu_rows)
                J.append(rec)
                print(f'[saved] cells={g.n_states} edges={g.n_edges} build={build:.3f}s score={measured_score} recovered={recovered} cpu_equal={cpu_equal}',flush=True)
                if args.adaptive and recovered: break
    

def load_balibase_cases(root,K,L,ncase):
    files=sorted(Path(root).rglob('*.tfa')); out=[]
    for k in K:
        made=0
        for p in files:
            d=read_fasta(p); names=list(d)
            if len(names)>=k:
                out.append((f'{p.stem}_k{k}',[d[x][:L] for x in names[:k]],k)); made+=1
                if made>=ncase: break
    return out

def load_scanner_cases(hit_dir,fasta,K,flank,ncase,seed):
    hits=read_scanner_hits(hit_dir,positive_strand_only=True); seqdb=read_fasta(fasta); by=defaultdict(list)
    for h in hits:
        if h['sequence_name'] in seqdb: by[h['motif_id']].append(h)
    rng=random.Random(seed+123); out=[]
    for k in K:
        made=0
        for mid in sorted(by):
            names=sorted({h['sequence_name'] for h in by[mid]})
            if len(names)<k: continue
            rng.shuffle(names); seqs=[]
            for name in names[:k]:
                h=rng.choice([x for x in by[mid] if x['sequence_name']==name]); s=seqdb[name]
                lo=max(0,int(h['start'])-flank); hi=min(len(s),int(h['stop'])+flank); seqs.append(s[lo:hi])
            out.append((f'{mid}_k{k}_{made}',seqs,k)); made+=1
            if made>=ncase: break
    return out

def ints(s): return [int(x) for x in s.split(',') if x.strip()]


def main():
    signal.signal(signal.SIGINT,_sigint)
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('experiments/ccr/results_v2'))
    ap.add_argument('--datasets',default='synthetic-global,synthetic-local')
    ap.add_argument('--k',default='3,4,5,6,7,8'); ap.add_argument('--part-sizes',default='1,2,4,8')
    ap.add_argument('--eps',default='0,1,2,3,4,5'); ap.add_argument('--q',default='2,3,4')
    ap.add_argument('--block-sizes',default='128,256,512'); ap.add_argument('--trials',type=int,default=5)
    ap.add_argument('--seed',type=int,default=12345); ap.add_argument('--global-L',type=int,default=12)
    ap.add_argument('--local-motif-len',type=int,default=9); ap.add_argument('--local-flank-len',type=int,default=12)
    ap.add_argument('--max-full-states',type=int,default=250000)
    ap.add_argument('--gpu-repeats',type=int,default=10); ap.add_argument('--gpu-warmup',type=int,default=2)
    ap.add_argument('--adaptive',action='store_true'); ap.add_argument('--autotune-block',action='store_true')
    ap.add_argument('--no-cuda-graph',action='store_true'); ap.add_argument('--cpu-timing',action='store_true')
    ap.add_argument('--detailed-halo',action='store_true')
    ap.add_argument('--fresh',action='store_true'); ap.add_argument('--no-resume',action='store_true')
    ap.add_argument('--balibase'); ap.add_argument('--balibase-L',type=int,default=12); ap.add_argument('--real-cases',type=int,default=5)
    ap.add_argument('--scanner-hits'); ap.add_argument('--jaspar-fasta'); ap.add_argument('--scanner-flank',type=int,default=10)
    args=ap.parse_args()
    K=ints(args.k); P=ints(args.part_sizes); E=sorted(ints(args.eps)); Q=ints(args.q); B=ints(args.block_sizes)
    datasets=sorted(x.strip() for x in args.datasets.split(',') if x.strip())
    protocol=dict(datasets=datasets,k=K,part_sizes=P,eps=E,q=Q,block_sizes=B,trials=args.trials,seed=args.seed,
                  global_L=args.global_L,local_motif_len=args.local_motif_len,local_flank_len=args.local_flank_len,
                  max_full_states=args.max_full_states,gpu_repeats=args.gpu_repeats,gpu_warmup=args.gpu_warmup,
                  adaptive=args.adaptive,autotune_block=args.autotune_block,cuda_graph=not args.no_cuda_graph,
                  cpu_timing=args.cpu_timing,detailed_halo=args.detailed_halo)
    J=Journal(args.out,protocol,resume=not args.no_resume,fresh=args.fresh)
    rng=random.Random(args.seed)
    print('CUDA:',cuda_available(),'adaptive:',args.adaptive,'checkpoint:',J.path,flush=True)
    try:
        if 'synthetic-global' in datasets:
            for k in K:
                for tr in range(args.trials):
                    seqs,known,meta=synthetic_global(k,args.global_L,rng)
                    ref=dict(score=known,reference_kind=meta['kind'],reference_certified=1)
                    run_family(J,'synthetic_global',f'k{k}_trial{tr}',seqs,'global',ref,k,P,E,Q,B,args)
                    if STOP_REQUESTED: break
                if STOP_REQUESTED: break
        if not STOP_REQUESTED and 'synthetic-local' in datasets:
            for k in K:
                for tr in range(args.trials):
                    seqs,target,meta=synthetic_local(k,args.local_motif_len,args.local_flank_len,rng)
                    exact=exact_reference_if_feasible(seqs,'local',args.max_full_states)
                    ref=exact or dict(score=target,reference_kind=meta['kind'],reference_certified=0)
                    run_family(J,'synthetic_local',f'k{k}_trial{tr}',seqs,'local',ref,k,P,E,Q,B,args)
                    if STOP_REQUESTED: break
                if STOP_REQUESTED: break
        if not STOP_REQUESTED and 'balibase' in datasets:
            if not args.balibase: raise SystemExit('--balibase required')
            for case,seqs,k in load_balibase_cases(args.balibase,K,args.balibase_L,args.real_cases):
                ref=exact_reference_if_feasible(seqs,'global',args.max_full_states)
                run_family(J,'balibase',case,seqs,'global',ref,k,P,E,Q,B,args)
                if STOP_REQUESTED: break
        if not STOP_REQUESTED and 'scanner' in datasets:
            if not args.scanner_hits or not args.jaspar_fasta: raise SystemExit('--scanner-hits and --jaspar-fasta required')
            for case,seqs,k in load_scanner_cases(args.scanner_hits,args.jaspar_fasta,K,args.scanner_flank,args.real_cases,args.seed):
                ref=exact_reference_if_feasible(seqs,'local',args.max_full_states)
                run_family(J,'scanner',case,seqs,'local',ref,k,P,E,Q,B,args)
                if STOP_REQUESTED: break
    finally:
        J.csv()
    print('[done] completed configs:',len(J.done),'safe to resume with identical command',flush=True)

if __name__=='__main__': main()
