#!/usr/bin/env python3
"""CPU reference regression tests for restored mmrepro modules."""
import sys, pathlib, math
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from mmrepro.sparse_dp import build_sparse_graph,run_cpu,MATCH
from mmrepro.sparse_graph_fast import build_sparse_graph_fast
from mmrepro.seeds import shared_qmer_anchors
from mmrepro.datasets import read_fasta,read_scanner_hits
for seqs in (['AC','AC','AC'],['AC','AG','AT'],['ACG','ACG']):
 for band in (None,0,1):
  a=build_sparse_graph(seqs,band,1); b=build_sparse_graph_fast(seqs,band,1)
  assert a.n_states==b.n_states and a.n_edges==b.n_edges,(seqs,band,a.n_states,b.n_states,a.n_edges,b.n_edges)
  for mode in ('global','local'):
   x=run_cpu(a,seqs,mode=mode,want_traceback=False)['score']
   y=run_cpu(b,seqs,mode=mode,want_traceback=False)['score']
   assert (math.isinf(x) and math.isinf(y)) or abs(x-y)<1e-9,(seqs,band,mode,x,y)
assert shared_qmer_anchors(['ACGTAC','TACGTA','AACGTT'],q=3)
print('PASS: slow/fast sparse graph and CPU max-plus reference agreement; q-mer anchors')
