"""Seed discovery and local-band offset estimation."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from .sparse_dp import middle_partition


def shared_qmer_anchors(seqs, q=3, max_anchors=8, max_occurrences_per_qmer=8):
    tables=[]
    for s in seqs:
        d=defaultdict(list)
        for i in range(len(s)-q+1): d[s[i:i+q]].append(i+1)  # DP coordinate
        tables.append(d)
    shared=set(tables[0])
    for t in tables[1:]: shared &= set(t)
    cand=[]
    for qmer in shared:
        lists=[t[qmer][:max_occurrences_per_qmer] for t in tables]
        for a in product(*lists):
            # Translation magnitude/spread is not penalised: local motifs are
            # expected to occur at different offsets. Prefer occurrence tuples
            # whose implied translated diagonal has small residual after centring.
            w=sum(a); mp=middle_partition(w,len(seqs)); delta=tuple(a[i]-mp[i] for i in range(len(seqs)))
            score=max(delta)-min(delta)
            cand.append((score,qmer,tuple(a),delta))
    cand.sort(key=lambda x:(x[0],x[1],x[2]))
    # de-duplicate offsets so repetitive q-mers do not consume all candidates.
    out=[]; seen=set()
    for x in cand:
        if x[3] in seen: continue
        seen.add(x[3]); out.append(x)
        if len(out)>=max_anchors: break
    return out


def oracle_offset(starts):
    """DP-coordinate motif starts -> translated-diagonal delta."""
    a=tuple(int(x)+1 for x in starts)
    mp=middle_partition(sum(a),len(a))
    return tuple(a[i]-mp[i] for i in range(len(a)))
