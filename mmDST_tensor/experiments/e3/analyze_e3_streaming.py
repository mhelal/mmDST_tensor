#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv',nargs='?',default='experiments/e3_stream/main/e3_stream_raw.csv')
    ap.add_argument('--out',default=None)
    a=ap.parse_args(); p=Path(a.csv); df=pd.read_csv(p)
    ok=df[df['status'].eq('OK')].copy()
    if ok.empty:
        print('No OK rows'); return
    grp=['dataset','mode','k','P','eps','q','divergence','rank']
    cols=[c for c in ['band_functional_error','states','accepted_edges','ccr_edge','geometry_sec','score_sec','stream_wall_sec',
                       'peak_active_bytes','launch_reduction_fraction','tr_compression_vs_dense','tr_val_rel_error','tr_functional_error','tr_fit_sec'] if c in ok]
    summary=ok.groupby(grp,dropna=False)[cols].median(numeric_only=True).reset_index()
    out=Path(a.out) if a.out else p.with_name('e3_stream_summary.csv'); summary.to_csv(out,index=False)
    print(summary.to_string(index=False,max_rows=200)); print('\nWrote',out)

if __name__=='__main__': main()
