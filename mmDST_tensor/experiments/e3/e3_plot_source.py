#!/usr/bin/env python3
"""Reproducible plots and summary tables for final Experiment 3 (E3).

Inputs
------
  e3_stream_raw(1).csv
  e3_stream_waves(1).csv
  e3_stream_protocol(1).json

The script intentionally derives every plotted quantity from the final CSVs.
It does not depend on notebook state and is suitable for repository release.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def savefig(fig, outbase: Path):
    fig.tight_layout()
    fig.savefig(outbase.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(outbase.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def load(raw_path, waves_path):
    raw = pd.read_csv(raw_path)
    waves = pd.read_csv(waves_path)
    graph = raw.drop_duplicates('graph_key').copy()
    raw['tr_finite'] = np.isfinite(raw['tr_functional_error'].to_numpy())
    return raw, graph, waves


def write_tables(raw, graph, out):
    out.mkdir(parents=True, exist_ok=True)
    km = graph.groupby(['mode','k']).agg(
        n_graphs=('graph_key','size'),
        median_states=('states','median'), max_states=('states','max'),
        median_edges=('accepted_edges','median'), max_edges=('accepted_edges','max'),
        median_wall_sec=('stream_wall_sec','median'),
        median_geometry_sec=('geometry_sec','median'), median_score_sec=('score_sec','median'),
        max_peak_wave_states=('peak_wave_states','max'),
        median_launch_reduction=('launch_reduction_fraction','median'),
    ).reset_index()
    km.to_csv(out/'e3_summary_by_mode_k.csv', index=False)

    pke = graph.groupby(['mode','k','P','eps']).agg(
        n_graphs=('graph_key','size'),
        median_states=('states','median'), max_states=('states','max'),
        median_edges=('accepted_edges','median'),
        median_ccr=('ccr_edge','median'), median_cross_fraction=('cross_edge_fraction','median'),
        median_wall_sec=('stream_wall_sec','median'),
        median_launch_reduction=('launch_reduction_fraction','median'),
    ).reset_index()
    pke.to_csv(out/'e3_summary_by_mode_k_P_eps.csv', index=False)

    tr = raw.groupby('rank').agg(
        n=('key','size'), finite=('tr_finite','sum'),
        median_val_rel_error=('tr_val_rel_error','median'),
        median_fit_sec=('tr_fit_sec','median'),
        median_params=('tr_params','median'),
        median_compression_dense=('tr_compression_vs_dense','median'),
        median_compression_generated=('tr_compression_vs_generated','median'),
    ).reset_index()
    tr['finite_rate'] = tr['finite']/tr['n']
    tr['median_functional_error_finite'] = [
        np.median(raw.loc[(raw['rank']==r)&raw['tr_finite'],'tr_functional_error']) for r in tr['rank']
    ]
    for t in (0.1,1.0):
        tr[f'fraction_functional_error_lt_{t}'] = [
            np.mean(np.isfinite(s) & (s<t)) for _, s in raw.groupby('rank')['tr_functional_error']
        ]
    tr.to_csv(out/'e3_tr_summary_by_rank.csv', index=False)

    exact = graph[np.isfinite(graph['exact_functional'])].groupby(['mode','P','eps']).agg(
        n=('graph_key','size'), median_band_functional_error=('band_functional_error','median'),
        mean_band_functional_error=('band_functional_error','mean'),
    ).reset_index()
    exact.to_csv(out/'e3_exact_control_band_error.csv', index=False)


def plot_state_scaling(graph, out):
    # Median states by k for each P/eps, separate mode panels.
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax, mode in zip(axes,['global','local']):
        s = graph[graph['mode']==mode].groupby(['k','P','eps'])['states'].median().reset_index()
        for (P,eps), d in s.groupby(['P','eps']):
            ax.plot(d['k'], d['states'], marker='o', label=f'P={P}, eps={eps}')
        ax.set_yscale('log'); ax.set_xlabel('Number of sequences k'); ax.set_title(mode.capitalize())
        ax.grid(True, which='both', alpha=.25)
    axes[0].set_ylabel('Median generated states (log scale)')
    axes[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('E3 sparse-state growth: partition size and band width')
    savefig(fig,out/'fig_e3_state_scaling')


def plot_ccr_tradeoff(graph,out):
    # CCR vs retained states; medians over divergence/trial for each k/P/eps.
    s=graph.groupby(['mode','k','P','eps']).agg(states=('states','median'),ccr=('ccr_edge','median')).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=False,sharey=False)
    for ax,mode in zip(axes,['global','local']):
        d=s[s['mode']==mode]
        for (k,eps), q in d.groupby(['k','eps']):
            ax.plot(q['states'],q['ccr'],marker='o',label=f'k={k}, eps={eps}')
            for _,r in q.iterrows(): ax.annotate(f"P{int(r.P)}",(r.states,r.ccr),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set_xscale('log'); ax.set_xlabel('Median generated states (log scale)'); ax.set_ylabel('Median edge CCR')
        ax.set_title(mode.capitalize()); ax.grid(True,which='both',alpha=.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle('E3 CCR improves with P while retained support grows')
    savefig(fig,out/'fig_e3_ccr_state_tradeoff')


def plot_exact_error(graph,out):
    e=graph[np.isfinite(graph['exact_functional'])].groupby(['mode','P','eps'])['band_functional_error'].median().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,mode in zip(axes,['global','local']):
        d=e[e['mode']==mode]
        for eps,q in d.groupby('eps'):
            ax.plot(q['P'],q['band_functional_error'],marker='o',label=f'eps={eps}')
        ax.set_yscale('log'); ax.set_xticks([1,2,4]); ax.set_xlabel('Partition size P'); ax.set_title(mode.capitalize())
        ax.grid(True,which='both',alpha=.25)
    axes[0].set_ylabel('Median |band functional - exact| (log scale)')
    axes[1].legend()
    fig.suptitle('Exact streamed controls (k=3,4): support expansion improves functional fidelity')
    savefig(fig,out/'fig_e3_exact_band_error')


def plot_launch(graph,out):
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    for mode,d in graph.groupby('mode'):
        ax.scatter(d['peak_wave_states'],100*d['launch_reduction_fraction'],s=20,alpha=.6,label=mode)
    ax.axvline(1024,linestyle='--',linewidth=1,label='small-wave threshold = 1024')
    ax.set_xscale('log'); ax.set_xlabel('Peak wave states (log scale)'); ax.set_ylabel('Launch reduction (%)')
    ax.set_title('Adaptive scheduling: fusion dominates narrow waves, wide waves remain parallel kernels')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    savefig(fig,out/'fig_e3_launch_reduction')


def plot_tr_rank(raw,out):
    ranks=sorted(raw['rank'].unique())
    finite_med=[]; finite_rate=[]; lt1=[]; val=[]
    for r in ranks:
        d=raw[raw['rank']==r]
        f=d[np.isfinite(d['tr_functional_error'])]
        finite_med.append(f['tr_functional_error'].median())
        finite_rate.append(100*np.mean(np.isfinite(d['tr_functional_error'])))
        lt1.append(100*np.mean(np.isfinite(d['tr_functional_error']) & (d['tr_functional_error']<1)))
        val.append(d['tr_val_rel_error'].median())
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    ax.plot(ranks,finite_med,marker='o',label='Median functional error (finite)')
    ax.plot(ranks,val,marker='s',label='Median validation relative error')
    ax.set_yscale('log'); ax.set_xticks(ranks); ax.set_xlabel('TR rank'); ax.set_ylabel('Error (log scale)')
    ax.set_title('Tensor Ring fidelity improves strongly with rank in aggregate')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    savefig(fig,out/'fig_e3_tr_rank_error')

    fig,ax=plt.subplots(figsize=(6.5,4.3))
    x=np.arange(len(ranks)); w=.35
    ax.bar(x-w/2,finite_rate,w,label='Finite functional')
    ax.bar(x+w/2,lt1,w,label='Functional error < 1')
    ax.set_xticks(x,ranks); ax.set_ylim(0,105); ax.set_xlabel('TR rank'); ax.set_ylabel('Configurations (%)')
    ax.set_title('Tensor Ring functional reliability by rank'); ax.grid(True,axis='y',alpha=.25); ax.legend()
    savefig(fig,out/'fig_e3_tr_finite_rate')


def plot_divergence(raw,out):
    fig,ax=plt.subplots(figsize=(7,4.4))
    for r,d in raw.groupby('rank'):
        vals=[]
        for div in sorted(raw['divergence'].unique()):
            x=d[(d['divergence']==div)&np.isfinite(d['tr_functional_error'])]['tr_functional_error']
            vals.append(x.median())
        ax.plot(sorted(raw['divergence'].unique()),vals,marker='o',label=f'rank {r}')
    ax.set_yscale('log'); ax.set_xlabel('Synthetic divergence'); ax.set_ylabel('Median finite TR functional error')
    ax.set_title('Divergence effect is weaker and less monotonic than rank effect')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    savefig(fig,out/'fig_e3_divergence_rank')


def plot_timing(graph,out):
    fig,ax=plt.subplots(figsize=(7,4.5))
    for mode,d in graph.groupby('mode'):
        ax.scatter(d['accepted_edges'],d['score_sec'],s=20,alpha=.6,label=mode)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Accepted predecessor evaluations'); ax.set_ylabel('Scoring time (s)')
    ax.set_title('GPU recurrence time scales with evaluated dependencies')
    ax.grid(True,which='both',alpha=.25); ax.legend()
    savefig(fig,out/'fig_e3_score_vs_edges')


def plot_wave_examples(graph,waves,out):
    # Choose the largest graph and the final completed graph for two intuitive profiles.
    keys=[graph.loc[graph['states'].idxmax(),'graph_key'], 'synthetic-local|k6_d0.4_trial2|local|k6|P4|e1|q3|d0.4']
    for idx,key in enumerate(keys,1):
        d=waves[waves['graph_key']==key].sort_values('wave')
        if d.empty: continue
        fig,ax=plt.subplots(figsize=(9,4.6))
        ax.plot(d['wave'],d['states'],linewidth=1.5,label='states in wave')
        ax.axhline(1024,linestyle='--',linewidth=1,label='small-wave threshold')
        ax.set_yscale('log'); ax.set_xlabel('Wave index'); ax.set_ylabel('States (log scale)')
        ax.set_title('Wavefront width profile\n'+key)
        ax.grid(True,which='both',alpha=.25); ax.legend()
        savefig(fig,out/f'fig_e3_wave_profile_{idx}')

        fig,ax=plt.subplots(figsize=(9,4.6))
        ax.plot(d['wave'],d['kernel_sec'],linewidth=1.3)
        ax.set_yscale('log'); ax.set_xlabel('Wave index'); ax.set_ylabel('Kernel time (s, log scale)')
        ax.set_title('Per-wave GPU kernel timing\n'+key)
        ax.grid(True,which='both',alpha=.25)
        savefig(fig,out/f'fig_e3_wave_kernel_{idx}')


def plot_compression(raw,out):
    s=raw.groupby(['k','rank']).agg(dense=('tr_compression_vs_dense','median'),generated=('tr_compression_vs_generated','median')).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True)
    for r,d in s.groupby('rank'):
        axes[0].plot(d['k'],d['dense'],marker='o',label=f'rank {r}')
        axes[1].plot(d['k'],d['generated'],marker='o',label=f'rank {r}')
    for ax,title in zip(axes,['Relative to dense Cartesian field','Relative to generated sparse field']):
        ax.set_yscale('log'); ax.set_xlabel('k'); ax.set_ylabel('Median compression ratio'); ax.set_title(title); ax.grid(True,which='both',alpha=.25)
    axes[1].legend()
    savefig(fig,out/'fig_e3_tr_compression_baselines')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',default='experiments/e3/results/e3_stream_raw.csv')
    ap.add_argument('--waves',default='experiments/e3/results/e3_stream_waves.csv')
    ap.add_argument('--out',default='experiments/figures')
    ap.add_argument('--tables',default='experiments/e3/tables')
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    raw,graph,waves=load(a.raw,a.waves)
    write_tables(raw,graph,Path(a.tables))
    plot_state_scaling(graph,out)
    plot_ccr_tradeoff(graph,out)
    plot_exact_error(graph,out)
    plot_launch(graph,out)
    plot_tr_rank(raw,out)
    plot_divergence(raw,out)
    plot_timing(graph,out)
    plot_wave_examples(graph,waves,out)
    plot_compression(raw,out)
    print(f'Generated figures in {out} and tables in {a.tables}')

if __name__=='__main__':
    main()
