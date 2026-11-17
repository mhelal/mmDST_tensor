#!/usr/bin/env python3
"""
build_paper_assets.py -- scan the experiment CSVs and emit paper-ready tables
(LaTeX) and figures (PNG), choosing the framings with the highest honest impact.

"High impact" here means: lead with the finding that is both true and
surprising, and never with a number that is an artifact. Concretely:

  * Headline table = value-vs-partition contrast on realistic sequences
    (the semiring dichotomy), NOT the raw compression factor.
  * Headline improvement = the beta lever (400x error reduction), reported as a
    curve so the trade-off is visible.
  * Divergence flatness = the alignment-scaling result that replaces the
    misleading tensor-size plot.
  * Band = the exactness/cost Pareto point (eps=1: 100% score, 2.2% cells).

Run after alignment_scaling.py has produced results_alignment/*.csv.
"""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results_alignment"
OUT = Path(__file__).parent / "paper_assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif", "savefig.dpi": 200})


def read(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- TABLES
def table_value_vs_partition():
    rows = read("value_vs_partition.csv")
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$k$ & value opt.\ err & value rel.\ err & partition $\Delta\log Z$ & part.\ comp \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['k']} & {float(r['value_opt_error']):.1f} & "
                     f"{float(r['value_relerr']):.3f} & "
                     f"{float(r['partition_logZ_error']):.1e} & "
                     f"{float(r['partition_comp']):.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_value_vs_partition.tex").write_text("\n".join(lines))
    print("wrote table_value_vs_partition.tex  (the semiring dichotomy -- headline)")


def table_divergence():
    rows = read("divergence.csv")
    lines = [r"\begin{tabular}{rrr}", r"\toprule",
             r"divergence & rel.\ error & $\Delta\log Z$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{float(r['divergence'])*100:.0f}\\% & "
                     f"{float(r['rel_error']):.4f} & "
                     f"{float(r['logZ_error']):.1e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_divergence.tex").write_text("\n".join(lines))
    print("wrote table_divergence.tex  (alignment scaling replaces tensor scaling)")


# ---------------------------------------------------------------- FIGURES
def fig_beta_lever():
    rows = read("beta_sweep.csv")
    beta = [float(r["beta"]) for r in rows]
    err = [float(r["rel_error"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(beta, err, "o-", color="#4A6FA5", lw=2)
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel("relative reconstruction error")
    ax.set_title(r"Reconstruction error is tunable via $\beta$")
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate(r"$400\times$ improvement", xy=(beta[0], err[0]),
                xytext=(beta[1], err[0]*3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#B0413E"))
    fig.tight_layout(); fig.savefig(OUT / "fig_beta_lever.png")
    print("wrote fig_beta_lever.png  (the honest improvement -- headline lever)")


def fig_rank_frontier():
    rows = read("rank_budget.csv")
    comp = [float(r["compression"]) for r in rows]
    err = [float(r["rel_error"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogy(comp, err, "s-", color="#2E7D8A", lw=2)
    for r in rows:
        ax.annotate(f"r={r['rank']}", (float(r['compression']), float(r['rel_error'])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("compression factor"); ax.set_ylabel("relative error")
    ax.set_title("Accuracy / compression frontier (partition tensor)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_rank_frontier.png")
    print("wrote fig_rank_frontier.png  (honest trade-off curve)")


def fig_dichotomy():
    rows = read("value_vs_partition.csv")
    ks = [r["k"] for r in rows]
    vopt = [float(r["value_opt_error"]) for r in rows]
    plz = [float(r["partition_logZ_error"]) for r in rows]
    x = np.arange(len(ks)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w/2, vopt, w, label="value: optimum error", color="#B0413E")
    ax.bar(x + w/2, plz, w, label=r"partition: $\Delta\log Z$", color="#2E7D8A")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("error (symlog)")
    ax.set_title("Semiring dichotomy on realistic sequences")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_dichotomy.png")
    print("wrote fig_dichotomy.png  (value breaks, partition holds -- the core result)")


def main():
    table_value_vs_partition()
    table_divergence()
    fig_beta_lever()
    fig_rank_frontier()
    fig_dichotomy()
    print(f"\nAll assets in {OUT}/. Recommended paper lead: dichotomy table/figure,"
          f"\nthen beta lever as the improvement, then divergence flatness,"
          f"\nthen band Pareto point. Retire raw compression-factor plots to a"
          f"\nsize-description role only.")


if __name__ == "__main__":
    main()
