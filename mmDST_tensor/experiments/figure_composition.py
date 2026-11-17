#!/usr/bin/env python3
"""
figure_composition.py -- conceptual illustration of the paper's key architectural
argument, as a two-panel figure.

Panel A: "Decomposition alone hits the wall." To decompose the tensor you must
first FORM it -- an (L+1)^k array -- and each DP entry's dependency cone reaches
back through the whole tensor, so nothing is saved: you build the exponential
object before compressing it.

Panel B: "Band + cross-approximation composes." The hyper-diagonal band bounds
each cell's dependency cone to a thin tube, making single-entry evaluation cheap;
cross-approximation (TT-cross) then samples only O(k n r^2) in-band entries to
build the low-rank representation directly -- the full tensor is never formed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

C_TENSOR = "#B0413E"     # exponential/danger red
C_BAND = "#2E7D8A"       # band teal
C_CORE = "#4A6FA5"       # low-rank blue
C_CONE = "#E8A33D"       # dependency cone amber
C_MUTE = "#9AA0A6"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))

# ============================================================ PANEL A
axA.set_title("A)  Decomposition alone hits the wall", fontsize=13, fontweight="bold", pad=12)
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")

# the big tensor (must be formed)
axA.add_patch(Rectangle((0.6, 2.2), 5.2, 5.2, facecolor=C_TENSOR, alpha=0.14,
                        edgecolor=C_TENSOR, lw=2))
# grid to suggest all cells
for gx in np.linspace(0.6, 5.8, 9):
    axA.plot([gx, gx], [2.2, 7.4], color=C_TENSOR, alpha=0.18, lw=0.6)
for gy in np.linspace(2.2, 7.4, 9):
    axA.plot([0.6, 5.8], [gy, gy], color=C_TENSOR, alpha=0.18, lw=0.6)
axA.text(3.2, 7.75, r"full tensor  $(L{+}1)^k$", ha="center", fontsize=11,
         color=C_TENSOR, fontweight="bold")

# dependency cone from sink reaching back through everything
sink = (5.5, 7.1)
for tx, ty in [(0.9, 2.5), (0.9, 4.5), (0.9, 6.5), (2.5, 2.5), (4.2, 2.5)]:
    axA.add_patch(FancyArrowPatch((tx, ty), sink, arrowstyle="-",
                  color=C_CONE, alpha=0.5, lw=1.1,
                  connectionstyle="arc3,rad=0.08"))
axA.plot(*sink, "o", color=C_TENSOR, ms=9)
axA.text(sink[0]+0.15, sink[1]+0.15, "sink entry", fontsize=9, color=C_TENSOR)
axA.text(3.2, 1.55, "every entry's dependency cone\nreaches through the whole tensor",
         ha="center", fontsize=9.5, color=C_CONE, style="italic")

# arrow to decomposition, which is downstream and doesn't help
axA.add_patch(FancyArrowPatch((6.0, 4.8), (7.2, 4.8), arrowstyle="-|>",
              color=C_MUTE, lw=2, mutation_scale=18))
axA.add_patch(FancyBboxPatch((7.3, 3.9), 2.3, 1.8, boxstyle="round,pad=0.1",
              facecolor=C_CORE, alpha=0.15, edgecolor=C_CORE, lw=1.8))
axA.text(8.45, 4.8, "low-rank\ncores", ha="center", va="center",
         fontsize=10, color=C_CORE, fontweight="bold")
axA.text(8.45, 3.35, "compression comes\nAFTER forming the\nexponential object",
         ha="center", va="top", fontsize=8.5, color=C_MUTE)
# big red X over the saving
axA.text(6.6, 6.6, r"$\times$", fontsize=34, color=C_TENSOR, ha="center",
         fontweight="bold")
axA.text(6.6, 5.75, "no saving", fontsize=9, color=C_TENSOR, ha="center")

# ============================================================ PANEL B
axB.set_title("B)  Band + cross-approximation composes", fontsize=13, fontweight="bold", pad=12)
axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")

# faint full tensor outline (never formed)
axB.add_patch(Rectangle((0.6, 2.2), 5.2, 5.2, facecolor="none",
                        edgecolor=C_MUTE, lw=1, ls="--", alpha=0.6))
axB.text(3.2, 7.75, "full tensor never formed", ha="center", fontsize=9.5,
         color=C_MUTE, style="italic")

# the band: a thin diagonal tube
band_pts_lo = [(0.6, 2.6), (5.8, 6.8)]
band_pts_hi = [(0.6, 3.4), (5.8, 7.4)]
xs = np.linspace(0.6, 5.8, 50)
lo = np.linspace(2.4, 6.6, 50)
hi = np.linspace(3.4, 7.6, 50)
axB.fill_between(xs, lo, hi, color=C_BAND, alpha=0.25)
axB.plot(xs, (lo+hi)/2, color=C_BAND, lw=1.5)
axB.text(2.0, 5.7, "hyper-diagonal\nband (radius $\\varepsilon$)", fontsize=10,
         color=C_BAND, fontweight="bold", rotation=32, ha="center")

# bounded cone: short local arrows within the band only
for cx in np.linspace(1.4, 5.0, 5):
    cy = 2.4 + (cx-0.6)*0.81 + 0.5
    axB.add_patch(FancyArrowPatch((cx-0.5, cy-0.4), (cx, cy), arrowstyle="-|>",
                  color=C_CONE, lw=1.3, mutation_scale=10))
axB.text(3.2, 1.55, "each entry's cone is bounded to the band\n=> single-entry"
         " evaluation is cheap", ha="center", fontsize=9.5, color=C_CONE, style="italic")

# sampled entries (cross-approximation) -- scattered dots in band
rng = np.random.default_rng(3)
sx = rng.uniform(1.0, 5.4, 14)
sy = np.array([2.4 + (x-0.6)*0.81 + rng.uniform(0.05, 0.75) for x in sx])
axB.plot(sx, sy, "o", color=C_CORE, ms=6, alpha=0.9)
axB.text(4.9, 3.0, r"sample $O(k\,n\,r^2)$" + "\nin-band entries", fontsize=8.5,
         color=C_CORE, ha="center")

# arrow to cores, which ARE the output (built directly)
axB.add_patch(FancyArrowPatch((6.0, 4.8), (7.2, 4.8), arrowstyle="-|>",
              color=C_CORE, lw=2.2, mutation_scale=18))
axB.add_patch(FancyBboxPatch((7.3, 3.9), 2.3, 1.8, boxstyle="round,pad=0.1",
              facecolor=C_CORE, alpha=0.22, edgecolor=C_CORE, lw=2))
axB.text(8.45, 4.8, "low-rank\ncores", ha="center", va="center",
         fontsize=10, color=C_CORE, fontweight="bold")
axB.text(8.45, 3.35, "built DIRECTLY from\nsampled entries;\nno dense tensor",
         ha="center", va="top", fontsize=8.5, color=C_CORE)
axB.text(6.6, 6.5, r"$\checkmark$", fontsize=30, color=C_BAND, ha="center",
         fontweight="bold")

# shared caption strip
fig.text(0.5, 0.02,
         "The partition (log-sum-exp) tensor is the field-native target; the band supplies the cheap entry oracle "
         "that cross-approximation needs.",
         ha="center", fontsize=9.5, style="italic", color="#333333")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("figure_composition.png", dpi=200, bbox_inches="tight")
print("wrote figure_composition.png")
