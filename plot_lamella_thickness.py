"""
plot_lamella_thickness.py
===========================

Box + jittered-point plot of estimated lamella thickness, grouped by
lamella (stage-position k-means cluster, matching this project's
`aretomo3-preprocess analyse` grouping -- see `n_lamellae` in
`aretomo3_project.json`'s `analyse.args`). Styled to match the lab's
existing "Lamella thickness (K-means k=..., stage pos.)" QC panel
(`bar_plot.png`): one coloured box per lamella, individual tomograms
overlaid as jittered points, tick labels showing each lamella's n.

DATA SOURCE
-----------
`LAMELLATHICKNESS.csv` has two columns:

    Lamella                          -- integer lamella/cluster index
    Thickness tomogram (A) Z score   -- per-tomogram estimated thickness,
                                         in Angstrom (column name is
                                         misleading -- these are raw
                                         thickness values, ~1600-3300 A,
                                         not z-scores; confirmed by range
                                         matching the reference plot's
                                         0-350 nm axis once converted)

Thickness is converted A -> nm (/10) to match the reference plot's axis.

USAGE
-----
    python3 plot_lamella_thickness.py \\
        --i LAMELLATHICKNESS.csv \\
        --o lamella_thickness.png
"""

from __future__ import annotations
import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on a headless HPC node; still writes the PNG fine
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_thickness_by_lamella(path):
    groups = defaultdict(list)
    with open(path) as fh:
        reader = csv.DictReader(fh)
        thickness_col = [c for c in reader.fieldnames if c.lower().startswith("thickness")][0]
        for row in reader:
            groups[int(row["Lamella"])].append(float(row[thickness_col]) / 10.0)  # A -> nm
    return dict(sorted(groups.items()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i", default="LAMELLATHICKNESS.csv", help="path to LAMELLATHICKNESS.csv")
    p.add_argument("--o", default="lamella_thickness.png", help="output image path")
    args = p.parse_args(argv)

    groups = load_thickness_by_lamella(args.i)
    n_lamellae = len(groups)
    labels = [f"L{k}\n(n={len(v)})" for k, v in groups.items()]
    data = list(groups.values())

    colors = [cm.tab10(i % 10) for i in range(n_lamellae)]

    fig, ax = plt.subplots(figsize=(1.1 * n_lamellae + 1, 5))
    positions = np.arange(1, n_lamellae + 1)

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(color="black"), capprops=dict(color="black"),
                     boxprops=dict(edgecolor="black", linewidth=0.8))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(pos + jitter, vals, color=color, edgecolor="black",
                   linewidth=0.3, s=18, alpha=0.85, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated thickness (nm)")
    ax.set_title(f"Lamella thickness\n(K-means k={n_lamellae}, stage pos.)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(args.o, dpi=200)
    print(f"Saved plot to {args.o}")
    for k, vals in groups.items():
        arr = np.array(vals)
        print(f"L{k}: n={len(arr)}, median={np.median(arr):.1f} nm, "
              f"mean={arr.mean():.1f} nm, range=[{arr.min():.1f}, {arr.max():.1f}] nm")


if __name__ == "__main__":
    main()
