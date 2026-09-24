"""
plot_tilt_removal_histogram.py
===============================

Histogram of tilt images removed per tilt-series (TS), broken down by the
two removal reasons used in this project's AreTomo3 preprocessing:

    - dark images   (AreTomo3's own `-DarkTol` frame exclusion)
    - low overlap   (post-alignment frames whose overlap with neighbouring
                      tilts fell below the QC threshold -- 80% in this
                      project)

DATA SOURCE, CONFIRMED AGAINST THIS PROJECT'S OWN FILES
--------------------------------------------------------
`aretomo3_project.json`'s `analyse` block records the
`aretomo3-preprocess analyse` step (run002-cmd1-tc1-1880/analyse), whose
own `args.threshold` is 80.0 -- i.e. the 80% overlap cutoff is this
project's actual QC setting, not an assumed default. That block's
`per_ts_qc` list has one entry per tilt series with, among other fields:

    name      -- e.g. "ts-001"
    n_dark    -- count of frames AreTomo3 removed as dark for this TS
    n_bad     -- count of frames flagged with overlap_pct below the
                 `threshold` above (see the sibling `flagged_frames.tsv`
                 in the same analyse output dir, which lists these by
                 name/section/overlap_pct and was used to derive n_bad)

Dark frames are excluded by AreTomo3 before overlap is even computed (see
the per-series `.aln` files' `# DarkFrame = ...` header lines vs. this
project's `flagged_frames.tsv` -- no section index appears in both), so
`n_dark` and `n_bad` are counting disjoint sets of frames and can be
summed to a per-TS total without double-counting.

USAGE
-----
    python3 plot_tilt_removal_histogram.py \\
        --i aretomo3_project.json \\
        --o tilt_removal_histogram.png
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on a headless HPC node; still writes the PNG fine
import matplotlib.pyplot as plt

# Categorical palette slots 1 (blue) and 2 (orange) -- validated colorblind-
# safe pair (fixed order, not cycled), from this project's dataviz palette.
COLOR_DARK = "#2a78d6"
COLOR_OVERLAP = "#eb6834"


def load_per_ts_qc(path):
    with open(path) as fh:
        d = json.load(fh)
    analyse = d.get("analyse")
    if analyse is None or "per_ts_qc" not in analyse:
        raise ValueError(
            f"{path}: no 'analyse.per_ts_qc' block found -- this does not "
            f"look like this project's aretomo3_project.json (or the "
            f"'analyse' preprocessing step has not been run)."
        )
    threshold = analyse.get("args", {}).get("threshold")
    return analyse["per_ts_qc"], threshold


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i", default="aretomo3_project.json", help="path to aretomo3_project.json")
    p.add_argument("--o", default="tilt_removal_histogram.png", help="output image path")
    args = p.parse_args(argv)

    qc, threshold = load_per_ts_qc(args.i)
    n_dark = np.array([ts["n_dark"] for ts in qc])
    n_bad = np.array([ts["n_bad"] for ts in qc])
    n_ts = len(qc)

    max_count = int(max(n_dark.max(), n_bad.max()))
    x = np.arange(0, max_count + 1)
    dark_counts = np.bincount(n_dark, minlength=max_count + 1)
    bad_counts = np.bincount(n_bad, minlength=max_count + 1)

    # Grouped (not overlaid) bars -- overlapping semi-transparent fills would
    # blend the two categorical colors into a muddy third color and hurt
    # legibility, so each tilt-count gets two side-by-side bars with a small
    # gap between them instead.
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, dark_counts, width=width * 0.92, color=COLOR_DARK,
           label=f"dark images (n_dark, {int((n_dark > 0).sum())}/{n_ts} TS affected)")
    ax.bar(x + width / 2, bad_counts, width=width * 0.92, color=COLOR_OVERLAP,
           label=f"overlap < {threshold:.0f}% (n_bad, {int((n_bad > 0).sum())}/{n_ts} TS affected)")

    ax.set_xlabel("Tilt images removed per tilt-series")
    ax.set_ylabel("Number of tilt-series")
    ax.set_title(f"Tilts removed per TS -- dark images vs. overlap < {threshold:.0f}% (n={n_ts} TS)")
    ax.set_xlim(-1, max_count + 1)
    ax.set_xticks(x[::2] if max_count > 20 else x)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.o, dpi=200)
    print(f"Saved plot to {args.o}")
    print(f"dark images   : mean {n_dark.mean():.2f}, median {np.median(n_dark):.0f}, "
          f"max {n_dark.max()}, {(n_dark > 0).sum()}/{n_ts} TS with >=1 removed")
    print(f"overlap<{threshold:.0f}%  : mean {n_bad.mean():.2f}, median {np.median(n_bad):.0f}, "
          f"max {n_bad.max()}, {(n_bad > 0).sum()}/{n_ts} TS with >=1 flagged")


if __name__ == "__main__":
    main()
