"""
plot_tilt_removal_histogram.py
===============================

Histogram of TOTAL tilt images removed per tilt-series (TS) in this
project's AreTomo3 preprocessing -- combining both removal reasons into
one per-TS count:

    - dark images   (AreTomo3's own `-DarkTol` frame exclusion)
    - low overlap   (post-alignment frames whose overlap with neighbouring
                      tilts fell below the QC threshold -- 80% in this
                      project)

Styled to match the lab's existing "Removed tilts" QC panel: plain grey
bars, a dashed vertical line at the median, x-axis ticked every 3 tilts.

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
`n_dark` and `n_bad` are counting disjoint sets of frames and are summed
here (`n_dark + n_bad`) into one total-removed-per-TS count without
double-counting.

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

BAR_COLOR = "#808080"
MEDIAN_COLOR = "#000000"


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
    total_removed_all = np.array([ts["n_dark"] + ts["n_bad"] for ts in qc])
    # Only TS with >=1 removed tilt are plotted/counted, matching this lab's
    # existing "Removed tilts (n=... TS)" QC panel convention.
    total_removed = total_removed_all[total_removed_all > 0]
    n_ts = len(total_removed)
    median = np.median(total_removed)

    max_count = int(total_removed.max())
    bins = np.arange(0, max_count + 2) - 0.5  # integer-centred bins, one per tilt count

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(total_removed, bins=bins, color=BAR_COLOR, edgecolor="black", linewidth=0.5)
    ax.axvline(median, color=MEDIAN_COLOR, linestyle="--", linewidth=1.2,
               label=f"Median = {median:.0f}")

    ax.set_xlabel("Tilts removed per TS")
    ax.set_ylabel("Number of tilt-series")
    ax.set_title(f"Removed tilts\n(n={n_ts} TS, thresh={threshold:.0f}%)")
    ax.set_xlim(-1, max_count + 1)
    ax.set_xticks(np.arange(0, max_count + 1, 3))
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.o, dpi=200)
    print(f"Saved plot to {args.o}")
    print(f"total removed per TS: mean {total_removed.mean():.2f}, median {median:.0f}, "
          f"max {max_count}, {(total_removed > 0).sum()}/{n_ts} TS with >=1 removed")


if __name__ == "__main__":
    main()
