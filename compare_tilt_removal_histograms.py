"""
compare_tilt_removal_histograms.py
====================================

Overlays the "tilts removed per TS" distribution (dark images + low
overlap, summed -- see plot_tilt_removal_histogram.py) from TWO AreTomo3
`analyse` runs on one plot, normalised to % of tilt-series so that
projects with different TS counts compare fairly. Written to let this
project's data collection be compared directly against an older/worse one
(e.g. bi38262-21-akinetes/relion2 vs. this project, bi38262-30-akinetes).

WHY TWO LOADING PATHS
----------------------
This project's `aretomo3_project.json` (`aretomo3-preprocess` from
2026-08 onward) stores a ready-made `analyse.per_ts_qc` list -- one entry
per TS with `n_dark`/`n_bad` already counted (see
plot_tilt_removal_histogram.py's docstring for how those were derived).

Older projects' `analyse` blocks (e.g. bi38262-21-akinetes/relion2, from
an earlier version of the tool) do NOT have `per_ts_qc` -- confirmed by
inspecting that project's own `run001-cmd0/analyse/aretomo3_project.json`,
which has `n_flagged_frames`/`n_ts_with_flags` totals but no per-TS
breakdown. For those, this script recomputes the same two counts directly
from that analyse run's own raw outputs, sitting alongside it on disk:

    n_bad  -- number of rows per `ts_name` in that analyse dir's own
              `flagged_frames.tsv` (same format/meaning as the current
              project's: one row per frame whose `overlap_pct` fell below
              that run's own `threshold`, confirmed identical column
              layout in both projects' files)
    n_dark -- number of `# DarkFrame = ...` header lines in each
              `ts-XXX.aln` file in the analyse dir's PARENT directory
              (AreTomo3's own alignment output dir, e.g. `run001-cmd0/`,
              of which `run001-cmd0/analyse/` is a subdirectory) --
              confirmed this is the same per-TS dark-frame count reported
              via AreTomo3's own log ("Remove image N at ... deg") for
              the current project's equivalent run.

Only tilt series with >=1 removed tilt (dark or low-overlap) are counted
and plotted, matching this lab's existing "Removed tilts (n=... TS)" QC
panel convention (see plot_tilt_removal_histogram.py).

USAGE
-----
    python3 compare_tilt_removal_histograms.py \\
        --i1 /path/to/old_project/run001-cmd0/analyse --label1 "Previous collection" \\
        --i2 /path/to/this_project/aretomo3_project.json --label2 "This collection" \\
        --o tilt_removal_comparison.png

`--i1`/`--i2` each accept EITHER an analyse output directory (containing
`aretomo3_project.json` and `flagged_frames.tsv`) OR a path directly to an
`aretomo3_project.json` file (its own directory is then treated as the
analyse dir).
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on a headless HPC node; still writes the PNG fine
import matplotlib.pyplot as plt

# Categorical palette slots 1 (blue) and 2 (orange) -- validated colorblind-
# safe pair (fixed order, not cycled), from this project's dataviz palette.
COLOR_1 = "#2a78d6"
COLOR_2 = "#eb6834"


def _resolve_paths(path):
    """Returns (analyse_dir, project_json_path, aln_dir) for either an
    analyse-directory path or a direct aretomo3_project.json path."""
    if os.path.isdir(path):
        analyse_dir = path.rstrip("/")
        project_json = os.path.join(analyse_dir, "aretomo3_project.json")
    else:
        project_json = path
        analyse_dir = os.path.dirname(os.path.abspath(path))
    aln_dir = os.path.dirname(analyse_dir)
    return analyse_dir, project_json, aln_dir


def load_total_removed(path):
    """Returns (dict: ts_name -> total tilts removed, threshold_pct)."""
    analyse_dir, project_json, aln_dir = _resolve_paths(path)

    with open(project_json) as fh:
        d = json.load(fh)
    analyse = d.get("analyse")
    if analyse is None:
        raise ValueError(f"{project_json}: no 'analyse' block found")
    threshold = analyse.get("args", {}).get("threshold")

    if "per_ts_qc" in analyse:
        totals = {ts["name"]: ts["n_dark"] + ts["n_bad"] for ts in analyse["per_ts_qc"]}
        return totals, threshold

    # Older-schema project: recompute n_bad from flagged_frames.tsv and
    # n_dark from the sibling .aln files' DarkFrame header lines.
    flagged_tsv = os.path.join(analyse_dir, "flagged_frames.tsv")
    n_bad = defaultdict(int)
    with open(flagged_tsv) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            n_bad[row["ts_name"]] += 1

    n_dark = {}
    aln_files = sorted(glob.glob(os.path.join(aln_dir, "ts-*.aln")))
    if not aln_files:
        raise ValueError(
            f"{path}: no per-TS 'per_ts_qc' in {project_json} AND no "
            f"ts-*.aln files found in {aln_dir} to recompute dark-frame "
            f"counts from -- can't determine tilts removed for this project."
        )
    for f in aln_files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            n_dark[name] = sum(1 for line in fh if line.startswith("# DarkFrame"))

    all_names = set(n_dark) | set(n_bad)
    totals = {name: n_dark.get(name, 0) + n_bad.get(name, 0) for name in all_names}
    return totals, threshold


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i1", required=True, help="first project's analyse dir or aretomo3_project.json")
    p.add_argument("--i2", required=True, help="second project's analyse dir or aretomo3_project.json")
    p.add_argument("--label1", default=None, help="legend label for --i1")
    p.add_argument("--label2", default=None, help="legend label for --i2")
    p.add_argument("--o", default="tilt_removal_comparison.png", help="output image path")
    args = p.parse_args(argv)

    totals1, thresh1 = load_total_removed(args.i1)
    totals2, thresh2 = load_total_removed(args.i2)

    label1 = args.label1 or args.i1
    label2 = args.label2 or args.i2

    vals1 = np.array([v for v in totals1.values() if v > 0])
    vals2 = np.array([v for v in totals2.values() if v > 0])
    n1, n2 = len(vals1), len(vals2)
    med1, med2 = np.median(vals1), np.median(vals2)

    max_count = int(max(vals1.max(), vals2.max()))
    bins = np.arange(0, max_count + 2) - 0.5  # integer-centred bins, one per tilt count

    fig, ax = plt.subplots(figsize=(8, 5.5))
    w1 = np.ones_like(vals1, dtype=float) * 100.0 / n1
    w2 = np.ones_like(vals2, dtype=float) * 100.0 / n2
    ax.hist(vals1, bins=bins, weights=w1, color=COLOR_1, alpha=0.55,
            edgecolor=COLOR_1, linewidth=1.0,
            label=f"{label1} (n={n1} TS, median={med1:.0f})")
    ax.hist(vals2, bins=bins, weights=w2, color=COLOR_2, alpha=0.55,
            edgecolor=COLOR_2, linewidth=1.0,
            label=f"{label2} (n={n2} TS, median={med2:.0f})")
    ax.axvline(med1, color=COLOR_1, linestyle="--", linewidth=1.2)
    ax.axvline(med2, color=COLOR_2, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Tilts removed per TS (dark + overlap-flagged)")
    ax.set_ylabel("% of tilt-series")
    ax.set_title("Tilts removed per TS: data-collection comparison")
    ax.set_xlim(-1, max_count + 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.o, dpi=200)
    print(f"Saved plot to {args.o}")
    print(f"{label1}: n={n1}, median={med1:.0f}, mean={vals1.mean():.2f}, max={vals1.max()}, threshold={thresh1}")
    print(f"{label2}: n={n2}, median={med2:.0f}, mean={vals2.mean():.2f}, max={vals2.max()}, threshold={thresh2}")


if __name__ == "__main__":
    main()
