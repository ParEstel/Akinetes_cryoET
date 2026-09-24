"""
compare_tilt_removal_histograms.py
====================================

WHAT THIS SCRIPT DOES (plain-English summary)
-----------------------------------------------
This is the sister script to `plot_tilt_removal_histogram.py`. Instead of
plotting ONE project's "tilts removed per tilt-series" data, it puts TWO
projects' data on the SAME chart, one drawn in blue and one in orange, so
you can directly compare them -- for example, "was this data collection
actually better than the previous one?"

Because the two projects might have a different total number of
tilt-series, the bars show PERCENTAGES ("what fraction of this project's
tilt-series lost this many tilts") rather than raw counts -- that way a
project with 100 tilt-series and one with 200 tilt-series can be compared
fairly, on the same scale.

A dashed line is drawn for each project's median (the "typical" value),
in that project's own colour, so you can read off two headline numbers at
a glance.

WHERE THE NUMBERS COME FROM (AND WHY THIS SCRIPT HAS TWO WAYS OF READING THEM)
---------------------------------------------------------------------------------
Both projects come from the same AreTomo3 preprocessing pipeline, but
older projects were processed with an older version of the tool that
saved its results slightly differently:

  - NEWER projects have a ready-made list of numbers already sitting in
    their `aretomo3_project.json` file (under `analyse` -> `per_ts_qc`),
    with one entry per tilt-series telling us straight away how many
    "dark" and how many "low overlap" images it lost.

  - OLDER projects don't have that ready-made list. For those, this
    script works the numbers out itself, from two files that are always
    produced regardless of tool version:
        * `flagged_frames.tsv` -- a spreadsheet-like text file listing
          every individual image that was flagged as "low overlap"; we
          just count how many times each tilt-series' name appears in it.
        * the `.aln` files (one per tilt-series) -- these contain a line
          starting with "# DarkFrame" for every image that was thrown out
          as "dark"; we count those lines.

Either way, we end up with the same thing: one number per tilt-series,
"how many tilt images did it lose in total".

Tilt-series that lost ZERO tilts are left out of the comparison, to match
how this lab has reported this number before.

HOW TO RUN THIS SCRIPT
-------------------------
    python3 compare_tilt_removal_histograms.py \\
        --i1 /path/to/OLDER_project/aretomo3_project.json --label1 "Previous collection" \\
        --i2 /path/to/THIS_project/aretomo3_project.json  --label2 "This collection" \\
        --o tilt_removal_comparison.png

`--i1` and `--i2` can each point EITHER straight at an
`aretomo3_project.json` file, OR at the folder that contains one (an
"analyse" output folder) -- the script figures out which you gave it.

`--label1`/`--label2` are just the names that will appear in the chart's
legend, so you can call them whatever makes sense (e.g. "Session A",
"March collection", etc).
"""

# --- Tools this script needs ---
import argparse   # reads --i1, --i2, --label1, etc. typed on the command line
import csv        # reads the flagged_frames.tsv spreadsheet-style file
import glob       # finds files that match a pattern, e.g. "all files named ts-*.aln"
import json       # reads .json files
import os         # for working with file/folder paths
from collections import defaultdict  # a dictionary that starts every new entry at 0

import numpy as np  # a toolkit for doing maths on lists of numbers quickly
import matplotlib
matplotlib.use("Agg")  # tells matplotlib to just save a file, not try to pop up a window
import matplotlib.pyplot as plt

# The two colours used to tell the two projects apart in the chart.
# These are colourblind-friendly and are used consistently across this
# lab's other charts (project 1 = blue, project 2 = orange).
COLOR_PROJECT_1 = "#2a78d6"  # blue
COLOR_PROJECT_2 = "#eb6834"  # orange


def find_the_analyse_folder_and_files(path_given_by_user):
    """
    The user can point us at either:
      (a) an "analyse" folder itself, e.g. ".../run001-cmd0/analyse", or
      (b) the aretomo3_project.json file directly inside that folder.

    Either way, we need three things: the analyse folder itself, the full
    path to its aretomo3_project.json file, and the folder ABOVE it (which
    is where the *.aln alignment files live, one per tilt-series, needed
    for older projects -- see the docstring above).

    This function works all three of those out and returns them.
    """
    if os.path.isdir(path_given_by_user):
        # The user gave us the folder itself.
        analyse_folder = path_given_by_user.rstrip("/")
        project_json_path = os.path.join(analyse_folder, "aretomo3_project.json")
    else:
        # The user gave us the .json file directly -- its folder IS the analyse folder.
        project_json_path = path_given_by_user
        analyse_folder = os.path.dirname(os.path.abspath(path_given_by_user))

    # The .aln files live one level up from the "analyse" folder.
    folder_above_analyse = os.path.dirname(analyse_folder)

    return analyse_folder, project_json_path, folder_above_analyse


def count_tilts_removed_the_new_way(analyse_section):
    """
    For newer projects, aretomo3_project.json already contains a
    ready-made list (per_ts_qc). We just add up the two numbers it gives
    us for each tilt-series.

    Returns a dictionary that looks like:
        {"ts-001": 5, "ts-002": 12, ...}
    i.e. tilt-series name -> total tilts removed.
    """
    totals_by_tilt_series = {}
    for one_tilt_series in analyse_section["per_ts_qc"]:
        name = one_tilt_series["name"]
        dark_count = one_tilt_series["n_dark"]
        low_overlap_count = one_tilt_series["n_bad"]
        totals_by_tilt_series[name] = dark_count + low_overlap_count
    return totals_by_tilt_series


def count_tilts_removed_the_old_way(analyse_folder, folder_above_analyse):
    """
    For older projects that don't have the ready-made list, work the same
    numbers out by hand from the raw files (see the big docstring at the
    top of this file for why this gives the same answer).

    Returns the same kind of dictionary as the "new way" function above.
    """
    # --- Count the "low overlap" images, from flagged_frames.tsv ---
    # This file has one row per flagged image, with a column called
    # "ts_name" saying which tilt-series it belongs to. We count how many
    # rows belong to each tilt-series.
    low_overlap_counts = defaultdict(int)  # defaultdict: unseen names start at 0
    flagged_frames_file = os.path.join(analyse_folder, "flagged_frames.tsv")
    with open(flagged_frames_file) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ts_name = row["ts_name"]
            low_overlap_counts[ts_name] += 1

    # --- Count the "dark" images, from each tilt-series' .aln file ---
    # Every removed dark image gets its own line starting with
    # "# DarkFrame" inside that tilt-series' .aln file, so we just count
    # those lines, once per file.
    dark_counts = {}
    aln_files = sorted(glob.glob(os.path.join(folder_above_analyse, "ts-*.aln")))
    if not aln_files:
        raise ValueError(
            f"Could not find any ts-*.aln files in {folder_above_analyse}, "
            f"and this project's aretomo3_project.json doesn't have the "
            f"newer ready-made numbers either -- there's no way to work out "
            f"how many tilts were removed."
        )
    for aln_file_path in aln_files:
        file_name_without_extension = os.path.splitext(os.path.basename(aln_file_path))[0]
        with open(aln_file_path) as f:
            number_of_dark_lines = 0
            for line in f:
                if line.startswith("# DarkFrame"):
                    number_of_dark_lines += 1
        dark_counts[file_name_without_extension] = number_of_dark_lines

    # --- Add the two counts together for every tilt-series we saw ---
    all_tilt_series_names = set(dark_counts.keys()) | set(low_overlap_counts.keys())
    totals_by_tilt_series = {}
    for name in all_tilt_series_names:
        totals_by_tilt_series[name] = dark_counts.get(name, 0) + low_overlap_counts.get(name, 0)

    return totals_by_tilt_series


def load_tilts_removed_for_one_project(path_given_by_user):
    """
    Works out how many tilts were removed per tilt-series for ONE
    project, trying the "new" (fast) way first and falling back to the
    "old" (recompute-from-scratch) way if needed.

    Returns (dictionary of {tilt_series_name: total_removed}, overlap_threshold_percent)
    """
    analyse_folder, project_json_path, folder_above_analyse = \
        find_the_analyse_folder_and_files(path_given_by_user)

    with open(project_json_path) as f:
        project_data = json.load(f)

    analyse_section = project_data.get("analyse")
    if analyse_section is None:
        raise ValueError(f"{project_json_path} doesn't have an 'analyse' section at all.")

    overlap_threshold = analyse_section.get("args", {}).get("threshold")

    if "per_ts_qc" in analyse_section:
        totals = count_tilts_removed_the_new_way(analyse_section)
    else:
        totals = count_tilts_removed_the_old_way(analyse_folder, folder_above_analyse)

    return totals, overlap_threshold


def main():
    # --- Read what the user typed on the command line ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1", required=True, help="first project's aretomo3_project.json (or its folder)")
    parser.add_argument("--i2", required=True, help="second project's aretomo3_project.json (or its folder)")
    parser.add_argument("--label1", default=None, help="name for project 1 in the chart legend")
    parser.add_argument("--label2", default=None, help="name for project 2 in the chart legend")
    parser.add_argument("--o", default="tilt_removal_comparison.png", help="filename to save the picture as")
    args = parser.parse_args()

    # If the user didn't give a custom legend name, just use the path they typed.
    label1 = args.label1 or args.i1
    label2 = args.label2 or args.i2

    # --- Step 1: load both projects' numbers ---
    totals1, threshold1 = load_tilts_removed_for_one_project(args.i1)
    totals2, threshold2 = load_tilts_removed_for_one_project(args.i2)

    # Only keep tilt-series that lost at least 1 tilt (see docstring above).
    values1 = np.array([count for count in totals1.values() if count > 0])
    values2 = np.array([count for count in totals2.values() if count > 0])

    number_of_ts_1 = len(values1)
    number_of_ts_2 = len(values2)
    median1 = float(np.median(values1))
    median2 = float(np.median(values2))

    # --- Step 2: decide how to divide the x-axis into bars ("bins") ---
    # One bar per whole number of tilts removed, from 0 up to the worst
    # case seen in EITHER project, so both histograms share the same scale.
    worst_case = int(max(values1.max(), values2.max()))
    bin_edges = np.arange(0, worst_case + 2) - 0.5

    # --- Step 3: turn "how many tilt-series" into "what % of tilt-series" ---
    # Each tilt-series in project 1 counts as (100 / number_of_ts_1) percent,
    # so the two bars add up to 100% each, making the two projects
    # comparable even if they don't have the same number of tilt-series.
    percent_per_ts_in_project_1 = np.full(number_of_ts_1, 100.0 / number_of_ts_1)
    percent_per_ts_in_project_2 = np.full(number_of_ts_2, 100.0 / number_of_ts_2)

    # --- Step 4: draw the chart ---
    fig, ax = plt.subplots(figsize=(8, 5.5))  # figsize is the picture size in inches

    ax.hist(
        values1, bins=bin_edges, weights=percent_per_ts_in_project_1,
        color=COLOR_PROJECT_1, alpha=0.55,  # alpha = see-through amount, so overlaps are visible
        edgecolor=COLOR_PROJECT_1, linewidth=1.0,
        label=f"{label1} (n={number_of_ts_1} TS, median={median1:.0f})",
    )
    ax.hist(
        values2, bins=bin_edges, weights=percent_per_ts_in_project_2,
        color=COLOR_PROJECT_2, alpha=0.55,
        edgecolor=COLOR_PROJECT_2, linewidth=1.0,
        label=f"{label2} (n={number_of_ts_2} TS, median={median2:.0f})",
    )

    # Dashed vertical line at each project's median, in its own colour.
    ax.axvline(median1, color=COLOR_PROJECT_1, linestyle="--", linewidth=1.2)
    ax.axvline(median2, color=COLOR_PROJECT_2, linestyle="--", linewidth=1.2)

    # --- Step 5: add titles, axis labels, and other text ---
    ax.set_xlabel("Tilts removed per TS (dark + overlap-flagged)")
    ax.set_ylabel("% of tilt-series")
    ax.set_title("Tilts removed per TS: data-collection comparison")
    ax.set_xlim(-1, worst_case + 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")  # faint horizontal gridlines, easier to read values off

    # --- Step 6: save the picture to disk ---
    fig.tight_layout()  # shrinks margins so nothing gets cut off
    fig.savefig(args.o, dpi=200)  # dpi=200 controls how sharp/high-resolution the image is
    print(f"Saved plot to {args.o}")

    # Also print a short plain-text summary to the terminal, for a quick check.
    print(f"{label1}: n={number_of_ts_1}, median={median1:.0f}, mean={values1.mean():.2f}, "
          f"max={values1.max()}, threshold={threshold1}")
    print(f"{label2}: n={number_of_ts_2}, median={median2:.0f}, mean={values2.mean():.2f}, "
          f"max={values2.max()}, threshold={threshold2}")


# This just means "if this file is run directly, call main()". It's a
# standard Python convention and can be ignored.
if __name__ == "__main__":
    main()
