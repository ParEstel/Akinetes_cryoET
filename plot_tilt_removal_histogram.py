"""
plot_tilt_removal_histogram.py
===============================

WHAT THIS SCRIPT DOES (plain-English summary)
-----------------------------------------------
During tilt-series alignment (AreTomo3), some of the individual tilt
images in each tilt-series get thrown away, for two reasons:

    1. "dark" images   -- the image was blank/black (e.g. the beam was
                           blanked, or the stage was moving), so AreTomo3
                           discards it automatically.
    2. "low overlap"    -- after alignment, the image didn't line up well
                           enough with its neighbours (less than 80% of
                           the image area overlapped), so it was flagged
                           as unreliable.

This script counts, for every tilt-series, how many tilt images were
thrown away in total (reason 1 + reason 2 added together), and then
draws a histogram: a bar chart showing "how many tilt-series lost 0
tilts, how many lost 1 tilt, how many lost 2 tilts", and so on. This
makes it easy to see at a glance whether most tilt-series are healthy
(losing very few tilts) or whether a lot of data had to be thrown away.

The chart also draws a dashed vertical line at the MEDIAN (the middle
value, if you lined up every tilt-series worst-to-best) so you have a
single "typical" number to quote, not just a wall of bars.

WHERE THE NUMBERS COME FROM
-----------------------------
All of this is read out of one file that AreTomo3's preprocessing
already produced: `aretomo3_project.json`. Inside it, there's a section
called "analyse" that already lists, for every tilt-series, how many
tilts were dropped for each reason (fields named `n_dark` and `n_bad`).
We simply add those two numbers together for each tilt-series.

Tilt-series that lost ZERO tilts are left out of the chart on purpose,
to match how this lab has reported this number before (a tilt-series
with a perfect score doesn't need to be in a "how bad was it" plot).

HOW TO RUN THIS SCRIPT
-------------------------
From a terminal, inside the project folder that contains
`aretomo3_project.json`, run:

    python3 plot_tilt_removal_histogram.py

That's it -- it will read `aretomo3_project.json` in the current folder
and write a picture called `tilt_removal_histogram.png` next to it.

If your file has a different name or is somewhere else, tell the script
where to look and what to call the picture:

    python3 plot_tilt_removal_histogram.py --i path/to/aretomo3_project.json --o my_picture.png
"""

# These lines "import" (load) the extra tools this script needs.
import argparse   # lets the script understand --i and --o typed on the command line
import json       # lets the script read .json files
import numpy as np  # a toolkit for doing maths on lists of numbers quickly

# matplotlib is the tool that actually draws the chart and saves it as a PNG picture.
import matplotlib
matplotlib.use("Agg")  # tells matplotlib to just save a file, not try to pop up a window
import matplotlib.pyplot as plt

# The colours used in the chart, as "hex codes" (a standard way of writing colours).
# Feel free to change these if you want a different look.
BAR_COLOR = "#808080"     # grey, used for the histogram bars
MEDIAN_LINE_COLOR = "#000000"  # black, used for the dashed median line


def get_tilts_removed_per_tilt_series(json_path):
    """
    Opens the aretomo3_project.json file and pulls out, for every tilt
    series, how many tilt images were removed in total.

    Returns two things:
      - a list of numbers (one per tilt-series: how many tilts it lost)
      - the overlap threshold (e.g. 80.0, meaning "80%") that was used
        when deciding which images counted as "low overlap"
    """
    with open(json_path) as f:
        project_data = json.load(f)

    # The per-tilt-series numbers live inside project_data["analyse"]["per_ts_qc"].
    analyse_section = project_data.get("analyse")
    if analyse_section is None or "per_ts_qc" not in analyse_section:
        raise ValueError(
            f"Could not find the expected data inside {json_path}. "
            f"Make sure this is the right aretomo3_project.json file, and "
            f"that the 'analyse' step has already been run."
        )

    per_tilt_series_info = analyse_section["per_ts_qc"]
    overlap_threshold = analyse_section.get("args", {}).get("threshold")

    # For each tilt series, add "dark images removed" + "low-overlap images
    # removed" together to get one total number for that tilt series.
    totals = []
    for one_tilt_series in per_tilt_series_info:
        dark_count = one_tilt_series["n_dark"]
        low_overlap_count = one_tilt_series["n_bad"]
        totals.append(dark_count + low_overlap_count)

    return totals, overlap_threshold


def main():
    # --- Read what the user typed on the command line (or use the defaults) ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", default="aretomo3_project.json",
                         help="the input file to read (default: aretomo3_project.json)")
    parser.add_argument("--o", default="tilt_removal_histogram.png",
                         help="the filename to save the picture as")
    args = parser.parse_args()

    # --- Step 1: get the data ---
    all_totals, overlap_threshold = get_tilts_removed_per_tilt_series(args.i)

    # Only keep tilt-series that lost at least 1 tilt (see docstring above
    # for why) -- this matches how the lab has plotted this before.
    totals_to_plot = [t for t in all_totals if t > 0]
    number_of_tilt_series = len(totals_to_plot)
    worst_case = max(totals_to_plot)
    median_value = float(np.median(totals_to_plot))

    # --- Step 2: decide how to divide the x-axis into bars ("bins") ---
    # We want one bar per possible whole number of tilts removed: 1, 2, 3, ...
    # up to the worst case seen. np.arange makes a list of numbers, and the
    # "-0.5" shifts the bar edges so each bar is centred ON its number
    # instead of starting at it (this is just a cosmetic/plotting detail).
    bin_edges = np.arange(0, worst_case + 2) - 0.5

    # --- Step 3: draw the chart ---
    fig, ax = plt.subplots(figsize=(5, 4))  # figsize is the picture size in inches

    ax.hist(
        totals_to_plot,
        bins=bin_edges,
        color=BAR_COLOR,
        edgecolor="black",   # a thin black outline around each bar, for clarity
        linewidth=0.5,
    )

    # Draw a dashed vertical line at the median, with a label for the legend.
    ax.axvline(
        median_value,
        color=MEDIAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Median = {median_value:.0f}",
    )

    # --- Step 4: add titles, axis labels, and other text ---
    ax.set_xlabel("Tilts removed per TS")
    ax.set_ylabel("Number of tilt-series")
    ax.set_title(f"Removed tilts\n(n={number_of_tilt_series} TS, thresh={overlap_threshold:.0f}%)")
    ax.set_xlim(-1, worst_case + 1)
    ax.set_xticks(np.arange(0, worst_case + 1, 3))  # a tick mark every 3 tilts
    ax.legend(frameon=False)  # shows the "Median = ..." label, no box around it

    # --- Step 5: save the picture to disk ---
    fig.tight_layout()  # shrinks margins so nothing gets cut off
    fig.savefig(args.o, dpi=200)  # dpi=200 controls how sharp/high-resolution the image is
    print(f"Saved plot to {args.o}")

    # Also print a short plain-text summary to the terminal, for a quick check.
    average_value = sum(totals_to_plot) / len(totals_to_plot)
    print(
        f"total removed per TS: mean {average_value:.2f}, median {median_value:.0f}, "
        f"max {worst_case}, {number_of_tilt_series}/{number_of_tilt_series} TS with >=1 removed"
    )


# This just means "if this file is run directly, call main()". It's a
# standard Python convention and can be ignored.
if __name__ == "__main__":
    main()
