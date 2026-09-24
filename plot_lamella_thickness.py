"""
plot_lamella_thickness.py
===========================

WHAT THIS SCRIPT DOES (plain-English summary)
-----------------------------------------------
Every tomogram in this project was cut from one of several "lamellae"
(thin slices of frozen sample). Lamellae were grouped into clusters based
on their position on the microscope stage (a method called "k-means
clustering" -- don't worry about the maths, just think of it as "grouping
nearby lamellae together"). This script draws one "box plot" per group,
showing how thick the lamellae in that group were estimated to be.

A BOX PLOT is a standard way to summarise a bunch of numbers at once:
    - the black line in the middle of each box is the MEDIAN (the
      "typical"/middle value)
    - the coloured box itself covers the middle 50% of the values
    - the thin black "whiskers" sticking out the top and bottom show the
      overall spread
On top of each box, we also draw every individual measurement as a small
dot, nudged left/right slightly at random ("jittered") purely so dots
that would otherwise land right on top of each other are still visible
individually.

Each group (L1, L2, L3, ...) gets its own colour, purely to make it easy
to tell them apart -- the colour itself doesn't mean anything.

WHERE THE NUMBERS COME FROM
-----------------------------
`LAMELLATHICKNESS.csv` is a simple spreadsheet-style text file with one
row per tomogram, and two columns: which lamella-group it belongs to, and
its estimated thickness. The thickness in that file is measured in
Angstrom (a very small unit -- 10 Angstrom = 1 nanometre), so this script
divides every value by 10 to convert it to nanometres (nm), which is the
more usual unit for reporting lamella thickness.

HOW TO RUN THIS SCRIPT
-------------------------
    python3 plot_lamella_thickness.py

That's it -- it will read `LAMELLATHICKNESS.csv` in the current folder
and write a picture called `lamella_thickness.png` next to it.

If your file has a different name or is somewhere else, tell the script
where to look and what to call the picture:

    python3 plot_lamella_thickness.py --i path/to/LAMELLATHICKNESS.csv --o my_picture.png
"""

# --- Tools this script needs ---
import argparse   # reads --i and --o typed on the command line
import csv        # reads the LAMELLATHICKNESS.csv spreadsheet-style file
from collections import defaultdict  # a dictionary that starts every new entry as an empty list

import numpy as np  # a toolkit for doing maths on lists of numbers quickly
import matplotlib
matplotlib.use("Agg")  # tells matplotlib to just save a file, not try to pop up a window
import matplotlib.pyplot as plt
import matplotlib.cm as colormap  # a ready-made set of distinct colours to pick from


def load_thickness_measurements_grouped_by_lamella(csv_path):
    """
    Reads the CSV file and sorts every thickness measurement into a
    "bucket" (a Python list) according to which lamella it belongs to.

    Returns a dictionary that looks like:
        {1: [304.2, 291.4, ...], 2: [332.5, 307.1, ...], ...}
    i.e. lamella number -> list of thickness measurements (in nm), sorted
    so that lamella 1 comes first, then 2, then 3, and so on.
    """
    # defaultdict(list) means: the first time we mention a new lamella
    # number, it automatically starts with an empty list, so we don't have
    # to check "have I seen this one before?" ourselves.
    measurements_by_lamella = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)

        # The thickness column's exact name is a bit odd/inconsistent, so
        # instead of hard-coding it, we find whichever column starts with
        # the word "Thickness".
        thickness_column_name = None
        for column_name in reader.fieldnames:
            if column_name.lower().startswith("thickness"):
                thickness_column_name = column_name
                break

        for row in reader:
            lamella_number = int(row["Lamella"])
            thickness_in_angstrom = float(row[thickness_column_name])
            thickness_in_nanometres = thickness_in_angstrom / 10.0  # 10 Angstrom = 1 nm
            measurements_by_lamella[lamella_number].append(thickness_in_nanometres)

    # Sort by lamella number (1, 2, 3, ...) so the chart reads left-to-right in order.
    return dict(sorted(measurements_by_lamella.items()))


def main():
    # --- Read what the user typed on the command line (or use the defaults) ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", default="LAMELLATHICKNESS.csv",
                         help="the input file to read (default: LAMELLATHICKNESS.csv)")
    parser.add_argument("--o", default="lamella_thickness.png",
                         help="the filename to save the picture as")
    args = parser.parse_args()

    # --- Step 1: get the data ---
    measurements_by_lamella = load_thickness_measurements_grouped_by_lamella(args.i)
    lamella_numbers = list(measurements_by_lamella.keys())
    number_of_lamellae = len(lamella_numbers)

    # One list of measurements per lamella, in the same order as lamella_numbers.
    list_of_measurement_lists = list(measurements_by_lamella.values())

    # x-axis labels like "L1\n(n=28)" -- the lamella name plus how many
    # tomograms went into it.
    x_axis_labels = []
    for lamella_number, measurements in measurements_by_lamella.items():
        x_axis_labels.append(f"L{lamella_number}\n(n={len(measurements)})")

    # Pick a distinct colour for each lamella from a ready-made palette of 10.
    colors_per_lamella = [colormap.tab10(i % 10) for i in range(number_of_lamellae)]

    # x-axis positions: 1, 2, 3, ... one per lamella.
    x_positions = np.arange(1, number_of_lamellae + 1)

    # --- Step 2: draw the box plots ---
    # figsize width grows with the number of lamellae, so the boxes don't get squashed together.
    fig, ax = plt.subplots(figsize=(1.1 * number_of_lamellae + 1, 5))

    boxplot_result = ax.boxplot(
        list_of_measurement_lists,
        positions=x_positions,
        widths=0.6,
        patch_artist=True,     # lets us fill each box with a colour
        showfliers=False,      # don't draw matplotlib's own "outlier" dots -- we draw ALL points ourselves below
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        boxprops=dict(edgecolor="black", linewidth=0.8),
    )
    # Colour in each box to match its lamella.
    for box_patch, color in zip(boxplot_result["boxes"], colors_per_lamella):
        box_patch.set_facecolor(color)
        box_patch.set_alpha(0.55)  # slightly see-through, so the black median line stands out

    # --- Step 3: draw every individual measurement as a dot on top ---
    # random_generator makes the left/right "jitter" for each dot. Using a
    # fixed starting point (seed 0) means the dots land in the same spots
    # every time you re-run the script, so the picture doesn't change
    # slightly each time for no reason.
    random_generator = np.random.default_rng(seed=0)

    for x_position, measurements, color in zip(x_positions, list_of_measurement_lists, colors_per_lamella):
        # Nudge each dot's x-position left or right by a small random
        # amount, so dots with similar thickness don't hide behind each other.
        jitter_amounts = random_generator.uniform(-0.18, 0.18, size=len(measurements))
        jittered_x_positions = x_position + jitter_amounts
        ax.scatter(
            jittered_x_positions, measurements,
            color=color, edgecolor="black", linewidth=0.3,
            s=18,        # dot size
            alpha=0.85,  # slightly see-through
            zorder=3,    # draw dots on top of the boxes, not behind them
        )

    # --- Step 4: add titles, axis labels, and other text ---
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_axis_labels)
    ax.set_ylabel("Estimated thickness (nm)")
    ax.set_title(f"Lamella thickness\n(K-means k={number_of_lamellae}, stage pos.)")
    ax.grid(alpha=0.3, axis="y")   # faint horizontal gridlines, easier to read values off
    ax.set_axisbelow(True)         # keep those gridlines behind the boxes/dots, not on top

    # --- Step 5: save the picture to disk ---
    fig.tight_layout()  # shrinks margins so nothing gets cut off
    fig.savefig(args.o, dpi=200)  # dpi=200 controls how sharp/high-resolution the image is
    print(f"Saved plot to {args.o}")

    # Also print a short plain-text summary for every lamella, for a quick check.
    for lamella_number, measurements in measurements_by_lamella.items():
        values = np.array(measurements)
        print(
            f"L{lamella_number}: n={len(values)}, median={np.median(values):.1f} nm, "
            f"mean={values.mean():.1f} nm, range=[{values.min():.1f}, {values.max():.1f}] nm"
        )


# This just means "if this file is run directly, call main()". It's a
# standard Python convention and can be ignored.
if __name__ == "__main__":
    main()
