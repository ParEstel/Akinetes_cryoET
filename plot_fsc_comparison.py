"""
plot_fsc_comparison.py
=======================

Overlays the corrected (gold-standard, masked) FSC curves from two RELION
`postprocess.star` files on one matplotlib plot -- e.g. to visually and
numerically compare a C1 vs. a C2-symmetric refinement.

COLUMN NAMES USED, CONFIRMED AGAINST REAL RELION SOURCE/OUTPUT
------------------------------------------------------------------
RELION's `postprocess.star` has a `data_fsc` loop table. Column names
confirmed directly from `src/postprocessing.cpp` (which labels its own
exported FSC plot "RELION masked-corrected FSC", axes "Resolution (A-1)"
vs. "Correlation Coefficient") and cross-checked against two independent
community tools built on real RELION output (starpy's `plot_star.py`,
Follow_Relion_gracefully) -- both use exactly:

    rlnResolution                                         (x-axis, 1/Angstrom)
    rlnFourierShellCorrelationCorrected                    (the main gold-standard curve -- what this script plots)
    rlnFourierShellCorrelationUnmaskedMaps
    rlnFourierShellCorrelationMaskedMaps
    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps

This matches the exact legend labels seen on a real postprocess.star FSC
plot from this dataset earlier in this project (rlnFourierShellCorrelation-
Corrected/UnmaskedMaps/MaskedMaps/CorrectedFourierShellCorrelationPhase-
RandomizedMaskedMaps), so these are read with confidence, not guessed.

USAGE
-----
    python3 plot_fsc_comparison.py \\
        --i1 PostProcess/job1/postprocess.star --label1 "C1" \\
        --i2 PostProcess/job2/postprocess.star --label2 "C2" \\
        --o fsc_comparison.png

Also prints the resolution (Angstrom) at which each curve crosses the
threshold (default 0.143), found by linear interpolation between the two
bracketing points -- not just read off the plot by eye.
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on a headless HPC node; still writes the PNG fine
import matplotlib.pyplot as plt


def read_star_loop(path, block_name, columns_wanted):
    """Minimal STAR loop reader, scoped to a single named block
    (e.g. 'data_fsc'). Returns a dict: column_name -> list[str]. Only the
    columns in `columns_wanted` are kept; raises a clear error if any of
    them are missing from the block, rather than silently returning
    partial/garbage data."""
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n and lines[i].strip() != block_name:
        i += 1
    if i >= n:
        raise ValueError(f"Block '{block_name}' not found in {path}")
    i += 1
    while i < n and lines[i].strip() == "":
        i += 1
    if i >= n or lines[i].strip() != "loop_":
        raise ValueError(f"Expected 'loop_' after {block_name} in {path}")
    i += 1

    columns = []
    while i < n and lines[i].strip().startswith("_"):
        columns.append(lines[i].strip().split()[0][1:])
        i += 1

    missing = [c for c in columns_wanted if c not in columns]
    if missing:
        raise ValueError(
            f"{path}, block {block_name}: missing expected column(s) "
            f"{missing}. Columns actually present: {columns}. This RELION "
            f"version/job type may use different FSC column names than "
            f"expected -- check the file directly before assuming this "
            f"script is wrong."
        )

    data = {c: [] for c in columns_wanted}
    while i < n and lines[i].strip() != "" and not lines[i].strip().startswith("data_"):
        parts = lines[i].split()
        if len(parts) != len(columns):
            i += 1
            continue
        row = dict(zip(columns, parts))
        for c in columns_wanted:
            data[c].append(row[c])
        i += 1

    return {c: np.array([float(x) for x in v]) for c, v in data.items()}


def crossing_resolution(resolution_invA, fsc, threshold):
    """Resolution (Angstrom) at which `fsc` first drops through
    `threshold`, going from high FSC (low resolution / low 1/A) to low FSC
    (high resolution / high 1/A), by linear interpolation between the two
    bracketing points. Returns None if the curve never crosses it."""
    order = np.argsort(resolution_invA)
    x = resolution_invA[order]
    y = fsc[order]
    for k in range(len(x) - 1):
        if y[k] >= threshold and y[k + 1] < threshold:
            # linear interpolation in (x, y) between the two points
            frac = (threshold - y[k]) / (y[k + 1] - y[k])
            x_cross = x[k] + frac * (x[k + 1] - x[k])
            if x_cross <= 0:
                return None
            return 1.0 / x_cross  # convert 1/Angstrom -> Angstrom
    return None


def load_fsc(path):
    """Auto-detects which of the two RELION FSC table formats a file uses,
    so --i1/--i2 can each be either a postprocess.star OR a Refine3D
    run_model.star, compared directly against each other:

    - postprocess.star: block 'data_fsc', columns 'rlnResolution' /
      'rlnFourierShellCorrelationCorrected' (masked, gold-standard,
      MTF-corrected -- confirmed from src/postprocessing.cpp and two
      independent community tools, see module docstring).
    - Refine3D's run_model.star: block 'data_model_class_1', columns
      'rlnResolution' / 'rlnGoldStandardFsc' (UNMASKED half-map FSC --
      confirmed from RELION's own documented command:
      `relion_star_printtable rootname_model.star data_model_class_1
      rlnResolution rlnGoldStandardFsc`, on
      relion's "Analyse results" documentation page).

    Comparing these two directly (Refine3D's own unmasked estimate vs.
    PostProcess's masked/sharpened estimate for the SAME job) is exactly
    the diagnostic check for "why did PostProcess resolution get worse
    than refinement" -- per RELION's own docs, PostProcess is expected to
    recover resolution the deliberately-conservative unmasked refinement
    estimate under-sold, not lose resolution; a worse PostProcess number
    points at the mask (wrong map, doesn't fully enclose the structure, or
    an edge/soft-mask artifact), not at the refinement itself.
    """
    try:
        d = read_star_loop(path, "data_fsc",
                            ["rlnResolution", "rlnFourierShellCorrelationCorrected"])
        return d["rlnResolution"], d["rlnFourierShellCorrelationCorrected"], "PostProcess (masked, corrected)"
    except ValueError:
        pass

    try:
        d = read_star_loop(path, "data_model_class_1",
                            ["rlnResolution", "rlnGoldStandardFsc"])
        return d["rlnResolution"], d["rlnGoldStandardFsc"], "Refine3D (unmasked, gold-standard)"
    except ValueError:
        pass

    raise ValueError(
        f"{path}: could not find either a PostProcess-style 'data_fsc' "
        f"block (rlnFourierShellCorrelationCorrected) or a Refine3D-style "
        f"'data_model_class_1' block (rlnGoldStandardFsc). This file may "
        f"be a different RELION output type than expected -- check its "
        f"actual contents before assuming this script is wrong."
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i1", required=True, help="first postprocess.star")
    p.add_argument("--i2", required=True, help="second postprocess.star")
    p.add_argument("--label1", default=None, help="legend label for --i1 (default: its path)")
    p.add_argument("--label2", default=None, help="legend label for --i2 (default: its path)")
    p.add_argument("--o", default="fsc_comparison.png", help="output image path")
    p.add_argument("--threshold", type=float, default=0.143,
                    help="FSC threshold for the reported crossing resolution (default 0.143)")
    args = p.parse_args(argv)

    label1 = args.label1 or args.i1
    label2 = args.label2 or args.i2

    res1, fsc1, kind1 = load_fsc(args.i1)
    res2, fsc2, kind2 = load_fsc(args.i2)

    if args.label1 is None:
        label1 = f"{label1} [{kind1}]"
    if args.label2 is None:
        label2 = f"{label2} [{kind2}]"

    if kind1 != kind2:
        print(
            f"NOTE: comparing a '{kind1}' curve against a '{kind2}' curve -- "
            f"these are different FSC calculations (unmasked half-map FSC "
            f"vs. masked/MTF-corrected/sharpened FSC), not two versions of "
            f"the same measurement. A gap between them is EXPECTED and is "
            f"exactly what this comparison is for (e.g. diagnosing whether "
            f"PostProcess's masking made resolution better or worse than "
            f"Refine3D's own unmasked estimate) -- it does not by itself "
            f"mean anything is wrong."
        )

    cross1 = crossing_resolution(res1, fsc1, args.threshold)
    cross2 = crossing_resolution(res2, fsc2, args.threshold)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res1, fsc1, label=label1, linewidth=1.8)
    ax.plot(res2, fsc2, label=label2, linewidth=1.8)
    ax.axhline(args.threshold, color="grey", linestyle="--", linewidth=1,
               label=f"threshold = {args.threshold}")
    ax.set_xlabel("Resolution (1/Å)")
    ax.set_ylabel("Fourier Shell Correlation (corrected)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("FSC comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    def annotate(x_res, label):
        if x_res is None:
            print(f"{label}: FSC never drops below {args.threshold} in this data -- "
                  f"no crossing resolution to report.")
            return
        print(f"{label}: resolution at FSC = {args.threshold}  ->  {x_res:.2f} Å")
        ax.axvline(1.0 / x_res, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    annotate(cross1, label1)
    annotate(cross2, label2)

    fig.tight_layout()
    fig.savefig(args.o, dpi=200)
    print(f"Saved plot to {args.o}")


if __name__ == "__main__":
    main()
