"""
check_ab_class_agreement.py
=============================

For the RsmA expanded-monomer no-alignment Class3D run: checks, for every
original dimer, whether its monomer-A copy and monomer-B copy landed in
the SAME class or DIFFERENT classes.

WHY THIS TEST, AND HOW TO READ IT
------------------------------------
Every row `relion_c2_recenter.py`'s `write_monomer_expanded()` writes gets
a unique `rlnTomoParticleName` via `unique_particle_name()`:
    "<original_name>_c2exp0"   <- monomer A copy
    "<original_name>_c2exp1"   <- monomer B copy
(see relion_c2_recenter.py, `_name_counter`/`unique_particle_name`). Both
copies share the same "<original_name>" prefix, which is what lets this
script re-pair them after classification.

If the two monomers within a dimer are genuinely structurally equivalent
(the dimer is symmetric all the way down to content, not just geometry),
A's and B's class assignments should be statistically INDEPENDENT of one
another -- any apparent pattern would just be classification noise. If
instead there is real content asymmetry between the two monomer positions
(e.g. one systematically differs from the other -- a bound factor, a
distinct conformational state), A's and B's classes should be correlated
(or anti-correlated) more than chance predicts.

This script does NOT just eyeball "same class / different class" --  it
builds the full A-class x B-class contingency table and runs a chi-square
test of independence, since a small excess of same-class or
different-class pairs can easily happen by chance with only a few classes
and a moderate dimer count; the test tells you whether the pattern seen is
actually statistically meaningful.

USAGE
-----
    python3 check_ab_class_agreement.py \\
        --i Class3D/jobXXX/run_it025_data.star \\
        --o ab_class_pairs.csv

`--i` is the Class3D job's own particle output star file (must contain
both rlnTomoParticleName and rlnClassNumber). `--o` is optional -- a CSV
of every matched dimer's (A class, B class) pair, for your own further
analysis/plotting.
"""

from __future__ import annotations
import argparse
import sys
from collections import defaultdict

try:
    from relion_c2_recenter import read_star
except ImportError:
    print(
        "Could not import read_star from relion_c2_recenter.py -- make "
        "sure this script sits in the same folder as relion_c2_recenter.py, "
        "or add that folder to PYTHONPATH.",
        file=sys.stderr,
    )
    raise


def split_name(rln_tomo_particle_name):
    """Returns (base_name, 'A' or 'B') for a name of the form
    '<base>_c2exp0' / '<base>_c2exp1', or (None, None) if it doesn't match
    that pattern (e.g. this row was never part of a C2 expansion)."""
    for suffix, tag in (("_c2exp0", "A"), ("_c2exp1", "B")):
        if rln_tomo_particle_name.endswith(suffix):
            return rln_tomo_particle_name[: -len(suffix)], tag
    return None, None


def load_ab_classes(star_path):
    sf = read_star(star_path)
    ptable = sf.tables["data_particles"]
    if "rlnTomoParticleName" not in ptable.columns:
        raise ValueError(f"{star_path}: no rlnTomoParticleName column found")
    if "rlnClassNumber" not in ptable.columns:
        raise ValueError(
            f"{star_path}: no rlnClassNumber column found -- is this "
            f"actually a Class3D output star file (e.g. run_itXXX_data.star), "
            f"not the input particles.star?"
        )

    pairs = defaultdict(dict)  # base_name -> {'A': class, 'B': class}
    unmatched = 0
    for row in ptable.rows:
        base, tag = split_name(row["rlnTomoParticleName"])
        if base is None:
            unmatched += 1
            continue
        pairs[base][tag] = int(row["rlnClassNumber"])

    complete = {b: v for b, v in pairs.items() if "A" in v and "B" in v}
    incomplete = len(pairs) - len(complete)
    return complete, unmatched, incomplete


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i", required=True, help="Class3D output particles star file")
    p.add_argument("--o", default=None, help="optional CSV of per-dimer (A class, B class) pairs")
    args = p.parse_args(argv)

    pairs, unmatched, incomplete = load_ab_classes(args.i)

    if unmatched:
        print(f"NOTE: {unmatched} row(s) did not match the '_c2exp0'/'_c2exp1' "
              f"naming pattern and were skipped (not part of a C2 expansion).")
    if incomplete:
        print(f"NOTE: {incomplete} dimer(s) had only one of their two monomer "
              f"copies present in this file (the partner may have been "
              f"removed earlier, e.g. by duplicate-dimer filtering or a "
              f"prior particle-selection step) and were excluded from the "
              f"comparison below.")

    n = len(pairs)
    if n == 0:
        print("No complete A/B pairs found -- nothing to compare.")
        return

    classes_A = sorted({v["A"] for v in pairs.values()})
    classes_B = sorted({v["B"] for v in pairs.values()})
    all_classes = sorted(set(classes_A) | set(classes_B))

    table = {a: {b: 0 for b in all_classes} for a in all_classes}
    same_class = 0
    for v in pairs.values():
        table[v["A"]][v["B"]] += 1
        if v["A"] == v["B"]:
            same_class += 1

    print(f"\nComplete A/B dimer pairs compared: {n}")
    print(f"Same class (A == B):     {same_class}  ({100*same_class/n:.1f}%)")
    print(f"Different class (A != B): {n - same_class}  ({100*(n-same_class)/n:.1f}%)\n")

    print("Contingency table (rows = monomer A's class, columns = monomer B's class):")
    header = "        " + "".join(f"B={b:<6}" for b in all_classes)
    print(header)
    for a in all_classes:
        row = "".join(f"{table[a][b]:<8}" for b in all_classes)
        print(f"A={a:<5} {row}")
    print()

    import numpy as np
    matrix = np.array([[table[a][b] for b in all_classes] for a in all_classes])

    try:
        from scipy.stats import chi2_contingency
        chi2, pval, dof, expected = chi2_contingency(matrix)
        print(f"Chi-square test of independence: chi2 = {chi2:.3f}, dof = {dof}, p = {pval:.4g}")
        if pval < 0.05:
            print(
                "p < 0.05: A's and B's class assignments are NOT independent -- "
                "there IS a statistically significant relationship between which "
                "class a dimer's two monomers end up in. This is worth following "
                "up (e.g. with difference maps between classes) as possible "
                "evidence of real content asymmetry -- or symmetry stronger than "
                "chance -- between the two monomer positions within each dimer."
            )
        else:
            print(
                "p >= 0.05: no significant evidence that A's and B's class "
                "assignments are related -- consistent with the two monomers "
                "being structurally equivalent at the level this classification "
                "can detect (this does not PROVE equivalence, only that this "
                "test found no evidence against it)."
            )
    except ImportError:
        print(
            "scipy not available -- install it (`pip install scipy "
            "--break-system-packages`) to get the chi-square significance test. "
            "The raw contingency table above is still valid without it."
        )

    if args.o:
        with open(args.o, "w") as fh:
            fh.write("dimer_base_name,class_A,class_B,same_class\n")
            for base, v in pairs.items():
                fh.write(f"{base},{v['A']},{v['B']},{int(v['A']==v['B'])}\n")
        print(f"\nPer-dimer pairs written to {args.o}")


if __name__ == "__main__":
    main()
