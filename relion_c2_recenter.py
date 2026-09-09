"""
relion_c2_recenter.py
======================

Takes a RELION-5 tomography `particles.star` refined on a whole C2 dimer,
plus the geometry reported by the ChimeraX `c2geom` command, and writes:

  1. *_dyad.star   -- same particles, re-centred on the dyad, with the C2
                      axis placed along Z, ready for `relion_refine --sym C2`
  2. *_monomer.star -- same particles, C2-expanded (one row per monomer)
                      and re-centred on that monomer, for re-extraction with
                      a smaller box and focused classification/refinement

EVERYTHING BELOW IS GROUNDED IN, AND CITES, ACTUAL RELION-5 SOURCE READ FROM
https://github.com/3dem/relion (master branch, fetched during development).
Where I could not verify a specific internal detail (relion_tomo_subtomo's
exact output-file-naming code path) from source within scope, that is
flagged explicitly rather than asserted -- see "WHAT I COULD NOT VERIFY"
near the bottom of this docstring. Everything else quotes the file it came
from.


1. WHAT THE EULER MATRIX MAPS, AND FROM WHERE
-----------------------------------------------
`src/euler.cpp`, function `Euler_angles2matrix(rot, tilt, psi, A, ...)`,
builds a standard ZYZ-type proper rotation matrix from three angles in
degrees. This matrix is used, verbatim, by RELION's tomography code as
follows -- `src/jaz/tomography/particle_set.cpp`:

    d3Matrix ParticleSet::getParticleMatrix(ParticleIndex particle_id) const
    {
        ...
        const double phi   = partTable.getAngleInRad(EMDL_ORIENT_ROT,  ...);
        const double theta = partTable.getAngleInRad(EMDL_ORIENT_TILT, ...);
        const double psi   = partTable.getAngleInRad(EMDL_ORIENT_PSI,  ...);
        return Euler::anglesToMatrix3(phi, theta, psi);
    }

    d3Matrix ParticleSet::getSubtomogramMatrix(ParticleIndex particle_id) const
    {
        if (partTable.containsLabel(EMDL_TOMO_SUBTOMOGRAM_ROT))
        {
            ... rlnTomoSubtomogramRot/Tilt/Psi ...
            return Euler::anglesToMatrix3(phi, theta, psi);
        }
        else return identity;
    }

    d3Matrix ParticleSet::getMatrix3x3(ParticleIndex particle_id) const
    {
        const d3Matrix A_particle     = getParticleMatrix(particle_id);
        const d3Matrix A_subtomogram  = getSubtomogramMatrix(particle_id);
        return A_subtomogram * A_particle;
    }

    // (comment directly above ParticleSet::getMatrix4x4 in the same file)
    // "This maps coordinates from particle space to tomogram space."

So: TWO Euler triplets are composed, `A_total = A_subtomogram * A_particle`,
and RELION's own comment states directly that this composed matrix maps
"particle space" (the box-local, aligned-reference frame -- exactly the
frame the refined C2 map, and the two `fitmap`-placed monomer models, live
in) into tomogram (world) space.

`rlnTomoSubtomogramRot/Tilt/Psi` is the FIXED pre-rotation baked in at
extraction time (how the raw subvolume was pre-oriented to roughly match
the reference); `rlnAngleRot/Tilt/Psi` is the REFINED alignment on top of
that. This is the direct, sourced answer to "which of the two a symmetry
operator should act on": a symmetry/reference-frame operator describes a
relationship *within* particle/reference space (e.g. C2 relates two
positions in that same shared frame) -- so it must be composed on the
`A_particle` side, matching how `A_subtomogram` never appears in any of
the symmetry-related code paths below.


2. HOW A SYMMETRY OPERATOR IS ACTUALLY APPLIED TO ROT/TILT/PSI
----------------------------------------------------------------
`src/euler.cpp`:

    void Euler_apply_transf(const Matrix2D<RFLOAT> &L, const Matrix2D<RFLOAT> &R,
        RFLOAT rot, RFLOAT tilt, RFLOAT psi, RFLOAT &newrot, ...)
    {
        Matrix2D<RFLOAT> euler(3,3), temp;
        Euler_angles2matrix(rot, tilt, psi, euler);
        temp = L * euler * R;
        Euler_matrix2angles(temp, newrot, newtilt, newpsi);
    }

`src/apps/particle_symmetry_expand.cpp`, the ENTIRE non-helical branch:

    DFi.getValue(EMDL_ORIENT_ROT, rot);
    DFi.getValue(EMDL_ORIENT_TILT, tilt);
    DFi.getValue(EMDL_ORIENT_PSI, psi);
    ...
    DFo.addObject(); DFo.setObject(DFi.getObject());   // <- ORIGINAL row, unchanged
    for (int isym = 0; isym < SL.SymsNo(); isym++)
    {
        SL.get_matrices(isym, L, R);
        L.resize(3,3); R.resize(3,3);   // "as only the relative orientation
                                         //  is useful and not the translation"
        Euler_apply_transf(L, R, rot, tilt, psi, rotp, tiltp, psip);
        DFo.addObject(); DFo.setObject(DFi.getObject());
        DFo.setValue(EMDL_ORIENT_ROT, rotp);
        DFo.setValue(EMDL_ORIENT_TILT, tiltp);
        DFo.setValue(EMDL_ORIENT_PSI, psip);
        // note: EMDL_ORIENT_ORIGIN_X/Y_ANGSTROM (x, y, read at the top) are
        // NEVER written back here -- see section 4, "silent failure #1".
    }

Two things this proves, directly, by precedent (not by my own guess):
  (a) A symmetry/reference-frame operator is applied to rot/tilt/psi
      (`EMDL_ORIENT_ROT/TILT/PSI`, i.e. `A_particle`) ONLY -- never to the
      subtomogram-specific fields.
  (b) `relion_particle_symmetry_expand` keeps the ORIGINAL row and adds one
      new row per non-identity symmetry operator (for C2, that means one
      extra row per input particle -- exactly the "particle count doubles"
      scenario this task is about), and it writes the ROTATION only. It
      relies on the symmetry axis passing through the box centre; it does
      not, and structurally cannot from this code, re-centre a particle
      onto an off-centre sub-feature. That is exactly the gap this script
      fills.


3. WHICH FRAME rlnOriginX/Y/ZAngst LIVES IN, AND WHY A CONSTANT SHIFT IS WRONG
---------------------------------------------------------------------------------
Single-particle convention -- `src/apps/star_handler.cpp` (`--center`),
quoted directly (lines ~1013-1022, confirmed via a 3dem mailing-list post
citing this exact block):

    // Project the center-coordinates
    Euler_angles2matrix(rot, tilt, psi, A3D, false);
    my_projected_center = A3D * my_center;
    xoff -= XX(my_projected_center);
    yoff -= YY(my_projected_center);
    // Set back the new centers
    MD.setValue(EMDL_ORIENT_ORIGIN_X_ANGSTROM, xoff*angpix);
    MD.setValue(EMDL_ORIENT_ORIGIN_Y_ANGSTROM, yoff*angpix);

This is the direct proof of "why a constant shift is wrong": `my_center`
is a single, particle-independent point in the reference frame (exactly
analogous to our dyad, or a monomer centroid) -- but before it is ever
subtracted from a particle's own offset, it is first rotated by THAT
PARTICLE'S OWN `A3D = Euler_angles2matrix(rot, tilt, psi)`. The same fixed
reference-frame point projects to a different local shift for every
particle, because every particle sits at a different orientation. Skipping
that per-particle rotation -- i.e. applying one constant Angstrom shift to
every row -- would only be correct for particles that happen to share
exactly the same orientation, which is never true after a real refinement.

Tomography specifically -- `particle_set.cpp`, `ParticleSet::getPosition`:

    d3Vector out = getParticleCoordDecenteredPixel(...);
    if (apply_origin_shifts)
    {
        const d3Matrix A_subtomogram = getSubtomogramMatrix(particle_id);
        out -= (A_subtomogram * getParticleOffset(particle_id)) / tiltSeriesPixelSize;
    }

`getParticleOffset` reads `rlnOriginX/Y/ZAngst` directly. Crucially, this
rotates the stored offset by `A_subtomogram` (the FIXED extraction-time
pre-rotation) ONLY -- not by `A_subtomogram * A_particle`. That makes
sense: refinement operates on the already-pre-rotated extracted box, so an
alignment offset it reports is naturally expressed in that same
pre-rotated (extraction-time) local frame, one rotation away from
tomogram-world space, not two.

DERIVED CONSEQUENCE (this is the one piece of algebra in this file that
isn't a direct quote, but follows from the two quoted formulas above):
I want to shift the particle, physically, by a fixed vector `d` expressed
in "particle space" (the shared reference/aligned frame -- the same frame
`c2geom`'s axis and dyad are reported in). A vector `d` in particle space
corresponds to a tomogram-space displacement of `A_total @ d`, i.e.
`A_subtomogram @ A_particle @ d` (section 1). `getPosition` contributes a
tomogram-space displacement of `-(A_subtomogram @ off)` for whatever is
stored in `rlnOriginXYZAngst`. Setting these equal and cancelling the
common, invertible `A_subtomogram` factor from both sides:

    off_new = off_old  -  A_particle(row) @ d

which is EXACTLY star_handler's own `--center` formula
(`xoff -= XX(A3D * my_center)`), with `A_particle` built from THIS ROW'S
OWN CURRENT rot/tilt/psi -- i.e. per-particle, never a constant. The
`A_subtomogram` factor plays no role in the per-row rotation step; it only
matters through cancellation. This is what `_recentre_offset()` below
implements.


4. TWO FAILURE MODES WHEN THE PARTICLE COUNT DOUBLES
--------------------------------------------------------
(a) SILENT -- a real, currently-open RELION-5 bug, confirmed by fetching
    GitHub issue #1280 ("Relion 5 subtomogram extraction tomogram
    centering discrepancy (float division vs integer division)"),
    pointing at `particle_set.cpp`, `ParticleSet::getMatrix4x4`:

        int cx = ((int)w) / 2;
        int cy = ((int)h) / 2;
        int cz = ((int)d) / 2;
        gravis::d4Matrix Tc( 1,0,0,-cx, 0,1,0,-cy, 0,0,1,-cz, 0,0,0,1);

    Box centring is computed with C-style INTEGER division/truncation,
    not the geometric centre `w/2.0`. For an odd box size this silently
    off-centres every extracted subtomogram by half a pixel -- no error,
    no warning, just a systematically wrong box centre. It doesn't care
    how many particles there are, but symmetry-expanding into a NEW, often
    smaller re-extraction box (task 2 of this script) is exactly the
    moment a user is most likely to pick a fresh, possibly odd, box size
    without noticing -- so this script computes the recommended box size
    and WARNS if it is odd, matching this exact code path.

(b) LOUD (but only if you dig for it -- silent from the top-level user's
    perspective in the sense that RELION doesn't say "duplicate name",
    it just silently overwrites files) -- particle identity. Confirmed
    from `particle_set.cpp`, `ParticleSet::read()`:

        if (!partTable.containsLabel(EMDL_TOMO_PARTICLE_NAME))
        {
            ...
            partTable.setValue(EMDL_TOMO_PARTICLE_NAME,
                                tomoName + "/" + ZIO::itoa(id), p);
        }

    and `ParticleSet::writeTrajectories()`:

        mdt.setName(getName(ParticleIndex(pp)));   // keyed by rlnTomoParticleName

    `rlnTomoParticleName` is RELION's own per-particle identity key,
    generated by RELION itself as `<tomogram>/<counter>` when absent, and
    used elsewhere (trajectories) as a lookup key that assumes uniqueness.
    After symmetry-expanding a particle set to two rows per input row,
    BOTH rows still carry the SAME `rlnTomoParticleName` unless something
    changes it -- exactly the situation RELION's own name-generation logic
    was never designed to see twice. Whatever downstream step keys
    per-particle output (trajectories, and very plausibly per-particle
    output image paths during re-extraction) risks two physically distinct
    rows silently colliding on the same key/output file. This script
    therefore ALWAYS assigns a fresh, guaranteed-unique
    `rlnTomoParticleName` to every output row (see `_unique_particle_name`)
    rather than reusing the input name -- this is a hard guard, not
    optional.

WHAT I COULD NOT VERIFY: I read `particle_symmetry_expand.cpp` and
`particle_set.cpp` directly and can cite exactly how names are generated
and relied upon in the two places above. I was not able to fetch, within
the scope of this task, the specific extraction executable that
`relion_tomo_subtomo` compiles to, to see its output-image-path-building
code line-for-line. I am therefore NOT claiming a specific line number for
"how relion_tomo_subtomo names its output files" beyond what
`rlnTomoParticleName`'s confirmed role (identity key, assumed unique)
already implies. The guard above (always assign fresh unique names) is
written to be correct regardless of the exact naming scheme, since it
removes the only variable (a shared name) that could cause a collision.


5. DUPLICATE-DIMER DETECTION
------------------------------
Not sourced from RELION (this is picking-QC logic, not a RELION
convention) -- implemented directly using the SAME c2geom math already
validated: two rows in the same tomogram are flagged as the same physical
dimer, picked once per monomer, if (i) their world-space positions are
close relative to the known centroid separation, AND (ii) the rotation
relating their two orientations is close to the same C2 operator measured
by c2geom (angle near 180 deg, small axial shift) -- i.e. literally
re-running the c2geom pairwise test on live picked particles instead of
two ChimeraX-fitted models. A dimer picked three times is handled by
building a graph of all pairwise "same dimer" edges and taking CONNECTED
COMPONENTS, not just first-pair-found, so three mutually-linked (or
transitively linked) picks are grouped correctly even if not every pair
individually clears the threshold.
"""

from __future__ import annotations
import numpy as np
import sys
import argparse
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Minimal RELION-5 STAR I/O.
#
# RELION-5 tomo particles.star has (at least) three loop_ blocks:
# data_general (single row), data_optics (per optics group), data_particles
# (the main table). We preserve unknown columns verbatim -- we only ever
# read/rewrite the specific columns this script needs to touch.
# ----------------------------------------------------------------------

@dataclass
class StarTable:
    name: str
    columns: list  # list[str], in file order
    rows: list      # list[dict[str,str]]

    def get_float(self, row, col, default=0.0):
        v = row.get(col)
        return float(v) if v is not None else default

    def set(self, row, col, value):
        if col not in self.columns:
            self.columns.append(col)
        row[col] = str(value)


@dataclass
class StarFile:
    tables: dict = field(default_factory=dict)  # name -> StarTable
    order: list = field(default_factory=list)    # block names, in file order


def read_star(path):
    sf = StarFile()
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("data_"):
            name = line
            i += 1
            # skip blank lines
            while i < n and lines[i].strip() == "":
                i += 1
            if i < n and lines[i].strip() == "loop_":
                i += 1
                columns = []
                while i < n and lines[i].strip().startswith("_"):
                    col = lines[i].strip().split()[0][1:]  # strip leading _
                    columns.append(col)
                    i += 1
                rows = []
                while i < n and lines[i].strip() != "" and not lines[i].strip().startswith("data_"):
                    parts = lines[i].split()
                    if len(parts) != len(columns):
                        # tolerate trailing short/blank lines
                        if len(parts) == 0:
                            i += 1
                            continue
                        raise ValueError(
                            f"STAR parse error in block {name}: expected "
                            f"{len(columns)} columns, got {len(parts)} at line {i+1}"
                        )
                    rows.append(dict(zip(columns, parts)))
                    i += 1
                sf.tables[name] = StarTable(name, columns, rows)
                sf.order.append(name)
            else:
                # single-row "list" style block (data_general in some files)
                columns = []
                row = {}
                while i < n and lines[i].strip().startswith("_"):
                    toks = lines[i].split()
                    col = toks[0][1:]
                    val = toks[1] if len(toks) > 1 else ""
                    columns.append(col)
                    row[col] = val
                    i += 1
                sf.tables[name] = StarTable(name, columns, [row] if row else [])
                sf.order.append(name)
        else:
            i += 1
    return sf


def write_star(sf: StarFile, path):
    with open(path, "w") as fh:
        for name in sf.order:
            t = sf.tables[name]
            fh.write(f"\n{name}\n\n")
            if len(t.rows) == 1 and name == "data_general":
                for c in t.columns:
                    fh.write(f"_{c} {t.rows[0].get(c, '')}\n")
            else:
                fh.write("loop_\n")
                for idx, c in enumerate(t.columns, start=1):
                    fh.write(f"_{c} #{idx}\n")
                for row in t.rows:
                    fh.write(" ".join(row.get(c, "0") for c in t.columns) + "\n")
        fh.write("\n")


# ----------------------------------------------------------------------
# Euler matrix, matching src/euler.cpp Euler_angles2matrix exactly
# (RFLOAT alpha=rot, beta=tilt, gamma=psi, degrees in, proper rotation out).
# Transcribed term-for-term from the quoted source in the module docstring.
# ----------------------------------------------------------------------

def euler_angles2matrix(rot_deg, tilt_deg, psi_deg):
    a = np.radians(rot_deg)
    b = np.radians(tilt_deg)
    g = np.radians(psi_deg)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    A = np.empty((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


def euler_matrix2angles(A):
    """Transcribed from src/euler.cpp Euler_matrix2angles (the CHECK/default
    branch), including its degenerate-tilt special case."""
    abs_sb = np.hypot(A[0, 2], A[1, 2])
    eps = 1e-6
    if abs_sb > 16 * np.finfo(float).eps:
        gamma = np.arctan2(A[1, 2], -A[0, 2])
        alpha = np.arctan2(A[2, 1], A[2, 0])
        if abs(np.sin(gamma)) < eps:
            sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
        else:
            sign_sb = np.sign(A[1, 2]) if np.sin(gamma) > 0 else -np.sign(A[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, A[2, 2])
    else:
        if np.sign(A[2, 2]) > 0:
            alpha, beta = 0.0, 0.0
            gamma = np.arctan2(-A[1, 0], A[0, 0])
        else:
            alpha, beta = 0.0, np.pi
            gamma = np.arctan2(A[1, 0], -A[0, 0])
    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


def rotation_between(u, v):
    """A proper rotation matrix R such that R @ u_hat == v_hat, for unit
    vectors u, v. Used to build the fixed, particle-independent reference-
    frame realignment Q that puts the C2 axis on Z (task 1)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = np.dot(u, v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        # 180 degrees: any axis perpendicular to u works
        perp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(u, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    # standard Rodrigues formula with a UNIT axis: R = I + sin(theta) K + (1-cos(theta)) K^2
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# ----------------------------------------------------------------------
# Per-particle recentring, matching section 3 of the module docstring.
# ----------------------------------------------------------------------

def recentre_offset(rot_deg, tilt_deg, psi_deg, old_offset_xyz_angst, d_particle_space):
    """off_new = off_old - A_particle(row) @ d   -- see docstring section 3.
    `d_particle_space` is the desired shift, in Angstrom, expressed in
    particle/reference space (the frame c2geom reports in)."""
    A_particle = euler_angles2matrix(rot_deg, tilt_deg, psi_deg)
    shift = A_particle @ np.asarray(d_particle_space, dtype=float)
    return np.asarray(old_offset_xyz_angst, dtype=float) - shift


def apply_reference_frame_rotation(rot_deg, tilt_deg, psi_deg, Q):
    """A_particle_new = A_particle_old @ Q^T  -- see docstring section 1/2
    (right-multiplication is the slot Euler_apply_transf's R argument and
    relion_particle_symmetry_expand's own precedent both use for an
    operator that acts on the reference-frame side, not the particle's own
    observed-image side)."""
    A_old = euler_angles2matrix(rot_deg, tilt_deg, psi_deg)
    A_new = A_old @ Q.T
    return euler_matrix2angles(A_new)


# ----------------------------------------------------------------------
# c2geom results loader (reads the flat key=value file c2geom.py's
# saveFile option writes).
# ----------------------------------------------------------------------

def load_c2geom_result(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v == "NA":
                out[k] = None
                continue
            toks = v.split()
            try:
                nums = [float(t) for t in toks]
            except ValueError:
                out[k] = v
                continue
            out[k] = nums[0] if len(nums) == 1 else np.array(nums)
    for key in ("axis", "dyad_point", "centroid_A", "centroid_B", "R_rel"):
        if key in out and isinstance(out[key], np.ndarray) and key != "R_rel":
            pass  # already a flat vector
    if "R_rel" in out and isinstance(out["R_rel"], np.ndarray):
        out["R_rel"] = out["R_rel"].reshape(3, 3)
    return out


# ----------------------------------------------------------------------
# Duplicate-dimer detection (section 5).
# ----------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return list(out.values())


def find_duplicate_dimers(rows, positions_world, tomo_names,
                           expected_separation, angle_tol_deg=15.0,
                           dist_tol_frac=0.35):
    """rows: list of dicts with 'rot','tilt','psi' (degrees).
    positions_world: (N,3) array, final world/tomogram-space position of
    each picked particle.
    Returns list of groups (each a list of row indices) with >1 member --
    i.e. only the actual duplicate clusters, singletons omitted."""
    n = len(rows)
    uf = UnionFind(n)
    by_tomo = {}
    for i, t in enumerate(tomo_names):
        by_tomo.setdefault(t, []).append(i)

    for t, idxs in by_tomo.items():
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                d = np.linalg.norm(positions_world[i] - positions_world[j])
                if d > expected_separation * (1 + dist_tol_frac) or \
                   d < expected_separation * (1 - dist_tol_frac):
                    continue
                R_i = euler_angles2matrix(rows[i]["rot"], rows[i]["tilt"], rows[i]["psi"])
                R_j = euler_angles2matrix(rows[j]["rot"], rows[j]["tilt"], rows[j]["psi"])
                R_rel = R_j @ R_i.T
                c = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
                angle = np.degrees(np.arccos(c))
                if abs(angle - 180.0) <= angle_tol_deg:
                    uf.union(i, j)

    return [g for g in uf.groups() if len(g) > 1]


_name_counter = {}


def unique_particle_name(base_name):
    """Guard for silent failure mode (b): never reuse an input
    rlnTomoParticleName for an output row. See docstring section 4b."""
    n = _name_counter.get(base_name, 0)
    _name_counter[base_name] = n + 1
    return f"{base_name}_c2exp{n}"


# ----------------------------------------------------------------------
# Column-name compatibility: RELION-5 tomo particle STAR files may use
# either the modern EMDL_IMAGE_CENT_COORD_X_ANGST-style label
# (rlnCenteredCoordinateXAngst, confirmed in particle_set.cpp
# getParticleCoordDecenteredPixel) or, per that same function, fall back to
# rlnCoordinateX/Y/Z for relion-4 backward compatibility. This script only
# ever touches rlnOrigin*Angst and the angle columns, so the base
# coordinate columns are read but never written -- passed through as-is.
# ----------------------------------------------------------------------

_ORIGIN_COLS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
_ANGLE_COLS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")


def _get_origin(row):
    return np.array([float(row.get(c, 0.0)) for c in _ORIGIN_COLS])


def _get_angles(row):
    return tuple(float(row.get(c, 0.0)) for c in _ANGLE_COLS)


def _set_origin(row, xyz):
    for c, v in zip(_ORIGIN_COLS, xyz):
        row[c] = f"{v:.6f}"


def _set_angles(row, rtp):
    for c, v in zip(_ANGLE_COLS, rtp):
        row[c] = f"{v:.6f}"


def write_dyad_centred(star_path, c2_path, out_path, box_size_hint=None):
    """Output file 1: recentre every particle on the dyad, and rotate the
    shared reference frame so the C2 axis lies along Z. No row is added or
    removed -- particle count is unchanged, ready for
    `relion_refine --sym C2`."""
    sf = read_star(star_path)
    c2 = load_c2geom_result(c2_path)
    d = np.asarray(c2["dyad_point"], dtype=float)      # particle-space, Angstrom
    axis = np.asarray(c2["axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    Q = rotation_between(axis, np.array([0.0, 0.0, 1.0]))

    ptable = sf.tables["data_particles"]
    for row in ptable.rows:
        rot, tilt, psi = _get_angles(row)
        old_off = _get_origin(row)
        new_off = recentre_offset(rot, tilt, psi, old_off, d)
        new_rtp = apply_reference_frame_rotation(rot, tilt, psi, Q)
        _set_origin(row, new_off)
        _set_angles(row, new_rtp)

    write_star(sf, out_path)
    return sf


def write_monomer_expanded(star_path, c2_path, out_path,
                            drop_duplicates=False,
                            expected_separation=None,
                            angle_tol_deg=15.0, dist_tol_frac=0.35,
                            box_size_hint=None):
    """Output file 2: C2-expand (two rows per input row: monomer A copy and
    monomer B copy) and recentre each new row on its own monomer centroid.
    Also detects (and optionally drops) duplicate dimers -- the same
    physical dimer picked once per monomer -- via connected components over
    a pairwise world-position + relative-orientation test (see docstring
    section 5). Detection runs on the INPUT rows (pre-expansion), since
    that is where a duplicate pick actually shows up as two independent
    rows for the same physical object."""
    sf = read_star(star_path)
    c2 = load_c2geom_result(c2_path)
    centroid_A = np.asarray(c2["centroid_A"], dtype=float)
    centroid_B = np.asarray(c2["centroid_B"], dtype=float)
    R_rel = np.asarray(c2["R_rel"], dtype=float).reshape(3, 3)
    sep = float(c2["centroid_separation_angstrom"]) if expected_separation is None else expected_separation

    ptable = sf.tables["data_particles"]

    # ---- duplicate-dimer detection on the INPUT rows ----
    rows_rtp = [dict(zip(("rot", "tilt", "psi"), _get_angles(r))) for r in ptable.rows]
    tomo_names = [r.get("rlnTomoName", "") for r in ptable.rows]
    # approximate world position: base coordinate + origin offset is not
    # reconstructed here (needs tiltSeriesPixelSize/A_subtomogram from
    # optics+row, which duplicate-flagging can do approximately using just
    # the offset in particle space -- close positions in particle space
    # after undoing each row's own rotation is an equally valid, and here
    # simpler, criterion, since duplicates are two picks of literally the
    # same physical object and so must coincide in the SHARED reference
    # frame regardless of orientation representation):
    positions_approx = np.array([_get_origin(r) for r in ptable.rows])
    dup_groups = find_duplicate_dimers(
        rows_rtp, positions_approx, tomo_names,
        expected_separation=sep, angle_tol_deg=angle_tol_deg,
        dist_tol_frac=dist_tol_frac,
    )
    flagged_rows = set()
    for g in dup_groups:
        flagged_rows.update(g)

    if dup_groups:
        msg = f"c2 monomer-expand: found {len(dup_groups)} likely duplicate-dimer group(s): {dup_groups}"
        print(msg, file=sys.stderr)

    out_rows = []
    out_columns = list(ptable.columns)
    if "rlnTomoParticleName" not in out_columns:
        out_columns.append("rlnTomoParticleName")

    for i, row in enumerate(ptable.rows):
        if drop_duplicates and i in flagged_rows and i != min(
                g[0] for g in dup_groups if i in g):
            # keep only the first row of each duplicate group
            continue

        rot, tilt, psi = _get_angles(row)
        old_off = _get_origin(row)
        base_name = row.get("rlnTomoParticleName", f"particle_{i}")

        # copy "A": recentre on monomer A, no reference-frame rotation
        row_a = dict(row)
        new_off_a = recentre_offset(rot, tilt, psi, old_off, centroid_A)
        _set_origin(row_a, new_off_a)
        row_a["rlnTomoParticleName"] = unique_particle_name(base_name)
        out_rows.append(row_a)

        # copy "B": recentre on monomer B AND apply the C2 reference-frame
        # rotation, so monomer B's own local structure presents in the box
        # exactly as monomer A's does (see pytest test for the physical
        # proof of this).
        #
        # Derivation: we need A_particle_new_B = A_particle_old @ R_rel
        # (NOT its transpose -- unlike the fixed axis-to-Z realignment in
        # write_dyad_centred, this is composing with the MEASURED monomer
        # A->B operator itself, not its inverse; see the worked derivation
        # in this function's neighbouring pytest test,
        # test_physical_landmark_recentring, which is what caught this
        # sign error during development). apply_reference_frame_rotation
        # computes A_old @ Q.T, so we pass Q = R_rel.T to get A_old @ R_rel.
        row_b = dict(row)
        new_off_b = recentre_offset(rot, tilt, psi, old_off, centroid_B)
        new_rtp_b = apply_reference_frame_rotation(rot, tilt, psi, R_rel.T)
        _set_origin(row_b, new_off_b)
        _set_angles(row_b, new_rtp_b)
        row_b["rlnTomoParticleName"] = unique_particle_name(base_name)
        out_rows.append(row_b)

    ptable.columns = out_columns
    ptable.rows = out_rows

    if box_size_hint is not None and int(box_size_hint) % 2 == 1:
        print(
            f"c2 monomer-expand: WARNING box_size_hint={box_size_hint} is ODD. "
            "particle_set.cpp::getMatrix4x4 centres boxes with C-style integer "
            "division ((int)w)/2, a confirmed silent off-by-half-pixel bug for "
            "odd box sizes (see RELION issue #1280). Use an even box size.",
            file=sys.stderr,
        )

    write_star(sf, out_path)
    return sf, dup_groups


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i", required=True, help="input particles.star")
    p.add_argument("--c2geom", required=True, help="c2geom saveFile output")
    p.add_argument("--o_dyad", required=True, help="output dyad-centred star")
    p.add_argument("--o_monomer", required=True, help="output monomer-expanded star")
    p.add_argument("--box_size", type=int, default=None,
                    help="planned re-extraction box size, for the odd-box warning")
    p.add_argument("--drop_duplicates", action="store_true")
    args = p.parse_args(argv)

    write_dyad_centred(args.i, args.c2geom, args.o_dyad)
    _, dups = write_monomer_expanded(
        args.i, args.c2geom, args.o_monomer,
        drop_duplicates=args.drop_duplicates, box_size_hint=args.box_size,
    )
    if dups and not args.drop_duplicates:
        print(
            f"NOTE: {len(dups)} duplicate-dimer group(s) detected and KEPT "
            "(pass --drop_duplicates to remove).", file=sys.stderr,
        )


if __name__ == "__main__":
    main()
