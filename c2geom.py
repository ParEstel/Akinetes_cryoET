"""
c2geom.py — measure C2 dimer geometry from two independently fitmap-fitted
copies of the same monomer model, sitting in a shared density map.

WHAT THIS ASSUMES
------------------
- `monomerA` and `monomerB` are two independent openings of the SAME
  reference PDB/mmCIF, each moved into place by `fitmap` (or by hand) inside
  the same map. Their local (per-model) atom coordinates are therefore
  identical; only their `scene_position` placements differ.
- The relationship between the two placements is being tested for whether it
  looks like a clean 2-fold (C2) point-group symmetry: a single ~180 deg
  rotation about some axis, with no translation ("shift") along that axis.
  It is not assumed to be exactly 180 deg — that is measured, not asserted.

MATH, DERIVED
-------------
A rigid placement in ChimeraX (`model.scene_position`) is an affine map
    P(x) = R x + t
from the model's own local atom coordinates x into scene coordinates,
with R a 3x3 proper rotation and t a 3-vector.

If the two fits are truly related by some rigid motion, that motion is the
transform that turns "a point already placed via A" into "the same local
point, but placed via B":
    T = P_B  o  P_A^-1
      T(x) = R x + t ,   R = R_B R_A^T ,   t = t_B - R t_A
(This composition is exact, given the two ChimeraX placements; it does not
by itself prove the relation is a clean 180 deg rotation -- that is what the
angle/axis/shift diagnostics below actually test.)

Axis direction (works at exactly 180 deg)
    The usual axis-from-skew-part formula, axis ~ (R - R^T), is built from
    the sin(theta) term in Rodrigues' formula and is IDENTICALLY ZERO at
    theta = 180 deg (sin 180 = 0) -- exactly where a C2 axis always sits.
    Using the symmetric part instead avoids this:
        S = (R + R^T)/2 = cos(theta) I + (1-cos theta) n n^T
    n (the axis) is an eigenvector of S with eigenvalue exactly 1, at every
    theta, including 180. We pick the eigenvector whose eigenvalue is
    closest to 1. Sign is arbitrary (a 2-fold axis has no intrinsic
    direction) and is canonicalised for reproducible reporting.

Dyad point (derived, not guessed)
    A rotation by theta about axis n through point p obeys, for all x:
        T(x) = p + R(x - p) = R x + (I - R) p
    so comparing to T(x) = R x + t, ANY point p solving
        (I - R) p = t                                         (*)
    lies on the rotation axis by the very definition of "rotation about an
    axis through p" -- this is the proof, not an assumption. At theta = 180,
    (I - R) is singular (its null space is the axis direction itself), so
    (*) has a 1-parameter family of solutions -- correct, since a line has
    no single point. We solve (*) by least squares (never by direct
    inversion, which is exactly what is ill-conditioned near 180 deg), then
    slide the minimum-norm solution along n to the point closest to a
    physically meaningful reference (the midpoint of the two fitted
    centroids), and report the residual of (*) at that point as a numerical
    proof that it really is on the axis.

Shift along axis (screw component)
    Any rigid motion decomposes into a rotation about an axis through some
    point, plus a translation along that same axis (a "screw" motion). The
    axial translation is exactly t . n (dot product with the unit axis),
    independent of where p is chosen along the line. For a true point-group
    C2 this must be ~0 Angstrom; a nonzero value means the two placements
    are related by a screw, not a point symmetry -- i.e. NOT what a rigid
    homodimer interface should look like.

Coordinate frame
    Everything is computed in SCENE coordinates (`model.scene_position`,
    `atoms.scene_coords`, and the map's own `volume.scene_position`), the
    one frame in which the map and both independently-moved models are
    simultaneously valid as currently displayed. This is deliberately robust
    to the map or either model having been separately moved/rotated after
    fitting.

Isotropic-voxel assumption
    The ONLY place voxel size is used is an optional "value in voxels"
    convenience print-out for the separation/dyad numbers, which only makes
    sense as a single scalar if voxel size is the same in x, y, z. This is
    checked explicitly; if the map is anisotropic that conversion is skipped
    and only Angstrom values are reported. Nothing else in this script
    depends on the map being cubic/isotropic.

Diagnostics: what actually tells you a fit landed wrong
    Because T = P_B o P_A^-1 is an EXACT algebraic composition, "does
    applying T to A's atoms reproduce B's atoms" is tautologically always
    true (to floating point) and is NOT a useful check -- it is guaranteed
    by construction regardless of whether the true relationship is a clean
    C2 or garbage. The genuinely informative, non-circular checks are:
      (1) how close the measured rotation ANGLE is to 180 deg
      (2) how close the axial SHIFT is to 0 Angstrom (point vs. screw)
      (3) whether the composed R is close to a proper rotation at all
          (orthogonality drift, reported directly)
      (4) an INDEPENDENT per-monomer check against the map itself: how well
          each monomer's own atoms sit in real density, computed separately
          for A and B, which does not depend on the A<->B relationship at
          all and so can catch "monomer B individually landed in the wrong
          place" even when the A-vs-B algebra above looks internally
          consistent.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Pure math, no ChimeraX dependency -- usable standalone / unit-testable,
# and importable from a plain "python c2geom.py"-style downstream script.
# ----------------------------------------------------------------------

def place_to_Rt(matrix_3x4):
    """Split a ChimeraX Place.matrix (3x4 array) into rotation R (3x3) and
    translation t (3,)."""
    m = np.asarray(matrix_3x4, dtype=float)
    return m[:, :3].copy(), m[:, 3].copy()


def nearest_rotation(M):
    """Orthogonal (polar) decomposition: the proper rotation matrix closest
    to M in Frobenius norm. Used to clean up tiny numerical drift in the
    composed transform before trusting its angle/axis."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def compose_B_after_Ainverse(R_A, t_A, R_B, t_B):
    """T = P_B o P_A^-1 : the rigid transform that turns 'placed via A' into
    'placed via B'. See module docstring for the derivation."""
    R_rel = R_B @ R_A.T
    t_rel = t_B - R_rel @ t_A
    return R_rel, t_rel


def rotation_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return np.degrees(np.arccos(c))


def rotation_axis(R):
    """Axis via the eigenvector (eigenvalue closest to +1) of the SYMMETRIC
    part S = (R + R^T)/2. Non-degenerate at theta = 180 deg; see module
    docstring. Returns (unit_axis, sorted_eigenvalues) -- the eigenvalues
    are themselves a diagnostic: for a clean rotation they should look like
    (1, cos theta, cos theta), i.e. two of them near -1 when theta ~ 180."""
    S = 0.5 * (R + R.T)
    w, v = np.linalg.eigh(S)  # ascending eigenvalues, orthonormal eigvecs
    idx = int(np.argmin(np.abs(w - 1.0)))
    axis = v[:, idx]
    axis = axis / np.linalg.norm(axis)
    # canonical sign: largest-magnitude component positive (axis direction
    # is not physically meaningful for a 2-fold -- this just makes repeated
    # runs reproducible)
    i = int(np.argmax(np.abs(axis)))
    if axis[i] < 0:
        axis = -axis
    return axis, w


def shift_along_axis(t, axis):
    """Axial (screw) component of translation -- should be ~0 for a true
    point-group C2. See module docstring."""
    return float(np.dot(t, axis))


def dyad_point(R, t, axis, reference_point):
    """Solve (I - R) p = t for a point p on the rotation axis (derivation in
    module docstring), then slide along the axis to the point nearest
    `reference_point`.

    Returns (point, implementation_check_residual). NOTE on that residual:
    (I - R) has the axis direction in its null space, so (I - R) can never
    reproduce any component of t that lies ALONG the axis -- for any point
    p on the line, ||(I-R)p - t|| is identically equal to |axial_shift|,
    not to zero. That is a fact about the geometry (a screw's axial part
    can't be removed by choosing p), not a numerical error, so it is NOT a
    useful precision check by itself. The genuine implementation self-check
    is whether (I-R)p reproduces the PERPENDICULAR part of t, which should
    be ~0 to machine precision regardless of how large the axial shift is;
    that is what is returned here.
    """
    A = np.eye(3) - R
    p0, *_ = np.linalg.lstsq(A, t, rcond=None)  # min-norm solution; lstsq
    # (not a direct inverse) is what stays well-behaved as A -> singular
    # near theta = 180 deg.
    p = p0 + axis * float(np.dot(reference_point - p0, axis))
    t_perp = t - float(np.dot(t, axis)) * axis
    perp_residual = float(np.linalg.norm(A @ p - t_perp))  # ~0 => lstsq/axis
    # step implemented correctly; independent of any real screw component.
    return p, perp_residual


def point_axis_distance(point, dyad, axis):
    d = point - dyad
    return float(np.linalg.norm(d - np.dot(d, axis) * axis))


def trilinear_sample(data_kji, ijk_points):
    """Manual trilinear interpolation of a (nk, nj, ni)-shaped map array at
    fractional grid points given as an (N, 3) array of (i, j, k). No SciPy
    dependency, so this does not rely on what happens to be bundled with a
    given ChimeraX Python build."""
    ijk = np.asarray(ijk_points, dtype=float)
    ni, nj, nk = data_kji.shape[2], data_kji.shape[1], data_kji.shape[0]
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    i0 = np.clip(np.floor(i).astype(int), 0, ni - 2 if ni > 1 else 0)
    j0 = np.clip(np.floor(j).astype(int), 0, nj - 2 if nj > 1 else 0)
    k0 = np.clip(np.floor(k).astype(int), 0, nk - 2 if nk > 1 else 0)
    i1, j1, k1 = np.minimum(i0 + 1, ni - 1), np.minimum(j0 + 1, nj - 1), np.minimum(k0 + 1, nk - 1)
    fi = np.clip(i - i0, 0.0, 1.0)
    fj = np.clip(j - j0, 0.0, 1.0)
    fk = np.clip(k - k0, 0.0, 1.0)

    def g(kk, jj, ii):
        return data_kji[kk, jj, ii]

    c000 = g(k0, j0, i0); c100 = g(k0, j0, i1)
    c010 = g(k0, j1, i0); c110 = g(k0, j1, i1)
    c001 = g(k1, j0, i0); c101 = g(k1, j0, i1)
    c011 = g(k1, j1, i0); c111 = g(k1, j1, i1)

    c00 = c000 * (1 - fi) + c100 * fi
    c10 = c010 * (1 - fi) + c110 * fi
    c01 = c001 * (1 - fi) + c101 * fi
    c11 = c011 * (1 - fi) + c111 * fi
    c0 = c00 * (1 - fj) + c10 * fj
    c1 = c01 * (1 - fj) + c11 * fj
    return c0 * (1 - fk) + c1 * fk


def c2geom_core(R_A, t_A, R_B, t_B, centroid_A, centroid_B):
    """The whole geometric calculation, given nothing but two placements
    (R, t) and two centroids -- no ChimeraX objects required. This is the
    'plain python' half of c2geom: importable and callable outside a
    ChimeraX session (e.g. from a downstream analysis script that has
    logged R_A/t_A/R_B/t_B from an earlier ChimeraX run).

    Returns a dict with everything a downstream script needs.
    """
    R_rel_raw, t_rel = compose_B_after_Ainverse(R_A, t_A, R_B, t_B)
    R_rel = nearest_rotation(R_rel_raw)
    orthogonality_drift = float(np.linalg.norm(R_rel_raw - R_rel))

    angle = rotation_angle_deg(R_rel)
    axis, sym_eigvals = rotation_axis(R_rel)
    axial_shift = shift_along_axis(t_rel, axis)

    midpoint = 0.5 * (np.asarray(centroid_A) + np.asarray(centroid_B))
    dyad, dyad_residual = dyad_point(R_rel, t_rel, axis, midpoint)

    separation = float(np.linalg.norm(np.asarray(centroid_B) - np.asarray(centroid_A)))

    v = np.asarray(centroid_B) - np.asarray(centroid_A)
    vn = np.linalg.norm(v)
    if vn > 1e-9:
        ang_v_axis = np.degrees(np.arccos(np.clip(np.dot(v / vn, axis), -1.0, 1.0)))
        # arccos already returns [0,180]; deviation-from-perpendicular is
        # |90 - angle| and this is invariant to the axis sign flip
        # (ang -> 180-ang leaves |90-ang| unchanged), so no extra folding
        # is needed -- an earlier version of this had a spurious double
        # fold that produced 90 instead of 0 for a perfect right angle.
        perp_deviation_deg = float(abs(90.0 - ang_v_axis))
    else:
        perp_deviation_deg = float("nan")

    dist_A = point_axis_distance(np.asarray(centroid_A), dyad, axis)
    dist_B = point_axis_distance(np.asarray(centroid_B), dyad, axis)

    return {
        "R_rel": R_rel,
        "t_rel": t_rel,
        "orthogonality_drift": orthogonality_drift,
        "rotation_angle_deg": angle,
        "angle_deviation_from_180_deg": float(abs(180.0 - angle)),
        "axis": axis,
        "axis_symmetric_part_eigenvalues": sym_eigvals,
        "axial_shift_angstrom": axial_shift,
        "dyad_point": dyad,
        "dyad_equation_residual": dyad_residual,
        "centroid_A": np.asarray(centroid_A, dtype=float),
        "centroid_B": np.asarray(centroid_B, dtype=float),
        "centroid_separation_angstrom": separation,
        "centroid_vector_perp_deviation_deg": perp_deviation_deg,
        "centroid_A_axis_distance_angstrom": dist_A,
        "centroid_B_axis_distance_angstrom": dist_B,
    }


# ----------------------------------------------------------------------
# ChimeraX glue: extracts placements/atoms/map data from live session
# objects and calls the pure math above. This half needs ChimeraX.
# ----------------------------------------------------------------------

def _map_density_stats(volume, scene_points):
    """Independent, non-circular per-monomer check: how well do this
    monomer's OWN atoms sit in real density, regardless of the other
    monomer. Returns (mean_value, fraction_above_contour, contour_level)."""
    inv = volume.scene_position.inverse()
    local_pts = inv.transform_points(np.asarray(scene_points, dtype=float))
    origin = np.array(volume.data.origin, dtype=float)
    step = np.array(volume.data.step, dtype=float)
    ijk = (local_pts - origin) / step  # fractional grid (i, j, k)
    data_kji = volume.data.matrix()  # numpy array shaped (nk, nj, ni)
    values = trilinear_sample(data_kji, ijk)

    level = None
    try:
        surfs = volume.surfaces
        if surfs:
            level = float(surfs[0].level)
    except Exception:
        level = None

    frac_above = float(np.mean(values >= level)) if level is not None else None
    return float(np.mean(values)), frac_above, level


def c2geom(session, map, monomerA, monomerB, saveFile=None):
    """ChimeraX command entry point.

    Usage:
        c2geom #1 monomerA #2 monomerB #3
    (map is the positional argument; monomerA/monomerB are keywords, each a
    single already-fitted atomic model.)
    """
    logger = session.logger

    nA, nB = monomerA.num_atoms, monomerB.num_atoms
    if nA != nB:
        logger.warning(
            f"c2geom: monomerA has {nA} atoms but monomerB has {nB}. "
            "They should be two openings of the same reference structure; "
            "results below may not be meaningful."
        )

    R_A, t_A = place_to_Rt(monomerA.scene_position.matrix)
    R_B, t_B = place_to_Rt(monomerB.scene_position.matrix)

    coordsA = monomerA.atoms.scene_coords
    coordsB = monomerB.atoms.scene_coords
    centroid_A = coordsA.mean(axis=0)
    centroid_B = coordsB.mean(axis=0)

    result = c2geom_core(R_A, t_A, R_B, t_B, centroid_A, centroid_B)

    # independent per-monomer density diagnostics (does NOT depend on the
    # A<->B relationship, so it can catch a monomer that individually
    # landed in the wrong place even if the symmetry algebra looks clean)
    meanA, fracA, levelA = _map_density_stats(map, coordsA)
    meanB, fracB, levelB = _map_density_stats(map, coordsB)
    result.update({
        "monomerA_mean_map_value": meanA,
        "monomerA_fraction_atoms_above_contour": fracA,
        "monomerB_mean_map_value": meanB,
        "monomerB_fraction_atoms_above_contour": fracB,
        "map_contour_level": levelA if levelA is not None else levelB,
    })

    # optional voxel-unit convenience conversion -- ONLY valid if isotropic
    step = np.array(map.data.step, dtype=float)
    isotropic = bool(np.allclose(step, step[0], atol=1e-6))
    voxel_size = float(step[0]) if isotropic else None
    if voxel_size:
        result["voxel_size_angstrom"] = voxel_size
        result["centroid_separation_voxels"] = result["centroid_separation_angstrom"] / voxel_size
        result["axial_shift_voxels"] = result["axial_shift_angstrom"] / voxel_size

    # dyad point also reported in the map's own grid-index frame, since
    # that is often what a downstream RELION/subvolume-extraction step
    # actually wants.
    dyad_local = map.scene_position.inverse().transform_points(
        result["dyad_point"].reshape(1, 3)
    )[0]
    origin = np.array(map.data.origin, dtype=float)
    dyad_ijk = (dyad_local - origin) / step
    result["dyad_point_grid_index"] = dyad_ijk

    # Map box size and the SCENE-coordinate position of the box's own
    # geometric centre, needed downstream to convert these (corner-
    # anchored) scene coordinates into RELION's box-centre-relative
    # "particle space" convention (see relion_c2_recenter.py). ChimeraX
    # scene coordinates for a freshly-opened MRC are anchored at grid
    # index (0,0,0), NOT the box centre -- these two are NOT the same
    # point in general, and conflating them was a real bug caught by
    # testing this pipeline against real data (see relion_c2_recenter.py
    # module docstring, section 6).
    grid_size = np.array(map.data.matrix().shape[::-1], dtype=float)  # (nk,nj,ni) -> (ni,nj,nk)
    box_centre_index = (grid_size // 2)  # RELION's own (int)w/2 convention
    box_centre_local = map.data.origin + box_centre_index * step
    box_centre_scene = map.scene_position.transform_points(
        np.asarray(box_centre_local, dtype=float).reshape(1, 3)
    )[0]
    result["map_grid_size"] = grid_size
    result["map_box_centre_scene"] = box_centre_scene

    _report(logger, result, isotropic)

    if saveFile:
        _write_result_file(saveFile, result)
        logger.info(f"c2geom: wrote {saveFile}")

    return result


def _report(logger, r, isotropic):
    lines = []
    lines.append("c2geom -- C2 dimer geometry")
    lines.append("-" * 40)
    lines.append(f"rotation angle           : {r['rotation_angle_deg']:.3f} deg "
                 f"(deviation from 180: {r['angle_deviation_from_180_deg']:.3f} deg)")
    lines.append(f"axis (scene, unit vector): {np.array2string(r['axis'], precision=6)}")
    lines.append(f"  symmetric-part eigvals : {np.array2string(r['axis_symmetric_part_eigenvalues'], precision=4)} "
                 "(expect ~[-1,-1,1] near a clean C2)")
    lines.append(f"axial shift (screw)      : {r['axial_shift_angstrom']:.3f} A "
                 "(expect ~0 for point symmetry)")
    lines.append(f"orthogonality drift of R : {r['orthogonality_drift']:.2e}")
    lines.append("")
    lines.append(f"dyad point (scene, A)    : {np.array2string(r['dyad_point'], precision=3)}")
    lines.append(f"  perp-part residual     : {r['dyad_equation_residual']:.2e} A "
                 "(implementation check, always ~0; NOT the same as axial shift -- see docstring)")
    lines.append(f"dyad point (map grid ijk): {np.array2string(r['dyad_point_grid_index'], precision=3)}")
    lines.append(f"map grid size            : {np.array2string(r['map_grid_size'], precision=0)}")
    lines.append(f"map box centre (scene)   : {np.array2string(r['map_box_centre_scene'], precision=3)} "
                 "(needed downstream to convert scene coords to RELION particle-space)")
    lines.append("")
    lines.append(f"centroid A (scene)       : {np.array2string(r['centroid_A'], precision=3)}")
    lines.append(f"centroid B (scene)       : {np.array2string(r['centroid_B'], precision=3)}")
    lines.append(f"centroid separation      : {r['centroid_separation_angstrom']:.3f} A")
    lines.append(f"centroid-vector vs axis  : {r['centroid_vector_perp_deviation_deg']:.3f} deg "
                 "from perpendicular (expect ~0 -- derived, see docstring)")
    lines.append(f"centroid A dist to axis  : {r['centroid_A_axis_distance_angstrom']:.3f} A")
    lines.append(f"centroid B dist to axis  : {r['centroid_B_axis_distance_angstrom']:.3f} A")
    lines.append("")
    lines.append("independent per-monomer density check (not derived from A<->B relation):")
    lines.append(f"  monomerA mean map value: {r['monomerA_mean_map_value']:.5g}"
                 + (f", frac atoms >= contour: {r['monomerA_fraction_atoms_above_contour']:.3f}"
                    if r['monomerA_fraction_atoms_above_contour'] is not None else ""))
    lines.append(f"  monomerB mean map value: {r['monomerB_mean_map_value']:.5g}"
                 + (f", frac atoms >= contour: {r['monomerB_fraction_atoms_above_contour']:.3f}"
                    if r['monomerB_fraction_atoms_above_contour'] is not None else ""))
    if isotropic and "voxel_size_angstrom" in r:
        lines.append("")
        lines.append(f"voxel size               : {r['voxel_size_angstrom']:.4f} A (isotropic)")
        lines.append(f"centroid separation      : {r['centroid_separation_voxels']:.3f} voxels")
        lines.append(f"axial shift              : {r['axial_shift_voxels']:.3f} voxels")
    else:
        lines.append("")
        lines.append("map is not isotropic (or step unavailable) -- voxel-unit numbers skipped")
    logger.info("\n".join(lines))


def _write_result_file(path, r):
    with open(path, "w") as fh:
        for key, val in r.items():
            if isinstance(val, np.ndarray):
                fh.write(f"{key} = {' '.join(f'{x:.6f}' for x in val.ravel())}\n")
            elif isinstance(val, float):
                fh.write(f"{key} = {val:.6f}\n")
            elif val is None:
                fh.write(f"{key} = NA\n")
            else:
                fh.write(f"{key} = {val}\n")


# ----------------------------------------------------------------------
# ChimeraX command registration -- this is what makes `c2geom ...` work
# on the command line once this file has been run at startup.
# ----------------------------------------------------------------------

def _register():
    from chimerax.core.commands import CmdDesc, register, SaveFileNameArg
    from chimerax.atomic import AtomicStructureArg
    from chimerax.map import MapArg

    desc = CmdDesc(
        required=[("map", MapArg)],
        keyword=[
            ("monomerA", AtomicStructureArg),
            ("monomerB", AtomicStructureArg),
            ("saveFile", SaveFileNameArg),
        ],
        required_arguments=["monomerA", "monomerB"],
        synopsis="Measure C2 axis/dyad/diagnostics between two independently "
                 "fitted copies of a monomer inside a dimer map",
    )
    register("c2geom", desc, c2geom, logger=None)


try:
    # Only attempt registration when actually running inside ChimeraX
    # (this file is also meant to be importable as a plain module for the
    # pure-python functions above, e.g. `from c2geom import c2geom_core`).
    import chimerax.core.commands  # noqa: F401
    _register()
except ImportError:
    pass
except Exception:
    import traceback
    traceback.print_exc()
    print("c2geom: registration failed (see traceback above)")
