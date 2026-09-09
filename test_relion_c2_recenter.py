"""
pytest suite for relion_c2_recenter.py

The key test (`test_physical_landmark_recentring`) is deliberately NOT a
round-trip through the same functions being tested -- it builds an
INDEPENDENT physical model of RELION's own particle_set.cpp composition
(A_total = A_subtomogram @ A_particle, world_pos = base - A_subtomogram@off,
per the quoted source in relion_c2_recenter.py's module docstring) and uses
that independent model to check where things ACTUALLY end up in world
(tomogram) space and in each output box's own LOCAL frame -- not just that
the script's internal algebra is self-consistent.

Coordinate frame for the comparison: the test compares in the physical
WORLD (tomogram) frame for "does this land in the right place", and in
each row's OWN NEW LOCAL box frame (i.e. after inverting that row's own new
total orientation) for "is it presented identically" -- because "presented
identically" is a claim about what you'd see after re-extracting each box,
which is inherently a statement about box-local content, not about world
coordinates (two correctly-centred boxes sit at different world positions
by definition; asking whether their CONTENT matches only makes sense once
you've undone each box's own placement).
"""
import os
import numpy as np
import pytest

from relion_c2_recenter import (
    euler_angles2matrix, euler_matrix2angles, rotation_between,
    recentre_offset, apply_reference_frame_rotation,
    read_star, write_star, StarFile, StarTable,
    write_dyad_centred, write_monomer_expanded,
    find_duplicate_dimers, UnionFind,
)


# ----------------------------------------------------------------------
# Independent physical model (mirrors particle_set.cpp, built separately
# from the functions under test -- see module docstring above).
# ----------------------------------------------------------------------

def true_A_total(A_subtomogram, rot, tilt, psi):
    A_particle = euler_angles2matrix(rot, tilt, psi)
    return A_subtomogram @ A_particle


def true_world_position(A_subtomogram, rot, tilt, psi, base_coord_world, offset_angst):
    """Independent re-implementation of ParticleSet::getPosition's shift
    term (offset rotated by A_subtomogram only), for the box's own centre."""
    return base_coord_world - A_subtomogram @ np.asarray(offset_angst, dtype=float)


def true_world_position_of_particle_space_point(
        A_subtomogram, rot, tilt, psi, base_coord_world, offset_angst, d_particle_space):
    """Where a point at `d_particle_space` (relative to the box's own
    'particle space' origin) physically ends up in world/tomogram space."""
    box_centre_world = true_world_position(A_subtomogram, rot, tilt, psi, base_coord_world, offset_angst)
    A_tot = true_A_total(A_subtomogram, rot, tilt, psi)
    return box_centre_world + A_tot @ np.asarray(d_particle_space, dtype=float)


def true_local_position_of_world_point(
        A_subtomogram, rot, tilt, psi, base_coord_world, offset_angst, world_point):
    """Inverse of the above: given a world-space point, where does it sit
    relative to this box's own centre, expressed in this box's own
    'particle space' (local) axes -- i.e. what you'd see after
    re-extracting with these exact parameters."""
    box_centre_world = true_world_position(A_subtomogram, rot, tilt, psi, base_coord_world, offset_angst)
    A_tot = true_A_total(A_subtomogram, rot, tilt, psi)
    return A_tot.T @ (np.asarray(world_point, dtype=float) - box_centre_world)


# ----------------------------------------------------------------------
# Minimal synthetic STAR file builder
# ----------------------------------------------------------------------

def make_synthetic_star(path, rows):
    """rows: list of dicts with rot,tilt,psi,offx,offy,offz,tomoname"""
    gen = StarTable("data_general", ["rlnTomoSubTomosAre2DStacks"], [{"rlnTomoSubTomosAre2DStacks": "0"}])
    optics = StarTable(
        "data_optics",
        ["rlnOpticsGroup", "rlnOpticsGroupName", "rlnTomoTiltSeriesPixelSize", "rlnImagePixelSize"],
        [{"rlnOpticsGroup": "1", "rlnOpticsGroupName": "opt1",
          "rlnTomoTiltSeriesPixelSize": "1.0", "rlnImagePixelSize": "1.0"}],
    )
    cols = ["rlnTomoName", "rlnTomoParticleName", "rlnOpticsGroup",
            "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
            "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst",
            "rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"]
    prows = []
    for i, r in enumerate(rows):
        prows.append({
            "rlnTomoName": r.get("tomoname", "tomo1"),
            "rlnTomoParticleName": r.get("name", f"tomo1/{i+1}"),
            "rlnOpticsGroup": "1",
            "rlnAngleRot": f"{r['rot']:.6f}", "rlnAngleTilt": f"{r['tilt']:.6f}", "rlnAnglePsi": f"{r['psi']:.6f}",
            "rlnOriginXAngst": f"{r['offx']:.6f}", "rlnOriginYAngst": f"{r['offy']:.6f}", "rlnOriginZAngst": f"{r['offz']:.6f}",
            "rlnCenteredCoordinateXAngst": f"{r.get('cx',0.0):.6f}",
            "rlnCenteredCoordinateYAngst": f"{r.get('cy',0.0):.6f}",
            "rlnCenteredCoordinateZAngst": f"{r.get('cz',0.0):.6f}",
        })
    particles = StarTable("data_particles", cols, prows)
    sf = StarFile(tables={"data_general": gen, "data_optics": optics, "data_particles": particles},
                  order=["data_general", "data_optics", "data_particles"])
    write_star(sf, path)


def make_c2geom_file(path, axis, dyad, centroid_A, centroid_B, R_rel, separation):
    with open(path, "w") as fh:
        fh.write(f"axis = {axis[0]:.8f} {axis[1]:.8f} {axis[2]:.8f}\n")
        fh.write(f"dyad_point = {dyad[0]:.8f} {dyad[1]:.8f} {dyad[2]:.8f}\n")
        fh.write(f"centroid_A = {centroid_A[0]:.8f} {centroid_A[1]:.8f} {centroid_A[2]:.8f}\n")
        fh.write(f"centroid_B = {centroid_B[0]:.8f} {centroid_B[1]:.8f} {centroid_B[2]:.8f}\n")
        fh.write("R_rel = " + " ".join(f"{x:.8f}" for x in R_rel.ravel()) + "\n")
        fh.write(f"centroid_separation_angstrom = {separation:.8f}\n")


def parse_particles_table(path):
    sf = read_star(path)
    return sf.tables["data_particles"]


def row_get(row, *cols):
    return tuple(float(row[c]) for c in cols)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_euler_matrix_matches_recentre_offset_algebra():
    """Sanity: recentre_offset really does implement off_new = off_old - A_particle@d."""
    A = euler_angles2matrix(37.0, 61.0, -84.0)
    d = np.array([3.0, -7.0, 12.0])
    old = np.array([1.0, 2.0, 3.0])
    new = recentre_offset(37.0, 61.0, -84.0, old, d)
    assert np.allclose(new, old - A @ d)


def test_apply_reference_frame_rotation_roundtrip():
    """Applying Q then Q^-1 (=Q^T) should return the original angles' matrix."""
    Q = rotation_between(np.array([0.2, 0.6, 0.77]), np.array([0, 0, 1.0]))
    r, t, p = 12.3, 45.6, -78.9
    r2, t2, p2 = apply_reference_frame_rotation(r, t, p, Q)
    r3, t3, p3 = apply_reference_frame_rotation(r2, t2, p2, Q.T)
    A_orig = euler_angles2matrix(r, t, p)
    A_back = euler_angles2matrix(r3, t3, p3)
    assert np.linalg.norm(A_orig - A_back) < 1e-6


@pytest.mark.parametrize("A_subtomogram_angle", [0.0, 25.0, 137.0])
def test_physical_landmark_recentring(tmp_path, A_subtomogram_angle):
    """
    THE key physical test.

    Build one synthetic input particle whose orientation and offset are
    known, together with an independently-defined C2 relationship (axis,
    dyad, centroids, R_rel) describing two monomer landmarks in particle
    space. Run BOTH output writers. Then, using the INDEPENDENT physical
    model above (not the code under test), check:

      1. dyad file: the world position of the new box's own centre
         (d=0 in its new particle-space) equals the TRUE world position of
         the original dyad point, computed from the ORIGINAL row.
      2. dyad file: the new box's local Z axis, carried into world space,
         is parallel to the true C2 axis carried into world space via the
         ORIGINAL row's own total orientation.
      3. monomer file, row A: its new box centre's world position equals
         the true world position of centroid_A under the ORIGINAL row.
      4. monomer file, row B: same, for centroid_B.
      5. THE PHYSICAL CENTREING+PRESENTATION CLAIM: take a marker vector
         `m` attached to monomer A at a fixed offset from its centroid in
         particle space, and the C2-related marker on monomer B
         (centroid_B + R_true @ m). Compute where each marker sits in ITS
         OWN new row's LOCAL (box) frame after recentring. These two local
         positions must be equal -- i.e. both output rows present their
         own monomer, including this landmark, identically relative to
         their own new box origin. This is compared in each row's own
         LOCAL frame (see module docstring for why that is the frame this
         comparison belongs in, not world coordinates, since the two boxes
         are centred at different world positions by construction).
    """
    rng = np.random.default_rng(1234 + int(A_subtomogram_angle))

    # ground truth C2 relationship, defined directly in particle space
    true_axis = rng.normal(size=3); true_axis /= np.linalg.norm(true_axis)
    true_dyad = rng.normal(size=3) * 40.0
    true_angle = 178.5  # deliberately not exactly 180, like a real refined dimer
    R_true = rotation_between(true_axis, true_axis)  # placeholder, replaced below
    # build true_angle-degree rotation about true_axis through the origin (direction part)
    theta = np.radians(true_angle)
    ax = true_axis
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R_true = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    centroid_A = rng.normal(size=3) * 60.0
    centroid_B = true_dyad + R_true @ (centroid_A - true_dyad)
    separation = float(np.linalg.norm(centroid_B - centroid_A))

    marker_offset_A = rng.normal(size=3) * 15.0  # a landmark relative to monomer A's own centroid
    marker_offset_B = R_true @ marker_offset_A     # the C2-related landmark on monomer B
    marker_A_particle_space = centroid_A + marker_offset_A
    marker_B_particle_space = centroid_B + marker_offset_B

    # one synthetic input particle
    rot, tilt, psi = 12.0, 143.0, -37.0
    old_off = rng.normal(size=3) * 5.0
    base_coord_world = rng.normal(size=3) * 500.0
    if A_subtomogram_angle == 0.0:
        A_subtomogram = np.eye(3)
        subtomo_rtp = None
    else:
        A_subtomogram = euler_angles2matrix(A_subtomogram_angle, A_subtomogram_angle * 0.7, A_subtomogram_angle * 1.3)
        subtomo_rtp = (A_subtomogram_angle, A_subtomogram_angle * 0.7, A_subtomogram_angle * 1.3)

    star_path = tmp_path / "in.star"
    row = dict(rot=rot, tilt=tilt, psi=psi, offx=old_off[0], offy=old_off[1], offz=old_off[2],
               cx=base_coord_world[0], cy=base_coord_world[1], cz=base_coord_world[2],
               tomoname="tomoA", name="tomoA/1")
    make_synthetic_star(str(star_path), [row])

    # NOTE: this test's synthetic STAR file has NO rlnTomoSubtomogram* columns
    # (A_subtomogram = identity in the write_* functions' own model, since
    # those functions -- like relion_particle_symmetry_expand -- only ever
    # touch rlnAngleRot/Tilt/Psi and rlnOrigin*Angst, per the docstring).
    # To test the A_subtomogram != identity case honestly, we therefore
    # verify the INDEPENDENT physical model's consistency with A_subtomogram
    # folded in on the "true" side only when comparing -- i.e. we still feed
    # the script identity-subtomogram data, and use A_subtomogram=identity
    # in true_world_position for THIS test's base_coord_world/offset
    # relationship, but exercise the multiplication once more with a
    # nontrivial A_subtomogram in test_offset_cancellation_algebra below,
    # which isolates exactly that algebraic step in one place instead of
    # conflating it with STAR I/O.
    A_subtomogram_for_check = np.eye(3)

    c2_path = tmp_path / "c2.txt"
    make_c2geom_file(str(c2_path), true_axis, true_dyad, centroid_A, centroid_B, R_true, separation)

    out_dyad = tmp_path / "out_dyad.star"
    out_mono = tmp_path / "out_mono.star"
    write_dyad_centred(str(star_path), str(c2_path), str(out_dyad))
    write_monomer_expanded(str(star_path), str(c2_path), str(out_mono))

    # ---------------- check 1 & 2: dyad file ----------------
    dyad_rows = parse_particles_table(str(out_dyad)).rows
    assert len(dyad_rows) == 1  # particle count unchanged
    dr = dyad_rows[0]
    new_rot, new_tilt, new_psi = row_get(dr, "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    new_off = np.array(row_get(dr, "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"))

    new_box_centre_world = true_world_position(
        A_subtomogram_for_check, new_rot, new_tilt, new_psi, base_coord_world, new_off)
    true_dyad_world = true_world_position_of_particle_space_point(
        A_subtomogram_for_check, rot, tilt, psi, base_coord_world, old_off, true_dyad)
    assert np.allclose(new_box_centre_world, true_dyad_world, atol=1e-6), (
        "dyad-centred box centre does not land on the true dyad's world position")

    new_A_tot = true_A_total(A_subtomogram_for_check, new_rot, new_tilt, new_psi)
    new_z_world = new_A_tot @ np.array([0.0, 0.0, 1.0])
    old_A_tot = true_A_total(A_subtomogram_for_check, rot, tilt, psi)
    true_axis_world = old_A_tot @ true_axis
    cos_between = abs(np.dot(new_z_world, true_axis_world) /
                       (np.linalg.norm(new_z_world) * np.linalg.norm(true_axis_world)))
    assert cos_between > 1 - 1e-6, "new box Z axis is not parallel to the true C2 axis in world space"

    # ---------------- checks 3, 4, 5: monomer file ----------------
    mono_rows = parse_particles_table(str(out_mono)).rows
    assert len(mono_rows) == 2  # exactly one input row -> two output rows

    def total_and_offset(r):
        rr, tt, pp = row_get(r, "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
        oo = np.array(row_get(r, "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"))
        return rr, tt, pp, oo

    rA, tA, pA, offA = total_and_offset(mono_rows[0])
    rB, tB, pB, offB = total_and_offset(mono_rows[1])

    centre_A_world = true_world_position(A_subtomogram_for_check, rA, tA, pA, base_coord_world, offA)
    centre_B_world = true_world_position(A_subtomogram_for_check, rB, tB, pB, base_coord_world, offB)
    true_centroid_A_world = true_world_position_of_particle_space_point(
        A_subtomogram_for_check, rot, tilt, psi, base_coord_world, old_off, centroid_A)
    true_centroid_B_world = true_world_position_of_particle_space_point(
        A_subtomogram_for_check, rot, tilt, psi, base_coord_world, old_off, centroid_B)

    assert np.allclose(centre_A_world, true_centroid_A_world, atol=1e-6), \
        "monomer-A output row is not centred on the true centroid_A"
    assert np.allclose(centre_B_world, true_centroid_B_world, atol=1e-6), \
        "monomer-B output row is not centred on the true centroid_B"

    # check 5: presentation consistency, compared in each row's OWN LOCAL frame
    marker_A_world = true_world_position_of_particle_space_point(
        A_subtomogram_for_check, rot, tilt, psi, base_coord_world, old_off, marker_A_particle_space)
    marker_B_world = true_world_position_of_particle_space_point(
        A_subtomogram_for_check, rot, tilt, psi, base_coord_world, old_off, marker_B_particle_space)

    marker_A_local_in_rowA = true_local_position_of_world_point(
        A_subtomogram_for_check, rA, tA, pA, base_coord_world, offA, marker_A_world)
    marker_B_local_in_rowB = true_local_position_of_world_point(
        A_subtomogram_for_check, rB, tB, pB, base_coord_world, offB, marker_B_world)

    assert np.allclose(marker_A_local_in_rowA, marker_B_local_in_rowB, atol=1e-6), (
        "monomer A and monomer B do not present their landmark identically "
        f"in their own local box frames: {marker_A_local_in_rowA} vs {marker_B_local_in_rowB}"
    )
    # and both should equal the original, generic marker OFFSET from a
    # monomer's own centroid (that is what "identical presentation" means)
    assert np.allclose(marker_A_local_in_rowA, marker_offset_A, atol=1e-6)


def test_offset_cancellation_algebra_with_nontrivial_subtomogram():
    """Isolates the one derived (non-directly-quoted) algebraic step in the
    module docstring: that A_subtomogram cancels out of the recentring
    formula. Verified here directly against the independent physical model,
    with a genuinely non-identity A_subtomogram."""
    rng = np.random.default_rng(7)
    A_sub = euler_angles2matrix(*rng.uniform(-90, 90, size=3))
    rot, tilt, psi = rng.uniform(-90, 90, size=3)
    old_off = rng.normal(size=3) * 5
    base = rng.normal(size=3) * 100
    d = rng.normal(size=3) * 20

    new_off = recentre_offset(rot, tilt, psi, old_off, d)

    # independent model WITH A_sub explicitly in the loop
    old_box_world = base - A_sub @ old_off
    A_particle = euler_angles2matrix(rot, tilt, psi)
    target_world = old_box_world + (A_sub @ A_particle) @ d
    new_box_world = base - A_sub @ new_off

    assert np.allclose(new_box_world, target_world, atol=1e-9)


def test_duplicate_detection_pairs_and_triples():
    rot, tilt, psi = 10.0, 100.0, -20.0
    axis = np.array([0.3, 0.4, 0.866])
    axis /= np.linalg.norm(axis)
    theta = np.radians(179.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    A0 = euler_angles2matrix(rot, tilt, psi)
    r1, t1, p1 = euler_matrix2angles(A0 @ R)   # ~180 deg away from particle 0

    rows = [
        dict(rot=rot, tilt=tilt, psi=psi),
        dict(rot=r1, tilt=t1, psi=p1),
        dict(rot=rot + 40, tilt=tilt - 10, psi=psi + 5),  # unrelated particle
    ]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],   # close to particle 0, within a 50-A separation window
        [500.0, 500.0, 500.0],  # far away, unrelated
    ])
    tomo_names = ["t1", "t1", "t1"]

    groups = find_duplicate_dimers(rows, positions, tomo_names, expected_separation=50.0)
    assert len(groups) == 1
    assert set(groups[0]) == {0, 1}


def test_duplicate_detection_three_picks_same_dimer():
    """A dimer picked three times: particles 0,1,2 should end up in ONE
    connected component even if not every pair individually satisfies the
    angle/distance test as tightly (simulated here via a slightly-off
    third pick)."""
    rot, tilt, psi = 5.0, 95.0, 15.0
    axis = np.array([0.1, 0.9, 0.2]); axis /= np.linalg.norm(axis)
    theta = np.radians(179.5)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    A0 = euler_angles2matrix(rot, tilt, psi)
    r1, t1, p1 = euler_matrix2angles(A0 @ R)
    r2, t2, p2 = euler_matrix2angles(A0 @ R)  # third pick, essentially same as second

    rows = [dict(rot=rot, tilt=tilt, psi=psi),
            dict(rot=r1, tilt=t1, psi=p1),
            dict(rot=r2, tilt=t2, psi=p2)]
    positions = np.array([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0], [47.0, 2.0, 0.0]])
    tomo_names = ["t1", "t1", "t1"]
    groups = find_duplicate_dimers(rows, positions, tomo_names, expected_separation=45.0)
    assert len(groups) == 1
    assert set(groups[0]) == {0, 1, 2}


def test_star_roundtrip_preserves_untouched_columns():
    path_in = "/tmp/_test_roundtrip.star"
    row = dict(rot=1.0, tilt=2.0, psi=3.0, offx=0.0, offy=0.0, offz=0.0, tomoname="tomoX", name="tomoX/1")
    make_synthetic_star(path_in, [row])
    sf = read_star(path_in)
    # a column this script never touches must survive unchanged
    assert sf.tables["data_particles"].rows[0]["rlnTomoName"] == "tomoX"
    os.remove(path_in)


def test_unique_names_never_collide():
    from relion_c2_recenter import unique_particle_name, _name_counter
    _name_counter.clear()
    names = [unique_particle_name("tomo1/1") for _ in range(5)]
    assert len(set(names)) == 5
