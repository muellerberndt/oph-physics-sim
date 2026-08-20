"""Receipts and mutation guards for the lane D2 family readout v2
(exploratory, non-evidential; design record ``oph_fpe/defects/DESIGN_V2.md``)."""

from __future__ import annotations

import random

import pytest

from oph_fpe.defects.z6_a5_action import (
    act_on_sector,
    declared_generator,
    rotation_group,
    sector_orbit,
)
from oph_fpe.defects.z6_carrier_defects import (
    MOD,
    CarrierDefectError,
    base_carrier_spec,
    chord_holonomies,
    gauge_move,
    zero_config,
)
from oph_fpe.defects.z6_family_readout import (
    all_seam_colored_vector,
    antipodal_port_map,
    antipodal_receipt,
    antipodal_seam_involution,
    a5_fixed_character_receipt,
    build_readout,
    canonical_tait_coloring,
    character_is_gauge_invariant,
    config_label_v2,
    crt_diagonal,
    diagonal_weights,
    family_label,
    family_weights,
    gauge_invariance_receipt,
    image_is_diagonal,
    label_image_subgroup,
    non_vacuity_receipt,
    orbit_label_multiset,
    seam_weight_vector,
    tait_receipt,
)
from oph_fpe.defects.z6_matter_grammar_verifier import CONTROL_LABEL
from oph_fpe.defects.z6_matter_grammar_verifier_v2 import (
    attach_v2_labels,
    receipt_json,
    run_v2,
    verify_census_v2,
)

SPEC = base_carrier_spec()
ROTATIONS = rotation_group(SPEC)
COLORING = canonical_tait_coloring(SPEC)
INVOLUTION = antipodal_seam_involution(SPEC)
WEIGHTS = family_weights(SPEC, COLORING, INVOLUTION)
TEST_STREAMS = (("uniform_iid", 11, 24), ("sparse_pair", 12, 12))


def _random_sectors(seed: int, count: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    return [
        tuple(rng.randrange(MOD) for _ in range(len(SPEC.chords)))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# Structure receipts: coloring and antipodal derivation
# ---------------------------------------------------------------------------

def test_tait_receipt_rainbow_and_classes():
    receipt = tait_receipt(SPEC, COLORING)
    assert receipt["class_sizes"] == [10, 10, 10]
    assert receipt["rainbow_faces"] == SPEC.faces


def test_tait_receipt_rejects_broken_coloring():
    from oph_fpe.defects.z6_family_readout import _face_seams

    seams0 = list(_face_seams(SPEC))[0]
    broken = list(COLORING)
    e1, e2 = seams0[0], seams0[1]
    broken[e1], broken[e2] = broken[e2], broken[e1]
    assert sorted(sum(1 for c in broken if c == v) for v in range(3)) == [10, 10, 10]
    with pytest.raises(CarrierDefectError, match="not rainbow"):
        tait_receipt(SPEC, broken)


def test_coloring_deterministic():
    assert canonical_tait_coloring(SPEC) == COLORING


def test_antipodal_port_map_involution():
    ports = antipodal_port_map(SPEC)
    assert all(ports[ports[p]] == p and ports[p] != p
               for p in range(SPEC.ports))


def test_antipodal_seam_involution_pairs():
    assert all(INVOLUTION[INVOLUTION[e]] == e and INVOLUTION[e] != e
               for e in range(SPEC.seams))
    pairs = {tuple(sorted((e, INVOLUTION[e]))) for e in range(SPEC.seams)}
    assert len(pairs) == SPEC.seams // 2


def test_antipodal_receipt_not_in_rotation_group():
    receipt = antipodal_receipt(SPEC, ROTATIONS)
    assert receipt["graph_automorphism"] is True
    assert receipt["in_rotation_group"] is False
    assert receipt["seam_pair_count"] == 15


# ---------------------------------------------------------------------------
# Gauge invariance: exact receipt and the gauge-variance mutant
# ---------------------------------------------------------------------------

def test_gauge_invariance_receipt_exact():
    receipt = gauge_invariance_receipt(SPEC, WEIGHTS)
    assert receipt["unit_port_moves_zero_label"] == SPEC.ports
    for name in ("q", "t", "d"):
        assert receipt["characters"][name]["boundary_zero_ports"] == 12


def test_gauge_variance_mutant_caught():
    mutant = all_seam_colored_vector(SPEC, COLORING)
    assert character_is_gauge_invariant(SPEC, mutant, 3) is False


def test_v2_weight_vectors_pass_the_mutant_check():
    for coeffs, modulus in (
        (WEIGHTS.w6, MOD), (WEIGHTS.w3, 3), (WEIGHTS.w2, 2),
    ):
        vector = seam_weight_vector(SPEC, coeffs)
        assert character_is_gauge_invariant(SPEC, vector, modulus) is True


def test_label_invariant_under_sampled_gauge_moves():
    rng = random.Random(101)
    for _ in range(25):
        config = [rng.randrange(MOD) for _ in range(SPEC.seams)]
        gauge = [rng.randrange(MOD) for _ in range(SPEC.ports)]
        moved = gauge_move(SPEC, config, gauge)
        assert config_label_v2(SPEC, WEIGHTS, config) == config_label_v2(
            SPEC, WEIGHTS, moved
        )


def test_label_zero_on_unit_port_moves():
    for p in range(SPEC.ports):
        gauge = [0] * SPEC.ports
        gauge[p] = 1
        moved = gauge_move(SPEC, zero_config(SPEC), gauge)
        assert config_label_v2(SPEC, WEIGHTS, moved) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Vacuity detector: the diagonal readout mutant and the v2 image
# ---------------------------------------------------------------------------

def test_diagonal_mutant_triggers_vacuity_detector():
    mutant = diagonal_weights(SPEC)
    image = label_image_subgroup(mutant)
    assert image_is_diagonal(image) is True
    with pytest.raises(CarrierDefectError):
        non_vacuity_receipt(SPEC, mutant)


def test_v2_image_full_lattice_and_live_checks():
    receipt = non_vacuity_receipt(SPEC, WEIGHTS)
    assert receipt["image_is_diagonal"] is False
    assert receipt["image_size"] == 36
    assert receipt["image_full_lattice"] is True
    assert receipt["control_label_reachable"] is True
    witness = receipt["off_diagonal_witness"]
    assert tuple(witness["label_qtd"]) not in crt_diagonal()
    assert witness["center_char"] != 0


def test_diagonal_mutant_labels_match_v1_shape():
    mutant = diagonal_weights(SPEC)
    for sector in _random_sectors(7, 10):
        q, t, d = family_label(mutant, sector)
        assert (t, d) == (q % 3, q % 2)


# ---------------------------------------------------------------------------
# A5 receipts: fixed characters and orbit multiset stability
# ---------------------------------------------------------------------------

def test_a5_fixed_character_spaces_zero():
    receipt = a5_fixed_character_receipt(SPEC, ROTATIONS)
    assert receipt["fixed_dim_mod3"] == 0
    assert receipt["fixed_dim_mod2"] == 0
    assert receipt["rotations"] == 60


def test_orbit_label_multiset_stable_across_members():
    generator = declared_generator(ROTATIONS)
    for sector in _random_sectors(13, 5):
        moved = act_on_sector(SPEC, generator, sector)
        assert orbit_label_multiset(
            SPEC, ROTATIONS, WEIGHTS, sector
        ) == orbit_label_multiset(SPEC, ROTATIONS, WEIGHTS, moved)


def test_labels_not_constant_on_some_orbit():
    # The fixed-character receipt forces label variation on some orbit;
    # exhibit one on the unit-sector orbits.
    varied = False
    for i in range(len(SPEC.chords)):
        unit = [0] * len(SPEC.chords)
        unit[i] = 1
        multiset = orbit_label_multiset(SPEC, ROTATIONS, WEIGHTS, unit)
        if len({tuple(lab) for lab in multiset}) > 1:
            varied = True
            break
    assert varied


# ---------------------------------------------------------------------------
# Build receipt bundle
# ---------------------------------------------------------------------------

def test_build_readout_receipts_complete():
    readout = build_readout(SPEC, ROTATIONS)
    receipts = readout["receipts"]
    assert receipts["readout"] == "tait_antipodal_family_readout.v2"
    assert receipts["non_vacuity"]["image_full_lattice"] is True
    assert receipts["a5_fixed_characters"]["fixed_dim_mod3"] == 0
    assert receipts["mutant_control"][
        "all_seam_colored_vector_invariant"
    ] is False
    assert readout["weights"] == WEIGHTS


# ---------------------------------------------------------------------------
# Verifier: planted control label and table consistency
# ---------------------------------------------------------------------------

def _sector_with_label(target) -> tuple[int, ...]:
    """A sector realizing the target label, by breadth-first closure over
    unit-chord increments (the image is the full lattice)."""
    n = len(SPEC.chords)
    zero = tuple([0] * n)
    seen = {(0, 0, 0): zero}
    frontier = [(0, 0, 0)]
    while frontier:
        state = frontier.pop(0)
        sector = seen[state]
        if state == tuple(target):
            return sector
        for i in range(n):
            bumped = list(sector)
            bumped[i] = (bumped[i] + 1) % MOD
            bumped_t = tuple(bumped)
            label = family_label(WEIGHTS, bumped_t)
            if label not in seen:
                seen[label] = bumped_t
                frontier.append(label)
    raise AssertionError("target label unreachable")


def test_planted_control_label_reported():
    sector = _sector_with_label(CONTROL_LABEL)
    assert family_label(WEIGHTS, sector) == CONTROL_LABEL
    orbit_rep, orbit_size = sector_orbit(SPEC, ROTATIONS, sector)
    census = {
        "classes": [{
            "sector": list(sector),
            "label_qtd": [sum(sector) % MOD, sum(sector) % 3,
                          sum(sector) % 2],
            "energy": 1,
            "multiplicity": 1,
            "neutral_escapable": True,
            "orbit_representative": list(orbit_rep),
            "orbit_size": orbit_size,
        }],
    }
    labeled = attach_v2_labels(census, SPEC, ROTATIONS, WEIGHTS)
    verifier = verify_census_v2(labeled)
    assert verifier["control"]["realized"] is True
    assert verifier["control"]["class_count"] == 1
    planted = [
        cell for cell in verifier["lattice_occupancy"]
        if tuple(cell["label_qtd"]) == CONTROL_LABEL
    ]
    assert planted[0]["class_count"] == 1


def test_verifier_tables_consistent_on_test_streams():
    receipt = run_v2(SPEC, TEST_STREAMS)
    verifier = receipt["verifier"]
    classes = receipt["class_labels"]
    total = len(classes)
    assert verifier["descent"]["classes_total"] == total
    assert sum(
        cell["class_count"] for cell in verifier["lattice_occupancy"]
    ) == total
    partition = verifier["committed_vs_complement"]
    assert partition["classes_in_committed"] + partition[
        "classes_in_complement"
    ] == total
    assert partition["committed_label_count"] == 6
    assert partition["complement_label_count"] == 30
    assert verifier["descent"]["diagonal_readout_detected"] is False
    for entry in classes:
        sector = tuple(entry["sector"])
        assert tuple(entry["label_qtd_v2"]) == family_label(
            WEIGHTS, sector
        )
    stable_count = sum(1 for c in classes if c["depth2_stable"])
    assert verifier["cross_tabulation"]["depth2_stable_classes"] == (
        stable_count
    )
    assert sum(
        cell["class_count"]
        for cell in verifier["cross_tabulation"]["stable_cells"]
    ) == stable_count


def test_orbit_multisets_cover_realized_members():
    receipt = run_v2(SPEC, TEST_STREAMS)
    for orbit in receipt["orbit_labels"]:
        full = [tuple(lab) for lab in orbit["label_multiset_full"]]
        for lab in orbit["label_multiset_realized"]:
            assert tuple(lab) in full
        assert orbit["orbit_size"] >= orbit["realized_classes"]


def test_receipt_bytes_stable_under_rerun():
    first = receipt_json(run_v2(SPEC, TEST_STREAMS))
    second = receipt_json(run_v2(SPEC, TEST_STREAMS))
    assert first == second
