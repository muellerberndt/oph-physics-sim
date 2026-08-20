"""Structural receipts for the exact base carrier of the #733 EM handoff.

Design-only tests: finite exact mathematics on the committed tables, no
physical claim, no instrument.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from oph_fpe.em import base_carrier
from oph_fpe.em.base_carrier import (
    BaseCarrierError,
    CHORD_SEAMS,
    FACES,
    PORTS,
    SEAM_LEFT,
    SEAM_RIGHT,
    SEAMS,
    TREE_SEAMS,
)

ZERO = Fraction(0)
ONE = Fraction(1)


def test_base_carrier_receipt_passes_with_committed_tables() -> None:
    receipt = base_carrier.base_carrier_receipt()
    assert receipt["rank_C"] == 19
    assert receipt["rank_d"] == 11
    assert receipt["C_compose_d_zero"] is True
    assert receipt["kernel_C_dimension"] == 11
    assert receipt["kernel_C_equals_image_d"] is True
    assert receipt["kernel_d_dimension"] == 1
    assert receipt["degree_sequence"] == [5] * PORTS
    assert receipt["local_kinetic"] == {
        "diagonal": 2,
        "nonzero_per_row": 5,
        "total_support": 150,
        "rank": 19,
    }
    assert receipt["design_only"] is True
    assert receipt["instrument_armed"] is False


def test_composition_c_after_d_is_zero_entrywise() -> None:
    c = base_carrier.face_incidence_matrix()
    d = base_carrier.coboundary_matrix()
    for f in range(FACES):
        for p in range(PORTS):
            total = sum(
                (c[f][e] * d[e][p] for e in range(SEAMS)), start=ZERO
            )
            assert total == 0


def test_boundary_of_codifferential_is_zero() -> None:
    face_field = [Fraction(f * f - 3, 7) for f in range(FACES)]
    seam_field = base_carrier.face_codifferential(face_field)
    assert base_carrier.boundary(seam_field) == [ZERO] * PORTS


def test_face_curvature_of_gradient_is_zero() -> None:
    phi = [Fraction(2 * p + 1, 3) for p in range(PORTS)]
    assert base_carrier.face_curvature(base_carrier.coboundary(phi)) == [
        ZERO
    ] * FACES


def test_orientation_erasure_control_matches_committed_value() -> None:
    # Committed adversarial control: with |C| in place of the signed
    # incidence, boundary-of-boundary on face zero at port zero is -2.
    control = base_carrier.unsigned_face_boundary_control()
    assert control == base_carrier.UNSIGNED_FACE_ZERO_PORT_ZERO_CONTROL == -2
    signed = sum(
        (
            base_carrier.face_incidence_entry(0, e)
            * base_carrier.incidence_entry(e, 0)
            for e in range(SEAMS)
        ),
        start=ZERO,
    )
    assert signed == 0


def test_orientation_erasure_fails_the_structural_receipt() -> None:
    unsigned = [
        [abs(x) for x in row] for row in base_carrier.face_incidence_matrix()
    ]
    with pytest.raises(BaseCarrierError):
        base_carrier.base_carrier_receipt(face_matrix=unsigned)


def test_single_sign_flip_in_c_fails_the_structural_receipt() -> None:
    flipped = [list(row) for row in base_carrier.face_incidence_matrix()]
    flipped[0][0] = -flipped[0][0]
    with pytest.raises(BaseCarrierError):
        base_carrier.base_carrier_receipt(face_matrix=flipped)


def test_seam_orientation_left_below_right() -> None:
    assert all(
        0 <= SEAM_LEFT[e] < SEAM_RIGHT[e] < PORTS for e in range(SEAMS)
    )
    assert len({(SEAM_LEFT[e], SEAM_RIGHT[e]) for e in range(SEAMS)}) == SEAMS


def test_spanning_tree_and_fundamental_cycles() -> None:
    assert len(TREE_SEAMS) == PORTS - 1
    assert len(CHORD_SEAMS) == 19
    cycles = base_carrier.fundamental_cycles()
    for chord, cycle in cycles.items():
        assert cycle[chord] == 1
        assert all(cycle[other] == 0 for other in CHORD_SEAMS if other != chord)
        assert base_carrier.boundary(cycle) == [ZERO] * PORTS


def test_tree_solution_solves_gauss_on_tree_support() -> None:
    load = [ZERO] * PORTS
    load[3], load[8] = ONE, -ONE
    current = base_carrier.tree_solution(load)
    assert base_carrier.boundary(current) == load
    assert all(current[chord] == 0 for chord in CHORD_SEAMS)


def test_tree_solution_rejects_non_neutral_load() -> None:
    load = [ZERO] * PORTS
    load[0] = ONE
    with pytest.raises(BaseCarrierError):
        base_carrier.tree_solution(load)


def test_laplacian_is_degree_five_regular_adjacency_form() -> None:
    lap = base_carrier.laplacian_matrix()
    for p in range(PORTS):
        assert lap[p][p] == 5
        assert sum(lap[p], start=ZERO) == 0
