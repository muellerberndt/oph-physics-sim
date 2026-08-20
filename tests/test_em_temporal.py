"""Exact leapfrog temporal Maxwell lane: per-step receipts and mutation
guards.

Design-only tests.  The step index is a declared evolution parameter, not
physical time; sources are declared inputs; nothing here is a frozen
instrument or a physical claim.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from oph_fpe.em import base_carrier, green, temporal
from oph_fpe.em.base_carrier import CHORD_SEAMS, FACES, PORTS, SEAMS
from oph_fpe.em.temporal import (
    LeapfrogLane,
    TemporalReceiptError,
    constant_history,
    zero_ports,
    zero_seams,
)

ZERO = Fraction(0)
ONE = Fraction(1)


def face_zero_kick() -> list[Fraction]:
    return base_carrier.face_codifferential(
        [ONE if f == 0 else ZERO for f in range(FACES)]
    )


# ---------------------------------------------------------------------------
# The committed nonstatic inhabitant and exact energy conservation
# ---------------------------------------------------------------------------

def test_demo_bundle_matches_committed_anchors_over_64_steps() -> None:
    demo = temporal.demo_bundle(steps=64)
    assert demo["anchors"] == {
        "A1_seam0": "1",
        "B0_zero": True,
        "B1_face0": "3",
        "nonstatic": True,
    }
    receipts = demo["receipts"]
    assert receipts["ampere_exact"] is True
    assert receipts["faraday_identity_exact"] is True
    assert receipts["codifferential_boundary_zero"] is True
    assert receipts["gauss"]["propagated_all_steps"] is True
    assert receipts["energy"]["source_free"] is True
    assert receipts["energy"]["drift_zero"] is True
    assert receipts["energy"]["initial"] == receipts["energy"]["final"] == "3/2"
    assert demo["wave"]["wave_recursion_exact"] is True
    assert demo["wave"]["interior_steps"] == 63


def test_energy_balance_carries_the_exact_source_work_term() -> None:
    lane = LeapfrogLane()
    steps = 8
    # Conserved seam current: a fundamental cycle, constant in the step index.
    cycle = base_carrier.fundamental_cycle(CHORD_SEAMS[2])
    current = constant_history(cycle, steps - 1)
    phi = constant_history(zero_ports(), steps)
    run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, verify=False
    )
    receipts = lane.verify_history(
        run["A"], phi, current, fail_closed=True
    )
    assert receipts["energy"]["balance_exact"] is True
    assert receipts["energy"]["source_free"] is False
    # The work term moves the energy: the balance is not vacuous here.
    a = run["A"]
    energies = [lane.field_energy(a, phi, n) for n in range(steps)]
    assert energies[0] != energies[-1]
    for n in range(steps - 1):
        electric_sum = [
            lane.electric_field(a, phi, n)[e]
            + lane.electric_field(a, phi, n + 1)[e]
            for e in range(SEAMS)
        ]
        work = Fraction(1, 2) * base_carrier.seam_inner(electric_sum, cycle)
        assert energies[n + 1] == energies[n] - work


# ---------------------------------------------------------------------------
# Gauss and continuity, both directions
# ---------------------------------------------------------------------------

def test_conserved_source_propagates_the_gauss_constraint() -> None:
    lane = LeapfrogLane()
    steps = 8
    cycle = base_carrier.fundamental_cycle(CHORD_SEAMS[0])
    current = constant_history(cycle, steps - 1)
    assert base_carrier.boundary(cycle) == [ZERO] * PORTS
    phi = constant_history(zero_ports(), steps)
    rho = constant_history(zero_ports(), steps)
    run = lane.evolve(
        zero_seams(), zero_seams(), phi, current, steps, rho=rho, verify=True
    )
    gauss = run["receipts"]["gauss"]
    assert gauss["initial"] is True
    assert gauss["continuity_all_steps"] is True
    assert gauss["propagated_all_steps"] is True
    assert gauss["step_iff_consistent"] is True


def test_continuity_violating_source_breaks_the_gauss_constraint() -> None:
    lane = LeapfrogLane()
    steps = 6
    bad_current = [
        [ONE if e == 0 else ZERO for e in range(SEAMS)]
        for _ in range(steps - 1)
    ]
    assert any(x != 0 for x in base_carrier.boundary(bad_current[0]))
    phi = constant_history(zero_ports(), steps)
    declared_rho = constant_history(zero_ports(), steps)
    run = lane.evolve(
        zero_seams(), zero_seams(), phi, bad_current, steps, verify=False
    )
    receipts = lane.verify_history(
        run["A"], phi, bad_current, rho=declared_rho, fail_closed=False
    )
    gauss = receipts["gauss"]
    assert gauss["initial"] is True
    assert gauss["continuity_all_steps"] is False
    assert gauss["propagated_all_steps"] is False
    assert gauss["first_gauss_failure_step"] == 1
    # The step equivalence itself still holds: Gauss breaks exactly where
    # continuity breaks.
    assert gauss["step_iff_consistent"] is True
    with pytest.raises(TemporalReceiptError, match="continuity violated"):
        lane.verify_history(
            run["A"], phi, bad_current, rho=declared_rho, fail_closed=True
        )


def test_declared_load_following_continuity_restores_gauss() -> None:
    # Other direction of the equivalence: the same nonconserved seam current
    # propagates the constraint once the declared load obeys continuity.
    lane = LeapfrogLane()
    steps = 6
    bad_current = [
        [ONE if e == 0 else ZERO for e in range(SEAMS)]
        for _ in range(steps - 1)
    ]
    phi = constant_history(zero_ports(), steps)
    rho_history = [zero_ports()]
    for n in range(steps - 1):
        drift = base_carrier.boundary(bad_current[n])
        rho_history.append(
            [rho_history[-1][p] - drift[p] for p in range(PORTS)]
        )
    run = lane.evolve(
        zero_seams(), zero_seams(), phi, bad_current, steps,
        rho=rho_history, verify=True,
    )
    gauss = run["receipts"]["gauss"]
    assert gauss["continuity_all_steps"] is True
    assert gauss["propagated_all_steps"] is True
    assert gauss["step_iff_consistent"] is True


# ---------------------------------------------------------------------------
# Gauge invariance
# ---------------------------------------------------------------------------

def test_time_dependent_gauge_transformation_leaves_everything_invariant() -> None:
    lane = LeapfrogLane()
    steps = 8
    phi = [[Fraction(3 * p - 2, 7) for p in range(PORTS)] for _ in range(steps)]
    current = constant_history(zero_seams(), steps - 1)
    run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, verify=False
    )
    chi = [
        [Fraction((n + 1) * (p + 2), 5) for p in range(PORTS)]
        for n in range(steps + 1)
    ]
    receipt = lane.gauge_invariance_receipt(run["A"], phi, current, chi)
    assert receipt == {
        "gauge_invariant": True,
        "steps": steps,
        "electric_invariant": True,
        "magnetic_invariant": True,
        "energy_invariant": True,
        "ampere_preserved": True,
    }


# ---------------------------------------------------------------------------
# Static Coulomb join, both directions
# ---------------------------------------------------------------------------

def test_static_coulomb_join_existence_direction() -> None:
    lane = LeapfrogLane()
    rho = [ZERO] * PORTS
    rho[0], rho[5], rho[9] = Fraction(2), Fraction(-1), Fraction(-1)
    receipt = lane.coulomb_static_solution(rho, steps=10)
    assert receipt["static_join"] is True
    assert receipt["existence_direction"] is True
    assert receipt["electric_equals_green_field"] is True
    assert receipt["magnetic_zero"] is True
    assert receipt["load"] == [str(x) for x in rho]


def test_static_join_uniqueness_direction_recovers_the_green_field() -> None:
    lane = LeapfrogLane()
    steps = 6
    rho = [ZERO] * PORTS
    rho[3], rho[8] = ONE, -ONE
    potential = green.green_potential(rho)
    phi = constant_history([-x for x in potential], steps)
    current = constant_history(zero_seams(), steps - 1)
    a = constant_history(zero_seams(), steps + 1)
    receipt = lane.static_join_receipt(a, phi, current)
    assert receipt["electric_equals_green_field"] is True
    assert receipt["neutral"] is True
    assert lane.electric_field(a, phi, 0) == green.coulomb_field(rho)


def test_static_join_rejects_a_nonconstant_history() -> None:
    lane = LeapfrogLane()
    steps = 4
    phi = constant_history(zero_ports(), steps)
    current = constant_history(zero_seams(), steps - 1)
    a = constant_history(zero_seams(), steps + 1)
    a[2] = face_zero_kick()
    with pytest.raises(TemporalReceiptError, match="not constant"):
        lane.static_join_receipt(a, phi, current)


# ---------------------------------------------------------------------------
# Wave recursion
# ---------------------------------------------------------------------------

def test_wave_recursion_holds_source_free_and_fails_with_a_source() -> None:
    lane = LeapfrogLane()
    steps = 10
    phi = [[Fraction(p - 5, 4) for p in range(PORTS)] for _ in range(steps)]
    current = constant_history(zero_seams(), steps - 1)
    run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, verify=False
    )
    receipt = lane.wave_recursion_receipt(run["A"], phi)
    assert receipt["wave_recursion_exact"] is True

    cycle = base_carrier.fundamental_cycle(CHORD_SEAMS[1])
    sourced = constant_history(cycle, steps - 1)
    sourced_run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, sourced, steps, verify=False
    )
    with pytest.raises(TemporalReceiptError, match="wave recursion"):
        lane.wave_recursion_receipt(sourced_run["A"], phi)


# ---------------------------------------------------------------------------
# Mutation guards
# ---------------------------------------------------------------------------

def test_sign_flip_in_c_is_caught_by_faraday_and_energy_receipts() -> None:
    flipped = [list(row) for row in base_carrier.face_incidence_matrix()]
    flipped[0][0] = -flipped[0][0]
    lane_bad = LeapfrogLane(face_matrix=flipped)
    steps = 8
    phi = [[Fraction(p * p + 1) for p in range(PORTS)] for _ in range(steps)]
    current = constant_history(zero_seams(), steps - 1)
    run = lane_bad.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, verify=False
    )
    receipts = lane_bad.verify_history(
        run["A"], phi, current, fail_closed=False
    )
    assert receipts["faraday_identity_exact"] is False
    assert receipts["energy"]["balance_exact"] is False
    assert receipts["codifferential_boundary_zero"] is False
    with pytest.raises(TemporalReceiptError):
        lane_bad.verify_history(run["A"], phi, current, fail_closed=True)


def test_energy_drift_detector_catches_a_tampered_history() -> None:
    lane = LeapfrogLane()
    steps = 8
    phi = constant_history(zero_ports(), steps)
    current = constant_history(zero_seams(), steps - 1)
    run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, verify=False
    )
    tampered = [list(vec) for vec in run["A"]]
    tampered[4][0] += ONE
    receipts = lane.verify_history(
        tampered, phi, current, fail_closed=False
    )
    assert receipts["ampere_exact"] is False
    assert (
        receipts["energy"]["balance_exact"] is False
        or receipts["energy"]["drift_zero"] is False
    )
    with pytest.raises(TemporalReceiptError):
        lane.verify_history(tampered, phi, current, fail_closed=True)


def test_evolve_is_fail_closed_by_default() -> None:
    # verify=True runs the full receipt set inside evolve; the committed
    # carrier passes, and the receipts land in the result.
    lane = LeapfrogLane()
    steps = 6
    phi = constant_history(zero_ports(), steps)
    current = constant_history(zero_seams(), steps - 1)
    rho = constant_history(zero_ports(), steps)
    run = lane.evolve(
        zero_seams(), face_zero_kick(), phi, current, steps, rho=rho
    )
    assert run["receipts"]["failures"] == []
    assert run["receipts"]["design_only"] is True
