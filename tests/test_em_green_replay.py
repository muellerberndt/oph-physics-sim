"""Green solve, handoff observables, and replay of the committed receipt.

Design-only tests.  The replay does not arm an instrument; the committed
decision-rule template stays unarmed; nothing is frozen.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from oph_fpe.em import base_carrier, green, replay
from oph_fpe.em.base_carrier import CHORD_SEAMS, PORTS, SEAMS

ZERO = Fraction(0)
ONE = Fraction(1)

RECEIPT_PATH = replay.default_receipt_path()

needs_committed_receipt = pytest.mark.skipif(
    not RECEIPT_PATH.is_file(),
    reason="committed RER receipt not present beside this checkout",
)


def load(pairs: dict[int, int]) -> list[Fraction]:
    rho = [ZERO] * PORTS
    for port, value in pairs.items():
        rho[port] = Fraction(value)
    return rho


DIPOLE = load({0: 1, 1: -1})
ANTIPODAL = load({3: 1, 8: -1})
THREE_PORT = load({0: 2, 5: -1, 9: -1})


# ---------------------------------------------------------------------------
# Exact Green solve
# ---------------------------------------------------------------------------

def test_green_matrix_satisfies_the_defining_identities() -> None:
    g = green.green_matrix()
    lap = base_carrier.laplacian_matrix()
    for i in range(PORTS):
        assert sum(g[i], start=ZERO) == 0
        for j in range(PORTS):
            assert g[i][j] == g[j][i]
            product = sum(
                (lap[i][k] * g[k][j] for k in range(PORTS)), start=ZERO
            )
            assert product == (ONE if i == j else ZERO) - Fraction(1, 12)


def test_dipole_potential_matches_committed_values() -> None:
    phi = green.green_potential(DIPOLE)
    committed = [
        "11/60", "-11/60", "0", "0", "1/20", "-1/20",
        "1/20", "-1/20", "0", "0", "1/60", "-1/60",
    ]
    assert [str(x) for x in phi] == committed


def test_thomson_energies_match_committed_values() -> None:
    expected = {
        tuple(DIPOLE): ("11/30", "1", "19/30"),
        tuple(ANTIPODAL): ("1/2", "3", "5/2"),
        tuple(THREE_PORT): ("7/5", "4", "13/5"),
    }
    for rho, (coulomb, tree, cycle) in expected.items():
        result = green.thomson_decomposition(list(rho))
        assert str(result["coulomb_energy"]) == coulomb
        assert str(result["tree_solution_energy"]) == tree
        assert str(result["cycle_part_energy"]) == cycle


def test_non_neutral_load_fails_closed() -> None:
    rho = [ZERO] * PORTS
    rho[0] = ONE
    with pytest.raises(green.GreenSolveError):
        green.green_potential(rho)


# ---------------------------------------------------------------------------
# The three handoff observables
# ---------------------------------------------------------------------------

def test_port_potential_difference_definition_and_antisymmetry() -> None:
    phi = green.green_potential(DIPOLE)
    for p in range(PORTS):
        for q in range(PORTS):
            value = green.port_potential_difference(DIPOLE, p, q)
            assert value == phi[p] - phi[q]
            assert value == -green.port_potential_difference(DIPOLE, q, p)


def test_seam_flux_satisfies_gauss_on_all_committed_loads() -> None:
    for rho in (DIPOLE, ANTIPODAL, THREE_PORT):
        flux = green.seam_flux(rho)
        assert len(flux) == SEAMS
        assert base_carrier.boundary(flux) == rho


def test_chord_field_strength_is_zero_on_the_coulomb_field() -> None:
    for rho in (DIPOLE, ANTIPODAL, THREE_PORT):
        components = green.chord_field_strength_components(
            green.coulomb_field(rho)
        )
        assert set(components) == set(CHORD_SEAMS)
        assert all(value == 0 for value in components.values())


def test_field_strength_is_gauge_invariant_and_fixes_cycles() -> None:
    seam_field = [Fraction(3 * e - 7, 11) for e in range(SEAMS)]
    chi = [Fraction(p * p - 4, 5) for p in range(PORTS)]
    gauged = [
        a + g
        for a, g in zip(
            seam_field, base_carrier.coboundary(chi), strict=True
        )
    ]
    assert green.field_strength(gauged) == green.field_strength(seam_field)
    chord = CHORD_SEAMS[0]
    cycle = base_carrier.fundamental_cycle(chord)
    assert green.field_strength(cycle) == cycle
    gradient = base_carrier.coboundary(chi)
    assert green.field_strength(gradient) == [ZERO] * SEAMS


def test_tree_solution_chord_components_equal_the_cycle_part() -> None:
    result = green.thomson_decomposition(THREE_PORT)
    components = green.chord_field_strength_components(result["tree"])
    for chord in CHORD_SEAMS:
        assert components[chord] == result["cycle_part"][chord]


# ---------------------------------------------------------------------------
# Replay of the committed receipt
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def replay_report() -> dict:
    return replay.replay_committed_receipt()


@needs_committed_receipt
def test_replay_verdict_is_replicated(replay_report: dict) -> None:
    assert replay_report["verdict"] == "REPLICATED"
    assert replay_report["mismatch_count"] == 0
    assert replay_report["mismatches"] == []


@needs_committed_receipt
def test_replay_covers_the_three_declared_observables(
    replay_report: dict,
) -> None:
    observables = replay_report["observables"]
    assert observables["port_potential_difference"] == {
        "comparisons": 396,
        "mismatches": 0,
    }
    assert observables["seam_flux"] == {"comparisons": 90, "mismatches": 0}
    assert observables["chord_field_strength_component"] == {
        "comparisons": 114,
        "mismatches": 0,
    }
    assert replay_report["declared_comparisons_total"] == 600
    assert replay_report["supporting_comparisons_total"] == 714


@needs_committed_receipt
def test_replay_stays_design_only_and_unarmed(replay_report: dict) -> None:
    assert replay_report["design_only"] is True
    assert replay_report["frozen"] is False
    assert replay_report["instrument_armed"] is False
    assert replay_report["template_armed"] is False
    template = replay_report["decision_rule_template"]
    assert template["template_only"] is True
    assert template["not_a_freeze"] is True
    assert template["armed"] is False
    assert template["verdict_labels"] == [
        "REPLICATED",
        "FAILED",
        "INCONCLUSIVE",
    ]


@needs_committed_receipt
def test_wrong_green_matrix_entry_fails_the_replay() -> None:
    mutated = green.green_matrix()
    mutated[0][0] += Fraction(1, 180)
    report = replay.replay_committed_receipt(green_override=mutated)
    assert report["verdict"] == "FAILED"
    assert report["mismatch_count"] > 0
    assert report["observables"]["port_potential_difference"]["mismatches"] > 0
    assert report["observables"]["seam_flux"]["mismatches"] > 0


def test_missing_receipt_is_inconclusive(tmp_path: Path) -> None:
    report = replay.replay_committed_receipt(
        receipt_path=tmp_path / "absent.json"
    )
    assert report["verdict"] == "INCONCLUSIVE"
    assert "missing committed receipt" in report["reason"]


@needs_committed_receipt
def test_tampered_receipt_copy_is_inconclusive_pin_drift(
    tmp_path: Path,
) -> None:
    data = json.loads(RECEIPT_PATH.read_text())
    data["green_matrix_times_180"][0][0] = 36
    target = tmp_path / "tampered.json"
    target.write_bytes(replay._canonical_json_bytes(data))
    report = replay.replay_committed_receipt(receipt_path=target)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "self-digest drift" in report["reason"]


def test_invalid_json_is_inconclusive(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{not json")
    report = replay.replay_committed_receipt(receipt_path=target)
    assert report["verdict"] == "INCONCLUSIVE"
