from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import fz11_conservative_time_lift as producer
from oph_fpe.dynamics.verify_fz11_conservative_time_lift_independent import (
    IndependentVerificationError,
    verify_receipt as verify_independent,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/repair_closure/fz11_conservative_time_lift_receipt.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _write_mutation(tmp_path: Path, report: dict, name: str = "mutated.json") -> Path:
    _rehash(report)
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _set_path(report: dict, path: tuple[object, ...], value: object) -> None:
    cursor: object = report
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


@pytest.fixture(scope="module")
def canonical() -> dict:
    return producer.load_receipt_strict(RECEIPT)


def test_canonical_packet_replays_and_verifies(canonical: dict) -> None:
    assert _canonical_bytes(producer.produce_receipt()) == _canonical_bytes(canonical)
    assert producer.verify_receipt(canonical)["receipt"] is True
    result = verify_independent(RECEIPT)
    assert result["receipt"] is True
    assert result["exact_rational_fixture"] is True
    assert result["checked_J_dimension"] == 14
    assert result["checked_laurent_terms"] == 13
    assert result["declared_orientation_permutation_presentations"] == 46080
    assert result["checked_orientation_generator_fixture"] is True
    assert result["energy_uniqueness_mutation_rejected"] is True
    assert result["producer_imported"] is False
    assert result["comparison_data_read"] is False


def test_exact_factorization_and_continuous_mode_relation(canonical: dict) -> None:
    fixture = canonical["declared_finite_factorization"]
    assert fixture["q_dimension"] == 2
    assert fixture["p_dimension"] == 12
    assert fixture["positive_axis_count"] == 6
    assert fixture["K_equals_B_star_B"] == [["6", "-6"], ["-6", "6"]]
    assert fixture["exact_modes"]["constant"]["eigenvalue"] == "0"
    assert fixture["exact_modes"]["alternating"]["eigenvalue"] == "12"
    assert fixture["identities"] == {
        "B_star_is_weighted_adjoint": True,
        "K_equals_B_star_B": True,
        "K_positive_semidefinite": True,
        "J_B_skew_adjoint": True,
        "J_B_squared_equals_diag_minus_K_minus_B_B_star": True,
    }
    continuous = canonical["continuous_time_lift"]
    assert continuous["coordinate_equation"] == "q_second_derivative+K*q=0"
    assert continuous["frozen_relation"] == "omega^2=lambda"
    assert continuous["conditional_continuous_evolution_supplied"] is True
    assert continuous["physical_clock_selected"] is False


def test_direct_factorization_is_canonical_only_after_translations(
    canonical: dict,
) -> None:
    binding = canonical["frozen_operator_binding"]
    assert binding["translation_action_source_selected"] is False
    assert (
        binding["factorization_arbitrary_within_declared_direct_incidence_class"]
        is False
    )
    assert (
        binding[
            "generic_psd_factorizations_remain_nonunique_outside_declared_class"
        ]
        is True
    )
    assert (
        binding["direct_factorization_canonical_up_to_momentum_frame_isometry"]
        is True
    )
    uniqueness = canonical["direct_factorization_uniqueness"]
    assert uniqueness["B_prime_star_B_prime_equals_B_star_B"] is True
    assert uniqueness["J_B_prime_equals_diag_I_U_J_B_diag_I_U_inverse"] is True
    assert uniqueness["labeled_presentation_operation_count"] == 46080
    assert "does not select the translations" in uniqueness["scope_boundary"]


def test_energy_conservation_selects_K_only_under_named_energy_premise(
    canonical: dict,
) -> None:
    row = canonical["time_law_uniqueness_from_energy"]
    assert row["declared_hypotheses"] == [
        "finite-dimensional real configuration space",
        "linear second-order law q_second_derivative+A*q=0",
        "canonical velocity inner product in the kinetic term",
        "self-adjoint frozen stiffness K",
        "conservation of E_K for every initial displacement and velocity",
    ]
    assert row["derivative_identity"] == "dE_K/dt=<v,(K-A)q>"
    assert row["A_equals_K_forced_under_declared_energy_premise"] is True
    assert row["exact_mutation_control"]["dE_K_dt_at_initial_state"] == "-1"
    assert row["exact_mutation_control"]["conservation_violated"] is True
    assert row["complete_response_energy_identified_with_E_K"] is False
    assert row["E_K_identified_with_conserved_phase_norm"] is False
    assert row["physical_clock_selected"] is False


def test_leapfrog_is_separate_and_sine_modified(canonical: dict) -> None:
    row = canonical["discrete_time_audit"]
    assert row["separate_from_continuous_packet"] is True
    assert row["stable_phase_relation"] == "4*h^-2*sin(theta/2)^2=lambda"
    assert row["omega_tilde_squared_identified_with_lambda"] is False
    assert row["leapfrog_map_identified_with_exp_h_J_B"] is False
    assert row["physical_discrete_clock_selected"] is False
    assert row["repair_tick_supplies_physical_time"] is False
    assert row["clock_or_continuum_theorem_required_for_physical_time"] is True
    exact = row["exact_fixture"]
    assert exact["h_squared_lambda"] == "3"
    assert exact["characteristic_polynomial"] == "r^2+r+1=0"
    assert exact["principal_stable_phase"] == "theta=2*pi/3"
    assert exact["frequencies_identified"] is False


def test_phase_norm_is_not_promoted_to_energy_or_clock(canonical: dict) -> None:
    row = canonical["continuous_time_lift"]
    assert (
        row["conserved_phase_norm_identified_with_canonical_second_order_energy"]
        is False
    )
    assert row["conserved_phase_norm_identified_with_physical_energy"] is False
    assert row["conserved_phase_norm_selects_physical_clock"] is False


def test_all_physical_boundaries_remain_closed(canonical: dict) -> None:
    assert canonical["comparison_data_read"] is False
    assert canonical["target_data_read"] is False
    assert canonical["target_data_paths"] == []
    attainment = canonical["attainment"]
    for key in (
        "B_source_selected",
        "phase_norm_identified_with_physical_energy",
        "physical_clock_selected",
        "lorentz_or_boost_law_derived",
        "physical_field_sector_selected",
        "continuum_limit_derived",
        "physical_scale_selected",
        "physical_readout_selected",
        "physical_prediction_promoted",
        "comparison_permitted",
        "issue_655_closure_supported",
    ):
        assert attainment[key] is False


def test_independent_verifier_imports_no_producer() -> None:
    path = (
        ROOT
        / "oph_fpe/dynamics/verify_fz11_conservative_time_lift_independent.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert "oph_fpe.dynamics.fz11_conservative_time_lift" not in imports
    assert not any(name.startswith("oph_fpe") for name in imports)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("declared_finite_factorization", "B", 0, 0), "0"),
        (("declared_finite_factorization", "B_star", 0, 0), "0"),
        (("declared_finite_factorization", "K_equals_B_star_B", 0, 0), "7"),
        (("declared_finite_factorization", "J_B", 0, 2), "-1/2"),
        (("declared_finite_factorization", "J_B_squared", 0, 0), "-11"),
        (("frozen_operator_binding", "translation_action_source_selected"), True),
        (
            (
                "frozen_operator_binding",
                "factorization_arbitrary_within_declared_direct_incidence_class",
            ),
            True,
        ),
        (
            (
                "frozen_operator_binding",
                "generic_psd_factorizations_remain_nonunique_outside_declared_class",
            ),
            False,
        ),
        (("continuous_time_lift", "physical_clock_selected"), True),
        (
            (
                "continuous_time_lift",
                "conserved_phase_norm_identified_with_physical_energy",
            ),
            True,
        ),
        (("continuous_time_lift", "frozen_relation"), "omega=lambda"),
        (("time_law_uniqueness_from_energy", "complete_response_energy_identified_with_E_K"), True),
        (("time_law_uniqueness_from_energy", "exact_mutation_control", "dE_K_dt_at_initial_state"), "0"),
        (("discrete_time_audit", "omega_tilde_squared_identified_with_lambda"), True),
        (("discrete_time_audit", "repair_tick_supplies_physical_time"), True),
        (("discrete_time_audit", "stable_phase_relation"), "omega_tilde^2=lambda"),
        (("attainment", "physical_field_sector_selected"), True),
        (("attainment", "lorentz_or_boost_law_derived"), True),
        (("attainment", "continuum_limit_derived"), True),
        (("attainment", "physical_scale_selected"), True),
        (("attainment", "comparison_permitted"), True),
        (("parent_pin", "sha256"), "sha256:" + "0" * 64),
        (("continuous_time_lift", "flow_parameter"), "physical time"),
        (("direct_factorization_uniqueness", "scope_boundary"), "globally unique"),
        (("claim_boundary",), "This is a physical photon prediction."),
    ],
)
def test_semantic_and_exact_mutations_fail_closed(
    tmp_path: Path,
    canonical: dict,
    path: tuple[object, ...],
    value: object,
) -> None:
    changed = copy.deepcopy(canonical)
    _set_path(changed, path, value)
    mutated = _write_mutation(tmp_path, changed)
    with pytest.raises(IndependentVerificationError):
        verify_independent(mutated)
    with pytest.raises(producer.ConservativeTimeLiftError):
        producer.load_receipt_strict(mutated)


def test_unknown_target_field_fails_closed(tmp_path: Path, canonical: dict) -> None:
    changed = copy.deepcopy(canonical)
    changed["observed_particle_mass"] = "forbidden"
    with pytest.raises(IndependentVerificationError, match="top-level schema drift"):
        verify_independent(_write_mutation(tmp_path, changed))


def test_duplicate_key_and_nonfinite_constants_fail_closed(
    tmp_path: Path,
    canonical: dict,
) -> None:
    rendered = json.dumps(canonical, sort_keys=True)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(rendered[:-1] + ',"issue":662}', encoding="utf-8")
    with pytest.raises(IndependentVerificationError, match="duplicate JSON key"):
        verify_independent(duplicate)
    with pytest.raises(producer.ConservativeTimeLiftError, match="duplicate JSON key"):
        producer.load_receipt_strict(duplicate)

    for token in ("NaN", "Infinity", "-Infinity"):
        malformed = tmp_path / f"nonfinite-{token.replace('-', 'minus')}.json"
        malformed.write_text(
            rendered[:-1] + f',"forbidden":{token}}}',
            encoding="utf-8",
        )
        with pytest.raises(IndependentVerificationError, match="non-finite"):
            verify_independent(malformed)
        with pytest.raises(producer.ConservativeTimeLiftError, match="non-finite"):
            producer.load_receipt_strict(malformed)


def test_both_clis_accept_the_canonical_receipt() -> None:
    producer_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.fz11_conservative_time_lift",
            "--verify",
            str(RECEIPT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(producer_result.stdout)["receipt"] is True
    independent_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_fz11_conservative_time_lift_independent",
            "--receipt",
            str(RECEIPT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(independent_result.stdout)["receipt"] is True
