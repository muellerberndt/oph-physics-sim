from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from oph_fpe.uhe_coefficients import (
    binary_logit_coefficient,
    coefficient_emission_report,
    poisson_opportunity_coefficient,
    small_signal_coefficient,
    write_uhe_coefficient_emission_report,
)


def test_default_uhe_coefficient_emitter_is_source_only_fixture() -> None:
    report = coefficient_emission_report()

    assert report["artifact_type"] == "UHE_COEFFICIENT_EMISSION_RECEIPT"
    assert report["schema"] == "oph_uhe_coefficient_emission_v2"
    assert report["status"] == "synthetic_demo_fixture"
    assert report["evidence_classification"]["label"] == "SYNTHETIC_DEMO_FIXTURE"
    assert report["evidence_classification"]["coefficient_evidence"] == "PLANTED_COEFFICIENT_RECOVERY_ONLY"
    assert report["evidence_classification"]["target_moment_origin"] == (
        "SYNTHETIC_MOMENTS_FROM_PLANTED_COEFFICIENTS"
    )
    assert report["claim_tier"] == "DEMO"
    assert report["strongest_allowed_claim"] == "SYNTHETIC_DEMO_FIXTURE_RECOVERED"
    assert report["physical_claim"] is False
    assert report["source_only"] is True
    assert report["readiness_gates"]["NO_UHE_DATA_USE"] is True
    assert report["readiness_gates"]["COMMON_SOURCE_LOCK"] is True
    assert report["solver"]["converged"] is True
    assert report["source_law"]["finite_support"] == {
        "kind": "finite_enumerated_support",
        "notation": "U_finite = {u_0, ..., u_7}",
        "cardinality": 8,
        "feature_dimension": 5,
        "baseline_full_support": True,
    }
    assert report["source_law"]["weight_formula"].startswith("p_eta(u) = m_0(u)")
    np.testing.assert_allclose(report["coefficients"], [0.25, 0.35, -0.20, 0.10, 0.05], atol=1.0e-8)


def test_explicit_uhe_source_inputs_are_not_reclassified_as_synthetic_fixture() -> None:
    report = coefficient_emission_report(
        features=[[0.0], [1.0]],
        baseline_weights=[1.0, 1.0],
        target_moments=[0.7],
        source_dag={"declared_inputs": ["source_model_moments"]},
    )

    assert report["status"] == "declared_source_inputs"
    assert report["evidence_classification"]["label"] == "DECLARED_SOURCE_INPUTS"
    assert report["evidence_classification"]["synthetic_demo"] is False
    assert report["claim_tier"] == "SOURCE_ONLY"
    assert report["strongest_allowed_claim"] == "SOURCE_ONLY_COEFFICIENT_EMITTED"
    assert report["physical_claim"] is False
    assert report["source_law"]["finite_support"]["cardinality"] == 2


def test_closed_form_corollaries() -> None:
    assert math.isclose(
        binary_logit_coefficient(0.7, 0.4),
        math.log(0.7 * 0.6 / (0.4 * 0.3)),
    )
    assert math.isclose(poisson_opportunity_coefficient(6.0, 2.0), math.log(3.0))
    assert small_signal_coefficient([[2.0, 0.0], [0.0, 4.0]], [1.0, 2.0]) == [0.5, 0.5]


def test_uhe_source_dag_with_event_data_invalidates_source_only_label() -> None:
    report = coefficient_emission_report(source_dag={"inputs": ["event_coordinates", "likelihood_values"]})

    assert report["claim_tier"] == "INVALIDATED"
    assert report["strongest_allowed_claim"] == "INVALIDATED_COEFFICIENT_DAG"
    assert report["readiness_gates"]["NO_UHE_DATA_USE"] is False
    assert "event_coordinates" in report["source_classifier"]["target_leak_hits"]


def test_uhe_nonminimal_features_fail_closed() -> None:
    report = coefficient_emission_report(
        features=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        baseline_weights=[1.0, 1.0, 1.0],
        target_moments=[0.5, 0.5],
    )

    assert report["claim_tier"] == "CONDITIONAL"
    assert report["readiness_gates"]["FEATURE_MINIMALITY"] is False
    assert "feature_nonminimal_remove_redundant_columns" in report["blockers"]


def test_uhe_moment_outside_polytope_fail_closed() -> None:
    report = coefficient_emission_report(
        features=[[0.0], [1.0], [2.0]],
        baseline_weights=[1.0, 1.0, 1.0],
        target_moments=[3.0],
    )

    assert report["readiness_gates"]["MOMENT_INTERIOR"] is False
    assert report["readiness_gates"]["COEFFICIENT_SOLVE_CONVERGED"] is False
    assert "target_moment_not_in_relative_interior" in report["blockers"]


def test_uhe_common_source_lock_blocks_species_separate_coefficients() -> None:
    report = coefficient_emission_report(
        species_coefficients={
            "neutrino": [0.1, 0.2],
            "cosmic_ray": [0.1, 0.2],
            "gamma": [0.1, 0.3],
        }
    )

    assert report["claim_tier"] == "CONDITIONAL"
    assert report["readiness_gates"]["COMMON_SOURCE_LOCK"] is False
    assert "separate_messenger_coefficients_break_common_source_lock" in report["blockers"]


def test_write_uhe_coefficient_emission_report(tmp_path: Path) -> None:
    report = write_uhe_coefficient_emission_report(tmp_path / "uhe")

    assert report["schema"] == "oph_uhe_coefficient_emission_v2"
    assert (tmp_path / "uhe" / "uhe_coefficient_emission_report.json").exists()
    assert (tmp_path / "uhe" / "uhe_coefficient_emission_report.md").exists()
    parsed = json.loads((tmp_path / "uhe" / "uhe_coefficient_emission_report.json").read_text(encoding="utf-8"))
    assert parsed["claim_tier"] == "DEMO"
    assert parsed["status"] == "synthetic_demo_fixture"
    assert parsed["evidence_classification"]["label"] == "SYNTHETIC_DEMO_FIXTURE"
