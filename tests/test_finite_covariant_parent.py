from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.finite_covariant_parent import (
    EXPLICIT_RECIPIENT_STRESS_RECEIPT,
    FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT,
    GAUGE_INDEPENDENCE_RECEIPT,
    PARENT_RECEIPT,
    STRESS_CLOSURE_RECEIPT,
    finite_covariant_collar_packet_parent_report,
    write_finite_covariant_collar_packet_parent_report,
)


def test_finite_covariant_parent_report_passes_closed_packet_artifact(tmp_path: Path):
    artifact = _artifact()

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is True
    assert report[STRESS_CLOSURE_RECEIPT] is True
    assert report[EXPLICIT_RECIPIENT_STRESS_RECEIPT] is True
    assert report[GAUGE_INDEPENDENCE_RECEIPT] is True
    assert report[FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT] is True
    assert report["physical_cmb_prediction"] is False
    assert report["blockers"] == []

    source = tmp_path / "parent.json"
    out = tmp_path / "finite_covariant_collar_packet_parent_report.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")

    written = write_finite_covariant_collar_packet_parent_report(source, out)

    assert written[PARENT_RECEIPT] is True
    assert out.exists()


def test_finite_covariant_parent_requires_recipient_stress_for_nonzero_gamma():
    artifact = _artifact()
    artifact["packets"]["states"] = [artifact["packets"]["states"][0]]

    report = finite_covariant_collar_packet_parent_report(artifact)

    assert report[PARENT_RECEIPT] is False
    assert report[EXPLICIT_RECIPIENT_STRESS_RECEIPT] is False
    assert "explicit_recipient_stress_missing_for_nonzero_Gamma_rec" in report["blockers"]


def _artifact() -> dict:
    return {
        "manifest": {
            "source_hash": "sha256:source",
            "regulator_id": "N64_eps0",
            "parent_theorem_version": "fccpp-v0",
        },
        "packets": {
            "states": [
                {"label": "anomaly", "rho": 0.2, "u_mu": [1.0, 0.0, 0.0, 0.0]},
                {"label": "recipient", "rho": 0.1, "u_mu": [1.0, 0.0, 0.0, 0.0]},
            ]
        },
        "repair": {"Gamma_rec": 0.05, "detailed_balance_residual": 0.0},
        "stress": {"stress_energy_closure_residual": 0.0},
        "causal_response": {
            "characteristic_speed_bound": 0.5,
            "kinetic_matrix": [[1.0]],
            "damping_matrix": [[1.0]],
            "propagation_matrix": [[0.5]],
            "source_matrix": [[1.0]],
            "output_matrix": [[1.0]],
        },
        "gauge": {"newtonian_synchronous_agree_within_error": True},
        "refinement": {"stress_and_response_converge": True},
        "cdm_limit": {"CDM_LIMIT_RECOVERY_RECEIPT": True},
        "frozen_run": {
            "source_hash": "sha256:source",
            "solver_hash": "sha256:solver",
            "likelihood_hash": "sha256:likelihood",
            "mutable_source_artifacts": False,
        },
    }
