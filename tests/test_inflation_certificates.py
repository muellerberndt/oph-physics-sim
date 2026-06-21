from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.inflation_certificates import (
    FORBIDDEN_DATA_USE_KEYS,
    emit_edge_center_certificate,
    emit_scalar_release_certificate_from_collar_run,
    inflation_certificate_bundle_report,
    write_inflation_certificate_bundle_report,
)


def test_inflation_certificate_report_keeps_gates_closed_when_missing(tmp_path: Path):
    report = write_inflation_certificate_bundle_report(tmp_path / "missing", tmp_path / "out", source_path=None)

    assert (tmp_path / "out" / "inflation_certificate_report.json").exists()
    assert (tmp_path / "out" / "inflation_certificate_report.md").exists()
    assert (tmp_path / "out" / "schemas" / "scalar_release.schema.json").exists()
    assert (tmp_path / "out" / "templates" / "scalar_release_certificate.template.json").exists()
    assert report["certificate_summary"]["found_count"] == 0
    assert report["certificate_summary"]["passed_count"] == 0
    assert report["inflation_certificate_stack_ready"] is False
    assert report["physical_cmb_prediction"] is False


def test_inflation_certificate_validators_recompute_toy_bundle(tmp_path: Path):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    _write_json(
        cert_dir / "scalar_release_certificate.json",
        {
            "id": "scalar",
            "type": "scalar_release",
            "release_packets": [
                {
                    "packet_id": "q0",
                    "scalar_visible": True,
                    "entropy": {"AB": 0.10, "BD": 0.20, "B": 0.05, "ABD": 0.10},
                },
                {
                    "packet_id": "q1",
                    "scalar_visible": True,
                    "entropy": {"AB": 0.20, "BD": 0.25, "B": 0.05, "ABD": 0.10},
                },
            ],
            "scalar_readout_normalization": {"kappa_rel": 2.0},
            "no_data_use_manifest": _no_data_use(),
        },
    )
    _write_json(
        cert_dir / "edge_center_certificate.json",
        {
            "id": "edge",
            "type": "edge_center",
            "P": P_STAR,
            "edge_center_basis": ["a", "b"],
            "scalar_event_diagonal": [1.0, 0.0],
            "z6_reserve_diagonal": [1.0, 0.0],
            "sector_local_generators": [{"name": "g0", "diagonal": [0.0, 1.0]}],
            "no_data_use_manifest": _no_data_use(),
        },
    )
    _write_json(
        cert_dir / "homogeneous_anomaly_certificate.json",
        {
            "id": "anomaly",
            "type": "homogeneous_anomaly",
            "refinement_levels": [
                {"level": 0, "a": 1.0, "V_com": 1.0, "ell_r": 1.0, "collars": [{"weight": 1.0, "cmi": 1.0}]},
                {
                    "level": 1,
                    "a": 1.0,
                    "V_com": 1.0,
                    "ell_r": 1.0,
                    "epsilon_r": 0.0,
                    "collars": [{"weight": 1.0, "cmi": 1.0}],
                },
            ],
            "no_data_use_manifest": _no_data_use(),
        },
    )
    _write_json(
        cert_dir / "parent_collar_certificate.json",
        {
            "id": "parent",
            "type": "parent_collar",
            "kernel_rows": [{"k": 0.1, "a": 1.0, "rho_b_bar": 2.0, "rho_A_bar": 10.0, "d_rho_A_eq_d_delta_b": 4.0}],
            "no_data_use_manifest": _no_data_use(),
        },
    )
    _write_json(
        cert_dir / "repair_matrix_certificate.json",
        {
            "id": "repair",
            "type": "repair_matrix",
            "transition_matrices": [
                {
                    "k": 0.1,
                    "a": 1.0,
                    "delta_eta": 0.5,
                    "stationary_distribution": [0.5, 0.5],
                    "K": [[0.9, 0.1], [0.1, 0.9]],
                }
            ],
            "no_data_use_manifest": _no_data_use(),
        },
    )
    _write_json(
        cert_dir / "boltzmann_handoff_certificate.json",
        {
            "id": "boltzmann",
            "type": "boltzmann_handoff",
            "required_outputs": {
                "background_A": "background_A.csv",
                "perturbation_A_grid": "perturbation_A_grid.h5",
                "B_A_grid": "B_A_grid.h5",
                "Gamma_rec_grid": "Gamma_rec_grid.h5",
                "primordial": "primordial.json",
                "neutrino_branch": "neutrino_branch.json",
                "solver_manifest": "solver_manifest.json",
            },
            "certificate_references": ["scalar", "edge", "anomaly", "parent", "repair", "neutrino"],
            "no_data_use_manifest": _no_data_use(),
        },
    )

    report = inflation_certificate_bundle_report(cert_dir, source_path=None)
    scalar = report["derived_outputs"]["scalar_release"]
    edge = report["derived_outputs"]["edge_center"]
    repair = report["derived_outputs"]["repair_matrix"]

    assert report["certificate_summary"]["passed_count"] == 6
    assert report["inflation_certificate_stack_ready"] is False
    assert report["readiness_gates"]["scalar_release_certificate"] is False
    assert math.isclose(scalar["epsilon_star"], 0.15)
    assert math.isclose(scalar["A_q_cmi_upper_bound"], 4.0 * math.log(2.0) * 2.0 * 0.15)
    assert scalar["A_q_energy"] is None
    assert scalar["A_zeta"] is None
    assert scalar["SCALAR_RELEASE_AMPLITUDE_CERTIFICATE"] is False
    assert math.isclose(edge["n_s"], 1.0 - P_STAR / 48.0)
    assert math.isclose(repair["mean_Gamma_rec"], -math.log(0.8) / 0.5)
    assert report["physical_cmb_prediction"] is False


def test_emit_scalar_release_certificate_from_collar_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": "fake_run_001",
            "patch_count": 4096,
            "claim_boundary": "test manifest",
        },
    )
    _write_json(
        run_dir / "collar_markov_report.json",
        {
            "mode": "diagonal_empirical_collar_state",
            "cap_count": 2,
            "median_epsilon_cmi": 0.04,
            "mean_epsilon_cmi": 0.045,
            "p90_epsilon_cmi": 0.07,
            "rows": [
                {
                    "cap_id": 0,
                    "theta0": 0.55,
                    "collar_width": 0.01,
                    "inside_count": 100,
                    "collar_count": 20,
                    "outside_count": 3976,
                    "epsilon_cmi": 0.05,
                    "sector_conditioned_cmi": {"identity": 0.02},
                    "sample_count": 4096,
                    "packet_alphabet_size": 16,
                },
                {
                    "cap_id": 1,
                    "theta0": 0.75,
                    "collar_width": 0.01,
                    "inside_count": 200,
                    "collar_count": 24,
                    "outside_count": 3872,
                    "epsilon_cmi": 0.03,
                    "sector_conditioned_cmi": {"identity": 0.01},
                    "sample_count": 4096,
                    "packet_alphabet_size": 18,
                },
            ],
        },
    )

    report = emit_scalar_release_certificate_from_collar_run(
        run_dir,
        tmp_path / "cert",
        kappa_rel=2.0,
        source_path=None,
    )
    cert = json.loads((tmp_path / "cert" / "scalar_release_certificate.json").read_text())
    scalar = report["derived_outputs"]["scalar_release"]

    assert cert["certificate_tier"] == "diagonal_collar_markov_proxy"
    assert cert["release_packets"][0]["packet_id"] == "cap_0"
    assert report["certificate_summary"]["found_count"] == 1
    assert report["certificate_summary"]["passed_count"] == 1
    assert report["inflation_certificate_stack_ready"] is False
    assert math.isclose(scalar["epsilon_star"], 0.03)
    assert math.isclose(scalar["A_q_cmi_upper_bound"], 4.0 * math.log(2.0) * 2.0 * 0.03)
    assert scalar["A_zeta"] is None


def test_scalar_release_rejects_a_zeta_shortcut(tmp_path: Path):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    _write_json(
        cert_dir / "scalar_release_certificate.json",
        {
            "id": "scalar",
            "type": "scalar_release",
            "release_packets": [{"packet_id": "q0", "scalar_visible": True, "cmi": 0.1}],
            "scalar_readout_normalization": {"kappa_rel": 1.0},
            "A_zeta": 2.1e-9,
            "Sachs_Wolfe_conversion_used": True,
            "no_data_use_manifest": _no_data_use(),
        },
    )

    report = inflation_certificate_bundle_report(cert_dir, source_path=None)
    validation = report["certificate_validations"]["scalar_release"]

    assert validation["validator_receipt"] is False
    assert "forbidden scalar-amplitude shortcut" in validation["reason"]


def test_emit_edge_center_certificate(tmp_path: Path):
    report = emit_edge_center_certificate(tmp_path / "cert", source_path=None)
    edge = report["derived_outputs"]["edge_center"]

    assert (tmp_path / "cert" / "edge_center_certificate.json").exists()
    assert report["certificate_summary"]["found_count"] == 1
    assert report["certificate_summary"]["passed_count"] == 1
    assert report["inflation_certificate_stack_ready"] is False
    assert math.isclose(edge["theta_OPH"], P_STAR / 48.0)
    assert math.isclose(edge["n_s"], 1.0 - P_STAR / 48.0)


def test_certificate_loader_ignores_templates_after_multiple_emits(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "manifest.json", {"run_id": "fake_run_002"})
    _write_json(
        run_dir / "collar_markov_report.json",
        {
            "mode": "diagonal_empirical_collar_state",
            "cap_count": 1,
            "rows": [{"cap_id": 0, "epsilon_cmi": 0.04, "sample_count": 128}],
        },
    )

    cert_dir = tmp_path / "cert"
    emit_scalar_release_certificate_from_collar_run(run_dir, cert_dir, source_path=None)
    report = emit_edge_center_certificate(cert_dir, source_path=None)

    assert report["certificate_summary"]["found_count"] == 2
    assert report["certificate_summary"]["passed_count"] == 2
    assert report["certificate_summary"]["missing_types"] == [
        "homogeneous_anomaly",
        "parent_collar",
        "repair_matrix",
        "boltzmann_handoff",
    ]


def _no_data_use() -> dict[str, object]:
    return {
        "observational_likelihoods_used": False,
        "used_data": [],
        "forbidden_data": list(FORBIDDEN_DATA_USE_KEYS),
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
