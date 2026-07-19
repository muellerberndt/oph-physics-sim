from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.edge_center_clock import edge_center_clock_target
from oph_fpe.cosmology.finite_repair_transition_clock import (
    validate_transition_clock_eligibility,
    write_finite_repair_transition_clock_report,
    write_finite_repair_transition_clock_sweep_report,
)
from oph_fpe.cosmology.repair_clock import repair_clock_report


def test_finite_repair_transition_clock_writes_matrix_and_scalar_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
    )

    report = write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    assert report["finite_transition_matrix_ready"] is True
    assert report["state_count"] == 2
    assert report["transition_count"] == 9
    assert report["primary"]["lambda_2"] is not None
    assert report["clock_modes"]["empirical"]["clock_mode"] == "empirical"
    assert report["clock_modes"]["edge_center_selected"]["clock_mode"] == "edge_center_selected"
    assert report["clock_modes"]["e_diagnostic"]["clock_mode"] == "e_diagnostic"
    assert report["clock_modes"]["e_diagnostic"]["promoting"] is False
    assert report["repair_clock_empirical_certificate"] is False
    assert report["eta_R_empirical_finite_lattice_derived"] is False
    assert report["finite_step_survival_exponent_derived"] is True
    assert report["finite_step_survival"]["distinct_from_full_collar_derivative"] is True
    assert report["repair_clock_edge_center_certificate"] is False
    assert report["EDGE_CENTER_CLOCK_RECEIPT"] is False
    target = edge_center_clock_target()
    assert report["target"]["required_eta_R"] == target.theta
    assert report["target"]["required_kappa_rep"] == target.kappa_rep
    assert (out / "finite_repair_transition_matrix.npz").exists()
    assert (out / "finite_repair_transition_matrix_report.json").exists()
    assert (out / "scalar_repair_semigroup_report.json").exists()


def test_finite_transition_clock_is_visible_to_repair_clock_but_not_auto_certified(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
    )
    write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    aggregate = repair_clock_report([out])

    assert aggregate["summary"]["estimator_count"] == 1
    assert aggregate["rows"][0]["estimator"] == "scalar_repair_semigroup_gap"
    assert aggregate["rows"][0]["source"] == "finite_state_transition_matrix"
    assert aggregate["rows"][0]["finite_lattice_derived"] is True
    assert aggregate["finite_repair_clock_certificate"] is False


def test_finite_transition_clock_sweep_reports_best_quotient(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
    )

    report = write_finite_repair_transition_clock_sweep_report(
        run,
        out,
        packet_fieldsets=(
            ("checkpoint", ("checkpoint_class",)),
            ("checkpoint_sector", ("checkpoint_class", "s3_sector_class")),
        ),
        primary_matrices=("raw_empirical", "reversible_empirical"),
        repair_step_times=(1.0, 2.0),
    )

    assert report["summary"]["row_count"] == 8
    assert report["summary"]["finite_ready_count"] >= 1
    assert report["summary"]["best_finite_row"]["field_set_name"] in {"checkpoint", "checkpoint_sector"}
    assert report["physical_cmb_prediction"] is False
    assert (out / "finite_repair_transition_clock_sweep_report.json").exists()
    assert (out / "finite_repair_transition_clock_sweep_rows.csv").exists()


def test_reducible_transition_chain_cannot_emit_physical_clock_certificate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 0],
            [1, 1, 1],
        ],
    )

    report = write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    assert report["state_count"] == 2
    assert report["primary"]["irreducible"] is False
    assert report["primary"]["aperiodic"] is False
    assert report["primary"]["detailed_balance_max_abs_error"] is None
    assert report["finite_transition_matrix_ready"] is False
    assert report["finite_lattice_derived"] is False
    assert report["repair_clock_empirical_certificate"] is False
    assert report["eta_R_empirical_finite_lattice_derived"] is False
    assert report["physical_cmb_eligible_eta_R_empirical"] is False
    assert any("reducible" in blocker for blocker in report["blockers"])
    assert any("aperiodic" in blocker for blocker in report["blockers"])


def test_periodic_transition_chain_cannot_emit_physical_clock_certificate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
    )

    report = write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    assert report["primary"]["irreducible"] is True
    assert report["primary"]["aperiodic"] is False
    assert report["primary"]["lambda_2"] == 1.0
    assert report["finite_transition_matrix_ready"] is False
    assert report["repair_clock_empirical_certificate"] is False
    assert report["physical_cmb_eligible_eta_R_empirical"] is False
    assert any("aperiodic" in blocker for blocker in report["blockers"])


def test_nonreversible_transition_chain_cannot_emit_physical_clock_certificate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1, 2, 2, 0, 0],
        ],
    )

    report = write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    assert report["primary"]["irreducible"] is True
    assert report["primary"]["aperiodic"] is True
    assert report["primary"]["lambda_2"] < 1.0
    assert report["primary"]["detailed_balance_max_abs_error"] > 1.0e-12
    assert report["finite_transition_matrix_ready"] is False
    assert report["repair_clock_empirical_certificate"] is False
    assert report["physical_cmb_eligible_eta_R_empirical"] is False
    assert any("reversible/GNS" in blocker for blocker in report["blockers"])


def test_cached_true_flags_cannot_override_invalid_transition_chain_metadata() -> None:
    stale_report = {
        "finite_transition_matrix_ready": True,
        "finite_lattice_derived": True,
        "physical_cmb_eligible_eta_R_empirical": True,
        "state_count": 2,
        "transition_count": 48,
        "primary": {
            "finite": True,
            "irreducible": False,
            "aperiodic": False,
            "lambda_2": 1.0,
            "detailed_balance_max_abs_error": None,
        },
    }

    eligibility = validate_transition_clock_eligibility(stale_report)

    assert eligibility["eligible"] is False
    assert eligibility["legacy_ready_flags_ignored"]["finite_transition_matrix_ready"] is True
    assert "primary_irreducible" in eligibility["blockers"]
    assert "primary_aperiodic" in eligibility["blockers"]
    assert "primary_spectral_gap" in eligibility["blockers"]
    assert "primary_detailed_balance" in eligibility["blockers"]


def test_transition_clock_eligibility_accepts_complete_valid_metadata() -> None:
    eligibility = validate_transition_clock_eligibility(
        {
            "state_count": 2,
            "transition_count": 48,
            "primary": {
                "finite": True,
                "irreducible": True,
                "aperiodic": True,
                "lambda_2": 0.5,
                "detailed_balance_max_abs_error": 0.0,
            },
        }
    )

    assert eligibility["eligible"] is True
    assert eligibility["blockers"] == []


def test_repair_clock_consumer_rejects_legacy_scalar_sidecar_without_raw_evidence(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "scalar_repair_semigroup_report.json").write_text(
        json.dumps(
            {
                "source": "finite_state_transition_matrix",
                "finite_lattice_derived": True,
                "eligible_for_repair_clock_certificate": True,
                "repair_clock_certificate": True,
                "semigroup_controls_passed": True,
                "semigroup": {
                    "kappa_rep_estimate": 2.627023712627471,
                    "eta_R_estimate": 0.033978504362582485,
                },
                "transition_matrix_certificate": {"matrix_ready": True},
            }
        ),
        encoding="utf-8",
    )

    report = repair_clock_report([run])

    assert report["summary"]["estimator_count"] == 1
    assert report["rows"][0]["finite_lattice_derived"] is False
    assert report["rows"][0]["eligible_for_certificate"] is False
    assert report["rows"][0]["transition_clock_eligibility"]["eligible"] is False


def _write_observer_views(path: Path, paths: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for observer_id, values in enumerate(paths):
            steps = [
                {
                    "checkpoint_class": value,
                    "stable_flag": 1,
                    "s3_sector_class": 1,
                    "repair_load_bucket": 0,
                }
                for value in values
            ]
            handle.write(
                json.dumps(
                    {
                        "observer_id": observer_id,
                        "transition_history_mean_modal_mass": 1.0,
                        "transition_history_descriptor": {
                            "fields": [
                                "checkpoint_class",
                                "stable_flag",
                                "s3_sector_class",
                                "repair_load_bucket",
                            ],
                            "steps": steps,
                        },
                    }
                )
                + "\n"
            )
