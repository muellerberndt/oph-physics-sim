from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.ba_kernel import B_A_kernel_receipt, estimate_B_A_from_paired_runs
from oph_fpe.cosmology.ba_kernel import ba_kernel_report_from_parent_report, write_ba_kernel_refinement_report


def test_estimate_B_A_from_paired_runs_groups_k_a_bins():
    base = [
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.0, "baryon_source": 2.0},
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.2, "baryon_source": 2.0},
    ]
    perturbed = [
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.5, "baryon_source": 3.0},
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.8, "baryon_source": 3.0},
    ]

    rows = estimate_B_A_from_paired_runs(base, perturbed)

    assert len(rows) == 1
    assert rows[0].k_bin == 0.1
    assert rows[0].a_bin == 0.5
    assert rows[0].sample_count == 2
    assert rows[0].B_A > 0.0


def test_B_A_kernel_receipt_requires_enough_samples():
    rows = estimate_B_A_from_paired_runs(
        [{"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.0, "baryon_source": 2.0}],
        [{"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.5, "baryon_source": 3.0}],
    )

    report = B_A_kernel_receipt(rows, min_good_rows=1, min_sample_count=16)

    assert report["B_A_KERNEL_RECEIPT"] is False
    assert report["B_A_DIAGNOSTIC_CANDIDATE_RECEIPT"] is False
    assert report["B_A_k_a_physical_emitted"] is False


def test_B_A_kernel_receipt_candidate_does_not_promote_to_physical():
    base = [
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 1.0, "baryon_source": 1.0}
        for _ in range(16)
    ]
    perturbed = [
        {"k_bin": 0.1, "a_bin": 0.5, "repair_anomaly": 2.0, "baryon_source": 2.0}
        for _ in range(16)
    ]
    rows = estimate_B_A_from_paired_runs(base, perturbed)

    report = B_A_kernel_receipt(rows, min_good_rows=1, min_sample_count=16)

    assert report["B_A_DIAGNOSTIC_CANDIDATE_RECEIPT"] is True
    assert report["B_A_KERNEL_CANDIDATE_RECEIPT"] is True
    assert report["B_A_KERNEL_RECEIPT"] is False
    assert "common_source_functional_receipt_missing" in report["promotion_blockers"]


def test_ba_kernel_from_parent_report_keeps_candidate_separate_from_physical_receipt(tmp_path: Path):
    parent = tmp_path / "b_a_parent_report.json"
    parent.write_text(
        json.dumps(
            {
                "B_A_PARENT_RECEIPT": False,
                "rows": [
                    {
                        "k_h_mpc": 1.0,
                        "a": 0.5,
                        "B_A_mean": 0.2,
                        "B_A_sem": 0.01,
                        "mode_count": 2,
                    },
                    {
                        "k_h_mpc": 1.0,
                        "a": 0.5,
                        "B_A_mean": 0.4,
                        "B_A_sem": 0.01,
                        "mode_count": 2,
                    },
                ],
                "readiness": {
                    "checks": {
                        "paired_perturb_resettle_rows_emitted": True,
                        "finite_difference_rows_emitted": True,
                        "control_rows_emitted": True,
                        "no_cmb_data_used": True,
                        "real_baryon_perturbation_runs_present": True,
                        "full_perturbation_rerun": True,
                        "report_backed_surrogate_parent": False,
                        "controls_fail": True,
                        "sign_stable": True,
                        "scale_calibrated_k_h_mpc": False,
                        "calibrated_a_evolution": False,
                        "finite_observer_view_parent_variation": False,
                        "energy_momentum_exchange_closed": False,
                        "gauge_consistency_audited": False,
                        "refinement_convergence_passed": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = ba_kernel_report_from_parent_report(parent, tmp_path / "out")

    assert report["B_A_KERNEL_CANDIDATE_RECEIPT"] is True
    assert report["B_A_DIAGNOSTIC_CANDIDATE_RECEIPT"] is True
    assert report["B_A_KERNEL_RECEIPT"] is False
    assert report["row_count"] == 1
    assert report["B_A_k_a"][0][2] == 0.30000000000000004
    assert "physical_check_failed_scale_calibrated_k_h_mpc" in report["promotion_blockers"]
    assert "B_A_source_lift_independence_receipt_missing" in report["promotion_blockers"]
    assert (tmp_path / "out" / "B_A_kernel_report.json").exists()


def test_ba_kernel_refinement_report_separates_two_scale_diagnostic_from_convergence(tmp_path: Path):
    left = tmp_path / "screen_4k"
    right = tmp_path / "screen_16k"
    left.mkdir()
    right.mkdir()
    (left / "manifest.json").write_text(json.dumps({"patch_count": 4096}), encoding="utf-8")
    (right / "manifest.json").write_text(json.dumps({"patch_count": 16384}), encoding="utf-8")
    _write_parent_kernel(
        left / "b_a_parent_report.json",
        [
            {"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0},
            {"k_h_mpc": 0.2, "a": 0.5, "B_A_mean": -1.0},
        ],
    )
    _write_parent_kernel(
        right / "b_a_parent_report.json",
        [
            {"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.4},
            {"k_h_mpc": 0.2, "a": 0.5, "B_A_mean": 1.2},
        ],
    )

    report = write_ba_kernel_refinement_report(
        [left / "b_a_parent_report.json", right / "b_a_parent_report.json"],
        tmp_path / "out",
    )

    assert report["mode"] == "B_A_kernel_refinement_v0"
    assert report["patch_counts"] == [4096, 16384]
    assert report["two_scale_diagnostic_receipt"] is True
    assert report["B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT"] is False
    assert "requires_at_least_three_patch_counts_for_refinement_convergence" in report["blockers"]
    assert "B_A_kernel_pairwise_drift_or_sign_instability" in report["blockers"]
    assert report["pair_rows"][0]["common_key_count"] == 2
    assert report["pair_rows"][0]["pair_refinement_pass"] is False
    assert report["key_pair_row_count"] == 2
    assert report["key_pair_stable_fraction"] == 0.0
    assert (tmp_path / "out" / "B_A_kernel_refinement_report.json").exists()
    assert (tmp_path / "out" / "B_A_kernel_refinement_pairs.csv").exists()
    assert (tmp_path / "out" / "B_A_kernel_refinement_key_pairs.csv").exists()


def _write_parent_kernel(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "paired_cap_collar_perturb_resettle_B_A_parent_v0",
                "rows": rows,
                "readiness": {"checks": {"paired_B_A_diagnostic_receipt": True}},
            }
        ),
        encoding="utf-8",
    )
