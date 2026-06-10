from __future__ import annotations

from oph_fpe.cosmology.ba_kernel import B_A_kernel_receipt, estimate_B_A_from_paired_runs


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
    assert report["B_A_k_a_physical_emitted"] is False
