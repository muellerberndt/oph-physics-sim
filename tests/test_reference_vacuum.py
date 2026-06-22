from pathlib import Path

import numpy as np

from oph_fpe.ensembles.reference_vacuum import (
    free_scalar_ensemble_spec,
    harmonic_gaussian_reference_report,
    sample_harmonic_coefficients,
    u1_lattice_gauge_reference_report,
    write_reference_vacuum_baseline_report,
)


def test_free_scalar_reference_keeps_vacuum_promotion_closed() -> None:
    report = harmonic_gaussian_reference_report(
        ell_max=5,
        sample_count=2048,
        amplitude=2.0,
        theta=0.25,
        seed_key="unit-free-scalar",
        smoothing_sigma=0.05,
        coarse_ell_max=4,
    )

    assert report["claim_tier"] == "E1"
    assert report["partition_randomness"]["partition_replay_receipt"] is True
    assert report["refinement_diagnostics"]["exact_mode_truncation_refinement_receipt"] is True
    assert report["reference_theory_regression_receipt"] is True
    assert report["OPH_NATIVE_VACUUM_PROMOTION_RECEIPT"] is False
    assert report["OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT"] is False
    assert report["covariance_diagnostics"]["integrated_autocorrelation_time"] == 1.0


def test_harmonic_reference_replays_shared_modes_across_regulators() -> None:
    fine = free_scalar_ensemble_spec(ell_max=6, amplitude=1.0, theta=0.1)
    coarse = free_scalar_ensemble_spec(ell_max=4, amplitude=1.0, theta=0.1)
    fine_modes, _, fine_coeffs = sample_harmonic_coefficients(fine, sample_count=8, seed_key="shared")
    coarse_modes, _, coarse_coeffs = sample_harmonic_coefficients(coarse, sample_count=8, seed_key="shared")
    retained = fine_coeffs[:, [index for index, mode in enumerate(fine_modes) if mode["ell"] <= 4]]

    assert coarse_modes == [mode for mode in fine_modes if mode["ell"] <= 4]
    assert np.array_equal(retained, coarse_coeffs)


def test_compact_u1_reference_is_diagnostic_only() -> None:
    report = u1_lattice_gauge_reference_report(
        lattice_size=2,
        sweeps=6,
        beta=0.2,
        step_size=0.7,
        seed_key="unit-u1",
    )

    assert report["claim_tier"] == "E1"
    assert report["lattice_gauge_stage"]["compact_u1_reference"] is True
    assert report["lattice_gauge_stage"]["su2_reference"] is False
    assert report["partition_randomness"]["partition_replay_receipt"] is True
    assert report["OPH_NATIVE_VACUUM_PROMOTION_RECEIPT"] is False
    assert len(report["plaquette_trace"]) == 6


def test_reference_vacuum_bundle_writes_artifacts(tmp_path: Path) -> None:
    report = write_reference_vacuum_baseline_report(
        tmp_path,
        ell_max=4,
        sample_count=64,
        amplitude=1.0,
        theta=0.0,
        seed_key="bundle",
        smoothing_sigma=0.1,
        u1_lattice_size=2,
        u1_sweeps=4,
    )

    assert report["claim_tier"] == "E1"
    assert report["OPH_NATIVE_VACUUM_PROMOTION_RECEIPT"] is False
    assert (tmp_path / "reference_vacuum_baseline_report.json").exists()
    assert (tmp_path / "free_scalar_harmonic_coefficients.npz").exists()
    assert (tmp_path / "free_scalar_spectra.csv").exists()
