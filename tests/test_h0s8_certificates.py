from __future__ import annotations

import math

from oph_fpe.cosmology.h0s8_certificates import (
    H0S8CertificateInputs,
    fake_trial_certificate,
    h0s8_lane8_certificate_report,
    hidden_export_capacity_bound,
    observer_template_trial_count_log2,
    payload_certificate,
    refinement_stability_gate,
)


def test_h0s8_payload_certificate_uses_hoeffding_fano_bound():
    cert = payload_certificate(
        h_x_bits=16.0,
        alphabet_size=2**16,
        empirical_error=0.0,
        calibration_samples=10_000,
        delta=1e-6,
    )

    assert cert.epsilon_delta > 0.0
    assert 0.0 < cert.i_rec_cert_bits < 16.0


def test_h0s8_fake_trial_certificate_counts_redundant_suppression():
    cert = fake_trial_certificate(
        fake_deficit_bits_value=30.0,
        trial_count_log2=20.0,
        redundant_copy_count=3,
        redundant_p_star=0.5,
        gamma_target_bits=0.0,
    )

    assert cert.redundant_suppression_bits == 2.0
    assert cert.gamma_margin_bits == 12.0
    assert cert.probability_bound == 2.0**-12
    assert cert.gate_pass is True


def test_h0s8_template_trial_count_uses_description_and_time():
    log_trials = observer_template_trial_count_log2(
        template_description_bits=10.0,
        branch_time=8.0,
        tau_min=2.0,
    )

    assert log_trials == 12.0


def test_h0s8_hidden_export_capacity_is_finite_and_charged():
    assert hidden_export_capacity_bound(
        hidden_log_dim_bits=8.0,
        hidden_initial_entropy_bits=2.0,
        continuation_slack_bits=1.0,
    ) == 7.0


def test_h0s8_refinement_gate_requires_gap_above_two_delta():
    assert refinement_stability_gate(coarse_minimizer_gap_bits=1.1, coarse_refinement_delta_bits=0.5)
    assert not refinement_stability_gate(coarse_minimizer_gap_bits=1.0, coarse_refinement_delta_bits=0.5)


def test_h0s8_lane8_defaults_do_not_become_physical_certificate():
    report = h0s8_lane8_certificate_report()

    assert report["values_are_run_derived"] is False
    assert report["theorem_gates"]["audited_record_payload_lower_bound"] is True
    assert report["theorem_gates"]["low_entropy_ancestry_certificate_ready"] is False
    assert report["physical_prediction_ready"] is False


def test_h0s8_lane8_run_derived_values_can_close_certificate():
    report = h0s8_lane8_certificate_report(
        H0S8CertificateInputs(
            h_x_bits=64.0,
            alphabet_size=2**16,
            empirical_error=0.0,
            calibration_samples=1_000_000,
            delta=1e-12,
            b_hid_max_bits=0.0,
            n_res_bits=0.0,
            hidden_log_dim_bits=0.0,
            template_description_bits=8.0,
            branch_time=8.0,
            tau_min=1.0,
            redundant_copy_count=16,
            redundant_p_star=0.5,
            best_nonfaithful_selector_cost=1.0,
            values_source="finite_collar_run",
        )
    )

    assert report["values_are_run_derived"] is True
    assert report["fake_history_certificate"]["gamma_margin_bits"] > 0.0
    assert report["theorem_gates"]["low_entropy_ancestry_certificate_ready"] is True
    assert math.isfinite(report["fake_history_certificate"]["probability_bound"])
