import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.bulk.modular_probe import (
    _clock_fit_from_rows,
    cap_state_density,
    collar_operator_system_density,
    geometric_permutation_operator,
    history_koopman_modular_generator,
    modular_generator_from_unitary_transition,
    modular_unitary,
    perturb_remeasure_response_density,
    perturb_remeasure_response_kernel_density,
    repair_affinity_response_density,
    regularized_modular_generator,
    state_derived_bw_report,
    state_derived_modular_transport,
    transition_response_modular_generator,
)
from oph_fpe.bulk.transition_selection import transition_scale_selection_report
from oph_fpe.core.graph import fibonacci_sphere_points


def test_regularized_modular_generator_finite_for_zero_probabilities():
    rho = np.diag([1.0, 0.0, 0.0])

    K = regularized_modular_generator(rho, 1e-3)

    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.conj().T)


def test_modular_transport_identity_at_t0_and_unitary_norm():
    rho = np.array([[0.7, 0.1], [0.1, 0.3]])
    rho = rho / np.trace(rho)
    O = np.array([[1.0, 0.2], [0.2, -1.0]])
    K = regularized_modular_generator(rho, 1e-3)
    U = modular_unitary(K, 0.25)

    assert np.allclose(state_derived_modular_transport(O, rho, 0.0, 1e-3), O)
    assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10)


def test_modular_unitary_uses_paper_negative_time_generator_convention():
    K = np.diag([1.0, 2.0])
    t = 0.25

    U = modular_unitary(K, t)

    assert np.allclose(U, np.diag(np.exp(-1j * t * np.asarray([1.0, 2.0]))))


def test_koopman_log_generator_reconstructs_negative_time_unitary() -> None:
    expected_generator = np.diag([-0.4, 0.2, 0.7])
    response_lag = 0.3
    unitary = modular_unitary(expected_generator, response_lag)

    recovered_generator = modular_generator_from_unitary_transition(unitary, response_lag)
    reconstructed_unitary = modular_unitary(recovered_generator, response_lag)

    assert np.allclose(recovered_generator, expected_generator, atol=1.0e-10)
    assert np.allclose(reconstructed_unitary, unitary, atol=1.0e-10)


def test_inferred_clock_fit_uses_nonstatic_clock_carrier_rows():
    rows = []
    for time in (0.125, 0.25, 0.5):
        for _ in range(4):
            rows.append(
                {
                    "time": time,
                    "s_hat": 0.0,
                    "kappa_row_hat": 0.0,
                    "known_scale_residuals": {
                        "0": 0.1,
                        "1x": 0.5,
                        "pi": 1.0,
                        "2pi": 2.0,
                        "4pi": 4.0,
                    },
                    "best_known_scale_label": "0",
                    "clock_carrier_row": False,
                }
            )
        rows.append(
            {
                "time": time,
                "s_hat": float(2.0 * np.pi * time),
                "kappa_row_hat": float(2.0 * np.pi),
                "known_scale_residuals": {
                    "0": 2.0,
                    "1x": 1.5,
                    "pi": 1.0,
                    "2pi": 0.1,
                    "4pi": 1.0,
                },
                "best_known_scale_label": "2pi",
                "clock_carrier_row": True,
            }
        )

    all_rows = _clock_fit_from_rows(
        rows,
        known={"1x": 1.0, "pi": float(np.pi), "2pi": float(2.0 * np.pi), "4pi": float(4.0 * np.pi)},
        fit_label="all_rows",
    )
    carrier_rows = _clock_fit_from_rows(
        [row for row in rows if row["clock_carrier_row"]],
        known={"1x": 1.0, "pi": float(np.pi), "2pi": float(2.0 * np.pi), "4pi": float(4.0 * np.pi)},
        fit_label="nonstatic_clock_carrier_rows",
    )

    assert all_rows["receipt"] is False
    assert carrier_rows["receipt"] is True
    assert carrier_rows["nearest_known_scale"] == "2pi"


def test_state_derived_bw_report_writes_matrix_rows():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=3, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=8)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature", "repair_load"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        max_basis=32,
        seed=11,
    )

    assert report["mode"] == "state_derived_modular_probe"
    assert report["receipt_name"] == "BW_KMS_BRANCH_REPLAY_RECEIPT"
    assert report["legacy_receipt_name"] == "BW_KMS_BRANCH_INSTANTIATION_RECEIPT"
    assert report["finite_lorentz_theorem_contract_receipt"] is False
    assert "claim_level" in report
    assert report["row_count"] == 4
    assert report["median"] >= 0.0
    assert report["rows"][0]["matrix_element_count"] == 32 * 32
    assert report["controls"]["wrong_1x_normalization"]["count"] == 4
    assert report["controls"]["no_modular_flow"]["count"] == 4
    assert "correct_beats_controls" in report
    assert "best_control" in report
    assert "state_selected_scale_label" in report
    assert "2pi" in report["target_scale_candidate_medians"]
    assert report["scale_controls_same_basis"] is True


def test_state_derived_bw_report_accepts_history_transition_kernel():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=4, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = []
    for shift in (3, 2, 1):
        history.append(
            {
                **fields,
                "record_signature": (np.arange(points.shape[0]) + shift) % 31,
                "stable_count": (np.arange(points.shape[0]) + shift) % 7,
                "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
            }
        )
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=8)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature", "repair_load"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        state_mode="history_transition_kernel",
        history_fields=history,
        max_basis=32,
        seed=13,
    )

    assert report["state_mode"] == "history_transition_kernel"
    assert report["endogenous_modular_generator"] is True
    assert report["row_count"] == 4
    assert report["rows"][0]["matrix_element_count"] == 32 * 32
    assert np.isfinite(report["median"])
    assert "wrong_1x_normalization" in report["controls"]
    assert report["state_selected_scale_label"] in {"1x", "2pi"}
    assert report["scale_controls_same_basis"] is True
    assert "target_scale_control_degenerate" in report
    assert "degenerate_target_scale_controls" in report


def test_history_koopman_generator_uses_visible_modular_time_lag():
    basis = np.arange(4)
    base = {
        "record_signature": np.arange(4),
        "stable_count": np.arange(4),
        "committed_mask": np.ones(4),
        "repair_load": np.linspace(0.1, 0.4, 4),
        "cumulative_repair_load": np.linspace(0.2, 0.5, 4),
        "local_mismatch_density": np.linspace(0.3, 0.6, 4),
        "modular_depth": np.linspace(0.4, 0.7, 4),
        "modular_time": np.zeros(4),
        "s3_class_density": np.linspace(0.5, 0.8, 4),
        "s3_sector_class": np.arange(4) % 3,
    }
    history = [
        base,
        {**base, "record_signature": np.arange(4) + 1, "modular_time": np.full(4, 0.03)},
    ]
    raw = {**base, "record_signature": np.arange(4) + 2, "modular_time": np.full(4, 0.06)}

    _generator, meta = history_koopman_modular_generator(raw, history, basis, response_lag=0.25)

    assert meta["response_lag_source"] == "observer_visible_modular_time_median_increment"
    assert meta["effective_response_lag"] == 0.03
    assert meta["configured_response_lag"] == 0.25
    assert meta["generator_reconstruction_error"] < 1.0e-10


def test_endogenous_generator_receipt_is_separate_from_2pi_clock():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=14, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "modular_depth": np.linspace(0.0, 1.0, points.shape[0]),
        "modular_time": np.full(points.shape[0], 0.09),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 31,
            "modular_time": np.full(points.shape[0], 0.03 * index),
        }
        for index, shift in enumerate((3, 2, 1), start=0)
    ]
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=8)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.125, 0.25, 0.5],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-6],
        controls=[],
        state_mode="history_koopman_generator_state",
        history_fields=history,
        max_basis=24,
        seed=15,
    )

    assert report["endogenous_modular_generator"] is True
    assert "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT" in report
    assert "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT" in report
    assert report["endogenous_generator_receipt_boundary"].startswith("L2 only")


def test_collar_operator_system_density_is_positive_finite():
    points = fibonacci_sphere_points(384)
    cap = sample_caps(points, count=1, theta_values=[0.55], seed=24, collar_width=0.1)[0]
    basis = np.arange(48)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 31,
            "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
        }
        for shift in (2, 1)
    ]

    rho = collar_operator_system_density(points, cap, basis, fields, history, regularizer=1e-6)
    eigvals = np.linalg.eigvalsh(rho)

    assert rho.shape == (48, 48)
    assert np.all(np.isfinite(rho))
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert float(np.min(eigvals)) > -1e-10


def test_cap_state_density_accepts_collar_operator_system_mode():
    points = fibonacci_sphere_points(384)
    cap = sample_caps(points, count=1, theta_values=[0.55], seed=25, collar_width=0.1)[0]
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }

    rho, support, packets = cap_state_density(
        points,
        cap,
        fields,
        max_basis=32,
        regularizer=1e-6,
        seed=26,
        state_mode="collar_operator_system",
    )

    assert rho.shape == (support.size, support.size)
    assert support.size == 32
    assert packets.size == points.shape[0]
    assert np.isclose(np.trace(rho).real, 1.0)


def test_state_derived_bw_report_accepts_collar_operator_system():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=27, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 31,
            "stable_count": (np.arange(points.shape[0]) + shift) % 7,
            "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
        }
        for shift in (3, 2, 1)
    ]
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=28)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature", "repair_load"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        state_mode="collar_operator_system",
        history_fields=history,
        max_basis=32,
        seed=29,
    )

    assert report["state_mode"] == "collar_operator_system"
    assert report["endogenous_modular_generator"] is True
    assert report["row_count"] == 4
    assert np.isfinite(report["median"])
    assert report["state_selected_scale_label"] in {"1x", "2pi"}
    assert report["scale_controls_same_basis"] is True


def test_state_derived_bw_report_accepts_offdiagonal_collar_current_observables():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.75], seed=30, collar_width=0.12)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 31,
            "stable_count": (np.arange(points.shape[0]) + shift) % 7,
            "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
        }
        for shift in (3, 2, 1)
    ]
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=31)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.09, 0.17],
        observables=["collar_crossing_current", "packet_flow_current"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        state_mode="collar_operator_system",
        history_fields=history,
        max_basis=32,
        seed=32,
    )

    assert report["state_mode"] == "collar_operator_system"
    assert report["row_count"] == 8
    assert np.isfinite(report["median"])
    assert {row["observable"] for row in report["rows"]} == {
        "collar_crossing_current",
        "packet_flow_current",
    }
    assert all(row["matrix_element_count"] == 32 * 32 for row in report["rows"])


def test_state_derived_bw_report_records_generator_scale():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=14, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 31,
            "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
        }
        for shift in (2, 1)
    ]
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=18)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization"],
        state_mode="history_transition_kernel",
        history_fields=history,
        generator_scale=2.0 * np.pi,
        max_basis=32,
        seed=17,
    )

    assert report["generator_scale"] == 2.0 * np.pi
    assert report["normalization_source"] == "bw_normalized_finite_state_generator"
    assert report["rows"][0]["generator_scale"] == 2.0 * np.pi
    assert report["generator_scale_audit"]["enabled"] is True
    assert report["generator_scale_audit"]["candidate_source"] == "auto_from_wrong_normalization_controls"
    assert {"0", "1x", "pi", "2pi", "4pi"}.issubset(report["generator_scale_audit"]["by_scale"])


def test_state_derived_bw_report_accepts_cap_flow_graph_generator():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=21, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=22)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature", "repair_load"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        state_mode="cap_flow_graph_generator",
        generator_scale=2.0 * np.pi,
        max_basis=32,
        seed=23,
    )

    assert report["state_mode"] == "cap_flow_graph_generator"
    assert report["endogenous_modular_generator"] is False
    assert report["declared_cap_flow_generator"] is True
    assert report["generator_scale"] == 2.0 * np.pi
    assert report["row_count"] == 4
    assert np.isfinite(report["median"])
    assert report["rows"][0]["generator_source"] == "declared_cap_flow_graph_generator"
    assert report["rows"][0]["matrix_element_count"] == 32 * 32


def test_state_derived_bw_report_writes_generator_scale_audit():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=1, theta_values=[0.55], seed=31, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=32)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.025],
        observables=["record_signature"],
        regularizers=[0.001],
        controls=["wrong_1x_normalization"],
        state_mode="cap_flow_graph_generator",
        generator_scale=2.0 * np.pi,
        generator_scale_candidates=[-2.0 * np.pi, 0.0, 1.0, 2.0 * np.pi],
        max_basis=24,
        seed=33,
    )

    audit = report["generator_scale_audit"]
    assert audit["enabled"] is True
    assert audit["candidate_source"] == "explicit_config"
    assert audit["configured_scale_label"] == "2pi"
    assert audit["target_scale_label"] == "2pi"
    assert "2pi" in audit["by_scale"]
    assert "minus_2pi" in audit["by_scale"]
    assert "0" in audit["by_scale"]
    assert audit["best_label"] in audit["by_scale"]
    assert audit["diagnosis"] in {
        "configured_scale_best",
        "opposite_sign_best",
        "no_flow_best",
        "different_scale_best",
    }


def test_transition_response_density_log_calibrates_density_log_lane():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=34, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=35)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-12],
        controls=[
            "wrong_1x_normalization",
            "wrong_pi_normalization",
            "shuffled_observables",
            "no_modular_flow",
        ],
        state_mode="transition_response_density_log",
        target_operator_mode="permutation",
        transition_response_time=0.1,
        transition_response_scale=2.0 * np.pi,
        density_inverse_temperature=0.1,
        generator_scale=10.0,
        generator_scale_candidates=[0.0, 1.0, 10.0],
        max_basis=32,
        seed=36,
    )

    assert report["state_mode"] == "transition_response_density_log"
    assert report["declared_transition_response_density"] is True
    assert report["endogenous_modular_generator"] is False
    assert report["normalization_source"] == "declared_transition_response_density_log"
    assert report["density_inverse_temperature"] == 0.1
    assert report["generator_scale"] == 10.0
    assert report["state_selected_scale_label"] == "2pi"
    assert report["correct_beats_controls"] is True
    assert report["TRANSITION_RESPONSE_DENSITY_LOG_CALIBRATION_RECEIPT"] is True
    assert "shuffled_observables" in report["not_applicable_controls"]
    assert report["rows"][0]["generator_source"] == "declared_transition_response_density_log"
    assert report["generator_scale_audit"]["enabled"] is True


def test_transition_response_unitary_skips_generator_scale_audit():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=37, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=38)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-12],
        controls=["wrong_1x_normalization", "wrong_pi_normalization", "no_modular_flow"],
        state_mode="transition_response_unitary",
        target_operator_mode="permutation",
        transition_response_time=0.1,
        transition_response_scale=2.0 * np.pi,
        generator_scale=2.0 * np.pi,
        generator_scale_candidates=[0.0, 1.0, 2.0 * np.pi],
        max_basis=32,
        seed=39,
    )

    assert report["direct_transition_automorphism"] is True
    assert report["normalization_declared"] is True
    assert report["correct_beats_controls"] is True
    assert report["generator_scale_audit"]["enabled"] is False
    assert report["generator_scale_audit"]["not_applicable"] is True


def test_repair_affinity_response_density_is_endogenous_positive_state():
    points = fibonacci_sphere_points(384)
    cap = sample_caps(points, count=1, theta_values=[0.55], seed=40, collar_width=0.1)[0]
    basis = np.arange(32)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }

    rho = repair_affinity_response_density(
        points,
        cap,
        basis,
        fields,
        response_time=0.1,
        regularizer=1e-12,
        density_inverse_temperature=0.1,
    )
    eigvals = np.linalg.eigvalsh(rho)

    assert rho.shape == (32, 32)
    assert np.all(np.isfinite(rho))
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert float(np.min(eigvals)) > -1e-10


def test_state_derived_bw_report_accepts_repair_affinity_response_density_log():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=41, collar_width=0.1)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": (np.arange(points.shape[0]) % 3),
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=42)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-12],
        controls=["wrong_1x_normalization", "wrong_pi_normalization", "no_modular_flow"],
        state_mode="repair_affinity_response_density_log",
        target_operator_mode="permutation",
        transition_response_time=0.1,
        density_inverse_temperature=0.1,
        generator_scale=10.0,
        generator_scale_candidates=[0.0, 1.0, 10.0],
        max_basis=32,
        seed=43,
    )

    assert report["state_mode"] == "repair_affinity_response_density_log"
    assert report["repair_affinity_response_density"] is True
    assert report["endogenous_modular_generator"] is True
    assert report["normalization_source"] == "endogenous_repair_affinity_response_density_log"
    assert report["rows"][0]["generator_source"] == "repair_affinity_response_density_log"
    assert report["generator_scale_audit"]["enabled"] is True
    assert report["row_count"] == 4
    assert np.isfinite(report["median"])


def test_perturb_remeasure_response_density_is_endogenous_positive_state():
    points = fibonacci_sphere_points(192)
    cap = sample_caps(points, count=1, theta_values=[0.75], seed=44, collar_width=0.12)[0]
    patch_count = points.shape[0]
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    basis = np.arange(24)
    fields = {
        "record_signature": np.arange(patch_count) % 31,
        "committed_mask": np.ones(patch_count),
        "stable_count": np.arange(patch_count) % 7,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(patch_count) % 3,
    }

    rho = perturb_remeasure_response_density(
        points,
        cap,
        basis,
        fields,
        graph_response={
            "left": left,
            "right": right,
            "port_left": np.zeros(left.size, dtype=np.int64),
            "port_right": np.zeros(left.size, dtype=np.int64),
            "group_order": 6,
            "patch_count": patch_count,
        },
        response_time=0.1,
        regularizer=1e-12,
        density_inverse_temperature=0.1,
        seed=45,
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
    )
    eigvals = np.linalg.eigvalsh(rho)

    assert rho.shape == (24, 24)
    assert np.all(np.isfinite(rho))
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert float(np.min(eigvals)) > -1e-10


def test_state_derived_bw_report_accepts_perturb_remeasure_response_density_log():
    points = fibonacci_sphere_points(192)
    caps = sample_caps(points, count=1, theta_values=[0.75], seed=46, collar_width=0.12)
    patch_count = points.shape[0]
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    fields = {
        "record_signature": np.arange(patch_count) % 31,
        "committed_mask": np.ones(patch_count),
        "stable_count": np.arange(patch_count) % 7,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(patch_count) % 3,
    }
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=47)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-12],
        controls=["wrong_1x_normalization", "wrong_pi_normalization", "no_modular_flow"],
        state_mode="perturb_remeasure_response_density_log",
        target_operator_mode="permutation",
        transition_response_time=0.1,
        density_inverse_temperature=0.1,
        generator_scale=10.0,
        generator_scale_candidates=[0.0, 1.0, 10.0],
        graph_response={
            "left": left,
            "right": right,
            "port_left": np.zeros(left.size, dtype=np.int64),
            "port_right": np.zeros(left.size, dtype=np.int64),
            "group_order": 6,
            "patch_count": patch_count,
        },
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
        max_basis=24,
        seed=48,
    )

    assert report["state_mode"] == "perturb_remeasure_response_density_log"
    assert report["perturb_remeasure_response_density"] is True
    assert report["endogenous_modular_generator"] is True
    assert report["normalization_source"] == "endogenous_perturb_remeasure_response_density_log"
    assert report["rows"][0]["generator_source"] == "perturb_remeasure_response_density_log"
    assert report["generator_scale_audit"]["enabled"] is True
    assert report["row_count"] == 2
    assert np.isfinite(report["median"])


def test_state_derived_bw_report_accepts_perturb_remeasure_response_kernel_log():
    points = fibonacci_sphere_points(192)
    caps = sample_caps(points, count=1, theta_values=[0.75], seed=49, collar_width=0.12)
    patch_count = points.shape[0]
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    fields = {
        "record_signature": np.arange(patch_count) % 31,
        "committed_mask": np.ones(patch_count),
        "stable_count": np.arange(patch_count) % 7,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(patch_count) % 3,
    }
    graph_response = {
        "left": left,
        "right": right,
        "port_left": np.zeros(left.size, dtype=np.int64),
        "port_right": np.zeros(left.size, dtype=np.int64),
        "group_order": 6,
        "patch_count": patch_count,
    }
    rho = perturb_remeasure_response_kernel_density(
        points,
        caps[0],
        np.arange(24),
        fields,
        graph_response=graph_response,
        regularizer=1e-12,
        seed=50,
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
    )
    assert rho.shape == (24, 24)
    assert np.all(np.isfinite(rho))
    assert np.isclose(np.trace(rho).real, 1.0)

    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=51)
    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1e-12],
        controls=["wrong_1x_normalization", "wrong_pi_normalization", "no_modular_flow"],
        state_mode="perturb_remeasure_response_kernel_log",
        target_operator_mode="permutation",
        transition_response_time=0.1,
        density_inverse_temperature=0.1,
        generator_scale=1.0,
        generator_scale_candidates=[0.0, 1.0, 2.0 * np.pi],
        graph_response=graph_response,
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
        max_basis=24,
        seed=52,
    )

    assert report["state_mode"] == "perturb_remeasure_response_kernel_log"
    assert report["perturb_remeasure_response_kernel"] is True
    assert report["endogenous_modular_generator"] is True
    assert report["normalization_source"] == "endogenous_perturb_remeasure_response_kernel_log"
    assert report["rows"][0]["generator_source"] == "perturb_remeasure_response_kernel_log"
    assert report["generator_scale_audit"]["enabled"] is True
    assert report["row_count"] == 2
    assert np.isfinite(report["median"])


def test_transition_response_generator_matches_pullback_orientation():
    points = fibonacci_sphere_points(256)
    cap = sample_caps(points, count=1, theta_values=[0.55], seed=12, collar_width=0.12)[0]
    basis = np.arange(32)
    response_time = 0.1
    response_scale = 2.0 * np.pi
    transition, _ = geometric_permutation_operator(points, cap, response_scale * response_time, basis)

    K, eta = transition_response_modular_generator(
        points,
        cap,
        basis,
        response_time=response_time,
        response_scale=response_scale,
    )
    U = modular_unitary(K, response_time)

    assert eta < 1e-10
    assert np.linalg.norm(U - transition.conj().T) < 1e-10


def test_transition_scale_selection_declared_sanity_selects_2pi():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=5, collar_width=0.12)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 31,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 7,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.abs(points[:, 1]),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(points.shape[0]) % 3,
    }

    report = transition_scale_selection_report(
        points,
        caps,
        fields,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        sources=["declared_geometric_sanity"],
        candidate_scales=[1.0, np.pi, 2.0 * np.pi, 4.0 * np.pi],
        declared_response_scale=2.0 * np.pi,
        max_basis=32,
        seed=19,
    )

    assert report["primary_source"] == "declared_geometric_sanity"
    assert report["selected_label"] == "2pi"
    assert report["two_pi_selected"] is True
    assert report["source_reports"]["declared_geometric_sanity"]["by_scale"]["2pi"]["selection_score_median"] < 1e-10


def test_transition_scale_selection_repair_affinity_is_primary_not_forced():
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=2, theta_values=[0.55], seed=7, collar_width=0.12)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 53,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 9,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(points.shape[0]) % 3,
    }

    report = transition_scale_selection_report(
        points,
        caps,
        fields,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        sources=["repair_affinity_response", "declared_geometric_sanity"],
        candidate_scales=[1.0, np.pi, 2.0 * np.pi, 4.0 * np.pi],
        max_basis=32,
        seed=23,
    )

    assert report["primary_source"] == "repair_affinity_response"
    assert report["selected_label"] in {"1x", "pi", "2pi", "4pi"}
    assert "repair_collar_observer_fields" in report["normalization_source"]
    assert report["source_reports"]["declared_geometric_sanity"]["two_pi_selected"] is True
    assert report["row_count"] == 2 * 2 * 1 * 4


def test_transition_scale_selection_perturb_remeasure_is_primary_with_graph_response():
    points = fibonacci_sphere_points(192)
    caps = sample_caps(points, count=1, theta_values=[0.75], seed=9, collar_width=0.14)
    patch_count = points.shape[0]
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    fields = {
        "record_signature": np.arange(patch_count) % 53,
        "committed_mask": np.ones(patch_count),
        "stable_count": np.arange(patch_count) % 9,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(patch_count) % 3,
    }

    report = transition_scale_selection_report(
        points,
        caps,
        fields,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        sources=["perturb_remeasure_response", "declared_geometric_sanity"],
        candidate_scales=[1.0, np.pi, 2.0 * np.pi, 4.0 * np.pi],
        graph_response={
            "left": left,
            "right": right,
            "port_left": np.zeros(left.size, dtype=np.int64),
            "port_right": np.zeros(left.size, dtype=np.int64),
            "group_order": 6,
            "patch_count": patch_count,
        },
        max_basis=16,
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
        seed=29,
    )

    assert report["primary_source"] == "perturb_remeasure_response"
    assert report["selected_label"] in {"1x", "pi", "2pi", "4pi"}
    assert report["source_reports"]["perturb_remeasure_response"]["by_scale"]["1x"]["count"] == 1
    assert report["rows"][0]["mean_perturbed_edges_per_source"] > 0.0
    assert report["source_reports"]["declared_geometric_sanity"]["two_pi_selected"] is True


def test_transition_scale_selection_kms_collar_transport_is_branch_instantiation():
    points = fibonacci_sphere_points(256)
    caps = sample_caps(points, count=1, theta_values=[0.75], seed=10, collar_width=0.12)
    patch_count = points.shape[0]
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    fields = {
        "record_signature": np.arange(patch_count) % 41,
        "committed_mask": np.ones(patch_count),
        "stable_count": np.arange(patch_count) % 5,
        "repair_load": np.abs(points[:, 0]),
        "cumulative_repair_load": np.abs(points[:, 2]),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(patch_count) % 3,
    }

    report = transition_scale_selection_report(
        points,
        caps,
        fields,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        sources=["kms_collar_transport_response", "perturb_remeasure_response", "declared_geometric_sanity"],
        candidate_scales=[1.0, np.pi, 2.0 * np.pi, 4.0 * np.pi],
        graph_response={
            "left": left,
            "right": right,
            "port_left": np.zeros(left.size, dtype=np.int64),
            "port_right": np.zeros(left.size, dtype=np.int64),
            "group_order": 6,
            "patch_count": patch_count,
        },
        kms_response_scale=2.0 * np.pi,
        kms_transport_steps=6,
        max_basis=24,
        probe_steps=2,
        probe_repairs_per_source=8,
        probe_max_incident_edges=2,
        seed=31,
    )

    assert report["primary_source"] == "kms_collar_transport_response"
    assert report["two_pi_selected"] is True
    assert report["source_reports"]["kms_collar_transport_response"]["two_pi_selected"] is True
    assert report["source_reports"]["perturb_remeasure_response"]["selected_label"] in {"1x", "pi", "2pi", "4pi"}
    assert report["normalization_source"] == "kms_bw_collar_transport_for_primary"
