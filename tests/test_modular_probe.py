import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.bulk.modular_probe import (
    geometric_permutation_operator,
    modular_unitary,
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
    assert report["receipt_name"] == "BW_KMS_BRANCH_INSTANTIATION_RECEIPT"
    assert "claim_level" in report
    assert report["row_count"] == 4
    assert report["median"] >= 0.0
    assert report["rows"][0]["matrix_element_count"] == 32 * 32
    assert report["controls"]["wrong_1x_normalization"]["count"] == 4
    assert report["controls"]["no_modular_flow"]["count"] == 4
    assert "correct_beats_controls" in report
    assert "best_control" in report


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
