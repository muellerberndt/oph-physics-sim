import numpy as np

from oph_fpe.algebra.maxent_cap_state import maxent_record_operator_cap_state
from oph_fpe.bulk.cap_geometry import fibonacci_sphere_points, sample_caps
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.bulk.modular_probe import state_derived_bw_report


def _fields(points: np.ndarray) -> dict[str, np.ndarray]:
    n = points.shape[0]
    return {
        "record_signature": np.arange(n) % 37,
        "committed_mask": np.ones(n),
        "stable_count": np.arange(n) % 11,
        "repair_load": np.sin(points[:, 2] * 3.0),
        "cumulative_repair_load": np.cos(points[:, 0] * 4.0),
        "local_mismatch_density": np.maximum(points[:, 1], 0.0),
        "s3_class_density": np.abs(points[:, 1]),
        "s3_sector_class": np.arange(n) % 3,
    }


def test_maxent_record_operator_cap_state_is_positive_geometry_free() -> None:
    points = fibonacci_sphere_points(256)
    fields = _fields(points)
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 37,
            "repair_load": np.sin(points[:, 2] * (3.0 + 0.1 * shift)),
        }
        for shift in (3, 2, 1)
    ]

    result = maxent_record_operator_cap_state(fields, history, np.arange(32), regularizer=1.0e-8)

    assert result.rho.shape == (32, 32)
    assert np.allclose(result.rho, result.rho.conj().T)
    assert np.isclose(np.trace(result.rho).real, 1.0)
    assert result.min_eigenvalue > -1.0e-10
    assert result.geometry_dependency_count == 0
    assert result.operator_count > 0
    assert result.as_jsonable()["mode"] == "maxent_record_operator_state"


def test_state_derived_bw_report_accepts_maxent_record_operator_state() -> None:
    points = fibonacci_sphere_points(384)
    caps = sample_caps(points, count=1, theta_values=[0.55], seed=27, collar_width=0.1)
    fields = _fields(points)
    history = [
        {
            **fields,
            "record_signature": (np.arange(points.shape[0]) + shift) % 37,
            "stable_count": (np.arange(points.shape[0]) + shift) % 11,
        }
        for shift in (2, 1)
    ]
    collar = collar_markov_report(points, caps, fields, max_triplets=128, seed=28)

    report = state_derived_bw_report(
        points,
        caps,
        fields,
        collar,
        times=[0.1],
        observables=["record_signature", "repair_load"],
        regularizers=[1.0e-8],
        controls=["wrong_1x_normalization", "no_modular_flow"],
        state_mode="maxent_record_operator_state",
        history_fields=history,
        max_basis=24,
        seed=29,
    )

    assert report["state_mode"] == "maxent_record_operator_state"
    assert report["maxent_record_operator_state"] is True
    assert report["endogenous_modular_generator"] is False
    assert report["surrogate_endogenous_modular_generator"] is True
    assert report["PRIME_GEOMETRIC_CAP_STATE_RECEIPT"] is False
    assert report["normalization_source"] == "surrogate_maxent_record_operator_state"
    assert report["rows"][0]["generator_source"] == "maxent_record_operator_state"
    assert np.isfinite(report["median"])
