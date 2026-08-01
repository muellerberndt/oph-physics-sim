from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import sph_harm_y

from oph_fpe.cosmology.a5_biposh_continuum_tail import (
    DEFAULT_RECEIPT,
    SCHEMA,
    STATUS,
    _canonical_bytes,
    _exact_difference_bound,
    _exact_midpoint_defect_bound,
    _generator_difference_bound,
    _generator_midpoint_defect_bound,
    _solid_harmonics,
    build_a5_biposh_continuum_tail_packet,
    exact_refinement_identity_report,
)
from oph_fpe.cosmology.verify_a5_biposh_continuum_tail_independent import (
    VerificationError,
    _verify_interval,
    _verify_inverse,
    _verify_tail,
    verify_packet,
)


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))


def _rehash(receipt: dict) -> None:
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(receipt)
    ).hexdigest()


def test_exact_refinement_identity_is_rational_and_complete() -> None:
    report = exact_refinement_identity_report()
    assert report["arithmetic"] == "fractions.Fraction over Q"
    assert report["coefficient_basis_cases"] == 36
    assert report["identity_verified"] is True


def test_polynomial_harmonics_match_independent_scipy_convention() -> None:
    generator = np.random.default_rng(659)
    points = generator.normal(size=(256, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    for ell in (2, 4):
        reference = np.column_stack(
            [sph_harm_y(ell, m, theta, phi) for m in range(-ell, ell + 1)]
        )
        assert np.max(np.abs(_solid_harmonics(points, ell) - reference)) < 3.0e-14


def test_conservative_generator_fallback_dominates_stable_exact_bounds() -> None:
    for ell in (2, 4):
        for edge_bound in (0.65, 0.1, 0.01, 1.0e-4, 1.0e-8):
            assert _generator_difference_bound(ell, edge_bound) >= (
                _exact_difference_bound(ell, edge_bound)
            )
            assert _generator_midpoint_defect_bound(ell, edge_bound) >= (
                _exact_midpoint_defect_bound(ell, edge_bound)
            )


def test_canonical_packet_rebuilds_byte_exact(canonical: dict) -> None:
    assert _canonical_bytes(build_a5_biposh_continuum_tail_packet()) == _canonical_bytes(
        canonical
    )


def test_independent_verifier_replays_canonical_packet(canonical: dict) -> None:
    verified = verify_packet(DEFAULT_RECEIPT)
    assert verified["schema"] == SCHEMA
    assert verified["status"] == STATUS


def test_conditional_interval_is_nonzero_without_physical_promotion(
    canonical: dict,
) -> None:
    interval = canonical["conditional_continuum_interval"]
    assert interval["conditional_interval_excludes_zero"] is True
    assert interval["primary_amplitude_free_statistic_interval"][0] > 0.0
    decisions = canonical["selection_decision"]
    assert decisions["conditional_stiffness_continuum_limit_exists"] is True
    assert (
        decisions["conditional_stiffness_l6_nonzero_under_numerical_envelope"]
        is True
    )
    assert decisions["equal_seam_refinement_extension_source_selected"] is False
    assert decisions["physical_covariance_selected"] is False
    assert decisions["physical_prediction"] is False
    assert decisions["promotion_allowed"] is False


def test_forged_tail_bound_is_rejected(canonical: dict) -> None:
    forged = copy.deepcopy(canonical["mesh_and_tail_certificate"])
    forged["block_rows"][0]["certified_tail_upper_bound"] *= 0.1
    with pytest.raises(VerificationError, match="tail certified_tail_upper_bound"):
        _verify_tail(forged)


def test_forged_stable_polynomial_is_rejected(canonical: dict) -> None:
    forged = copy.deepcopy(canonical["mesh_and_tail_certificate"])
    forged["stable_polynomial_forms"]["M4_squared_over_c4_squared"] = "4*t^2"
    with pytest.raises(VerificationError, match="stable addition-theorem"):
        _verify_tail(forged)


def test_forged_continuum_interval_is_rejected(canonical: dict) -> None:
    forged = copy.deepcopy(canonical)
    forged["conditional_continuum_interval"][
        "primary_amplitude_free_statistic_interval"
    ][0] *= 2.0
    with pytest.raises(VerificationError, match="continuum interval"):
        _verify_interval(forged)


def test_inverse_covariance_cannot_claim_a_continuum_tail(canonical: dict) -> None:
    forged = copy.deepcopy(canonical["conditional_inverse_covariance"])
    forged["continuum_tail_enclosed"] = True
    with pytest.raises(VerificationError, match="inverse covariance boundary"):
        _verify_inverse(forged)


def test_payload_mutation_is_rejected_before_replay(
    tmp_path: Path,
    canonical: dict,
) -> None:
    forged = copy.deepcopy(canonical)
    forged["claim_boundary"] = "promoted"
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(VerificationError, match="payload hash"):
        verify_packet(path)


def test_rehashed_physical_promotion_is_rejected(
    tmp_path: Path,
    canonical: dict,
) -> None:
    forged = copy.deepcopy(canonical)
    forged["selection_decision"]["physical_prediction"] = True
    _rehash(forged)
    path = tmp_path / "promoted.json"
    path.write_bytes(_canonical_bytes(forged))
    with pytest.raises(VerificationError, match="selection boundary"):
        verify_packet(path)
