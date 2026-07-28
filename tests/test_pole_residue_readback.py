"""Regression and adversarial tests for the pole-residue readback (issue #569)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.core.charged_response import ChargedResponseError, Q5
from oph_fpe.core import pole_residue_readback as prr

REPO_ROOT = Path(__file__).resolve().parents[1]
RER = REPO_ROOT.parent / "reverse-engineering-reality" / "code" / "a5_closure" / "manifests"
CARRIER_PATH = RER / "echosahedral_federation_reference.json"
PARENT_PATH = RER / "charged_response_semantic_artifact.json"

pytestmark = pytest.mark.skipif(
    not (CARRIER_PATH.is_file() and PARENT_PATH.is_file()),
    reason="the research carrier and parent artifact are not checked out",
)


@pytest.fixture(scope="module")
def carrier_manifest() -> dict:
    return json.loads(CARRIER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parent_artifact() -> dict:
    return json.loads(PARENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact(carrier_manifest: dict, parent_artifact: dict) -> dict:
    return prr.produce_pole_residue_artifact(carrier_manifest, parent_artifact)


def test_artifact_is_deterministic(carrier_manifest: dict, parent_artifact: dict, artifact: dict) -> None:
    again = prr.produce_pole_residue_artifact(carrier_manifest, parent_artifact)
    assert again == artifact


def test_family_band_residue_receipt(artifact: dict) -> None:
    receipt = artifact["pole_residue_readback"]["family_band_residue"]
    assert receipt == {
        "band": "frame",
        "measured_rank": 3,
        "equals_exact_frame_projector": True,
        "lowest_positive_generator_frequency": True,
        "unitary_mode_norms_conserved": True,
        "faithful_kernel_order": 1,
        "equivariant_under_all_automorphisms": True,
        "galois_partner_at_maximal_pole": True,
    }
    poles = artifact["pole_residue_readback"]["measured_poles"]
    assert poles["unit"]["multiplicity"] == 1
    assert poles["frame"]["multiplicity"] == 3
    assert poles["quintet"]["multiplicity"] == 5
    assert poles["kernel"]["multiplicity"] == 3


def test_artifact_pins_parent_and_dynamics(artifact: dict, parent_artifact: dict) -> None:
    binding = artifact["carrier_binding"]
    assert binding["parent_artifact_sha256"] == parent_artifact["artifact_sha256"]
    assert artifact["dynamics_binding"]["dynamics_sha256"].startswith("sha256:")
    assert artifact["schema"] == prr.ARTIFACT_SCHEMA


def test_no_floats_in_artifact(artifact: dict) -> None:
    def walk(value, path="$"):
        assert not isinstance(value, float), path
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(artifact)


def test_controls_failed_closed(artifact: dict) -> None:
    for name, verdict in artifact["controls"].items():
        assert verdict["expected_failure"] is True, name
        assert verdict["failed"] is True, name


def test_reconstruction_is_exact_and_guarded() -> None:
    assert prr.reconstruct_q5((30.0 + 6.0 * prr.SQRT5) / 120.0, 120) == Q5.of(
        __import__("fractions").Fraction(30, 120), __import__("fractions").Fraction(6, 120)
    )
    with pytest.raises(ChargedResponseError):
        prr.reconstruct_q5(
            (30.0 + 6.0 * prr.SQRT5) / 120.0 + 50.0 * prr.RECONSTRUCTION_TOLERANCE, 120
        )


def test_truncated_series_refused() -> None:
    series = prr.measure_response_series()[:2]
    with pytest.raises(ChargedResponseError):
        prr.extract_pole_residues(series)


def test_doctored_step_refused() -> None:
    series = prr.measure_response_series()
    series[1][0, 0] += 1.0e-3
    with pytest.raises(ChargedResponseError):
        prr.extract_pole_residues(series)


def test_parent_carrier_pin_enforced(carrier_manifest: dict, parent_artifact: dict) -> None:
    doctored = dict(parent_artifact)
    doctored["carrier_binding"] = dict(parent_artifact["carrier_binding"])
    doctored["carrier_binding"]["carrier_manifest_sha256"] = "doctored"
    with pytest.raises(ChargedResponseError):
        prr.produce_pole_residue_artifact(carrier_manifest, doctored)


def test_measured_series_is_a_unitary_channel() -> None:
    series = prr.measure_response_series()
    step = series[1]
    assert float(np.max(np.abs(step.conj().T @ step - np.eye(12)))) <= 1.0e-8
    assert float(np.max(np.abs(series[3] - step @ step @ step))) <= 1.0e-7
