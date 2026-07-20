from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.bosons.brst_blocks import block_structure_receipt
from oph_fpe.bosons.pipeline import (
    _load_artifact_reference,
    build_wzh_campaign_report,
    load_wzh_config,
    write_wzh_campaign_bundle,
)
from oph_fpe.bosons.pole_enclosure import pole_enclosure_receipt
from oph_fpe.bosons.rg_transport import piecewise_affine_rg_receipt
from oph_fpe.bosons.source_clock import PLANCK_H_J_S, source_clock_receipt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bosons" / "wzh_source_closure_diagnostic_v1.yml"


def test_source_clock_computes_gap_but_fails_closed_without_source_packet() -> None:
    receipt = source_clock_receipt(
        np.diag([0.0, 2.0, 5.0]),
        frequency_hz=10.0,
        perturbation_norm_bound=0.1,
        source_packet_verified=False,
        no_target_ancestry=False,
    )

    assert receipt["dimensionless_gap"] == 2.0
    assert receipt["gap_interval"] == [1.8, 2.2]
    assert receipt["E_star_joule"] == PLANCK_H_J_S * 5.0
    assert receipt["E_star_GeV"] is not None
    assert receipt["source_clock_receipt"] is False
    assert "source_clock_packet_not_verified" in receipt["blockers"]


def test_piecewise_rg_matches_scalar_affine_solution() -> None:
    receipt = piecewise_affine_rg_receipt(
        [2.0],
        initial_scale=math.e,
        segments=[
            {
                "end_scale": 1.0,
                "matrix": [[1.0]],
                "offset": [0.0],
                "matching_matrix": [[1.0]],
                "matching_offset": [0.0],
            }
        ],
        source_packet_verified=True,
        same_branch=True,
        scheme="test",
        loop_order="exact_affine",
    )

    assert receipt["final_state"][0] == pytest.approx(2.0 / math.e, abs=1.0e-10)
    assert receipt["affine_rg_numerical_control_receipt"] is True
    assert receipt["declared_candidate_conditions_met"] is True
    assert receipt["rg_matching_receipt"] is False
    assert receipt["promotion_allowed"] is False


def test_affine_rg_rejects_negative_error_bound() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        piecewise_affine_rg_receipt(
            [2.0],
            initial_scale=math.e,
            segments=[
                {
                    "end_scale": 1.0,
                    "matrix": [[1.0]],
                    "offset": [0.0],
                    "matching_matrix": [[1.0]],
                    "matching_offset": [0.0],
                    "truncation_error_bound": -1.0,
                }
            ],
            source_packet_verified=True,
            same_branch=True,
            scheme="test",
            loop_order="exact_affine",
        )


def test_neutral_block_factors_photon_and_massive_control_root() -> None:
    block = block_structure_receipt(
        [
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-3.0, 0.2]]],
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
        ],
        block_id="Z",
        block_kind="neutral",
    )

    assert block["ward_photon_factor_at_s_zero"] is True
    assert block["brst_block_receipt"] is False
    pole = pole_enclosure_receipt(
        block["determinant_coefficients"],
        contour_center=(3.0, -0.2),
        contour_radius=0.25,
    )
    assert pole["numerical_zero_control_receipt"] is True
    assert pole["physical_pole_receipt"] is False


def test_wzh0_helpers_cannot_promote_from_all_true_caller_flags() -> None:
    block = block_structure_receipt(
        [
            [[[1.0, 0.0]]],
            [[[1.0, 0.0]]],
        ],
        block_id="W",
        block_kind="charged",
        source_kernel_verified=True,
        ward_identity_verified=True,
        slavnov_taylor_verified=True,
        nielsen_identity_verified=True,
    )
    assert block["declared_candidate_conditions_met"] is True
    assert block["brst_block_receipt"] is False
    assert block["promotion_allowed"] is False

    pole = pole_enclosure_receipt(
        [[-1.0, 0.1], [1.0, 0.0]],
        reference_coefficients=[[-1.0, 0.1], [1.0, 0.0]],
        contour_center=(1.0, -0.1),
        contour_radius=0.25,
        physical_sheet_verified=True,
        nonzero_residue_verified=True,
        uncertainty_bound_present=True,
        source_block_receipt=True,
    )
    assert pole["declared_candidate_conditions_met"] is True
    assert pole["numerical_zero_control_receipt"] is True
    assert pole["physical_pole_receipt"] is False
    assert pole["promotion_allowed"] is False

    clock = source_clock_receipt(
        np.diag([0.0, 2.0, 5.0]),
        frequency_hz=10.0,
        selected_levels=(0, 1),
        source_packet_verified=True,
        no_target_ancestry=True,
        frequency_role="source_emitted",
    )
    assert clock["declared_candidate_conditions_met"] is True
    assert clock["finite_gap_numerical_control_receipt"] is True
    assert clock["source_clock_receipt"] is False
    assert clock["promotion_allowed"] is False


def test_default_campaign_is_hash_pinned_fail_closed_diagnostic(tmp_path: Path) -> None:
    config = load_wzh_config(CONFIG)
    report = build_wzh_campaign_report(config, repository_root=ROOT, generated_utc="2026-07-12T00:00:00Z")

    assert report["overall_status"] == (
        "WZH0_SYNTHETIC_CONTROL_SPECIFICATION_ONLY_NONPROMOTING"
    )
    assert report["promotion_allowed"] is False
    assert report["promotion_gates"]["certificate_candidate_claim"] is False
    assert report["promotion_gates"]["paper_inputs_hash_pinned"] is False
    assert all(
        item["numerical_zero_control_receipt"] is True
        and item["physical_pole_receipt"] is False
        and item["physical_readout_attached"] is False
        and item["M_GeV"] is None
        for item in report["pole_enclosures"].values()
    )

    output = tmp_path / "bundle"
    written = write_wzh_campaign_bundle(CONFIG, output, repository_root=ROOT)
    assert written["promotion_allowed"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["promotion_allowed"] is False


def test_legacy_artifact_loader_rejects_absolute_and_parent_paths() -> None:
    for unsafe in ("../outside.json", str((ROOT / "pyproject.toml").resolve())):
        result = _load_artifact_reference(unsafe, ROOT)
        assert result["present"] is False
        assert result["path_contained"] is False
        assert result["hash_matches_expected"] is False


def test_legacy_campaign_cannot_self_promote_when_every_declared_gate_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_wzh_config(CONFIG)
    config["claim_level"] = "certificate_candidate"
    config["source_root"] = {"verified": True, "same_branch": True}
    config["provenance"] = {
        "no_target_ancestry": True,
        "prospective_claim_freeze": True,
    }
    config["source_clock"]["source_packet_verified"] = True
    config["source_clock"]["no_target_ancestry"] = True
    config["rg_transport"]["source_packet_verified"] = True
    config["rg_transport"]["same_branch"] = True
    for block in config["pole_controls"]["blocks"].values():
        block["coordinate_unit"] = "GeV_squared"
        for field in (
            "source_kernel_verified",
            "ward_identity_verified",
            "slavnov_taylor_verified",
            "nielsen_identity_verified",
            "physical_sheet_verified",
            "nonzero_residue_verified",
            "uncertainty_bound_present",
        ):
            block[field] = True

    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline._source_root_artifact_verified", lambda _value: True
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline._certificate_verified",
        lambda _value, _kind: True,
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline._trusted_packet_verified", lambda _value: True
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline._load_artifact_reference",
        lambda *_args, **_kwargs: {
            "present": True,
            "hash_matches_expected": True,
            "payload": {"promotion_allowed": True},
        },
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.source_clock_receipt",
        lambda **_kwargs: {"source_clock_receipt": True, "E_star_GeV": 1.0},
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.piecewise_affine_rg_receipt",
        lambda **_kwargs: {"rg_matching_receipt": True},
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.block_structure_receipt",
        lambda *_args, block_id, **_kwargs: {
            "block_id": block_id,
            "determinant_coefficients": [[1.0, 0.0]],
            "brst_block_receipt": True,
        },
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.pole_enclosure_receipt",
        lambda *_args, **_kwargs: {
            "physical_pole_receipt": True,
            "mass_coordinate": 1.0,
            "width_coordinate": 0.1,
        },
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.decode_matrix_polynomial", lambda _value: object()
    )
    monkeypatch.setattr(
        "oph_fpe.bosons.pipeline.determinant_polynomial", lambda _value: [1.0 + 0.0j]
    )

    report = build_wzh_campaign_report(
        config,
        repository_root=ROOT,
        generated_utc="2026-07-20T00:00:00Z",
    )

    assert all(
        report["promotion_gates"][key] is True
        for key in (
            "certificate_candidate_claim",
            "paper_inputs_hash_pinned",
            "strict_source_root",
            "same_branch",
            "source_clock",
            "d10_QT1_QT5",
            "d11_source_character",
            "rg_matching",
            "no_target_ancestry",
            "prospective_claim_freeze",
        )
    )
    assert report["promotion_gates"]["brst_complex_poles"] is False
    assert report["promotion_gates"]["runtime_subject_binding"] is False
    assert report["promotion_allowed"] is False
    assert all(
        pole["physical_readout_attached"] is False
        and pole["M_GeV"] is None
        and pole["Gamma_GeV"] is None
        for pole in report["pole_enclosures"].values()
    )
    assert "legacy_v1_caller_claims_are_diagnostic_only" in report["blockers"]
