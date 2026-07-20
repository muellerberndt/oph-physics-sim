from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.bulk import physical_h3_kms_campaign as campaign
from oph_fpe.bulk.physical_h3_kms_prerun import (
    REQUIRED_CLOCK_CANDIDATES,
    REQUIRED_GEOMETRY_MODELS,
    REQUIRED_RUNGS,
    canonical_sha256,
    frozen_campaign_family_sha256,
)
from oph_fpe.bulk.physical_h3_kms_replay import (
    registered_instrument_code_bundle,
    registered_role_code_registries,
)
from oph_fpe.bulk.physical_h3_kms_runtime import THREAD_ENVIRONMENT_KEYS


@pytest.fixture(autouse=True)
def _canonical_numerical_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_builder_freezes_exact_controls_ladder_and_real_code_bytes() -> None:
    package = campaign.build_frozen_campaign()
    preregistration = package["preregistration"]
    plan = preregistration["plan"]

    assert package["preregistration_report"]["admission_status"] == "VALID_PASS"
    assert package["preregistration_report"]["scientific_status"] == "NOT_EVALUATED"
    assert tuple(plan["rungs"]) == REQUIRED_RUNGS
    assert len(set(plan["seeds"])) >= 3
    assert tuple(plan["clock_candidates"]) == REQUIRED_CLOCK_CANDIDATES
    assert tuple(plan["geometry_models"]) == REQUIRED_GEOMETRY_MODELS
    assert len(plan["run_matrix"]) == len(plan["seeds"]) * len(REQUIRED_RUNGS)
    assert all(row["status"] == "NOT_EVALUATED" for row in plan["run_matrix"])
    assert all(
        row["cell_config"]["source_federation"]["carrier_count"]
        == row["cell"]["rung"]
        for row in plan["run_matrix"]
    )
    assert plan["producer_registry"] == registered_role_code_registries()[
        "producer_registry"
    ]
    assert plan["checker_registry"] == registered_role_code_registries()[
        "checker_registry"
    ]
    assert plan["instrument_commit_sha256"] == registered_instrument_code_bundle()[
        "instrument_commit_sha256"
    ]
    assert plan["plan_sha256"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    assert plan["archive_boundary"] == {
        "frozen_before_source_capture": True,
        "retune_after_freeze": False,
        "archived_16k_failure_preserved": True,
        "archived_outcomes_used_for_threshold_selection": False,
        "historical_receipt_byte_sha256": (
            campaign.HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256
        ),
        "historical_campaign_sha256": campaign.HISTORICAL_CAMPAIGN_SHA256,
        "historical_16k_source_seed": campaign.DEFAULT_SEEDS[0],
        "historical_16k_rung": 16_384,
        "historical_16k_joint_independent_receipt": False,
        "historical_stable_branch_failure_established": False,
    }
    assert package["numerical_runtime"]["thread_environment_receipt"] is True


def test_cell_selection_does_not_change_frozen_campaign_family() -> None:
    four_k = campaign.build_frozen_campaign(current_rung=4_096)["preregistration"]
    sixty_four_k = campaign.build_frozen_campaign(current_rung=65_536)[
        "preregistration"
    ]

    assert four_k["plan"]["plan_sha256"] != sixty_four_k["plan"]["plan_sha256"]
    assert frozen_campaign_family_sha256(four_k["plan"]) == (
        frozen_campaign_family_sha256(sixty_four_k["plan"])
    )
    assert four_k["config"] != sixty_four_k["config"]


def test_calibrations_are_disjoint_threshold_fixtures_not_physical_receipts() -> None:
    package = campaign.build_frozen_campaign()
    plan = package["preregistration"]["plan"]
    artifacts = package["calibration_artifacts"]
    source_seeds = set(plan["seeds"])
    observed_seed_sets: list[set[int]] = []

    for name, artifact in artifacts.items():
        assert set(artifact) == {
            "schema",
            "calibration_id",
            "calibration_seeds",
            "independent_of_campaign_source_seeds",
            "frozen_before_source_capture",
            "protocol",
            "thresholds",
        }
        seeds = set(artifact["calibration_seeds"])
        assert seeds
        assert seeds.isdisjoint(source_seeds)
        assert all(seeds.isdisjoint(previous) for previous in observed_seed_sets)
        observed_seed_sets.append(seeds)
        assert artifact["protocol"]["campaign_source_rng_used"] is False
        assert artifact["protocol"]["archived_campaign_outcomes_used"] is False
        assert artifact["protocol"]["scope"] == (
            "deterministic_threshold_fixture_only"
        )
        assert artifact["protocol"]["physical_threshold_calibration_receipt"] is False
        assert artifact["protocol"]["physical_gate_eligible"] is False
        assert artifact["thresholds"]
        assert plan["calibrations"][f"{name}_sha256"] == canonical_sha256(
            artifact
        )

    assert plan["thresholds"] == {
        "clock_absolute_residual_max": 0.2,
        "clock_win_margin_min": 0.1,
        "geometry_win_margin_min": 0.05,
        "curvature_minimum_power": 0.9,
    }
    assert plan["calibrations"]["physical_threshold_calibration_receipt"] is False
    assert len(
        artifacts["clock_calibration"]["protocol"]["absolute_residual_samples"]
    ) == 3 * 64
    assert len(
        artifacts["geometry_calibration"]["protocol"]["winning_margin_samples"]
    ) == 3 * 64
    assert len(
        artifacts["curvature_calibration"]["protocol"]["detection_power_samples"]
    ) == 3 * 64


def test_demo_force_and_nudge_identifiers_cannot_enter_physical_builder() -> None:
    with pytest.raises(campaign.PhysicalCampaignError, match="forbidden"):
        campaign.build_frozen_campaign(campaign_id="physical-demo-family")
    with pytest.raises(campaign.PhysicalCampaignError, match="forbidden"):
        campaign.build_frozen_campaign(
            replicate_ids=("primary", "force-pass"),
            current_replicate_id="primary",
        )
    with pytest.raises(TypeError):
        campaign.build_frozen_campaign(demo_mode=True)  # type: ignore[call-arg]


def test_campaign_seed_cannot_reuse_an_independent_calibration_seed() -> None:
    with pytest.raises(campaign.PhysicalCampaignError, match="disjoint"):
        campaign.build_frozen_campaign(
            seeds=(910_001, 20_260_751, 20_260_761), current_seed=910_001
        )


def test_runner_freezes_before_capture_and_exports_preflight_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "physical-cell"
    observations: dict[str, object] = {}

    def fake_capture(source_inputs: object) -> dict:
        observations["source_inputs"] = source_inputs
        assert (output / "freeze" / "freeze_receipt.json").is_file()
        assert (output / "freeze" / "clock_calibration.json").is_file()
        assert (output / "freeze" / "geometry_calibration.json").is_file()
        assert (output / "freeze" / "curvature_calibration.json").is_file()
        assert not (output / "replay_bundle").exists()
        return {
            "input_config": source_inputs,
            "config": {"seed": 20_260_751},
            "reports": {
                "source_observer": {"schema": "source"},
                "refinement": {"schema": "refinement"},
                "prime_geometric_state": {"schema": "cap"},
                "independent_geometry": {"schema": "geometry"},
            },
            "capture_sha256": "sha256:" + "a" * 64,
        }

    def fake_bundle_writer(
        output_dir: str | Path,
        preregistration: object,
        source_capture: object,
        calibration_artifacts: object,
        *,
        freeze_receipt: object,
        historical_campaign_receipt_bytes: bytes,
        numerical_runtime: object,
    ) -> Path:
        observations["preregistration"] = preregistration
        observations["calibration_artifacts"] = calibration_artifacts
        observations["freeze_receipt"] = freeze_receipt
        observations["historical_campaign_receipt_bytes"] = (
            historical_campaign_receipt_bytes
        )
        observations["numerical_runtime"] = numerical_runtime
        root = Path(output_dir)
        rows = []
        plan = preregistration["plan"]  # type: ignore[index]
        for frozen in plan["run_matrix"]:
            selected = frozen["cell"] == plan["current_cell"]
            rows.append(
                {
                    **frozen["cell"],
                    "preflight": "PASS" if selected else "PENDING",
                    "powered_and_complete": selected,
                    "status": "VALID_FAIL" if selected else "NOT_EVALUATED",
                }
            )
        postrun = {
            "native_bw": {"schema": "native"},
            "candidate_interventions": {"schema": "clock"},
            "geometry_controls": {"schema": "controls"},
            "semantic_event": {"schema": "event"},
            "campaign": {"run_matrix": rows},
            "postrun_scientific_failures": ["measured_clock_selection_failed"],
            "postrun_not_evaluated_reasons": [],
        }
        _write_canonical(root / "artifacts" / "postrun_report.json", postrun)
        manifest = root / "replay_manifest.json"
        _write_canonical(manifest, {"schema": "test-manifest"})
        return manifest

    monkeypatch.setattr(campaign, "capture_physical_source", fake_capture)
    monkeypatch.setattr(
        campaign, "write_physical_h3_kms_replay_bundle", fake_bundle_writer
    )
    monkeypatch.setattr(
        campaign,
        "replay_physical_h3_kms_bundle",
        lambda _path: {
            "PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT": True,
            "blockers": [],
        },
    )

    receipt = campaign.run_frozen_campaign_cell(output)

    assert receipt["instrument_status"] == "VALID_PASS"
    assert receipt["cell_scientific_status"] == "VALID_FAIL"
    assert receipt["campaign_complete"] is False
    assert receipt["physical_promotion_allowed"] is False
    assert receipt["retune_after_freeze"] is False
    assert receipt["demo_or_nudge_controls_accepted"] is False
    assert receipt["replay_manifest_path"] == "replay_bundle/replay_manifest.json"
    assert observations["source_inputs"] == campaign.build_frozen_campaign()[
        "preregistration_report"
    ]["source_inputs"]
    for filename in (
        "config.json",
        "physical_source_observer_contract_report.json",
        "physical_h3_kms_refinement_report.json",
        "prime_geometric_cap_state_report.json",
        "physical_h3_kms_independent_geometry_report.json",
        "physical_h3_kms_native_bw_payload.json",
        "physical_h3_kms_candidate_interventions_report.json",
        "physical_h3_kms_geometry_controls_report.json",
        "semantic_event_reconstruction_report.json",
        "physical_h3_kms_campaign_manifest.json",
        "physical_h3_kms_replay_verification.json",
        "campaign_run_receipt.json",
    ):
        assert (output / filename).is_file()
    freeze_receipt = json.loads(
        (output / "freeze" / "freeze_receipt.json").read_text(encoding="utf-8")
    )
    assert observations["freeze_receipt"] == freeze_receipt
    assert "freeze_receipt" not in freeze_receipt["artifact_descriptors"]
    assert "numerical_runtime" in freeze_receipt["artifact_descriptors"]
    assert observations["numerical_runtime"] == json.loads(
        (output / "freeze" / "numerical_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze_receipt["scientific_status"] == "NOT_EVALUATED"
    assert freeze_receipt["retune_after_freeze"] is False
    assert freeze_receipt["demo_or_nudge_controls_accepted"] is False
    archived = output / "freeze" / "historical_campaign_receipt.json"
    assert archived.is_file()
    assert (
        "sha256:" + __import__("hashlib").sha256(archived.read_bytes()).hexdigest()
        == campaign.HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256
    )


def test_runner_refuses_changed_historical_receipt_before_output_or_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    altered = tmp_path / "altered_campaign_receipt.json"
    altered.write_bytes(campaign.DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT.read_bytes() + b" ")
    called = False

    def fake_capture(_source_inputs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(campaign, "capture_physical_source", fake_capture)
    output = tmp_path / "must-not-exist"
    with pytest.raises(campaign.PhysicalCampaignError, match="bytes changed"):
        campaign.run_frozen_campaign_cell(
            output, historical_campaign_receipt=altered
        )
    assert called is False
    assert not output.exists()


def test_runner_refuses_noncanonical_runtime_before_output_or_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_capture(_source_inputs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(campaign, "capture_physical_source", fake_capture)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    output = tmp_path / "must-not-exist-runtime"
    with pytest.raises(campaign.PhysicalCampaignError, match="numerical runtime"):
        campaign.run_frozen_campaign_cell(output)
    assert called is False
    assert not output.exists()


def test_runner_refuses_larger_rung_without_replayed_prerequisites_before_output_or_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_capture(_source_inputs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(campaign, "capture_physical_source", fake_capture)
    output = tmp_path / "must-not-exist-unready-16k"
    with pytest.raises(campaign.PhysicalCampaignError, match="lower-rung"):
        campaign.run_frozen_campaign_cell(output, current_rung=16_384)
    assert called is False
    assert not output.exists()


def test_runner_rejects_caller_prerequisites_when_fresh_aggregation_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        campaign,
        "aggregate_physical_h3_kms_family",
        lambda _directories: {
            "aggregation_instrument_status": "INSTRUMENT_INVALID",
            "blockers": ["tampered_prerequisite"],
        },
    )
    output = tmp_path / "must-not-exist-invalid-prerequisites"
    with pytest.raises(campaign.PhysicalCampaignError, match="instrument invalid"):
        campaign.run_frozen_campaign_cell(
            output,
            current_rung=65_536,
            prerequisite_run_directories=(tmp_path / "purported-cell",),
        )
    assert not output.exists()


def test_runner_is_append_never_and_replay_failure_is_not_scientific_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(campaign.PhysicalCampaignError, match="must not already exist"):
        campaign.run_frozen_campaign_cell(existing)

    output = tmp_path / "failed-replay"

    monkeypatch.setattr(
        campaign,
        "capture_physical_source",
        lambda inputs: {
            "input_config": inputs,
            "config": {},
            "reports": {},
            "capture_sha256": "sha256:" + "b" * 64,
        },
    )

    def fake_bundle_writer(
        output_dir: str | Path,
        _preregistration: object,
        _source_capture: object,
        _calibrations: object,
        **_freeze_evidence: object,
    ) -> Path:
        manifest = Path(output_dir) / "replay_manifest.json"
        _write_canonical(manifest, {"schema": "test-manifest"})
        return manifest

    monkeypatch.setattr(
        campaign, "write_physical_h3_kms_replay_bundle", fake_bundle_writer
    )
    monkeypatch.setattr(
        campaign,
        "replay_physical_h3_kms_bundle",
        lambda _path: {
            "PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT": False,
            "blockers": ["tampered"],
        },
    )
    with pytest.raises(campaign.PhysicalCampaignError, match="disk replay failed"):
        campaign.run_frozen_campaign_cell(output)
    failure = json.loads(
        (output / "campaign_run_receipt.json").read_text(encoding="utf-8")
    )
    assert failure["instrument_status"] == "INSTRUMENT_INVALID"
    assert failure["cell_scientific_status"] == "NOT_EVALUATED"
    assert failure["physical_promotion_allowed"] is False
