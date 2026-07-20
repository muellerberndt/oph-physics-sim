from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import oph_fpe.bulk.physical_h3_kms_replay as replay_module
from oph_fpe.bulk.physical_h3_kms_campaign import (
    DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT,
)
from oph_fpe.bulk.physical_h3_kms_replay import (
    REPLAY_MANIFEST_SCHEMA,
    ReplayBundleError,
    registered_instrument_code_bundle,
    registered_role_code_bindings,
    registered_role_code_registries,
    replay_physical_h3_kms_bundle,
    write_physical_h3_kms_replay_bundle,
)
from oph_fpe.bulk.physical_h3_kms_preflight import (
    physical_h3_kms_preflight_report,
)
from oph_fpe.bulk.physical_h3_kms_prerun import (
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.physical_h3_kms_runtime import (
    THREAD_ENVIRONMENT_KEYS,
    observed_numerical_runtime,
)


COMPATIBILITY_RECEIPTS = (
    "REPLAY_MANIFEST_VERIFICATION_RECEIPT",
    "PRE_SOURCE_FREEZE_REPLAY_RECEIPT",
    "HISTORICAL_16K_ARCHIVE_BYTE_REPLAY_RECEIPT",
    "SOURCE_CAPTURE_REPLAY_RECEIPT",
    "PER_CELL_CONTROL_ARTIFACTS_REPLAY_RECEIPT",
    "PER_CELL_SCIENTIFIC_PREDICATES_RECOMPUTED_RECEIPT",
    "SINGLE_BUNDLE_COMMITMENT_RECEIPT",
    "NUMERICAL_RUNTIME_REPLAY_RECEIPT",
)
ReplayInputs = tuple[dict, dict, dict, dict, bytes, dict]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _byte_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.fixture
def replay_inputs(monkeypatch: pytest.MonkeyPatch) -> ReplayInputs:
    for key in THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")
    numerical_runtime = observed_numerical_runtime()
    source = capture_physical_source(
        {
            "carrier_count": 4,
            "rung": 4,
            "seed": 811,
            "replicate_id": "primary",
            "preregistered_plan_sha256": "sha256:" + "a" * 64,
            "propagation_steps": 2,
            "intrinsic_step": 0.137,
            "coupling_strength": 1.0,
            "state_space": "normalized_complex_amplitude_in_C12",
            "rng_family": "numpy_generator_pcg64_v1",
            "initialization_distribution": "normalized_complex_gaussian_v1",
            "intrinsic_phase_distribution": "uniform_unit_interval_v1",
            "seam_update_rule": (
                "disjoint_single_port_endpoint_arithmetic_mean_v1"
            ),
            "observer_count": 2,
            "observer_support_size": 2,
            "observer_samples": 4,
            "support_refinement_level": 1,
            "geometry_sample_count": 4,
        }
    )
    source_inputs = copy.deepcopy(source["input_config"])
    registries = registered_role_code_registries()
    thresholds = {
        "clock_absolute_residual_max": 0.2,
        "clock_win_margin_min": 0.1,
        "geometry_win_margin_min": 0.05,
        "curvature_minimum_power": 0.9,
    }
    calibration_artifacts = {
        "clock_calibration": {
            "schema": "oph.physical-h3-kms.clock-calibration.v1",
            "calibration_id": "clock-independent-v1",
            "calibration_seeds": [9001, 9002],
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "protocol": {
                "algorithm": "synthetic-clock-power-v1",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
            },
            "thresholds": {
                "clock_absolute_residual_max": thresholds[
                    "clock_absolute_residual_max"
                ],
                "clock_win_margin_min": thresholds["clock_win_margin_min"],
            },
        },
        "geometry_calibration": {
            "schema": "oph.physical-h3-kms.geometry-calibration.v1",
            "calibration_id": "geometry-independent-v1",
            "calibration_seeds": [9101, 9102],
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "protocol": {
                "algorithm": "synthetic-geometry-power-v1",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
            },
            "thresholds": {
                "geometry_win_margin_min": thresholds[
                    "geometry_win_margin_min"
                ]
            },
        },
        "curvature_calibration": {
            "schema": "oph.physical-h3-kms.curvature-calibration.v1",
            "calibration_id": "curvature-independent-v1",
            "calibration_seeds": [9201, 9202],
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "protocol": {
                "algorithm": "synthetic-curvature-power-v1",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
            },
            "thresholds": {
                "curvature_minimum_power": thresholds[
                    "curvature_minimum_power"
                ]
            },
        },
    }
    calibration_plan = {
        "clock_calibration_sha256": replay_module.canonical_sha256(
            calibration_artifacts["clock_calibration"]
        ),
        "geometry_calibration_sha256": replay_module.canonical_sha256(
            calibration_artifacts["geometry_calibration"]
        ),
        "curvature_calibration_sha256": replay_module.canonical_sha256(
            calibration_artifacts["curvature_calibration"]
        ),
        "independent_of_campaign_source_seeds": True,
        "frozen_before_source_capture": True,
        "physical_threshold_calibration_receipt": False,
    }
    preregistration = {
        "schema": "test-preregistration",
        "config": {"cell": "small-test"},
        "plan": {
            "instrument_commit_sha256": registered_instrument_code_bundle()[
                "instrument_commit_sha256"
            ],
            "seeds": [101, 202, 303],
            "thresholds": thresholds,
            "calibrations": calibration_plan,
            "producer_registry": registries["producer_registry"],
            "checker_registry": registries["checker_registry"],
            "archive_boundary": {
                "frozen_before_source_capture": True,
                "retune_after_freeze": False,
                "archived_16k_failure_preserved": True,
                "archived_outcomes_used_for_threshold_selection": False,
                "historical_receipt_byte_sha256": (
                    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
                ),
                "historical_campaign_sha256": (
                    REGISTERED_HISTORICAL_CAMPAIGN_SHA256
                ),
                "historical_16k_source_seed": (
                    REGISTERED_HISTORICAL_16K_SOURCE_SEED
                ),
                "historical_16k_rung": 16_384,
                "historical_16k_joint_independent_receipt": False,
                "historical_stable_branch_failure_established": False,
            },
        },
    }
    preregistration["plan"]["plan_sha256"] = replay_module.canonical_sha256(
        preregistration["plan"]
    )
    preregistration_report = {
        "schema": "test-preregistration-report",
        "SOURCE_CAPTURE_ALLOWED": True,
        "admission_status": "VALID_PASS",
        "scientific_status": "NOT_EVALUATED",
        "source_inputs": source_inputs,
        "scientific_failures": [],
    }

    def fake_prerun(value: object) -> dict:
        assert isinstance(value, dict)
        return copy.deepcopy(preregistration_report)

    def fake_source_inputs(value: object) -> dict:
        assert isinstance(value, dict)
        return copy.deepcopy(source_inputs)

    def fake_postrun(capture: object, envelope: object) -> dict:
        assert capture == source
        assert isinstance(envelope, dict)
        not_evaluated_stages = (
            "P1_nested_refinement_and_expectations",
            "P2_prime_geometric_cap_state",
            "P3_independent_geometric_parameter",
            "P4_native_bw01_bw08",
            "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage",
            "P7_semantic_event_e1_e4_and_frame_fiber_separation",
            "P8_frozen_multiseed_four_rung_campaign",
        )
        stage_epistemics = {
            stage_id: {
                "measurement_status": "NOT_EVALUATED",
                "physical_gate_eligible": False,
                "not_evaluated_reasons": ["independent_producer_missing"],
                "scientific_failures": [],
                "structural_receipt": {
                    "scope": "artifact_integrity_only",
                    "status": "COMPLETE",
                    "artifact_hash": _byte_sha256(stage_id.encode()),
                    "physical_claim": False,
                },
                "instrument_receipt": {
                    "scope": "independent_physical_producer",
                    "status": "MISSING_REQUIRED_PRODUCER",
                    "physical_claim": False,
                },
                "sensitivity_receipt": {
                    "scope": "diagnostic_sensitivity_only",
                    "status": "COMPLETE",
                    "artifact_hash": _byte_sha256((stage_id + ":s").encode()),
                    "physical_gate_eligible": False,
                    "diagnostic_findings": [],
                },
            }
            for stage_id in not_evaluated_stages
        }
        stage_epistemics["P5_frozen_candidate_interventions"] = {
            "measurement_status": "NOT_EVALUATED"
        }
        return {
            "schema": "test-postrun-report",
            "source_root_sha256": source["source_root_sha256"],
            "stage_epistemics": stage_epistemics,
            "native_bw": {"schema": "test-native-bw"},
            "candidate_interventions": {"schema": "test-candidates"},
            "geometry_controls": {"schema": "test-geometry-controls"},
            "semantic_event": {"schema": "test-semantic-event"},
            "campaign": {"schema": "test-campaign"},
            "postrun_scientific_failures": ["expected_negative_measurement"],
            # The disk layer must never copy this into its own promotion field.
            "physical_promotion_allowed": True,
        }

    monkeypatch.setattr(replay_module, "physical_h3_kms_prerun_report", fake_prerun)
    monkeypatch.setattr(replay_module, "physical_h3_kms_source_inputs", fake_source_inputs)
    monkeypatch.setattr(
        replay_module,
        "_compute_postrun_reports_from_verified_source",
        fake_postrun,
    )
    historical_bytes = DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT.read_bytes()
    frozen_values = {
        "preregistration": preregistration,
        "preregistration_report": preregistration_report,
        "postrun_preregistration": replay_module._postrun_envelope(
            preregistration, preregistration_report
        ),
        "role_code_bindings": registered_role_code_bindings(),
        "instrument_code_bundle": registered_instrument_code_bundle(),
        "numerical_runtime": numerical_runtime,
        **calibration_artifacts,
    }
    filenames = {
        "preregistration": "preregistration.json",
        "preregistration_report": "preregistration_report.json",
        "postrun_preregistration": "postrun_preregistration.json",
        "role_code_bindings": "role_code_bindings.json",
        "instrument_code_bundle": "instrument_code_bundle.json",
        "numerical_runtime": "numerical_runtime.json",
        "clock_calibration": "clock_calibration.json",
        "geometry_calibration": "geometry_calibration.json",
        "curvature_calibration": "curvature_calibration.json",
        "historical_campaign_receipt": "historical_campaign_receipt.json",
    }
    frozen_bytes = {
        name: _json_bytes(value) for name, value in frozen_values.items()
    }
    frozen_bytes["historical_campaign_receipt"] = historical_bytes
    descriptors = {
        name: {
            "path": f"freeze/{filenames[name]}",
            "byte_sha256": _byte_sha256(data),
            "byte_count": len(data),
        }
        for name, data in frozen_bytes.items()
    }
    freeze_receipt = {
        "schema": "oph.physical-h3-kms.pre-source-freeze-receipt.v2",
        "frozen_at_utc": "2026-07-20T00:00:00.000000Z",
        "plan_sha256": preregistration["plan"]["plan_sha256"],
        "preregistration_sha256": replay_module.canonical_sha256(preregistration),
        "instrument_commit_sha256": preregistration["plan"][
            "instrument_commit_sha256"
        ],
        "artifact_descriptors": descriptors,
        "source_capture_allowed": True,
        "scientific_status": "NOT_EVALUATED",
        "retune_after_freeze": False,
        "archived_16k_failure_preserved": True,
        "archived_outcomes_used_for_threshold_selection": False,
        "demo_or_nudge_controls_accepted": False,
        "claim_boundary": "test-only internal chronology receipt",
    }
    return (
        preregistration,
        source,
        calibration_artifacts,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    )


def _write_bundle(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> Path:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    return write_physical_h3_kms_replay_bundle(
        tmp_path / "bundle",
        preregistration,
        source,
        calibrations,
        freeze_receipt=freeze_receipt,
        historical_campaign_receipt_bytes=historical_bytes,
        numerical_runtime=numerical_runtime,
    )


def _rewrite_artifact(
    manifest_path: Path,
    artifact_name: str,
    value: object | None = None,
    *,
    raw: bytes | None = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["artifacts"][artifact_name]
    artifact_path = manifest_path.parent / descriptor["path"]
    data = raw if raw is not None else _json_bytes(value)
    artifact_path.write_bytes(data)
    descriptor["byte_sha256"] = _byte_sha256(data)
    descriptor["byte_count"] = len(data)
    manifest_path.write_bytes(_json_bytes(manifest))


def test_role_registry_helper_binds_actual_implementation_bytes() -> None:
    bindings = registered_role_code_bindings()
    registries = registered_role_code_registries()

    assert set(bindings) == {"producer_registry", "checker_registry"}
    for role, row in bindings["producer_registry"].items():
        assert row["byte_sha256"] == registries["producer_registry"][role][
            "source_code_sha256"
        ]
        assert row["byte_count"] > 0
        assert row["implementation_path"].endswith(".py")
    for role, row in bindings["checker_registry"].items():
        assert row["byte_sha256"] == registries["checker_registry"][role][
            "checker_code_sha256"
        ]

    bundle_paths = {
        row["implementation_path"]
        for row in registered_instrument_code_bundle()["files"]
    }
    assert {
        "oph_fpe/bulk/physical_h3_kms_preflight.py",
        "oph_fpe/bulk/physical_h3_kms_runtime.py",
        "oph_fpe/bulk/physical_h3_kms_aggregate.py",
        "oph_fpe/bulk/bw_native_preflight.py",
        "oph_fpe/bulk/bw_certificate_308.py",
        "oph_fpe/core/echosahedral_dynamics.py",
        "oph_fpe/core/echosahedral_federation.py",
        "oph_fpe/core/icosahedral.py",
        "oph_fpe/core/screen_ports.py",
        "oph_fpe/gauge/covariant_overlap.py",
        "oph_fpe/finite_groups.py",
        "oph_fpe/claims.py",
    }.issubset(bundle_paths)


def test_exact_disk_bundle_replays_without_scientific_promotion(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    report = replay_physical_h3_kms_bundle(manifest_path)

    assert report["PREREGISTRATION_REPORT_EXACT_REPLAY_RECEIPT"]
    assert report["PREREGISTRATION_SOURCE_CAPTURE_ADMISSION_RECEIPT"]
    assert report["PRE_SOURCE_FREEZE_ARTIFACT_BINDING_RECEIPT"]
    assert report["HISTORICAL_16K_ARCHIVE_BYTE_BINDING_RECEIPT"]
    assert report["REGISTERED_ROLE_CODE_BYTE_BINDING_RECEIPT"]
    assert report["INSTRUMENT_CODE_BUNDLE_BINDING_RECEIPT"]
    assert report["CALIBRATION_ARTIFACT_BINDING_RECEIPT"]
    assert report["NUMERICAL_RUNTIME_ARTIFACT_BINDING_RECEIPT"]
    assert report["SOURCE_INPUT_BINDING_RECEIPT"]
    assert report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"]
    assert report["POSTRUN_REPORT_EXACT_REPLAY_RECEIPT"]
    assert report["PREFLIGHT_EXPORT_PROJECTION_RECEIPT"]
    assert set(report["preflight_export_hashes"]) == {
        "config",
        "source_observer",
        "refinement",
        "prime_geometric_state",
        "independent_geometry",
        "native_bw",
        "candidate_interventions",
        "geometry_controls",
        "semantic_event",
        "campaign",
    }
    assert report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"]
    assert all(report[name] is True for name in COMPATIBILITY_RECEIPTS)
    assert all(
        row["BYTE_SHA256_RECEIPT"]
        for row in report["artifact_byte_receipts"].values()
    )
    assert report["scientific_evaluation_performed_by_replay_layer"] is False
    assert report["physical_promotion_allowed"] is False
    assert report["scientific_promotion_allowed"] is False
    assert "not positive scientific" in report["compatibility_receipt_semantics"][
        "claim_boundary"
    ]
    assert report["blockers"] == []


def test_writer_and_reader_each_execute_exact_source_replay_once(
    tmp_path: Path,
    replay_inputs: ReplayInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_verifier = replay_module.verify_physical_source_capture
    actual_postrun = replay_module._compute_postrun_reports_from_verified_source
    call_count = 0
    postrun_count = 0

    def counted_verifier(source: object) -> dict:
        nonlocal call_count
        call_count += 1
        assert isinstance(source, dict)
        return actual_verifier(source)

    def counted_postrun(capture: object, envelope: object) -> dict:
        nonlocal postrun_count
        postrun_count += 1
        return actual_postrun(capture, envelope)

    monkeypatch.setattr(
        replay_module,
        "verify_physical_source_capture",
        counted_verifier,
    )
    monkeypatch.setattr(
        replay_module,
        "_compute_postrun_reports_from_verified_source",
        counted_postrun,
    )
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    assert call_count == 1
    assert postrun_count == 1

    call_count = 0
    postrun_count = 0
    report = replay_physical_h3_kms_bundle(manifest_path)
    assert report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"] is True
    assert call_count == 1
    assert postrun_count == 1


def test_preflight_rejects_tampered_top_level_export_despite_valid_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    run_root = tmp_path / "run"
    manifest_path = write_physical_h3_kms_replay_bundle(
        run_root / "replay_bundle",
        preregistration,
        source,
        calibrations,
        freeze_receipt=freeze_receipt,
        historical_campaign_receipt_bytes=historical_bytes,
        numerical_runtime=numerical_runtime,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    postrun_descriptor = manifest["artifacts"]["postrun_report"]
    postrun = json.loads(
        (manifest_path.parent / postrun_descriptor["path"]).read_text(
            encoding="utf-8"
        )
    )
    exports = {
        "config.json": source["config"],
        "physical_source_observer_contract_report.json": source["reports"][
            "source_observer"
        ],
        "physical_h3_kms_refinement_report.json": source["reports"]["refinement"],
        "prime_geometric_cap_state_report.json": source["reports"][
            "prime_geometric_state"
        ],
        "physical_h3_kms_independent_geometry_report.json": source["reports"][
            "independent_geometry"
        ],
        "physical_h3_kms_native_bw_payload.json": postrun["native_bw"],
        "physical_h3_kms_candidate_interventions_report.json": postrun[
            "candidate_interventions"
        ],
        "physical_h3_kms_geometry_controls_report.json": postrun[
            "geometry_controls"
        ],
        "semantic_event_reconstruction_report.json": postrun["semantic_event"],
        "physical_h3_kms_campaign_manifest.json": postrun["campaign"],
        "physical_h3_kms_replay_verification.json": replay_physical_h3_kms_bundle(
            manifest_path
        ),
    }
    for filename, value in exports.items():
        (run_root / filename).write_bytes(_json_bytes(value))

    baseline = physical_h3_kms_preflight_report(run_root)
    assert baseline["artifact_replay_admission"][
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"
    ], baseline["artifact_replay_admission"]
    assert baseline["artifact_replay_admission"][
        "top_level_preflight_exports_exact"
    ]

    tampered_path = run_root / "physical_h3_kms_geometry_controls_report.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["caller_authored_pass"] = True
    tampered_path.write_bytes(_json_bytes(tampered))

    rejected = physical_h3_kms_preflight_report(run_root)
    admission = rejected["artifact_replay_admission"]
    assert not admission["PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"]
    assert not admission["top_level_preflight_exports_exact"]
    assert (
        "top_level_preflight_export_mismatch:geometry_controls"
        in admission["blockers"]
    )


def test_writer_refuses_plan_hashes_not_bound_to_actual_code(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    broken = copy.deepcopy(preregistration)
    broken["plan"]["producer_registry"]["source_dynamics"][
        "source_code_sha256"
    ] = "sha256:" + "0" * 64

    with pytest.raises(ReplayBundleError, match="actual registered code bytes"):
        write_physical_h3_kms_replay_bundle(
            tmp_path / "broken",
            broken,
            source,
            calibrations,
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_bytes,
            numerical_runtime=numerical_runtime,
        )


def test_artifact_byte_tampering_fails_before_json_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = manifest_path.parent / manifest["artifacts"]["source_capture"]["path"]
    source_path.write_bytes(source_path.read_bytes() + b" ")

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert not report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"]
    assert not report["artifact_byte_receipts"]["source_capture"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert "byte commitment mismatch" in report["blockers"][0]
    assert all(report[name] is False for name in COMPATIBILITY_RECEIPTS)


def test_recommitted_preregistration_report_tampering_fails_exact_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = (
        manifest_path.parent / manifest["artifacts"]["preregistration_report"]["path"]
    )
    supplied = json.loads(report_path.read_text(encoding="utf-8"))
    supplied["scientific_status"] = "VALID_PASS"
    _rewrite_artifact(manifest_path, "preregistration_report", supplied)

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert report["artifact_byte_receipts"]["preregistration_report"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert not report["PREREGISTRATION_REPORT_EXACT_REPLAY_RECEIPT"]
    assert "not an exact replay" in report["blockers"][0]
    assert all(report[name] is False for name in COMPATIBILITY_RECEIPTS)


def test_recommitted_source_tampering_fails_deterministic_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = manifest_path.parent / manifest["artifacts"]["source_capture"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["claim_boundary"] = "tampered but byte-recommitted"
    _rewrite_artifact(manifest_path, "source_capture", source)

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert report["SOURCE_INPUT_BINDING_RECEIPT"]
    assert not report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"]
    assert "source capture is not an exact replay" in report["blockers"][0]


def test_recommitted_postrun_tampering_fails_exact_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    postrun_path = manifest_path.parent / manifest["artifacts"]["postrun_report"]["path"]
    postrun = json.loads(postrun_path.read_text(encoding="utf-8"))
    postrun["postrun_scientific_failures"] = []
    _rewrite_artifact(manifest_path, "postrun_report", postrun)

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"]
    assert not report["POSTRUN_REPORT_EXACT_REPLAY_RECEIPT"]
    assert "postrun report is not an exact replay" in report["blockers"][0]


def test_recommitted_historical_archive_substitution_fails_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    _rewrite_artifact(
        manifest_path,
        "historical_campaign_receipt",
        raw=b'{"schema":"substitute"}\n',
    )

    report = replay_physical_h3_kms_bundle(manifest_path)

    assert report["artifact_byte_receipts"]["historical_campaign_receipt"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert report["HISTORICAL_16K_ARCHIVE_BYTE_BINDING_RECEIPT"] is False
    assert "historical campaign receipt byte hash mismatch" in report["blockers"][0]


def test_recommitted_freeze_anchor_substitution_fails_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze_path = manifest_path.parent / manifest["artifacts"]["freeze_receipt"][
        "path"
    ]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["retune_after_freeze"] = True
    _rewrite_artifact(manifest_path, "freeze_receipt", freeze)

    report = replay_physical_h3_kms_bundle(manifest_path)

    assert report["artifact_byte_receipts"]["freeze_receipt"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert report["PRE_SOURCE_FREEZE_ARTIFACT_BINDING_RECEIPT"] is False
    assert "pre-source freeze receipt anchor mismatch" in report["blockers"][0]


def test_recommitted_numerical_runtime_substitution_fails_before_source_replay(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_path = manifest_path.parent / manifest["artifacts"][
        "numerical_runtime"
    ]["path"]
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["numpy"]["version"] = "substituted"
    material = {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    runtime["runtime_sha256"] = replay_module.canonical_sha256(material)
    _rewrite_artifact(manifest_path, "numerical_runtime", runtime)

    report = replay_physical_h3_kms_bundle(manifest_path)

    assert report["artifact_byte_receipts"]["numerical_runtime"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"] is False
    assert "current_numpy_mismatch" in report["blockers"][0]


def test_current_thread_environment_mismatch_fails_before_source_replay(
    tmp_path: Path,
    replay_inputs: ReplayInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    report = replay_physical_h3_kms_bundle(manifest_path)

    assert report["SOURCE_CAPTURE_EXACT_REPLAY_RECEIPT"] is False
    assert "current_observed_thread_environment_mismatch" in report["blockers"][0]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"a","schema":"b"}',
        b'{"schema":NaN}',
        b'{"schema":1e999}',
    ],
)
def test_manifest_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path, raw: bytes
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(raw)

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert not report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"]
    assert report["blockers"]


def test_artifact_json_rejects_nonfinite_even_when_bytes_are_recommitted(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    _rewrite_artifact(manifest_path, "postrun_report", raw=b'{"value":NaN}\n')

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert report["artifact_byte_receipts"]["postrun_report"][
        "BYTE_SHA256_RECEIPT"
    ]
    assert "nonfinite JSON constant" in report["blockers"][0]


def test_safe_relative_path_rejects_parent_traversal(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["source_capture"]["path"] = "../source_capture.json"
    manifest_path.write_bytes(_json_bytes(manifest))

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert not report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"]
    assert "safe POSIX relative path" in report["blockers"][0]


def test_manifest_role_binding_tampering_fails_closed(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["role_code_bindings"]["checker_registry"]["cell_postflight"][
        "byte_sha256"
    ] = "sha256:" + "0" * 64
    manifest_path.write_bytes(_json_bytes(manifest))

    report = replay_physical_h3_kms_bundle(manifest_path)
    assert not report["REGISTERED_ROLE_CODE_BYTE_BINDING_RECEIPT"]
    assert "differ from actual bytes" in report["blockers"][0]


def test_plan_instrument_commit_must_bind_replay_verifier_code(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    broken = copy.deepcopy(preregistration)
    broken["plan"]["instrument_commit_sha256"] = "sha256:" + "f" * 64

    with pytest.raises(ReplayBundleError, match="recomputed code-bundle digest"):
        write_physical_h3_kms_replay_bundle(
            tmp_path / "broken",
            broken,
            source,
            calibrations,
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_bytes,
            numerical_runtime=numerical_runtime,
        )


def test_hash_bound_but_unrelated_calibration_threshold_is_rejected(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    broken_preregistration = copy.deepcopy(preregistration)
    broken_calibrations = copy.deepcopy(calibrations)
    broken_calibrations["clock_calibration"]["thresholds"][
        "clock_win_margin_min"
    ] = 99.0
    broken_preregistration["plan"]["calibrations"][
        "clock_calibration_sha256"
    ] = replay_module.canonical_sha256(broken_calibrations["clock_calibration"])

    with pytest.raises(ReplayBundleError, match="does not match frozen plan"):
        write_physical_h3_kms_replay_bundle(
            tmp_path / "broken-calibration",
            broken_preregistration,
            source,
            broken_calibrations,
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_bytes,
            numerical_runtime=numerical_runtime,
        )


def test_calibration_seeds_must_be_disjoint_from_campaign_and_each_other(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs
    broken_preregistration = copy.deepcopy(preregistration)
    broken_calibrations = copy.deepcopy(calibrations)
    broken_calibrations["geometry_calibration"]["calibration_seeds"] = [101]
    broken_preregistration["plan"]["calibrations"][
        "geometry_calibration_sha256"
    ] = replay_module.canonical_sha256(
        broken_calibrations["geometry_calibration"]
    )

    with pytest.raises(ReplayBundleError, match="intersect campaign source seeds"):
        write_physical_h3_kms_replay_bundle(
            tmp_path / "overlap-calibration",
            broken_preregistration,
            source,
            broken_calibrations,
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_bytes,
            numerical_runtime=numerical_runtime,
        )


def test_writer_never_overwrites_an_existing_bundle(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    _write_bundle(tmp_path, replay_inputs)
    (
        preregistration,
        source,
        calibrations,
        freeze_receipt,
        historical_bytes,
        numerical_runtime,
    ) = replay_inputs

    with pytest.raises(ReplayBundleError, match="refusing to overwrite"):
        write_physical_h3_kms_replay_bundle(
            tmp_path / "bundle",
            preregistration,
            source,
            calibrations,
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_bytes,
            numerical_runtime=numerical_runtime,
        )


def test_written_manifest_uses_exact_schema(
    tmp_path: Path, replay_inputs: ReplayInputs
) -> None:
    manifest_path = _write_bundle(tmp_path, replay_inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == REPLAY_MANIFEST_SCHEMA
    assert set(manifest["artifacts"]) == {
        "preregistration",
        "preregistration_report",
        "freeze_receipt",
        "numerical_runtime",
        "historical_campaign_receipt",
        "source_capture",
        "postrun_report",
        "clock_calibration",
        "geometry_calibration",
        "curvature_calibration",
    }
