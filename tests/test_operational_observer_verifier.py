from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.common_source_tower import (
    C0_RECEIPT_KEYS,
    REPORT_ARTIFACT_TYPE as COMMON_SOURCE_REPORT_ARTIFACT_TYPE,
)
from oph_fpe.core.echosahedral_federation import (
    EchosahedralFederation,
    ExternalBoundaryBundle,
    ObserverSupport,
    interface_algebra_sha256,
    reference_echosahedral_carrier,
    reference_federation_instrument_bundle,
)
from oph_fpe.observers.operational_verifier import (
    A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT,
    A3_READBACK_PREDICTION_CONTROL_RECEIPT,
    A3_RECORD_COMMIT_REPLAY_RECEIPT,
    A4_BOUNDED_INTERFACE_RECEIPT,
    A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT,
    A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT,
    A4_FEEDBACK_ABLATION_RECEIPT,
    MANIFEST_SCHEMA,
    OBSERVER_ARTIFACT_INTEGRITY_RECEIPT,
    OBSERVER_CONTRACT_BINDING_RECEIPT,
    OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT,
    OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT,
    OBSERVER_SOURCE_FIREWALL_RECEIPT,
    OPERATIONAL_SELF_READING_OBSERVER_RECEIPT,
    TRANSACTION_PARENT_ENVELOPE_SCHEMA,
    compute_observer_contract_binding,
    compute_outcome_generator_precommitment,
    compute_outcome_secret_commitment,
    compute_prediction_phase_commitment,
    frozen_shuffle_permutation,
    frozen_source_outcomes,
    semantic_observer_event_id,
    verify_operational_observer_manifest,
    write_operational_observer_report,
)
from oph_fpe.ontology import canonical_hash
from oph_fpe.repair.transaction import (
    build_repair_replay_envelope,
    verify_repair_replay_envelope,
)


def _raw_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest_hash(manifest_path: Path, role: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_id = manifest["role_bindings"][role]
    for row in manifest["artifacts"]:
        if row["artifact_id"] == artifact_id:
            row["sha256"] = _raw_hash(manifest_path.parent / row["path"])
            break
    _write_json(manifest_path, manifest)


def _rewrite_parent_hash(
    manifest_path: Path,
    *,
    role: str,
    manifest_field: str,
) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_id = manifest["role_bindings"][role]
    parent_path: Path | None = None
    for row in manifest["artifacts"]:
        if row["artifact_id"] == artifact_id:
            parent_path = manifest_path.parent / row["path"]
            row["sha256"] = _raw_hash(parent_path)
            break
    assert parent_path is not None
    manifest[manifest_field] = _raw_hash(parent_path)
    _write_json(manifest_path, manifest)
    return manifest[manifest_field]


def _binding_arrays(binding: dict[str, str]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in binding.items()}


def _canonical_federation_bundle() -> dict[str, Any]:
    carrier = reference_echosahedral_carrier("c0")
    algebra_hash = interface_algebra_sha256({"algebra": "observer-test"})
    federation = EchosahedralFederation(
        federation_id="observer-parent-federation",
        carriers=(carrier,),
        seams=(),
        external_boundaries=(
            ExternalBoundaryBundle(
                boundary_id="external",
                carrier_id="c0",
                ports=tuple(range(12)),
                boundary_condition="open_external",
                boundary_algebra_sha256=algebra_hash,
            ),
        ),
        observer_supports=(
            ObserverSupport(
                observer_token="observer-parent",
                carrier_ids=frozenset({"c0"}),
                visible_seam_ids=frozenset(),
                record_algebra_sha256=algebra_hash,
                checkpoint_cut_sha256=algebra_hash,
            ),
        ),
    )
    return reference_federation_instrument_bundle(federation)


def _canonical_repair_artifact(proposal_id: str) -> dict[str, Any]:
    zero = {"terms": [], "constant": 0, "transform": "identity"}
    components = {
        name: dict(zero)
        for name in ("record", "sector", "holonomy", "local_constraint")
    }
    components["overlap"] = {
        "terms": [{"register": "x", "coefficient": 1}],
        "constant": 0,
        "transform": "absolute",
    }
    return build_repair_replay_envelope(
        initial_state={"x": 2},
        initial_versions={"x": 0},
        mismatch_evaluator={
            "kind": "exact_affine_ledger_v1",
            "components": components,
            "physical_auxiliary": {},
        },
        proposals=[
            {
                "proposal_id": proposal_id,
                "transition_kind": "STRICT_REPAIR",
                "proposal_class": "EXACT_SPLICE",
                "collar": {
                    "collar_id": "observer-parent-collar",
                    "visible_read_set": ["x"],
                    "writable_registers": ["x"],
                    "protected_boundary": [],
                    "sector_registers": [],
                    "record_registers": [],
                    "checkpoint_registers": [],
                    "interior_registers": ["x"],
                    "carrier_ids": [],
                    "seam_ids": [],
                    "forbidden_target_fields": [],
                },
                "declared_read_set": ["x"],
                "recovery": {
                    "kind": "literal_updates_v1",
                    "updates": {"x": 1},
                },
                "inverse_updates": {},
                "source_parameters": {"source": "frozen local splice"},
                "parent_event_ids": [],
            }
        ],
    )


def _record_commit_envelope(
    *,
    proposal_id: str,
    record_id: str,
    value: int,
    source_primitive_commitment: str,
    parent_event_ids: list[str],
    leak_outcome_commitment: str | None = None,
) -> dict[str, Any]:
    zero = {"terms": [], "constant": 0, "transform": "identity"}
    components = {
        name: dict(zero)
        for name in (
            "record",
            "sector",
            "holonomy",
            "overlap",
            "local_constraint",
        )
    }
    source_parameters = {
        "source_primitive_commitment": source_primitive_commitment
    }
    if leak_outcome_commitment is not None:
        source_parameters["outcome_secret_commitment"] = leak_outcome_commitment
    register = f"record:{record_id}"
    return build_repair_replay_envelope(
        initial_state={register: []},
        initial_versions={register: 0},
        mismatch_evaluator={
            "kind": "exact_affine_ledger_v1",
            "components": components,
            "physical_auxiliary": {},
        },
        proposals=[
            {
                "proposal_id": proposal_id,
                "transition_kind": "RECORD_COMMIT",
                "proposal_class": "EXACT_SPLICE",
                "collar": {
                    "collar_id": f"record-collar:{record_id}",
                    "visible_read_set": [register],
                    "writable_registers": [register],
                    "protected_boundary": [],
                    "sector_registers": [],
                    "record_registers": [register],
                    "checkpoint_registers": [],
                    "interior_registers": [],
                    "carrier_ids": ["c0"],
                    "seam_ids": [],
                    "forbidden_target_fields": [],
                },
                "declared_read_set": [register],
                "recovery": {
                    "kind": "append_literal_v1",
                    "register": register,
                    "value": {"record_id": record_id, "value": value},
                },
                "inverse_updates": {},
                "source_parameters": source_parameters,
                "parent_event_ids": parent_event_ids,
            }
        ],
    )


def _build_bundle(
    root: Path,
    *,
    bundle_id: str = "observer-bundle-a",
    source_label: str = "source-a",
    source_artifact_type: str | None = None,
    break_ancestry: bool = False,
    all_zero_predictions: bool = False,
    leak_outcome_commitment_into_record: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stimuli = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    features_path = root / "source_features.npy"
    np.save(features_path, stimuli, allow_pickle=False)
    decoded_features_hash = canonical_hash(
        {
            "dtype": stimuli.dtype.str,
            "shape": list(stimuli.shape),
            "data": stimuli.reshape(-1).tolist(),
        },
        domain="oph.primitive-artifact.value.v1",
    )
    source_primitive_commitment = canonical_hash(
        {"source_label": source_label}, domain="test.source.v1"
    )
    outcome_secret = "secret-1"
    outcome_secret_commitment = compute_outcome_secret_commitment(outcome_secret)
    outcome_generator_precommitment = compute_outcome_generator_precommitment(
        source_primitive_commitment=source_primitive_commitment,
        generator_id="sha256_source_feature_counter_v1",
        outcome_secret_commitment=outcome_secret_commitment,
        action_modulus=2,
        sample_count=len(stimuli),
    )
    source_path = root / "source_receipt.json"
    source_payload = {
        "schema": "oph.test.source-receipt.v1",
        "source_label": source_label,
        "primitive_commitment": source_primitive_commitment,
        "outcome_secret_commitment": outcome_secret_commitment,
        "outcome_generator_precommitment": outcome_generator_precommitment,
    }
    if source_artifact_type is not None:
        source_payload["artifact_type"] = source_artifact_type
    _write_json(source_path, source_payload)
    source_hash = _raw_hash(source_path)
    federation_parent_path = root / "federation_bundle.json"
    _write_json(federation_parent_path, _canonical_federation_bundle())
    federation_hash = _raw_hash(federation_parent_path)
    repair_path = root / "canonical_repair.json"
    _write_json(repair_path, _canonical_repair_artifact(f"{bundle_id}-repair"))
    repair_hash = _raw_hash(repair_path)
    transaction_envelope_path = root / "transaction_parent_envelope.json"
    _write_json(
        transaction_envelope_path,
        {
            "schema": TRANSACTION_PARENT_ENVELOPE_SCHEMA,
            "source_bundle_receipt_hash": source_hash,
            "federation_bundle_receipt_hash": federation_hash,
            "canonical_repair_artifact_hash": repair_hash,
        },
    )
    transaction_envelope_hash = _raw_hash(transaction_envelope_path)
    outcomes = frozen_source_outcomes(
        stimuli,
        source_primitive_commitment=source_primitive_commitment,
        outcome_secret=outcome_secret,
        action_modulus=2,
    )
    predictions = (
        np.zeros(6, dtype=np.int64)
        if all_zero_predictions
        else outcomes.copy()
    )

    evaluator_id = "observer-small-model-evaluator-v1"
    evaluator_path = root / "evaluator.json"
    _write_json(
        evaluator_path,
        {
            "schema": "oph.operational-observer.evaluator.v1",
            "evaluator_id": evaluator_id,
            "prediction_rule": "single_committed_record_value_v1",
            "action_rule": "modular_feedback_v1",
            "shuffle_rule": "sha256_fisher_yates_v1",
            "checkpoint_rule": "exact_suffix_replay_v1",
        },
    )
    configuration_id = "observer-small-model-config-v1"
    configuration_path = root / "configuration.json"
    _write_json(
        configuration_path,
        {
            "schema": "oph.operational-observer.configuration.v2",
            "configuration_id": configuration_id,
            "action_modulus": 2,
            "neutral_feedback": 0,
            "max_interface_ports": 24,
            "minimum_prediction_advantage_count": 2,
        },
    )
    seed_id = "observer-seeds-a"
    seed_path = root / "seed.json"
    _write_json(
        seed_path,
        {
            "schema": "oph.operational-observer.seed.v1",
            "seed_id": seed_id,
            "run_seed": 101,
            "shuffle_seed": 0,
        },
    )

    artifact_ids = {
        "source_bundle_receipt": "source-parent",
        "federation_bundle_receipt": "federation-parent",
        "canonical_repair_artifact": "canonical-repair-parent",
        "transaction_parent_envelope": "transaction-parent-envelope",
        "evaluator": "observer-evaluator",
        "configuration": "observer-configuration",
        "seed": "observer-seed",
        "federation_support": "federation-support",
        "semantic_trace": "semantic-trace",
        "checkpoint": "observer-checkpoint",
        "source_features": "source-features",
        "source_features_binding": "source-features-binding",
        "record_commit_provenance": "record-commit-provenance",
        "outcome_provenance": "outcome-provenance",
        "frozen_control": "frozen-control",
    }
    contract_hash = compute_observer_contract_binding(
        bundle_id=bundle_id,
        source_bundle_receipt_hash=source_hash,
        federation_bundle_receipt_hash=federation_hash,
        canonical_repair_artifact_hash=repair_hash,
        transaction_parent_envelope_hash=transaction_envelope_hash,
        evaluator_artifact_id=artifact_ids["evaluator"],
        evaluator_sha256=_raw_hash(evaluator_path),
        evaluator_id=evaluator_id,
        configuration_artifact_id=artifact_ids["configuration"],
        configuration_sha256=_raw_hash(configuration_path),
        configuration_id=configuration_id,
        seed_artifact_id=artifact_ids["seed"],
        seed_sha256=_raw_hash(seed_path),
        seed_id=seed_id,
        run_seed=101,
        shuffle_seed=0,
    )
    binding = {
        "source_bundle_receipt_hash": source_hash,
        "federation_bundle_receipt_hash": federation_hash,
        "canonical_repair_artifact_hash": repair_hash,
        "transaction_parent_envelope_hash": transaction_envelope_hash,
        "evaluator_artifact_id": artifact_ids["evaluator"],
        "configuration_artifact_id": artifact_ids["configuration"],
        "seed_artifact_id": artifact_ids["seed"],
        "evaluator_id": evaluator_id,
        "configuration_id": configuration_id,
        "seed_id": seed_id,
        "contract_binding_sha256": contract_hash,
    }

    support_path = root / "support.json"
    interface_ports = [["c0", port] for port in range(12)]
    _write_json(
        support_path,
        {
            "schema": "oph.operational-observer.federation-support.v1",
            "binding": binding,
            "federation_id": "observer-parent-federation",
            "observer_id": "observer-parent",
            "carriers": ["c0"],
            "seams": [],
            "external_boundaries": [
                {"carrier_id": "c0", "ports": list(range(12))},
            ],
            "observer_support": {
                "carrier_ids": ["c0"],
                "visible_seam_ids": [],
                "interface_ports": interface_ports,
            },
        },
    )

    features_binding_path = root / "source_features_binding.json"
    _write_json(
        features_binding_path,
        {
            "schema": "oph.operational-observer.source-features-binding.v1",
            "binding": binding,
            "source_features_artifact_id": artifact_ids["source_features"],
            "source_features_sha256": _raw_hash(features_path),
            "source_features_decoded_sha256": decoded_features_hash,
        },
    )

    events: list[dict[str, Any]] = []
    record_provenance_entries: list[dict[str, Any]] = []
    last_id: str | None = None
    record_origins: dict[str, str] = {}
    for index, prediction in enumerate(predictions.tolist()):
        carrier = "c0"
        record_id = f"r{index}"
        event_parents = [] if last_id is None else [last_id]
        record_envelope = _record_commit_envelope(
            proposal_id=f"{bundle_id}-record-{index}",
            record_id=record_id,
            value=prediction,
            source_primitive_commitment=source_primitive_commitment,
            parent_event_ids=event_parents,
            leak_outcome_commitment=(
                outcome_secret_commitment
                if leak_outcome_commitment_into_record and index == 0
                else None
            ),
        )
        record_replay = verify_repair_replay_envelope(record_envelope)
        assert record_replay["RECORD_COMMIT_REPLAY_RECEIPT"] is True
        commit_without_id = {
            "kind": "RECORD_COMMIT",
            "parents": event_parents,
            "carrier_id": carrier,
            "record_id": record_id,
            "value": prediction,
            "commit_status": "COMMITTED",
            "canonical_repair_artifact_hash": repair_hash,
            "record_replay_envelope_sha256": (
                "sha256:" + record_replay["artifact_hash"]
            ),
            "semantic_record_event_id": (
                "sha256:" + record_replay["semantic_record_event_id"]
            ),
        }
        commit_id = semantic_observer_event_id(
            commit_without_id,
            observer_id="observer-parent",
            contract_binding_sha256=contract_hash,
        )
        events.append({**commit_without_id, "event_id": commit_id})
        record_provenance_entries.append(
            {
                "observer_event_id": commit_id,
                "record_id": record_id,
                "record_register": f"record:{record_id}",
                "value": prediction,
                "replay_envelope": record_envelope,
            }
        )
        record_origins[record_id] = commit_id

        prediction_parents = [] if break_ancestry and index == 0 else [commit_id]
        prediction_without_id = {
            "kind": "READBACK_PREDICTION",
            "parents": prediction_parents,
            "carrier_id": carrier,
            "reads": [record_id],
            "feature_index": index,
            "prediction": prediction,
        }
        prediction_id = semantic_observer_event_id(
            prediction_without_id,
            observer_id="observer-parent",
            contract_binding_sha256=contract_hash,
        )
        events.append({**prediction_without_id, "event_id": prediction_id})

        action = int((stimuli[index] + prediction) % 2)
        action_without_id = {
            "kind": "LOCAL_ACTION",
            "parents": [prediction_id],
            "carrier_id": carrier,
            "reads": [record_id],
            "feature_index": index,
            "feedback_event_id": prediction_id,
            "action": action,
        }
        action_id = semantic_observer_event_id(
            action_without_id,
            observer_id="observer-parent",
            contract_binding_sha256=contract_hash,
        )
        events.append({**action_without_id, "event_id": action_id})
        last_id = action_id

    trace_path = root / "trace.json"
    _write_json(
        trace_path,
        {
            "schema": "oph.operational-observer.semantic-trace.v1",
            "binding": binding,
            "observer_id": "observer-parent",
            "events": events,
        },
    )
    record_provenance_path = root / "record_commit_provenance.json"
    _write_json(
        record_provenance_path,
        {
            "schema": (
                "oph.operational-observer.record-commit-provenance.v1"
            ),
            "binding": binding,
            "entries": record_provenance_entries,
        },
    )
    outcome_provenance_path = root / "outcome_provenance.json"
    _write_json(
        outcome_provenance_path,
        {
            "schema": "oph.operational-observer.outcome-provenance.v1",
            "binding": binding,
            "generator_id": "sha256_source_feature_counter_v1",
            "source_parent_primitive_commitment": (
                source_primitive_commitment
            ),
            "source_generator_precommitment": (
                outcome_generator_precommitment
            ),
            "source_features_artifact_id": artifact_ids["source_features"],
            "source_features_sha256": _raw_hash(features_path),
            "source_features_decoded_sha256": decoded_features_hash,
            "outcome_secret": outcome_secret,
            "outcome_secret_commitment": outcome_secret_commitment,
            "prediction_phase_commitment": (
                compute_prediction_phase_commitment(events)
            ),
            "action_modulus": 2,
            "sample_count": len(stimuli),
            "generated_outcomes_commitment": canonical_hash(
                outcomes.tolist(),
                domain="oph.operational-observer.generated-outcomes.v1",
            ),
        },
    )

    permutation = frozen_shuffle_permutation(len(predictions), 0)
    control_path = root / "control.npz"
    np.savez(
        control_path,
        schema=np.asarray("oph.operational-observer.frozen-control.v2"),
        **_binding_arrays(binding),
        permutation=permutation,
        shuffled_predictions=predictions[permutation],
    )

    cut_index = 8
    prefix_record_state = {f"r{index}": int(predictions[index]) for index in range(3)}
    prefix_origins = {f"r{index}": record_origins[f"r{index}"] for index in range(3)}
    suffix_predictions = [
        {
            "event_id": events[index * 3 + 1]["event_id"],
            "feature_index": index,
            "prediction": int(predictions[index]),
        }
        for index in range(3, 6)
    ]
    suffix_actions = [
        {
            "event_id": events[index * 3 + 2]["event_id"],
            "feature_index": index,
            "action": int((stimuli[index] + predictions[index]) % 2),
        }
        for index in range(3, 6)
    ]
    event_ids = [event["event_id"] for event in events]
    final_record_state = {
        f"r{index}": int(predictions[index]) for index in range(len(predictions))
    }
    continuation_hash = canonical_hash(
        {
            "record_state": final_record_state,
            "suffix_predictions": suffix_predictions,
            "suffix_actions": suffix_actions,
            "last_event_id": event_ids[-1],
        },
        domain="oph.operational-observer.continuation-state.v1",
    )
    checkpoint_material = {
        "schema": "oph.operational-observer.checkpoint.v1",
        "binding": binding,
        "observer_id": "observer-parent",
        "continuation_observer_id": "observer-parent",
        "cut_event_id": event_ids[cut_index],
        "next_event_index": cut_index + 1,
        "record_state": prefix_record_state,
        "record_origins": prefix_origins,
        "semantic_history_root": canonical_hash(
            event_ids[: cut_index + 1],
            domain="oph.operational-observer.history-root.v1",
        ),
        "committed_suffix_event_ids": event_ids[cut_index + 1 :],
        "continuation_state_hash": continuation_hash,
    }
    checkpoint_path = root / "checkpoint.json"
    _write_json(
        checkpoint_path,
        {
            **checkpoint_material,
            "checkpoint_hash": canonical_hash(
                checkpoint_material,
                domain="oph.operational-observer.checkpoint.v1",
            ),
        },
    )

    paths = {
        "source_bundle_receipt": source_path,
        "federation_bundle_receipt": federation_parent_path,
        "canonical_repair_artifact": repair_path,
        "transaction_parent_envelope": transaction_envelope_path,
        "evaluator": evaluator_path,
        "configuration": configuration_path,
        "seed": seed_path,
        "federation_support": support_path,
        "semantic_trace": trace_path,
        "checkpoint": checkpoint_path,
        "source_features": features_path,
        "source_features_binding": features_binding_path,
        "record_commit_provenance": record_provenance_path,
        "outcome_provenance": outcome_provenance_path,
        "frozen_control": control_path,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle_id": bundle_id,
        "source_bundle_receipt_hash": source_hash,
        "federation_bundle_receipt_hash": federation_hash,
        "canonical_repair_artifact_hash": repair_hash,
        "transaction_parent_envelope_hash": transaction_envelope_hash,
        "evaluator_artifact_id": artifact_ids["evaluator"],
        "configuration_artifact_id": artifact_ids["configuration"],
        "seed_artifact_id": artifact_ids["seed"],
        "contract_binding_sha256": contract_hash,
        "artifacts": [
            {
                "artifact_id": artifact_ids[role],
                "path": path.name,
                "format": path.suffix.removeprefix("."),
                "role": role,
                "sha256": _raw_hash(path),
                "contract_binding_sha256": contract_hash,
            }
            for role, path in paths.items()
        ],
        "role_bindings": artifact_ids,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_exact_small_model_emits_only_finite_operational_receipts(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    report = verify_operational_observer_manifest(manifest_path)

    for key in (
        OBSERVER_ARTIFACT_INTEGRITY_RECEIPT,
        OBSERVER_CONTRACT_BINDING_RECEIPT,
        OBSERVER_SOURCE_FIREWALL_RECEIPT,
        OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT,
        OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT,
        A3_RECORD_COMMIT_REPLAY_RECEIPT,
        A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT,
        A3_READBACK_PREDICTION_CONTROL_RECEIPT,
        A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT,
        A4_BOUNDED_INTERFACE_RECEIPT,
        A4_FEEDBACK_ABLATION_RECEIPT,
        A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT,
        OPERATIONAL_SELF_READING_OBSERVER_RECEIPT,
    ):
        assert report[key] is True
    assert report["receipt"] is True
    assert report["frozen_prediction_control"]["direct_match_count"] == 6
    assert report["frozen_prediction_control"]["shuffled_match_count"] <= 4
    assert report["feedback_ablation"]["changed_action_count"] == 4
    assert report["outcome_provenance"]["claim_tier"] == (
        "SYNTHETIC_COMMIT_REVEAL_PROTOCOL_MECHANICS"
    )
    assert report["outcome_provenance"]["temporal_external_timestamp_claim"] is False
    assert report["PHYSICAL_PREDICTIVE_INDEPENDENCE_RECEIPT"] is False
    predictor_config = json.loads(
        (manifest_path.parent / "configuration.json").read_text(encoding="utf-8")
    )
    assert not any(
        token in key for key in predictor_config for token in ("outcome", "secret")
    )
    assert report["PHYSICAL_GEOMETRY_RECEIPT"] is False
    assert report["INDEPENDENT_PHYSICAL_CLOCK_RECEIPT"] is False
    assert report["GRAVITY_EMERGENCE_RECEIPT"] is False
    assert report["STANDARD_MODEL_EMERGENCE_RECEIPT"] is False


def test_emergence_ladder_replays_observer_report_and_requires_root_binding(
    tmp_path: Path,
) -> None:
    from oph_fpe.emergence_ladder import audit_emergence_ladder

    bundle_dir = tmp_path / "bundle"
    manifest_path = _build_bundle(bundle_dir)
    report_path = bundle_dir / "observer_verification.json"
    write_operational_observer_report(manifest_path, report_path)

    ladder = audit_emergence_ladder(bundle_dir)
    a3 = ladder["dag"]["stages"]["A3"]
    a4 = ladder["dag"]["stages"]["A4"]

    assert all(item["passed"] is True for item in a3["evidence"].values())
    assert all(item["passed"] is True for item in a4["evidence"].values())
    assert a3["common_source_binding_required"] is True
    assert a3["common_source_binding_verified"] is False
    assert any(
        blocker.startswith("common_source_commitment_unbound:")
        for blocker in a3["blockers"]
    )
    assert a3["passed"] is False
    assert a4["passed"] is False
    registry_row = next(
        row
        for row in ladder["source_inventory"]["registered_artifacts"]
        if row["report_path"] == "observer_verification.json"
    )
    assert registry_row["passed"] is True
    assert registry_row["source_binding_status"] == (
        "observer_source_parent_is_not_verified_c0_report"
    )

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered[A3_RECORD_COMMIT_REPLAY_RECEIPT] = False
    _write_json(report_path, tampered)
    rejected = audit_emergence_ladder(bundle_dir)
    rejected_a3 = rejected["dag"]["stages"]["A3"]
    assert rejected_a3["evidence"]["observer_artifact_integrity"]["passed"] is False
    assert rejected["source_inventory"]["input_errors"]


def test_ladder_binds_observer_federation_and_repair_to_one_verified_source(
    tmp_path: Path, monkeypatch
) -> None:
    from oph_fpe.emergence_ladder import audit_emergence_ladder

    bundle_dir = tmp_path / "bundle"
    manifest_path = _build_bundle(
        bundle_dir,
        source_artifact_type=COMMON_SOURCE_REPORT_ARTIFACT_TYPE,
    )
    report_path = bundle_dir / "observer_verification.json"
    write_operational_observer_report(manifest_path, report_path)
    commitment = "one-source-commitment"
    artifact_rows = [
        {
            "artifact_id": "federation-parent",
            "semantic_role": "authoritative_presentation",
            "actual_sha256": _raw_hash(bundle_dir / "federation_bundle.json"),
            "passed": True,
        },
        {
            "artifact_id": "repair-parent",
            "semantic_role": "repair_log",
            "actual_sha256": _raw_hash(bundle_dir / "canonical_repair.json"),
            "passed": True,
        },
    ]

    def replay(_path: Path) -> dict[str, Any]:
        return {
            "passed": True,
            "blockers": [],
            "recomputed_report": {
                **{key: True for key in C0_RECEIPT_KEYS},
                "computed_bundle_commitment": commitment,
                "artifact_verification": {"rows": artifact_rows},
            },
        }

    monkeypatch.setattr(
        "oph_fpe.emergence_ladder.verify_common_source_tower_report_file",
        replay,
    )
    ladder = audit_emergence_ladder(bundle_dir)
    a3 = ladder["dag"]["stages"]["A3"]
    registry_row = next(
        row
        for row in ladder["source_inventory"]["registered_artifacts"]
        if row["report_path"] == "observer_verification.json"
    )

    assert a3["common_source_binding_required"] is True
    assert a3["common_source_binding_verified"] is True
    assert set(a3["closure_source_bindings"].values()) == {commitment}
    assert registry_row["source_binding_status"] == (
        "observer_parents_bound_to_verified_common_source"
    )
    assert registry_row["observer_parent_binding"] == {
        "source_bundle_receipt_sha256": _raw_hash(
            bundle_dir / "source_receipt.json"
        ).removeprefix("sha256:"),
        "canonical_repair_artifact_sha256": _raw_hash(
            bundle_dir / "canonical_repair.json"
        ).removeprefix("sha256:"),
        "federation_bundle_receipt_sha256": _raw_hash(
            bundle_dir / "federation_bundle.json"
        ).removeprefix("sha256:"),
        "source_parent_commitments": [commitment],
        "repair_parent_commitments": [commitment],
        "federation_parent_commitments": [commitment],
        "repair_parent_is_verified_repair": True,
        "federation_parent_is_verified_bundle": True,
    }


def test_manifest_booleans_cannot_authorize_a_receipt(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["receipt"] = True
    _write_json(manifest_path, manifest)

    report = verify_operational_observer_manifest(manifest_path)

    assert report["receipt"] is False
    assert any("manifest_fields_invalid" in blocker for blocker in report["blockers"])


def test_artifact_hash_drift_fails_before_semantic_replay(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    trace_path = manifest_path.parent / "trace.json"
    trace_path.write_bytes(trace_path.read_bytes() + b" ")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is False
    assert report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any("declared_sha256_mismatch" in blocker for blocker in report["blockers"])


def test_required_artifact_deletion_fails_closed(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    (manifest_path.parent / "checkpoint.json").unlink()

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is False
    assert report[A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT] is False


def test_cross_source_splice_is_rejected_even_when_spliced_hash_is_declared(
    tmp_path: Path,
) -> None:
    manifest_a = _build_bundle(tmp_path / "a", source_label="source-a")
    manifest_b = _build_bundle(tmp_path / "b", source_label="source-b")
    support_a = manifest_a.parent / "support.json"
    support_a.write_bytes((manifest_b.parent / "support.json").read_bytes())
    _rewrite_manifest_hash(manifest_a, "federation_support")

    report = verify_operational_observer_manifest(manifest_a)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT] is False
    assert any("evidence_contract_binding_mismatch" in blocker for blocker in report["blockers"])


def test_transaction_parent_envelope_rejects_partial_repair_splice(
    tmp_path: Path,
) -> None:
    manifest_a = _build_bundle(tmp_path / "a", bundle_id="bundle-a")
    manifest_b = _build_bundle(tmp_path / "b", bundle_id="bundle-b")
    repair_a = manifest_a.parent / "canonical_repair.json"
    repair_a.write_bytes((manifest_b.parent / "canonical_repair.json").read_bytes())
    repair_hash = _rewrite_parent_hash(
        manifest_a,
        role="canonical_repair_artifact",
        manifest_field="canonical_repair_artifact_hash",
    )
    envelope_path = manifest_a.parent / "transaction_parent_envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["canonical_repair_artifact_hash"] = repair_hash
    _write_json(envelope_path, envelope)
    _rewrite_parent_hash(
        manifest_a,
        role="transaction_parent_envelope",
        manifest_field="transaction_parent_envelope_hash",
    )

    report = verify_operational_observer_manifest(manifest_a)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[OBSERVER_CONTRACT_BINDING_RECEIPT] is False
    assert report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any(
        blocker == "computed_contract_binding_mismatch"
        for blocker in report["blockers"]
    )


def test_transaction_parent_envelope_rejects_forged_repair_hash(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    envelope_path = manifest_path.parent / "transaction_parent_envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["canonical_repair_artifact_hash"] = "sha256:" + "f" * 64
    _write_json(envelope_path, envelope)
    _rewrite_parent_hash(
        manifest_path,
        role="transaction_parent_envelope",
        manifest_field="transaction_parent_envelope_hash",
    )

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[OBSERVER_CONTRACT_BINDING_RECEIPT] is False
    assert any(
        blocker
        == "transaction_parent_envelope_mismatch:canonical_repair_artifact_hash"
        for blocker in report["blockers"]
    )


def test_canonical_repair_tamper_is_independently_replayed(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    repair_path = manifest_path.parent / "canonical_repair.json"
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    repair["TRANSACTIONAL_REPAIR_RECEIPT"] = False
    _write_json(repair_path, repair)
    repair_hash = _rewrite_parent_hash(
        manifest_path,
        role="canonical_repair_artifact",
        manifest_field="canonical_repair_artifact_hash",
    )
    envelope_path = manifest_path.parent / "transaction_parent_envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["canonical_repair_artifact_hash"] = repair_hash
    _write_json(envelope_path, envelope)
    _rewrite_parent_hash(
        manifest_path,
        role="transaction_parent_envelope",
        manifest_field="transaction_parent_envelope_hash",
    )

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[OBSERVER_CONTRACT_BINDING_RECEIPT] is False
    assert any(
        blocker
        == "canonical_repair_verification_failed:REPAIR_ARTIFACT_INTEGRITY_RECEIPT"
        for blocker in report["blockers"]
    )


def test_runtime_or_target_metadata_is_forbidden_from_semantic_trace(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    trace_path = manifest_path.parent / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["worker_id"] = "worker-7"
    _write_json(trace_path, trace)
    _rewrite_manifest_hash(manifest_path, "semantic_trace")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_SOURCE_FIREWALL_RECEIPT] is False
    assert report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any("forbidden_semantic_field" in blocker for blocker in report["blockers"])

    target_manifest = _build_bundle(tmp_path / "target-bundle")
    descriptor_path = target_manifest.parent / "source_features_binding.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["target_geometry"] = "H3"
    _write_json(descriptor_path, descriptor)
    _rewrite_manifest_hash(target_manifest, "source_features_binding")

    target_report = verify_operational_observer_manifest(target_manifest)

    assert target_report[OBSERVER_SOURCE_FIREWALL_RECEIPT] is False
    assert target_report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any(
        "source_features_binding" in blocker
        for blocker in target_report["blockers"]
    )


def test_record_read_requires_causal_write_ancestry(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle", break_ancestry=True)

    report = verify_operational_observer_manifest(manifest_path)

    assert report[A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT] is False
    assert report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any("record_write_not_causal_ancestor" in blocker for blocker in report["blockers"])


def test_hand_authored_record_event_without_primitive_replay_cannot_pass(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    provenance_path = manifest_path.parent / "record_commit_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["entries"] = provenance["entries"][1:]
    _write_json(provenance_path, provenance)
    _rewrite_manifest_hash(manifest_path, "record_commit_provenance")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT] is True
    assert report[A3_RECORD_COMMIT_REPLAY_RECEIPT] is False
    assert report[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] is False
    assert any(
        "record_commit_has_no_primitive_replay" in blocker
        for blocker in report["blockers"]
    )


def test_direct_outcome_array_leak_is_not_an_admitted_control_input(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    control_path = manifest_path.parent / "control.npz"
    with np.load(control_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["heldout_outcomes"] = np.asarray([1, 1, 1, 1, 0, 0], dtype=np.int64)
    np.savez(control_path, **values)
    _rewrite_manifest_hash(manifest_path, "frozen_control")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[A3_READBACK_PREDICTION_CONTROL_RECEIPT] is False
    assert any(
        "frozen_control_fields_not_exact" in blocker
        for blocker in report["blockers"]
    )


def test_predictor_side_record_provenance_cannot_depend_on_outcome_commitment(
    tmp_path: Path,
) -> None:
    manifest_path = _build_bundle(
        tmp_path / "bundle",
        leak_outcome_commitment_into_record=True,
    )

    report = verify_operational_observer_manifest(manifest_path)

    assert report[OBSERVER_ARTIFACT_INTEGRITY_RECEIPT] is True
    assert report[OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT] is False
    assert report[OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT] is False
    assert report[A3_RECORD_COMMIT_REPLAY_RECEIPT] is False
    assert report[A3_READBACK_PREDICTION_CONTROL_RECEIPT] is False
    assert report["PHYSICAL_PREDICTIVE_INDEPENDENCE_RECEIPT"] is False
    assert any(
        "record_replay_source_parent_mismatch" in blocker
        for blocker in report["blockers"]
    )
    assert any(
        "outcome_secret_or_commitment_leaked_to_prediction_side" in blocker
        for blocker in report["blockers"]
    )


def test_frozen_shuffle_and_checkpoint_are_recomputed_not_trusted(tmp_path: Path) -> None:
    manifest_path = _build_bundle(tmp_path / "bundle")
    control_path = manifest_path.parent / "control.npz"
    with np.load(control_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["permutation"] = np.arange(6, dtype=np.int64)
    np.savez(control_path, **values)
    _rewrite_manifest_hash(manifest_path, "frozen_control")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[A3_READBACK_PREDICTION_CONTROL_RECEIPT] is False
    assert any("frozen_permutation_seed_replay_mismatch" in blocker for blocker in report["blockers"])

    manifest_path = _build_bundle(tmp_path / "checkpoint-bundle")
    checkpoint_path = manifest_path.parent / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["record_state"]["r0"] = 1 - checkpoint["record_state"]["r0"]
    checkpoint_material = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_hash"
    }
    checkpoint["checkpoint_hash"] = canonical_hash(
        checkpoint_material,
        domain="oph.operational-observer.checkpoint.v1",
    )
    _write_json(checkpoint_path, checkpoint)
    _rewrite_manifest_hash(manifest_path, "checkpoint")

    report = verify_operational_observer_manifest(manifest_path)

    assert report[A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT] is False
    assert any("checkpoint_record_state_mismatch" in blocker for blocker in report["blockers"])


def test_feedback_ablation_requires_a_changed_later_action(tmp_path: Path) -> None:
    manifest_path = _build_bundle(
        tmp_path / "bundle",
        all_zero_predictions=True,
    )

    report = verify_operational_observer_manifest(manifest_path)

    assert report[A4_FEEDBACK_ABLATION_RECEIPT] is False
    assert any(
        "feedback_ablation_changes_no_later_local_action" in blocker
        for blocker in report["blockers"]
    )
