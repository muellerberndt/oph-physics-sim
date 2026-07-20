from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.common_source_tower import (
    C0_RECEIPT_KEYS,
    DEFAULT_MANIFEST_NAME,
    ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT,
    MANIFEST_SCHEMA,
    REQUIRED_ROLES,
    SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT,
    SOURCE_TOWER_NO_TARGET_PATH_RECEIPT,
    SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT,
    _generated_permutation_group,
    verify_common_domain_source_tower,
    verify_common_source_tower_report,
    write_common_domain_source_tower_report,
)
from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations


class _FixtureBuilder:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts: list[dict[str, Any]] = []
        self.processes: list[dict[str, Any]] = []

    def _record(
        self,
        artifact_id: str,
        path: Path,
        format_name: str,
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        raw = path.read_bytes()
        self.artifacts.append(
            {
                "artifact_id": artifact_id,
                "path": path.relative_to(self.root).as_posix(),
                "format": format_name,
                "artifact_class": artifact_class,
                "semantic_role": semantic_role,
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }
        )
        return artifact_id

    def json(
        self,
        artifact_id: str,
        value: Any,
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        path = self.root / f"{artifact_id}.json"
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return self._record(
            artifact_id, path, "json", artifact_class, semantic_role
        )

    def npy(
        self,
        artifact_id: str,
        value: Any,
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        path = self.root / f"{artifact_id}.npy"
        np.save(path, np.asarray(value), allow_pickle=False)
        return self._record(
            artifact_id, path, "npy", artifact_class, semantic_role
        )

    def npz(
        self,
        artifact_id: str,
        value: dict[str, np.ndarray],
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        path = self.root / f"{artifact_id}.npz"
        np.savez(path, **value)
        return self._record(
            artifact_id, path, "npz", artifact_class, semantic_role
        )

    def process(
        self,
        process_id: str,
        inputs: list[str],
        output: str,
        evaluator: str,
        config: str,
        seed: str,
    ) -> None:
        self.processes.append(
            {
                "process_id": process_id,
                "data_input_artifact_ids": inputs,
                "output_artifact_id": output,
                "evaluator_artifact_id": evaluator,
                "configuration_artifact_id": config,
                "seed_artifact_id": seed,
            }
        )


def _a5_generators() -> tuple[np.ndarray, np.ndarray]:
    permutations = icosahedral_a5_port_permutations()
    for left in permutations:
        if left == tuple(range(12)):
            continue
        for right in permutations:
            if right == tuple(range(12)) or right == left:
                continue
            if len(_generated_permutation_group((left, right))) == 60:
                return np.asarray(left), np.asarray(right)
    raise AssertionError("the exact icosahedral action has no two-generator A5 pair")


def _write_valid_fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    builder = _FixtureBuilder(root)
    x_f = np.asarray([1.0, 2.0, 3.0, 4.0])
    coarse = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    x_c = coarse @ x_f
    patch_count = 20  # the unrefined icosahedral face count
    # Integer-valued invariant functional keeps the exact-envelope test about
    # relabeling, not floating-point reduction order.
    weights = np.ones(12, dtype=float)
    channel_values: dict[str, np.ndarray] = {}
    channels = (
        "accessible_algebras",
        "port_restrictions",
        "records",
        "repairs",
        "checkpoints",
        "semantic_event_history",
        "physical_quotient",
    )
    for index, channel in enumerate(channels):
        channel_values[f"concrete_{channel}"] = (
            np.arange(patch_count * 12, dtype=float).reshape(patch_count, 12)
            + 100.0 * index
        )
        channel_values[f"abstract_{channel}"] = (
            channel_values[f"concrete_{channel}"] @ weights
        )

    generator_a, generator_b = _a5_generators()
    initial = np.zeros((patch_count, 12), dtype=float)
    dimension = initial.size
    repair_linear = np.stack((np.eye(dimension), np.eye(dimension)))
    repair_bias = np.stack(
        (
            np.full(dimension, 1.25),
            np.full(dimension, -0.25),
        )
    )
    repair_reference = (
        initial.reshape(-1) + repair_bias[0] + repair_bias[1]
    ).reshape(initial.shape) @ weights

    main_arrays: dict[str, np.ndarray] = {
        "physical_coarse_maps": coarse,
        "refine_fine_source": x_f,
        "refine_coarse_source": x_c,
        "fine_operator": np.eye(4),
        "coarse_operator": np.eye(2),
        "readout_down_map": coarse,
        "port_weights": weights,
        "gauge_generator_a": generator_a,
        "gauge_generator_b": generator_b,
        "repair_initial": initial,
        "repair_linear": repair_linear,
        "repair_bias": repair_bias,
        "repair_reference": repair_reference,
        **channel_values,
    }
    for role in REQUIRED_ROLES:
        if role not in {"authoritative_presentation", "physical_coarse_maps"}:
            main_arrays[role] = x_f.copy()
    for role in (
        "cap_algebras",
        "state",
        "modular_data",
        "semantic_event_graph",
        "null_charges",
        "stress",
        "entropy",
        "scale",
    ):
        main_arrays[f"coarse_{role}"] = x_c.copy()

    foreign_arrays = {
        "state": x_f.copy(),
        "entropy": x_f.copy(),
        "port_weights": weights.copy(),
        "foreign_nonce": np.asarray([572], dtype=np.int64),
    }

    evaluator_identity = builder.json(
        "eval_identity",
        {
            "schema": "oph.source-tower-evaluator.v1",
            "operation": "identity",
            "seed_policy": "bound_no_random_draws",
        },
        "evaluator",
        "identity_evaluator",
    )
    evaluator_extract = builder.json(
        "eval_npz_extract",
        {
            "schema": "oph.source-tower-evaluator.v1",
            "operation": "npz_extract",
            "seed_policy": "bound_no_random_draws",
        },
        "evaluator",
        "npz_extract_evaluator",
    )
    empty_config = builder.json(
        "config_identity", {}, "configuration", "identity_config"
    )
    seed = builder.json("bound_seed", {"seed": 573}, "seed", "frozen_seed")

    main_primitive = builder.npz(
        "main_primitive",
        main_arrays,
        "source_primitive",
        "source_primitive",
    )
    main_source = builder.npz(
        "main_source",
        main_arrays,
        "authoritative_source",
        "authoritative_presentation",
    )
    builder.process(
        "produce_main_source",
        [main_primitive],
        main_source,
        evaluator_identity,
        empty_config,
        seed,
    )

    foreign_primitive = builder.npz(
        "foreign_primitive",
        foreign_arrays,
        "source_primitive",
        "foreign_source_primitive",
    )
    foreign_source = builder.npz(
        "foreign_source",
        foreign_arrays,
        "authoritative_source",
        "foreign_authoritative_presentation",
    )
    builder.process(
        "produce_foreign_source",
        [foreign_primitive],
        foreign_source,
        evaluator_identity,
        empty_config,
        seed,
    )

    def extract(
        output_id: str,
        source_id: str,
        key: str,
        value: np.ndarray,
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        output = builder.npy(output_id, value, artifact_class, semantic_role)
        config = builder.json(
            f"config_{output_id}",
            {"input_artifact_id": source_id, "key": key},
            "configuration",
            f"extract_config_{output_id}",
        )
        builder.process(
            f"extract_{output_id}",
            [source_id],
            output,
            evaluator_extract,
            config,
            seed,
        )
        return output

    role_bindings = {"authoritative_presentation": main_source}
    for role in REQUIRED_ROLES:
        if role == "authoritative_presentation":
            continue
        artifact_class = "typed_arrow" if role == "physical_coarse_maps" else "readout"
        role_bindings[role] = extract(
            f"role_{role}",
            main_source,
            role,
            main_arrays[role],
            artifact_class,
            role,
        )

    extra: dict[str, str] = {}
    for key in (
        "refine_fine_source",
        "refine_coarse_source",
        "fine_operator",
        "coarse_operator",
        "readout_down_map",
        "port_weights",
        "gauge_generator_a",
        "gauge_generator_b",
        "repair_initial",
        "repair_linear",
        "repair_bias",
        "repair_reference",
        *channel_values.keys(),
        *(f"coarse_{role}" for role in (
            "cap_algebras",
            "state",
            "modular_data",
            "semantic_event_graph",
            "null_charges",
            "stress",
            "entropy",
            "scale",
        )),
    ):
        artifact_class = (
            "typed_arrow"
            if key in {
                "fine_operator",
                "coarse_operator",
                "readout_down_map",
                "port_weights",
                "gauge_generator_a",
                "gauge_generator_b",
                "repair_linear",
            }
            else "readout"
        )
        extra[key] = extract(
            key,
            main_source,
            key,
            main_arrays[key],
            artifact_class,
            key,
        )

    foreign_state = extract(
        "foreign_state",
        foreign_source,
        "state",
        foreign_arrays["state"],
        "negative_control",
        "state",
    )
    foreign_entropy = extract(
        "foreign_entropy",
        foreign_source,
        "entropy",
        foreign_arrays["entropy"],
        "negative_control",
        "entropy",
    )
    foreign_arrow = extract(
        "foreign_port_weights",
        foreign_source,
        "port_weights",
        foreign_arrays["port_weights"],
        "negative_control",
        "foreign_port_weights",
    )
    exact_envelope = builder.json(
        "exact_envelope", {"mode": "exact"}, "configuration", "error_envelope"
    )
    # This disconnected artifact lets target-path rejection be tested without
    # changing the positive fixture's declared artifact inventory.
    target_scale = builder.npy(
        "disconnected_target_scale",
        x_f,
        "forbidden_target",
        "target_scale",
    )

    refinement_squares = []
    for role in (
        "cap_algebras",
        "state",
        "modular_data",
        "semantic_event_graph",
        "null_charges",
        "stress",
        "entropy",
        "scale",
    ):
        refinement_squares.append(
            {
                "square_id": f"refinement_{role}",
                "readout_role": role,
                "fine_source_artifact_id": extra["refine_fine_source"],
                "coarse_source_artifact_id": extra["refine_coarse_source"],
                "source_coarse_map_artifact_id": role_bindings["physical_coarse_maps"],
                "fine_readout_artifact_id": role_bindings[role],
                "coarse_readout_artifact_id": extra[f"coarse_{role}"],
                "fine_readout_operator_artifact_id": extra["fine_operator"],
                "coarse_readout_operator_artifact_id": extra["coarse_operator"],
                "readout_coarse_map_artifact_id": extra["readout_down_map"],
                "error_envelope_artifact_id": exact_envelope,
            }
        )

    realization_channels = {}
    for channel in channels:
        realization_channels[channel] = {
            "concrete_artifact_id": extra[f"concrete_{channel}"],
            "abstract_artifact_id": extra[f"abstract_{channel}"],
            "arrow_artifact_id": extra["port_weights"],
            "arrow_kind": "port_weighted_sum",
            "error_envelope_artifact_id": exact_envelope,
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle_id": "issue-572-valid-adversarial-fixture",
        "artifacts": builder.artifacts,
        "role_bindings": role_bindings,
        "provenance_processes": builder.processes,
        "splice_controls": {
            "cap_state": {
                "foreign_state_artifact_id": foreign_state,
                "foreign_source_anchor_artifact_id": foreign_source,
            },
            "stress_entropy": {
                "foreign_entropy_artifact_id": foreign_entropy,
                "foreign_source_anchor_artifact_id": foreign_source,
            },
        },
        "refinement_squares": refinement_squares,
        "realization": {
            "patch_count": patch_count,
            "ports_per_patch": 12,
            "channels": realization_channels,
            "gauge_relabeling_control": {
                "concrete_artifact_id": extra["concrete_physical_quotient"],
                "quotient_arrow_artifact_id": extra["port_weights"],
                "quotient_arrow_kind": "port_weighted_sum",
                "reference_quotient_artifact_id": extra["abstract_physical_quotient"],
                "generator_artifact_ids": [
                    extra["gauge_generator_a"],
                    extra["gauge_generator_b"],
                ],
                "error_envelope_artifact_id": exact_envelope,
            },
            "repair_schedule_control": {
                "initial_state_artifact_id": extra["repair_initial"],
                "repair_linear_artifact_id": extra["repair_linear"],
                "repair_bias_artifact_id": extra["repair_bias"],
                "quotient_arrow_artifact_id": extra["port_weights"],
                "quotient_arrow_kind": "port_weighted_sum",
                "reference_quotient_artifact_id": extra["repair_reference"],
                "error_envelope_artifact_id": exact_envelope,
            },
            "lookalike_arrow_control": {
                "channel": "physical_quotient",
                "foreign_arrow_artifact_id": foreign_arrow,
                "foreign_source_anchor_artifact_id": foreign_source,
            },
        },
    }
    manifest_path = root / DEFAULT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Expose useful IDs for adversarial rewrites without adding manifest fields.
    manifest["_test_ids"] = {
        "target_scale": target_scale,
        "main_state": role_bindings["state"],
        "foreign_arrow": foreign_arrow,
    }
    return manifest_path, manifest


def _load_manifest_without_test_ids(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest = {key: value for key, value in manifest.items() if key != "_test_ids"}
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_valid_typed_source_tower_recomputes_every_c0_receipt(tmp_path: Path) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)

    report = verify_common_domain_source_tower(manifest_path)

    assert report["receipt"] is True
    assert all(report[key] is True for key in C0_RECEIPT_KEYS)
    assert report["artifact_verification"]["all_declared_hashes_recomputed"] is True
    assert report["provenance"]["required_roles_share_one_source"] is True
    assert "graph" not in report["provenance"]
    assert len(report["refinement_commutation"]["rows"]) == 8
    realization = report["echosahedral_abstract_realization"]
    assert realization["materialized_local_port_coordinate_count"] == 20 * 12
    assert realization["gauge_relabeling_control"]["generated_group_order"] == 60
    assert realization["repair_schedule_control"]["enumerated_schedule_count"] == 2
    json.dumps(report, allow_nan=False)


def test_written_report_is_replayed_and_exactly_compared(tmp_path: Path) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    report_path = tmp_path / "verified.json"
    report = write_common_domain_source_tower_report(manifest_path, report_path)

    assert verify_common_source_tower_report(
        report, report_path=report_path
    )["passed"] is True

    tampered = copy.deepcopy(report)
    tampered["receipt"] = False
    validation = verify_common_source_tower_report(tampered, report_path=report_path)
    assert validation["passed"] is False
    assert "stored_report_not_exact_verifier_output" in validation["blockers"]


def test_declared_hash_mismatch_fails_all_c0_receipts(tmp_path: Path) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    manifest["artifacts"][0]["sha256"] = "sha256:" + "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report["receipt"] is False
    assert all(report[key] is False for key in C0_RECEIPT_KEYS)


def test_rehashed_cap_state_and_stress_entropy_splices_are_rejected(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    report = verify_common_domain_source_tower(manifest_path)

    rows = {
        row["control"]: row for row in report["cross_source_splice_controls"]["rows"]
    }
    assert report[SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT] is True
    assert rows["cap_state"]["rehashed_foreign_lookalike"] is True
    assert rows["cap_state"]["splice_rejected"] is True
    assert rows["stress_entropy"]["rehashed_foreign_lookalike"] is True
    assert rows["stress_entropy"]["splice_rejected"] is True


def test_foreign_lookalike_arrow_cannot_replace_main_arrow(tmp_path: Path) -> None:
    manifest_path, fixture = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    manifest["realization"]["channels"]["physical_quotient"][
        "arrow_artifact_id"
    ] = fixture["_test_ids"]["foreign_arrow"]
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report[ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT] is False
    assert any(
        "channel_objects_not_main_source_derived" in blocker
        for blocker in report["blockers"]
    )


def test_unproduced_foreign_anchor_is_not_independent_negative_control(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    manifest["provenance_processes"] = [
        row
        for row in manifest["provenance_processes"]
        if row["output_artifact_id"] != "foreign_source"
    ]
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report[SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT] is False
    assert report[ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT] is False
    assert any(
        "not_independently_reconstructed_from_source_primitive" in blocker
        for blocker in report["blockers"]
    )


def test_forbidden_target_path_into_state_is_detected(tmp_path: Path) -> None:
    manifest_path, fixture = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    state_id = fixture["_test_ids"]["main_state"]
    target_id = fixture["_test_ids"]["target_scale"]
    state_process = next(
        row for row in manifest["provenance_processes"] if row["output_artifact_id"] == state_id
    )
    state_process["data_input_artifact_ids"] = [target_id]
    config_id = state_process["configuration_artifact_id"]
    config_row = next(row for row in manifest["artifacts"] if row["artifact_id"] == config_id)
    config_path = tmp_path / config_row["path"]
    config_path.write_text(
        json.dumps({"input_artifact_id": target_id, "key": "unused"}) + "\n",
        encoding="utf-8",
    )
    config_row["sha256"] = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    # npz_extract cannot read the target NPY, so this also demonstrates that
    # rehashing the splice does not make its evaluator provenance valid.
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report[SOURCE_TOWER_NO_TARGET_PATH_RECEIPT] is False
    assert report["target_free_source_path"]["target_to_protected_paths"]


def test_refinement_square_rejects_dimensionally_plausible_wrong_arrow(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    manifest["refinement_squares"][0]["readout_coarse_map_artifact_id"] = (
        manifest["refinement_squares"][0]["fine_readout_operator_artifact_id"]
    )
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report[SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT] is False
    assert report["receipt"] is False


def test_manifest_parser_rejects_extra_caller_receipt_fields(tmp_path: Path) -> None:
    manifest_path, _ = _write_valid_fixture(tmp_path)
    manifest = _load_manifest_without_test_ids(manifest_path)
    manifest["COMMON_DOMAIN_SOURCE_TOWER_RECEIPT"] = True
    _rewrite_manifest(manifest_path, manifest)

    report = verify_common_domain_source_tower(manifest_path)

    assert report["receipt"] is False
    assert "manifest_fields_invalid" in report["blockers"][0]


def test_emergence_ladder_c0_requires_replayed_verifier_artifact(
    tmp_path: Path,
) -> None:
    from oph_fpe.emergence_ladder import STAGE_SPECS, audit_emergence_ladder

    observer_keys = {
        requirement.receipt_keys[0]
        for stage in STAGE_SPECS
        if stage.stage_id in {"A0", "A1", "A2", "A3", "A4"}
        for requirement in stage.requirements
    }
    forged_dir = tmp_path / "forged"
    forged_dir.mkdir()
    (forged_dir / "caller_booleans.json").write_text(
        json.dumps(
            {"receipts": {**{key: True for key in observer_keys}, **{key: True for key in C0_RECEIPT_KEYS}}}
        ),
        encoding="utf-8",
    )

    forged = audit_emergence_ladder(forged_dir)

    assert forged["dag"]["stages"]["A4"]["passed"] is True
    assert forged["dag"]["stages"]["C0"]["passed"] is False

    valid_dir = tmp_path / "valid"
    manifest_path, _ = _write_valid_fixture(valid_dir)
    report_path = valid_dir / "source_tower_verification.json"
    write_common_domain_source_tower_report(manifest_path, report_path)
    (valid_dir / "observer_receipts.json").write_text(
        json.dumps({"receipts": {key: True for key in observer_keys}}),
        encoding="utf-8",
    )

    verified = audit_emergence_ladder(valid_dir)

    c0 = verified["dag"]["stages"]["C0"]
    assert c0["passed"] is True
    assert c0["source_report_paths"] == ["source_tower_verification.json"]
    assert verified["source_inventory"]["common_source_tower_reports"] == [
        {
            "report_path": "source_tower_verification.json",
            "passed": True,
            "blockers": [],
        }
    ]
