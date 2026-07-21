"""Physical producer for the issue-572 common-domain source tower.

`common_source_tower.py` is the fail-closed verifier for the Einstein-branch
typed tower: thirteen semantic roles, restricted evaluator algebra, hash-bound
provenance, refinement squares, splice controls, and the echosahedral
realization channels.  Until now nothing produced a bundle from the physical
source: the four Einstein-facing readouts (``null_charges``, ``stress``,
``entropy``, ``scale``) existed only as role slots.

This module derives all thirteen role readouts deterministically from one
frozen :func:`capture_physical_source` run and emits a manifest in the exact
shape the verifier demands.  Every bound artifact is an ``npz_extract`` of a
single authoritative source archive, so the verifier's independent
reconstruction re-derives each readout from the same source object.  A second
capture at a declared foreign seed supplies the cross-source splice and
look-alike negative controls.

Claim boundary.  The produced bundle targets the array-level receipts:
provenance, declared target-path firewall, refinement commutation, splice
rejection, and the array-channel realization diagnostic.  The verifier keeps
``SOURCE_TOWER_NO_TARGET_PATH_RECEIPT`` and the physical federation
realization receipt pinned ``False`` pending the generator-code firewall and
the replayed-federation binding; this producer does not and must not unpin
them.  :func:`verify_physical_source_binding` adds the missing source-side
binding as a separate composite layer: it replays the capture from the frozen
configuration, recomputes every role readout, and compares the authoritative
archive byte-for-byte, failing closed on any drift.  No Einstein conclusion,
coupling, vacuum, signature, or scale target enters the source data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import (
    capture_physical_source,
    verify_physical_source_capture,
)
from oph_fpe.common_source_tower import (
    MANIFEST_SCHEMA,
    REQUIRED_ROLES,
    REFINEMENT_ROLES,
    REALIZATION_CHANNELS,
    verify_common_domain_source_tower,
)
from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations

PRODUCER_SCHEMA = "oph.einstein-tower-producer.v1"
BINDING_SCHEMA = "oph.einstein-tower-physical-binding.v1"
DEFAULT_MAIN_CONFIG: Mapping[str, int] = {
    "carrier_count": 64,
    "cycles": 8,
    "seed": 20260751,
}
DEFAULT_FOREIGN_CONFIG: Mapping[str, int] = {
    "carrier_count": 64,
    "cycles": 8,
    "seed": 20260861,
}
PORTS = 12
COARSE_FACTOR = 4

# The producer never emits a physical-promotion claim.
PHYSICAL_PROMOTION_ALLOWED = False


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _carrier_index(carrier_id: str) -> int:
    return int(carrier_id.rsplit("-", 1)[1])


def _trajectory_matrices(capture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = capture["postrun_capture"]["carrier_port_trajectories"]
    count = len(rows)
    initial = np.zeros((count, PORTS))
    settled = np.zeros((count, PORTS))
    repaired = np.zeros((count, PORTS))
    phase_initial = np.zeros(count)
    phase_settled = np.zeros(count)
    for row in rows:
        index = _carrier_index(row["carrier_id"])
        initial[index] = np.asarray(row["initial_port_intensities"], dtype=float)
        settled[index] = np.asarray(row["settled_port_intensities"], dtype=float)
        repaired[index] = np.asarray(row["repaired_port_intensities"], dtype=float)
        phase_initial[index] = float(row["initial_intrinsic_phase"])
        phase_settled[index] = float(row["settled_intrinsic_phase"])
    return {
        "initial": initial,
        "settled": settled,
        "repaired": repaired,
        "phase_initial": phase_initial,
        "phase_settled": phase_settled,
    }


def _per_carrier_port_repair_deltas(
    capture: Mapping[str, Any], count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed repair deltas accumulated per carrier-port slot.

    Returns the total delta matrix and the two cycle-parity halves used by
    the repair-schedule control (addition of the halves commutes exactly).
    """

    total = np.zeros((count, PORTS))
    even = np.zeros((count, PORTS))
    odd = np.zeros((count, PORTS))
    for row in capture["source_artifacts"]["dynamics"]["repair_event_log"]:
        bucket = even if int(row["cycle"]) % 2 == 0 else odd
        reads = {
            (entry["carrier_id"], int(entry["port"])): float(entry["value"])
            for entry in row["read_set"]
        }
        for entry in row["write_set"]:
            key = (entry["carrier_id"], int(entry["port"]))
            delta = float(entry["value"]) - reads[key]
            index = _carrier_index(entry["carrier_id"])
            total[index, int(entry["port"])] += delta
            bucket[index, int(entry["port"])] += delta
    return total, even, odd


def _per_carrier_event_counts(capture: Mapping[str, Any], count: int) -> np.ndarray:
    counts = np.zeros(count)
    for event in capture["postrun_capture"]["semantic_events"]:
        touched: set[int] = set()
        for resource in event["visible_footprint"]:
            carrier = resource.split(":", 1)[0]
            if carrier.startswith("carrier-"):
                touched.add(_carrier_index(carrier))
        for index in touched:
            counts[index] += 1.0
    return counts


def _shannon_entropy_rows(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1)
    out = np.zeros(matrix.shape[0])
    for index in range(matrix.shape[0]):
        if sums[index] <= 0.0:
            continue
        p = matrix[index] / sums[index]
        mask = p > 0.0
        out[index] = float(-(p[mask] * np.log(p[mask])).sum())
    return out


def derive_role_readouts(capture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """All thirteen role readouts, deterministic in the capture."""

    matrices = _trajectory_matrices(capture)
    count = matrices["settled"].shape[0]
    total_delta, even_delta, odd_delta = _per_carrier_port_repair_deltas(
        capture, count
    )
    event_counts = _per_carrier_event_counts(capture, count)
    settled = matrices["settled"]
    repaired = matrices["repaired"]
    initial = matrices["initial"]

    median = float(np.median(settled))
    snapshots = capture["source_artifacts"]["dynamics"]["record_state_snapshots"]
    snapshot_digest = np.asarray(
        [
            float(
                int(
                    hashlib.sha256(
                        _canonical_json(snapshot).encode("utf-8")
                    ).hexdigest()[:12],
                    16,
                )
            )
            for snapshot in snapshots
        ]
    )
    repair_rows = capture["source_artifacts"]["dynamics"]["repair_event_log"]
    repair_log = np.asarray(
        [
            [
                float(row["cycle"]),
                float(row["transaction_index"]),
                float(row["mismatch_before"]),
                float(row["mismatch_after"]),
            ]
            for row in repair_rows
        ]
    )

    fine: dict[str, np.ndarray] = {
        # Per-carrier count of ports above the global settled median: a
        # finite occupancy readout of the bound cap algebras.
        "cap_algebras": (settled > median).sum(axis=1).astype(float),
        # Per-carrier mean settled port intensity: the coarse state readout.
        "state": settled.mean(axis=1),
        # Per-carrier intrinsic-phase transport accumulated by the dynamics.
        "modular_data": np.abs(
            matrices["phase_settled"] - matrices["phase_initial"]
        ),
        # Per-carrier semantic-event incidence.
        "semantic_event_graph": event_counts,
        # Per-carrier net signed repair charge moved across seams.
        "null_charges": total_delta.sum(axis=1),
        # Per-carrier residual deviation between repaired and settled ports.
        "stress": np.abs(repaired - settled).mean(axis=1),
        # Per-carrier Shannon entropy of the settled port distribution.
        "entropy": _shannon_entropy_rows(settled),
        # Per-carrier activity scale of the dynamics.
        "scale": np.linalg.norm(settled - initial, axis=1),
    }
    quotient_normal_form = np.sort(settled.mean(axis=1))
    return {
        **fine,
        "quotient_normal_form": quotient_normal_form,
        "protected_boundary": snapshot_digest,
        "repair_log": repair_log,
        "initial": initial,
        "settled": settled,
        "repaired": repaired,
        "total_delta": total_delta,
        "even_delta": even_delta,
        "odd_delta": odd_delta,
        "event_counts": event_counts,
    }


def _block_mean_matrix(count: int, factor: int) -> np.ndarray:
    coarse_count = count // factor
    matrix = np.zeros((coarse_count, count))
    for index in range(coarse_count):
        matrix[index, index * factor : (index + 1) * factor] = 1.0 / factor
    return matrix


class _BundleBuilder:
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
        self.artifacts.append(
            {
                "artifact_id": artifact_id,
                "path": path.relative_to(self.root).as_posix(),
                "format": format_name,
                "artifact_class": artifact_class,
                "semantic_role": semantic_role,
                "sha256": _sha256_file(path),
            }
        )
        return artifact_id

    def json(
        self, artifact_id: str, value: Any, artifact_class: str, semantic_role: str
    ) -> str:
        path = self.root / f"{artifact_id}.json"
        path.write_text(_canonical_json(value), encoding="utf-8")
        return self._record(artifact_id, path, "json", artifact_class, semantic_role)

    def npy(
        self, artifact_id: str, value: Any, artifact_class: str, semantic_role: str
    ) -> str:
        path = self.root / f"{artifact_id}.npy"
        np.save(path, np.asarray(value), allow_pickle=False)
        return self._record(artifact_id, path, "npy", artifact_class, semantic_role)

    def npz(
        self,
        artifact_id: str,
        value: dict[str, np.ndarray],
        artifact_class: str,
        semantic_role: str,
    ) -> str:
        path = self.root / f"{artifact_id}.npz"
        np.savez(path, **value)
        return self._record(artifact_id, path, "npz", artifact_class, semantic_role)

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


def _tower_arrays(capture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    readouts = derive_role_readouts(capture)
    count = readouts["settled"].shape[0]
    if count % COARSE_FACTOR != 0:
        raise ValueError("carrier count must be divisible by the coarse factor")
    block = _block_mean_matrix(count, COARSE_FACTOR)
    coarse_count = count // COARSE_FACTOR
    role_order = list(REFINEMENT_ROLES)
    stack = np.concatenate([readouts[role] for role in role_order])
    blocks = len(role_order)
    coarse_full = np.zeros((blocks * coarse_count, blocks * count))
    for index in range(blocks):
        coarse_full[
            index * coarse_count : (index + 1) * coarse_count,
            index * count : (index + 1) * count,
        ] = block
    stack_coarse = coarse_full @ stack

    arrays: dict[str, np.ndarray] = {
        "refine_fine_source": stack,
        "refine_coarse_source": stack_coarse,
        "physical_coarse_maps": coarse_full,
        "readout_down_map": block,
        "port_weights": np.ones(PORTS),
        "quotient_normal_form": readouts["quotient_normal_form"],
        "protected_boundary": readouts["protected_boundary"],
        "repair_log": readouts["repair_log"],
    }
    for index, role in enumerate(role_order):
        selector = np.zeros((count, blocks * count))
        selector[:, index * count : (index + 1) * count] = np.eye(count)
        selector_coarse = np.zeros((coarse_count, blocks * coarse_count))
        selector_coarse[
            :, index * coarse_count : (index + 1) * coarse_count
        ] = np.eye(coarse_count)
        arrays[role] = readouts[role]
        arrays[f"coarse_{role}"] = selector_coarse @ stack_coarse
        arrays[f"fine_operator_{role}"] = selector
        arrays[f"coarse_operator_{role}"] = selector_coarse

    # Realization channels: N x 12 concrete data from the capture, abstracted
    # by the exact port-weighted sum.  Channel values are quantized to a fixed
    # integer grid (declared step 2^-20) so permuted and re-ordered exact sums
    # are bitwise stable; integers below 2^53 add exactly in float64 in any
    # order, which keeps the gauge and schedule controls about relabeling
    # rather than floating-point reduction order.
    def _grid(values: np.ndarray) -> np.ndarray:
        return np.rint(np.asarray(values, dtype=float) * (2.0 ** 20))

    template_dims = np.ones((count, PORTS))
    channel_concrete = {
        "accessible_algebras": template_dims * 4.0,
        "port_restrictions": _grid(readouts["initial"]),
        "records": _grid(readouts["repaired"]),
        "repairs": _grid(np.abs(readouts["total_delta"])),
        "checkpoints": template_dims,
        "semantic_event_history": np.repeat(
            readouts["event_counts"][:, None], PORTS, axis=1
        ),
        "physical_quotient": _grid(readouts["settled"]),
    }
    weights = arrays["port_weights"]
    for channel in REALIZATION_CHANNELS:
        arrays[f"concrete_{channel}"] = channel_concrete[channel]
        arrays[f"abstract_{channel}"] = channel_concrete[channel] @ weights

    permutations = icosahedral_a5_port_permutations()
    generator_a = None
    generator_b = None
    identity = tuple(range(PORTS))
    for left in permutations:
        if left == identity:
            continue
        for right in permutations:
            if right in (identity, left):
                continue
            generated = {identity}
            frontier = [identity]
            while frontier:
                base = frontier.pop()
                for gen in (left, right):
                    composed = tuple(gen[base[k]] for k in range(PORTS))
                    if composed not in generated:
                        generated.add(composed)
                        frontier.append(composed)
            if len(generated) == 60:
                generator_a = np.asarray(left)
                generator_b = np.asarray(right)
                break
        if generator_a is not None:
            break
    if generator_a is None or generator_b is None:
        raise RuntimeError("no two-generator A5 pair found")
    arrays["gauge_generator_a"] = generator_a
    arrays["gauge_generator_b"] = generator_b

    dimension = count * PORTS
    arrays["repair_initial"] = _grid(readouts["initial"])
    arrays["repair_linear"] = np.stack((np.eye(dimension), np.eye(dimension)))
    arrays["repair_bias"] = np.stack(
        (
            _grid(readouts["even_delta"]).reshape(-1),
            _grid(readouts["odd_delta"]).reshape(-1),
        )
    )
    arrays["repair_reference"] = (
        arrays["repair_initial"].reshape(-1)
        + arrays["repair_bias"][0]
        + arrays["repair_bias"][1]
    ).reshape(count, PORTS) @ weights
    return arrays


def _foreign_arrays(
    capture: Mapping[str, Any], main_arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Foreign-source negative controls.

    The splice gate requires value lookalikes: arrays numerically identical
    to the main readouts whose only difference is foreign provenance.  The
    foreign capture anchors the lineage; its seed is recorded beside the
    copied values so the archive is distinct at the byte level.
    """

    return {
        "state": np.asarray(main_arrays["state"]).copy(),
        "entropy": np.asarray(main_arrays["entropy"]).copy(),
        "port_weights": np.ones(PORTS),
        "foreign_seed_marker": np.asarray(
            [int(capture["config"]["seed"])], dtype=np.int64
        ),
    }


def produce_common_source_tower_bundle(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    foreign_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a physical issue-572 bundle and verify it fail-closed."""

    root = Path(output_dir)
    main_config = dict(DEFAULT_MAIN_CONFIG if config is None else config)
    other_config = dict(
        DEFAULT_FOREIGN_CONFIG if foreign_config is None else foreign_config
    )
    if main_config.get("seed") == other_config.get("seed"):
        raise ValueError("foreign capture must use a distinct seed")

    main_capture = capture_physical_source(main_config)
    foreign_capture = capture_physical_source(other_config)
    verify_physical_source_capture(main_capture)
    verify_physical_source_capture(foreign_capture)

    builder = _BundleBuilder(root)
    main_arrays = _tower_arrays(main_capture)
    foreign = _foreign_arrays(foreign_capture, main_arrays)

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
    seed_value = int(main_config["seed"])
    seed = builder.json(
        "bound_seed", {"seed": seed_value}, "seed", "frozen_seed"
    )

    main_primitive = builder.npz(
        "main_primitive", main_arrays, "source_primitive", "source_primitive"
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
        "foreign_primitive", foreign, "source_primitive", "foreign_source_primitive"
    )
    foreign_source = builder.npz(
        "foreign_source",
        foreign,
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
        config_artifact = builder.json(
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
            config_artifact,
            seed,
        )
        return output

    role_bindings: dict[str, str] = {"authoritative_presentation": main_source}
    for role in REQUIRED_ROLES:
        if role == "authoritative_presentation":
            continue
        artifact_class = (
            "typed_arrow" if role == "physical_coarse_maps" else "readout"
        )
        role_bindings[role] = extract(
            f"role_{role}",
            main_source,
            role,
            main_arrays[role],
            artifact_class,
            role,
        )

    extra: dict[str, str] = {}
    typed_arrow_keys = {
        "readout_down_map",
        "port_weights",
        "gauge_generator_a",
        "gauge_generator_b",
        "repair_linear",
    } | {f"fine_operator_{role}" for role in REFINEMENT_ROLES} | {
        f"coarse_operator_{role}" for role in REFINEMENT_ROLES
    }
    extra_keys = [
        "refine_fine_source",
        "refine_coarse_source",
        "readout_down_map",
        "port_weights",
        "gauge_generator_a",
        "gauge_generator_b",
        "repair_initial",
        "repair_linear",
        "repair_bias",
        "repair_reference",
        *(f"fine_operator_{role}" for role in REFINEMENT_ROLES),
        *(f"coarse_operator_{role}" for role in REFINEMENT_ROLES),
        *(f"coarse_{role}" for role in REFINEMENT_ROLES),
        *(f"concrete_{channel}" for channel in REALIZATION_CHANNELS),
        *(f"abstract_{channel}" for channel in REALIZATION_CHANNELS),
    ]
    for key in extra_keys:
        artifact_class = "typed_arrow" if key in typed_arrow_keys else "readout"
        extra[key] = extract(key, main_source, key, main_arrays[key], artifact_class, key)

    foreign_state = extract(
        "foreign_state", foreign_source, "state", foreign["state"], "negative_control", "state"
    )
    foreign_entropy = extract(
        "foreign_entropy",
        foreign_source,
        "entropy",
        foreign["entropy"],
        "negative_control",
        "entropy",
    )
    foreign_arrow = extract(
        "foreign_port_weights",
        foreign_source,
        "port_weights",
        foreign["port_weights"],
        "negative_control",
        "foreign_port_weights",
    )
    exact_envelope = builder.json(
        "exact_envelope", {"mode": "exact"}, "configuration", "error_envelope"
    )

    refinement_squares = [
        {
            "square_id": f"refinement_{role}",
            "readout_role": role,
            "fine_source_artifact_id": extra["refine_fine_source"],
            "coarse_source_artifact_id": extra["refine_coarse_source"],
            "source_coarse_map_artifact_id": role_bindings["physical_coarse_maps"],
            "fine_readout_artifact_id": role_bindings[role],
            "coarse_readout_artifact_id": extra[f"coarse_{role}"],
            "fine_readout_operator_artifact_id": extra[f"fine_operator_{role}"],
            "coarse_readout_operator_artifact_id": extra[f"coarse_operator_{role}"],
            "readout_coarse_map_artifact_id": extra["readout_down_map"],
            "error_envelope_artifact_id": exact_envelope,
        }
        for role in REFINEMENT_ROLES
    ]
    realization_channels = {
        channel: {
            "concrete_artifact_id": extra[f"concrete_{channel}"],
            "abstract_artifact_id": extra[f"abstract_{channel}"],
            "arrow_artifact_id": extra["port_weights"],
            "arrow_kind": "port_weighted_sum",
            "error_envelope_artifact_id": exact_envelope,
        }
        for channel in REALIZATION_CHANNELS
    }
    count = int(main_arrays["state"].shape[0])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle_id": f"einstein-tower-physical-{seed_value}",
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
            "patch_count": count,
            "ports_per_patch": PORTS,
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
    manifest_path = root / "common_source_tower_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = verify_common_domain_source_tower(manifest_path)
    producer_receipt = {
        "schema": PRODUCER_SCHEMA,
        "issue": 572,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "foreign_config": other_config,
        "main_capture_sha256": main_capture["capture_sha256"],
        "foreign_capture_sha256": foreign_capture["capture_sha256"],
        "manifest_path": manifest_path.name,
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_commitment": report.get("computed_bundle_commitment"),
        "verifier_report_sha256": report.get("verification_report_sha256"),
    }
    receipt_path = root / "einstein_tower_producer_receipt.json"
    receipt_path.write_text(_canonical_json(producer_receipt), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "verifier_report": report,
        "producer_receipt": producer_receipt,
    }


def verify_physical_source_binding(
    manifest_path: str | Path,
    *,
    main_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Composite binding layer: replay the capture and recompute every readout.

    This does not unpin the C0 verifier's physical receipts.  It adds the
    source-side binding those receipts name as missing: the authoritative
    archive must equal, array for array, the deterministic readouts of a fresh
    capture replay under the frozen configuration.
    """

    manifest_path = Path(manifest_path)
    blockers: list[str] = []
    report = verify_common_domain_source_tower(manifest_path)
    receipt_path = manifest_path.parent / "einstein_tower_producer_receipt.json"
    try:
        producer_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        producer_receipt = None
        blockers.append("producer_receipt_missing_or_invalid")

    replay_equal = False
    capture_hash_equal = False
    if producer_receipt is not None:
        config = dict(
            producer_receipt["main_config"]
            if main_config is None
            else main_config
        )
        replayed = capture_physical_source(config)
        capture_hash_equal = bool(
            replayed["capture_sha256"] == producer_receipt["main_capture_sha256"]
        )
        if not capture_hash_equal:
            blockers.append("capture_replay_hash_mismatch")
        expected = _tower_arrays(replayed)
        stored_path = manifest_path.parent / "main_source.npz"
        try:
            with np.load(stored_path) as stored:
                stored_keys = set(stored.files)
                if stored_keys != set(expected):
                    blockers.append("authoritative_archive_key_set_mismatch")
                    replay_equal = False
                else:
                    replay_equal = all(
                        np.array_equal(stored[key], np.asarray(expected[key]))
                        for key in expected
                    )
                    if not replay_equal:
                        blockers.append("authoritative_archive_array_drift")
        except OSError:
            blockers.append("authoritative_archive_unreadable")

    array_receipts_pass = bool(
        report.get("COMMON_DOMAIN_SOURCE_TOWER_RECEIPT", False)
        if "COMMON_DOMAIN_SOURCE_TOWER_RECEIPT" in report
        else report.get("common_domain_source_tower_receipt", False)
    )
    # The verifier exposes receipt keys as module-level constants; read the
    # canonical booleans directly from the report by their known names.
    keys = {
        key: bool(report.get(key, False))
        for key in report
        if key.endswith("_receipt") or key.endswith("_RECEIPT")
    }
    binding = {
        "schema": BINDING_SCHEMA,
        "issue": 572,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "c0_receipts": keys,
        "capture_replay_hash_equal": capture_hash_equal,
        "authoritative_archive_replay_equal": replay_equal,
        "passed": bool(
            not blockers and capture_hash_equal and replay_equal
        ),
        "blockers": sorted(blockers),
        "claim_boundary": (
            "Source-binding replay for the issue-572 physical bundle: the "
            "authoritative archive is recomputed from a fresh deterministic "
            "capture under the frozen configuration and compared exactly. "
            "The C0 verifier's pinned physical receipts stay authoritative "
            "and are not modified by this layer. No Einstein-branch physical "
            "claim is promoted."
        ),
    }
    return binding
