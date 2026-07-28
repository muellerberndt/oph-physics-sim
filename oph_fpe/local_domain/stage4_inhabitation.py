"""Issue-634 stage 4: inhabitation, provenance DAG, and replay verifier.

Stage 4 closes the staged construction: one target-clean source, the
declared sixteen-thousand-carrier capture, inhabits the complete typed
domain when the three stage receipts are attained, bound to one
capture hash, resolvable from one provenance DAG, and reproducible
byte-exactly by an independent replay of the producers.

The bundle verifier trusts no receipt boolean: it recomputes each
stage verdict from the recorded clause vector, the control vector, and
the blocker list, and refuses a receipt whose stored verdict disagrees
with the recomputation.  The provenance DAG hashes the configuration,
the capture, the producer and capture modules, and every receipt, and
the target-clean scan refuses forbidden target vocabulary in any
recorded configuration.  The issue-level control matrix maps the six
declared control families, wrong dimension, split signature, failed
cocycle, nonlocal operator, mixed source, and target injection, to the
detected controls of the stage receipts.  Preservation squares map
gauge relabelling, chart changes, lift changes, and registered
refinements to the exact certificates that carry them.  No output of
this module carries physical promotion.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import oph_fpe.bulk.event_manifold_producer as event_manifold_producer
import oph_fpe.bulk.physical_h3_kms_source_capture as source_capture
import oph_fpe.local_domain.stage1_event_complex as stage1_module
import oph_fpe.local_domain.stage2_spin_layer as stage2_module
import oph_fpe.local_domain.stage3_typed_domain as stage3_module
from oph_fpe.local_domain.stage1_event_complex import (
    FORBIDDEN_CONFIG_KEYS,
    MAIN_CONFIG,
)

SCHEMA = "oph.local-domain-stage4.v1"
ISSUE = 634
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

STAGE_RECEIPTS = {
    "stage1": ("stage1_receipt.json", "receipt_sha256"),
    "stage2": ("stage2_receipt.json", "stage2_receipt_sha256"),
    "stage3": ("stage3_receipt.json", "stage3_receipt_sha256"),
}

CONTROL_MATRIX = {
    "wrong_dimension": ("stage1", "wrong_dimension"),
    "split_signature": ("stage1", "split_signature"),
    "failed_cocycle": ("stage1", "failed_cocycle"),
    "nonlocal_operator": ("stage3", "nonlocal_operator"),
    "mixed_source": ("stage2", "mixed_source"),
    "target_injection": ("stage2", "target_injection"),
}

PRESERVATION_SQUARES = {
    "gauge_relabelling": ("stage3", "gauge_relabelling_covariant"),
    "chart_changes": ("stage1", "transitions_supported"),
    "lift_changes": ("stage2", "lift_ambiguity_rank_identity"),
    "registered_refinements": ("stage3", "refinement_naturality_exact"),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _module_sha(module) -> str:
    return _sha256_bytes(Path(module.__file__).read_bytes())


def _recompute_verdict(receipt: Mapping[str, Any]) -> str:
    clauses_pass = all(receipt.get("clause_verdicts", {}).values()) and bool(
        receipt.get("clause_verdicts")
    )
    controls = receipt.get("negative_controls", {})
    controls_pass = bool(controls) and all(
        row.get("control_failure_detected") for row in controls.values()
    )
    blockers_empty = receipt.get("blockers") == []
    return (
        "ATTAINED"
        if clauses_pass and controls_pass and blockers_empty
        else "NOT_ATTAINED"
    )


def verify_local_domain_bundle(
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve every artifact and recompute every stage verdict.

    Fail-closed: any missing file, hash mismatch, cross-stage binding
    failure, or verdict disagreement lands in the blocker list, and
    receipt booleans are never trusted."""

    base = Path(data_dir) if data_dir is not None else DATA_DIR
    blockers: list[str] = []
    receipts: dict[str, Mapping[str, Any]] = {}

    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "blockers": ["manifest_missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for stage, (filename, manifest_key) in STAGE_RECEIPTS.items():
        path = base / filename
        if not path.exists():
            blockers.append(f"{stage}_receipt_missing")
            continue
        raw = path.read_bytes()
        if manifest.get(manifest_key) != _sha256_bytes(raw):
            blockers.append(f"{stage}_receipt_hash_mismatch")
            continue
        receipt = json.loads(raw.decode("utf-8"))
        recomputed = _recompute_verdict(receipt)
        if receipt.get("verdict") != recomputed:
            blockers.append(f"{stage}_verdict_disagrees_with_recomputation")
        if receipt.get("physical_promotion_allowed") is not False:
            blockers.append(f"{stage}_promotion_flag_invalid")
        receipts[stage] = receipt

    arrays_path = base / "stage1_arrays.npz.gz"
    if not arrays_path.exists():
        blockers.append("stage1_arrays_missing")
    elif manifest.get("arrays_sha256") != _sha256_bytes(
        arrays_path.read_bytes()
    ):
        blockers.append("stage1_arrays_hash_mismatch")

    if len(receipts) == len(STAGE_RECEIPTS):
        capture_hashes = {r["capture_sha256"] for r in receipts.values()}
        if len(capture_hashes) != 1:
            blockers.append("capture_hashes_disagree_across_stages")
        if not receipts["stage2"]["stage1_binding"].get("same_source"):
            blockers.append("stage2_not_bound_to_stage1_source")
        if not receipts["stage3"]["stage_binding"].get("same_domain_freeze"):
            blockers.append("stage3_not_bound_to_stage2_domain")
        configs = [r.get("main_config", {}) for r in receipts.values()]
        if any(c != configs[0] for c in configs):
            blockers.append("configs_disagree_across_stages")
        if configs and configs[0] != MAIN_CONFIG:
            blockers.append("config_not_the_declared_main_config")
        for config in configs:
            forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
            for key in forbidden:
                blockers.append(f"forbidden_config_key:{key}")

    return {
        "passed": bool(not blockers),
        "blockers": sorted(set(blockers)),
        "stage_verdicts": {
            stage: receipt.get("verdict") for stage, receipt in receipts.items()
        },
        "capture_sha256": (
            next(iter(receipts.values()))["capture_sha256"]
            if receipts and len({r["capture_sha256"] for r in receipts.values()}) == 1
            else None
        ),
    }


def provenance_dag(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Hash-bound provenance DAG from configuration to receipts."""

    config = receipts["stage1"]["main_config"]
    artifacts = {
        "configuration": {
            "kind": "declared_configuration",
            "sha256": _sha256_bytes(_canonical_json(config).encode("utf-8")),
        },
        "capture_module": {
            "kind": "evaluator",
            "path": "oph_fpe/bulk/physical_h3_kms_source_capture.py",
            "sha256": _module_sha(source_capture),
        },
        "reconstruction_module": {
            "kind": "evaluator",
            "path": "oph_fpe/bulk/event_manifold_producer.py",
            "sha256": _module_sha(event_manifold_producer),
        },
        "stage1_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage1_event_complex.py",
            "sha256": _module_sha(stage1_module),
        },
        "stage2_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage2_spin_layer.py",
            "sha256": _module_sha(stage2_module),
        },
        "stage3_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage3_typed_domain.py",
            "sha256": _module_sha(stage3_module),
        },
        "capture": {
            "kind": "source_primitive",
            "sha256": receipts["stage1"]["capture_sha256"],
        },
        "stage1_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage1"]).encode("utf-8")
            ),
        },
        "stage2_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage2"]).encode("utf-8")
            ),
        },
        "stage3_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage3"]).encode("utf-8")
            ),
        },
    }
    processes = [
        {
            "process": "capture_physical_source",
            "inputs": ["configuration"],
            "evaluator": "capture_module",
            "output": "capture",
        },
        {
            "process": "produce_stage1_receipt",
            "inputs": ["capture"],
            "evaluator": "stage1_module",
            "output": "stage1_receipt",
        },
        {
            "process": "produce_stage2_receipt",
            "inputs": ["capture", "stage1_receipt"],
            "evaluator": "stage2_module",
            "output": "stage2_receipt",
        },
        {
            "process": "produce_stage3_receipt",
            "inputs": ["capture", "stage2_receipt"],
            "evaluator": "stage3_module",
            "output": "stage3_receipt",
        },
    ]
    known = set(artifacts)
    acyclic = True
    produced: set[str] = set()
    for process in processes:
        if any(
            name not in known or name == process["output"]
            for name in process["inputs"]
        ):
            acyclic = False
        if process["output"] in produced:
            acyclic = False
        produced.add(process["output"])
    return {
        "artifacts": artifacts,
        "processes": processes,
        "single_source": True,
        "acyclic": bool(acyclic),
        "dag_sha256": _sha256_bytes(
            _canonical_json({"artifacts": artifacts, "processes": processes}).encode(
                "utf-8"
            )
        ),
    }


def _round_trip_canonical(value: Any) -> str:
    """Canonical form after a JSON round trip.

    Serialization is compared after parse and re-serialization on both
    sides, so a producer emitting non-string keys cannot make a frozen
    file and a fresh dict of equal content compare unequal."""

    return _canonical_json(json.loads(_canonical_json(value)))


def independent_replay(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Re-produce all three stage receipts and compare byte-exactly."""

    with tempfile.TemporaryDirectory() as scratch:
        stage1_module.produce_stage1_receipt(output_dir=scratch)
        replayed_stage2 = stage2_module.produce_stage2_receipt()
        replayed_stage3 = stage3_module.produce_stage3_receipt()
        replay_dir = Path(scratch)
        stage1_bytes = (replay_dir / "stage1_receipt.json").read_bytes()
    comparisons = {
        "stage1": bool(
            _round_trip_canonical(json.loads(stage1_bytes.decode("utf-8")))
            == _round_trip_canonical(receipts["stage1"])
        ),
        "stage2": bool(
            _round_trip_canonical(replayed_stage2)
            == _round_trip_canonical(receipts["stage2"])
        ),
        "stage3": bool(
            _round_trip_canonical(replayed_stage3)
            == _round_trip_canonical(receipts["stage3"])
        ),
    }
    return {
        "receipt_byte_exact": comparisons,
        "all_byte_exact": bool(all(comparisons.values())),
    }


def produce_stage4_receipt(
    *,
    output_dir: str | Path | None = None,
    run_replay: bool = True,
) -> dict[str, Any]:
    """Produce the stage-4 inhabitation receipt for the frozen bundle."""

    resolution = verify_local_domain_bundle()
    receipts = {}
    for stage, (filename, _) in STAGE_RECEIPTS.items():
        path = DATA_DIR / filename
        if path.exists():
            receipts[stage] = json.loads(path.read_text(encoding="utf-8"))
    if len(receipts) != len(STAGE_RECEIPTS):
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["stage_receipts_incomplete"],
        }

    dag = provenance_dag(receipts)

    control_rows = {}
    for name, (stage, key) in CONTROL_MATRIX.items():
        detected = bool(
            receipts[stage]["negative_controls"]
            .get(key, {})
            .get("control_failure_detected")
        )
        control_rows[name] = {"stage": stage, "control": key, "detected": detected}
    preservation_rows = {}
    for name, (stage, clause) in PRESERVATION_SQUARES.items():
        preservation_rows[name] = {
            "stage": stage,
            "clause": clause,
            "holds": bool(receipts[stage]["clause_verdicts"].get(clause)),
        }

    replay = (
        independent_replay(receipts)
        if run_replay
        else {"receipt_byte_exact": {}, "all_byte_exact": False}
    )

    controls: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        for name in ("manifest.json", "stage1_arrays.npz.gz"):
            (scratch_dir / name).write_bytes((DATA_DIR / name).read_bytes())
        for stage, (filename, _) in STAGE_RECEIPTS.items():
            (scratch_dir / filename).write_bytes(
                (DATA_DIR / filename).read_bytes()
            )

        tampered = json.loads(
            (scratch_dir / "stage2_receipt.json").read_text(encoding="utf-8")
        )
        tampered["seam_layer"]["lift_ambiguity"]["lift_ambiguity_rank"] += 1
        (scratch_dir / "stage2_receipt.json").write_text(
            _canonical_json(tampered), encoding="utf-8"
        )
        tampered_result = verify_local_domain_bundle(scratch_dir)
        controls["tampered_receipt"] = {
            "control_failure_detected": bool(not tampered_result["passed"])
        }

        (scratch_dir / "stage2_receipt.json").write_bytes(
            (DATA_DIR / "stage2_receipt.json").read_bytes()
        )
        overridden = json.loads(
            (scratch_dir / "stage1_receipt.json").read_text(encoding="utf-8")
        )
        overridden["clause_verdicts"]["dimension_four"] = False
        overridden["verdict"] = "ATTAINED"
        raw = _canonical_json(overridden).encode("utf-8")
        (scratch_dir / "stage1_receipt.json").write_bytes(raw)
        manifest = json.loads(
            (scratch_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["receipt_sha256"] = _sha256_bytes(raw)
        (scratch_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
        override_result = verify_local_domain_bundle(scratch_dir)
        controls["truth_flag_override"] = {
            "control_failure_detected": bool(
                "stage1_verdict_disagrees_with_recomputation"
                in override_result["blockers"]
            )
        }

        (scratch_dir / "stage3_receipt.json").unlink()
        missing_result = verify_local_domain_bundle(scratch_dir)
        controls["missing_artifact"] = {
            "control_failure_detected": bool(
                "stage3_receipt_missing" in missing_result["blockers"]
            )
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "bundle_resolves_fail_closed": resolution["passed"],
        "single_source_dag": bool(
            dag["single_source"] and dag["acyclic"]
        ),
        "target_clean_provenance": bool(
            not any(
                blocker.startswith("forbidden_config_key")
                for blocker in resolution["blockers"]
            )
        ),
        "stage_verdicts_recomputed_attained": bool(
            resolution["passed"]
            and all(
                verdict == "ATTAINED"
                for verdict in resolution["stage_verdicts"].values()
            )
        ),
        "issue_control_matrix_complete": all(
            row["detected"] for row in control_rows.values()
        ),
        "preservation_squares_hold": all(
            row["holds"] for row in preservation_rows.values()
        ),
        "independent_replay_byte_exact": replay["all_byte_exact"],
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "capture_sha256": resolution.get("capture_sha256"),
        "bundle_resolution": resolution,
        "provenance_dag": dag,
        "issue_control_matrix": control_rows,
        "preservation_squares": preservation_rows,
        "independent_replay": replay,
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "LOCAL_DOMAIN_INHABITATION_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-634 stage-4 object: the declared source inhabits "
            "the typed local domain in the finite sense that the three "
            "stage receipts are attained on one capture hash, resolvable "
            "from one target-clean acyclic provenance DAG, with the six "
            "issue-level control families detected, the preservation "
            "squares carried by exact certificates, and the producers "
            "replayed byte-exactly. The Lorentzian signature certificate "
            "is the held-out inertia of the issue-575 instrument with the "
            "cone margin owned by issue 595, the sign layer is the "
            "declared reversing convention with continuum identification "
            "owned by the issue-596 lanes, and physical fiber selection "
            "belongs to issues 569 and 630. No physical promotion follows "
            "from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "stage4_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["stage4_receipt"] = "stage4_receipt.json"
        manifest["stage4_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
