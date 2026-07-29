"""Issue-633 finite-interface audit: no physical unit mount is emitted.

The declared serialized issue-634 interface contains a positive
dimensionless gap and no physical unit mount.  This certificate checks
that bounded statement in three fail-closed parts:

* Every numeric leaf of the listed frozen receipts is classified and
  every key is scanned for declared unit vocabulary.  This is a schema
  audit; an unlabeled real is not declared dimensionless merely because
  its key lacks a unit suffix.
* The sector producer is run under two values of a named environment
  channel.  Two attained, byte-identical outputs show that this producer
  does not read that channel.  They do not constitute a mathematical
  two-completion theorem.
* The explicitly listed producer modules are scanned for a declared set
  of SI tokens, and the capture configuration allowlist is checked for
  unit keys.  This is a bounded lexical scan, not a transitive dependency
  closure proof.

The positive finite clause is retained: the source Hamiltonian has an
exactly positive dimensionless gap.  The receipt emits
PHYSICAL_UNITS_NOT_EVALUABLE_ON_DECLARED_SERIALIZED_INTERFACE.  It does
not prove that every extension of A1--A3 lacks a physical clock or that
a future matter or measurement producer cannot supply one.  No physical
promotion follows from any output.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from oph_fpe.local_domain.receipt_io import (
    load_manifest_pinned_receipt,
    manifest_pinned_artifact_sha256,
)

SCHEMA = "oph.local-domain-clock-unit-verdict.v1"
ISSUE = 633
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"
MODULE_DIR = Path(__file__).resolve().parent

RECEIPT_FILES = (
    "stage1_receipt.json",
    "stage2_receipt.json",
    "stage3_receipt.json",
    "stage4_receipt.json",
    "source_gap_receipt.json",
    "defect_sector_receipt.json",
    "matter_attachment_receipt.json",
)

UNIT_VOCABULARY = (
    "hertz",
    "_hz",
    "gev",
    "mev_",
    "joule",
    "kelvin",
    "metre",
    "meter_",
    "second_si",
    "planck_constant",
    "boltzmann",
    "caesium",
    "cesium",
    "si_frequency",
    "si_unit",
    "_mass",
    "mass_",
    "_energy",
    "energy_",
    "electronvolt",
)

SOURCE_SCAN_TOKENS = (
    "9192631770",
    "6.62607015",
    "1.380649",
    "299792458",
    "caesium",
    "cesium",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def classify_leaf(key_path: str, value: Any) -> str:
    """Classify one emitted leaf of the domain interface."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean_leaf"
    if isinstance(value, int):
        return "integer_leaf"
    if isinstance(value, float):
        return "float_leaf"
    if isinstance(value, str):
        if value.startswith("sha256:"):
            return "sha256_string"
        return "string_leaf"
    raise TypeError(f"unclassifiable leaf at {key_path}: {type(value)}")


def walk_interface(
    value: Any, key_path: str = ""
) -> tuple[dict[str, int], list[str]]:
    """Walk a payload, count leaf classes, and flag unit vocabulary."""

    census: dict[str, int] = {}
    unit_hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in UNIT_VOCABULARY):
                unit_hits.append(f"{key_path}/{key}")
            child_census, child_hits = walk_interface(
                child, f"{key_path}/{key}"
            )
            for name, count in child_census.items():
                census[name] = census.get(name, 0) + count
            unit_hits.extend(child_hits)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_census, child_hits = walk_interface(
                child, f"{key_path}[{index}]"
            )
            for name, count in child_census.items():
                census[name] = census.get(name, 0) + count
            unit_hits.extend(child_hits)
    else:
        leaf_class = classify_leaf(key_path, value)
        census[leaf_class] = census.get(leaf_class, 0) + 1
    return census, unit_hits


def classify_array_artifact(path: Path) -> dict[str, Any]:
    """Classify the emitted array artifact by name, dtype, and shape.

    The gzipped array bundle is the largest emitted numeric surface of
    the domain; its array names are checked against the unit
    vocabulary and every array is recorded with dtype and shape, so
    the classification covers it rather than skipping it."""

    import gzip
    import io

    import numpy as np

    raw = path.read_bytes()
    bundle = np.load(io.BytesIO(gzip.decompress(raw)))
    arrays = {}
    hits = []
    for name in bundle.files:
        lowered = name.lower()
        if any(token in lowered for token in UNIT_VOCABULARY):
            hits.append(f"{path.name}:{name}")
        arrays[name] = {
            "dtype": str(bundle[name].dtype),
            "shape": list(bundle[name].shape),
        }
    return {
        "artifact": path.name,
        "sha256": _sha256_bytes(raw),
        "arrays": arrays,
        "unit_vocabulary_hits": hits,
        "clean": bool(not hits),
    }


def interface_classification() -> dict[str, Any]:
    """Classify the explicitly declared frozen finite interface.

    Each walked receipt is first pinned against the manifest hash, so
    the classification is of the certified bytes rather than whatever
    is on disk; the array bundle is classified alongside the JSON
    receipts."""

    manifest_path = DATA_DIR / "manifest.json"
    invalid_artifacts: list[str] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest, Mapping):
                raise TypeError("manifest must be an object")
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ):
            manifest = {}
            invalid_artifacts.append(manifest_path.name)
    else:
        manifest = {}
    manifest_key_of = {
        "stage1_receipt.json": "receipt_sha256",
        "stage2_receipt.json": "stage2_receipt_sha256",
        "stage3_receipt.json": "stage3_receipt_sha256",
        "stage4_receipt.json": "stage4_receipt_sha256",
        "source_gap_receipt.json": "source_gap_receipt_sha256",
        "defect_sector_receipt.json": "defect_sector_receipt_sha256",
        "matter_attachment_receipt.json": "matter_attachment_receipt_sha256",
    }
    per_receipt = {}
    artifact_sha256: dict[str, str] = {}
    total_census: dict[str, int] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    pin_failures: list[str] = []
    schema_failures: list[str] = []
    expected_identity = {
        "stage1_receipt.json": ("oph.local-domain-stage1.v1", 634),
        "stage2_receipt.json": ("oph.local-domain-stage2.v1", 634),
        "stage3_receipt.json": ("oph.local-domain-stage3.v1", 634),
        "stage4_receipt.json": ("oph.local-domain-stage4.v1", 634),
        "source_gap_receipt.json": ("oph.source-clock-gap.v1", 633),
        "defect_sector_receipt.json": (
            "oph.local-domain-defect-sector-spectra.v1",
            311,
        ),
        "matter_attachment_receipt.json": (
            "oph.local-domain-matter-attachment.v1",
            569,
        ),
    }
    for name in RECEIPT_FILES:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        raw = path.read_bytes()
        expected = manifest.get(manifest_key_of[name])
        if expected != _sha256_bytes(raw):
            pin_failures.append(name)
            continue
        artifact_sha256[name] = _sha256_bytes(raw)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("receipt must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            invalid_artifacts.append(name)
            continue
        expected_schema, expected_issue = expected_identity[name]
        if (
            payload.get("schema") != expected_schema
            or payload.get("issue") != expected_issue
            or payload.get("physical_promotion_allowed") is not False
        ):
            schema_failures.append(name)
            continue
        census, hits = walk_interface(payload, name)
        per_receipt[name] = {
            "leaf_census": census,
            "unit_vocabulary_hits": hits[:8],
        }
        for key, count in census.items():
            total_census[key] = total_census.get(key, 0) + count
        all_hits.extend(hits)

    array_path = DATA_DIR / "stage1_arrays.npz.gz"
    if array_path.exists():
        array_raw = array_path.read_bytes()
        if manifest.get("arrays_sha256") != _sha256_bytes(array_raw):
            pin_failures.append(array_path.name)
            array_report = None
        else:
            artifact_sha256[array_path.name] = _sha256_bytes(array_raw)
            try:
                array_report = classify_array_artifact(array_path)
            except (OSError, ValueError, EOFError):
                invalid_artifacts.append(array_path.name)
                array_report = None
            if array_report is not None:
                all_hits.extend(array_report["unit_vocabulary_hits"])
    else:
        array_report = None
        missing.append(array_path.name)

    return {
        "receipts_walked": sorted(per_receipt),
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "missing_receipts": missing,
        "manifest_pin_failures": pin_failures,
        "schema_failures": schema_failures,
        "invalid_artifacts": sorted(set(invalid_artifacts)),
        "total_leaf_census": dict(sorted(total_census.items())),
        "array_artifact": array_report,
        "unit_vocabulary_hits": all_hits[:16],
        "no_unit_labeled_field": bool(
            not all_hits
            and not missing
            and not pin_failures
            and not invalid_artifacts
            and not schema_failures
        ),
        "scope": (
            "the explicitly listed serialized receipts and stage-1 array "
            "bundle; absence of a unit-labeled key does not type an "
            "arbitrary unlabeled value"
        ),
    }


SI_ATTACHMENT_CHANNEL = "OPH_SI_UNIT_ATTACHMENT"


def _call_with_temporary_attachment(
    producer: Callable[[], Any], attachment: str
) -> Any:
    """Run one producer while preserving the caller's environment."""

    existed = SI_ATTACHMENT_CHANNEL in os.environ
    previous = os.environ.get(SI_ATTACHMENT_CHANNEL)
    os.environ[SI_ATTACHMENT_CHANNEL] = attachment
    try:
        return producer()
    finally:
        if existed:
            assert previous is not None
            os.environ[SI_ATTACHMENT_CHANNEL] = previous
        else:
            os.environ.pop(SI_ATTACHMENT_CHANNEL, None)


def producer_channel_nonuse_experiment() -> dict[str, Any]:
    """Two producer runs under distinct named SI channel values.

    The attachment is placed in the process environment, a channel any
    unit-mounting producer could read, and the finite sector producer
    is executed once under each attachment.  The two emitted payloads
    are canonicalized and compared byte-level.  Equality of two attained
    runs shows only that this producer does not read this named channel."""

    from oph_fpe.local_domain.defect_sector_spectra import (
        produce_defect_sector_receipt,
    )

    outputs = {}
    verdicts = {}
    for label, attachment in (("one", "9192631770"), ("two", "2")):
        payload = _call_with_temporary_attachment(
            produce_defect_sector_receipt, attachment
        )
        outputs[label] = _sha256_bytes(
            _canonical_json(payload).encode("utf-8")
        )
        verdicts[label] = payload.get("verdict")
    producer_attained = all(
        verdict == "ATTAINED" for verdict in verdicts.values()
    )
    identical = bool(outputs["one"] == outputs["two"] and producer_attained)
    return {
        "completion_count": 2,
        "attachment_channel": SI_ATTACHMENT_CHANNEL,
        "attachments": {"one": "9192631770", "two": "2"},
        "producer_rerun": "produce_defect_sector_receipt",
        "payload_hashes": outputs,
        "producer_verdicts": verdicts,
        "both_producer_runs_attained": producer_attained,
        "domain_restrictions_identical": identical,
        "witness_holds": identical,
        "form": (
            "bounded implementation check: two attained producer runs "
            "under distinct values of the named SI channel emit identical "
            "bytes"
        ),
    }


def unit_reading_probe() -> dict[str, Any]:
    """A producer that reads the attachment channel, for the control."""

    return {"probe_payload": os.environ.get(SI_ATTACHMENT_CHANNEL, "")}


DECLARED_SUPPORTING_MODULES = (
    "bulk/physical_h3_kms_source_capture.py",
    "bulk/event_manifold_producer.py",
    "core/charged_response.py",
    "core/pole_residue_readback.py",
    "core/spin_statistics_response.py",
)


def source_input_closure() -> dict[str, Any]:
    """Scan a declared source list for SI vocabulary and constants.

    The scan covers the local-domain producers and explicitly listed
    supporting modules.  It is not a computed transitive import closure.
    The capture configuration allowlist is checked against the unit
    vocabulary explicitly."""

    package_root = MODULE_DIR.parent
    scanned = {}
    hits: list[str] = []
    own_name = Path(__file__).name
    paths = [
        path
        for path in sorted(MODULE_DIR.glob("*.py"))
        if path.name != own_name
    ] + [package_root / rel for rel in DECLARED_SUPPORTING_MODULES]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        file_hits = [
            token for token in SOURCE_SCAN_TOKENS if token in text
        ]
        key = str(path.relative_to(package_root).as_posix())
        scanned[key] = {
            "sha256": _sha256_bytes(path.read_bytes()),
            "hits": file_hits,
        }
        hits.extend(f"{key}:{token}" for token in file_hits)

    from oph_fpe.bulk.physical_h3_kms_source_capture import (
        _ALLOWED_CONFIG_KEYS,
    )
    from oph_fpe.local_domain.stage1_event_complex import MAIN_CONFIG

    allowlist_hits = [
        key
        for key in sorted(_ALLOWED_CONFIG_KEYS)
        if any(token in key.lower() for token in UNIT_VOCABULARY)
    ]
    config_within_allowlist = bool(
        set(MAIN_CONFIG).issubset(set(_ALLOWED_CONFIG_KEYS))
    )
    return {
        "self_exclusion_note": (
            "this certificate carries the scan vocabulary and is excluded "
            "from its own scan; its integrity is carried by the manifest "
            "hash"
        ),
        "modules_scanned": sorted(scanned),
        "module_pins": {
            name: row["sha256"] for name, row in scanned.items()
        },
        "si_token_hits": hits[:8],
        "capture_allowlist_unit_hits": allowlist_hits,
        "main_config_within_allowlist": config_within_allowlist,
        "bounded_scan_clean": bool(
            not hits and not allowlist_hits and config_within_allowlist
        ),
        "scope": (
            "declared module and token lists; not a proof over the "
            "transitive Python dependency graph"
        ),
    }


def _load_pinned_gap_receipt() -> dict[str, Any] | None:
    """Load the finite gap receipt only when its manifest pin matches."""

    return load_manifest_pinned_receipt(
        DATA_DIR,
        "source_gap_receipt.json",
        "source_gap_receipt_sha256",
    )


def positive_gap_binding(
    gap_receipt: Mapping[str, Any] | None,
    bundle_resolution: Mapping[str, Any],
) -> dict[str, bool]:
    """Bind the retained positive gap to the resolved finite domain."""

    present = gap_receipt is not None
    schema_valid = bool(
        present
        and gap_receipt.get("schema") == "oph.source-clock-gap.v1"
        and gap_receipt.get("issue") == 633
        and gap_receipt.get("physical_promotion_allowed") is False
    )
    exact_gap = gap_receipt.get("exact_gap", {}) if present else {}
    attained = bool(
        schema_valid
        and gap_receipt.get("verdict") == "ATTAINED"
        and isinstance(exact_gap, Mapping)
        and exact_gap.get("positive")
    )
    same_source = bool(
        present
        and gap_receipt.get("source_projection_sha256")
        == bundle_resolution.get("source_projection_sha256")
    )
    same_domain = bool(
        present
        and gap_receipt.get("domain_freeze_sha256")
        == bundle_resolution.get("domain_freeze_sha256")
    )
    return {
        "receipt_present_and_pinned": present,
        "receipt_schema_valid": schema_valid,
        "receipt_attained_with_positive_gap": attained,
        "same_source_projection": same_source,
        "same_domain_freeze": same_domain,
        "retained": bool(attained and same_source and same_domain),
    }


def source_gap_proof_binding(
    gap_receipt: Mapping[str, Any] | None,
    receipt_sha256: str | None,
) -> dict[str, Any]:
    """Project the exact finite-gap proof into a portable parent binding."""

    kernel = (
        gap_receipt.get("kernel_certificate", {})
        if isinstance(gap_receipt, Mapping)
        else {}
    )
    rows = (
        kernel.get("component_certificates", [])
        if isinstance(kernel, Mapping)
        else []
    )
    rows_valid = isinstance(rows, list)
    negative_rows = (
        [row for row in rows if isinstance(row, Mapping) and not row.get("consistent")]
        if rows_valid
        else []
    )
    witnesses_valid = bool(
        rows_valid
        and all(
            isinstance(row.get("negative_cycle_witness"), Mapping)
            and row["negative_cycle_witness"].get("negative") is True
            and row["negative_cycle_witness"].get("sign_product") == -1
            for row in negative_rows
        )
    )
    frustrated_count = (
        kernel.get("frustrated_component_count")
        if isinstance(kernel, Mapping)
        else None
    )
    rank_theorem = (
        kernel.get("rank_theorem", {})
        if isinstance(kernel, Mapping)
        else {}
    )
    exact_gap = (
        gap_receipt.get("exact_gap", {})
        if isinstance(gap_receipt, Mapping)
        else {}
    )
    proof_complete = bool(
        receipt_sha256 is not None
        and isinstance(exact_gap, Mapping)
        and exact_gap.get("positive") is True
        and isinstance(kernel, Mapping)
        and kernel.get("twisted_kernel_dimension") == 0
        and kernel.get("frustrated_component_witnesses_verified") is True
        and isinstance(rank_theorem, Mapping)
        and rank_theorem.get("applied") is True
        and isinstance(frustrated_count, int)
        and frustrated_count == len(negative_rows)
        and witnesses_valid
    )
    return {
        "source_gap_receipt_sha256": receipt_sha256,
        "source_projection_sha256": (
            gap_receipt.get("source_projection_sha256")
            if isinstance(gap_receipt, Mapping)
            else None
        ),
        "domain_freeze_sha256": (
            gap_receipt.get("domain_freeze_sha256")
            if isinstance(gap_receipt, Mapping)
            else None
        ),
        "exact_gap_positive": bool(
            isinstance(exact_gap, Mapping)
            and exact_gap.get("positive") is True
        ),
        "twisted_kernel_dimension": (
            kernel.get("twisted_kernel_dimension")
            if isinstance(kernel, Mapping)
            else None
        ),
        "rank_theorem_applied": bool(
            isinstance(rank_theorem, Mapping)
            and rank_theorem.get("applied") is True
        ),
        "frustrated_component_count": frustrated_count,
        "negative_cycle_witness_count": len(negative_rows),
        "negative_cycle_witnesses_verified": witnesses_valid,
        "proof_projection_complete": proof_complete,
    }


def produce_clock_unit_verdict(
    *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Produce the issue-633 serialized finite-interface audit."""

    classification = interface_classification()
    witness = producer_channel_nonuse_experiment()
    closure = source_input_closure()

    from oph_fpe.local_domain.stage4_inhabitation import (
        verify_local_domain_bundle,
    )

    bundle = verify_local_domain_bundle()

    gap_receipt = _load_pinned_gap_receipt()
    gap_receipt_sha256 = manifest_pinned_artifact_sha256(
        DATA_DIR,
        "source_gap_receipt.json",
        "source_gap_receipt_sha256",
    )
    gap_binding = positive_gap_binding(gap_receipt, bundle)
    gap_proof = source_gap_proof_binding(
        gap_receipt,
        gap_receipt_sha256,
    )
    gap_retained = bool(
        gap_binding["retained"] and gap_proof["proof_projection_complete"]
    )
    expected_parent_artifacts = set(RECEIPT_FILES) | {
        "stage1_arrays.npz.gz"
    }
    parent_artifacts_pinned = bool(
        set(classification["artifact_sha256"]) == expected_parent_artifacts
        and classification["artifact_sha256"].get(
            "source_gap_receipt.json"
        )
        == gap_receipt_sha256
    )

    controls: dict[str, dict[str, Any]] = {}

    doctored = {"declared_si_frequency_hz": 9192631770}
    _, doctored_hits = walk_interface(doctored, "doctored")
    controls["unit_key_injection"] = {
        "control_failure_detected": bool(doctored_hits),
        "note": "an injected SI-labeled key is flagged by the walker",
    }

    probe_hashes = {}
    for label, attachment in (("one", "9192631770"), ("two", "2")):
        probe_payload = _call_with_temporary_attachment(
            unit_reading_probe, attachment
        )
        probe_hashes[label] = _sha256_bytes(
            _canonical_json(probe_payload).encode("utf-8")
        )
    controls["unit_reading_producer_detected"] = {
        "control_failure_detected": bool(
            probe_hashes["one"] != probe_hashes["two"]
        ),
        "note": (
            "the same two-run comparison applied to a producer that "
            "reads the attachment channel detects the difference, so "
            "the experiment can fail on a unit-mounting producer"
        ),
    }

    poisoned_source = "frequency = 9192631770  # caesium"
    controls["source_token_injection"] = {
        "control_failure_detected": bool(
            any(token in poisoned_source for token in SOURCE_SCAN_TOKENS)
        ),
    }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "audited_parent_artifact_bytes_pinned": parent_artifacts_pinned,
        "declared_serialized_interface_resolved": bool(
            bundle["passed"]
            and not classification["missing_receipts"]
            and not classification["manifest_pin_failures"]
            and not classification["invalid_artifacts"]
            and not classification["schema_failures"]
        ),
        "declared_interface_has_no_unit_labeled_field": classification[
            "no_unit_labeled_field"
        ],
        "declared_si_channel_not_read_by_attained_producer": witness[
            "witness_holds"
        ],
        "bounded_source_scan_clean": closure["bounded_scan_clean"],
        "positive_gap_clause_retained": gap_retained,
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    attained = not blockers

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "source_projection_sha256": bundle.get(
            "source_projection_sha256"
        ),
        "domain_freeze_sha256": bundle.get("domain_freeze_sha256"),
        "declared_domain": (
            "the explicitly listed serialized issue-634 local event, "
            "operator, and finite attachment receipts"
        ),
        "interface_classification": classification,
        "producer_channel_nonuse_experiment": witness,
        "bounded_source_scan": closure,
        "upstream_pins": classification["artifact_sha256"],
        "positive_clause_retained": {
            "source_hamiltonian_with_positive_gap": gap_retained,
            "gap_receipt": "source_gap_receipt.json",
            "binding": gap_binding,
            "proof_binding": gap_proof,
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": (
            "PHYSICAL_UNITS_NOT_EVALUABLE_ON_DECLARED_SERIALIZED_INTERFACE"
            if attained
            else "NOT_ATTAINED"
        ),
        "CLOCK_UNIT_BOUNDED_INTERFACE_AUDIT": bool(attained),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-633 boundary result on the declared serialized "
            "interface: no emitted key carries a physical-unit label, two "
            "attained runs of the sector producer ignore the named SI "
            "attachment channel, and the bounded source scan finds none of "
            "its declared SI tokens. The source "
            "Hamiltonian with its exactly positive dimensionless gap is "
            "retained. These checks justify no GeV claim from this "
            "serialized interface. They are not a transitive source-"
            "closure proof or a no-go theorem for every extended A1--A3 "
            "domain; a future physical transition or measurement producer "
            "is not judged here. No physical promotion follows from any "
            "output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "clock_unit_verdict.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["clock_unit_verdict"] = "clock_unit_verdict.json"
        manifest["clock_unit_verdict_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
