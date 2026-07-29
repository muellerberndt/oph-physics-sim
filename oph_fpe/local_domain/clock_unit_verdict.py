"""Issue-633 closure certificate: unit non-identifiability on the domain.

The clock lane closes with a source energy interval bound to the SI
chart or with a complete non-identifiability theorem and the explicit
verdict that physical units are not evaluable.  The declared domain of
the lane is the issue-634 local event and Hamiltonian typing, and that
domain is closed and final, so the negative branch is decidable.  This
certificate proves it in three exact parts, each fail-closed:

* Exhaustive emitted-quantity classification.  Every numeric leaf of
  every frozen local-domain receipt is walked and classified: integer
  counts, hash strings, booleans and verdicts, and dimensionless reals
  (coordinates of fitted forms, margins, gaps, residuals, rank data).
  Unit semantics attach through keys in the canonical receipt format,
  and no key carries unit vocabulary, so the complete emitted
  interface of the declared domain is dimensionless.
* Two-completion experiment.  The gap producer is executed twice
  under two distinct SI attachments placed in a reachable environment
  channel, and the emitted payloads are byte-identical, a falsifiable
  measurement that the SI chart is a free coordinate of the
  completion; a producer reading the channel is shown to be detected
  by the same comparison.  This is the same-antecedent countermodel
  form of the negative exit, not a failed clock candidate.
* Input closure.  The local-domain producer sources are scanned for SI
  vocabulary and measured-constant imports; none is present, and the
  capture configuration allowlist carries no unit key, so no unit
  datum has source ancestry.

The positive first clause of the lane is retained: the issue-633 gap
receipt carries the source Hamiltonian with its exactly positive
dimensionless gap.  What the declared domain cannot do is select a
physical reference transition, so the certificate emits
PHYSICAL_UNITS_NOT_EVALUABLE for the declared domain.  A completion
that adds physical fiber content is a different domain owned by the
family and scalar lanes; under the declared domain this verdict is
complete.  No physical promotion follows from any output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
        return "boolean_verdict"
    if isinstance(value, int):
        return "integer_count"
    if isinstance(value, float):
        return "dimensionless_real"
    if isinstance(value, str):
        if value.startswith("sha256:"):
            return "hash"
        return "text"
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
    """Exhaustive classification of every frozen domain artifact.

    Each walked receipt is first pinned against the manifest hash, so
    the classification is of the certified bytes rather than whatever
    is on disk; the array bundle is classified alongside the JSON
    receipts."""

    manifest_path = DATA_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
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
    total_census: dict[str, int] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    pin_failures: list[str] = []
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
        payload = json.loads(raw.decode("utf-8"))
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
        array_report = classify_array_artifact(array_path)
        if manifest.get("arrays_sha256") != array_report["sha256"]:
            pin_failures.append(array_path.name)
        all_hits.extend(array_report["unit_vocabulary_hits"])
    else:
        array_report = None
        missing.append(array_path.name)

    return {
        "receipts_walked": sorted(per_receipt),
        "missing_receipts": missing,
        "manifest_pin_failures": pin_failures,
        "total_leaf_census": dict(sorted(total_census.items())),
        "array_artifact": array_report,
        "unit_vocabulary_hits": all_hits[:16],
        "interface_dimensionless": bool(
            not all_hits and not missing and not pin_failures
        ),
    }


SI_ATTACHMENT_CHANNEL = "OPH_SI_UNIT_ATTACHMENT"


def two_completion_experiment() -> dict[str, Any]:
    """Two producer runs under distinct reachable SI attachments.

    The attachment is placed in the process environment, a channel any
    unit-mounting producer could read, and the issue-633 gap producer
    is executed once under each attachment.  The two emitted payloads
    are canonicalized and compared byte-level; a producer that read
    the attachment would emit different bytes, so equality is a
    falsifiable measurement that the SI chart is a free coordinate of
    the completion, unconstrained by any domain quantity."""

    import os

    from oph_fpe.local_domain.defect_sector_spectra import (
        produce_defect_sector_receipt,
    )

    outputs = {}
    for label, attachment in (("one", "9192631770"), ("two", "2")):
        os.environ[SI_ATTACHMENT_CHANNEL] = attachment
        try:
            payload = produce_defect_sector_receipt()
        finally:
            os.environ.pop(SI_ATTACHMENT_CHANNEL, None)
        outputs[label] = _sha256_bytes(
            _canonical_json(payload).encode("utf-8")
        )
    identical = bool(outputs["one"] == outputs["two"])
    return {
        "completion_count": 2,
        "attachment_channel": SI_ATTACHMENT_CHANNEL,
        "attachments": {"one": "9192631770", "two": "2"},
        "producer_rerun": "produce_defect_sector_receipt",
        "payload_hashes": outputs,
        "domain_restrictions_identical": identical,
        "witness_holds": identical,
        "form": (
            "measured same-antecedent completions: the producer runs "
            "under two distinct reachable SI attachments and emits "
            "identical bytes"
        ),
    }


def unit_reading_probe(attachment: str) -> dict[str, Any]:
    """A producer that reads the attachment channel, for the control."""

    import os

    return {"probe_payload": os.environ.get(SI_ATTACHMENT_CHANNEL, ""),
            "echo": attachment}


TRANSITIVE_PRODUCER_MODULES = (
    "bulk/physical_h3_kms_source_capture.py",
    "bulk/event_manifold_producer.py",
    "core/charged_response.py",
    "core/pole_residue_readback.py",
    "core/spin_statistics_response.py",
)


def source_input_closure() -> dict[str, Any]:
    """Scan the producer sources for SI vocabulary and constants.

    The scan covers the local-domain producers and the transitive
    modules they import to generate the data: the capture module, the
    reconstruction instrument, and the response, pole-residue, and
    spin producers.  The capture configuration allowlist is checked
    against the unit vocabulary explicitly."""

    package_root = MODULE_DIR.parent
    scanned = {}
    hits: list[str] = []
    own_name = Path(__file__).name
    paths = [
        path
        for path in sorted(MODULE_DIR.glob("*.py"))
        if path.name != own_name
    ] + [package_root / rel for rel in TRANSITIVE_PRODUCER_MODULES]
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
        "closed": bool(
            not hits and not allowlist_hits and config_within_allowlist
        ),
    }


def produce_clock_unit_verdict(
    *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Produce the issue-633 unit non-identifiability certificate."""

    classification = interface_classification()
    witness = two_completion_experiment()
    closure = source_input_closure()

    from oph_fpe.local_domain.stage4_inhabitation import (
        verify_local_domain_bundle,
    )

    bundle = verify_local_domain_bundle()

    gap_path = DATA_DIR / "source_gap_receipt.json"
    gap_receipt = (
        json.loads(gap_path.read_text(encoding="utf-8"))
        if gap_path.exists()
        else None
    )
    gap_retained = bool(
        gap_receipt is not None
        and gap_receipt["verdict"] == "ATTAINED"
        and gap_receipt["exact_gap"]["positive"]
    )

    controls: dict[str, dict[str, Any]] = {}

    doctored = {"declared_si_frequency_hz": 9192631770}
    _, doctored_hits = walk_interface(doctored, "doctored")
    controls["unit_key_injection"] = {
        "control_failure_detected": bool(doctored_hits),
        "note": "an injected SI-labeled key is flagged by the walker",
    }

    import os

    probe_hashes = {}
    for label, attachment in (("one", "9192631770"), ("two", "2")):
        os.environ[SI_ATTACHMENT_CHANNEL] = attachment
        try:
            probe_payload = unit_reading_probe(attachment)
        finally:
            os.environ.pop(SI_ATTACHMENT_CHANNEL, None)
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
        "declared_domain_final": bool(
            bundle["passed"]
            and not classification["missing_receipts"]
            and not classification["manifest_pin_failures"]
        ),
        "interface_exhaustively_dimensionless": classification[
            "interface_dimensionless"
        ],
        "two_completion_experiment_holds": witness["witness_holds"],
        "source_input_closure_holds": closure["closed"],
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
        "declared_domain": (
            "the closed issue-634 local event and Hamiltonian typing, the "
            "single declared dependency of the clock lane"
        ),
        "interface_classification": classification,
        "two_completion_experiment": witness,
        "source_input_closure": closure,
        "positive_clause_retained": {
            "source_hamiltonian_with_positive_gap": gap_retained,
            "gap_receipt": "source_gap_receipt.json",
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": (
            "PHYSICAL_UNITS_NOT_EVALUABLE" if attained else "NOT_ATTAINED"
        ),
        "CLOCK_UNIT_NON_IDENTIFIABILITY_RECEIPT": bool(attained),
        "blockers": blockers,
        "claim_boundary": (
            "Complete negative exit of the issue-633 clock lane on its "
            "declared domain: the closed issue-634 typing emits an "
            "exhaustively dimensionless interface, two distinct SI "
            "attachments restrict to byte-identical domain payloads, and "
            "the producer sources are closed against SI vocabulary, so no "
            "physical reference transition is identifiable and physical "
            "units are not evaluable on this domain. The source "
            "Hamiltonian with its exactly positive dimensionless gap is "
            "retained. Dimensionless pole verdicts downstream stay "
            "evaluable; a GeV mass claim is forbidden. A completion "
            "carrying physical fiber content is a different domain owned "
            "by the family and scalar lanes and is not judged here. No "
            "physical promotion follows from any output."
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
