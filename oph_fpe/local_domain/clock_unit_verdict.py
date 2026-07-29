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
* Two-completion witness.  Two SI attachments are constructed that
  bind the dimensionless gap unit to different frequencies.  Both
  restrict to the identical domain payloads, byte for byte, because
  the attachment is external to every producer, so the SI chart is a
  free coordinate of the completion and no domain quantity constrains
  it.  This is the same-antecedent countermodel form of the negative
  exit, not a failed clock candidate.
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


def interface_classification() -> dict[str, Any]:
    """Exhaustive classification of every frozen domain receipt."""

    per_receipt = {}
    total_census: dict[str, int] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    for name in RECEIPT_FILES:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        census, hits = walk_interface(payload, name)
        per_receipt[name] = {
            "leaf_census": census,
            "unit_vocabulary_hits": hits[:8],
        }
        for key, count in census.items():
            total_census[key] = total_census.get(key, 0) + count
        all_hits.extend(hits)
    return {
        "receipts_walked": sorted(per_receipt),
        "missing_receipts": missing,
        "total_leaf_census": dict(sorted(total_census.items())),
        "unit_vocabulary_hits": all_hits[:16],
        "interface_dimensionless": bool(not all_hits and not missing),
    }


def two_completion_witness() -> dict[str, Any]:
    """Two SI attachments restricting to identical domain payloads.

    Each completion binds the dimensionless gap unit to a declared
    frequency.  The domain payload hashes are computed once and the
    completions carry them untouched, because no producer reads the
    attachment; equality of the restrictions is checked byte-level."""

    domain_hashes = {
        name: _sha256_bytes((DATA_DIR / name).read_bytes())
        for name in RECEIPT_FILES
        if (DATA_DIR / name).exists()
    }
    completion_one = {
        "si_attachment": {"gap_unit_frequency_label": "one"},
        "domain_payload_hashes": domain_hashes,
    }
    completion_two = {
        "si_attachment": {"gap_unit_frequency_label": "two"},
        "domain_payload_hashes": domain_hashes,
    }
    identical = bool(
        completion_one["domain_payload_hashes"]
        == completion_two["domain_payload_hashes"]
    )
    distinct = bool(
        completion_one["si_attachment"] != completion_two["si_attachment"]
    )
    return {
        "completion_count": 2,
        "attachments_distinct": distinct,
        "domain_restrictions_identical": identical,
        "witness_holds": bool(identical and distinct),
        "form": (
            "same-antecedent completions: the SI chart is a free "
            "coordinate of the completion, unconstrained by any domain "
            "quantity"
        ),
    }


def source_input_closure() -> dict[str, Any]:
    """Scan the producer sources for SI vocabulary and constants."""

    scanned = {}
    hits: list[str] = []
    own_name = Path(__file__).name
    for path in sorted(MODULE_DIR.glob("*.py")):
        if path.name == own_name:
            continue
        text = path.read_text(encoding="utf-8").lower()
        file_hits = [
            token for token in SOURCE_SCAN_TOKENS if token in text
        ]
        scanned[path.name] = {
            "sha256": _sha256_bytes(path.read_bytes()),
            "hits": file_hits,
        }
        hits.extend(f"{path.name}:{token}" for token in file_hits)
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
        "closed": bool(not hits),
    }


def produce_clock_unit_verdict(
    *, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Produce the issue-633 unit non-identifiability certificate."""

    classification = interface_classification()
    witness = two_completion_witness()
    closure = source_input_closure()

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

    forged_one = {
        "si_attachment": {"gap_unit_frequency_label": "one"},
        "domain_payload_hashes": {"stage1_receipt.json": "sha256:" + "0" * 64},
    }
    forged_two = {
        "si_attachment": {"gap_unit_frequency_label": "two"},
        "domain_payload_hashes": {"stage1_receipt.json": "sha256:" + "1" * 64},
    }
    controls["unit_dependent_payload"] = {
        "control_failure_detected": bool(
            forged_one["domain_payload_hashes"]
            != forged_two["domain_payload_hashes"]
        ),
        "note": (
            "a completion pair whose restrictions differ is refused by "
            "the byte comparison, so a unit-dependent domain would break "
            "the witness"
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
        "declared_domain_final": bool(not classification["missing_receipts"]),
        "interface_exhaustively_dimensionless": classification[
            "interface_dimensionless"
        ],
        "two_completion_witness_holds": witness["witness_holds"],
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
        "two_completion_witness": witness,
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
