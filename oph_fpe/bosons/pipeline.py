from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from oph_fpe.bosons.brst_blocks import (
    block_structure_receipt,
    decode_matrix_polynomial,
    determinant_polynomial,
)
from oph_fpe.bosons.pole_enclosure import pole_enclosure_receipt
from oph_fpe.bosons.rg_transport import piecewise_affine_rg_receipt
from oph_fpe.bosons.source_clock import source_clock_receipt
from oph_fpe.evidence.hashes import stable_json_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA = REPOSITORY_ROOT / "schemas" / "bosons" / "wzh_source_closure_run.schema.json"
REPORT_SCHEMA = REPOSITORY_ROOT / "schemas" / "bosons" / "wzh_source_closure_receipt.schema.json"


def load_wzh_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    return payload


def build_wzh_campaign_report(
    config: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    paper_inputs = {
        name: _load_artifact_reference(
            value,
            repository_root,
            expected_sha256=config["paper_input_hashes"].get(name),
        )
        for name, value in config["paper_inputs"].items()
    }
    clock = source_clock_receipt(**config["source_clock"])
    rg = piecewise_affine_rg_receipt(**config["rg_transport"])

    blocks: dict[str, Any] = {}
    poles: dict[str, Any] = {}
    for boson, block_config in config["pole_controls"]["blocks"].items():
        structure = block_structure_receipt(
            block_config["coefficients"],
            block_id=boson,
            block_kind=block_config["block_kind"],
            source_kernel_verified=block_config.get("source_kernel_verified", False),
            ward_identity_verified=block_config.get("ward_identity_verified", False),
            slavnov_taylor_verified=block_config.get("slavnov_taylor_verified", False),
            nielsen_identity_verified=block_config.get("nielsen_identity_verified", False),
        )
        blocks[boson] = structure
        reference_raw = block_config.get("reference_coefficients", block_config["coefficients"])
        reference_determinant = determinant_polynomial(decode_matrix_polynomial(reference_raw))
        poles[boson] = pole_enclosure_receipt(
            structure["determinant_coefficients"],
            reference_coefficients=[
                [float(value.real), float(value.imag)] for value in reference_determinant
            ],
            contour_center=block_config["contour"]["center"],
            contour_radius=block_config["contour"]["radius"],
            contour_samples=block_config["contour"].get("samples", 2048),
            physical_sheet_verified=block_config.get("physical_sheet_verified", False),
            nonzero_residue_verified=block_config.get("nonzero_residue_verified", False),
            uncertainty_bound_present=block_config.get("uncertainty_bound_present", False),
            source_block_receipt=structure["brst_block_receipt"],
        )
        coordinate_unit = block_config["coordinate_unit"]
        mass_coordinate = poles[boson]["mass_coordinate"]
        width_coordinate = poles[boson]["width_coordinate"]
        mass_gev = width_gev = None
        physical_readout_attached = False
        if mass_coordinate is not None and width_coordinate is not None:
            if coordinate_unit == "GeV_squared":
                mass_gev = mass_coordinate
                width_gev = width_coordinate
                physical_readout_attached = True
            elif (
                coordinate_unit == "dimensionless_E_star_squared"
                and clock["source_clock_receipt"]
                and clock["E_star_GeV"] is not None
            ):
                mass_gev = mass_coordinate * clock["E_star_GeV"]
                width_gev = width_coordinate * clock["E_star_GeV"]
                physical_readout_attached = True
        poles[boson]["coordinate_unit"] = coordinate_unit
        poles[boson]["physical_readout_attached"] = physical_readout_attached
        poles[boson]["M_GeV"] = mass_gev
        poles[boson]["Gamma_GeV"] = width_gev

    source_root = bool(
        config["source_root"].get("verified", False)
        and _source_root_artifact_verified(paper_inputs.get("source_root_receipt"))
    )
    same_branch = bool(config["source_root"].get("same_branch", False))
    d10_verified = _certificate_verified(paper_inputs.get("d10_certificate"), "d10")
    d11_verified = _certificate_verified(paper_inputs.get("d11_certificate"), "d11")
    clock_packet_verified = _trusted_packet_verified(paper_inputs.get("source_clock_certificate"))
    rg_packet_verified = _trusted_packet_verified(paper_inputs.get("rg_packet"))
    pole_kernel_packet_verified = _trusted_packet_verified(paper_inputs.get("pole_kernel_manifest"))
    provenance_packet_verified = _trusted_packet_verified(paper_inputs.get("provenance_manifest"))
    provenance = config["provenance"]
    pole_packet = (
        all(
            item["physical_pole_receipt"] and item["physical_readout_attached"]
            for item in poles.values()
        )
        and set(poles) == {"W", "Z", "H"}
    )
    required_pins = {
        "source_root_receipt",
        "d10_receipt",
        "d11_receipt",
        "d10_certificate",
        "d11_certificate",
        "source_clock_certificate",
        "rg_packet",
        "pole_kernel_manifest",
        "provenance_manifest",
    }
    inputs_pinned = all(
        paper_inputs.get(name, {}).get("hash_matches_expected") is True for name in required_pins
    )
    gates = {
        "certificate_candidate_claim": config["claim_level"] == "certificate_candidate",
        "paper_inputs_hash_pinned": inputs_pinned,
        "strict_source_root": source_root,
        "same_branch": same_branch,
        "source_clock": bool(clock["source_clock_receipt"] and clock_packet_verified),
        "d10_QT1_QT5": d10_verified,
        "d11_source_character": d11_verified,
        "rg_matching": bool(rg["rg_matching_receipt"] and rg_packet_verified),
        "brst_complex_poles": bool(pole_packet and pole_kernel_packet_verified),
        "no_target_ancestry": bool(
            provenance.get("no_target_ancestry", False) and provenance_packet_verified
        ),
        "prospective_claim_freeze": bool(
            provenance.get("prospective_claim_freeze", False) and provenance_packet_verified
        ),
    }
    promotion_allowed = all(gates.values())
    blockers = [name for name, passed in gates.items() if not passed]

    report = {
        "schema": "oph_wzh_source_closure_receipt_v1",
        "generated_utc": generated_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_id": config["claim_id"],
        "branch_id": config["branch_id"],
        "claim_level": config["claim_level"],
        "observer_like_system": {
            "bounded_local_state": True,
            "typed_ports_or_boundaries": True,
            "readback_records": True,
            "feedback_or_repair_moves": True,
            "public_evidence_bundle": True,
            "statement": (
                "The numerical backend consumes observer-like self-reading carrier receipts; "
                "it does not replace or synthesize the paper-side D10/D11 carriers."
            ),
        },
        "paper_inputs": paper_inputs,
        "source_clock": clock,
        "rg_transport": rg,
        "brst_blocks": blocks,
        "pole_enclosures": poles,
        "promotion_gates": gates,
        "promotion_allowed": promotion_allowed,
        "overall_status": (
            "full_source_only_wzh_pole_packet"
            if promotion_allowed
            else "diagnostic_backend_source_packets_incomplete"
        ),
        "blockers": blockers,
        "claim_boundary": (
            "Finite numerical controls do not promote W/Z/H masses. Promotion requires one "
            "same-branch root, source clock, validated D10 and D11 certificates, frozen RG "
            "packet, BRST-complete pole receipts, uncertainty, and prospective provenance."
        ),
    }
    jsonschema.Draft202012Validator(
        json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    ).validate(report)
    return report


def write_wzh_campaign_bundle(
    config_path: Path,
    out_dir: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    config = load_wzh_config(config_path)
    report = build_wzh_campaign_report(config, repository_root=repository_root)
    out_dir.mkdir(parents=True)
    (out_dir / "config.yml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "wzh_source_closure_receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "oph_wzh_source_closure_bundle_manifest_v1",
        "claim_id": config["claim_id"],
        "branch_id": config["branch_id"],
        "config_sha256": _sha256(config_path),
        "report_sha256": _sha256(out_dir / "wzh_source_closure_receipt.json"),
        "report_semantic_hash": stable_json_hash(report),
        "promotion_allowed": report["promotion_allowed"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _load_artifact_reference(
    value: str | None,
    repository_root: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if value is None:
        return {
            "path": None,
            "present": False,
            "sha256": None,
            "expected_sha256": expected_sha256,
            "hash_matches_expected": False,
            "payload": None,
        }
    path = Path(value)
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "sha256": None,
            "expected_sha256": expected_sha256,
            "hash_matches_expected": False,
            "payload": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
    actual_sha256 = _sha256(path)
    return {
        "path": str(path),
        "present": True,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "hash_matches_expected": expected_sha256 is not None and actual_sha256 == expected_sha256,
        "payload": payload,
    }


def _certificate_verified(reference: dict[str, Any] | None, kind: str) -> bool:
    if (
        not reference
        or not reference.get("present")
        or reference.get("hash_matches_expected") is not True
        or not isinstance(reference.get("payload"), dict)
    ):
        return False
    payload = reference["payload"]
    if kind == "d10":
        return bool(
            payload.get("promotion_allowed") is True
            and payload.get("exact_verifier_receipt", {}).get("all_checks_pass") is True
        )
    return bool(
        payload.get("promotion_allowed") is True
        and payload.get("trusted_checker", {}).get("accepted") is True
    )


def _source_root_artifact_verified(reference: dict[str, Any] | None) -> bool:
    if (
        not reference
        or reference.get("hash_matches_expected") is not True
        or not isinstance(reference.get("payload"), dict)
    ):
        return False
    payload = reference["payload"]
    return bool(
        payload.get("promotion_allowed") is True
        or (
            payload.get("unique") is True
            and payload.get("map_into_interior") is True
            and payload.get("no_target_ancestry") is True
        )
    )


def _trusted_packet_verified(reference: dict[str, Any] | None) -> bool:
    if (
        not reference
        or reference.get("hash_matches_expected") is not True
        or not isinstance(reference.get("payload"), dict)
    ):
        return False
    payload = reference["payload"]
    return bool(
        payload.get("promotion_allowed") is True
        and payload.get("trusted_checker", {}).get("accepted") is True
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
