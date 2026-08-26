from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

from oph_fpe.cosmology.public_measurement_source_binding import (
    CASSINI_ROLES,
    DEFAULT_MANIFEST_PATH,
    PASS_STATUS,
    PLANCK_ROLES,
    REPOSITORY_ROOT,
    SPARC_ROLES,
    binding_errors_for_roles,
    validate_public_measurement_sources,
)


PLANCK_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "measurements"
    / "planck2018"
    / "COM_PowerSpect_CMB-TT-binned_R3.01.txt"
)
SPARC_DIR = REPOSITORY_ROOT / "data" / "measurements" / "sparc"
CASSINI_PATH = (
    REPOSITORY_ROOT / "data" / "measurements" / "cassini" / "cassini_q2_2026.json"
)


def _validate(
    *,
    planck: Path = PLANCK_PATH,
    sparc: Path = SPARC_DIR,
    cassini: Path = CASSINI_PATH,
    manifest: Path = DEFAULT_MANIFEST_PATH,
) -> dict:
    return validate_public_measurement_sources(
        planck_tt_path=planck,
        sparc_dir=sparc,
        cassini_summary_path=cassini,
        manifest_path=manifest,
    )


def test_committed_public_measurement_bundle_is_exactly_bound() -> None:
    receipt = _validate()

    assert receipt["status"] == PASS_STATUS
    assert receipt["canonical_source_binding_receipt"] is True
    assert receipt["integrity_receipt"] is True
    assert receipt["integrity_errors"] == []
    assert receipt["contains_external_published_measurements"] is True
    assert receipt["raw_instrument_telemetry_included"] is False
    assert receipt["remote_source_authentication_performed"] is False
    assert set(receipt["files"]) == set((*PLANCK_ROLES, *SPARC_ROLES, *CASSINI_ROLES))
    assert all(row["integrity_receipt"] for row in receipt["files"].values())
    rar_bins = receipt["files"]["sparc_rar_bins"]
    assert rar_bins["representation"] == "normalized_published_snapshot"
    assert (
        rar_bins["declared_upstream_sha256"]
        == "d543cab7b720a4f14152ccc8158f7823072ce65a9c5b403d7c401b3f039a79d7"
    )
    assert rar_bins["declared_upstream_bytes"] - rar_bins["actual_bytes"] == 14
    cassini = receipt["files"]["cassini_q2_summary"]
    assert cassini["representation"] == "transcribed_published_summary"
    assert cassini["declared_upstream_sha256"] is None
    assert "No raw instrument telemetry" in receipt["claim_boundary"]


def test_well_formed_cassini_value_mutation_fails_byte_binding(tmp_path: Path) -> None:
    source = json.loads(CASSINI_PATH.read_text(encoding="utf-8"))
    source["observable"]["central_value_s2"] = 1.7e-27
    mutated = tmp_path / CASSINI_PATH.name
    mutated.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    receipt = _validate(cassini=mutated)

    row = receipt["files"]["cassini_q2_summary"]
    assert receipt["integrity_receipt"] is False
    assert row["integrity_receipt"] is False
    assert "selected_file_sha256_mismatch" in row["integrity_errors"]
    assert binding_errors_for_roles(receipt, CASSINI_ROLES)


def test_planck_and_sparc_mutations_fail_their_lanes(tmp_path: Path) -> None:
    planck = tmp_path / PLANCK_PATH.name
    planck.write_bytes(PLANCK_PATH.read_bytes() + b"\n")

    sparc = tmp_path / "sparc"
    shutil.copytree(SPARC_DIR, sparc)
    rar = sparc / "RAR.mrt"
    rar.write_bytes(rar.read_bytes().replace(b"-11.23", b"-11.24", 1))

    receipt = _validate(planck=planck, sparc=sparc)

    assert receipt["integrity_receipt"] is False
    assert (
        "selected_file_sha256_mismatch"
        in receipt["files"]["planck_tt_binned"]["integrity_errors"]
    )
    assert (
        "selected_file_sha256_mismatch"
        in receipt["files"]["sparc_rar"]["integrity_errors"]
    )
    assert binding_errors_for_roles(receipt, PLANCK_ROLES)
    assert binding_errors_for_roles(receipt, SPARC_ROLES)


def test_manifest_tamper_and_missing_manifest_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(manifest)
    cassini = next(
        row for row in tampered["files"] if row["role"] == "cassini_q2_summary"
    )
    cassini["sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered_manifest.json"
    tampered_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    tampered_receipt = _validate(manifest=tampered_path)
    missing_receipt = _validate(manifest=tmp_path / "missing.json")

    assert tampered_receipt["integrity_receipt"] is False
    assert tampered_receipt["canonical_source_binding_receipt"] is False
    assert "manifest_path_noncanonical" in tampered_receipt["global_integrity_errors"]
    assert (
        "canonical_manifest_sha256_mismatch"
        in tampered_receipt["global_integrity_errors"]
    )
    assert (
        "selected_file_sha256_mismatch"
        in tampered_receipt["files"]["cassini_q2_summary"]["integrity_errors"]
    )
    assert missing_receipt["integrity_receipt"] is False
    assert "manifest_missing" in missing_receipt["global_integrity_errors"]
    assert all(
        "manifest_role_unavailable" in row["integrity_errors"]
        for row in missing_receipt["files"].values()
    )


def test_self_consistent_synthetic_data_and_manifest_cannot_green_canonical_receipt(
    tmp_path: Path,
) -> None:
    source = json.loads(CASSINI_PATH.read_text(encoding="utf-8"))
    source["observable"]["central_value_s2"] = 9.9e-27
    synthetic = tmp_path / CASSINI_PATH.name
    synthetic.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = next(
        row for row in manifest["files"] if row["role"] == "cassini_q2_summary"
    )
    entry["sha256"] = hashlib.sha256(synthetic.read_bytes()).hexdigest()
    entry["bytes"] = synthetic.stat().st_size
    synthetic_manifest = tmp_path / "self_consistent_synthetic_manifest.json"
    synthetic_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    receipt = _validate(cassini=synthetic, manifest=synthetic_manifest)

    assert receipt["files"]["cassini_q2_summary"]["integrity_receipt"] is True
    assert receipt["canonical_source_binding_receipt"] is False
    assert receipt["integrity_receipt"] is False
    assert receipt["contains_external_published_measurements"] is False
    assert receipt["status"] == "NONCONFORMANT"
    assert "manifest_path_noncanonical" in receipt["global_integrity_errors"]
    assert "canonical_manifest_sha256_mismatch" in receipt["global_integrity_errors"]
