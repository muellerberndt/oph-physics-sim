"""Fail-closed byte binding for the committed public measurement bundle.

The public-data comparison suite consumes published Planck and SPARC tables
and a locally transcribed Cassini summary statistic.  Recording the digest of
whatever path a caller supplies is not a source-integrity check: a mutated but
well-formed file would merely receive a new digest and could retain a positive
comparison receipt.  This module compares every selected input with an
independently committed manifest entry before the input is eligible for the
suite's integrity receipt.

The positive verdict is deliberately narrow.  It proves exact byte identity
with the reviewed repository copies and binds those copies to declared public
source URLs.  It does not contact or authenticate the remote hosts, attest to
download custody, recover raw instrument telemetry, evaluate an official
likelihood, or support an OPH physics claim.  In particular, the Cassini file
contains a published summary statistic rather than the underlying tracking
data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_ID = "oph.public_measurement_source_manifest.v1"
VALIDATOR_ID = "oph.public_measurement_source_binding.v1"
PASS_STATUS = "PUBLIC_MEASUREMENT_BYTES_BOUND"
FAIL_STATUS = "NONCONFORMANT"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "measurements"
    / "public_measurement_source_manifest_v1.json"
)
CANONICAL_MANIFEST_SHA256 = (
    "665a44f9056af9d59ef3d47f493e97bab8e06f8550933160b56e76040d730050"
)

ROLE_FILENAMES = {
    "planck_tt_binned": "COM_PowerSpect_CMB-TT-binned_R3.01.txt",
    "sparc_rar": "RAR.mrt",
    "sparc_rar_bins": "RARbins.mrt",
    "sparc_btfr": "BTFR_Lelli2019.mrt",
    "sparc_mass_models": "MassModels_Lelli2016c.mrt",
    "cassini_q2_summary": "cassini_q2_2026.json",
}

PLANCK_ROLES = ("planck_tt_binned",)
SPARC_ROLES = (
    "sparc_rar",
    "sparc_rar_bins",
    "sparc_btfr",
    "sparc_mass_models",
)
CASSINI_ROLES = ("cassini_q2_summary",)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RAR_BINS_DATA_ROW = re.compile(r"^\s*-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\.\d+\s+\d+\s*$")

REPRESENTATIONS = (
    "exact_published_bytes",
    "normalized_published_snapshot",
    "transcribed_published_summary",
)
RAR_BINS_NORMALIZATION_ID = "strip_one_trailing_ascii_space_from_each_data_row_v1"

CLAIM_BOUNDARY = (
    "A positive receipt proves only that the selected local Planck, SPARC, "
    "and Cassini-summary bytes equal the exact local representations pinned by "
    "the committed manifest: exact published copies, one documented normalized "
    "snapshot, and one structured transcription. The URLs and upstream digests "
    "are declarations and are not fetched or authenticated by this validator. "
    "No raw instrument telemetry, independent custody, official likelihood, "
    "OPH prediction, validation, or falsification follows."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["manifest_missing"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"manifest_unreadable:{type(exc).__name__}"]
    if not isinstance(value, dict):
        return {}, ["manifest_not_object"]
    return value, []


def _safe_manifest_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _selected_paths(
    planck_tt_path: Path,
    sparc_dir: Path,
    cassini_summary_path: Path,
) -> dict[str, Path]:
    return {
        "planck_tt_binned": Path(planck_tt_path),
        "sparc_rar": Path(sparc_dir) / ROLE_FILENAMES["sparc_rar"],
        "sparc_rar_bins": Path(sparc_dir) / ROLE_FILENAMES["sparc_rar_bins"],
        "sparc_btfr": Path(sparc_dir) / ROLE_FILENAMES["sparc_btfr"],
        "sparc_mass_models": Path(sparc_dir) / ROLE_FILENAMES["sparc_mass_models"],
        "cassini_q2_summary": Path(cassini_summary_path),
    }


def _manifest_entries(
    manifest: Mapping[str, Any], errors: list[str]
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != SCHEMA_ID:
        errors.append("manifest_schema_mismatch")
    if not isinstance(manifest.get("bundle_id"), str) or not manifest.get("bundle_id"):
        errors.append("manifest_bundle_id_missing")
    if not isinstance(manifest.get("claim_boundary"), str) or not manifest.get(
        "claim_boundary"
    ):
        errors.append("manifest_claim_boundary_missing")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        errors.append("manifest_files_not_list")
        return {}

    entries: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, dict):
            errors.append(f"manifest_entry_not_object:{index}")
            continue
        role = raw_entry.get("role")
        if role not in ROLE_FILENAMES:
            errors.append(f"manifest_unknown_role:{role}")
            continue
        if role in entries:
            errors.append(f"manifest_duplicate_role:{role}")
            continue
        entries[role] = raw_entry

    for role in ROLE_FILENAMES:
        if role not in entries:
            errors.append(f"manifest_missing_role:{role}")
    if len(raw_files) != len(ROLE_FILENAMES):
        errors.append("manifest_file_count_mismatch")
    return entries


def validate_public_measurement_sources(
    *,
    planck_tt_path: Path,
    sparc_dir: Path,
    cassini_summary_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate selected public-data inputs against the committed byte pins."""

    selected = _selected_paths(planck_tt_path, sparc_dir, cassini_summary_path)
    manifest_file = Path(manifest_path)
    manifest, global_errors = _load_json_object(manifest_file)
    manifest_hash = _sha256(manifest_file) if manifest_file.is_file() else None
    if manifest_file.resolve() != DEFAULT_MANIFEST_PATH.resolve():
        global_errors.append("manifest_path_noncanonical")
    if manifest_hash != CANONICAL_MANIFEST_SHA256:
        global_errors.append("canonical_manifest_sha256_mismatch")
    entries = _manifest_entries(manifest, global_errors) if manifest else {}

    files: dict[str, dict[str, Any]] = {}
    for role, selected_path in selected.items():
        entry = entries.get(role, {})
        errors: list[str] = []

        manifest_relpath = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        public_source = entry.get("public_source")
        data_kind = entry.get("data_kind")

        if entry and not _safe_manifest_path(manifest_relpath):
            errors.append("manifest_path_not_safe_relative")
        if entry and Path(str(manifest_relpath)).name != ROLE_FILENAMES[role]:
            errors.append("manifest_filename_mismatch")
        if entry and (
            not isinstance(expected_hash, str)
            or _HEX64.fullmatch(expected_hash) is None
        ):
            errors.append("manifest_sha256_invalid")
        if entry and (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            errors.append("manifest_byte_count_invalid")
        if entry and (
            not isinstance(public_source, str)
            or not public_source.startswith("https://")
        ):
            errors.append("manifest_public_source_not_https")
        if entry and (not isinstance(data_kind, str) or not data_kind):
            errors.append("manifest_data_kind_missing")

        representation = entry.get("representation")
        upstream_hash = entry.get("upstream_sha256")
        upstream_bytes = entry.get("upstream_bytes")
        normalization = entry.get("normalization")
        transcription_scope = entry.get("transcription_scope")
        if entry and representation not in REPRESENTATIONS:
            errors.append("manifest_representation_invalid")
        elif representation == "exact_published_bytes":
            if upstream_hash != expected_hash:
                errors.append("exact_representation_upstream_sha256_mismatch")
            if upstream_bytes != expected_bytes:
                errors.append("exact_representation_upstream_byte_count_mismatch")
        elif representation == "normalized_published_snapshot":
            if (
                not isinstance(upstream_hash, str)
                or _HEX64.fullmatch(upstream_hash) is None
            ):
                errors.append("normalized_representation_upstream_sha256_invalid")
            if (
                not isinstance(upstream_bytes, int)
                or isinstance(upstream_bytes, bool)
                or upstream_bytes <= 0
            ):
                errors.append("normalized_representation_upstream_byte_count_invalid")
            if not isinstance(normalization, dict) or not normalization.get("id"):
                errors.append("normalized_representation_contract_missing")
        elif representation == "transcribed_published_summary":
            if upstream_hash is not None or upstream_bytes is not None:
                errors.append("transcribed_summary_must_not_claim_upstream_file_digest")
            if not isinstance(transcription_scope, str) or not transcription_scope:
                errors.append("transcribed_summary_scope_missing")
        if not entry:
            errors.append("manifest_role_unavailable")

        actual_hash: str | None = None
        actual_bytes: int | None = None
        if not selected_path.is_file():
            errors.append("selected_file_missing")
        else:
            try:
                actual_hash = _sha256(selected_path)
                actual_bytes = selected_path.stat().st_size
            except OSError as exc:
                errors.append(f"selected_file_unreadable:{type(exc).__name__}")

        if actual_hash is not None and isinstance(expected_hash, str):
            if actual_hash != expected_hash:
                errors.append("selected_file_sha256_mismatch")
        if actual_bytes is not None and isinstance(expected_bytes, int):
            if actual_bytes != expected_bytes:
                errors.append("selected_file_byte_count_mismatch")

        if (
            role == "sparc_rar_bins"
            and representation == "normalized_published_snapshot"
        ):
            if (
                not isinstance(normalization, dict)
                or normalization.get("id") != RAR_BINS_NORMALIZATION_ID
            ):
                errors.append("rar_bins_normalization_id_mismatch")
            if isinstance(upstream_bytes, int) and isinstance(expected_bytes, int):
                if upstream_bytes - expected_bytes != 14:
                    errors.append("rar_bins_normalization_byte_delta_mismatch")
            if selected_path.is_file():
                try:
                    lines = selected_path.read_text(encoding="ascii").splitlines()
                except (OSError, UnicodeError) as exc:
                    errors.append(
                        f"rar_bins_normalized_text_unreadable:{type(exc).__name__}"
                    )
                else:
                    data_rows = [
                        line for line in lines if _RAR_BINS_DATA_ROW.fullmatch(line)
                    ]
                    if len(lines) != 27:
                        errors.append("rar_bins_normalized_line_count_mismatch")
                    if len(data_rows) != 14:
                        errors.append("rar_bins_normalized_data_row_count_mismatch")
                    if any(line.endswith((" ", "\t")) for line in data_rows):
                        errors.append("rar_bins_normalized_data_row_has_trailing_space")

        files[role] = {
            "selected_path": str(selected_path),
            "manifest_path": manifest_relpath,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "data_kind": data_kind,
            "representation": representation,
            "declared_upstream_sha256": upstream_hash,
            "declared_upstream_bytes": upstream_bytes,
            "normalization": normalization,
            "transcription_scope": transcription_scope,
            "public_source": public_source,
            "integrity_receipt": not errors,
            "integrity_errors": sorted(set(errors)),
        }

    global_errors = sorted(set(global_errors))
    all_errors = [*global_errors]
    all_errors.extend(
        f"{role}:{error}"
        for role, record in files.items()
        for error in record["integrity_errors"]
    )
    all_errors = sorted(set(all_errors))
    passed = not all_errors

    return {
        "schema": "oph.public_measurement_source_binding_receipt.v1",
        "validator": VALIDATOR_ID,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_hash,
        "canonical_manifest_path": str(DEFAULT_MANIFEST_PATH),
        "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "bundle_id": manifest.get("bundle_id") if manifest else None,
        "contains_external_published_measurements": passed,
        "raw_instrument_telemetry_included": False,
        "remote_source_authentication_performed": False,
        "files": files,
        "global_integrity_errors": global_errors,
        "canonical_source_binding_receipt": passed,
        "integrity_receipt": passed,
        "integrity_errors": all_errors,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def binding_errors_for_roles(
    receipt: Mapping[str, Any], roles: tuple[str, ...]
) -> list[str]:
    """Project bundle and per-file errors into one comparison lane."""

    errors = [
        f"public_measurement_binding:{value}"
        for value in receipt.get("global_integrity_errors", [])
    ]
    files = receipt.get("files") if isinstance(receipt.get("files"), dict) else {}
    for role in roles:
        record = files.get(role) if isinstance(files.get(role), dict) else {}
        if not record:
            errors.append(f"public_measurement_binding:{role}:receipt_missing")
            continue
        errors.extend(
            f"public_measurement_binding:{role}:{value}"
            for value in record.get("integrity_errors", [])
        )
    return sorted(set(errors))
