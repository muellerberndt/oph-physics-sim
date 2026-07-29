"""Shared fail-closed loaders for frozen local-domain receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return the repository's tagged SHA-256 representation."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_manifest_pinned_receipt(
    data_dir: str | Path,
    filename: str,
    manifest_key: str,
) -> dict[str, Any] | None:
    """Load a JSON receipt only when the manifest pins its exact bytes."""

    base = Path(data_dir)
    manifest_path = base / "manifest.json"
    receipt_path = base / filename
    if not manifest_path.exists() or not receipt_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            return None
        raw = receipt_path.read_bytes()
        if manifest.get(manifest_key) != sha256_bytes(raw):
            return None
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def manifest_pinned_artifact_sha256(
    data_dir: str | Path,
    filename: str,
    manifest_key: str,
) -> str | None:
    """Return the exact artifact hash only when the manifest pin matches."""

    base = Path(data_dir)
    manifest_path = base / "manifest.json"
    artifact_path = base / filename
    if not manifest_path.exists() or not artifact_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            return None
        digest = sha256_bytes(artifact_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return digest if manifest.get(manifest_key) == digest else None


def stage2_matches_source_domain(
    receipt: dict[str, Any] | None,
    source_projection_sha256: str,
    domain_freeze_sha256: str,
) -> bool:
    """Check the attained Stage-2 schema and its source/domain binding."""

    if receipt is None:
        return False
    try:
        frozen_domain = receipt["seam_layer"]["domain_complex"][
            "complex_freeze_sha256"
        ]
    except (KeyError, TypeError):
        return False
    return bool(
        receipt.get("schema") == "oph.local-domain-stage2.v1"
        and receipt.get("issue") == 634
        and receipt.get("physical_promotion_allowed") is False
        and receipt.get("verdict") == "ATTAINED"
        and receipt.get("source_projection_sha256")
        == source_projection_sha256
        and frozen_domain == domain_freeze_sha256
    )
