from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SCHEMA_PATH = _PROJECT_ROOT / "docs" / "oph_universe_timeline_visualization_payload_v1.schema.json"
DISTRIBUTED_SCHEMA_PATH = _PROJECT_ROOT / "docs" / "oph_distributed_universe_visualization_payload_v1.schema.json"
PACK_SCHEMA_PATH = _PROJECT_ROOT / "docs" / "oph_visualizer_pack_v2.schema.json"


def visualization_schema_path(payload: Mapping[str, Any]) -> Path:
    if payload.get("distributedSchema") == "oph_distributed_universe_visualization_payload_v1":
        return DISTRIBUTED_SCHEMA_PATH
    return CANONICAL_SCHEMA_PATH


def validate_visualization_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one canonical, compact-fallback, or distributed visualizer payload.

    Validation happens at export time so malformed references and schema drift do
    not become multi-gigabyte artifacts that fail only in the browser.
    """

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("visualization validation requires the jsonschema package") from exc

    schema_path = visualization_schema_path(payload)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dict(payload)), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = "/" + "/".join(str(part) for part in first.absolute_path)
        raise ValueError(
            f"visualization payload failed {schema_path.name} at {location or '/'}: {first.message}"
        )
    return {
        "schema": "oph_visualization_payload_validation_v1",
        "valid": True,
        "payloadSchema": payload.get("schemaVersion") or payload.get("schema"),
        "variant": "distributed" if payload.get("distributedSchema") else "canonical_or_fallback",
        "schemaFile": schema_path.name,
        "schemaId": schema.get("$id"),
    }


def validate_visualization_payload_file(path: Path) -> dict[str, Any]:
    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result = validate_visualization_payload(payload)
    return {**result, "payloadPath": str(payload_path), "payloadByteCount": payload_path.stat().st_size}


def validate_visualizer_pack_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("visualizer pack validation requires the jsonschema package") from exc
    schema = json.loads(PACK_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(manifest)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/" + "/".join(str(part) for part in first.absolute_path)
        raise ValueError(f"visualizer pack manifest failed at {location or '/'}: {first.message}")
    return {
        "valid": True,
        "schemaFile": PACK_SCHEMA_PATH.name,
        "schemaId": schema.get("$id"),
        "schema": manifest.get("schema"),
    }
