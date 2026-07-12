from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.viz.visualization_schema import (
    validate_visualization_payload,
    validate_visualizer_pack_manifest,
)


PACK_SCHEMA = "oph_visualizer_pack_v2"
DEFAULT_MAX_BYTES = 256_000_000
DEFAULT_TARGET_BYTES = 128_000_000
DEFAULT_CHUNK_BYTES = 4_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_visualizer_pack(
    *,
    bundle_dir: Path,
    out_path: Path,
    payload: Mapping[str, Any] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    target_bytes: int = DEFAULT_TARGET_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    compression_level: int = 10,
) -> dict[str, Any]:
    """Build a deterministic, content-addressed, hard-budgeted ``tar.zst`` pack.

    The canonical payload is split by top-level section. Large nested arrays are
    replaced by explicit chunk-list descriptors, so a web app can request only
    the observer, time, or worldline chunks currently in view. The original
    multi-gigabyte JSON and embedded HTML are deliberately not copied.
    """

    bundle_path = Path(bundle_dir)
    destination = Path(out_path)
    if payload is None:
        payload_path = bundle_path / "visualization_payload.json"
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_dict = _canonical_json_roundtrip(dict(payload))
    validation = validate_visualization_payload(payload_dict)
    max_bytes = int(max_bytes)
    target_bytes = int(target_bytes)
    chunk_bytes = int(chunk_bytes)
    if max_bytes <= 0 or max_bytes > DEFAULT_MAX_BYTES:
        raise ValueError(f"max_bytes must be in 1..{DEFAULT_MAX_BYTES}")
    if target_bytes <= 0 or target_bytes > DEFAULT_TARGET_BYTES or target_bytes >= max_bytes:
        raise ValueError(
            f"target_bytes must be positive, <= {DEFAULT_TARGET_BYTES}, and strictly below max_bytes"
        )
    if chunk_bytes < 64_000:
        raise ValueError("chunk_bytes must be at least 64000")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="oph_visualizer_pack_") as temp_name:
        staging = Path(temp_name) / "pack"
        staging.mkdir(parents=True)
        state = _ChunkState(staging=staging, chunk_bytes=chunk_bytes)
        section_refs: dict[str, Any] = {}
        inline: dict[str, Any] = {}
        for key in sorted(payload_dict):
            value = payload_dict[key]
            if isinstance(value, (dict, list)):
                transformed = state.externalize(value, path=(str(key),))
                section_path = f"sections/{_safe_name(str(key))}.json"
                _write_json_bytes(staging / section_path, transformed)
                section_refs[key] = {
                    "$ref": section_path,
                    "sha256": _sha256_file(staging / section_path),
                }
            else:
                inline[key] = value

        payload_index = {
            "schema": "oph_visualizer_payload_index_v2",
            "payloadSchema": payload_dict.get("schemaVersion") or payload_dict.get("schema"),
            "inline": inline,
            "sections": section_refs,
            "chunkDescriptor": {
                "type": "chunked_json_list_v1",
                "reconstruction": (
                    "Replace each {$chunkList:[{path,...}],count:N} object with the concatenation of "
                    "the JSON arrays stored at those paths, then load top-level sections from sections/*.json."
                ),
            },
        }
        _write_json_bytes(staging / "payload_index.json", payload_index)

        included_sidecars = []
        sidecar_sources = [
            *bundle_path.glob("screen_full_*.bin"),
            *bundle_path.glob("screen_frames_*.bin"),
            *bundle_path.glob("observers_full_*.json"),
            *bundle_path.glob("cameras_full_*.json"),
            *bundle_path.glob("repair_trace_full.csv"),
            *bundle_path.glob("visualization_export_manifest.json"),
        ]
        for source in sorted(sidecar_sources, key=lambda item: item.name):
            target = staging / "sidecars" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            included_sidecars.append(str(target.relative_to(staging)))
        included_documentation = []
        for name in ("VISUALIZATION_INSTRUCTIONS.md", "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md"):
            source = bundle_path / name
            if not source.exists():
                continue
            target = staging / "docs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                _portable_bundle_text(source.read_text(encoding="utf-8"), bundle_path=bundle_path),
                encoding="utf-8",
            )
            included_documentation.append(str(target.relative_to(staging)))
        for name in (
            "oph_universe_timeline_visualization_payload_v1.schema.json",
            "oph_distributed_universe_visualization_payload_v1.schema.json",
            "oph_visualizer_pack_v2.schema.json",
            "SIMULATION_ASSUMPTION_POLICY.md",
        ):
            source = PROJECT_ROOT / "docs" / name
            if not source.exists():
                continue
            target = staging / "docs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            included_documentation.append(str(target.relative_to(staging)))

        portable_summary = _portable_payload_summary(payload_dict)
        _write_json_bytes(staging / "universe_timeline_summary.json", portable_summary)
        viewer = bundle_path / "oph_universe_timeline_viewer.html"
        if viewer.exists() and viewer.stat().st_size <= 1_000_000:
            shutil.copyfile(viewer, staging / viewer.name)

        file_rows = _file_rows(staging)
        manifest = {
            "schema": PACK_SCHEMA,
            "payloadSchema": payload_dict.get("schemaVersion") or payload_dict.get("schema"),
            "validation": validation,
            "payloadIndex": "payload_index.json",
            "canonicalPayloadSha256": _sha256_bytes(_json_bytes(payload_dict)),
            "fileCountBeforeManifest": len(file_rows),
            "files": file_rows,
            "includedSidecars": included_sidecars,
            "includedDocumentation": sorted(included_documentation),
            "chunking": {
                "format": "content_addressed_json_array_chunks_v1",
                "targetUncompressedChunkBytes": chunk_bytes,
                "chunkCount": state.chunk_count,
                "deduplicatedChunkReferenceCount": state.deduplicated_references,
            },
            "archiveBudget": {
                "targetBytes": target_bytes,
                "hardMaximumBytesExclusive": max_bytes,
                "policy": "archive_is_deleted_and_build_fails_when_size_is_not_strictly_below_maximum",
            },
            "manifestSelfHashPolicy": (
                "manifest.json is the sole self-excluded member; every other archived payload, chunk, "
                "viewer, and sidecar file is listed with byte count and SHA-256 under files"
            ),
            "provenance": "simulator_visualization_export",
            "claimBoundary": (
                "Renderer data package. Simulation assumptions remain explicitly tagged in the payload; "
                "portable instructions and schemas are current for this payload, stale outer summaries are "
                "ignored, and this archive does not upgrade theorem or physical receipts."
            ),
        }
        pack_manifest_validation = validate_visualizer_pack_manifest(manifest)
        _write_json_bytes(staging / "manifest.json", manifest)

        temporary_archive = destination.with_name(f".{destination.name}.tmp")
        if temporary_archive.exists():
            temporary_archive.unlink()
        _write_deterministic_tar_zst(
            staging=staging,
            out_path=temporary_archive,
            compression_level=compression_level,
        )
        archive_bytes = temporary_archive.stat().st_size
        if archive_bytes >= max_bytes:
            temporary_archive.unlink(missing_ok=True)
            raise ValueError(
                f"visualizer pack is {archive_bytes} bytes; hard gate requires < {max_bytes} bytes"
            )
        temporary_archive.replace(destination)

    return {
        "schema": PACK_SCHEMA,
        "path": str(destination),
        "byte_count": int(destination.stat().st_size),
        "sha256": _sha256_file(destination),
        "target_bytes": target_bytes,
        "hard_maximum_bytes_exclusive": max_bytes,
        "within_target": bool(destination.stat().st_size <= target_bytes),
        "under_hard_limit": True,
        "chunk_count": state.chunk_count,
        "deduplicated_chunk_reference_count": state.deduplicated_references,
        "payload_validation": validation,
        "pack_manifest_validation": pack_manifest_validation,
    }


def read_visualizer_pack_payload(pack_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all manifested members and exactly reconstruct the canonical payload.

    This is primarily a release/CI verifier. Production web clients should
    resolve only the chunks needed for the current view instead of rebuilding
    the entire payload in memory.
    """

    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("visualizer pack reading requires the zstandard package") from exc

    archive_path = Path(pack_path)
    with tempfile.TemporaryDirectory(prefix="oph_visualizer_unpack_") as temp_name:
        root = Path(temp_name)
        with archive_path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        relative = Path(member.name)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError(f"unsafe path in visualizer pack: {member.name}")
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError(f"could not read visualizer pack member: {member.name}")
                        destination = root / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as output:
                            shutil.copyfileobj(source, output)

        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("visualizer pack is missing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_validation = validate_visualizer_pack_manifest(manifest)
        expected = {row["path"]: row for row in manifest.get("files", []) if isinstance(row, dict)}
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if actual_paths != set(expected):
            missing = sorted(set(expected) - actual_paths)
            unexpected = sorted(actual_paths - set(expected))
            raise ValueError(f"visualizer pack member mismatch: missing={missing}, unexpected={unexpected}")
        for relative, row in expected.items():
            path = _safe_pack_path(root, relative)
            if path.stat().st_size != int(row.get("byteCount", -1)):
                raise ValueError(f"visualizer pack byte-count mismatch: {relative}")
            if _sha256_file(path) != row.get("sha256"):
                raise ValueError(f"visualizer pack SHA-256 mismatch: {relative}")

        index = json.loads((root / manifest["payloadIndex"]).read_text(encoding="utf-8"))
        payload = dict(index.get("inline") or {})
        for key, ref in (index.get("sections") or {}).items():
            section_path = _safe_pack_path(root, ref["$ref"])
            if _sha256_file(section_path) != ref.get("sha256"):
                raise ValueError(f"payload section SHA-256 mismatch: {key}")
            transformed = json.loads(section_path.read_text(encoding="utf-8"))
            payload[str(key)] = _resolve_chunk_lists(transformed, root=root)
        payload_validation = validate_visualization_payload(payload)
        reconstructed_sha256 = _sha256_bytes(_json_bytes(payload))
        if reconstructed_sha256 != manifest.get("canonicalPayloadSha256"):
            raise ValueError("reconstructed visualizer payload does not match canonical payload SHA-256")
        report = {
            "schema": "oph_visualizer_pack_reconstruction_report_v1",
            "packPath": str(archive_path),
            "manifestValidation": manifest_validation,
            "payloadValidation": payload_validation,
            "manifestedFileCount": len(expected),
            "allManifestedHashesVerified": True,
            "manifestSelfExcludedByPolicy": True,
            "exactPayloadReconstructionReceipt": True,
            "canonicalPayloadSha256": reconstructed_sha256,
        }
        return payload, report


def _resolve_chunk_lists(value: Any, *, root: Path) -> Any:
    if isinstance(value, list):
        return [_resolve_chunk_lists(item, root=root) for item in value]
    if not isinstance(value, dict):
        return value
    chunks = value.get("$chunkList")
    if isinstance(chunks, list):
        result = []
        expected_start = 0
        for row in chunks:
            if not isinstance(row, dict) or int(row.get("startIndex", -1)) != expected_start:
                raise ValueError("visualizer chunk ordering is invalid")
            path = _safe_pack_path(root, row.get("path"))
            if _sha256_file(path) != row.get("sha256"):
                raise ValueError(f"visualizer chunk SHA-256 mismatch: {row.get('path')}")
            items = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(items, list) or len(items) != int(row.get("itemCount", -1)):
                raise ValueError(f"visualizer chunk item-count mismatch: {row.get('path')}")
            result.extend(_resolve_chunk_lists(item, root=root) for item in items)
            expected_start += len(items)
        if expected_start != int(value.get("count", -1)):
            raise ValueError("visualizer chunk-list total count mismatch")
        return result
    return {key: _resolve_chunk_lists(item, root=root) for key, item in value.items()}


class _ChunkState:
    def __init__(self, *, staging: Path, chunk_bytes: int) -> None:
        self.staging = staging
        self.chunk_bytes = int(chunk_bytes)
        self.chunk_count = 0
        self.deduplicated_references = 0
        self._known_chunks: set[str] = set()

    def externalize(self, value: Any, *, path: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self.externalize(item, path=(*path, str(key)))
                for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            }
        if not isinstance(value, list):
            return value
        transformed = [self.externalize(item, path=(*path, str(index))) for index, item in enumerate(value)]
        encoded_items = [_json_bytes(item) for item in transformed]
        total_size = 2 + max(0, len(encoded_items) - 1) + sum(len(item) for item in encoded_items)
        if total_size <= self.chunk_bytes:
            return transformed
        chunks = []
        current: list[bytes] = []
        current_size = 2
        start_index = 0
        for item in encoded_items:
            added = len(item) + (1 if current else 0)
            if current and current_size + added > self.chunk_bytes:
                chunks.append(self._write_chunk(current, start_index=start_index))
                start_index += len(current)
                current = []
                current_size = 2
            current.append(item)
            current_size += len(item) + (1 if len(current) > 1 else 0)
        if current:
            chunks.append(self._write_chunk(current, start_index=start_index))
        return {
            "$chunkList": chunks,
            "count": len(transformed),
            "sourcePath": "/" + "/".join(path),
        }

    def _write_chunk(self, items: list[bytes], *, start_index: int) -> dict[str, Any]:
        raw = b"[" + b",".join(items) + b"]"
        digest = hashlib.sha256(raw).hexdigest()
        relative = f"chunks/{digest[:2]}/{digest}.json"
        target = self.staging / relative
        if digest in self._known_chunks:
            self.deduplicated_references += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            self._known_chunks.add(digest)
            self.chunk_count += 1
        return {
            "path": relative,
            "sha256": f"sha256:{digest}",
            "startIndex": start_index,
            "itemCount": len(items),
            "byteCount": len(raw),
        }


def _write_deterministic_tar_zst(*, staging: Path, out_path: Path, compression_level: int) -> None:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("visualizer pack creation requires the zstandard package") from exc

    with tempfile.NamedTemporaryFile(prefix="oph_visualizer_pack_", suffix=".tar", delete=False) as handle:
        tar_path = Path(handle.name)
    try:
        with tarfile.open(tar_path, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for source in sorted((path for path in staging.rglob("*") if path.is_file()), key=lambda item: item.as_posix()):
                relative = source.relative_to(staging).as_posix()
                info = tarfile.TarInfo(name=relative)
                info.size = source.stat().st_size
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with source.open("rb") as source_handle:
                    archive.addfile(info, source_handle)
        compressor = zstandard.ZstdCompressor(
            level=int(compression_level),
            threads=0,
            write_checksum=True,
            write_content_size=True,
        )
        with tar_path.open("rb") as source_handle, Path(out_path).open("wb") as output_handle:
            compressor.copy_stream(source_handle, output_handle)
    finally:
        tar_path.unlink(missing_ok=True)


def _file_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "byteCount": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
    ]


def _write_json_bytes(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _canonical_json_roundtrip(value: Any) -> Any:
    """Normalize Python-only key/container types to the bytes a JSON consumer sees."""

    raw = json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return json.loads(raw)


def _portable_bundle_text(value: str, *, bundle_path: Path) -> str:
    roots = {str(Path(bundle_path)), str(Path(bundle_path).resolve())}
    text = str(value)
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(f"{root}/", "../")
        text = text.replace(root, "..")
    return text


def _portable_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    observer = payload.get("observerModularTime") if isinstance(payload.get("observerModularTime"), Mapping) else {}
    bulk = payload.get("consensusBulk") if isinstance(payload.get("consensusBulk"), Mapping) else {}
    proto = bulk.get("protoParticleCandidates") if isinstance(bulk.get("protoParticleCandidates"), Mapping) else {}
    ds4 = payload.get("assumedDs4Spacetime") if isinstance(payload.get("assumedDs4Spacetime"), Mapping) else {}
    completeness = (
        (payload.get("visualizationRenderData") or {}).get("visualUniverseCompleteness", {})
        if isinstance(payload.get("visualizationRenderData"), Mapping)
        else {}
    )
    return {
        "schema": "oph_visualizer_pack_portable_summary_v1",
        "payload_schema": payload.get("schemaVersion") or payload.get("schema"),
        "observer_count": len(observer.get("observers") or []),
        "objective_observer_view_count": len(observer.get("objectiveObserverViews") or []),
        "h3_object_count": len(bulk.get("objects") or []),
        "proto_particle_candidate_worldline_count": len(proto.get("worldlines") or []),
        "assumed_ds4_visualization_layer_receipt": bool(
            (ds4.get("receipts") or {}).get("assumed_ds4_visualization_layer_receipt", False)
        ),
        "visual_universe_render_ready_receipt": bool(
            (completeness.get("receipts") or {}).get("VISUAL_UNIVERSE_RENDER_READY_RECEIPT", False)
        ),
        "claim_boundary": payload.get("claimBoundary"),
        "source": "canonical_payload_sections_in_this_pack",
    }


def _sha256_bytes(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _safe_pack_path(root: Path, value: Any) -> Path:
    relative = Path(str(value))
    if not str(value) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe visualizer pack member reference: {value}")
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"unsafe visualizer pack member reference: {value}")
    return candidate
