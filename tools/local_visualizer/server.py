"""Dependency-free local server for OPH visualization payloads.

The server has two deliberately small trust surfaces:

* three bundled, immutable frontend files; and
* one selected visualization payload plus sidecars explicitly listed by a
  recognized visualization export manifest.

It never exposes a directory tree.  Data files are addressed by opaque IDs and
served through bounded page, row, or HTTP Range endpoints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import io
import json
import mimetypes
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser


API_SCHEMA = "oph.local-visualizer-api/1.0.0"
RECOGNIZED_MANIFEST_SCHEMA = "oph_universe_visualization_sidecars_v1"
DEFAULT_PAGE_SIZE = 256 * 1024
MAX_PAGE_SIZE = 4 * 1024 * 1024
MAX_DIRECT_FILE_BYTES = 4 * 1024 * 1024
MAX_ROW_LIMIT = 2_000
MAX_ROW_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_JSON_SIDECAR_SCAN_BYTES = 16 * 1024 * 1024
MAX_TEXT_SIDECAR_SCAN_BYTES = 64 * 1024 * 1024
ALLOWED_DATA_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".bin", ".npy", ".npz"})
SNAPSHOTTED_SIDECAR_SUFFIXES = frozenset({".json", ".jsonl", ".csv"})
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/static/app.js": "app.js",
    "/static/styles.css": "styles.css",
}
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "authorization",
        "password",
        "passphrase",
        "private_key",
        "secret_key",
        "client_secret",
        "session_token",
        "cookie",
        "set_cookie",
    }
)
SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_EMBEDDED_PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:Users|home|private|tmp|var|opt|root|Volumes)(?:/[^\s'\"<>]*)?"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b"),
)
PRIVATE_PATH_KEYS = frozenset(
    {
        "source_paths",
        "source_path",
        "payload_path",
        "manifest_path",
        "bundle_dir",
        "viewer_path",
        "instructions_path",
        "data_root",
        "sidecar_root",
    }
)
PHYSICAL_A5_REPORT_SCHEMA = "oph.physical-a5-sm.requirements-audit/1.0.0"
PHYSICAL_A5_REPORT_ARTIFACT = "OPH_PHYSICAL_A5_SM_REQUIREMENTS_AUDIT"
PUBLIC_REDACTION_MARKER = "[REDACTED_BY_LOCAL_VISUALIZER]"


class UnsafeDataError(ValueError):
    """Raised when a payload or sidecar crosses the local serving boundary."""


@dataclass(frozen=True)
class ServedFile:
    """One immutable file admitted to the local data API."""

    file_id: str
    path: Path
    relative_path: str
    byte_count: int
    sha256: str
    media_type: str
    logical_name: str
    snapshot: bytes = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def public_row(self) -> dict[str, Any]:
        row = {
            "fileId": self.file_id,
            "name": self.path.name,
            "relativePath": self.relative_path,
            "logicalName": self.logical_name,
            "byteCount": self.byte_count,
            "sha256": self.sha256,
            "mediaType": self.media_type,
            "rangeEndpoint": f"/api/files/{self.file_id}",
            "pageEndpoint": f"/api/pages/{self.file_id}",
        }
        if self.path.suffix.lower() in {".csv", ".jsonl", ".json"}:
            row["rowsEndpoint"] = f"/api/rows/{self.file_id}"
        row.update(self.metadata)
        return row


class VisualizerDataStore:
    """Validated payload, bounded summary, and manifested sidecar inventory."""

    def __init__(
        self,
        payload_path: Path | str,
        *,
        data_root: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ) -> None:
        raw_payload_path = Path(payload_path).expanduser().absolute()
        raw_root = (
            Path(data_root).expanduser().absolute()
            if data_root is not None
            else raw_payload_path.parent
        )
        if not raw_root.exists() or not raw_root.is_dir():
            raise UnsafeDataError("data_root_must_be_existing_directory")
        self.root = raw_root.resolve(strict=True)
        self.payload_path = self._admit_file(raw_payload_path, require_json=True)
        try:
            payload = _strict_json_loads(
                _read_stable_bytes(self.payload_path).decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, UnsafeDataError) as exc:
            raise UnsafeDataError("payload_must_be_valid_utf8_json") from exc
        if not isinstance(payload, dict):
            raise UnsafeDataError("payload_root_must_be_object")
        sensitive = _find_sensitive_key(payload)
        if sensitive is not None:
            raise UnsafeDataError(f"payload_contains_sensitive_key:{sensitive}")
        self.payload = _public_payload_snapshot(payload)
        self.standalone_screen_a5 = (
            payload.get("schema") == "oph.screen-a5-visualization-ladder/1.0.0"
        )
        self.files: dict[str, ServedFile] = {}
        self._add_file(
            self.payload_path,
            logical_name="visualization_payload",
            file_id="payload",
            snapshot=_json_bytes(self.payload),
        )

        source_manifest = (
            Path(manifest_path).expanduser().absolute()
            if manifest_path is not None
            else self.payload_path.parent / "visualization_export_manifest.json"
        )
        self.manifest_status = "absent"
        if source_manifest.exists():
            self._load_export_manifest(source_manifest)
        self.summary = _build_summary(self.payload, self.files)

    def _admit_file(self, candidate: Path, *, require_json: bool = False) -> Path:
        if candidate.is_symlink():
            raise UnsafeDataError("symlink_files_are_not_served")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise UnsafeDataError("served_file_missing") from exc
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise UnsafeDataError("served_file_outside_data_root") from exc
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise UnsafeDataError("symlink_path_component_not_served")
        if not resolved.is_file():
            raise UnsafeDataError("served_path_must_be_regular_file")
        if _looks_sensitive_path(relative):
            raise UnsafeDataError("sensitive_filename_not_served")
        suffix = resolved.suffix.lower()
        if require_json and suffix != ".json":
            raise UnsafeDataError("payload_must_have_json_suffix")
        if suffix not in ALLOWED_DATA_SUFFIXES:
            raise UnsafeDataError("unsupported_visualization_file_type")
        return resolved

    def _add_file(
        self,
        path: Path,
        *,
        logical_name: str,
        file_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        snapshot: bytes | None = None,
    ) -> None:
        relative = path.relative_to(self.root).as_posix()
        identifier = file_id or f"data-{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]}"
        if not _SAFE_ID_RE.fullmatch(identifier):
            raise UnsafeDataError("invalid_internal_file_id")
        if identifier in self.files:
            return
        immutable_bytes = snapshot if snapshot is not None else _read_stable_bytes(path)
        media_type = _media_type(path)
        safe_metadata = _public_manifest_metadata(metadata or {})
        self.files[identifier] = ServedFile(
            file_id=identifier,
            path=path,
            relative_path=relative,
            byte_count=len(immutable_bytes),
            sha256=hashlib.sha256(immutable_bytes).hexdigest(),
            media_type=media_type,
            logical_name=logical_name,
            snapshot=immutable_bytes,
            metadata=safe_metadata,
        )

    def _load_export_manifest(self, manifest_path: Path) -> None:
        try:
            admitted = self._admit_file(manifest_path, require_json=True)
            manifest = _strict_json_loads(
                _read_stable_bytes(admitted).decode("utf-8")
            )
        except (UnsafeDataError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            self.manifest_status = "rejected"
            return
        if not isinstance(manifest, dict) or manifest.get("schema") != RECOGNIZED_MANIFEST_SCHEMA:
            self.manifest_status = "unrecognized_schema"
            return
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            self.manifest_status = "invalid_files_table"
            return
        admitted_count = 0
        rejected_count = 0
        for logical_name, entry, raw_path in _manifest_path_entries(files):
            if "written" in entry and type(entry.get("written")) is not bool:
                rejected_count += 1
                continue
            if entry.get("written") is False:
                continue
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = admitted.parent / candidate
            try:
                safe_path = self._admit_file(candidate.absolute())
                immutable_bytes = _safe_sidecar_snapshot(
                    safe_path,
                    logical_name=logical_name,
                )
                _verify_declared_integrity(
                    immutable_bytes,
                    entry=entry,
                )
                self._add_file(
                    safe_path,
                    logical_name=logical_name,
                    metadata=entry,
                    snapshot=immutable_bytes,
                )
            except UnsafeDataError:
                rejected_count += 1
                continue
            admitted_count += 1
        self.manifest_status = f"accepted:{admitted_count}:rejected:{rejected_count}"

    def file(self, file_id: str) -> ServedFile | None:
        if not _SAFE_ID_RE.fullmatch(file_id):
            return None
        return self.files.get(file_id)

    def api_manifest(self) -> dict[str, Any]:
        return {
            "schema": API_SCHEMA,
            "payloadSchema": self.payload.get("schemaVersion") or self.payload.get("schema"),
            "manifestStatus": self.manifest_status,
            "files": [record.public_row() for record in self.files.values()],
            "paging": {
                "defaultPageSize": DEFAULT_PAGE_SIZE,
                "maximumPageSize": MAX_PAGE_SIZE,
                "maximumDirectFileBytes": MAX_DIRECT_FILE_BYTES,
                "maximumRowLimit": MAX_ROW_LIMIT,
                "archivesServed": False,
                "directoryListing": False,
            },
            "epistemicBoundary": _immutable_epistemic_boundary(),
        }


class LocalVisualizerHTTPServer(ThreadingHTTPServer):
    """Threaded local HTTP server carrying one validated data store."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        store: VisualizerDataStore,
        static_root: Path,
    ) -> None:
        self.store = store
        self.static_root = static_root.resolve(strict=True)
        super().__init__(server_address, LocalVisualizerRequestHandler)


class LocalVisualizerRequestHandler(BaseHTTPRequestHandler):
    """Exact-route handler; no URL is ever mapped directly to the filesystem."""

    protocol_version = "HTTP/1.1"
    server_version = "OPHLocalVisualizer/1.0"

    @property
    def app_server(self) -> LocalVisualizerHTTPServer:
        return self.server  # type: ignore[return-value]

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(head_only=False)

    def do_POST(self) -> None:  # noqa: N802
        self._json_error(HTTPStatus.METHOD_NOT_ALLOWED, "read_only_server")

    def do_PUT(self) -> None:  # noqa: N802
        self._json_error(HTTPStatus.METHOD_NOT_ALLOWED, "read_only_server")

    def do_DELETE(self) -> None:  # noqa: N802
        self._json_error(HTTPStatus.METHOD_NOT_ALLOWED, "read_only_server")

    def _dispatch(self, *, head_only: bool) -> None:
        if not _request_authority_is_loopback(
            host_header=self.headers.get("Host"),
            origin_header=self.headers.get("Origin"),
            expected_port=int(self.app_server.server_address[1]),
        ):
            self._json_error(
                HTTPStatus.MISDIRECTED_REQUEST,
                "loopback_host_or_origin_required",
                head_only=head_only,
            )
            return
        parsed = urlsplit(self.path)
        try:
            route = unquote(parsed.path, errors="strict")
        except UnicodeError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_url_encoding", head_only=head_only)
            return
        if "\x00" in route or ".." in route.split("/"):
            self._json_error(HTTPStatus.NOT_FOUND, "route_not_found", head_only=head_only)
            return
        if route in STATIC_FILES:
            self._serve_static(STATIC_FILES[route], head_only=head_only)
            return
        if route == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon", head_only=head_only)
            return
        if route == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "schema": API_SCHEMA,
                    "readOnly": True,
                    "loopback": _is_loopback_host(self.app_server.server_address[0]),
                },
                head_only=head_only,
            )
            return
        if route == "/api/manifest":
            self._send_json(
                HTTPStatus.OK,
                self.app_server.store.api_manifest(),
                head_only=head_only,
            )
            return
        if route == "/api/summary":
            self._send_json(
                HTTPStatus.OK,
                self.app_server.store.summary,
                head_only=head_only,
            )
            return
        for prefix, handler in (
            ("/api/files/", self._serve_file),
            ("/api/pages/", self._serve_page),
            ("/api/rows/", self._serve_rows),
        ):
            if route.startswith(prefix):
                file_id = route[len(prefix) :]
                if not file_id or "/" in file_id:
                    self._json_error(HTTPStatus.NOT_FOUND, "file_not_found", head_only=head_only)
                    return
                handler(file_id, parsed.query, head_only=head_only)
                return
        self._json_error(HTTPStatus.NOT_FOUND, "route_not_found", head_only=head_only)

    def _serve_static(self, name: str, *, head_only: bool) -> None:
        if name not in {"index.html", "app.js", "styles.css"}:
            self._json_error(HTTPStatus.NOT_FOUND, "asset_not_found", head_only=head_only)
            return
        path = self.app_server.static_root / name
        try:
            payload = path.read_bytes()
        except OSError:
            self._json_error(HTTPStatus.NOT_FOUND, "asset_not_found", head_only=head_only)
            return
        self._send_bytes(
            HTTPStatus.OK,
            payload,
            _media_type(path),
            head_only=head_only,
            extra_headers={"Cache-Control": "no-store"},
        )

    def _serve_file(self, file_id: str, _query: str, *, head_only: bool) -> None:
        record = self.app_server.store.file(file_id)
        if record is None:
            self._json_error(HTTPStatus.NOT_FOUND, "file_not_found", head_only=head_only)
            return
        range_header = self.headers.get("Range")
        if not range_header:
            if record.byte_count > MAX_DIRECT_FILE_BYTES and not head_only:
                self._json_error(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    "range_or_page_required_for_large_file",
                    head_only=head_only,
                    extra_headers={"Accept-Ranges": "bytes"},
                )
                return
            self._stream_file(
                record,
                start=0,
                length=record.byte_count,
                status=HTTPStatus.OK,
                head_only=head_only,
            )
            return
        parsed_range = _parse_range(range_header, record.byte_count)
        if parsed_range is None:
            self._json_error(
                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                "invalid_or_unsatisfied_range",
                head_only=head_only,
                extra_headers={"Content-Range": f"bytes */{record.byte_count}"},
            )
            return
        start, end = parsed_range
        self._stream_file(
            record,
            start=start,
            length=end - start + 1,
            status=HTTPStatus.PARTIAL_CONTENT,
            head_only=head_only,
            extra_headers={"Content-Range": f"bytes {start}-{end}/{record.byte_count}"},
        )

    def _serve_page(self, file_id: str, query: str, *, head_only: bool) -> None:
        record = self.app_server.store.file(file_id)
        if record is None:
            self._json_error(HTTPStatus.NOT_FOUND, "file_not_found", head_only=head_only)
            return
        params = parse_qs(query, keep_blank_values=True)
        try:
            page_size = _bounded_int(
                params.get("pageSize", [str(DEFAULT_PAGE_SIZE)])[0],
                minimum=1,
                maximum=MAX_PAGE_SIZE,
            )
            page = _bounded_int(params.get("page", ["0"])[0], minimum=0, maximum=2**53 - 1)
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_page_parameters", head_only=head_only)
            return
        start = page * page_size
        if start >= record.byte_count and record.byte_count != 0:
            self._json_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "page_out_of_range", head_only=head_only)
            return
        length = min(page_size, max(0, record.byte_count - start))
        self._stream_file(
            record,
            start=start,
            length=length,
            status=HTTPStatus.OK,
            head_only=head_only,
            extra_headers={
                "X-Page": str(page),
                "X-Page-Size": str(page_size),
                "X-Page-Offset": str(start),
                "X-Page-Returned": str(length),
                "X-Page-Complete": "true" if start + length >= record.byte_count else "false",
            },
        )

    def _serve_rows(self, file_id: str, query: str, *, head_only: bool) -> None:
        record = self.app_server.store.file(file_id)
        if record is None:
            self._json_error(HTTPStatus.NOT_FOUND, "file_not_found", head_only=head_only)
            return
        params = parse_qs(query, keep_blank_values=True)
        try:
            offset = _bounded_int(params.get("offset", ["0"])[0], minimum=0, maximum=2**53 - 1)
            limit = _bounded_int(params.get("limit", ["250"])[0], minimum=1, maximum=MAX_ROW_LIMIT)
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_row_parameters", head_only=head_only)
            return
        try:
            rows, has_more = _read_rows(record, offset=offset, limit=limit)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"rows_unavailable:{type(exc).__name__}", head_only=head_only)
            return
        response = {
            "schema": f"{API_SCHEMA}/rows",
            "fileId": file_id,
            "offset": offset,
            "limit": limit,
            "returned": len(rows),
            "hasMore": has_more,
            "rows": rows,
        }
        encoded = _json_bytes(response)
        if len(encoded) > MAX_ROW_RESPONSE_BYTES:
            self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "row_page_too_large", head_only=head_only)
            return
        self._send_bytes(
            HTTPStatus.OK,
            encoded,
            "application/json; charset=utf-8",
            head_only=head_only,
            extra_headers={"Cache-Control": "no-store"},
        )

    def _stream_file(
        self,
        record: ServedFile,
        *,
        start: int,
        length: int,
        status: HTTPStatus,
        head_only: bool,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        }
        headers.update(extra_headers or {})
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", record.media_type)
        self.send_header("Content-Length", str(length))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if head_only or length == 0:
            return
        end = start + length
        for offset in range(start, end, 64 * 1024):
            self.wfile.write(record.snapshot[offset : min(end, offset + 64 * 1024)])

    def _send_json(
        self,
        status: HTTPStatus,
        value: Any,
        *,
        head_only: bool,
    ) -> None:
        self._send_bytes(
            status,
            _json_bytes(value),
            "application/json; charset=utf-8",
            head_only=head_only,
            extra_headers={"Cache-Control": "no-store"},
        )

    def _json_error(
        self,
        status: HTTPStatus,
        code: str,
        *,
        head_only: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_bytes(
            status,
            _json_bytes({"error": code, "status": int(status)}),
            "application/json; charset=utf-8",
            head_only=head_only,
            extra_headers=extra_headers,
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        media_type: str,
        *,
        head_only: bool,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only and payload:
            self.wfile.write(payload)

    def _security_headers(self) -> None:
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def log_message(self, format_string: str, *args: object) -> None:
        # BaseHTTPRequestHandler's route/status log is useful; never append data paths.
        sys.stderr.write(f"[local-visualizer] {self.address_string()} {format_string % args}\n")


def make_server(
    payload_path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    data_root: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> LocalVisualizerHTTPServer:
    """Create, but do not start, one local visualizer server."""

    if not _is_loopback_host(host):
        raise UnsafeDataError("local_visualizer_requires_loopback_bind")
    store = VisualizerDataStore(
        payload_path,
        data_root=data_root,
        manifest_path=manifest_path,
    )
    static_root = Path(__file__).with_name("static")
    return LocalVisualizerHTTPServer((host, int(port)), store, static_root)


def _manifest_path_entries(files: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any], str]]:
    def walk(logical_name: str, value: Any) -> Iterable[tuple[str, Mapping[str, Any], str]]:
        if isinstance(value, Mapping):
            raw_path = value.get("path")
            if isinstance(raw_path, str) and raw_path:
                yield logical_name, value, raw_path
            for key, nested in value.items():
                if key in {"path", "source", "payload_path"}:
                    continue
                if isinstance(nested, (Mapping, list)):
                    yield from walk(f"{logical_name}.{key}", nested)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                yield from walk(f"{logical_name}.{index}", nested)

    for key, value in files.items():
        yield from walk(str(key), value)


def _public_manifest_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    key_map = {
        "row_count": "rowCount",
        "dtype": "dtype",
        "layout": "layout",
        "schema": "dataSchema",
        "frame_count": "frameCount",
        "field_name": "fieldName",
    }
    return {
        public_key: entry[source_key]
        for source_key, public_key in key_map.items()
        if source_key in entry and isinstance(entry[source_key], (str, int, float, bool))
    }


def _looks_sensitive_path(relative: Path) -> bool:
    for part in relative.parts:
        lower = part.lower()
        if lower.startswith(".") or lower in SENSITIVE_FILENAMES:
            return True
        if any(token in lower for token in ("credential", "private-key", "private_key", "secret")):
            return True
    return relative.suffix.lower() in SENSITIVE_SUFFIXES


def _find_sensitive_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalize_key(str(key))
            if _is_sensitive_key_name(normalized):
                return f"{path}.{key}"
            found = _find_sensitive_key(nested, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _find_sensitive_key(nested, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _public_payload_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create the immutable, redacted, fail-closed payload exposed to the UI."""

    copied = json.loads(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    if not isinstance(copied, dict):  # Defensive; the caller already checked this.
        raise UnsafeDataError("payload_root_must_be_object")
    _harden_physical_stage_statuses(copied)
    public = _redact_public_value(copied)
    if not isinstance(public, dict):
        raise UnsafeDataError("public_payload_copy_must_be_object")
    for ladder in _screen_a5_ladders(public):
        a5_to_sm = ladder.get("a5ToSm")
        if not isinstance(a5_to_sm, dict):
            continue
        snapshot = a5_to_sm.get("physicalReceiptSnapshot")
        if isinstance(snapshot, Mapping):
            a5_to_sm["physicalReceiptSnapshotPublicDigest"] = _canonical_digest(
                snapshot
            )
            a5_to_sm["physicalReceiptSnapshotPublicCopyRedacted"] = (
                _find_redaction_marker(snapshot)
            )
    return public


def _screen_a5_ladders(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    ladders: list[dict[str, Any]] = []
    if payload.get("schema") == "oph.screen-a5-visualization-ladder/1.0.0" and isinstance(
        payload, dict
    ):
        ladders.append(payload)
    embedded = payload.get("screenA5Ladder")
    if isinstance(embedded, dict) and embedded not in ladders:
        ladders.append(embedded)
    return ladders


def _harden_physical_stage_statuses(payload: dict[str, Any]) -> None:
    for ladder in _screen_a5_ladders(payload):
        a5_to_sm = ladder.get("a5ToSm")
        if not isinstance(a5_to_sm, dict):
            continue
        snapshot = a5_to_sm.get("physicalReceiptSnapshot")
        supplied_digest = a5_to_sm.get("physicalReceiptSnapshotDigest")
        digest_ok = bool(
            isinstance(snapshot, Mapping)
            and isinstance(supplied_digest, str)
            and supplied_digest == _canonical_digest(snapshot)
        )
        trusted = bool(
            a5_to_sm.get("physicalSnapshotTrusted") is True
            and digest_ok
            and isinstance(snapshot, Mapping)
            and snapshot.get("schema") == PHYSICAL_A5_REPORT_SCHEMA
            and snapshot.get("artifact_type") == PHYSICAL_A5_REPORT_ARTIFACT
        )
        a5_to_sm["physicalSnapshotTrusted"] = trusted
        blockers = a5_to_sm.get("physicalSnapshotBlockers")
        if not isinstance(blockers, list):
            blockers = []
            a5_to_sm["physicalSnapshotBlockers"] = blockers
        if not trusted and "local_visualizer_snapshot_validation_failed" not in blockers:
            blockers.append("local_visualizer_snapshot_validation_failed")
        snapshot_stages = (
            snapshot.get("stages", {})
            if trusted and isinstance(snapshot.get("stages"), Mapping)
            else {}
        )
        rows = a5_to_sm.get("stageNodes")
        stage_id_counts: dict[str, int] = {}
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and isinstance(row.get("stageId"), str):
                    stage_id = str(row["stageId"])
                    stage_id_counts[stage_id] = stage_id_counts.get(stage_id, 0) + 1
        verified_pass_ids: set[str] = set()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                stage_id = row.get("stageId")
                duplicate = (
                    not isinstance(stage_id, str)
                    or stage_id_counts.get(stage_id, 0) != 1
                )
                source_stage = (
                    snapshot_stages.get(stage_id, {})
                    if isinstance(stage_id, str)
                    and isinstance(snapshot_stages.get(stage_id), Mapping)
                    else {}
                )
                valid_pass = bool(
                    trusted
                    and not duplicate
                    and row.get("physicalPassed") is True
                    and row.get("physicalStatus") == "PASS"
                    and source_stage.get("passed") is True
                    and source_stage.get("status") == "PASS"
                )
                if valid_pass:
                    verified_pass_ids.add(stage_id)
                    continue
                row["physicalPassed"] = False
                if row.get("physicalStatus") == "PASS":
                    source_status = source_stage.get("status")
                    row["physicalStatus"] = (
                        source_status
                        if source_status in {"FAIL", "OPEN", "UNRESOLVED", "NOT_APPLICABLE"}
                        else "OPEN"
                    )
                if row.get("displayStatus") == "PASS":
                    row["displayStatus"] = row.get("physicalStatus", "OPEN")
                row["localVisualizerPhysicalStatusVerified"] = False
            for row in rows:
                if isinstance(row, dict) and row.get("physicalPassed") is True:
                    row["localVisualizerPhysicalStatusVerified"] = True
        receipts = ladder.get("receipts")
        if isinstance(receipts, dict):
            receipts["PHYSICAL_A5_SM_SNAPSHOT_TRUSTED"] = trusted
            snapshot_receipts = (
                snapshot.get("receipts", {})
                if trusted and isinstance(snapshot.get("receipts"), Mapping)
                else {}
            )
            stage_order = (
                snapshot.get("stage_order", [])
                if trusted and isinstance(snapshot.get("stage_order"), list)
                else []
            )
            global_pass = bool(
                trusted
                and snapshot.get("status") == "PASS"
                and snapshot.get("passed") is True
                and snapshot_receipts.get("PHYSICAL_A5_SM_GLOBAL_PASS") is True
                and stage_order
                and all(
                    isinstance(stage_id, str) and stage_id in verified_pass_ids
                    for stage_id in stage_order
                )
            )
            receipts["PHYSICAL_A5_SM_GLOBAL_PASS"] = global_pass


def _redact_public_value(value: Any, *, key: str | None = None) -> Any:
    normalized = _normalize_key(key) if key is not None else ""
    if normalized in PRIVATE_PATH_KEYS:
        return PUBLIC_REDACTION_MARKER
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_public_value(nested, key=str(child_key))
            for child_key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_public_value(nested) for nested in value]
    if isinstance(value, str) and _string_requires_redaction(value):
        return PUBLIC_REDACTION_MARKER
    return value


def _find_redaction_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_find_redaction_marker(nested) for nested in value.values())
    if isinstance(value, list):
        return any(_find_redaction_marker(nested) for nested in value)
    return value == PUBLIC_REDACTION_MARKER


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _normalize_key(value: str) -> str:
    snake_hint = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", snake_hint.lower()).strip("_")


def _string_requires_redaction(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower.startswith("file://") or stripped.startswith(("/", "~/")):
        return True
    if _WINDOWS_ABSOLUTE_RE.match(stripped) or _EMBEDDED_PRIVATE_PATH_RE.search(value):
        return True
    return any(pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS)


def _immutable_epistemic_boundary() -> dict[str, Any]:
    return {
        "modeDefault": "RECEIPT",
        "demoMode": "DEMO_ASSUMPTION",
        "promotion_allowed": False,
        "scientific_receipts_unchanged": True,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "target_ancestry_eligible": False,
        "demoWritesToSimulator": False,
        "frozenTargets": "post_exposure_display_only",
    }


def _build_summary(payload: Mapping[str, Any], files: Mapping[str, ServedFile]) -> dict[str, Any]:
    standalone_screen_a5 = payload.get("schema") == "oph.screen-a5-visualization-ladder/1.0.0"
    screen_a5 = payload if standalone_screen_a5 else payload.get("screenA5Ladder")
    screen = payload.get("screen") if isinstance(payload.get("screen"), Mapping) else {}
    observers = (
        payload.get("observerModularTime")
        if isinstance(payload.get("observerModularTime"), Mapping)
        else {}
    )
    consensus = payload.get("consensusBulk") if isinstance(payload.get("consensusBulk"), Mapping) else {}
    federation = (
        screen_a5.get("federation", {})
        if isinstance(screen_a5, Mapping) and isinstance(screen_a5.get("federation"), Mapping)
        else {}
    )
    demo_universe = (
        screen_a5.get("demoUniverse", {})
        if isinstance(screen_a5, Mapping) and isinstance(screen_a5.get("demoUniverse"), Mapping)
        else {}
    )
    finite_census = (
        demo_universe.get("finiteCensus", {})
        if isinstance(demo_universe.get("finiteCensus"), Mapping)
        else {}
    )
    segment_records = {
        str(row.get("segmentId")): [
            record for record in row.get("records", []) if isinstance(record, Mapping)
        ]
        for row in demo_universe.get("segments", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("segmentId"), str)
        and isinstance(row.get("records"), list)
    } if isinstance(demo_universe.get("segments"), list) else {}
    sm_records = segment_records.get("forced_sm_catalogue_and_interactions", [])
    particle_actor_count = sum(
        row.get("recordKind") == "particle_actor" for row in sm_records
    )
    particle_worldline_count = sum(
        row.get("recordKind") == "particle_worldline_sample" for row in sm_records
    )
    interaction_family_count = sum(
        row.get("recordKind") == "interaction_family" for row in sm_records
    )
    interaction_event_count = sum(
        row.get("recordKind") == "interaction_event" for row in sm_records
    )
    actual_literal_rows = sum(len(rows) for rows in segment_records.values())
    exported_carrier_samples = (
        len(federation.get("carrierInstances", []))
        if isinstance(federation.get("carrierInstances"), list)
        else 0
    )
    census = {
        "declaredCarrierCount": _nonnegative_int(
            finite_census.get("carrierCount", federation.get("declaredCarrierCount")),
            fallback=len(screen.get("points", [])) if isinstance(screen.get("points"), list) else 0,
        ),
        "exportedCarrierSampleCount": exported_carrier_samples,
        "loadedCarrierCount": exported_carrier_samples,
        "carrierPulseCount": _nonnegative_int(finite_census.get("carrierPulseCount"), fallback=0),
        "loadedCarrierPulseCount": len(
            segment_records.get("carrier_light_readback_settling", [])
        ),
        "screenPointCount": len(screen.get("points", [])) if isinstance(screen.get("points"), list) else 0,
        "observerCount": len(observers.get("observers", []))
        if isinstance(observers.get("observers"), list)
        else 0,
        "h3ObjectCount": _first_list_count(consensus, ("objects", "h3Objects", "recordObjects")),
        "particleActorCount": particle_actor_count,
        "particleWorldlineSampleCount": particle_worldline_count,
        "particleOrWorldlineCount": (
            particle_actor_count + particle_worldline_count
            if sm_records
            else _named_collection_count(payload, re.compile(r"particle|worldline", re.I))
        ),
        "interactionFamilyCount": interaction_family_count,
        "interactionEventCount": interaction_event_count,
        "atomCount": _nonnegative_int(
            finite_census.get("atomCount"),
            fallback=_named_collection_count(payload, re.compile(r"(^|_)atoms?$|atomCensus|atomicObjects|atomCount", re.I)),
        ),
        "loadedAtomCount": len(segment_records.get("finite_atom_census", [])),
        "literalDemoRowsLoaded": actual_literal_rows,
        "declaredLiteralDemoRows": _nonnegative_int(
            finite_census.get("literalRowsEmitted"), fallback=0
        ),
        "literalDemoRowCensusMatches": bool(
            _nonnegative_int(finite_census.get("literalRowsEmitted"), fallback=0)
            == actual_literal_rows
        ),
        "addressingContract": (
            "Every declared carrier and every exported census row has a stable index/ID. "
            "The browser renders only the requested visible window; it does not claim simultaneous literal rendering."
        ),
    }
    selected_sections: dict[str, Any] = {}
    section_names = (
        "screen",
        "smallUniverse",
        "observerModularTime",
        "subjectiveObserverCameras",
        "observerAnatomy",
        "consensusBulk",
        "emergentCurvedSpacetime",
        "assumedDs4Spacetime",
        "cmbComparison",
        "comparableObservations",
        "paperAccuracy",
        "visualizationRenderData",
        "visualizationViews",
    )
    for name in section_names:
        if name in payload:
            selected_sections[name] = _bounded_copy(payload[name], budget=[35_000], max_list=256)
    if isinstance(screen_a5, Mapping):
        # This contract contains the exact 12/30/20 carrier, 60 actions, and stage DAG;
        # it is intentionally retained whole (with sensitive-key redaction still applied).
        selected_sections["screenA5Ladder"] = _bounded_copy(screen_a5, budget=[250_000], max_list=2_500)
    return {
        "schema": f"{API_SCHEMA}/summary",
        "payloadSchema": payload.get("schemaVersion") or payload.get("schema"),
        "payloadVariant": "standalone_screen_a5_ladder" if standalone_screen_a5 else "universe_timeline",
        "title": payload.get("title") or (
            "OPH screen/A5 finite demo visualizer"
            if standalone_screen_a5
            else "OPH finite observer-patch visualizer"
        ),
        "claimBoundary": payload.get("claimBoundary"),
        "ophDifferentiator": payload.get("ophDifferentiator"),
        "sections": selected_sections,
        "census": census,
        "fileCount": len(files),
        "epistemicBoundary": _immutable_epistemic_boundary(),
    }


def _bounded_copy(value: Any, *, budget: list[int], max_list: int, depth: int = 0) -> Any:
    if budget[0] <= 0:
        return {"truncated": True, "reason": "summary_budget"}
    budget[0] -= 1
    if depth > 14:
        return {"truncated": True, "reason": "summary_depth"}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            normalized = _normalize_key(str(key))
            if normalized in SENSITIVE_KEYS or normalized in PRIVATE_PATH_KEYS:
                result[str(key)] = PUBLIC_REDACTION_MARKER
            else:
                result[str(key)] = _bounded_copy(
                    nested,
                    budget=budget,
                    max_list=max_list,
                    depth=depth + 1,
                )
            if budget[0] <= 0:
                result["_truncated"] = True
                break
        return result
    if isinstance(value, list):
        copied = [
            _bounded_copy(item, budget=budget, max_list=max_list, depth=depth + 1)
            for item in value[:max_list]
        ]
        if len(value) > max_list:
            copied.append({"truncated": True, "remainingItems": len(value) - max_list})
        return copied
    if isinstance(value, str):
        return PUBLIC_REDACTION_MARKER if _string_requires_redaction(value) else value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _named_collection_count(value: Any, pattern: re.Pattern[str], depth: int = 0) -> int:
    if depth > 10:
        return 0
    best = 0
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if pattern.search(str(key)):
                if isinstance(nested, list):
                    best = max(best, len(nested))
                elif isinstance(nested, Mapping):
                    for count_key in ("count", "totalCount", "declaredCount", "rowCount"):
                        best = max(best, _nonnegative_int(nested.get(count_key), fallback=0))
            best = max(best, _named_collection_count(nested, pattern, depth + 1))
    elif isinstance(value, list):
        for nested in value[:128]:
            best = max(best, _named_collection_count(nested, pattern, depth + 1))
    return best


def _first_list_count(value: Mapping[str, Any], keys: Iterable[str]) -> int:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, list):
            return len(candidate)
    return 0


def _nonnegative_int(value: Any, *, fallback: int) -> int:
    return int(value) if type(value) is int and value >= 0 else int(fallback)


def _read_rows(record: ServedFile, *, offset: int, limit: int) -> tuple[list[Any], bool]:
    suffix = record.path.suffix.lower()
    text = record.snapshot.decode("utf-8")
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = _slice_iterable(reader, offset=offset, count=limit + 1)
    elif suffix == ".jsonl":
        parsed = (json.loads(line) for line in text.splitlines() if line.strip())
        rows = _slice_iterable(parsed, offset=offset, count=limit + 1)
    elif suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("json_rows_require_top_level_array")
        rows = data[offset : offset + limit + 1]
    else:
        raise ValueError("unsupported_row_file")
    sensitive = _find_sensitive_key(rows)
    if sensitive is not None:
        raise ValueError("row_page_contains_sensitive_key")
    return rows[:limit], len(rows) > limit


def _is_sensitive_key_name(normalized: str) -> bool:
    if normalized in SENSITIVE_KEYS:
        return True
    sensitive_fragments = (
        "password",
        "passwd",
        "credential",
        "api_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "private_key",
        "secret_key",
        "client_secret",
        "session_token",
        "authorization",
    )
    squashed = normalized.replace("_", "")
    return any(
        fragment in normalized or fragment.replace("_", "") in squashed
        for fragment in sensitive_fragments
    )


def _safe_sidecar_snapshot(path: Path, *, logical_name: str) -> bytes:
    """Return one fully scanned immutable text snapshot or reject the file.

    Raw binary sidecars are deliberately not admitted here.  A filename or
    manifest label cannot prove that opaque bytes are credential-free, and a
    local browser should not deserialize NPY/NPZ object payloads.  A future
    typed binary endpoint can restore those formats after validating their
    complete descriptor and element encoding.
    """

    suffix = path.suffix.lower()
    if suffix not in SNAPSHOTTED_SIDECAR_SUFFIXES:
        raise UnsafeDataError("opaque_binary_sidecar_not_admitted")
    normalized_name = _normalize_key(logical_name)
    if (
        _is_sensitive_key_name(normalized_name)
        or normalized_name in PRIVATE_PATH_KEYS
        or _string_requires_redaction(logical_name)
    ):
        raise UnsafeDataError("sensitive_sidecar_logical_name")
    maximum = (
        MAX_JSON_SIDECAR_SCAN_BYTES
        if suffix == ".json"
        else MAX_TEXT_SIDECAR_SCAN_BYTES
    )
    if path.stat().st_size > maximum:
        raise UnsafeDataError("sidecar_too_large_for_complete_scan")
    raw = _read_stable_bytes(path)
    try:
        text = raw.decode("utf-8")
        if suffix == ".json":
            value = _strict_json_loads(text)
            _assert_public_structured_content(value)
        elif suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = _strict_json_loads(line)
                    _assert_public_structured_content(value)
                except UnsafeDataError as exc:
                    raise UnsafeDataError(
                        f"unsafe_jsonl_line:{line_number}:{exc}"
                    ) from exc
        else:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None:
                raise UnsafeDataError("csv_header_missing")
            for field in reader.fieldnames:
                normalized = _normalize_key(str(field))
                if _is_sensitive_key_name(normalized) or normalized in PRIVATE_PATH_KEYS:
                    raise UnsafeDataError("csv_sensitive_header")
            for row_number, row in enumerate(reader, start=2):
                for field, value in row.items():
                    values = value if isinstance(value, list) else [value]
                    if field is None:
                        raise UnsafeDataError(f"csv_extra_columns:row_{row_number}")
                    for cell in values:
                        if isinstance(cell, str) and _string_requires_redaction(cell):
                            raise UnsafeDataError(
                                f"csv_sensitive_value:{field}:row_{row_number}"
                            )
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise UnsafeDataError("sidecar_invalid_utf8_or_structure") from exc
    return raw


def _verify_declared_integrity(raw: bytes, *, entry: Mapping[str, Any]) -> None:
    declared_bytes = None
    for key in ("byte_count", "byteCount"):
        if key in entry:
            value = entry[key]
            if type(value) is not int or value < 0:
                raise UnsafeDataError("manifest_byte_count_invalid")
            declared_bytes = value
            break
    if declared_bytes is not None and declared_bytes != len(raw):
        raise UnsafeDataError("manifest_byte_count_mismatch")
    if "sha256" in entry:
        declared_hash = entry["sha256"]
        if not isinstance(declared_hash, str) or not _SHA256_RE.fullmatch(
            declared_hash.lower()
        ):
            raise UnsafeDataError("manifest_sha256_invalid")
        if hashlib.sha256(raw).hexdigest() != declared_hash.lower():
            raise UnsafeDataError("manifest_sha256_mismatch")


def _read_stable_bytes(path: Path) -> bytes:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise UnsafeDataError("served_file_unreadable") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        raise UnsafeDataError("served_file_changed_while_snapshotting")
    return raw


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise UnsafeDataError(f"nonfinite_json_value:{value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise UnsafeDataError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _assert_public_structured_content(value: Any) -> None:
    sensitive = _find_sensitive_key(value)
    if sensitive is not None:
        raise UnsafeDataError(f"sidecar_contains_sensitive_key:{sensitive}")
    private = _find_private_or_secret_value(value)
    if private is not None:
        raise UnsafeDataError(f"sidecar_contains_private_value:{private}")


def _find_private_or_secret_value(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalize_key(str(key)) in PRIVATE_PATH_KEYS:
                return f"{path}.{key}"
            found = _find_private_or_secret_value(nested, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _find_private_or_secret_value(nested, f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str) and _string_requires_redaction(value):
        return path
    return None


def _slice_iterable(rows: Iterable[Any], *, offset: int, count: int) -> list[Any]:
    selected: list[Any] = []
    for index, row in enumerate(rows):
        if index < offset:
            continue
        if len(selected) >= count:
            break
        selected.append(row)
    return selected


def _parse_range(value: str, size: int) -> tuple[int, int] | None:
    if not value.startswith("bytes=") or "," in value or size < 0:
        return None
    spec = value[6:].strip()
    if "-" not in spec:
        return None
    start_text, end_text = spec.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0 or size == 0:
                return None
            start = max(0, size - suffix_length)
            return start, size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= size:
        return None
    return start, min(end, size - 1)


def _bounded_int(value: str, *, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(value)
    return parsed


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _media_type(path: Path) -> str:
    overrides = {
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".bin": "application/octet-stream",
        ".npy": "application/octet-stream",
        ".npz": "application/octet-stream",
    }
    return overrides.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _is_loopback_host(host: str) -> bool:
    normalized = host.lower().rstrip(".")
    if normalized in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _request_authority_is_loopback(
    *,
    host_header: str | None,
    origin_header: str | None,
    expected_port: int,
) -> bool:
    """Reject DNS-rebinding authorities and cross-origin browser requests."""

    host = _parse_loopback_authority(host_header)
    if host is None:
        return False
    _hostname, port = host
    if port is not None and port != expected_port:
        return False
    if expected_port not in {80, 443} and port is None:
        return False
    if origin_header is None:
        return True
    try:
        origin = urlsplit(origin_header)
        if origin.scheme != "http" or origin.username is not None or origin.password is not None:
            return False
        origin_host = (origin.hostname or "").lower().rstrip(".")
        origin_port = origin.port
    except ValueError:
        return False
    if not _is_loopback_host(origin_host) or origin_port != expected_port:
        return False
    return origin.path in {"", "/"} and not origin.query and not origin.fragment


def _parse_loopback_authority(value: str | None) -> tuple[str, int | None] | None:
    if value is None or not value.strip() or any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(f"//{value}")
        if parsed.username is not None or parsed.password is not None:
            return None
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return None
    if not _is_loopback_host(hostname) or parsed.path or parsed.query or parsed.fragment:
        return None
    return hostname, port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an OPH visualization payload locally")
    parser.add_argument("--payload", required=True, type=Path, help="visualization_payload.json")
    parser.add_argument("--data-root", type=Path, help="containment root; defaults to payload directory")
    parser.add_argument("--manifest", type=Path, help="optional visualization_export_manifest.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="deprecated safety flag; remote binds are no longer admitted",
    )
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _is_loopback_host(args.host):
        raise SystemExit(
            "Refusing non-loopback bind: this data browser is loopback-only; "
            "--allow-remote is intentionally non-operative"
        )
    server = make_server(
        args.payload,
        host=args.host,
        port=args.port,
        data_root=args.data_root,
        manifest_path=args.manifest,
    )
    host, port = server.server_address[:2]
    url_host = "[::1]" if host == "::1" else host
    url = f"http://{url_host}:{port}/"
    print(f"OPH local visualizer: {url}")
    print("Read-only; demo controls never write receipts or launch campaigns.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local visualizer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
