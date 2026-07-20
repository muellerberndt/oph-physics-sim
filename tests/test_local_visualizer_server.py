from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
from typing import Iterator

import pytest

from oph_fpe.viz.screen_a5_ladder import build_screen_a5_ladder_payload
from tools.local_visualizer.server import UnsafeDataError, VisualizerDataStore, make_server


def _ladder_payload(*, carrier_count: int = 4096, atom_count: int = 8192) -> dict:
    return {
        "schema": "oph.screen-a5-visualization-ladder/1.0.0",
        "epistemicStatus": "VISUALIZATION_CONTRACT",
        "localCarrier": {
            "counts": {"portCount": 12, "edgeCount": 30, "faceCount": 20},
            "ports": [],
            "edges": [],
            "faces": [],
            "a5": {"order": 60, "actions": [], "sectors": []},
        },
        "federation": {
            "declaredCarrierCount": carrier_count,
            "carrierInstances": [{"carrierId": "carrier-000000"}],
            "seams": [],
        },
        "observerRepairBridge": {"steps": []},
        "a5ToSm": {
            "stageNodes": [
                {
                    "stageId": "ROOT_IMMUTABLE_PACKET",
                    "physicalStatus": "OPEN",
                    "physicalPassed": False,
                    "claimBoundary": "inventory is not a physical pass",
                }
            ],
            "stageEdges": [],
        },
        "clockSeparation": {},
        "demoControls": {"enabled": True, "forceAllStages": True, "toggleCatalog": []},
        "demoUniverse": {
            "finiteCensus": {
                "carrierCount": carrier_count,
                "carrierPulseCount": carrier_count * 12,
                "atomCount": atom_count,
                "literalRowsEmitted": 48,
            },
            "segments": [
                {
                    "segmentId": "carrier_light_readback_settling",
                    "explicitRecordCount": 12,
                    "records": [{"recordId": f"pulse-{index}"} for index in range(12)],
                },
                {
                    "segmentId": "finite_atom_census",
                    "explicitRecordCount": 32,
                    "records": [{"recordId": f"atom-{index}"} for index in range(32)],
                },
                {
                    "segmentId": "forced_sm_catalogue_and_interactions",
                    "explicitRecordCount": 4,
                    "records": [
                        {"recordKind": "particle_actor", "recordId": "actor-0"},
                        {"recordKind": "particle_worldline_sample", "recordId": "line-0"},
                        {"recordKind": "particle_worldline_sample", "recordId": "line-1"},
                        {"recordKind": "interaction_event", "recordId": "event-0"},
                    ],
                },
            ],
        },
        "receipts": {
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
        "claimBoundary": "Demo rows are display assumptions, not evidence.",
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@contextmanager
def _running_server(payload: Path, **kwargs: object) -> Iterator[tuple[str, int]]:
    server = make_server(payload, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _request(
    address: tuple[str, int],
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection(*address, timeout=5)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    result = response.status, {key.lower(): value for key, value in response.getheaders()}, body
    connection.close()
    return result


def test_standalone_ladder_summary_prefers_exact_demo_census(tmp_path: Path) -> None:
    payload = _write_json(tmp_path / "screen_a5.json", _ladder_payload())
    store = VisualizerDataStore(payload)

    assert store.summary["payloadVariant"] == "standalone_screen_a5_ladder"
    assert store.summary["sections"]["screenA5Ladder"]["schema"] == (
        "oph.screen-a5-visualization-ladder/1.0.0"
    )
    census = store.summary["census"]
    assert census["declaredCarrierCount"] == 4096
    assert census["carrierPulseCount"] == 4096 * 12
    assert census["atomCount"] == 8192
    assert census["loadedCarrierCount"] == 1
    assert census["loadedCarrierPulseCount"] == 12
    assert census["loadedAtomCount"] == 32
    assert census["particleActorCount"] == 1
    assert census["particleWorldlineSampleCount"] == 2
    assert census["particleOrWorldlineCount"] == 3
    assert census["interactionEventCount"] == 1
    assert census["literalDemoRowsLoaded"] == 48
    assert census["literalDemoRowCensusMatches"] is True
    assert store.summary["epistemicBoundary"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_authoritative_standalone_builder_is_summarized_without_contract_drift(
    tmp_path: Path,
) -> None:
    value = build_screen_a5_ladder_payload(
        demo_config={"enabled": True, "forceAllStages": True},
        federation_carrier_count=64,
    )
    store = VisualizerDataStore(_write_json(tmp_path / "screen_a5.json", value))

    ladder = store.summary["sections"]["screenA5Ladder"]
    assert len(ladder["localCarrier"]["ports"]) == 12
    assert len(ladder["localCarrier"]["edges"]) == 30
    assert len(ladder["localCarrier"]["faces"]) == 20
    assert len(ladder["localCarrier"]["a5"]["actions"]) == 60
    assert store.summary["census"]["declaredCarrierCount"] == 64
    assert store.summary["census"]["h3ObjectCount"] == 0
    assert store.summary["census"]["literalDemoRowCensusMatches"] is True


def test_manifest_range_page_rows_and_read_only_endpoints(tmp_path: Path) -> None:
    payload = _write_json(tmp_path / "visualization_payload.json", {
        "schema": "oph_universe_timeline_visualization_payload_v1",
        "screenA5Ladder": _ladder_payload(carrier_count=12, atom_count=60),
        "claimBoundary": "test boundary",
    })
    csv_path = tmp_path / "screen_points.csv"
    csv_bytes = b"index,x,y\n0,1,2\n1,3,4\n2,5,6\n"
    csv_path.write_bytes(csv_bytes)
    _write_json(tmp_path / "visualization_export_manifest.json", {
        "schema": "oph_universe_visualization_sidecars_v1",
        "files": {
                "screen_points_csv": {
                    "path": str(csv_path),
                    "written": True,
                    "row_count": 3,
                    "byte_count": len(csv_bytes),
                    "sha256": hashlib.sha256(csv_bytes).hexdigest(),
                }
        },
    })

    with _running_server(payload) as address:
        status, headers, body = _request(address, "/api/health")
        assert status == 200
        assert json.loads(body)["readOnly"] is True
        assert "default-src 'self'" in headers["content-security-policy"]

        status, _, body = _request(address, "/api/manifest")
        assert status == 200
        manifest = json.loads(body)
        assert manifest["epistemicBoundary"]["promotion_allowed"] is False
        assert manifest["epistemicBoundary"]["SCALE_CAMPAIGN_ALLOWED"] is False
        csv_entry = next(row for row in manifest["files"] if row["logicalName"] == "screen_points_csv")
        assert csv_entry["sha256"] == hashlib.sha256(csv_bytes).hexdigest()

        assert _request(
            address,
            "/api/summary",
            headers={"Host": "attacker.invalid"},
        )[0] == 421
        loopback_host = f"{address[0]}:{address[1]}"
        assert _request(
            address,
            "/api/summary",
            headers={"Host": loopback_host, "Origin": "https://attacker.invalid"},
        )[0] == 421
        assert _request(
            address,
            "/api/summary",
            headers={"Host": loopback_host, "Origin": f"http://{loopback_host}"},
        )[0] == 200

        status, headers, body = _request(
            address,
            csv_entry["rangeEndpoint"],
            headers={"Range": "bytes=6-14"},
        )
        assert status == 206
        assert headers["content-range"] == f"bytes 6-14/{len(csv_bytes)}"
        assert body == csv_bytes[6:15]

        status, headers, body = _request(address, f"{csv_entry['pageEndpoint']}?page=1&pageSize=7")
        assert status == 200
        assert headers["x-page-offset"] == "7"
        assert body == csv_bytes[7:14]
        assert _request(address, f"{csv_entry['pageEndpoint']}?page=0&pageSize=99999999")[0] == 400

        status, _, body = _request(address, f"{csv_entry['rowsEndpoint']}?offset=1&limit=1")
        assert status == 200
        rows = json.loads(body)
        assert rows["returned"] == 1
        assert rows["hasMore"] is True
        assert rows["rows"] == [{"index": "1", "x": "3", "y": "4"}]

        assert _request(address, "/api/manifest", method="POST")[0] == 405


def test_static_frontend_and_traversal_are_exact_routed(tmp_path: Path) -> None:
    payload = _write_json(tmp_path / "payload.json", _ladder_payload())
    with _running_server(payload) as address:
        status, _, body = _request(address, "/")
        assert status == 200
        assert b"Receipt mode" in body
        assert b"DEMO_ASSUMPTION" in body
        assert b"SCALE CAMPAIGN: LOCKED FALSE" in body
        status, _, javascript = _request(address, "/static/app.js")
        assert status == 200
        assert b"drawCarrierReadbackPulses" in javascript
        assert b"drawRepairResidual" in javascript
        assert b"drawDemoWorldlineActor" in javascript
        assert b"visibilitychange" in javascript
        assert b"prefers-reduced-motion" in javascript
        assert b"SPIN_EXCHANGE_314" in javascript
        assert b"REFINEMENT_COMPLETENESS" in javascript
        assert b"PHYSICAL_A5_SM_SNAPSHOT_TRUSTED" in javascript
        assert b'physicalPassed === true || stage?.physicalStatus === "PASS"' not in javascript
        assert b"physicalStatusLabel" in javascript
        assert b"clockData.candidates" in javascript
        assert b"particle_worldline_sample" in javascript
        assert b"traceability" in javascript
        assert b"sourceInteractionEventIds" in javascript
        assert b"ports.length !== 12 || edges.length !== 30" in javascript
        assert b"resolveProceduralCensusRecord" in javascript
        assert b"deterministicAtomSample" in javascript
        assert b"drawVisibleFederationSeams" in javascript
        assert b"drawA5ActionNavigator" in javascript
        assert b"requested === \"demo\"" in javascript
        assert _request(address, "/static/styles.css")[0] == 200
        assert _request(address, "/static/../server.py")[0] == 404
        assert _request(address, "/api/files/../../payload")[0] == 404
        assert _request(address, "/api/files/not-present")[0] == 404


def test_payload_and_manifested_text_sidecars_reject_sensitive_keys(tmp_path: Path) -> None:
    unsafe_payload = _write_json(tmp_path / "unsafe.json", {
        "schema": "test",
        "nested": {"clientSecret": "do-not-serve"},
    })
    with pytest.raises(UnsafeDataError, match="sensitive_key"):
        VisualizerDataStore(unsafe_payload)

    payload = _write_json(tmp_path / "payload.json", _ladder_payload())
    sensitive_json = _write_json(tmp_path / "visualization_rows.json", [
        {"observer": 1, "access_token": "do-not-serve"}
    ])
    sensitive_csv = tmp_path / "screen_secret_rows.csv"
    sensitive_csv.write_text("index,user_password\n0,do-not-serve\n", encoding="utf-8")
    sensitive_jsonl = tmp_path / "observer_rows.jsonl"
    sensitive_jsonl.write_text('{"index":0,"api_key":"do-not-serve"}\n', encoding="utf-8")
    _write_json(tmp_path / "visualization_export_manifest.json", {
        "schema": "oph_universe_visualization_sidecars_v1",
        "files": {
            "visualization_rows_json": {"path": str(sensitive_json), "written": True},
            "screen_secret_rows_csv": {"path": str(sensitive_csv), "written": True},
            "observer_rows_jsonl": {"path": str(sensitive_jsonl), "written": True},
        },
    })

    store = VisualizerDataStore(payload)
    assert list(store.files) == ["payload"]
    assert store.manifest_status == "accepted:0:rejected:3"
    serialized_manifest = json.dumps(store.api_manifest())
    assert "do-not-serve" not in serialized_manifest


def test_outside_paths_and_intermediary_symlink_escape_are_not_served(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = _write_json(root / "payload.json", _ladder_payload())
    outside_csv = outside / "screen_points.csv"
    outside_csv.write_text("index,value\n0,7\n", encoding="utf-8")
    link = root / "linked-outside"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    _write_json(root / "visualization_export_manifest.json", {
        "schema": "oph_universe_visualization_sidecars_v1",
        "files": {
            "direct_escape": {"path": str(outside_csv), "written": True},
            "symlink_escape": {"path": str(link / "screen_points.csv"), "written": True},
        },
    })

    store = VisualizerDataStore(payload, data_root=root)
    assert list(store.files) == ["payload"]
    assert store.manifest_status == "accepted:0:rejected:2"


def test_unrecognized_manifest_never_becomes_a_file_allowlist(tmp_path: Path) -> None:
    payload = _write_json(tmp_path / "payload.json", _ladder_payload())
    sidecar = tmp_path / "screen_points.csv"
    sidecar.write_text("index,value\n0,1\n", encoding="utf-8")
    _write_json(tmp_path / "visualization_export_manifest.json", {
        "schema": "some_other_manifest",
        "files": {"screen_points": {"path": str(sidecar), "written": True}},
    })

    store = VisualizerDataStore(payload)
    assert list(store.files) == ["payload"]
    assert store.manifest_status == "unrecognized_schema"


def test_public_payload_redacts_paths_and_fails_closed_on_forged_stage_pass(
    tmp_path: Path,
) -> None:
    value = _ladder_payload()
    value["sourcePaths"] = {"run": "/Users/person/private/run"}
    value["a5ToSm"]["stageNodes"][0].update(
        {
            "physicalStatus": "PASS",
            "physicalPassed": False,
            "displayStatus": "PASS",
        }
    )
    value["a5ToSm"]["physicalReceiptSnapshot"] = {
        "schema": "forged",
        "artifact_type": "forged",
        "stages": {},
        "manifest_path": "/private/tmp/receipt.json",
    }
    value["a5ToSm"]["physicalReceiptSnapshotDigest"] = "sha256:" + "0" * 64
    value["a5ToSm"]["physicalSnapshotTrusted"] = True
    payload = _write_json(tmp_path / "payload.json", value)

    store = VisualizerDataStore(payload)
    public = store.summary["sections"]["screenA5Ladder"]
    stage = public["a5ToSm"]["stageNodes"][0]
    assert stage["physicalPassed"] is False
    assert stage["physicalStatus"] == "OPEN"
    assert stage["displayStatus"] == "OPEN"
    assert stage["localVisualizerPhysicalStatusVerified"] is False
    assert public["a5ToSm"]["physicalSnapshotTrusted"] is False
    assert public["sourcePaths"] == "[REDACTED_BY_LOCAL_VISUALIZER]"
    assert public["a5ToSm"]["physicalReceiptSnapshot"]["manifest_path"] == (
        "[REDACTED_BY_LOCAL_VISUALIZER]"
    )
    assert b"/Users/" not in store.file("payload").snapshot  # type: ignore[union-attr]
    assert b"/private/" not in store.file("payload").snapshot  # type: ignore[union-attr]


def test_stage_pass_survives_only_when_node_and_embedded_snapshot_agree(
    tmp_path: Path,
) -> None:
    value = _ladder_payload()
    stage_id = value["a5ToSm"]["stageNodes"][0]["stageId"]
    snapshot = {
        "schema": "oph.physical-a5-sm.requirements-audit/1.0.0",
        "artifact_type": "OPH_PHYSICAL_A5_SM_REQUIREMENTS_AUDIT",
        "stages": {stage_id: {"status": "PASS", "passed": True}},
    }
    digest_bytes = json.dumps(
        snapshot,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value["a5ToSm"].update(
        {
            "physicalReceiptSnapshot": snapshot,
            "physicalReceiptSnapshotDigest": (
                "sha256:" + hashlib.sha256(digest_bytes).hexdigest()
            ),
            "physicalSnapshotTrusted": True,
            "physicalSnapshotBlockers": [],
        }
    )
    value["a5ToSm"]["stageNodes"][0].update(
        {"physicalStatus": "PASS", "physicalPassed": True, "displayStatus": "PASS"}
    )

    store = VisualizerDataStore(_write_json(tmp_path / "payload.json", value))
    stage = store.summary["sections"]["screenA5Ladder"]["a5ToSm"]["stageNodes"][0]
    assert stage["physicalStatus"] == "PASS"
    assert stage["physicalPassed"] is True
    assert stage["localVisualizerPhysicalStatusVerified"] is True


def test_actual_records_override_forged_explicit_record_counts(tmp_path: Path) -> None:
    value = _ladder_payload()
    value["demoUniverse"]["segments"][0]["explicitRecordCount"] = 999_999
    value["demoUniverse"]["segments"][1]["explicitRecordCount"] = 999_999
    value["demoUniverse"]["finiteCensus"]["literalRowsEmitted"] = 7

    census = VisualizerDataStore(
        _write_json(tmp_path / "payload.json", value)
    ).summary["census"]
    assert census["loadedCarrierPulseCount"] == 12
    assert census["loadedAtomCount"] == 32
    assert census["literalDemoRowsLoaded"] == 48
    assert census["declaredLiteralDemoRows"] == 7
    assert census["literalDemoRowCensusMatches"] is False


def test_manifest_integrity_full_scan_binary_rejection_and_immutable_snapshot(
    tmp_path: Path,
) -> None:
    payload = _write_json(tmp_path / "payload.json", _ladder_payload())
    safe = tmp_path / "screen_rows.csv"
    safe_bytes = b"index,value\n0,1\n"
    safe.write_bytes(safe_bytes)
    bad_hash = tmp_path / "bad_hash.csv"
    bad_hash.write_bytes(safe_bytes)
    bad_size = tmp_path / "bad_size.csv"
    bad_size.write_bytes(safe_bytes)
    late_secret = tmp_path / "late_rows.jsonl"
    late_secret.write_text(
        "".join('{"index":%d}\n' % index for index in range(100))
        + '{"index":100,"api_key":"must-not-serve"}\n',
        encoding="utf-8",
    )
    private_value = tmp_path / "private_value.csv"
    private_value.write_text("index,note\n0,/Users/person/private/run\n", encoding="utf-8")
    opaque = tmp_path / "screen_visualization.bin"
    opaque.write_bytes(b"opaque")
    _write_json(
        tmp_path / "visualization_export_manifest.json",
        {
            "schema": "oph_universe_visualization_sidecars_v1",
            "files": {
                "screen_rows": {
                    "path": str(safe),
                    "written": True,
                    "byte_count": len(safe_bytes),
                    "sha256": hashlib.sha256(safe_bytes).hexdigest(),
                },
                "hash_mismatch": {
                    "path": str(bad_hash),
                    "written": True,
                    "sha256": "0" * 64,
                },
                "size_mismatch": {
                    "path": str(bad_size),
                    "written": True,
                    "byte_count": len(safe_bytes) + 1,
                },
                "late_sensitive_jsonl": {"path": str(late_secret), "written": True},
                "private_value_csv": {"path": str(private_value), "written": True},
                "screen_visualization": {"path": str(opaque), "written": True},
            },
        },
    )

    store = VisualizerDataStore(payload)
    assert store.manifest_status == "accepted:1:rejected:5"
    served = next(
        row for row in store.files.values() if row.logical_name == "screen_rows"
    )
    safe.write_bytes(b"index,value\n0,999\n")
    assert served.snapshot == safe_bytes
    assert served.sha256 == hashlib.sha256(safe_bytes).hexdigest()
    assert all(row.logical_name != "screen_visualization" for row in store.files.values())


def test_non_loopback_bind_is_rejected_before_socket_creation(tmp_path: Path) -> None:
    payload = _write_json(tmp_path / "payload.json", _ladder_payload())
    with pytest.raises(UnsafeDataError, match="requires_loopback_bind"):
        make_server(payload, host="0.0.0.0", port=0)
