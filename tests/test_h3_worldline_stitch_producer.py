from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.bulk.h3_worldline_stitch import (
    H3_PRIMITIVE_SCHEMA,
    H3_PRIMITIVE_SOURCE_KIND,
    h3_worldline_stitch_certificate_report,
)
from oph_fpe.bulk.h3_worldline_stitch_producer import (
    MEASUREMENT_LANE_BLOCKER,
    MISSING_INPUTS_REPORT_NAME,
    PRIMITIVES_ARTIFACT_NAME,
    h3_worldline_stitch_producer_report,
    write_h3_worldline_stitch_producer_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _monolithic_run(root: Path, name: str, *, with_support_ids: bool = False) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    event: dict[str, object] = {
        "cycle": 0,
        "h3_spatial_point": [0.1, 0.2, 0.3],
        "fit_residual": 0.5,
        "support_node_count": 3,
    }
    if with_support_ids:
        event["support_nodes"] = [1, 2, 3]
    _write_json(
        run / "defect_h3_worldlines_report.json",
        {"worldlines": [{"worldline_id": 0, "events": [event]}]},
    )
    (run / "s3_gauge_state.npz").write_bytes(b"\x00")
    return run


def test_producer_refuses_when_run_dirs_are_absent(tmp_path: Path) -> None:
    report = write_h3_worldline_stitch_producer_report(
        tmp_path / "missing_left",
        tmp_path / "missing_right",
        tmp_path / "out",
    )

    assert report["artifact_emitted"] is False
    assert report["producer_fail_closed"] is True
    assert "distinct_run_dirs" in report["missing_inputs"]
    assert "left_worldline_catalog" in report["missing_inputs"]
    assert (tmp_path / "out" / MISSING_INPUTS_REPORT_NAME).exists()
    # Fail-closed: no primitives artifact anywhere.
    assert not list(tmp_path.rglob(PRIMITIVES_ARTIFACT_NAME))


def test_producer_reports_exact_gaps_for_monolithic_run_data(tmp_path: Path) -> None:
    left = _monolithic_run(tmp_path, "run_left")
    right = _monolithic_run(tmp_path, "run_right")

    report = h3_worldline_stitch_producer_report(left, right)

    assert report["artifact_emitted"] is False
    inputs = report["inputs"]
    # What the run data genuinely carries is recorded as present.
    assert inputs["distinct_run_dirs"]["present"] is True
    assert inputs["left_worldline_catalog"]["present"] is True
    assert inputs["right_gauge_state"]["present"] is True
    # The cross-shard measurement gaps are named, one by one.
    assert inputs["left_worldline_event_support_node_ids"]["present"] is False
    assert inputs["left_seam_interface_crossings"]["present"] is False
    assert inputs["cross_shard_manifest"]["present"] is False
    assert inputs["seam_adjacency_for_run_pair"]["present"] is False
    assert inputs["coarse_fine_refinement_pair"]["present"] is False
    assert set(report["missing_inputs"]) == {
        "left_worldline_event_support_node_ids",
        "right_worldline_event_support_node_ids",
        "left_seam_interface_crossings",
        "right_seam_interface_crossings",
        "cross_shard_manifest",
        "seam_adjacency_for_run_pair",
        "coarse_fine_refinement_pair",
    }
    assert report["target_schema"] == H3_PRIMITIVE_SCHEMA
    assert report["target_source_kind"] == H3_PRIMITIVE_SOURCE_KIND


def test_producer_never_emits_primitives_even_for_complete_looking_inputs(
    tmp_path: Path,
) -> None:
    # Negative control: file layout shaped like complete inputs.  These files
    # are synthetic and must not become a primitives artifact; the producer has
    # no measurement lane and must keep refusing with an explicit blocker.
    left = _monolithic_run(tmp_path / "shards", "run_shard0000", with_support_ids=True)
    right = _monolithic_run(tmp_path / "shards", "run_shard0001", with_support_ids=True)
    (left / "seam_interface_crossings.jsonl").write_text("{}\n", encoding="utf-8")
    (right / "seam_interface_crossings.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        tmp_path / "shards" / "distributed_universe_manifest.json",
        {
            "shards": [
                {
                    "run_id": "run_shard0000",
                    "shard_index": 0,
                    "seam_neighbor_indices": [1],
                },
                {
                    "run_id": "run_shard0001",
                    "shard_index": 1,
                    "seam_neighbor_indices": [0],
                },
            ]
        },
    )

    report = write_h3_worldline_stitch_producer_report(left, right, tmp_path / "out")

    assert report["missing_inputs"] == ["coarse_fine_refinement_pair"]
    assert report["artifact_emitted"] is False
    assert not list(tmp_path.rglob(PRIMITIVES_ARTIFACT_NAME))


def test_blockers_are_the_missing_inputs_while_any_input_is_absent(
    tmp_path: Path,
) -> None:
    report = h3_worldline_stitch_producer_report(tmp_path / "a", tmp_path / "b")

    # While inputs are missing, the missing inputs themselves are the blockers;
    # the measurement-lane blocker is reserved for the complete-inputs branch.
    assert MEASUREMENT_LANE_BLOCKER not in report["blockers"]
    assert report["blockers"] == report["missing_inputs"]


def test_verifier_stays_fail_closed_on_producerless_artifacts(tmp_path: Path) -> None:
    # Wired-chain check: the verifier that would consume the producer output
    # rejects an artifact lacking producer provenance, so the missing producer
    # lane cannot be bypassed.
    report = h3_worldline_stitch_certificate_report({})

    assert report["h3_worldline_stitch_certificate_receipt"] is False
    assert report["primitive_provenance_verified"] is False
    assert "h3_primitive_producer_provenance_missing" in report["missing_obligations"]


def test_cli_writes_producer_missing_inputs_report(tmp_path: Path) -> None:
    from oph_fpe.cli import main

    out = tmp_path / "out"
    code = main(
        [
            "h3-worldline-stitch-producer",
            "--left-run-dir",
            str(tmp_path / "left"),
            "--right-run-dir",
            str(tmp_path / "right"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads((out / MISSING_INPUTS_REPORT_NAME).read_text(encoding="utf-8"))
    assert payload["mode"] == "oph_h3_worldline_stitch_producer_v0"
    assert payload["artifact_emitted"] is False
    assert payload["missing_inputs"]
