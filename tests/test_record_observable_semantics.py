"""Regression guards for the physical record-observable surface.

The exported record observables are computed from physical record content
(the canonical record port rows), never from the internal SplitMix64 record
hash.  These tests fail if a hash-shaped quantity returns to
the exported path, if a hash-to-state feedback mode becomes selectable, or if
the gauge-blind mismatch fallback becomes reachable without an explicit
negative-control request.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from oph_fpe.evidence.controls import mandatory_control_report
from oph_fpe.scale import bw_array as bw_array_module
from oph_fpe.scale.array_screen import (
    _entropy,
    _node_signature,
    _record_packet_entropy,
    _record_packet_ids,
    _record_port_entropy,
)
from oph_fpe.scale.bw_array import (
    REMOVED_HASH_RECORD_FEEDBACK_MODES,
    _apply_observer_readback_drive,
    _observable_fields,
    _observer_raw_fields,
    run_bw_array_config,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny_config(run_id: str) -> dict:
    config = dict(load_config(Path("configs/e1_s3_bw_screen_64k.yml")))
    config["run_id"] = run_id
    config["graph"] = dict(config["graph"], patch_count=256, neighbors=8)
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=256)
    config["bw"] = dict(config["bw"], cap_count=2, times=[0.025], n_jobs=1)
    config["observers"] = dict(config.get("observers", {}), sample_count=8, neighborhood_size=16)
    config["cosmology"] = {
        "freezeout": {"enabled": False},
        "oph_cmb": {"enabled": False},
        "harmonic_time_trace": {
            "enabled": True,
            "sample_count": 2,
            "ell_max": 4,
            "fields": ["record_port_entropy", "stable_count"],
            "harmonic_batch_size": 128,
        },
    }
    return config


def _ring_edges(patch_count: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.arange(patch_count, dtype=np.int64)
    right = (left + 1) % patch_count
    return left, right


def test_physical_record_entropy_reads_zero_where_the_hash_saturates():
    # Constructed low-entropy state: every patch carries the identical record
    # packet (uniform canonical port rows), all committed.
    patch_count = 64
    record_port_state = np.zeros((patch_count, 12), dtype=np.int64)
    committed = np.ones(patch_count, dtype=bool)

    physical = _record_packet_entropy(record_port_state, committed)
    assert physical == 0.0

    # The removed hash observable saturates at ln(patch_count) on the same
    # structured state because the hash keys include per-node incidence slots.
    left, right = _ring_edges(patch_count)
    port_left = np.zeros(patch_count, dtype=np.int64)
    port_right = np.zeros(patch_count, dtype=np.int64)
    hash_signature = _node_signature(port_left, port_right, left, right, patch_count)
    hash_entropy = _entropy(hash_signature[committed])
    assert hash_entropy >= 0.9 * math.log(patch_count)

    # Detector: an exported record-entropy observable computed this way from a
    # structured state must stay far below the hash saturation value.
    assert physical < 0.1 * hash_entropy


def test_record_packet_entropy_counts_distinct_physical_content_only():
    record_port_state = np.zeros((8, 12), dtype=np.int64)
    record_port_state[4:] = 1
    committed = np.ones(8, dtype=bool)
    value = _record_packet_entropy(record_port_state, committed)
    assert value == pytest.approx(math.log(2.0))
    # Uncommitted patches do not contribute.
    committed[4:] = False
    assert _record_packet_entropy(record_port_state, committed) == 0.0
    # Identical content shares one packet id; distinct content does not.
    ids = _record_packet_ids(record_port_state)
    assert len(set(ids[:4].tolist())) == 1
    assert set(ids[:4].tolist()) != set(ids[4:].tolist())


def test_record_port_entropy_tracks_local_port_diversity():
    state = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    values = _record_port_entropy(state, 6)
    assert values[0] == 0.0
    assert values[1] == pytest.approx(math.log(6.0))


def test_exported_field_lists_carry_physical_record_fields_not_the_hash():
    patch_count = 6
    left, right = _ring_edges(patch_count)
    shared = {
        "left": left,
        "right": right,
        "gauge": np.zeros(patch_count, dtype=np.int16),
        "patch_count": patch_count,
        "record_port_entropy": np.linspace(0.0, 1.0, patch_count),
        "stable_count": np.ones(patch_count),
        "committed": np.ones(patch_count, dtype=bool),
        "repair_load": np.zeros(patch_count),
        "mismatch_density": np.zeros(patch_count),
        "modular_depth": np.zeros(patch_count),
        "modular_time": np.zeros(patch_count),
        "cumulative_repair_load": np.zeros(patch_count),
    }
    screen_fields = _observable_fields(
        port_left=np.zeros(patch_count, dtype=np.int16),
        port_right=np.zeros(patch_count, dtype=np.int16),
        **shared,
    )
    raw_fields = _observer_raw_fields(
        record_packet_id=np.arange(patch_count, dtype=np.int64),
        **shared,
    )
    assert "record_signature" not in screen_fields
    assert "record_signature" not in raw_fields
    assert "record_port_entropy" in screen_fields
    assert "record_port_entropy" in raw_fields
    assert "record_packet_id" in raw_fields
    # The packet-id token is bookkeeping, not a harmonic screen field.
    assert "record_packet_id" not in screen_fields


@pytest.mark.parametrize("mode", sorted(REMOVED_HASH_RECORD_FEEDBACK_MODES))
def test_hash_feedback_mode_selection_raises(mode: str):
    patch_count = 8
    left, right = _ring_edges(patch_count)
    port_left = np.zeros(patch_count, dtype=np.int16)
    port_right = np.zeros(patch_count, dtype=np.int16)
    with pytest.raises(ValueError, match="OPH_HASH_RECORD_FEEDBACK_MODE_REMOVED"):
        _apply_observer_readback_drive(
            port_left,
            port_right,
            left,
            right,
            group_order=6,
            rng=np.random.default_rng(1),
            cycle=0,
            config={"enabled": True, "mode": mode, "edge_fraction": 0.5},
            node_labels=None,
        )


def test_hash_feedback_config_rejected_with_receipt(tmp_path: Path):
    config = _tiny_config("hash_feedback_rejection")
    config["dynamics"] = dict(
        config["dynamics"],
        observer_readback_drive={
            "enabled": True,
            "mode": "committed_record_feedback",
            "edge_fraction": 0.1,
        },
    )
    with pytest.raises(ValueError, match="OPH_HASH_RECORD_FEEDBACK_MODE_REMOVED"):
        run_bw_array_config(config, tmp_path)
    receipt_path = tmp_path / "hash_feedback_rejection" / "config_rejection_receipt.json"
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["receipt"] == "OPH_HASH_RECORD_FEEDBACK_MODE_REMOVED"
    assert receipt["rejected_mode"] == "committed_record_feedback"
    assert receipt["reason"] == "hash feedback into physical state"


def test_physical_boundary_refresh_mode_still_drives_ports():
    patch_count = 16
    left, right = _ring_edges(patch_count)
    port_left = np.zeros(patch_count, dtype=np.int16)
    port_right = np.zeros(patch_count, dtype=np.int16)
    node_labels = np.arange(patch_count, dtype=np.int64) % 6
    driven = _apply_observer_readback_drive(
        port_left,
        port_right,
        left,
        right,
        group_order=6,
        rng=np.random.default_rng(7),
        cycle=0,
        config={
            "enabled": True,
            "mode": "support_visible_boundary_refresh",
            "edge_fraction": 1.0,
        },
        node_labels=node_labels,
    )
    assert driven == patch_count
    assert int(np.sum(port_left != 0)) + int(np.sum(port_right != 0)) > 0


def test_gauge_blind_controls_fallback_raises_without_explicit_request():
    points = np.eye(3)
    left = np.asarray([0, 1, 2])
    right = np.asarray([1, 2, 0])
    ports = np.zeros(3, dtype=np.int16)
    kwargs = {
        "requested_controls": ["no_repair"],
        "points": points,
        "left": left,
        "right": right,
        "initial_port_left": ports,
        "initial_port_right": ports,
        "final_port_left": ports,
        "final_port_right": ports,
        "seed": 5,
    }
    with pytest.raises(ValueError, match="OPH_GAUGE_BLIND_CONTROL_FENCE"):
        mandatory_control_report(**kwargs)
    report = mandatory_control_report(**kwargs, allow_gauge_blind_negative_control=True)
    assert report["mismatch_definition"] == "gauge_blind_negative_control"
    assert report["gauge_blind_negative_control_requested"] is True
    assert report["controls"]["no_repair"]["mismatch_definition"] == "gauge_blind_negative_control"
    coupled = mandatory_control_report(
        **kwargs,
        initial_gauge=np.zeros(3, dtype=np.int16),
        final_gauge=np.zeros(3, dtype=np.int16),
        group_name="S3",
        group_order=6,
    )
    assert coupled["gauge_coupled_overlap_state_supplied"] is True
    assert coupled["mismatch_definition"] != "gauge_blind_negative_control"


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_tiny_run_exports_physical_record_semantics(tmp_path: Path):
    config = _tiny_config("record_semantics_smoke")
    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])

    rows = _read_trace(run_path / "mismatch_trace.csv")
    assert rows
    assert "record_packet_entropy" in rows[0]
    assert "record_entropy" not in rows[0]
    committed = int(float(rows[-1]["committed_records"]))
    packet_entropy = float(rows[-1]["record_packet_entropy"])
    assert packet_entropy >= 0.0
    if committed > 0:
        assert packet_entropy <= math.log(committed) + 1.0e-9

    # The exported value is reproducible from the physical record content in
    # the patch-state artifact; a hash-based export would not be.
    state_path = run_path / "echosahedral_patch_state.npz"
    if state_path.exists():
        with np.load(state_path) as payload:
            recomputed = _record_packet_entropy(
                payload["canonical_record_port_state"],
                payload["committed"],
            )
        assert packet_entropy == pytest.approx(recomputed, abs=1.0e-9)

    receipt = json.loads(
        (run_path / "record_observable_semantics_receipt.json").read_text(encoding="utf-8")
    )
    assert "record_entropy" in receipt["removed_exports"]
    assert "record_signature" in receipt["removed_exports"]
    assert "record_packet_entropy" in receipt["replacement_exports"]
    assert "record_port_entropy" in receipt["replacement_exports"]
    assert sorted(receipt["removed_state_feedback_modes"]) == sorted(
        REMOVED_HASH_RECORD_FEEDBACK_MODES
    )

    bw_report = json.loads((run_path / "bw_report.json").read_text(encoding="utf-8"))
    skipped = {row["observable"] for row in bw_report.get("skipped_scalar_observables", [])}
    field_names = set(bw_report.get("fields", {}) or [])
    assert "record_signature" not in field_names
    assert "record_signature" not in skipped

    with np.load(run_path / "harmonic_time_trace.npz") as harmonic:
        names = set(harmonic.files)
    assert "record_port_entropy" in names
    assert "record_signature" not in names


def test_trace_record_entropy_is_wired_to_the_physical_packet_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinel = 12345.5
    monkeypatch.setattr(
        bw_array_module,
        "_record_packet_entropy",
        lambda *args, **kwargs: sentinel,
    )
    config = _tiny_config("record_semantics_wiring")
    config["cosmology"] = {"freezeout": {"enabled": False}, "oph_cmb": {"enabled": False}}
    result = run_bw_array_config(config, tmp_path)
    rows = _read_trace(Path(result["path"]) / "mismatch_trace.csv")
    values = {float(row["record_packet_entropy"]) for row in rows}
    assert values == {sentinel}


def test_fixed_seed_runs_are_identical(tmp_path: Path):
    config_a = _tiny_config("record_semantics_determinism")
    config_b = _tiny_config("record_semantics_determinism")
    result_a = run_bw_array_config(config_a, tmp_path / "a")
    result_b = run_bw_array_config(config_b, tmp_path / "b")
    trace_a = (Path(result_a["path"]) / "mismatch_trace.csv").read_bytes()
    trace_b = (Path(result_b["path"]) / "mismatch_trace.csv").read_bytes()
    assert trace_a == trace_b
    report_a = json.loads(
        (Path(result_a["path"]) / "echosahedral_patch_state_report.json").read_text(encoding="utf-8")
    )
    report_b = json.loads(
        (Path(result_b["path"]) / "echosahedral_patch_state_report.json").read_text(encoding="utf-8")
    )
    assert report_a["patch_port_state_sha256"] == report_b["patch_port_state_sha256"]
