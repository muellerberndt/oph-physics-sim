"""Fail-closed tests for the four-row Einstein scaling instrument."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.einstein_convergence_ladder import (
    PAIR_CAP,
    RUNG_SPECS,
    LadderValidationError,
    _canonical_artifact_bytes,
    _canonical_json_bytes,
    _load_artifact_bytes,
    _sha,
    normalize_existing_ladder,
    validate_ladder,
    write_ladder,
)
from oph_fpe.bulk.event_manifold_producer import _fit_quadratic_form

DATA_DIR = Path("data/einstein_convergence")


def _fake_runner(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ordinal = [row["rung_id"] for row in RUNG_SPECS].index(spec["rung_id"])
    chart = np.random.default_rng(ordinal + 80).normal(size=(12, 4))
    pairs = [
        (left, right)
        for left in range(len(chart))
        for right in range(left + 1, len(chart))
    ]
    causal = np.asarray(
        [pair for pair in pairs if sum(pair) % 3 == 0],
        dtype=np.int32,
    )
    spacelike = np.asarray(
        [pair for pair in pairs if sum(pair) % 3 != 0],
        dtype=np.int32,
    )
    flux = np.arange(1, 7, dtype=np.float64) + ordinal
    coefficients = np.linspace(0.2, 0.25, 6) + ordinal / 1000
    fit = _fit_quadratic_form(
        chart,
        {
            "causal": [tuple(map(int, pair)) for pair in causal],
            "spacelike": [tuple(map(int, pair)) for pair in spacelike],
        },
    )
    assert fit["fitted"]
    eigenvalues = np.asarray(fit["eigenvalues"])
    spread = round(
        float((flux.max() - flux.min()) / abs(float(np.median(flux)))),
        6,
    )
    config = {
        "carrier_count": spec["carrier_count"],
        "cycles": 16,
        "seed": 20260751,
        "observer_count": spec["observer_count"],
        "observer_support_size": spec["observer_support_size"],
        "observer_samples": spec["observer_samples"],
        "observer_cross_reads": True,
        "snapshot_coverage": "spanning",
        "geometry_transport": "held_out_flow",
    }
    summary = {
        "config": config,
        "capture_sha256": (
            "sha256:" + hashlib.sha256(spec["rung_id"].encode("utf-8")).hexdigest()
        ),
        "event_count": len(chart),
        "cross_observer_edges": ordinal + 1,
        "causal_pairs_total": len(causal),
        "spacelike_pairs_total": len(spacelike),
        "pair_cap": PAIR_CAP,
        "causal_stride": 1,
        "spacelike_stride": 1,
        "held_out_inertia": list(fit["inertia"]),
        "cone_margin": fit["cone_margin"],
        "eigenvalues": eigenvalues.tolist(),
        "normalization_coefficients": [
            round(float(value), 6) for value in coefficients
        ],
        "coupling_spread": spread,
        "raw_moment_min_eig": 0.5,
    }
    arrays = {
        "chart": chart,
        "causal_pairs": causal,
        "spacelike_pairs": spacelike,
        "flux": flux,
        "coefficients": coefficients,
        "form_eigenvalues": eigenvalues,
    }
    return summary, arrays


def _all_file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_declared_ladder_has_two_distinct_262144_rows() -> None:
    rows = [row for row in RUNG_SPECS if row["carrier_count"] == 262_144]
    assert len(rows) == 2
    assert {row["observer_support_size"] for row in rows} == {96, 384}
    assert len({row["stem"] for row in rows}) == 2
    assert len({row["rung_id"] for row in rows}) == 2


def test_four_row_writer_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_ladder(first, runner=_fake_runner)
    write_ladder(second, runner=_fake_runner)
    assert _all_file_bytes(first) == _all_file_bytes(second)
    manifest = validate_ladder(first)
    assert len(manifest["rungs"]) == 4
    assert manifest["bundle_origin"]["mode"] == "full_physical_source_run"
    assert manifest["bundle_origin"]["source_replay_claimed"] is True
    assert set(manifest["bundle_origin"]["runtime_versions"]) == {
        "python",
        "numpy",
        "scipy",
        "zlib",
    }


def test_manifest_row_loss_fails_closed(tmp_path: Path) -> None:
    write_ladder(tmp_path, runner=_fake_runner)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rungs"].pop()
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    with pytest.raises(LadderValidationError, match="manifest_row_count"):
        validate_ladder(tmp_path)


def test_coordinated_wall_clock_injection_fails_closed(tmp_path: Path) -> None:
    write_ladder(tmp_path, runner=_fake_runner)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rungs"][0]
    summary_path = tmp_path / row["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["capture_seconds"] = 12.5
    summary_bytes = _canonical_json_bytes(summary)
    summary_path.write_bytes(summary_bytes)
    row["summary_sha256"] = _sha(summary_bytes)
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    with pytest.raises(LadderValidationError, match="wall_clock"):
        validate_ladder(tmp_path)


def test_coordinated_eigenvalue_tamper_fails_closed(tmp_path: Path) -> None:
    write_ladder(tmp_path, runner=_fake_runner)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rungs"][0]
    artifact_path = tmp_path / row["artifact"]
    arrays = _load_artifact_bytes(artifact_path.read_bytes())
    arrays["form_eigenvalues"] = arrays["form_eigenvalues"] + 0.25
    artifact_bytes = _canonical_artifact_bytes(arrays)
    artifact_path.write_bytes(artifact_bytes)
    row["artifact_sha256"] = _sha(artifact_bytes)
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    with pytest.raises(
        LadderValidationError,
        match="form_eigenvalue_summary_mismatch",
    ):
        validate_ladder(tmp_path)


def test_coordinated_cone_margin_tamper_fails_closed(tmp_path: Path) -> None:
    write_ladder(tmp_path, runner=_fake_runner)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rungs"][0]
    summary_path = tmp_path / row["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cone_margin"] += 1.0e-9
    summary_bytes = _canonical_json_bytes(summary)
    summary_path.write_bytes(summary_bytes)
    row["summary_sha256"] = _sha(summary_bytes)
    row["cone_margin"] = summary["cone_margin"]
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    with pytest.raises(
        LadderValidationError,
        match="frozen_pair_refit_cone_margin_mismatch",
    ):
        validate_ladder(tmp_path)


def test_normalizer_removes_legacy_wall_clock_field(tmp_path: Path) -> None:
    write_ladder(tmp_path, runner=_fake_runner)
    summary_path = tmp_path / "rung_16384.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["capture_seconds"] = 1.25
    summary_path.write_bytes(_canonical_json_bytes(summary))
    manifest = normalize_existing_ladder(tmp_path)
    normalized = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "capture_seconds" not in normalized
    assert manifest["bundle_origin"]["mode"] == (
        "deterministic_reserialization_of_frozen_arrays"
    )
    assert manifest["bundle_origin"]["source_replay_claimed"] is False
    validate_ladder(tmp_path)


def test_frozen_four_row_bundle_validates() -> None:
    manifest = validate_ladder(DATA_DIR)
    assert [row["rung_id"] for row in manifest["rungs"]] == [
        spec["rung_id"] for spec in RUNG_SPECS
    ]
