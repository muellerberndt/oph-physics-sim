import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.bulk.observer_agreement import (
    MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT,
    OBSERVER_SPACETIME_CONSENSUS_RECEIPT,
    dressed_edge_views,
    observer_agreement_report,
    observer_frame,
    recover_regauging,
)
from oph_fpe.finite_groups import S3_INV, S3_MUL


def _ring_with_chords(node_count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for index in range(node_count):
        left.append(index)
        right.append((index + 1) % node_count)
    # Dense chords: every node gets two chords so that window overlaps carry
    # non-tree edges and pin the re-gauging section uniquely.
    for index in range(node_count):
        left.append(index)
        right.append((index + 3) % node_count)
        left.append(index)
        right.append((index + 5) % node_count)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def test_recover_regauging_exact_and_control():
    rng = np.random.default_rng(7)
    left, right = _ring_with_chords()
    gauge = rng.integers(0, 6, size=left.size, dtype=np.int64)
    nodes = np.arange(24, dtype=np.int64)
    frame_a = {int(n): int(v) for n, v in zip(nodes, observer_frame(3, 11, nodes), strict=True)}
    frame_b = {int(n): int(v) for n, v in zip(nodes, observer_frame(3, 12, nodes), strict=True)}
    view_a = dressed_edge_views(left, right, gauge, frame_a)
    view_b = dressed_edge_views(left, right, gauge, frame_b)

    recovery = recover_regauging(left, right, view_a, view_b)
    assert recovery["defect"] == 0.0
    assert recovery["checkable_edges"] > 0
    # Recovered map must equal f_b f_a^{-1} up to nothing: it is exact.
    for node in nodes:
        expected = int(S3_MUL[frame_b[int(node)], S3_INV[frame_a[int(node)]]])
        assert recovery["h_by_node"][int(node)] == expected

    shuffled = rng.integers(0, 6, size=view_b.size, dtype=np.int64)
    control = recover_regauging(left, right, view_a, shuffled)
    assert control["defect"] > 0.3


def _write_synthetic_run(tmp_path: Path, observer_specs: list[list[int]]) -> Path:
    rng = np.random.default_rng(5)
    left, right = _ring_with_chords(30)
    gauge = rng.integers(0, 6, size=left.size, dtype=np.int64)
    points = rng.standard_normal((30, 3))
    np.savez_compressed(
        tmp_path / "s3_gauge_state.npz",
        left=left,
        right=right,
        gauge=gauge,
        points=points,
    )
    with (tmp_path / "observer_views.jsonl").open("w") as handle:
        for observer_id, support in enumerate(observer_specs):
            handle.write(
                json.dumps(
                    {
                        "view_type": "patch_observer",
                        "observer_id": observer_id,
                        "support_nodes": support,
                        "modular_depth_mean": 1.0 + 0.01 * observer_id,
                        "observer_relative_times": [0.0, 0.5, 1.0],
                    }
                )
                + "\n"
            )
    (tmp_path / "observer_modular_experience_report.json").write_text(
        json.dumps({"observer_facing_3p1d_h3_experience_receipt": True})
    )
    return tmp_path


def test_observer_agreement_report_synthetic(tmp_path):
    # Twenty overlapping observers around the ring: every pair of neighbors
    # shares half a window, so the certificate has pairs and triples.
    specs = [[(start + offset) % 30 for offset in range(12)] for start in range(0, 30, 2)]
    run = _write_synthetic_run(tmp_path, specs)
    report = observer_agreement_report(run, seed=2, max_pairs=64, max_triples=32, min_overlap_edges=4)

    assert report["status"] == "evaluated"
    assert report["bulk_dimension_claim"] is None
    assert report["pair_agreement"]["median_defect"] == 0.0
    assert report["pair_agreement"]["perfect_fraction"] == 1.0
    assert report["control"]["median_defect_shuffled"] > 0.3
    assert report["cocycle"]["median_defect"] == 0.0
    assert report["experienced_chart"]["spatial_dimension"] == 3
    assert report["experienced_chart"]["time_dimension"] == 1
    assert report[OBSERVER_SPACETIME_CONSENSUS_RECEIPT] is True
    assert report[MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT] is True


def test_observer_agreement_missing_gauge(tmp_path):
    report = observer_agreement_report(tmp_path)
    assert report["status"] == "missing_s3_gauge_state"
    assert report["bulk_dimension_claim"] is None


@pytest.mark.skipif(
    not Path("runs/k1_fusion_universe_64k_20260712_rerun/s3_gauge_state.npz").exists(),
    reason="real run artifacts absent",
)
def test_observer_agreement_real_run_smoke():
    report = observer_agreement_report(
        "runs/k1_fusion_universe_64k_20260712_rerun",
        seed=1,
        max_pairs=24,
        max_triples=8,
        min_overlap_edges=6,
        max_observers=192,
    )
    assert report["status"] == "evaluated"
    assert report["bulk_dimension_claim"] is None
    assert report["population"]["evaluated_pairs"] >= 1


def test_agreement_bulk_field_synthetic(tmp_path):
    from oph_fpe.bulk.observer_agreement import write_agreement_bulk_field

    specs = [[(start + offset) % 30 for offset in range(12)] for start in range(0, 30, 2)]
    run = _write_synthetic_run(tmp_path, specs)
    summary = write_agreement_bulk_field(run, seed=2, max_pairs=64, max_triples=32, min_overlap_edges=4)
    assert summary["status"] == "evaluated"
    assert summary["certified_pairs_used"] > 0
    assert summary["pair_certified_patch_fraction"] > 0.0
    payload = np.load(tmp_path / "agreement_bulk_field.npz")
    assert payload["coverage"].shape == (30,)
    assert int(payload["pair_certified"].max()) >= 1
