import numpy as np
import pytest

from oph_fpe.bulk.modular_probe import (
    _perturb_remeasure_pullback,
    _perturb_remeasure_response_matrix,
)
from oph_fpe.gauge.covariant_overlap import (
    covariant_discrepancy,
    covariant_mismatch_mask,
    repair_covariant_port_pairs,
    repair_production_sector_links,
)


def _graph(node_count: int = 30, seed: int = 5):
    rng = np.random.default_rng(seed)
    left, right = [], []
    for index in range(node_count):
        left.append(index)
        right.append((index + 1) % node_count)
        left.append(index)
        right.append((index + 4) % node_count)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    gauge = rng.integers(0, 6, size=left.size, dtype=np.int16)
    port_left = rng.integers(0, 6, size=left.size, dtype=np.int64)
    port_right = rng.integers(0, 6, size=left.size, dtype=np.int64)
    theta = 2.0 * np.pi * np.arange(node_count) / node_count
    points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(node_count)])
    return points, left, right, port_left, port_right, gauge


SECTOR_CFG = {"enabled": True, "probability": 0.5, "mode": "repair_coupled_group_compose"}


def _graph_response(left, right, port_left, port_right, gauge, *, enabled, config):
    payload = {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "gauge": gauge,
        "group_name": "S3",
        "group_order": 6,
        "patch_count": int(max(int(left.max()), int(right.max())) + 1),
        "production_sector_repair_enabled": enabled,
    }
    if config is not None:
        payload["production_sector_repair_config"] = config
    return payload


def test_replay_step_restores_covariant_consistency():
    _, left, right, port_left, port_right, gauge = _graph()
    rng = np.random.default_rng(9)
    pl, pr, ga = port_left.copy(), port_right.copy(), gauge.copy()
    active = np.flatnonzero(covariant_mismatch_mask(pl, pr, ga, group_name="S3", group_order=6))
    assert active.size > 0
    delta = covariant_discrepancy(pl[active], pr[active], ga[active], group_name="S3", group_order=6)
    repair_production_sector_links(ga, active, delta, group_name="S3", group_order=6, rng=rng, config=SECTOR_CFG)
    repair_covariant_port_pairs(pl, pr, ga, active, rng.random(active.size) < 0.5, group_name="S3", group_order=6)
    still = covariant_mismatch_mask(pl, pr, ga, group_name="S3", group_order=6)[active]
    assert not np.any(still)


def test_probe_replays_sector_repair_with_config():
    points, left, right, port_left, port_right, gauge = _graph()
    basis = np.arange(0, 30, 3, dtype=np.int64)
    fields = {"record_signature": np.linspace(-1.0, 1.0, 30)}
    response, meta = _perturb_remeasure_response_matrix(
        points,
        fields,
        basis,
        graph_response=_graph_response(left, right, port_left, port_right, gauge, enabled=True, config=SECTOR_CFG),
        seed=3,
        probe_steps=6,
        probe_repairs_per_source=32,
        probe_max_incident_edges=8,
    )
    assert meta["production_sector_repair_replayed"] is True
    assert meta["probe_side_selection"] == "production_coin"
    assert meta["gauge_covariant_probe_receipt"] is True
    assert meta["mean_repaired_edges_per_source"] > 0.0
    assert response.shape == (basis.size, basis.size)
    assert float(np.sum(response)) > 0.0
    # The shared gauge array is never mutated by the probe.
    _, _, _, _, _, gauge_fresh = _graph()
    assert np.array_equal(gauge, gauge_fresh)


def test_probe_fail_closed_without_config():
    points, left, right, port_left, port_right, gauge = _graph()
    basis = np.arange(0, 30, 3, dtype=np.int64)
    fields = {"record_signature": np.linspace(-1.0, 1.0, 30)}
    with pytest.raises(ValueError, match="production_sector_repair_config"):
        _perturb_remeasure_response_matrix(
            points,
            fields,
            basis,
            graph_response=_graph_response(left, right, port_left, port_right, gauge, enabled=True, config=None),
            seed=3,
            probe_steps=4,
            probe_repairs_per_source=16,
            probe_max_incident_edges=8,
        )
    matrix, gap, meta = _perturb_remeasure_pullback(
        points,
        None,  # cap is unused on the fail-closed path
        fields,
        basis,
        graph_response=_graph_response(left, right, port_left, port_right, gauge, enabled=True, config=None),
        seed=3,
        probe_steps=4,
        probe_repairs_per_source=16,
        probe_max_incident_edges=8,
    )
    assert meta["response_source"] == "perturb_remeasure_response_fail_closed"
    assert "production_sector_repair_config_missing_for_replay" in meta["proof_blockers"]


def test_legacy_path_unchanged_when_sector_repair_disabled():
    points, left, right, port_left, port_right, gauge = _graph()
    basis = np.arange(0, 30, 3, dtype=np.int64)
    fields = {"record_signature": np.linspace(-1.0, 1.0, 30)}
    response, meta = _perturb_remeasure_response_matrix(
        points,
        fields,
        basis,
        graph_response=_graph_response(left, right, port_left, port_right, gauge, enabled=False, config=None),
        seed=3,
        probe_steps=4,
        probe_repairs_per_source=16,
        probe_max_incident_edges=8,
    )
    assert meta["production_sector_repair_replayed"] is False
    assert meta["probe_side_selection"] == "affinity_score"
    assert response.shape == (basis.size, basis.size)
