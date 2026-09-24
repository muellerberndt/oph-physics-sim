"""The texture readout is a deterministic function of a run directory and rejects tampered states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_exact import federation_huge as H
from oph_exact import federation_texture as T


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, Path]:
    cache = tmp_path_factory.mktemp("geo")
    out = tmp_path_factory.mktemp("run")
    H.build(2, out, cache, schedules=3, workers=1, float_sweeps=4, float_schedules=1, kernel_cells=2, kernel_steps=(1,), long_steps=(5,), long_cells=1, log=lambda m: None)
    return out, cache


def test_retention_slope_is_one_for_the_initial_loads() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(2000, 3))
    cell0 = rng.integers(0, 6, size=(2000, 12)).mean(axis=1)
    rows = T.retention(points, cell0, [cell0], (3, 6))
    assert rows and all(abs(r["slope_median"] - 1.0) < 1e-9 and abs(r["correlation_median"] - 1.0) < 1e-9 for r in rows)


def test_crossover_interpolates_in_log_scale() -> None:
    rows = [{"cells_per_cap": 100.0, "slope_median": 0.9}, {"cells_per_cap": 10.0, "slope_median": 0.1}]
    c = T.crossover(rows)
    assert c is not None and 10.0 < c < 100.0
    assert T.crossover([{"cells_per_cap": 100.0, "slope_median": 0.9}, {"cells_per_cap": 10.0, "slope_median": 0.8}]) is None


def test_build_is_deterministic_and_consistent(run) -> None:
    out, cache = run
    a = T.build(out, cache, terminals=3)
    b = T.build(out, cache, terminals=3)
    assert T.canonical(a) == T.canonical(b)
    receipt = json.loads((out / "receipt.json").read_text())
    assert a["descent_curve"]["sweeps"]["max"] == receipt["integer_law"]["sweeps"]["max"]
    assert a["descent_curve"]["median_by_sweep"][0] == 1.0 and a["descent_curve"]["median_by_sweep"][-1] == 0.0
    assert 1.0 <= a["transport"]["transfers_over_lower_bound"]["min"] <= a["transport"]["transfers_over_lower_bound"]["max"]
    assert a["retention"]["terminal_states"] == [e["seed"] for e in receipt["integer_law"]["entries"]]


def test_tampered_terminal_state_is_rejected(run, tmp_path) -> None:
    out, cache = run
    receipt = json.loads((out / "receipt.json").read_text())
    seed = receipt["integer_law"]["entries"][0]["seed"]
    path = out / f"integer_{seed}" / "terminal_state.npy"
    x = np.load(path)
    saved = x.copy()
    x[0] += 1
    np.save(path, x)
    try:
        with pytest.raises(ValueError, match="digest"):
            T.build(out, cache, terminals=1)
    finally:
        np.save(path, saved)
