"""The reserve accounting kernel replays the engine exactly and its readouts respond to the controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_exact import federation_huge as H
from oph_exact import federation_reserve as R


@pytest.fixture(scope="module")
def level2(tmp_path_factory) -> tuple[Path, Path]:
    cache = tmp_path_factory.mktemp("geo")
    out = tmp_path_factory.mktemp("run")
    H.build(2, out, cache, schedules=2, workers=1, float_sweeps=2, float_schedules=1, kernel_cells=2, kernel_steps=(1,), long_steps=(5,), long_cells=0, log=lambda m: None)
    return cache, out


def test_collar_count_and_engine_replay(level2) -> None:
    cache, out = level2
    geo = H.Geometry(cache, 2)
    collar = R.Collar(geo)
    assert collar.collar_seams == 30 * 2**2 and collar.counts["intra_face"] == geo.inter_count - collar.collar_seams
    loads = H.initial_loads(2, geo.ports)
    receipt = json.loads((out / "receipt.json").read_text())
    for e in receipt["integer_law"]["entries"]:
        sched = R.run_schedule(geo, collar, loads, e["seed"])
        assert sched["terminal_sha256"] == e["terminal_state"]["sha256"] and sched["sweeps"] == e["sweeps"]
        assert sched["V_ledger"] == e["V_ledger"]
        totals = np.asarray(sched["totals"])
        assert int(totals[:, :, 0].sum()) == e["descents"] and int(totals[:, :, 1].sum()) == e["swaps"] and int(totals[:, :, 2].sum()) == e["waits"]
        assert int(totals[:, :, 3].sum()) == e["unit_transfers"]


def test_readout_and_controls(level2) -> None:
    cache, out = level2
    geo = H.Geometry(cache, 2)
    collar = R.Collar(geo)
    loads = H.initial_loads(2, geo.ports)
    sched = R.run_schedule(geo, collar, loads, H.schedule_seed(2, 0))
    read = R.survival_readout(sched, 2, int(loads.astype(np.int64).sum()))
    fl = read["face_load"]
    assert 0.0 < fl["hazard_per_sweep_first_8"][0] < 1.0 and fl["ticks"]["0"]["sweeps_per_tick"] == 4
    assert 0.3 < fl["forward_share"] < 0.7
    assert 0.0 < read["descent_excess"]["contraction_per_sweep"] < 1.0
    # shuffled seam classes: the same trajectory, a different collar hazard
    rng = np.random.default_rng(1)
    shuffled = collar.cls_inter.copy()
    rng.shuffle(shuffled)
    ctrl = R.run_schedule(geo, collar, loads, H.schedule_seed(2, 0), cls_override=shuffled)
    assert ctrl["terminal_sha256"] == sched["terminal_sha256"]
    assert ctrl["totals"] != sched["totals"]


def test_refinement_survival_white_noise_reads_theta_two() -> None:
    lo = {"level": 5, "retention": {"rows": [{"resolution": 6, "cells_per_cap": 100.0, "initial_excess_sd": 0.04, "terminal_excess_sd_median": 0.02}]}}
    hi = {"level": 6, "retention": {"rows": [{"resolution": 6, "cells_per_cap": 400.0, "initial_excess_sd": 0.02, "terminal_excess_sd_median": 0.01}]}}
    r = R.refinement_survival(lo, hi)
    assert abs(r["rows"][0]["lambda_initial"] - 0.25) < 1e-9 and abs(r["rows"][0]["theta_initial"] - 2.0) < 1e-9
    assert abs(r["rows"][0]["theta_terminal"] - 2.0) < 1e-9
