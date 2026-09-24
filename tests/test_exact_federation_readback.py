"""The slow-band readback readout is consistent with the carrier projector and rejects tampered states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_exact import carrier
from oph_exact import federation_huge as H
from oph_exact import federation_readback as R


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, Path]:
    cache = tmp_path_factory.mktemp("geo")
    out = tmp_path_factory.mktemp("run")
    H.build(2, out, cache, schedules=2, workers=1, float_sweeps=2, float_schedules=1, kernel_cells=2, kernel_steps=(1,), long_steps=(5,), long_cells=0, log=lambda m: None)
    return out, cache


def test_fields_split_the_centred_energy() -> None:
    p_slow = carrier.slow_band_projector()
    p_fast = np.eye(12) - p_slow - np.ones((12, 12)) / 12.0
    x = np.random.default_rng(0).integers(0, 6, size=12 * 500)
    f = R.fields(x, 500, p_slow, p_fast)
    assert np.allclose(f["slow"] + f["fast"], f["centred"])
    assert abs(np.sum(f["slow"] * f["fast"])) < 1e-8
    s = R.summarise(f)
    assert abs(s["slow_band_energy_share"] + s["fast_band_energy_share"] - 1.0) < 1e-6


def test_build_reads_terminal_states(run) -> None:
    out, cache = run
    result = R.build(out, cache, terminals=2)
    assert result["level"] == 2 and len(result["terminal"]) == 2
    for t in result["terminal"]:
        assert 0.0 <= t["slow_band_energy_share"] <= 1.0 and 0.0 <= t["slow_norm_zero_fraction"] <= 1.0


def test_tampered_state_rejected(run) -> None:
    out, cache = run
    receipt = json.loads((out / "receipt.json").read_text())
    seed = receipt["integer_law"]["entries"][0]["seed"]
    path = out / f"integer_{seed}" / "terminal_state.npy"
    x = np.load(path); saved = x.copy(); x[0] += 1; np.save(path, x)
    try:
        with pytest.raises(ValueError, match="digest"):
            R.build(out, cache, terminals=1)
    finally:
        np.save(path, saved)
