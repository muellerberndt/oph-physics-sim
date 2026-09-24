"""The FLRW record-density readout is a deterministic function of the record-metric receipt."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from oph_exact import flrw_record_density_readout as R


def test_readout_is_current() -> None:
    assert R.canonical(R.build()) == R.OUTPUT.read_bytes()


def test_identities_hold_on_every_level() -> None:
    out = R.build()
    assert out["summary"]["sandwich_holds_all"]
    assert out["summary"]["clock_enclosure_holds_all"]
    assert out["summary"]["redshift_enclosure_holds_all"]
    for lv in out["levels"]:
        const = lv["profiles"]["constant"]
        # constant profile: the physical clock is the flat clock and the redshift reading is one
        assert const["expanding_count_clock"]["physical_clock"] == lv["flat_count_clock"]
        assert const["redshift"]["one_plus_z_reading"] == 1.0
        for name in ("de_sitter", "radiation", "matter"):
            p = lv["profiles"][name]
            assert p["sigma_0"] == 1.0 and abs(p["sigma_K"] - 2.0) < 1e-9
            assert p["expanding_count_clock"]["physical_clock"] > lv["flat_count_clock"]


def test_mass_identity_against_direct_sum() -> None:
    receipt = json.loads(R.RECEIPT.read_text())
    row = receipt["levels"][-1]
    fam = row["families"][0]
    K = fam["layer_steps"]
    counts = [int(c) for c in fam["vertical_intervals"][-1]["counts_by_layer"]]
    h = 1.0 / (2.0 * K)
    direct = sum((1.0 / (1.0 - h * j)) ** 4 * n for j, n in enumerate(counts))
    prof = R.profiles(K)["de_sitter"]
    assert abs(R.mass(prof["sigma"], counts) - direct) <= 1e-9 * direct


def test_mutated_receipt_changes_readout(tmp_path) -> None:
    receipt = json.loads(R.RECEIPT.read_text())
    mutated = copy.deepcopy(receipt)
    fam = mutated["levels"][-1]["families"][0]
    fam["vertical_intervals"][-1]["counts_by_layer"][3] += 1
    fam["count_clock"]["interval_count"] += 1
    path = tmp_path / "receipt.json"
    path.write_bytes(R.canonical(mutated))
    assert R.canonical(R.build(path)) != R.OUTPUT.read_bytes()
    broken = copy.deepcopy(receipt)
    broken["levels"][-1]["families"][0]["vertical_intervals"][-1]["counts_by_layer"].append(7)
    path.write_bytes(R.canonical(broken))
    with pytest.raises(ValueError):
        R.build(path)
