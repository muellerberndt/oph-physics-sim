"""The lean settlement engine reproduces the reference federation lane and its verifier rejects tampering."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oph_exact import federation as F
from oph_exact import federation_huge as H
from oph_exact import verify_federation_huge_independent as V
from oph_exact.federation_archive import array_sha256


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("geo")


@pytest.fixture(scope="module")
def level1(cache) -> tuple[H.Geometry, np.ndarray, F.ExactFederation]:
    H.build_geometry(1, cache, log=lambda m: None)
    geo = H.Geometry(cache, 1)
    return geo, H.initial_loads(1, geo.ports), F.build_federation(1, "port_pair")


def test_geometry_matches_reference_lane(level1) -> None:
    geo, loads, fed = level1
    assert geo.carriers == fed.carriers and geo.seams == fed.seams and geo.inter_count == fed.inter_count
    assert np.array_equal(loads.astype(np.int64), F.initial_loads(1, fed.ports))
    a, b = geo.seam_endpoints(np.arange(geo.seams))
    assert np.array_equal(a, fed.seam_a) and np.array_equal(b, fed.seam_b)
    assert geo.mean_denominator == fed.mean_denominator
    exp = H.expectation(geo, loads)
    assert exp["expected_integer_quotient_hash"] == fed.expected_integer_quotient_hash(loads.astype(np.int64))
    assert exp["balanced_minimum"] == fed.integer_minimum_descent(loads.astype(np.int64))


@pytest.mark.parametrize("index", [0, 1])
def test_integer_law_equals_reference(level1, tmp_path, index) -> None:
    geo, loads, fed = level1
    seed = H.schedule_seed(1, index)
    assert seed == F.schedule_seed(1, "port_pair", index)
    mine = H.run_integer(geo, loads, seed, tmp_path / "s", log=lambda m: None)
    ref = F.run_integer_law(fed, loads.astype(np.int64), seed)
    for key in ("quotient_hash", "sweeps", "attempts_to_balanced_class", "descents", "swaps", "waits", "unit_transfers",
                "odd_tie_seams_at_termination", "max_seam_difference_at_termination"):
        assert mine[key] == ref[key], key
    assert np.array_equal(np.load(tmp_path / "s" / "terminal_state.npy").astype(np.int64), np.asarray(ref["state"], dtype=np.int64))
    assert mine["quotient_hash_equals_expected"] and mine["conservation_exact"] and mine["descent_ledger_exact"]


def test_mean_law_equals_reference(level1, tmp_path) -> None:
    geo, loads, fed = level1
    seed = H.schedule_seed(1, 0)
    mine = H.run_mean(geo, loads, seed, tmp_path / "m", sweeps_budget=12, log=lambda m: None)
    ref = F.run_mean_law(fed, loads.astype(np.float64), seed, max_sweeps=12)
    assert mine["sweeps"] == ref["sweeps"] == 12
    assert mine["phi_terminal"] == ref["phi_terminal"]
    assert mine["V_terminal"] == ref["descent_functional_terminal"]
    assert mine["waits"] == ref["waits"]
    assert mine["terminal_quotient_hash"] is None and not mine["lattice_snap_unambiguous"]


def test_chunked_draws_reproduce_single_call_stream() -> None:
    seams = 100_003
    rng = np.random.default_rng(4242)
    seq, coin, digests = H.draw_sweep(rng, seams, coins=True, chunk=7_777)
    ref = np.random.default_rng(4242)
    rseq = ref.integers(0, seams, size=seams, dtype=np.int64)
    rcoin = ref.integers(0, 2, size=seams, dtype=np.int64)
    assert np.array_equal(seq, rseq) and np.array_equal(coin, rcoin)
    assert digests == [array_sha256(rseq, "int64"), array_sha256(rcoin, "int64")]


def test_local_ball_kernel_equals_whole_federation(level1) -> None:
    geo, _loads, fed = level1
    adj = H.cell_adjacency(geo)
    for cell in (0, 37):
        mine = H.response_kernels_local(geo, adj, cell, (1, 5))
        ref = F.response_kernels(fed, cell, (1, 5))
        for n in (1, 5):
            assert np.max(np.abs(np.asarray(mine["kernels"][str(n)]) - ref[n])) < 1e-12


@pytest.fixture(scope="module")
def built(cache, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("run")
    H.build(1, out, cache, schedules=2, workers=1, float_sweeps=8, float_schedules=1, kernel_cells=3, kernel_steps=(1, 5), long_steps=(30,), long_cells=1, log=lambda m: None)
    return out


def test_receipt_verifies(built) -> None:
    result = V.verify(built, replay_max_sweeps=1000, kernel_sample=2, log=lambda m: None)
    assert result["verdict"] == "PASS"
    assert all(r["replay_from"] == 0 for r in result["integer"])


def _copy(built: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(built, dst)
    return dst


def test_verifier_rejects_terminal_state_mutation(built, tmp_path) -> None:
    run = _copy(built, tmp_path)
    receipt = json.loads((run / "receipt.json").read_text())
    seed = receipt["integer_law"]["entries"][0]["seed"]
    path = run / f"integer_{seed}" / "terminal_state.npy"
    x = np.load(path)
    x[0] += 1
    np.save(path, x)
    with pytest.raises(V.VerificationError):
        V.verify(run, log=lambda m: None)


def test_verifier_rejects_draw_digest_mutation(built, tmp_path) -> None:
    run = _copy(built, tmp_path)
    receipt = json.loads((run / "receipt.json").read_text())
    seed = receipt["integer_law"]["entries"][1]["seed"]
    sched_path = run / f"integer_{seed}" / "schedule.json"
    sched = json.loads(sched_path.read_text())
    sched["draw_sha256_per_sweep"][0][1] = "0" * 64
    sched_path.write_bytes(H.canonical(sched))
    with pytest.raises(V.VerificationError, match="draw digests"):
        V.verify(run, log=lambda m: None)


def test_verifier_rejects_expected_hash_mutation(built, tmp_path) -> None:
    run = _copy(built, tmp_path)
    path = run / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["expected"]["expected_integer_quotient_hash"] = "f" * 64
    path.write_bytes(H.canonical(receipt))
    with pytest.raises(V.VerificationError, match="expected hash"):
        V.verify(run, log=lambda m: None)


def test_verifier_rejects_kernel_mutation(built, tmp_path) -> None:
    run = _copy(built, tmp_path)
    path = run / "receipt.json"
    receipt = json.loads(path.read_text())
    k = receipt["response_kernels"]["cells"][0]["kernels"]["1"]
    k[0][1] += 1e-6
    k[1][0] += 1e-6
    path.write_bytes(H.canonical(receipt))
    with pytest.raises(V.VerificationError, match="kernel"):
        V.verify(run, kernel_sample=1, log=lambda m: None)
