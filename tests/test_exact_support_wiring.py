"""Tests for lane A-wire: the full S2 support wiring receipt and its independent verifier."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from oph_exact import carrier
from oph_exact import support_wiring as W
from oph_exact import verify_support_wiring_independent as V

RECEIPT = W.RECEIPT_PATH


@pytest.fixture(scope="module")
def receipt() -> dict:
    if not RECEIPT.exists():
        pytest.skip("receipt not built")
    return W.load_receipt(RECEIPT)


def test_twelve_neighbour_graph_at_level_one() -> None:
    left, right, shared = W.neighbour_pairs(1)
    assert left.size == 450
    degree = np.bincount(np.concatenate([left, right]), minlength=80)
    counts = dict(zip(*np.unique(degree, return_counts=True)))
    assert counts == {11: 60, 12: 20}
    geometry = W.cell_geometry(1)
    assert int(geometry.pentagonal_cell.sum()) == 60
    assert np.array_equal(degree == 11, geometry.pentagonal_cell)
    assert int((shared == 2).sum()) == 120 and int((shared == 1).sum()) == 330
    # sorted lexicographically, left < right
    assert np.all(left < right)
    assert np.array_equal(np.lexsort((right, left)), np.arange(left.size))


def test_port_assignment_is_a_bijection_per_cell_with_one_unglued_port() -> None:
    wiring = W.build_wiring(1)
    slots = np.concatenate([wiring.left * 12 + wiring.left_port, wiring.right * 12 + wiring.right_port])
    assert np.unique(slots).size == slots.size
    assert wiring.receipt["assignment_acceptance"]["accepted"] is True
    assert int((wiring.unglued_port >= 0).sum()) == 60
    used = np.zeros((80, 12), dtype=bool)
    used[wiring.left, wiring.left_port] = True
    used[wiring.right, wiring.right_port] = True
    for cell in range(80):
        if wiring.degree[cell] == 12:
            assert used[cell].all() and wiring.unglued_port[cell] == -1
        else:
            assert used[cell].sum() == 11 and not used[cell, wiring.unglued_port[cell]]
    assert wiring.receipt["three_port_production_gluing"]["edge_adjacent_pairs_checked"] == 120


def test_antipodal_consistency_is_reproducible() -> None:
    a = W.build_wiring(2).receipt["antipodal_consistency"]
    W.build_wiring.cache_clear()
    b = W.build_wiring(2).receipt["antipodal_consistency"]
    assert a == b
    anti = np.asarray(carrier.antipode())
    wiring = W.build_wiring(2)
    assert a["consistent_pairs"] == int(np.sum(anti[wiring.left_port] == wiring.right_port))
    assert 0.0 < a["fraction"] < 1.0


def test_descent_functional_on_a_short_provenance_run() -> None:
    system = W.single_level_system(1)
    loads = W.initial_loads(1, system.ports)
    log = W.run_provenance(system, 6, 4242, loads)
    assert log.ledger_exact and log.descent_violations == 0
    v = [r["descent_functional_after_round"] for r in log.per_round]
    assert v[0] <= log.v_initial and all(b <= a for a, b in zip(v, v[1:]))
    assert log.v_terminal >= log.v_minimum
    assert log.attempts == 6 * system.external
    # replay through the verifier's plain loop gives the same log
    own = V.replay_schedule(V.wiring(1), 6, 4242)
    assert np.array_equal(own["seam"].astype(np.int64), log.seam.astype(np.int64))
    assert np.array_equal(own["outcome"].astype(np.int64), log.outcome.astype(np.int64))
    assert own["state_sha256"] == log.terminal_state_sha256


def test_provenance_rule_on_a_hand_built_log() -> None:
    # Four events on three carriers: e0 writes (0, 1); e1 reads (1, 2) and writes;
    # e2 waits on (0, 2) (reads only); e3 writes (0, 1).
    cell_a = np.array([0, 1, 0, 0])
    cell_b = np.array([1, 2, 2, 1])
    writes = np.array([True, True, False, True])
    prov = W.provenance_from_log(cell_a, cell_b, writes, 3)
    assert prov.parent_a.tolist() == [-1, 0, 0, 0]
    assert prov.parent_b.tolist() == [-1, -1, 1, 1]
    assert prov.version_read_a.tolist() == [0, 1, 1, 1]
    assert prov.version_read_b.tolist() == [0, 0, 1, 2]
    assert prov.version_written_a.tolist() == [1, 2, 1, 2]
    assert prov.version_written_b.tolist() == [1, 1, 1, 3]
    reference = W.provenance_rule_reference(cell_a.tolist(), cell_b.tolist(), writes.tolist(), 3)
    assert reference["parent_a"] == prov.parent_a.tolist() and reference["parent_b"] == prov.parent_b.tolist()
    # e2 is maximal (it wrote nothing); the interval [e0, e3] is {e0, e1, e3}.
    ids = W.interval_ids(prov, 0, 3)
    assert ids.tolist() == [0, 1, 3]
    indptr, indices = W.local_parents(prov, ids)
    pairs = W.dag_pairs(indptr, indices, np.zeros(3, dtype=np.int64))
    assert pairs["strict_pair_count"] == 3  # e0<e1, e0<e3, e1<e3
    assert W.dag_depth(indptr, indices).tolist() == [1, 2, 3]
    assert pairs["edge_is_cover"].tolist() == [True, False, True]


def test_pair_counter_matches_python_bitsets_on_a_random_dag() -> None:
    rng = np.random.default_rng(3)
    n = 400
    parents = [sorted(set(rng.integers(0, v, size=min(v, 2)).tolist())) if v else [] for v in range(n)]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(p) for p in parents])
    indices = np.asarray([p for ps in parents for p in ps], dtype=np.int64)
    exact = W.dag_pairs(indptr, indices, rng.integers(0, 5, size=n))
    past = [0] * n
    total = 0
    for v in range(n):
        bits = 0
        for p in parents[v]:
            bits |= past[p]
        total += bits.bit_count()
        past[v] = bits | (1 << v)
    assert exact["strict_pair_count"] == total
    sampled = W.dag_pairs(indptr, indices, rng.integers(0, 5, size=n), exact_limit=10, sample_seed=1)
    assert sampled["pair_counting"] == "stratified_sample_by_round"
    assert abs(sampled["strict_pair_count"] - total) < 5 * max(sampled["strict_pair_count_standard_error"], 1.0) + 0.2 * total


def test_source_net_block_reproduces_the_receipt_row() -> None:
    block = W.source_net_block()
    top = block["vertical_intervals"][-1]
    assert top["events"] == 1529 and top["strict_pair_count"] == 102990
    assert abs(top["myrheim_meyer_dimension"] - 4.149) < 1e-2
    assert block["reproduces_source_net_receipt_top_interval"] is True


def test_receipt_frozen(receipt: dict) -> None:
    assert receipt["schema"] == W.SCHEMA
    assert RECEIPT.read_text(encoding="ascii") == W.canonical_json(receipt)
    assert receipt["pins"] == W.file_pins()
    assert W.canonical_json(receipt["wiring"]["L3"]) == W.canonical_json(W.build_wiring(3).receipt)
    assert W.canonical_json(receipt["source_net_comparison"]) == W.canonical_json(W.source_net_block())
    for level_key, block in receipt["provenance"].items():
        assert set(block["primary"]["summary"]["expected_2plus1"]) == {"ordering_fraction", "dimension", "growth_exponent"}
    assert all(receipt["verdicts"].values())
    stored = receipt["provenance"]["L3"]["stored_log"]
    assert stored is not None and (W.REPO_ROOT / stored["path"]).exists()
    assert (W.REPO_ROOT / stored["path"]).stat().st_size < 5_000_000


def test_verifier_passes_on_receipt_and_fails_on_mutation(receipt: dict, tmp_path: Path) -> None:
    report = V.verify(RECEIPT)
    assert report["ok"] and report["pins_verified"] == len(W.PIN_FILES)
    assert report.get("L3_intervals_recomputed", 0) > 0
    mutated = copy.deepcopy(receipt)
    mutated["wiring"]["L3"]["antipodal_consistency"]["consistent_pairs"] += 1
    path = tmp_path / "mutated.json"
    path.write_text(W.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path, replay=False)
    mutated = copy.deepcopy(receipt)
    row = mutated["provenance"]["L3"]["primary"]["tips"][0]["ladder"][-1]
    row["strict_pair_count"] += 1
    path.write_text(W.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path, replay=False)
    mutated = copy.deepcopy(receipt)
    mutated["pins"]["oph_exact/support_wiring.py"] = "0" * 64
    path.write_text(W.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path, replay=False)
