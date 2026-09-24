"""Source-net causal-limit lane: frozen receipt, RER digests, pair counts, controls, verifier."""

from __future__ import annotations

import copy
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from oph_exact import source_net as producer
from oph_exact import verify_source_net_independent as verifier
from oph_fpe.bulk.causet_likeness import invert_myrheim_meyer_fraction, myrheim_meyer_fraction

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/exact/source_net_causal_limit_receipt.json"
RER_DIGESTS = {
    5: "cf3d2ee0d16f1e6659c49e31c34b0901ac709ae78d55d8584acb441a3bb83f13",
    8: "03600e54b13d987edf51cf77743a177ee683e700e899fc0c35efd081398204c1",
    13: "5c8f5681bba63a50d993326f642eec5e0cd89f76723e94712a1ba617e068d94a",
}
RER_EDGES = {5: 1330, 8: 16276, 13: 145997}
RER_CENTRE_COUNTS = {5: [1, 39, 39, 1], 8: [1, 93, 93, 1], 13: [1, 179, 1169, 179, 1]}
RER_RECORD_DIGESTS = {
    5: "cc8aa7f94012d3bc64108c89a827a64e62c2660ab9f91c3398bd5a226a80f240",
    8: "c679a9b20853483fdd23127004ca16814132f72e5fc77c6071ee5ebc18b95ff3",
    13: "18c37b118569e19079b444860eda7342685e14b34dad3a0971a81dbf5a275f18",
}


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_bytes())


@pytest.fixture(scope="module")
def small_families(tmp_path_factory) -> dict:
    workdir = tmp_path_factory.mktemp("source_net")
    return {(n, dim): producer.build_family(n, dim, processes=1, workdir=workdir)
            for n in (5, 6, 7) for dim in (3, 2, 1)}


def _family(receipt: dict, n: int, dim: int) -> dict:
    for level in receipt["levels"]:
        if level["fibonacci_index"] == n:
            return next(f for f in level["families"] if f["dimension"] == dim)
    raise KeyError((n, dim))


def _all_pairs_distance(q: int, dim: int) -> np.ndarray:
    indptr, indices, *_ = producer.build_site_graph(q, dim)
    count = q ** dim
    csr = csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr), shape=(count, count))
    return shortest_path(csr, unweighted=True, directed=False).astype(int)


def _brute_force_pairs(events: list[tuple[int, int]], dist: np.ndarray) -> int:
    return sum(1 for (j, s) in events for (jj, t) in events if j < jj and dist[s, t] <= jj - j)


# --------------------------------------------------------------------------
# Frozen, deterministic receipt
# --------------------------------------------------------------------------


def test_receipt_is_canonical_frozen_and_free_of_wall_clock(receipt: dict) -> None:
    data = RECEIPT.read_bytes()
    assert producer.canonical(receipt) == data
    assert receipt["schema"] == producer.SCHEMA
    assert receipt["scope"] == producer.SCOPE
    assert receipt["claim_boundary"] == producer.CLAIM_BOUNDARY
    text = data.decode("ascii")
    assert re.search(r"20\d\d-\d\d-\d\dT", text) is None
    assert "timestamp" not in text and "wall_clock" not in text
    assert [lv["fibonacci_index"] for lv in receipt["levels"]] == list(producer.LEVELS)
    assert [lv["q"] for lv in receipt["levels"]] == [5, 8, 13, 21, 34, 55, 89]
    for level in receipt["levels"]:
        assert [f["dimension"] for f in level["families"]] == [3, 2, 1]


def test_small_levels_rebuild_deterministically_to_the_frozen_rows(receipt: dict, small_families: dict,
                                                                   tmp_path: Path) -> None:
    for (n, dim), fam in small_families.items():
        frozen = _family(receipt, n, dim)
        frozen = {k: v for k, v in frozen.items() if k != "rer_cross_check"}
        assert producer.canonical(fam) == producer.canonical(frozen), (n, dim)
    again = producer.build_family(7, 3, processes=1, workdir=tmp_path)
    assert producer.canonical(again) == producer.canonical(small_families[(7, 3)])


def test_pins_match_the_working_tree(receipt: dict) -> None:
    for rel in producer.LOCAL_PINS:
        assert receipt["source_pins"][rel] == producer.file_sha256(ROOT / rel)
    assert receipt["rer_cross_check"]["receipt_sha256"] == producer.RER_RECEIPT_SHA256


# --------------------------------------------------------------------------
# RER cross-check at the shared levels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,q", [(5, 5), (6, 8), (7, 13)])
def test_rer_structural_digests_are_reproduced(receipt: dict, small_families: dict, n: int, q: int) -> None:
    fam = small_families[(n, 3)]
    assert fam["neighbors_including_wait_sha256"] == RER_DIGESTS[q]
    assert fam["undirected_spatial_edges"] == RER_EDGES[q]
    assert fam["source_records_sha256"] == RER_RECORD_DIGESTS[q]
    assert fam["center_intervals"][-1]["counts_by_layer"] == RER_CENTRE_COUNTS[q]
    frozen = _family(receipt, n, 3)
    block = frozen["rer_cross_check"]
    assert block["all_agree"] is True
    assert set(block["fields"]) == set(producer.RER_COMPARED_FIELDS)
    assert block["fields"]["neighbors_including_wait_sha256"]["rer"] == RER_DIGESTS[q]
    assert receipt["rer_cross_check"]["all_agree"] is True
    assert receipt["rer_cross_check"]["levels_compared"] == [5, 6, 7]


def test_rer_receipt_pin_and_digests_when_theory_checkout_is_present(small_families: dict) -> None:
    path = producer.RER_ROOT / producer.RER_RECEIPT
    if not path.is_file():
        pytest.skip("theory checkout absent")
    theirs = {lv["q"]: lv for lv in producer.rer_receipt()["levels"]}
    for q in (5, 8, 13):
        fam = small_families[({5: 5, 8: 6, 13: 7}[q], 3)]
        check = producer.cross_check(fam, theirs[q])
        assert check["all_agree"] is True, [k for k, v in check["fields"].items() if not v["agree"]]


def test_neighbor_digest_streams_the_canonical_json() -> None:
    indptr, indices, *_ = producer.build_site_graph(8, 3)
    rows = [indices[indptr[i]:indptr[i + 1]].tolist() for i in range(512)]
    producer._worker_init({"indptr": indptr, "indices": indices})
    assert producer.neighbor_digest(512) == producer.digest(rows) == RER_DIGESTS[8]


# --------------------------------------------------------------------------
# Exact arithmetic
# --------------------------------------------------------------------------


def test_vectorized_sign_rule_matches_the_scalar_rule() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(-2000, 2000, size=4000)
    b = rng.integers(-1200, 1200, size=4000)
    a[:50] = 0
    b[50:100] = 0
    expected = np.array([producer.phi_sign((int(x), int(y))) for x, y in zip(a, b)])
    assert np.array_equal(producer.phi_sign_array(a, b), expected)
    # phi^2 = phi + 1 and the conjugate 1 - phi are exact fixed points of the rule.
    assert producer.phi_sign((-1, 1)) == 1 and producer.phi_sign((1, -1)) == -1
    assert producer.phi_sign((-1618034, 1000000)) == -1 and producer.phi_sign((-1618033, 1000000)) == 1


def test_orbit_and_floors_agree_with_the_verifier_basis() -> None:
    for q in (5, 8, 13, 21, 34, 55):
        mine = producer.orbit(q)
        theirs = verifier.orbit_twice(q)
        assert [(2 * m + b, b) for m, b in mine] == theirs
        floors = verifier.golden_floors(q)
        assert [-m for m, _ in mine] == floors
        assert all(f == int(b * (1 + 5 ** 0.5) / 2) for b, f in enumerate(floors))


# --------------------------------------------------------------------------
# Ordering fraction, pair counting and Myrheim-Meyer inversion
# --------------------------------------------------------------------------


def test_myrheim_meyer_reference_values() -> None:
    assert abs(myrheim_meyer_fraction(4.0) - 0.1) < 1e-12
    assert abs(myrheim_meyer_fraction(3.0) - 8 / 35) < 1e-12
    assert abs(myrheim_meyer_fraction(2.0) - 0.5) < 1e-12
    for d in (2.0, 3.0, 4.0):
        assert abs(invert_myrheim_meyer_fraction(myrheim_meyer_fraction(d)) - d) < 1e-9
        assert abs(verifier.mm_dimension(verifier.mm_fraction(d)) - d) < 1e-9
    assert producer.REFERENCE_FRACTION[3] == Fraction(1, 10)


@pytest.mark.parametrize("n,dim", [(5, 3), (6, 3), (6, 2), (7, 2), (7, 1)])
def test_pair_counts_agree_with_explicit_event_enumeration(small_families: dict, n: int, dim: int) -> None:
    fam = small_families[(n, dim)]
    q, K, centre = fam["q"], fam["layer_steps"], fam["intervention_source_id"]
    dist = _all_pairs_distance(q, dim)
    count = q ** dim
    for row in fam["vertical_intervals"]:
        k = row["layers"]
        events = [(j, s) for j in range(k + 1) for s in range(count) if dist[centre, s] <= min(j, k - j)]
        assert len(events) == row["inclusive_event_count"]
        C = _brute_force_pairs(events, dist)
        assert C == row["strict_pair_count"]
        N = len(events)
        assert Fraction(row["ordering_fraction"]) == Fraction(2 * C, N * (N - 1))
    moving = fam["moving_tip_interval"]
    if moving is not None:
        x, y = moving["x_site"], moving["y_site"]
        events = [(j, s) for j in range(K + 1) for s in range(count) if dist[x, s] <= j and dist[s, y] <= K - j]
        assert len(events) == moving["inclusive_event_count"]
        assert _brute_force_pairs(events, dist) == moving["strict_pair_count"]


def test_weight_tables_count_layer_pairs() -> None:
    K = 5
    Wv = producer.vertical_weights(K)
    for k in range(1, K + 1):
        for a in range(K + 1):
            for b in range(K + 1):
                for d in range(K + 2):
                    expected = sum(1 for j in range(a, k - a + 1) for jj in range(b, k - b + 1)
                                   if jj - j >= max(d, 1))
                    assert Wv[k, a, b, d] == expected
    Wm = producer.moving_weights(3)
    assert Wm[0, 0, 0, 0, 0] == 6  # tips: j in 0..3, j' in 0..3, j' > j
    assert Wm[1, 1, 1, 1, 1] == 1  # both restricted to layers 1..2


def test_cone_pruned_bfs_matches_full_distances_on_relevant_pairs() -> None:
    q, dim = 13, 3
    indptr, indices, values, A, B, sites = producer.build_site_graph(q, dim)
    count = q ** dim
    K = producer.ceil_sqrt(q)
    dist_full = _all_pairs_distance(q, dim)
    centre = 732
    alpha = dist_full[centre]
    At, Bt = producer.metric_tables(q, A, B, sites, centre)
    cones = producer.cone_masks(q, At, Bt, K)
    rng = np.random.default_rng(3)
    for s in rng.choice(np.flatnonzero(alpha <= K // 2), size=12, replace=False):
        cap = K - int(alpha[s]) - 1
        if cap <= 0:
            continue
        allowed = [cones[K - int(alpha[s]) - m] for m in range(cap)]
        pruned = producer.bfs(indptr, indices, int(s), count, cap=cap, allowed=allowed)
        assert np.all((pruned < 0) | (pruned == dist_full[s]))
        for t in np.flatnonzero(alpha <= K // 2):
            if dist_full[s, t] <= K - alpha[s] - alpha[t] and t != centre:
                assert pruned[t] == dist_full[s, t]


# --------------------------------------------------------------------------
# Dimension controls
# --------------------------------------------------------------------------


def test_controls_respond_to_dimension(small_families: dict) -> None:
    rows = {dim: small_families[(7, dim)]["vertical_intervals"][-1] for dim in (1, 2, 3)}
    f1, f2, f3 = (rows[d]["ordering_fraction_float"] for d in (1, 2, 3))
    assert f1 > f2 > f3
    assert abs(f1 - 0.5) < 0.05
    assert abs(f2 - 8 / 35) < 0.03
    assert abs(f3 - 0.1) < 0.03
    assert abs(rows[1]["myrheim_meyer_dimension"] - 2.0) < 0.1
    assert abs(rows[2]["myrheim_meyer_dimension"] - 3.0) < 0.1
    assert abs(rows[3]["myrheim_meyer_dimension"] - 4.0) < 0.2


def test_control_populations_share_the_read_law(small_families: dict) -> None:
    for dim in (1, 2):
        fam = small_families[(6, dim)]
        assert fam["q"] == 8 and fam["layer_steps"] == 3 and fam["site_count"] == 8 ** dim
        assert fam["radius_squared_over_L2"] == "1/8"
        assert fam["minimum_neighbor_count_including_wait"] >= 2
    one = small_families[(6, 1)]
    indptr, indices, values, *_ = producer.build_site_graph(8, 1)
    xi = [producer.phi_float(v) for v in values]
    for i in range(8):
        for j in range(8):
            expected = (xi[i] - xi[j]) ** 2 <= 1 / 8 + 1e-12
            assert (j in indices[indptr[i]:indptr[i + 1]].tolist()) == expected


# --------------------------------------------------------------------------
# Continuum references
# --------------------------------------------------------------------------


def test_clipped_volume_reduces_to_the_diamond_inside_the_cube() -> None:
    for dim in (1, 2, 3):
        T = 0.4
        assert abs(producer.clipped_diamond_volume([0.5] * dim, T, dim)
                   - producer.diamond_volume(dim, T)) < 1e-9
        assert producer.clipped_diamond_volume([0.5] * dim, 1.4, dim) < producer.diamond_volume(dim, 1.4)
    assert abs(producer.diamond_volume(3, 1.0) - np.pi / 24) < 1e-15
    assert abs(producer.ball_box_volume([0.5, 0.5, 0.5], 0.3, 3) - 4 * np.pi * 0.027 / 3) < 1e-9
    assert abs(producer.ball_box_volume([0.0, 0.5, 0.5], 0.3, 3) - 2 * np.pi * 0.027 / 3) < 1e-9
    assert abs(producer.ball_box_volume([0.5, 0.5], 0.3, 2) - np.pi * 0.09) < 1e-12
    assert abs(producer.ball_box_volume([0.0, 0.0], 0.3, 2) - np.pi * 0.09 / 4) < 1e-9
    assert abs(verifier.clipped_volume([0.45, 0.5, 0.55], 1.1, 3)
               - producer.clipped_diamond_volume([0.45, 0.5, 0.55], 1.1, 3)) < 1e-6


def test_stratified_estimate_arithmetic() -> None:
    strata = {0: [0], 1: list(range(1, 11)), 2: list(range(11, 41))}
    chosen = producer.allocate_sample(strata, 11)
    assert len(chosen[0]) == 1 and len(chosen[1]) >= 2 and len(chosen[2]) >= 2
    values = {s: s * s for s in range(41)}
    est = producer.stratified_estimate(strata, chosen, values)
    total = Fraction(0)
    for key, members in strata.items():
        picked = chosen[key]
        total += Fraction(len(members) * sum(values[int(s)] for s in picked), len(picked))
    assert Fraction(est["value"]) == total
    assert est["sample_size"] == sum(len(v) for v in chosen.values())


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def test_verifier_accepts_the_committed_receipt() -> None:
    result = verifier.verify(verifier.load(RECEIPT))
    assert result["accepted"] is True
    assert result["rebuilt_levels"] == [5, 6, 7]
    assert len(result["families"]) == 3 * len(producer.LEVELS)


def _mutations(receipt: dict):
    r = copy.deepcopy(receipt)
    r["schema"] = "oph.exact.source-net-causal-limit.v0"
    yield "schema", r
    r = copy.deepcopy(receipt)
    r["scope"]["native_repair_selected"] = True
    yield "scope", r
    r = copy.deepcopy(receipt)
    r["source_pins"]["oph_exact/source_net.py"] = "0" * 64
    yield "pin", r
    r = copy.deepcopy(receipt)
    _family(r, 5, 3)["vertical_intervals"][-1]["strict_pair_count"] += 1
    yield "pair_count", r
    r = copy.deepcopy(receipt)
    _family(r, 6, 3)["neighbors_including_wait_sha256"] = "0" * 64
    yield "digest", r
    r = copy.deepcopy(receipt)
    _family(r, 7, 3)["vertical_intervals"][-1]["inclusive_event_count"] += 1
    yield "event_count", r
    r = copy.deepcopy(receipt)
    _family(r, 7, 2)["moving_tip_interval"]["rank_shift"] += 1
    yield "moving_tips", r
    r = copy.deepcopy(receipt)
    _family(r, 7, 3)["rer_cross_check"]["fields"]["undirected_spatial_edges"]["rer"] = 1
    yield "embedded_rer_value", r
    sampled = [(lv["fibonacci_index"], fam["dimension"]) for lv in receipt["levels"] for fam in lv["families"]
               if fam["vertical_pair_counting"] == "stratified_sample"]
    if sampled:  # the committed receipt counts every level exactly; the sampled path keeps its mutations when present
        n_s, d_s = sampled[0]
        r = copy.deepcopy(receipt)
        row = _family(r, n_s, d_s)["vertical_intervals"][-1]
        row["strict_pair_count_estimate"]["strata"][1]["sum"] += 1
        yield "sampled_sum", r
        r = copy.deepcopy(receipt)
        _family(r, n_s, d_s)["vertical_sample"]["seed"] += 1
        yield "sampled_seed", r
        r = copy.deepcopy(receipt)
        _family(r, n_s, d_s)["vertical_intervals"][-1]["strict_pair_count_estimate"]["sample_size"] -= 1
        yield "sample_size", r
    r = copy.deepcopy(receipt)
    _family(r, 10, 3)["vertical_intervals"][-1]["strict_pair_count"] += 1
    yield "exact_pair_count_q55", r
    r = copy.deepcopy(receipt)
    _family(r, 11, 3)["moving_tip_interval"]["strict_pair_count"] += 1
    yield "exact_moving_pair_count_q89", r
    r = copy.deepcopy(receipt)
    _family(r, 9, 3)["count_clock"]["count_clock"] *= 1.01
    yield "clock", r
    r = copy.deepcopy(receipt)
    _family(r, 8, 3)["vertical_intervals"][2]["ordering_fraction"] = "1/10"
    yield "fraction", r
    r = copy.deepcopy(receipt)
    _family(r, 9, 1)["shell_counts_from_centre"][1] += 1
    yield "shells", r
    r = copy.deepcopy(receipt)
    r["rer_cross_check"]["all_agree"] = False
    yield "summary", r


def test_verifier_rejects_mutated_receipts(receipt: dict) -> None:
    for name, mutated in _mutations(receipt):
        with pytest.raises(verifier.SourceNetVerificationError):
            verifier.verify(mutated)


def test_verifier_does_not_import_the_producer() -> None:
    source = (ROOT / "oph_exact/verify_source_net_independent.py").read_text(encoding="utf-8")
    assert "source_net import" not in source and "import source_net" not in source
    assert "oph_exact.source_net" not in source


def test_strict_loader_rejects_noncanonical_and_nonfinite(tmp_path: Path) -> None:
    for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a": 1}\n'):
        path = tmp_path / "bad.json"
        path.write_bytes(data)
        with pytest.raises(verifier.SourceNetVerificationError):
            verifier.load(path)


# --------------------------------------------------------------------------
# Receipt content at the large levels
# --------------------------------------------------------------------------


def test_large_levels_report_declared_counting_modes(receipt: dict) -> None:
    limit = receipt["pair_counting"]["exact_support_limit"]
    for level in receipt["levels"]:
        for fam in level["families"]:
            exact = fam["vertical_support_site_count"] <= limit
            assert fam["vertical_pair_counting"] == ("exact_all_pairs" if exact else "stratified_sample")
            if not exact:
                est = fam["vertical_intervals"][-1]["strict_pair_count_estimate"]
                assert est["sample_size"] >= receipt["pair_counting"]["sample_size_minimum"]
                assert fam["vertical_intervals"][-1]["ordering_fraction_standard_error"] > 0
    for n in (9, 10, 11):
        fam = _family(receipt, n, 3)
        assert fam["vertical_pair_counting"] == "exact_all_pairs"
        assert fam["moving_tip_interval"]["pair_counting"] == "exact_all_pairs"
        assert fam["vertical_intervals"][-1]["strict_pair_count"] > 0


def test_three_dimensional_family_moves_toward_one_tenth(receipt: dict) -> None:
    fractions = {}
    for level in receipt["levels"]:
        fam = _family(receipt, level["fibonacci_index"], 3)
        fractions[fam["q"]] = fam["vertical_intervals"][-1]["ordering_fraction_float"]
    assert abs(fractions[55] - 0.1) < abs(fractions[5] - 0.1)
    assert abs(fractions[55] - 0.1) < 0.02
    one = _family(receipt, 10, 1)["vertical_intervals"][-1]["ordering_fraction_float"]
    two = _family(receipt, 10, 2)["vertical_intervals"][-1]["ordering_fraction_float"]
    assert abs(one - 0.5) < 0.02 and abs(two - 8 / 35) < 0.02
