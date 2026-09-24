"""Carrier source-net lane (A-real): placement, readback metric, neighbours, provenance order, RER digests, verifier."""

from __future__ import annotations

import copy
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from oph_exact import carrier
from oph_exact import carrier_source_net as producer
from oph_exact import source_net
from oph_exact import verify_carrier_source_net_independent as verifier

RER_AVAILABLE = (producer.RER_ROOT / producer.RER_RECEIPT).is_file()
requires_rer = pytest.mark.skipif(
    not RER_AVAILABLE,
    reason="theory checkout absent (set OPH_RER_ROOT to reverse-engineering-reality)",
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/exact/carrier_source_net_receipt.json"
Q5_NEIGHBOUR_DIGEST = "cf3d2ee0d16f1e6659c49e31c34b0901ac709ae78d55d8584acb441a3bb83f13"
Q5_RECORDS_DIGEST = "cc8aa7f94012d3bc64108c89a827a64e62c2660ab9f91c3398bd5a226a80f240"
Q5_RER_FORWARD_CHAIN = "35b50b2a2883e7e02081db475591bde25f1ea0879ee9a42072c960ef07abde5c"
Q5_RER_FORWARD_LAYERS = [
    "5a686ecbfee8721c2a66eee6809949708a429e4091178856eb38ea0ca47ba43a",
    "ec1aba64c6884566d6fcb66c47f0f24f018504d4c276d75f5b3b3795e2fa041e",
    "8f313df4233e647ca244edf7935d6d3848568ff72498c1075708a4c52ae4aeb4",
    "9d6ca4a045e6d8b19e83de3317c314843ed018782d570a391d88d177f865296b",
]
Q5_RER_FORWARD_SUMS = [7875, 185593, 4865047, 130604068]
Q5_RER_INTERVENTION_CHAIN = "78a923711b5560f38baf5bd8d98e3f74fda444dbf6b8ea8e2cd8698bf9ede5be"
Q5_RER_INTERVENTION_SUMS = [7876, 185632, 4866313, 130640453]
Q5_RER_SUPPORTS = [
    ("c832d78ce28ff0b3e45bcac5d5b58a49b8286312413ee77721b1482074e945a3", 1, 1),
    ("bedd14d46af4bcff2f82ca606b5f5118c47adc1e8bd5231ca45450757131859a", 39, 39),
    ("96ca4a2b98bb4a2eddac5f871f2c55c1aec2f25846f682442c3420b43a83b082", 125, 1266),
    ("96ca4a2b98bb4a2eddac5f871f2c55c1aec2f25846f682442c3420b43a83b082", 125, 36385),
]
Q5_CENTRE = 124
Q5_STRICT_PAIRS = {1: 1, 2: 79, 3: 968}
Q5_COUNTS = [1, 39, 39, 1]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_bytes())


@pytest.fixture(scope="module")
def level5(tmp_path_factory) -> dict:
    frozen = producer.source_net_families()[5]
    rer = {lv["q"]: lv for lv in producer.rer_receipt()["levels"]}[5]
    return producer.build_level(5, frozen, rer, processes=1, workdir=tmp_path_factory.mktemp("csn"), keep_log=False)


def _level(receipt: dict, q: int) -> dict:
    return next(lv for lv in receipt["levels"] if lv["q"] == q)


# --------------------------------------------------------------------------
# Load placement and readback
# --------------------------------------------------------------------------


def test_load_placement_round_trip() -> None:
    sites, z, currents = producer.source_records(5)
    loads = producer.place_loads(z)
    assert np.array_equal(producer.antipodal_odd_quotient(loads), z)
    assert loads.min() >= 0 and int(loads.sum()) == int(np.abs(z).sum())
    anti = carrier.antipode()
    for k, p in enumerate(producer.POSITIVE_PORTS):
        assert np.all((loads[:, p] == 0) | (loads[:, anti[p]] == 0))
    rng = np.random.default_rng(1)
    random_z = rng.integers(-40, 41, size=(200, 6))
    assert np.array_equal(producer.antipodal_odd_quotient(producer.place_loads(random_z)), random_z)
    assert np.all(z.sum(axis=1) % 2 == 0)
    assert np.array_equal(z.sum(axis=1), 2 * sites.sum(axis=1))


def test_readback_is_the_signed_generator_sum_and_only_the_odd_part_moves_it() -> None:
    sites, z, _ = producer.source_records(5)
    loads = producer.place_loads(z)
    x = producer.readback(loads)
    assert np.max(np.abs(x - z.astype(float) @ producer.readback_generators())) < 1e-9
    anti = carrier.antipode()
    even = np.zeros(12)
    even[2] = even[anti[2]] = 7.0
    even[5] = even[anti[5]] = -3.0
    shifted = producer.readback(loads + even[None, :].astype(np.int64))
    assert np.max(np.abs(shifted - x)) < 1e-9
    assert np.linalg.matrix_rank(x, tol=1e-9) == 3


def test_readback_metric_equals_source_metric_up_to_the_global_scale_at_q5() -> None:
    q = 5
    sites, z, _ = producer.source_records(q)
    loads = producer.place_loads(z)
    x = producer.readback(loads)
    sig = producer.gram_sign_pattern()
    identity = producer.metric_identity_all_or_sampled(sites, z, sig, q)
    assert identity == {"mode": "all_pairs", "pairs_checked": 125 * 124 // 2, "exact": True, "seed": None}
    l2 = carrier.q5_float(producer.L_SQUARED_QSQRT5)
    assert abs(l2 - (12 / 5 - (4 / 5) * (1 + 5 ** 0.5) / 2)) < 1e-15
    A, B = source_net.axis_tables(source_net.orbit(q))
    for i in range(0, 125, 7):
        for j in range(125):
            source = sum(source_net.phi_float((int(A[sites[i, ax], sites[j, ax]]), int(B[sites[i, ax], sites[j, ax]])))
                         for ax in range(3))
            assert abs(np.sum((x[i] - x[j]) ** 2) - l2 * source) < 1e-9
    checks = producer.exact_sample_checks(sites, z, loads, producer.sample_sites(125, Q5_CENTRE))
    assert checks["exact_gram_identity"] and checks["exact_position_equals_paper_contraction"]
    assert checks["exact_metric_identity"] and checks["exact_metric_identity_pairs"] == 45
    assert producer.generator_frame_is_isometric()


def test_neighbour_digest_equality_at_q5() -> None:
    q = 5
    sites, z, _ = producer.source_records(q)
    x3 = producer.readback(producer.place_loads(z)) @ producer.chart_basis()
    nb = producer.find_neighbours(x3, z, q, producer.gram_sign_pattern())
    assert producer.neighbour_digest(nb["indptr"], nb["indices"]) == Q5_NEIGHBOUR_DIGEST
    indptr, indices, *_ = source_net.build_site_graph(q, 3)
    assert np.array_equal(nb["indptr"], indptr) and np.array_equal(nb["indices"], indices)
    assert nb["rejected_candidates"] == 0 and nb["accepted_pairs"] == 1330
    # The exact decision on its own: one pair inside and one pair outside the radius, decided exactly.
    dz = np.array([z[0] - z[0], z[0] - z[124]])
    assert producer.neighbour_decision(dz, q, producer.gram_sign_pattern()).tolist() == [True, False]


# --------------------------------------------------------------------------
# Reads, provenance order, intervention, RER digests
# --------------------------------------------------------------------------


@requires_rer
def test_rer_trace_digests_at_q5(level5: dict) -> None:
    fwd = level5["reads"]["forward"]
    assert fwd["audit_trace_sha256"] == Q5_RER_FORWARD_CHAIN
    assert fwd["layer_value_sha256"] == Q5_RER_FORWARD_LAYERS
    assert fwd["layer_value_sums"] == Q5_RER_FORWARD_SUMS
    assert fwd["event_count"] == 500 and fwd["authenticated_read_count"] == 8355
    itv = level5["reads"]["intervention"]
    assert itv["audit_trace_sha256"] == Q5_RER_INTERVENTION_CHAIN
    assert itv["layer_value_sums"] == Q5_RER_INTERVENTION_SUMS
    assert level5["reads"]["rer_cross_check"]["all_agree"] is True
    assert level5["records"]["source_records_sha256"] == Q5_RECORDS_DIGEST


@requires_rer
def test_provenance_order_equals_layered_order_at_q5(level5: dict) -> None:
    pv = level5["provenance"]
    for key in ("single_writer", "all_reads_resolved", "parents_precede_children", "derived_rank_equals_round",
                "read_relation_identical_across_rounds", "read_relation_equals_neighbour_digest"):
        assert pv[key] is True
    ci = pv["centre_interval"]
    assert ci["inclusive_event_count"] == 80 and ci["counts_by_round"] == Q5_COUNTS
    assert ci["equals_layered_order"] and ci["equals_source_net_layered_order"]
    assert ci["strict_pair_count"] == Q5_STRICT_PAIRS[3] and ci["equals_source_net_strict_pair_count"]
    assert pv["edge_count"] == 3 * 2785
    # Brute force from the log alone: dictionary provenance, explicit descendant sets.
    indptr, indices, *_ = source_net.build_site_graph(5, 3)
    run = producer.run_reads(indptr, indices, 3, chain=False)
    log = producer.EventLog(indptr, indices, 3)
    n = 125
    writer_of = {}
    for e, (register, version, _value) in enumerate(producer.write_records(run["values"])):
        assert (register, version) not in writer_of
        writer_of[(register, version)] = e
    parents = [[] for _ in range(4 * n)]
    for j, readers, registers, versions in log.forward():
        for reader, register, version in zip(readers.tolist(), registers.tolist(), versions.tolist()):
            parents[j * n + reader].append(writer_of[(register, version)])
    children = [[] for _ in range(4 * n)]
    for e, ps in enumerate(parents):
        for p in ps:
            children[p].append(e)
    desc = [set() for _ in range(4 * n)]
    for e in range(4 * n - 1, -1, -1):
        for c in children[e]:
            desc[e].add(c)
            desc[e] |= desc[c]
    seed, top = Q5_CENTRE, 3 * n + Q5_CENTRE
    members = sorted(e for e in desc[seed] | {seed} if e == top or top in desc[e])
    inside = set(members)
    assert len(members) == 80
    assert sum(len(desc[e] & inside) for e in members) == Q5_STRICT_PAIRS[3]
    assert producer.digest([[e // n, e % n] for e in members]) == ci["event_set_sha256"]
    for k, C in Q5_STRICT_PAIRS.items():
        topk = k * n + Q5_CENTRE
        mem = sorted(e for e in desc[seed] | {seed} if e == topk or topk in desc[e])
        ins = set(mem)
        assert sum(len(desc[e] & ins) for e in mem) == C
        row = next(r for r in level5["manifold"]["intervals"] if r["layers"] == k)
        assert row["strict_pair_count"] == C and row["equals_source_net"] is True


@requires_rer
def test_intervention_support_equals_future_cone_and_rer_supports(level5: dict) -> None:
    rows = level5["intervention"]["rounds"]
    assert level5["intervention"]["equals_future_cone_all_rounds"] is True
    assert level5["intervention"]["equals_source_net_reachable_ids_all_rounds"] is True
    for row, (sha, count, delta_sum) in zip(rows, Q5_RER_SUPPORTS):
        assert row["support_ids_sha256"] == sha and row["support_count"] == count
        assert row["positive_integer_delta_sum"] == delta_sum and row["equals_rer_support"] is True
        assert row["equals_future_cone"] is True and row["all_deltas_nonnegative"] is True
    assert level5["provenance"]["future_cone_counts"] == [1, 39, 125, 125]


@requires_rer
def test_manifold_readouts_and_costs_at_q5(level5: dict, receipt: dict) -> None:
    frozen = producer.source_net_families()[5]
    for row in level5["manifold"]["intervals"]:
        fr = next(r for r in frozen["vertical_intervals"] if r["layers"] == row["layers"])
        assert row["ordering_fraction"] == fr["ordering_fraction"]
        assert row["myrheim_meyer_dimension"] == fr["myrheim_meyer_dimension"]
    top = level5["manifold"]["intervals"][-1]
    assert top["ordering_fraction"] == str(Fraction(2 * 968, 80 * 79))
    assert level5["manifold"]["count_clock"]["equals_source_net"] is True
    oc = level5["operation_costs"]
    assert oc["total_reads"] == 3 * 2785 and oc["total_writes"] == 500
    assert oc["per_round"][0]["reads"] == 0 and all(r["reads"] == 2785 for r in oc["per_round"][1:])
    assert oc["sum_word_lengths"] == 1890 and oc["maximum_word_length"] == 32
    assert oc["total_bytes"] == oc["total_read_bytes"] + oc["total_write_bytes"] > 0
    assert oc["per_round"][1]["read_bytes"] == sum(
        (2 * producer.ID_BYTES + producer.VERSION_BYTES + producer.byte_length(i + 1)) * deg
        for i, deg in enumerate(np.diff(source_net.build_site_graph(5, 3)[0]).tolist()))


# --------------------------------------------------------------------------
# Frozen receipt
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
    assert [lv["q"] for lv in receipt["levels"]] == list(producer.LEVELS)
    assert receipt["readback_metric"]["scale_to_source_net_units_L_squared_qsqrt5"] == ["2", "-2/5"]
    assert receipt["readback_metric"]["scale_to_paper_position_s"] == "1"
    assert "work in progress" in receipt["dynamic_population_note"]
    assert all(v is False for v in receipt["scope"]["not_claimed"].values())
    assert receipt["rer_cross_check"]["all_agree"] is True and receipt["rer_cross_check"]["levels_compared"] == [5, 8, 13]
    assert all(v is True for k, v in receipt["source_net_cross_check"].items() if k != "receipt")
    for lv in receipt["levels"]:
        assert lv["neighbours"]["equals_source_net_digest"] is True
        assert lv["provenance"]["centre_interval"]["equals_source_net_strict_pair_count"] is True
        assert lv["manifold"]["all_intervals_equal_source_net"] is True


@requires_rer
def test_q5_level_rebuilds_deterministically_to_the_frozen_row(receipt: dict, level5: dict) -> None:
    frozen = {k: v for k, v in _level(receipt, 5).items() if k != "stored_log"}
    mine = {k: v for k, v in level5.items() if k != "stored_log"}
    assert producer.canonical(mine) == producer.canonical(frozen)


def test_pins_match_the_working_tree(receipt: dict) -> None:
    for rel in producer.LOCAL_PINS:
        assert receipt["source_pins"][rel] == producer.file_sha256(ROOT / rel)
    for q in producer.STORED_LOG_LEVELS:
        rel = f"data/exact/carrier_source_net_logs/q{q}_event_log.json.gz"
        assert receipt["source_pins"][rel] == producer.file_sha256(ROOT / rel)
    assert receipt["rer_cross_check"]["receipt_sha256"] == producer.RER_RECEIPT_SHA256


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def test_verifier_accepts_the_committed_receipt() -> None:
    result = verifier.verify(verifier.load(RECEIPT))
    assert result["accepted"] is True
    assert result["rebuilt_levels"] == [5, 8]
    assert [row["q"] for row in result["levels"]] == [5, 8, 13, 21, 34, 55, 89]


def _mutations(receipt: dict):
    r = copy.deepcopy(receipt)
    r["schema"] = "oph.exact.carrier-source-net.v0"
    yield "schema", r
    r = copy.deepcopy(receipt)
    r["scope"]["not_claimed"]["spacetime"] = True
    yield "scope", r
    r = copy.deepcopy(receipt)
    r["source_pins"]["oph_exact/carrier.py"] = "0" * 64
    yield "pin", r
    r = copy.deepcopy(receipt)
    _level(r, 5)["provenance"]["centre_interval"]["strict_pair_count"] += 1
    yield "pair_count", r
    r = copy.deepcopy(receipt)
    _level(r, 8)["neighbours"]["neighbors_including_wait_sha256"] = "0" * 64
    yield "digest", r
    r = copy.deepcopy(receipt)
    _level(r, 5)["reads"]["forward"]["audit_trace_sha256"] = "0" * 64
    yield "audit_chain", r
    r = copy.deepcopy(receipt)
    _level(r, 8)["intervention"]["rounds"][2]["support_count"] += 1
    yield "intervention", r
    r = copy.deepcopy(receipt)
    _level(r, 13)["provenance"]["centre_interval"]["inclusive_event_count"] += 1
    yield "event_count", r
    r = copy.deepcopy(receipt)
    _level(r, 21)["provenance"]["future_cone_sha256"][2] = "0" * 64
    yield "future_cone", r
    r = copy.deepcopy(receipt)
    _level(r, 34)["manifold"]["count_clock"]["count_clock"] *= 1.01
    yield "clock", r
    r = copy.deepcopy(receipt)
    _level(r, 13)["operation_costs"]["per_round"][2]["read_bytes"] += 1
    yield "costs", r
    r = copy.deepcopy(receipt)
    _level(r, 21)["manifold"]["intervals"][-1]["ordering_fraction"] = "1/10"
    yield "fraction", r
    r = copy.deepcopy(receipt)
    _level(r, 34)["provenance"]["derived_rank_equals_round"] = False
    yield "flag", r
    r = copy.deepcopy(receipt)
    r["carrier"]["gram_sign_pattern"][0][1] *= -1
    yield "sign_pattern", r
    r = copy.deepcopy(receipt)
    r["readback_metric"]["scale_to_source_net_units_L_squared_qsqrt5"] = ["2", "2/5"]
    yield "scale", r
    r = copy.deepcopy(receipt)
    r["source_net_cross_check"]["neighbour_digests_equal"] = False
    yield "summary", r


def test_verifier_rejects_mutated_receipts(receipt: dict) -> None:
    for name, mutated in _mutations(receipt):
        with pytest.raises(verifier.CarrierSourceNetVerificationError):
            verifier.verify(mutated)


def test_verifier_does_not_import_the_producer_or_the_carrier_module() -> None:
    source = (ROOT / "oph_exact/verify_carrier_source_net_independent.py").read_text(encoding="utf-8")
    assert "carrier_source_net import" not in source and "import carrier_source_net" not in source
    assert "from oph_exact" not in source and "import oph_exact" not in source
    assert "from oph_fpe" not in source and "import oph_fpe" not in source
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert all(("oph_" not in line) for line in imports), imports


def test_strict_loader_rejects_noncanonical_and_nonfinite(tmp_path: Path) -> None:
    for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a": 1}\n'):
        path = tmp_path / "bad.json"
        path.write_bytes(data)
        with pytest.raises(verifier.CarrierSourceNetVerificationError):
            verifier.load(path)
