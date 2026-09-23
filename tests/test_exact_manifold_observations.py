"""Manifold-observations lane: frozen receipt, q = 8 rebuild, profile and action identities, multipoles, verifier."""

from __future__ import annotations

import copy
import json
import math
import re
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from oph_exact import manifold_observations as producer
from oph_exact import verify_manifold_observations_independent as verifier

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/exact/manifold_observations_receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_bytes())


@pytest.fixture(scope="module")
def q8_rebuild() -> tuple[dict, dict]:
    references: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        fam = producer.build_family(6, 3, 1, Path(tmp), references)
    return fam, references


def _family(receipt: dict, n: int, dim: int) -> dict:
    for level in receipt["levels"]:
        if level["fibonacci_index"] == n:
            return next(f for f in level["families"] if f["dimension"] == dim)
    raise KeyError((n, dim))


def _future(relations: list[tuple[int, int]], n: int) -> np.ndarray:
    """Strict future matrix of the transitive closure of the given relations."""
    F = np.zeros((n, n), dtype=bool)
    for a, b in relations:
        F[a, b] = True
    for k in range(n):
        F |= F[:, k][:, None] & F[k, :][None, :]
    return F


# --------------------------------------------------------------------------
# Frozen receipt
# --------------------------------------------------------------------------


def test_receipt_is_canonical_frozen_and_free_of_wall_clock(receipt: dict) -> None:
    data = RECEIPT.read_bytes()
    assert producer.canonical(receipt) == data
    assert receipt["schema"] == producer.SCHEMA
    assert receipt["scope"] == producer.SCOPE
    assert receipt["claim_boundary"] == producer.CLAIM_BOUNDARY
    assert receipt["certificate_rows"] == producer.CERTIFICATE_ROWS
    statuses = {k: v["status"] for k, v in receipt["certificate_rows"].items()}
    assert statuses["C1_physical_event_interpretation"] == "not addressed"
    assert statuses["C6_distinguishing_lorentzian_geometry_with_uniqueness_control"] == "declared"
    assert all(v == "computed" for k, v in statuses.items() if k[:2] in ("C2", "C3", "C4", "C5"))
    text = data.decode("ascii")
    assert re.search(r"20\d\d-\d\d-\d\dT", text) is None
    assert "timestamp" not in text and "wall_clock" not in text
    assert [lv["fibonacci_index"] for lv in receipt["levels"]] == list(producer.LEVELS)
    for lv in receipt["levels"]:
        assert [f["dimension"] for f in lv["families"]] == list(producer.DIMENSIONS)


def test_pins_match_the_working_tree(receipt: dict) -> None:
    pins = receipt["source_pins"]
    assert set(pins) == set(producer.PINS)
    for rel, sha in pins.items():
        assert producer.file_sha256(ROOT / rel) == sha, rel


def test_style_of_status_strings(receipt: dict) -> None:
    text = json.dumps(receipt["claim_boundary"]) + json.dumps(receipt["certificate_rows"]) + json.dumps(receipt["conventions"])
    assert "\u2014" not in text and "---" not in text
    lowered = text.lower()
    for token in (" now ", " already ", " still ", "no longer", "not yet", " remains", "honest",
                  "crucially", "moreover", "delve", "serves as"):
        assert token not in lowered, token


# --------------------------------------------------------------------------
# q = 8 rebuild against the frozen rows
# --------------------------------------------------------------------------


def test_q8_family_rebuilds_to_the_frozen_block(receipt: dict, q8_rebuild: tuple[dict, dict]) -> None:
    fam, references = q8_rebuild
    frozen = _family(receipt, 6, 3)
    assert producer.canonical(fam) == producer.canonical(frozen)
    for key, ref in references.items():
        assert producer.canonical(ref) == producer.canonical(receipt["sprinkling_references"][key])


def test_q8_homogeneity_readouts(receipt: dict) -> None:
    fam = _family(receipt, 6, 3)
    sizes = {b["layers"]: b for b in fam["homogeneity"]["sizes"]}
    assert set(sizes) == {3, 2}
    two = sizes[2]
    assert two["diamond_count"] >= 5 and two["tip_rule"] == "buffer_positive"
    dims = [r["myrheim_meyer_dimension"] for r in two["diamonds"]]
    assert two["dimension_summary"]["count"] == len(dims)
    assert abs(two["dimension_summary"]["mean"] - np.mean(dims)) < 1e-9
    assert two["dimension_summary"]["max"] - two["dimension_summary"]["min"] < 0.2
    for row in two["diamonds"]:
        N = row["inclusive_event_count"]
        assert Fraction(row["ordering_fraction"]) == Fraction(2 * row["strict_pair_count"], N * (N - 1))
        assert row["continuum_diamond_inside_cube"] is True
    centre = fam["centre_vertical_intervals"][2]
    assert centre["layers"] == 3 and centre["inclusive_event_count"] == 188 and centre["strict_pair_count"] == 4244
    assert fam["source_net_cross_check"]["all_agree"] is True


# --------------------------------------------------------------------------
# Profile, action, multipoles, sampling
# --------------------------------------------------------------------------


def test_abundance_profile_identity_on_a_hand_built_diamond() -> None:
    # a < b, a < c, b < d, c < d: four links, one 4-element interval (a, d) with two events between.
    F = _future([(0, 1), (0, 2), (1, 3), (2, 3)], 4)
    prof = producer.between_profile_from_future(F)
    assert prof["counts"][:4] == [4, 0, 1, 0]
    assert prof["total_related_pairs"] == 5 and sum(prof["log2_bins_above_max_m"]) == 0
    assert producer.profile_ratios(prof["counts"])[:3] == [1.0, 0.0, 0.25]
    chain = _future([(i, i + 1) for i in range(4)], 5)
    prof = producer.between_profile_from_future(chain)
    assert prof["counts"][:5] == [4, 3, 2, 1, 0] and prof["total_related_pairs"] == 10
    bins = producer.profile_bins(np.array([0, 15, 16, 31, 32, 1023, 1024, 2 ** 19, 2 ** 25]))
    assert bins.tolist() == [0, 15, 16, 16, 17, 21, 22, 31, 31]


def test_benincasa_dowker_action_against_the_formula() -> None:
    chain = producer.benincasa_dowker_action(5, [4, 3, 2, 1])
    assert chain["bracket"] == 5 - 4 + 27 - 32 + 8 == 4
    assert abs(chain["action"] - 4.0 / math.sqrt(6.0) * 4) < 1e-9
    diamond = producer.benincasa_dowker_action(4, [4, 0, 1, 0])
    assert diamond["bracket"] == -16 and abs(diamond["action_over_N"] + 16 * 4.0 / math.sqrt(6.0) / 4) < 1e-9
    assert "(i+1)-element" in producer.BD_CONVENTION and "N_1 = links" in producer.BD_CONVENTION


def test_multipole_powers_of_symmetric_direction_sets() -> None:
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    p = producer.multipole_powers(axes)
    assert abs(p[0] - 1.0) < 1e-12
    assert max(p[1], p[2], p[3]) < 1e-12 and p[4] > 0.1
    g = (1.0 + math.sqrt(5.0)) / 2.0
    ico = np.array([[0, s1, s2 * g] for s1 in (1, -1) for s2 in (1, -1)]
                   + [[s1, s2 * g, 0] for s1 in (1, -1) for s2 in (1, -1)]
                   + [[s2 * g, 0, s1] for s1 in (1, -1) for s2 in (1, -1)], dtype=float)
    p = producer.multipole_powers(ico)
    assert max(p[1:5]) < 1e-12
    polar = np.array([[s * 0.1 * math.cos(t), s * 0.1 * math.sin(t), s] for t in np.linspace(0, 6, 40) for s in (1, -1)])
    p = producer.multipole_powers(polar)
    assert p[1] < 1e-12 and p[2] > 0.5
    rng = np.random.default_rng(3)
    n = rng.standard_normal((5000, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    blocks = producer.sphere_harmonics(n)
    for l, block in enumerate(blocks):
        assert np.allclose((block ** 2).sum(axis=0), 2 * l + 1)
    reference = producer.uniform_direction_reference(20000, 3, seeds=4)
    for l in range(1, 5):
        assert reference["power_mean"][l] < 4 * (2 * l + 1) / 20000


def test_sprinkling_reads_its_spacetime_dimension() -> None:
    for dim, expected in ((3, 4.0), (2, 3.0)):
        t, x = producer.sprinkle_diamond(1500, dim, 11)
        F = np.zeros((1500, 1500), dtype=bool)
        for lo in range(0, 1500, 500):
            F[lo:lo + 500] = producer.sprinkling_future_rows(t, x, np.arange(lo, lo + 500))
        f = 2 * int(F.sum()) / (1500 * 1499)
        assert abs(producer.mm_dimension(f) - expected) < 0.3
        assert np.all(np.linalg.norm(x, axis=1) <= np.minimum(t, 1.0 - t) + 1e-12)


def test_lower_event_between_counts_match_the_matrix_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fam = producer.Family(6, 3)
        fam.open_pool(1, Path(tmp))
        dd = fam.distances(fam.centre, fam.centre, fam.K, Path(tmp), "t")
    exact = producer.exact_profile(dd["D"], dd["a"], dd["g"], fam.K)
    total = np.zeros(producer.PROFILE_BINS, dtype=np.int64)
    for i in range(len(dd["a"])):
        for j in range(int(dd["a"][i]), fam.K - int(dd["g"][i]) + 1):
            total += producer.between_counts_from_lower_event(dd["D"], dd["g"], fam.K, j, i)
    assert total[:16].tolist() == exact["counts"] and total[16:].tolist() == exact["log2_bins_above_max_m"]


def test_stratified_totals_arithmetic() -> None:
    strata = {"a": [0, 1, 2, 3], "b": [4, 5]}
    chosen = {"a": np.array([0, 2]), "b": np.array([4, 5])}
    values = {0: [1, 0], 2: [3, 0], 4: [2, 1], 5: [4, 1]}
    est = producer.stratified_totals(strata, chosen, values, 2)
    assert est["estimate"] == ["14", "2"]
    s2 = ((1 + 9) - 16 / 2) / 1
    assert abs(est["standard_error"][0] - math.sqrt(16 * 0.5 * s2 / 2)) < 1e-9
    assert est["standard_error"][1] == 0.0 and est["sample_size"] == 4


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def test_verifier_accepts_the_committed_receipt() -> None:
    result = verifier.verify(verifier.load(RECEIPT))
    assert result["accepted"] is True and result["rebuilt_levels"] == [6, 7]
    assert len(result["families"]) == len(producer.LEVELS) * len(producer.DIMENSIONS)


def test_verifier_rejects_mutated_receipts(receipt: dict) -> None:
    fam = _family(receipt, 6, 3)
    mutations = []
    m = copy.deepcopy(receipt)
    _family(m, 6, 3)["centre_vertical_intervals"][2]["myrheim_meyer_dimension"] += 1e-6
    mutations.append(m)
    m = copy.deepcopy(receipt)
    _family(m, 6, 3)["centre_vertical_intervals"][2]["strict_pair_count"] += 1
    mutations.append(m)
    m = copy.deepcopy(receipt)
    _family(m, 6, 3)["homogeneity"]["sizes"][1]["dimension_summary"]["mean"] += 1e-6
    mutations.append(m)
    m = copy.deepcopy(receipt)
    _family(m, 6, 3)["abundance"]["counts"][0] += 1
    mutations.append(m)
    m = copy.deepcopy(receipt)
    _family(m, 6, 3)["link_directions"]["power_relative_to_degree_zero"][3] *= 1.001
    mutations.append(m)
    m = copy.deepcopy(receipt)
    m["certificate_rows"]["C6_distinguishing_lorentzian_geometry_with_uniqueness_control"]["status"] = "computed"
    mutations.append(m)
    m = copy.deepcopy(receipt)
    m["source_pins"]["oph_exact/source_net.py"] = "0" * 64
    mutations.append(m)
    assert fam["abundance"]["mode"] == "exact_between_count_matrix"
    for mutated in mutations:
        with pytest.raises(verifier.ManifoldObservationsVerificationError):
            verifier.verify(mutated)


def test_verifier_does_not_import_the_producer() -> None:
    source = (ROOT / "oph_exact/verify_manifold_observations_independent.py").read_text()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports
    assert not any("oph_exact" in line or "manifold_observations" in line or "source_net" in line or "oph_fpe" in line
                   for line in imports)


def test_strict_loader_rejects_noncanonical_receipts(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema": producer.SCHEMA}, indent=2))
    with pytest.raises(verifier.ManifoldObservationsVerificationError):
        verifier.load(path)
