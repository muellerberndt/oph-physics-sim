"""The exploratory sourced read law reduces to the declared law at g = 0 and its geometry matches the exact lane."""

from __future__ import annotations

import json

import numpy as np

from oph_exact import sourced_read_law as S
from oph_exact.source_net import OUTPUT as RECEIPT


def test_windowed_neighbour_counts_match_the_exact_lane() -> None:
    receipt = json.loads(RECEIPT.read_text())
    for row in receipt["levels"][:2]:  # q = 5 and q = 8
        fam = row["families"][0]
        pop = S.Population(row["q"], periodic=False)
        assert int(pop.fixed_counts.min()) == fam["minimum_neighbor_count_including_wait"]
        assert int(pop.fixed_counts.max()) == fam["maximum_neighbor_count_including_wait"]
        assert int(pop.fixed_counts.sum() - pop.n) == 2 * fam["undirected_spatial_edges"]


def test_g_zero_is_the_declared_law() -> None:
    pop = S.Population(8)
    n = S.step(pop, pop.fixed_counts * 3.0, g=0.0, beta=1.0, mode="physical", cap=None)
    assert np.array_equal(n, pop.fixed_counts.astype(float))
    r = S.run(pop, g=0.0, beta=1.0, rounds=3, initial="seeded")
    assert r["history"][1]["contrast_sd"] == r["history"][3]["contrast_sd"]


def test_linear_prediction_values() -> None:
    p = S.linear_prediction(1.0, 1.0)
    assert abs(p["lambda_short"] - 0.625) < 1e-12 and abs(p["lambda_long"] - 0.25) < 1e-12
    assert abs(S.linear_prediction(4.0 / 2.5, 1.0)["lambda_short"] - 1.0) < 1e-12


def test_uniform_map_is_exact_power_law() -> None:
    pop = S.Population(8)
    n0 = np.full(pop.n, 2.0 * pop.N0)
    # on a uniform field every site reads the fixed neighbourhood, so n -> N(s) (n/N0)^(beta g / 4)
    n1 = S.step(pop, n0, g=2.0, beta=1.0, mode="physical", cap=None)
    expected = pop.fixed_counts * (2.0) ** (2.0 * 1.0 / 4.0)
    assert np.allclose(n1, expected)


def test_capacity_binds() -> None:
    pop = S.Population(8)
    r = S.run(pop, g=3.0, beta=1.0, rounds=12, cap=2.0 * pop.N0, initial="native")
    assert r["history"][-1]["at_capacity_fraction"] > 0.0
    assert all(h["mean_records"] <= 2.0 * pop.N0 + 1e-9 for h in r["history"])
