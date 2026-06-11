from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from oph_fpe.bulk.h3_chart import h3_distance_matrix, random_h3_points
from oph_fpe.bulk.latent_geometry_selection import (
    select_latent_geometry,
    strict_neutral_latent_geometry_gate,
)


def test_latent_geometry_selection_recovers_planted_h3():
    distance = h3_distance_matrix(random_h3_points(96, seed=12, radius=1.25))

    report = select_latent_geometry(distance, seed=17, max_points=96, heldout_fraction=0.20)

    assert report["mode"] == "strict_neutral_heldout_latent_geometry_selection_v0"
    assert report["selected_model"] == "H3"
    assert report["h3_selected"] is True
    assert report["STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT"] is True


def test_latent_geometry_selection_recovers_three_dimensional_euclidean_family():
    rng = np.random.default_rng(41)
    coords = rng.normal(size=(96, 3))
    distance = squareform(pdist(coords, metric="euclidean"))

    report = select_latent_geometry(distance, seed=42, max_points=96, heldout_fraction=0.20)

    assert report["selected_dimension"] == 3
    assert report["selected_model"] in {"E3", "H3"}


def test_strict_neutral_latent_geometry_gate_requires_h3_fraction():
    report = strict_neutral_latent_geometry_gate(
        [
            {"h3_selected": True},
            {"h3_selected": True},
            {"h3_selected": True},
            {"h3_selected": False},
        ],
        min_fraction=0.75,
    )

    assert report["h3_selected_fraction"] == 0.75
    assert report["STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT"] is True

    failed = strict_neutral_latent_geometry_gate(
        [{"h3_selected": True}, {"h3_selected": False}],
        min_fraction=0.75,
    )

    assert failed["STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT"] is False


def test_latent_geometry_selector_rejects_tiny_distance_matrix():
    report = select_latent_geometry(np.zeros((4, 4)), seed=1)

    assert report["h3_selected"] is False
    assert report["selected_model"] is None
    assert report["STRICT_NEUTRAL_LATENT_H3_SELECTION_RECEIPT"] is False

