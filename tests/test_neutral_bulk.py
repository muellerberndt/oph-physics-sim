from __future__ import annotations

import numpy as np

from oph_fpe.bulk.neutral_bulk import (
    build_neutral_observer_views,
    neutral_distance_matrix,
    neutral_leakage_audit,
    neutral_model_selection,
    planted_neutral_control_report,
    strict_neutral_bulk_receipt,
    strict_neutral_bulk_report,
)


def test_build_neutral_observer_views_ignores_forbidden_geometry_fields():
    observer_views = [
        _observer_view(
            1,
            records=[1, 2, 3, 1],
            checkpoints=[4, 4, 5, 5],
            sectors=[0, 1, 1, 2],
            repairs=[2, 3, 2, 3],
            axis=[1.0, 0.0, 0.0],
            support_nodes=[10, 11],
            h3_point=[0.1, 0.2, 0.3],
            cap_axis=[0.0, 1.0, 0.0],
            radial_depth=17.0,
            modular_depth=23.0,
        )
    ]

    views = build_neutral_observer_views(observer_views)

    assert len(views) == 1
    view = views[0]
    assert view.record_transition_hist.shape == (32,)
    assert view.checkpoint_transition_hist.shape == (32,)
    assert view.sector_transition_hist.shape == (6,)
    assert view.repair_response_hist.shape == (16,)
    assert np.isclose(view.record_transition_hist.sum(), 1.0)

    # If forbidden geometry fields were used, this radically different copy
    # would change the neutral feature vectors. It must not.
    changed_geometry = [
        {
            **observer_views[0],
            "axis": [0.0, 0.0, 1.0],
            "support_nodes": [999],
            "h3_point": [9.0, 9.0, 9.0],
            "cap_axis": [1.0, 0.0, 0.0],
            "radial_depth": 999.0,
            "modular_depth": 999.0,
        }
    ]
    changed_views = build_neutral_observer_views(changed_geometry)

    assert np.allclose(view.record_transition_hist, changed_views[0].record_transition_hist)
    assert np.allclose(view.checkpoint_transition_hist, changed_views[0].checkpoint_transition_hist)
    assert np.allclose(view.sector_transition_hist, changed_views[0].sector_transition_hist)
    assert np.allclose(view.repair_response_hist, changed_views[0].repair_response_hist)


def test_neutral_distance_matrix_is_symmetric_and_zero_diagonal():
    observer_views = [
        _observer_view(1, records=[1, 1, 2], checkpoints=[1], sectors=[0, 1], repairs=[1]),
        _observer_view(2, records=[2, 3, 3], checkpoints=[2], sectors=[1, 1], repairs=[2]),
        _observer_view(3, records=[7, 8, 9], checkpoints=[3], sectors=[2, 2], repairs=[3]),
    ]
    views = build_neutral_observer_views(observer_views)

    distance = neutral_distance_matrix(views)

    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance[np.triu_indices(3, k=1)] >= 0.0)


def test_neutral_leakage_audit_is_posthoc_and_flags_primary_no_geometry_use():
    observer_views = [
        _observer_view(1, records=[1, 2], checkpoints=[1], sectors=[0], repairs=[1], axis=[1.0, 0.0, 0.0]),
        _observer_view(2, records=[2, 3], checkpoints=[2], sectors=[1], repairs=[2], axis=[0.0, 1.0, 0.0]),
        _observer_view(3, records=[3, 4], checkpoints=[3], sectors=[2], repairs=[3], axis=[0.0, 0.0, 1.0]),
    ]
    views = build_neutral_observer_views(observer_views)
    distance = neutral_distance_matrix(views)

    audit = neutral_leakage_audit(distance, observer_views)

    assert audit["h3_coordinates_used"] is False
    assert audit["cap_normals_used"] is False
    assert audit["screen_axes_used_in_primary_distance"] is False
    assert "s2_distance_correlation" in audit


def test_strict_neutral_bulk_receipt_requires_refinement():
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement={"stable_across_64k_256k_1m": False},
    )

    assert receipt["receipt"] == "STRICT_NEUTRAL_BULK_RECEIPT"
    assert receipt["strict_neutral_bulk"] is False
    assert receipt["physical_claim"] is False


def test_strict_neutral_bulk_receipt_can_pass_all_gates():
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement={"stable_across_64k_256k_1m": True},
    )

    assert receipt["strict_neutral_bulk"] is True
    assert receipt["physical_claim"] is True


def test_current_rank3_candidate_not_strict_without_refinement():
    report = strict_neutral_bulk_report(
        [_observer_view(i, records=[i, i + 1, i + 2], checkpoints=[i], sectors=[i % 3], repairs=[i % 5]) for i in range(12)],
        model_selection={"best_model": "H3", "h3_beats_s2": True, "h3_beats_h2_h4": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement={"stable_across_64k_256k_1m": False},
    )

    assert report["strict_neutral_bulk"] is False
    assert report["receipt"]["strict_neutral_bulk"] is False
    assert "H3 fitted points" in report["forbidden_primary_features"]


def test_neutral_model_selection_reports_required_metric_families():
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(48, 3))
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    report = neutral_model_selection(distance, seed=2, max_points=48)

    assert set(report["models"]) == {"S2", "E2", "E3", "H2", "H3", "H4"}
    assert report["best_model"] in report["models"]
    assert report["fit_observer_count"] == 48
    assert report["heldout_pair_count"] > 0


def test_planted_neutral_controls_are_diagnostic_and_include_hard_flags():
    report = planted_neutral_control_report(point_count=48, seed=3, max_points=48)

    assert set(report["rows"]) == {"planted_2d", "planted_3d", "planted_4d", "planted_h3"}
    assert "planted_2d_returns_2d" in report["controls"]
    assert "planted_3d_returns_3d" in report["controls"]
    assert "planted_4d_returns_4d" in report["controls"]
    assert "planted_h3_returns_h3" in report["controls"]
    assert isinstance(report["controls"]["planted_h3_returns_h3"], bool)


def _observer_view(
    observer_id: int,
    *,
    records: list[int],
    checkpoints: list[int],
    sectors: list[int],
    repairs: list[int],
    axis: list[float] | None = None,
    support_nodes: list[int] | None = None,
    h3_point: list[float] | None = None,
    cap_axis: list[float] | None = None,
    radial_depth: float | None = None,
    modular_depth: float | None = None,
) -> dict:
    steps = []
    count = max(len(records), len(checkpoints), len(sectors), len(repairs))
    for index in range(count):
        steps.append(
            {
                "record_family": records[index % len(records)],
                "checkpoint_class": checkpoints[index % len(checkpoints)],
                "s3_sector_class": sectors[index % len(sectors)],
                "repair_load_bucket": repairs[index % len(repairs)],
            }
        )
    return {
        "view_type": "patch_observer",
        "observer_id": observer_id,
        "axis": axis,
        "support_nodes": support_nodes or [],
        "h3_point": h3_point,
        "cap_axis": cap_axis,
        "radial_depth": radial_depth,
        "modular_depth": modular_depth,
        "transition_history_descriptor": {"steps": steps},
        "counterfactual_continuation_hist": [1.0, 2.0, 3.0],
        "record_persistence": 0.75,
        "sector_persistence": 0.5,
        "stable_fraction": 0.8,
    }
