from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.bulk.neutral_object_bulk import (
    extract_neutral_objects,
    neutral_object_distance_matrix,
    strict_neutral_object_bulk_report,
    write_strict_neutral_object_bulk_report,
)


def test_extract_neutral_objects_ignores_geometry_fields():
    rows = [_observer_view(i, group=i // 4) for i in range(16)]
    changed = [
        {
            **row,
            "axis": [0.0, 0.0, 1.0],
            "support_nodes": [9999, int(row["observer_id"])],
            "h3_point": [9.0, 9.0, 9.0],
            "cap_axis": [1.0, 0.0, 0.0],
            "radial_depth": 1000.0,
            "modular_depth": 2000.0,
        }
        for row in rows
    ]

    objects = extract_neutral_objects(rows, min_observers_per_object=3)
    changed_objects = extract_neutral_objects(changed, min_observers_per_object=3)

    assert len(objects) == len(changed_objects) >= 4
    assert [obj.visible_signature_key for obj in objects] == [
        obj.visible_signature_key for obj in changed_objects
    ]
    for obj, changed_obj in zip(objects, changed_objects):
        assert np.allclose(obj.record_lineage_hist, changed_obj.record_lineage_hist)
        assert np.allclose(obj.sector_transport_hist, changed_obj.sector_transport_hist)


def test_neutral_object_distance_matrix_is_symmetric():
    rows = [_observer_view(i, group=i // 4) for i in range(20)]
    objects = extract_neutral_objects(rows, min_observers_per_object=3)

    distance = neutral_object_distance_matrix(objects)

    assert distance.shape == (len(objects), len(objects))
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance[np.triu_indices(len(objects), k=1)] >= 0.0)


def test_vector_sector_signatures_do_not_collapse_to_zero_bucket():
    rows = []
    for i in range(18):
        group = i // 6
        row = _observer_view(i, group=group)
        row["sector_change_signature"] = [0.0, 0.0, 0.0]
        row["sector_change_signature"][group] = 1.0
        row["transition_history_histograms"] = {
            key: value
            for key, value in row["transition_history_histograms"].items()
            if key != "s3_sector_class"
        }
        rows.append(row)

    objects = extract_neutral_objects(rows, min_observers_per_object=3)

    assert len(objects) >= 3
    assert len({obj.visible_signature_key.split(":")[2] for obj in objects}) >= 3


def test_strict_neutral_object_bulk_blocks_too_few_objects():
    rows = [_observer_view(i, group=0) for i in range(8)]

    report = strict_neutral_object_bulk_report(rows, min_objects=4, min_observers_per_object=3)

    assert report["mode"] == "strict_neutral_object_bulk_v0"
    assert report["STRICT_NEUTRAL_OBJECT_BULK_RECEIPT"] is False
    assert "too_few_neutral_objects" in report["blockers"]
    assert "axis" in report["forbidden_primary_features"]
    assert report["dimension"]["reason"] == "too_few_neutral_objects"


def test_strict_neutral_object_bulk_writer_outputs_report_and_objects(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(32):
            handle.write(json.dumps(_observer_view(i, group=i // 4)) + "\n")

    report = write_strict_neutral_object_bulk_report(
        run,
        seed=11,
        min_objects=4,
        min_observers_per_object=3,
        max_model_points=16,
    )

    assert (run / "strict_neutral_object_bulk_report.json").exists()
    assert (run / "neutral_objects.jsonl").exists()
    assert report["object_count"] >= 4
    assert report["dimension"]["not_the_support_visible_chart_dimension"] is True
    assert report["physical_claim"] is False


def _observer_view(observer_id: int, *, group: int) -> dict:
    record = 10 + group
    packet = 20 + group
    checkpoint = group % 4
    sector = group % 3
    repair = group % 5
    return {
        "view_type": "patch_observer",
        "observer_id": observer_id,
        "record_signature_histogram": {str(record): 3.0, str(record + 1): 1.0},
        "object_packet_histogram": {str(packet): 2.0},
        "dominant_record_signature": record,
        "dominant_object_packet": packet,
        "checkpoint_class_transition": checkpoint,
        "sector_change_signature": sector,
        "transition_history_histograms": {
            "checkpoint_class": {str(checkpoint): 3.0},
            "s3_sector_class": {str(sector): 3.0},
            "repair_load_bucket": {str(repair): 2.0},
            "local_transition_token": {str(1000 + group): 1.0},
        },
        "transition_affinity_histograms": {"record_family": {str(group): 1.0}},
        "repair_response_spectrum": [float((group + j) % 3) for j in range(32)],
        "counterfactual_stability": 0.2 + 0.1 * group,
        "transition_history_persistence": 4 + group,
        "transition_history_mean_modal_mass": 0.7,
        "axis": [1.0, 0.0, 0.0],
        "support_nodes": [observer_id],
        "h3_point": [0.1, 0.2, 0.3],
        "cap_axis": [0.0, 1.0, 0.0],
        "radial_depth": 1.0,
        "modular_depth": 2.0,
    }
