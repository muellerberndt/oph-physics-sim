from __future__ import annotations

import csv
import json
from pathlib import Path

from oph_fpe.bulk.observer_consensus_bulk import (
    observer_consensus_bulk_readout_report,
    write_observer_consensus_bulk_readout_report,
)


def test_observer_consensus_bulk_readout_is_theorem_assisted_not_strict(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "claims.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "theorem_assisted_h3_bulk": True,
            "strict_neutral_bulk": False,
            "physical_cmb_output_usable_data_receipt": True,
            "physical_cmb_output_prediction_receipt": False,
        },
    )
    _write_json(
        run / "bulk_proof_certificate_report.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT": True,
            "STRICT_NEUTRAL_BULK_RECEIPT": False,
            "selected_object_chart_report": "observer_chart_object_h3_lineage_report.json",
            "selected_object_chart_incidence_mode": "record_sector_checkpoint_lineage",
        },
    )
    _write_json(
        run / "observer_modular_experience_report.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": False,
        },
    )
    _write_json(
        run / "strict_neutral_bulk_frontier_report.json",
        {
            "strict_neutral_bulk": False,
            "blockers": ["independent_svd_rank3_selector_not_stable_or_false"],
        },
    )
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
            "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
        },
    )
    (run / "observer_views.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "view_type": "patch_observer",
                        "observer_id": 7,
                        "axis": [0.0, 0.0, 1.0],
                        "support_patch_count": 3,
                        "support_entropy_capacity": 12.0,
                        "observer_relative_times": [0.0, 6.283185307179586],
                        "dominant_record_signature": 4,
                        "modular_depth_mean": 2.0,
                        "repair_load_mean": 0.1,
                        "mismatch_density_mean": 0.0,
                        "visible_signature_entropy": 0.5,
                        "visible_readout_hash": "abcdef0123456789",
                    }
                ),
                json.dumps(
                    {
                        "view_type": "cap_observer",
                        "cap_index": 0,
                        "axis": [1.0, 0.0, 0.0],
                        "theta0": 0.3,
                        "collar_width": 0.1,
                        "observer_relative_times": [0.0, 6.283185307179586],
                        "cap_area_planck": 8.0,
                        "cap_entropy_capacity": 5.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with (run / "h3_objects.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_id",
                "record_family_id",
                "family_mode",
                "observer_count",
                "support_size",
                "h3_compactness",
                "h3_compactness_normalized",
                "h3_spatial_point",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "object_id": "obj_001",
                "record_family_id": "family_1",
                "family_mode": "record_family_modular_response_mixture",
                "observer_count": "5",
                "support_size": "9",
                "h3_compactness": "0.2",
                "h3_compactness_normalized": "0.1",
                "h3_spatial_point": "[1.0, 2.0, 3.0]",
            }
        )
    (run / "neutral_objects.jsonl").write_text(
        json.dumps(
            {
                "object_id": "neutral_1",
                "observer_ids": [1, 2, 3],
                "visible_signature_key": "1:2:3",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = observer_consensus_bulk_readout_report([run])
    written = write_observer_consensus_bulk_readout_report([run], tmp_path / "out")

    assert report["OBSERVER_LIKE_SELF_READING_SYSTEM_RECEIPT"] is True
    assert report["observer_modular_time_receipt"] is True
    assert report["observer_facing_3p1d_h3_experience_receipt"] is True
    assert report["THEOREM_ASSISTED_CONSENSUS_3D_BULK_READOUT_RECEIPT"] is True
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["physical_cmb_output_comparison_receipt"] is True
    assert report["physical_cmb_prediction_receipt"] is False
    assert report["bulk_status"] == "theorem_assisted_observer_facing_consensus_3d_bulk"
    assert report["observer_readout"]["observer_view_count"] == 2
    assert report["h3_object_readout"]["spatial_dimension"] == 3
    assert report["h3_object_readout"]["object_count"] == 1
    assert report["neutral_object_readout"]["median_observers_per_neutral_object"] == 3.0
    assert written["strict_neutral_blockers"] == ["independent_svd_rank3_selector_not_stable_or_false"]
    assert (tmp_path / "out" / "observer_consensus_bulk_readout_report.json").exists()
    assert (tmp_path / "out" / "observer_consensus_bulk_readout_report.md").exists()
    assert (tmp_path / "out" / "observer_perspective_rows.csv").exists()
    assert (tmp_path / "out" / "consensus_h3_object_rows.csv").exists()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
