from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.bulk.prime_geometric_response import (
    attach_prime_geometric_response_to_rows,
    control_quotient_response_matrix,
    grouped_modular_response_matrix,
    response_component_spectrum,
    write_prime_geometric_response_attachment,
)


def test_grouped_modular_response_matrix_excludes_record_and_repair_for_prime_geometry():
    matrix = np.arange(24, dtype=float).reshape(3, 8)
    rows = [
        {"cap_index": 0, "time_index": 0, "observable": "checkpoint_class", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "checkpoint_class", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "record_family", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "repair_load_bucket", "feature_type": "class_distribution_delta"},
        {"cap_index": 1, "time_index": 0, "observable": "s3_sector_class", "feature_type": "change_probability_delta"},
        {"cap_index": 1, "time_index": 0, "observable": "stable_flag", "feature_type": "entropy_delta"},
        {"cap_index": 1, "time_index": 0, "observable": "checkpoint_class", "feature_type": "unsupported"},
        {"cap_index": 1, "time_index": 0, "observable": "record_family", "feature_type": "change_probability_delta"},
    ]

    grouped, report = grouped_modular_response_matrix(
        matrix,
        rows,
        observables=("checkpoint_class", "stable_flag", "s3_sector_class"),
    )

    assert grouped.shape == (3, 3)
    assert report["selected_feature_count"] == 4
    assert report["grouped_feature_count"] == 3
    assert np.allclose(grouped[:, 0], np.mean(matrix[:, [0, 1]], axis=1))


def test_attach_prime_geometric_response_ignores_geometry_fields():
    rows = [
        {"view_type": "patch_observer", "observer_id": 10, "axis": [1, 0, 0], "support_nodes": [1, 2]},
        {"view_type": "patch_observer", "observer_id": 20, "axis": [0, 1, 0], "support_nodes": [3, 4]},
    ]
    matrix = np.asarray([[1.0, 0.0, -1.0, 0.2], [0.1, 1.0, -0.5, 0.7]], dtype=float)
    feature_rows = [
        {"cap_index": 0, "time_index": 0, "observable": "checkpoint_class", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "s3_sector_class", "feature_type": "change_probability_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "record_family", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 0, "observable": "repair_load_bucket", "feature_type": "entropy_delta"},
    ]
    kernel = {"matrix": matrix, "observer_ids": [10, 20], "feature_rows": feature_rows}

    report = attach_prime_geometric_response_to_rows(rows, kernel, spectrum_width=8, component_bins=4)

    assert report["attached_observer_count"] == 2
    assert len(rows[0]["prime_geometric_modular_spectrum"]) == 8
    assert len(rows[0]["prime_geometric_control_quotient_spectrum"]) == 8
    assert len(rows[0]["support_visible_modular_spectrum"]) == 8
    assert len(rows[0]["repair_modular_spectrum"]) == 8 or len(rows[0]["repair_modular_spectrum"]) == 32
    assert "prime_geometric_modular_component_0" in rows[0]["modular_response_histograms"]
    assert "prime_geometric_control_quotient_component_0" in rows[0]["modular_response_histograms"]
    assert rows[0]["axis"] == [1, 0, 0]
    assert rows[0]["support_nodes"] == [1, 2]


def test_response_component_spectrum_has_fixed_width():
    matrix = np.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=float)

    spectrum, report = response_component_spectrum(matrix, width=5)

    assert spectrum.shape == (3, 5)
    assert report["component_count"] == 2
    assert report["rank_selection"]["mode"] == "svd_rank_selection_v0"
    assert report["rank_selection"]["component_count"] == 2
    assert "effective_rank" in report["rank_selection"]
    assert report["rank_selection"]["independent_rank3_selector_receipt"] is False
    assert np.all(np.isfinite(spectrum))


def test_control_quotient_response_matrix_removes_control_direction():
    raw = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
        ],
        dtype=float,
    )
    control = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=float,
    )
    feature_rows = [
        {"cap_index": 0, "time_index": 0, "observable": "checkpoint_class", "feature_type": "class_distribution_delta"},
        {"cap_index": 0, "time_index": 1, "observable": "s3_sector_class", "feature_type": "change_probability_delta"},
    ]

    quotient, report = control_quotient_response_matrix(
        raw,
        {"matrix": raw, "s2_boundary_control": control},
        feature_rows,
        observables=("checkpoint_class", "s3_sector_class"),
        max_control_rank=1,
    )

    assert quotient.shape == raw.shape
    assert report["control_count"] == 1
    assert report["control_basis_rank"] == 1
    assert report["residual_frobenius_norm"] < report["raw_centered_frobenius_norm"]


def test_write_prime_geometric_response_attachment_updates_existing_run(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"view_type": "patch_observer", "observer_id": 1},
        {"view_type": "patch_observer", "observer_id": 2},
    ]
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    cache = {
        "observer_ids": [1, 2],
        "feature_rows": [
            {"cap_index": 0, "time_index": 0, "observable": "checkpoint_class", "feature_type": "class_distribution_delta"},
            {"cap_index": 0, "time_index": 0, "observable": "s3_sector_class", "feature_type": "change_probability_delta"},
            {"cap_index": 0, "time_index": 0, "observable": "repair_load_bucket", "feature_type": "entropy_delta"},
        ],
        "cap_count": 1,
        "time_count": 1,
        "field_names": ["checkpoint_class", "s3_sector_class", "repair_load_bucket"],
        "wrong_scale_controls": [],
    }
    (run / "modular_response_kernel_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    np.savez(run / "modular_response_kernel_payload.npz", matrix=np.asarray([[1.0, 0.2, 0.4], [0.3, 1.0, 0.8]]))

    report = write_prime_geometric_response_attachment(run, spectrum_width=8, component_bins=4)
    updated = [json.loads(line) for line in (run / "observer_views.jsonl").read_text().splitlines()]

    assert report["attached_observer_count"] == 2
    assert (run / "observer_views.before_prime_geometric_response.jsonl").exists()
    assert (run / "prime_geometric_response_attachment_report.json").exists()
    assert "prime_geometric_modular_spectrum" in updated[0]
