from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.cosmology.freezeout import write_freezeout_products


def test_freezeout_products_include_configured_derived_fields(tmp_path: Path):
    points = fibonacci_sphere_points(128)
    fields = {
        "record_signature": np.sin(points[:, 0] * 4.0),
        "cumulative_repair_load": np.cos(points[:, 1] * 3.0),
    }
    config = {
        "output_profile": "evidence",
        "freezeout": {
            "fields": [
                "record_signature",
                "record_signature_smooth_k8",
                "record_repair_mix",
            ],
            "derived_fields": [
                {
                    "name": "record_signature_smooth_k8",
                    "kind": "knn_smooth",
                    "source": "record_signature",
                    "k": 8,
                    "steps": 2,
                    "alpha": 1.0,
                },
                {
                    "name": "record_repair_mix",
                    "kind": "linear_combo",
                    "terms": [
                        {"field": "record_signature", "weight": 1.0},
                        {"field": "cumulative_repair_load", "weight": 0.5},
                    ],
                },
            ],
        },
        "angular_power": {"ell_max": 8, "controls": ["shuffled_field"], "harmonic_batch_size": 128},
    }

    report = write_freezeout_products(
        tmp_path,
        points=points,
        fields=fields,
        cell_area_planck=np.ones(points.shape[0]),
        cell_entropy=np.ones(points.shape[0]),
        freezeout_cycle=4,
        committed_fraction=1.0,
        config=config,
        seed=9,
        gate_report={"allowed": True},
    )

    cl_report = json.loads((tmp_path / "cl_comparison_report.json").read_text(encoding="utf-8"))
    assert "record_signature_smooth_k8" in cl_report["fields"]
    assert "record_repair_mix" in cl_report["fields"]
    assert report["fields"]["record_signature_smooth_k8"]["peak_ell"] >= 2
