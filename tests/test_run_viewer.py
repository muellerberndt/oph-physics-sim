import json
from pathlib import Path

import numpy as np

from oph_fpe.viz import write_run_viewer


def test_run_viewer_writes_html_with_gate_boundary(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "viewer_smoke", "patch_count": 16}),
        encoding="utf-8",
    )
    (run_dir / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "support_visible_lorentz_3p1_kinematics_receipt": True,
                "conformal_h3_spatial_chart_receipt": True,
                "bulk_3d_established": False,
                "defect_cluster_h3_support_receipt": True,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "freezeout_fields.npz",
        points=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        record_signature=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        cell_area_planck=np.ones(3, dtype=np.float32),
        cell_entropy=np.ones(3, dtype=np.float32),
    )
    (run_dir / "observer_views.jsonl").write_text(
        json.dumps({"axis": [1.0, 0.0, 0.0], "visible_signature_entropy": 0.5}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "array_holonomy_report.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "cluster_id": "d0",
                        "centroid": [0.0, 1.0, 0.0],
                        "class": "threecycle",
                        "support_node_count": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "defect_timeline_report.json").write_text(
        json.dumps(
            {
                "persistent_worldline_count": 1,
                "snapshots": [
                    {
                        "cycle": 0,
                        "cluster_count": 1,
                        "clusters": [
                            {
                                "cluster_id": "d0",
                                "centroid": [0.0, 1.0, 0.0],
                                "class": "threecycle",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "defect_cluster_h3_report.json").write_text(
        json.dumps({"h3_fit": {"sample_fitted_h3_points": [[1.1, 0.2, 0.3, 0.4]]}}),
        encoding="utf-8",
    )
    (run_dir / "defect_h3_worldlines_report.json").write_text(
        json.dumps(
            {
                "worldlines": [
                    {
                        "worldline_id": "w0",
                        "observation_count": 2,
                        "events": [
                            {"cycle": 0, "h3_spatial_point": [0.1, 0.2, 0.3]},
                            {"cycle": 1, "h3_spatial_point": [0.2, 0.3, 0.4]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = write_run_viewer(run_dir)

    html_path = Path(summary["viewer_path"])
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    assert "OPH-FPE Receipt Viewer" in text
    assert "Diagnostic viewer only" in text
    assert summary["bulk_3d_established"] is False
    assert summary["defect_timeline_snapshots"] == 1
    assert summary["persistent_defect_worldlines"] == 1
    assert summary["h3_worldline_count"] == 1
