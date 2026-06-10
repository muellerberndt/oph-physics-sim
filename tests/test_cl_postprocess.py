from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.cl_postprocess import write_cl_from_freezeout_npz
from oph_fpe.core.graph import fibonacci_sphere_points


def test_write_cl_from_freezeout_npz_recomputes_spectrum(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    points = fibonacci_sphere_points(128)
    np.savez_compressed(
        run / "freezeout_fields.npz",
        points=points.astype(np.float32),
        cell_area_planck=np.ones(points.shape[0], dtype=np.float32),
        cell_entropy=np.ones(points.shape[0], dtype=np.float32),
        record_signature=np.sin(points[:, 0] * 3.0).astype(np.float32),
        stable_count=np.cos(points[:, 1] * 4.0).astype(np.float32),
    )
    (run / "cl_comparison_report.json").write_text(
        json.dumps({"freezeout_cycle": 5, "committed_fraction": 0.75, "gate_report": {"allowed": True}}),
        encoding="utf-8",
    )

    out = tmp_path / "post"
    summary = write_cl_from_freezeout_npz(
        run,
        out,
        ell_max=6,
        fields=["record_signature"],
        harmonic_batch_size=64,
        n_jobs=1,
    )

    assert summary["ell_max"] == 6
    assert summary["fields"] == ["record_signature"]
    report = json.loads((out / "cl_comparison_report.json").read_text(encoding="utf-8"))
    assert report["postprocess_only"] is True
    assert report["freezeout_cycle"] == 5
    assert report["committed_fraction"] == 0.75
    assert report["gate_report"]["allowed"] is True
    assert report["ell_max"] == 6
    assert list(report["fields"]) == ["record_signature"]
    assert len(report["fields"]["record_signature"]["spectrum"]) == 7
    assert (out / "cl_proxy.csv").read_text(encoding="utf-8").startswith("field,ell,C_ell,D_ell")
    assert (out / "cl_controls.csv").read_text(encoding="utf-8").startswith("field,control,ell,C_ell,D_ell")
