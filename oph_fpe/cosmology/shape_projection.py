from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    SHAPE_SCREEN_PROJECTION_RECEIPT,
)
from oph_fpe.cosmology.angular_power import angular_power_report
from oph_fpe.microphysics.dodeca_cell import DodecaCell, dodecahedron_face_normals, dodecahedral_cell


def project_dodeca_to_screen(
    face_rows: list[dict[str, Any]],
    *,
    cell: DodecaCell | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cell = cell or dodecahedral_cell()
    points = dodecahedron_face_normals(cell)
    row_by_face = {int(row.get("face_id", index)): row for index, row in enumerate(face_rows)}

    def field(name: str, default: float = 0.0) -> np.ndarray:
        return np.asarray(
            [float(row_by_face.get(index, {}).get(name, default)) for index in range(points.shape[0])],
            dtype=float,
        )

    fields = {
        "loop_mode_energy": field("mode_energy"),
        "loop_particle_density": field("particle_density"),
        "phi_wave_density": field("phi_wave"),
        "minus_dphi_dt_density": field("minus_dphi_dt"),
        "phase_coherence": field("phase_coherence"),
    }
    return points, fields


def shape_cl_report(
    points: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    ell_max: int = 8,
    seed: int = 1,
    n_jobs: int = 1,
) -> dict[str, Any]:
    report = angular_power_report(
        points,
        fields,
        ell_max=int(ell_max),
        seed=int(seed),
        controls=["shuffled_field", "random_gaussian"],
        estimator="spherical_harmonic",
        harmonic_batch_size=128,
        n_jobs=int(n_jobs),
    )
    report.update(
        {
            "receipt": SHAPE_SCREEN_PROJECTION_RECEIPT,
            "receipt_name": SHAPE_SCREEN_PROJECTION_RECEIPT,
            "passed": bool(points.shape[0] >= 12 and len(fields) >= 1),
            "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
            "neutral_oph_bulk_claim": False,
            "declared_3d_substrate": True,
            "physical_cmb_prediction": False,
            "claim_boundary": (
                "Declared Shape substrate screen projection and angular-power diagnostic. "
                "This is not neutral OPH bulk emergence and not a physical CMB prediction."
            ),
        }
    )
    return report


def write_shape_freezeout_npz(path: Path, points: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    payload = {"points": np.asarray(points, dtype=float)}
    payload.update({name: np.asarray(values, dtype=float) for name, values in fields.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(path), **payload)


def write_shape_projection_reports(
    out_dir: Path,
    face_rows: list[dict[str, Any]],
    *,
    ell_max: int = 8,
    seed: int = 1,
    n_jobs: int = 1,
    cell: DodecaCell | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    points, fields = project_dodeca_to_screen(face_rows, cell=cell)
    write_shape_freezeout_npz(out_dir / "shape_freezeout_fields.npz", points, fields)
    report = shape_cl_report(points, fields, ell_max=ell_max, seed=seed, n_jobs=n_jobs)
    (out_dir / "shape_screen_projection_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "shape_cl_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "cl_comparison_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report
