from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.oph_screen_power import C_ell_oph, D_ell_from_C_ell, OPHScreenPowerParams
from oph_fpe.cosmology.fossil_spectrum import write_fossil_spectrum_report


def test_fossil_spectrum_report_keeps_target_scan_diagnostic(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    cycles = np.array([0, 1, 2], dtype=np.int32)
    ell = np.arange(0, 40, dtype=np.float32)
    fit_ell = np.maximum(ell, 1.0)
    good = D_ell_from_C_ell(fit_ell, C_ell_oph(fit_ell, OPHScreenPowerParams(eta_R=0.035)))
    bad = D_ell_from_C_ell(fit_ell, C_ell_oph(fit_ell, OPHScreenPowerParams(eta_R=0.2)))
    field = np.vstack([bad, good, bad]).astype(np.float32)
    control = np.vstack([bad, bad, bad]).astype(np.float32)
    np.savez(
        run / "harmonic_time_trace.npz",
        cycles=cycles,
        ell=ell,
        record_signature=field,
        control__record_signature__shuffled_field=control,
    )
    with (run / "mismatch_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cycle", "phi_before", "phi", "committed_records", "committed_fraction"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"cycle": 0, "phi_before": 10, "phi": 10, "committed_records": 0, "committed_fraction": 0.0},
                {"cycle": 1, "phi_before": 10, "phi": 4, "committed_records": 8, "committed_fraction": 0.5},
                {"cycle": 2, "phi_before": 10, "phi": 0, "committed_records": 16, "committed_fraction": 1.0},
            ]
        )
    (run / "manifest.json").write_text(json.dumps({"patch_count": 4096}), encoding="utf-8")
    (run / "freezeout_map_summary.json").write_text(json.dumps({"freezeout_cycle": 2}), encoding="utf-8")

    report = write_fossil_spectrum_report(run, tmp_path / "out", fields=["record_signature"], ell_min=8, ell_max=32)

    best = report["best_target_closeness_diagnostic"]
    assert best["cycle"] == 1
    assert report["near_scale_invariant_transient"] is True
    assert report["best_beats_same_field_controls"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["cycle_markers"]["phi_zero_cycle"] == 2
    assert (tmp_path / "out" / "fossil_spectrum_rows.csv").exists()
