from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_SOURCE_CANDIDATE
from oph_fpe.cosmology.silence_to_observation import (
    silence_to_observation_report,
    write_silence_to_observation_report,
)


def _write_minimal_run(run: Path, *, initial_silent: bool = True, h3_objects: int = 3) -> None:
    run.mkdir()
    (run / "screen_microphysics.json").write_text(
        json.dumps(
            {
                "patch_count": 64,
                "pixel_scale": {"P": P_SOURCE_CANDIDATE},
                "screen_units": {"regulator_entropy_weight_sum": 64 * P_SOURCE_CANDIDATE / 4.0},
            }
        ),
        encoding="utf-8",
    )
    first_records = 0 if initial_silent else 1
    first_fraction = 0.0 if initial_silent else 0.1
    first_entropy = 0.0 if initial_silent else 0.2
    (run / "mismatch_trace.csv").write_text(
        "cycle,phase,phi,committed_records,committed_fraction,record_entropy,modular_depth_mean,mismatch_edges\n"
        f"0,exploration,10,{first_records},{first_fraction},{first_entropy},0.0,10\n"
        "63,exploration,0,64,1.0,2.0,0.7,0\n",
        encoding="utf-8",
    )
    (run / "theorem_core_receipts.json").write_text(
        json.dumps({"FINITE_CONSENSUS_THEOREM_RECEIPT": True}),
        encoding="utf-8",
    )
    (run / "observer_modular_experience_report.json").write_text(
        json.dumps(
            {
                "observer_modular_time_receipt": True,
                "observer_facing_3p1d_h3_experience_receipt": True,
                "observer_count": 8,
            }
        ),
        encoding="utf-8",
    )
    (run / "observer_chart_object_h3_report.json").write_text(
        json.dumps(
            {
                "object_count": h3_objects,
                "observer_chart_bulk_population_receipt": h3_objects > 0,
            }
        ),
        encoding="utf-8",
    )
    readout = run / "observer_consensus_bulk"
    readout.mkdir()
    (readout / "observer_consensus_bulk_readout_report.json").write_text(
        json.dumps({"observer_h3_object_population_receipt": h3_objects > 0}),
        encoding="utf-8",
    )
    (run / "defect_h3_worldlines_report.json").write_text(
        json.dumps({"persistent_h3_worldline_count": 1}),
        encoding="utf-8",
    )


def test_silence_to_observation_report_passes_scale_compressed_witness(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_minimal_run(run)

    report = silence_to_observation_report(run)

    assert report["mode"] == "oph_pn_silence_to_observation_witness_v1"
    assert report["scale_compressed_pn_silence_to_observation_receipt"] is True
    assert report["literal_global_N_capacity_simulated_receipt"] is False
    assert report["dynamic_p_detuning_control_receipt"] is False
    assert report["p_role"] == "post_hoc_analytic_branch_association"
    assert report["relaxation_dynamics_consumed_p"] is False
    assert "did not consume P" in report["claim_boundary"]
    assert report["silence_initial_state"]["initial_record_silence_receipt"] is True
    assert report["observation_emergence"]["h3_object_count"] == 3
    assert math.isclose(
        report["finite_regulator_depth"]["regulator_entropy_capacity_N_eff"],
        64 * P_SOURCE_CANDIDATE / 4.0,
    )
    assert report["detuning_controls"]["no_detuning_phi_equilibrium"]["blocks_pn_bridge"] is True
    assert all(row["blocks_selected_bridge"] for row in report["detuning_controls"]["wrong_detuning_multipliers"])


def test_silence_to_observation_report_blocks_without_silence_or_h3(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_minimal_run(run, initial_silent=False, h3_objects=0)

    report = silence_to_observation_report(run)

    assert report["scale_compressed_pn_silence_to_observation_receipt"] is False
    assert report["readiness_gates"]["initial_record_silence"] is False
    assert report["readiness_gates"]["h3_object_emergence"] is False


def test_silence_to_observation_report_writes_json_and_markdown(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    _write_minimal_run(run)

    report = write_silence_to_observation_report(run, out)
    loaded = json.loads((out / "silence_to_observation_report.json").read_text(encoding="utf-8"))

    assert (out / "silence_to_observation_report.md").exists()
    assert loaded["scale_compressed_pn_silence_to_observation_receipt"] == report[
        "scale_compressed_pn_silence_to_observation_receipt"
    ]


def test_silence_to_observation_is_available_from_lazy_cosmology_exports(tmp_path: Path) -> None:
    from oph_fpe.cosmology import silence_to_observation_report as exported_report

    run = tmp_path / "run"
    _write_minimal_run(run)

    assert exported_report(run)["scale_compressed_pn_silence_to_observation_receipt"] is True
