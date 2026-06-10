import json

from oph_fpe.scale.shape_substrate import (
    build_arc_tables,
    run_shape_dodeca_smoke,
    run_shape_dodeca_trace,
)
from oph_fpe.microphysics.dodeca_cell import dodecahedral_cell


def test_shape_arc_tables_degree_three():
    cell = dodecahedral_cell()
    tables = build_arc_tables(cell.graph)

    assert tables.incoming_arcs.shape == (20, 3)
    assert tables.outgoing_arcs.shape == (20, 3)
    assert len(tables.tail) == 60


def test_shape_trace_emits_faces_and_particles():
    trace = run_shape_dodeca_trace(
        {
            "seed": 7,
            "cycles": 12,
            "repair_rate": 0.04,
            "particle_energy_threshold": 0.0,
        }
    )

    assert len(trace["phi_trace"]) == 12
    assert len(trace["face_rows"]) == 12
    assert trace["state"].amp.shape[0] == 60


def test_shape_smoke_writes_declared_witness_reports(tmp_path):
    report = run_shape_dodeca_smoke(
        {
            "name": "shape_test",
            "seed": 11,
            "cycles": 16,
            "repair_rate": 0.04,
            "particle_energy_threshold": 0.0,
            "ell_max": 4,
            "certificate_detuning_multipliers": [0.5, 1.0, 1.5, 2.0],
        },
        tmp_path,
    )
    run_path = tmp_path / report["run_id"]
    summary = json.loads((run_path / "shape_substrate_summary.json").read_text(encoding="utf-8"))

    assert summary["shape_vertex_scattering_receipt"] is True
    assert summary["shape_dodeca_cell_receipt"] is True
    assert summary["shape_screen_projection_receipt"] is True
    assert summary["shape_selector_elimination_target_input_receipt"] is True
    assert summary["shape_q_IR_candidate"] == 0.25
    assert summary["shape_ell_IR_candidate"] == 32
    assert summary["neutral_oph_bulk_claim"] is False
    assert summary["physical_cmb_prediction"] is False
    assert (run_path / "shape_freezeout_fields.npz").exists()
    assert (run_path / "shape_settling_trace.csv").exists()
    assert (run_path / "shape_loop_particles.csv").exists()
    assert (run_path / "shape_screen_cl.csv").exists()
    assert "phi_loop_strain" in (run_path / "shape_settling_trace.csv").read_text(encoding="utf-8")
    assert "particle_class" in (run_path / "shape_loop_particles.csv").read_text(encoding="utf-8")
    assert "D_ell" in (run_path / "shape_screen_cl.csv").read_text(encoding="utf-8")
