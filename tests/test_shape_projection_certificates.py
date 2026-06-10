import numpy as np

from oph_fpe.cosmology.shape_certificates import (
    dodeca_visible_ir_target_certificate,
    ell_ir_from_visible_covariance_rank,
    four_sector_q_ir_certificate,
    gamma_rec_from_transition_matrix,
    q_ir_from_zero_mode,
    repair_transition_matrix,
)
from oph_fpe.cosmology.shape_projection import project_dodeca_to_screen, shape_cl_report


def test_shape_projection_and_cl_report_are_diagnostic_only():
    face_rows = [
        {
            "face_id": face_id,
            "mode_energy": float(face_id + 1),
            "particle_density": float(face_id % 2),
            "phi_wave": 0.1 * face_id,
            "minus_dphi_dt": 0.01,
            "phase_coherence": 0.9,
        }
        for face_id in range(12)
    ]
    points, fields = project_dodeca_to_screen(face_rows)
    report = shape_cl_report(points, fields, ell_max=4)

    assert points.shape == (12, 3)
    assert "loop_mode_energy" in fields
    assert report["passed"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["neutral_oph_bulk_claim"] is False


def test_shape_certificate_helpers_emit_finite_values():
    assert q_ir_from_zero_mode(np.ones(8)) == 1.0
    rank = ell_ir_from_visible_covariance_rank(np.eye(12))
    assert rank["visible_covariance_rank"] >= 1

    matrix, _ = repair_transition_matrix([0, 1, 1, 2], [1, 1, 2, 2])
    gamma = gamma_rec_from_transition_matrix(matrix)

    assert gamma["finite"] is True


def test_shape_selector_elimination_targets_are_separate_from_runtime_rank():
    ell = dodeca_visible_ir_target_certificate()
    q = four_sector_q_ir_certificate()

    assert ell["passed"] is True
    assert ell["visible_scalar_record_channels"] == 32
    assert ell["ell_IR_candidate"] == 32
    assert ell["finite_lattice_derived"] is False
    assert q["passed"] is True
    assert q["q_IR_candidate"] == 0.25
    assert q["physical_cmb_prediction"] is False
