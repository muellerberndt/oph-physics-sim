from __future__ import annotations

import numpy as np

from oph_fpe.flavor import build_poft_transport_emission_report


def _write_state(path, gauge: np.ndarray) -> None:
    edge_count = gauge.size
    np.savez_compressed(
        path,
        left=np.arange(edge_count, dtype=np.int64),
        right=np.arange(1, edge_count + 1, dtype=np.int64),
        gauge=gauge,
    )


def test_uniform_s3_state_is_haar_null_and_not_poft(tmp_path) -> None:
    state = tmp_path / "state.npz"
    _write_state(state, np.tile(np.arange(6, dtype=np.int64), 20))
    report = build_poft_transport_emission_report([("uniform", state)])
    row = report["states"][0]
    assert row["haar_rank_one_compatible"] is True
    assert row["poft_T0_necessary_spectral_match"] is False
    assert report["receipts"]["poft_T0_T1_physical_emission_receipt"] is False
    assert report["verdict"] == "CURRENT_S3_EDGE_CARRIER_HAAR_EQUILIBRATED_NOT_POFT"


def test_identity_state_does_not_pass_full_emission_without_complex_and_refinement_receipts(tmp_path) -> None:
    state = tmp_path / "identity.npz"
    _write_state(state, np.zeros(60, dtype=np.int64))
    report = build_poft_transport_emission_report([("identity", state)])
    assert report["states"][0]["complex_oriented_amplitude_exported"] is False
    assert report["receipts"]["coarse_to_fine_edge_intertwiner_exported"] is False
    assert report["receipts"]["poft_T0_T1_physical_emission_receipt"] is False
