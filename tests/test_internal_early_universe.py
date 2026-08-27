from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.core.icosahedral import geodesic_icosahedral_patch_arrays
from oph_fpe.cosmology.internal_early_universe import (
    PRIMARY_OBSERVABLE,
    internal_early_universe_report,
    write_internal_early_universe_report,
)


def _write_fixture(root: Path, *, censored: bool = False) -> None:
    points, left, right = geodesic_icosahedral_patch_arrays(0, patch_basis="cells")
    node_count = points.shape[0]
    # A nondegenerate, spatially structured completed-record surface.  This is
    # a unit fixture, not a physics sample.
    first_commit = np.floor(4.0 + 3.0 * (points[:, 2] + 1.0)).astype(np.int32)
    if censored:
        first_commit[0] = -1
    first_repair = np.maximum(first_commit - 4, 0).astype(np.int32)
    last_repair = np.maximum(first_commit - 3, first_repair).astype(np.int32)
    first_quiescence = np.maximum(first_commit - 2, 0).astype(np.int32)
    np.savez_compressed(
        root / "screen_event_times.npz",
        points=points.astype(np.float32),
        edge_left=left.astype(np.int32),
        edge_right=right.astype(np.int32),
        cell_entropy=np.ones(node_count, dtype=np.float32),
        initial_incident_mismatch_count=np.ones(node_count, dtype=np.int16),
        first_repair_cycle=first_repair,
        last_repair_cycle=last_repair,
        last_mismatch_cycle=np.maximum(first_quiescence - 1, -1).astype(np.int32),
        first_quiescence_cycle=first_quiescence,
        first_commit_cycle=first_commit,
        repair_span_cycles=(last_repair - first_repair).astype(np.int32),
        commit_latency_after_quiescence=(first_commit - first_quiescence).astype(np.int32),
        cumulative_repair_load=(1.0 + points[:, 2]).astype(np.float32),
        final_mismatch_density=np.zeros(node_count, dtype=np.float32),
        committed_final=(first_commit >= 0).astype(np.uint8),
        cycles=np.asarray([12], dtype=np.int32),
        record_commit_cycles=np.asarray([2], dtype=np.int32),
    )
    with (root / "mismatch_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cycle",
                "phi_before",
                "phi",
                "chosen_edges",
                "repair_budget",
                "observer_readback_drive_edges",
                "sector_edges_changed",
                "committed_fraction",
                "record_packet_entropy",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "cycle": 0,
                    "phi_before": 10,
                    "phi": 8,
                    "chosen_edges": 2,
                    "repair_budget": 4,
                    "observer_readback_drive_edges": 0,
                    "sector_edges_changed": 0,
                    "committed_fraction": 0.0,
                    "record_packet_entropy": 0.0,
                },
                {
                    "cycle": 1,
                    "phi_before": 8,
                    "phi": 4,
                    "chosen_edges": 4,
                    "repair_budget": 4,
                    "observer_readback_drive_edges": 0,
                    "sector_edges_changed": 0,
                    "committed_fraction": 0.4,
                    "record_packet_entropy": 0.8,
                },
                {
                    "cycle": 2,
                    "phi_before": 4,
                    "phi": 0,
                    "chosen_edges": 4,
                    "repair_budget": 4,
                    "observer_readback_drive_edges": 0,
                    "sector_edges_changed": 0,
                    "committed_fraction": 1.0,
                    "record_packet_entropy": 1.2,
                },
            ]
        )


def test_internal_diagnostics_are_target_blind_and_fail_closed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = internal_early_universe_report(
        tmp_path,
        ell_max=4,
        pair_samples=200,
        shuffle_draws=4,
        thresholds=(0.5, 0.75),
        seed=9,
        harmonic_batch_size=64,
    )

    assert report["primary_observable"] == PRIMARY_OBSERVABLE
    assert report["diagnostic_eligible"] is True
    assert report["input_custody"]["target_data_read"] is False
    assert report["input_custody"]["measurement_files_read"] == []
    assert report["physical_early_universe_claim"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["angular_spectrum"]["fields"][PRIMARY_OBSERVABLE]["spectrum"]
    assert report["late_record_morphology"]["threshold_rows"]
    assert report["global_settling_arrow"]["final_phi"] == 0
    assert report["global_settling_arrow"]["initial_phi"] == 10
    assert report["independent_seam_null"]["available"] is True
    assert report["factorized_graph_scheduler_control"]["available"] is True


def test_censored_record_surface_cannot_be_diagnostic_eligible(tmp_path: Path) -> None:
    _write_fixture(tmp_path, censored=True)
    report = internal_early_universe_report(
        tmp_path,
        ell_max=0,
        pair_samples=100,
        shuffle_draws=3,
    )

    assert report["diagnostic_eligible"] is False
    assert report["eligibility_checks"]["all_patches_committed"] is False
    assert report["censored_patch_count"] == 1


def test_writer_emits_json_markdown_and_machine_rows(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_fixture(run)

    report = write_internal_early_universe_report(
        run,
        out,
        ell_max=0,
        pair_samples=100,
        shuffle_draws=3,
    )

    assert report["schema"] == "oph_internal_early_universe_diagnostics_v1"
    assert (out / "internal_early_universe_diagnostics.json").is_file()
    assert (out / "internal_early_universe_diagnostics.md").is_file()
    assert (out / "internal_angular_correlation.csv").is_file()
    assert (out / "internal_late_record_morphology.csv").is_file()
    assert (out / "internal_graph_shell_correlation.csv").is_file()
    assert (out / "internal_independent_seam_null.csv").is_file()


def test_missing_required_event_array_is_rejected(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "screen_event_times.npz", points=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="missing arrays"):
        internal_early_universe_report(tmp_path, ell_max=0)
