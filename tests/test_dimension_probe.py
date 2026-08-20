"""Tests for the exploratory dimension probe (lane D1, non-evidential).

Calibration on graphs of known dimension, dense/stochastic estimator
agreement, mutation guards, and receipt canonicalization.  Pins and
tolerances are DESIGN.md sections 4, 5, and 7; a calibration failure is
repaired in the estimator, never in the tolerance.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

from oph_fpe.dimension import EVIDENTIAL_STATUS
from oph_fpe.dimension import calibration as cal
from oph_fpe.dimension import estimators as est
from oph_fpe.dimension import geometry, operators, probe, receipts


@pytest.fixture(scope="module")
def calibration_rows():
    return {row["name"]: row for row in probe.run_calibration()}


@pytest.fixture(scope="module")
def cross_check_rows():
    return {row["name"]: row for row in probe.run_cross_checks()}


@pytest.fixture(scope="module")
def small_window():
    _, level_graphs, incidences = geometry.build_tower_window(2)
    return level_graphs, incidences


@pytest.mark.parametrize("name", ["cycle_4096", "torus2d_64", "torus3d_24"])
def test_calibration_integer_cases(calibration_rows, name):
    row = calibration_rows[name]
    assert row["d_s_within_band"], row["name"]
    assert row["d_weyl_within_band"], row["name"]
    assert row["symmetry_max_abs_asymmetry"] == 0.0
    assert not row["window_degenerate"]
    assert row["kernel_dim"] == 1


def test_calibration_tree_anomalous_control(calibration_rows):
    row = calibration_rows["tree4_depth8"]
    assert row["anomalous_control_assertion"] == "exact_exponential_shell_growth"
    assert row["anomalous_control_pass"]
    assert row["exact_tree_growth"]["independent_of_weyl_eigencount"]
    assert row["exact_tree_growth"]["minimum_interior_shell_growth_ratio"] == 3.0
    assert isinstance(row["d_s_median"], float)
    assert "d_s_median_outside_all_integer_bands" in row
    assert set(row["weyl_k_scan_audit"]) == {str(k) for k in est.WEYL_K_SCAN}


def test_cross_checks_within_tolerance(cross_check_rows):
    assert set(cross_check_rows) == {case["name"] for case in cal.CROSS_CHECK_CASES}
    for row in cross_check_rows.values():
        assert row["within_tolerance"], row["name"]
        assert row["comparison_point_count"] > 0


def test_mutation_kappa_zero_reproduces_decoupled_control(small_window):
    level_graphs, incidences = small_window
    union, meta = operators.union_laplacian(level_graphs, incidences, kappa=0.0)
    blocks = [operators.single_level_laplacian(graph)[0] for graph in level_graphs]
    block_diagonal = sp.block_diag(blocks, format="csr")
    difference = (union - block_diagonal).tocoo()
    assert difference.nnz == 0
    assert meta["inter_edge_count"] == 0
    count, _ = operators.component_structure(union)
    assert count == len(level_graphs)


def test_mutation_broken_symmetry_caught(small_window):
    level_graphs, incidences = small_window
    union, _ = operators.union_laplacian(level_graphs, incidences, kappa=1.0)
    tampered = union.tolil()
    tampered[0, 1] = tampered[0, 1] + 0.5
    assert operators.symmetry_max_abs_asymmetry(tampered.tocsr()) > 0.0
    with pytest.raises(ValueError, match="asymmetry"):
        operators.require_symmetric(tampered.tocsr())


def test_mutation_3d_lattice_mislabeled_as_2d_fails(calibration_rows):
    row = calibration_rows["torus3d_24"]
    assert not cal.integer_band_check(row["d_s_median"], 2)
    assert not cal.integer_band_check(row["d_weyl"], 2)


def test_coupled_union_connected_with_kernel_dim_one(small_window):
    level_graphs, incidences = small_window
    for kappa in (0.25, 1.0):
        union, _ = operators.union_laplacian(level_graphs, incidences, kappa=kappa)
        count, _ = operators.component_structure(union)
        assert count == 1
        spectrum = est.dense_spectrum(union)
        assert int(np.count_nonzero(spectrum <= est.ZERO_MODE_CUT)) == 1


def test_static_control_unit_weights(small_window):
    level_graphs, incidences = small_window
    union, meta = operators.union_laplacian(
        level_graphs, incidences, kappa=0.0, static_control=True
    )
    matrix = union.tocoo()
    off_diagonal = matrix.data[matrix.row != matrix.col]
    assert off_diagonal.size > 0
    assert np.all(off_diagonal == -1.0)
    assert meta["inter_edge_count"] == sum(
        incidence.child_to_parent.size for incidence in incidences
    )
    assert meta["inter_level_weight_model"] == "all_ones_coupling_scale_control"


def test_matched_scale_control_changes_only_fiber_weight_shape(small_window):
    level_graphs, incidences = small_window
    kappa = 0.5
    matched, meta = operators.union_laplacian(
        level_graphs,
        incidences,
        kappa=kappa,
        uniform_fiber_weights=True,
    )
    assert meta["inter_level_weight_model"] == "matched_scale_uniform_one_quarter"
    offsets, _ = operators.union_offsets(level_graphs)
    _, _, weights = operators.inter_level_edges(
        incidences,
        offsets,
        kappa,
        uniform_fiber_weights=True,
    )
    assert np.all(weights == kappa / 4.0)
    assert operators.require_symmetric(matched) == 0.0


def test_tower_configuration_labels_separate_shape_and_scale_controls():
    rows = probe.tower_configurations()
    assert any(row["kind"] == "matched_scale_weight_shape_control" for row in rows)
    assert any(row["kind"] == "coupling_scale_control" for row in rows)
    assert not any(row["kind"] == "construction_control" for row in rows)


def test_committed_tower_receipts(small_window):
    level_graphs, incidences = small_window
    for graph in level_graphs:
        assert graph.cell_count == 20 * 4**graph.level
        assert 2 * graph.edge_count == 3 * graph.cell_count
    for incidence in incidences:
        assert incidence.max_parent_weight_sum_residual <= 1.0e-12
        assert 0.0 < incidence.max_uniform_deviation < 0.25


def test_window_rule_degenerate_paths():
    sigmas = est.SIGMA_GRID
    curve = np.full(sigmas.size, 2.0)
    degenerate = est.window_statistics(sigmas, curve, None)
    assert degenerate["window_degenerate"]
    assert degenerate["d_s_median"] is None
    tight = est.window_statistics(sigmas, curve, 1.0 / 4.5)
    assert tight["window_degenerate"]
    assert tight["window_point_count"] < est.WINDOW_MIN_POINTS


def test_reported_window_excludes_kernel_saturated_tail():
    sigmas = est.SIGMA_GRID
    curve = np.full(sigmas.size, 2.0)
    p_return = np.ones(sigmas.size)
    unguarded = est.window_statistics(sigmas, curve, 1.0e-4)
    p_return[sigmas >= 100.0] = 0.03
    guarded = est.window_statistics(
        sigmas,
        curve,
        1.0e-4,
        p_return=p_return,
        kernel_dim=1,
        node_count=100,
    )
    assert guarded["window_saturation_guard_applied"]
    assert guarded["window_saturation_floor"] == pytest.approx(0.04)
    assert guarded["window_point_count"] < unguarded["window_point_count"]


def test_stochastic_measurement_reports_seed_uncertainty(small_window):
    level_graphs, incidences = small_window
    union, _ = operators.union_laplacian(level_graphs, incidences, kappa=1.0)
    _, labels = operators.component_structure(union)
    basis = operators.kernel_basis_from_components(labels)
    measured = est.measure_dimensions(
        union,
        basis,
        expected_kernel_dim=1,
        probes=8,
        steps=32,
        force_stochastic=True,
    )
    assert len(measured["hutchinson_seed_ensemble"]) == 8
    assert measured["probes_per_seed"] == 1
    assert len(measured["d_s_median_seed_estimates"]) >= 2
    assert measured["d_s_median_seed_standard_error"] is not None
    assert measured["d_s_median_seed_mean"] is not None
    assert measured["d_s_median_seed_standard_deviation"] is not None
    assert measured["d_s_median_seed_standard_deviation"] == pytest.approx(
        measured["d_s_median_seed_standard_error"] * np.sqrt(8.0)
    )
    assert measured["p_return_standard_error"].shape == est.SIGMA_GRID.shape


def test_sigma_grid_has_calibration_window_margin(calibration_rows):
    assert est.SIGMA_GRID_POINTS == 141
    row = calibration_rows["torus3d_24"]
    assert row["window_point_count"] > est.WINDOW_MIN_POINTS
    assert row["window_point_margin"] > 0


def test_weyl_k_scan_records_admissible_range(calibration_rows):
    assert est.WEYL_ADMISSIBLE_K_RANGE == (100, 600)
    for name, expected in (("torus2d_64", 2), ("torus3d_24", 3)):
        scan = calibration_rows[name]["weyl_k_scan"]
        assert set(scan) == {str(k) for k in est.WEYL_K_SCAN}
        for k in (100, 128, 200, 300, 400, 600):
            assert cal.integer_band_check(scan[str(k)], expected)


def test_probe_reduction_ledger_is_derived():
    rows = [
        {"name": "dense", "probes": None},
        {"name": "full", "probes": est.HUTCHINSON_PROBES},
        {"name": "reduced", "probes": est.HUTCHINSON_PROBES // 2},
    ]
    assert probe.derive_probe_count_reductions(rows) == [
        {
            "name": "reduced",
            "configured_probes": est.HUTCHINSON_PROBES,
            "actual_probes": est.HUTCHINSON_PROBES // 2,
        }
    ]


def test_evidential_status_constant_is_canonical():
    assert EVIDENTIAL_STATUS == "exploratory_non_evidential"


def test_stochastic_estimator_deterministic(small_window):
    level_graphs, incidences = small_window
    union, _ = operators.union_laplacian(level_graphs, incidences, kappa=1.0)
    _, labels = operators.component_structure(union)
    basis = operators.kernel_basis_from_components(labels)
    first = est.return_probability_stochastic(union, basis, est.SIGMA_GRID, probes=4)
    second = est.return_probability_stochastic(union, basis, est.SIGMA_GRID, probes=4)
    assert np.array_equal(first, second)


def test_receipt_canonical_bytes():
    document = {
        "b": np.float64(1.23456789012345),
        "a": [np.int64(3), float("nan"), {"z": np.bool_(True)}],
    }
    first = receipts.canonical_bytes(document)
    second = receipts.canonical_bytes(document)
    assert first == second
    assert first.endswith(b"\n")
    text = first.decode("utf-8")
    assert '"b":1.23456789' in text
    assert "null" in text
    assert text.index('"a"') < text.index('"b"')
    assert receipts.receipt_sha256(document) == hashlib.sha256(first).hexdigest()
