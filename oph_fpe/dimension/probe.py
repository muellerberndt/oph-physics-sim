"""Exploratory sweep orchestration for the dimension probe.

Non-evidential.  The sweep is DESIGN.md section 6: calibration cases,
dense/stochastic cross-checks, single-level controls, coupled unions over
the pinned kappa grid, decoupled union controls, and static construction
controls.  Every configuration is reported; none is thresholded.  The
receipt is canonical JSON with a printed SHA-256.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy
import scipy.sparse as sp

from oph_fpe.dimension import calibration as cal
from oph_fpe.dimension import estimators as est
from oph_fpe.dimension import geometry, operators, receipts

KAPPA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
UNION_TOP_LEVELS = (3, 4, 5)
SINGLE_CONTROL_LEVELS = (3, 4, 5)
LEVEL_CEILING = 5

CROSS_CHECK_DS_POINTWISE_TOL = 0.15
CROSS_CHECK_DS_MEDIAN_TOL = 0.10
CROSS_CHECK_P_REL_TOL = 0.05
CROSS_CHECK_SATURATION_GUARD = 4.0

CLAIM_BOUNDARY = (
    "Exploratory, non-evidential spectral statistics of declared finite "
    "operators on the committed tower and on calibration graphs. The static "
    "union statistic is partially a property of the constructed graph. No "
    "number is a measurement of physical spatial dimension; no verdict is "
    "attached; open premise rows of the committed corpus stay open."
)

CONVENTIONS = (
    "C1 union presentation: levels kept as separate nodes and coupled; "
    "ascending-level offset indexing",
    "C2 intra-level normalization: unit seam weight at every level "
    "(level-0 convention of oph_fpe/em/base_carrier.laplacian_matrix "
    "extended level-uniformly, cell patch basis)",
    "C3 inter-level quadratic form: one parent-child edge per committed "
    "lineage pair, weight kappa times the committed "
    "conditional_expectation_weights entry; single-field stationarity "
    "reproduces the committed embed and conditional_expectation",
    "C4 coupling strength kappa grid {0.0, 0.25, 0.5, 1.0, 2.0}; the corpus "
    "is silent on the strength",
    "C5 estimator pins: sigma grid logspace(-3,4,71), window "
    "[4.0, 1/lambda_2] with minimum 5 points, 64 Rademacher probes, 120 "
    "Lanczos steps with full reorthogonalization, dense cap 2000, Weyl "
    "k=200 rank floor 8 shift -1e-6 zero cut 1e-9, pinned seeds "
    "(recorded fix of DESIGN.md 4.3: k moved from 64 to 200)",
    "C6 receipt floats rounded to 10 significant digits before canonical "
    "serialization",
    "C7 fiber weights: committed spherical-area expectation weights in "
    "place of the exact uniform 1/4; per-pair deviation recorded",
    "C8 cross-check comparison window: grid points with sigma <= 1/lambda_2, "
    "finite d_s on both paths, and dense P(sigma) >= 4*kernel_dim/N "
    "(saturation guard); the plateau window floor sigma_lo does not apply "
    "to the agreement check",
    "C9 timing sidecar: wall-clock values live outside the hashed receipt, "
    "in dimension_probe_timings.json",
)

CITATIONS = {
    "join_structure": "reverse-engineering-reality/Lean/QFT/CarrierJoinTransport.lean",
    "record_layer_expectation": "reverse-engineering-reality/Lean/QFT/JoinNetMorphism.lean",
    "refinement_maps": "oph_fpe/core/icosahedral.py (CellRefinementMap, GeodesicIcosahedralTower)",
    "level_adjacency": "oph_fpe/core/icosahedral.py (geodesic_icosahedral_patch_arrays, cells basis)",
    "intra_level_normalization": "oph_fpe/em/base_carrier.py (laplacian_matrix), oph_fpe/em/green.py",
    "carrier_family": "oph_fpe/core/array_geometry.py (icosahedral_tower)",
    "theory_frame": (
        "reverse-engineering-reality/paper/"
        "recovering_observer_spacetime_and_einstein_dynamics_from_overlap_consistency.tex"
    ),
}


def measure_operator(
    matrix: sp.csr_matrix,
    *,
    expected_components: int,
    probe_seed: int,
    force_stochastic: bool = False,
) -> dict:
    """Symmetry receipt, component check, and both estimators on one operator."""

    started = time.perf_counter()
    symmetry_residual = operators.require_symmetric(matrix)
    component_count, labels = operators.component_structure(matrix)
    if component_count != expected_components:
        raise ValueError(
            f"component count {component_count} does not match the "
            f"construction expectation {expected_components}"
        )
    kernel_basis = operators.kernel_basis_from_components(labels)
    measured = est.measure_dimensions(
        matrix,
        kernel_basis,
        expected_kernel_dim=component_count,
        hutchinson_seed=probe_seed,
        force_stochastic=force_stochastic,
    )
    measured["symmetry_max_abs_asymmetry"] = symmetry_residual
    measured["component_count"] = component_count
    measured["runtime_seconds"] = time.perf_counter() - started
    return measured


def run_calibration() -> list[dict]:
    rows = []
    for index, case in enumerate(cal.CALIBRATION_CASES):
        matrix = cal.build_calibration_graph(case["builder"])
        row = measure_operator(
            matrix,
            expected_components=1,
            probe_seed=est.HUTCHINSON_SEED + 1000 + index,
        )
        row["name"] = case["name"]
        row["kind"] = case["kind"]
        row.update(cal.calibration_verdict(row, case))
        rows.append(row)
    return rows


def run_cross_checks() -> list[dict]:
    """Dense against stochastic on dense-cap graphs (DESIGN.md 4.1)."""

    rows = []
    sigmas = est.SIGMA_GRID
    for index, case in enumerate(cal.CROSS_CHECK_CASES):
        matrix = cal.build_calibration_graph(case["builder"])
        seed = est.HUTCHINSON_SEED + 2000 + index
        dense = measure_operator(
            matrix, expected_components=1, probe_seed=seed
        )
        stochastic = measure_operator(
            matrix, expected_components=1, probe_seed=seed, force_stochastic=True
        )
        lambda_2 = dense["lambda_2"]
        saturation_floor = (
            CROSS_CHECK_SATURATION_GUARD
            * float(dense["kernel_dim"])
            / float(dense["node_count"])
        )
        mask = (
            (sigmas <= (1.0 / lambda_2 if lambda_2 else np.inf))
            & np.isfinite(dense["d_s_curve"])
            & np.isfinite(stochastic["d_s_curve"])
            & (dense["p_return"] >= saturation_floor)
        )
        if not np.any(mask):
            raise ValueError(
                f"empty cross-check comparison window for {case['name']}"
            )
        ds_pointwise = float(
            np.max(np.abs(dense["d_s_curve"][mask] - stochastic["d_s_curve"][mask]))
        )
        ds_median = float(
            abs(
                np.median(dense["d_s_curve"][mask])
                - np.median(stochastic["d_s_curve"][mask])
            )
        )
        p_rel = float(
            np.max(
                np.abs(dense["p_return"][mask] - stochastic["p_return"][mask])
                / dense["p_return"][mask]
            )
        )
        rows.append(
            {
                "name": case["name"],
                "node_count": dense["node_count"],
                "d_s_median_dense": dense["d_s_median"],
                "d_s_median_stochastic": stochastic["d_s_median"],
                "max_abs_d_s_pointwise_difference": ds_pointwise,
                "abs_d_s_median_difference": ds_median,
                "max_rel_p_return_error": p_rel,
                "comparison_point_count": int(np.count_nonzero(mask)),
                "comparison_window": (
                    "sigma <= 1/lambda_2, finite d_s on both paths, "
                    "dense P >= 4*kernel_dim/N"
                ),
                "tolerances": {
                    "d_s_pointwise": CROSS_CHECK_DS_POINTWISE_TOL,
                    "d_s_median": CROSS_CHECK_DS_MEDIAN_TOL,
                    "p_return_relative": CROSS_CHECK_P_REL_TOL,
                },
                "within_tolerance": bool(
                    ds_pointwise <= CROSS_CHECK_DS_POINTWISE_TOL
                    and ds_median <= CROSS_CHECK_DS_MEDIAN_TOL
                    and p_rel <= CROSS_CHECK_P_REL_TOL
                ),
                "probe_seed": seed,
            }
        )
    return rows


def tower_configurations() -> list[dict]:
    """The pinned sweep list, in deterministic order."""

    configs: list[dict] = []
    for level in SINGLE_CONTROL_LEVELS:
        configs.append(
            {
                "name": f"single_level_{level}",
                "kind": "single_level_control",
                "levels": [level],
                "kappa": 0.0,
                "static": False,
            }
        )
    for top in UNION_TOP_LEVELS:
        for kappa in KAPPA_GRID:
            kind = "decoupled_union_control" if kappa == 0.0 else "coupled_union"
            kappa_tag = str(kappa).replace(".", "p")
            configs.append(
                {
                    "name": f"union_0_to_{top}_kappa_{kappa_tag}",
                    "kind": kind,
                    "levels": list(range(top + 1)),
                    "kappa": float(kappa),
                    "static": False,
                }
            )
        configs.append(
            {
                "name": f"union_0_to_{top}_static",
                "kind": "construction_control",
                "levels": list(range(top + 1)),
                "kappa": None,
                "static": True,
            }
        )
    return configs


def run_tower_sweep(level_graphs, incidences) -> list[dict]:
    graphs_by_level = {graph.level: graph for graph in level_graphs}
    rows = []
    for index, config in enumerate(tower_configurations()):
        window_graphs = [graphs_by_level[level] for level in config["levels"]]
        window_incidences = [
            incidence
            for incidence in incidences
            if incidence.coarse_level in config["levels"]
            and incidence.fine_level in config["levels"]
        ]
        if config["static"]:
            matrix, meta = operators.union_laplacian(
                window_graphs, window_incidences, kappa=1.0, static_control=True
            )
        elif len(config["levels"]) == 1:
            matrix, meta = operators.single_level_laplacian(window_graphs[0])
        else:
            matrix, meta = operators.union_laplacian(
                window_graphs, window_incidences, kappa=config["kappa"]
            )
        expected_components = (
            len(config["levels"])
            if (not config["static"] and config["kappa"] == 0.0)
            else 1
        )
        if len(config["levels"]) == 1:
            expected_components = 1
        seed = est.HUTCHINSON_SEED + 3000 + index
        row = measure_operator(
            matrix, expected_components=expected_components, probe_seed=seed
        )
        row.update(
            {
                "name": config["name"],
                "kind": config["kind"],
                "levels": config["levels"],
                "kappa": config["kappa"],
                "intra_edge_count": meta["intra_edge_count"],
                "inter_edge_count": meta["inter_edge_count"],
            }
        )
        rows.append(row)
    return rows


def build_receipt() -> dict:
    started = time.perf_counter()
    _, level_graphs, incidences = geometry.build_tower_window(LEVEL_CEILING)
    calibration_rows = run_calibration()
    cross_check_rows = run_cross_checks()
    configuration_rows = run_tower_sweep(level_graphs, incidences)
    document = {
        "schema": "oph.dimension_probe.v1",
        "evidential_status": "exploratory_non_evidential",
        "claim_boundary": CLAIM_BOUNDARY,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "pins": {
            "sigma_grid": {"start_exponent": -3.0, "stop_exponent": 4.0, "num": 71},
            "sigma_grid_values": est.SIGMA_GRID,
            "hutchinson_probes": est.HUTCHINSON_PROBES,
            "lanczos_steps": est.LANCZOS_STEPS,
            "dense_cap": est.DENSE_CAP,
            "weyl_eigencount": est.WEYL_EIGENCOUNT,
            "weyl_rank_floor": est.WEYL_RANK_FLOOR,
            "weyl_shift": est.WEYL_SHIFT,
            "zero_mode_cut": est.ZERO_MODE_CUT,
            "window_sigma_lo": est.SIGMA_LO,
            "window_rule": "sigma in [4.0, 1/lambda_2], minimum 5 grid points",
            "window_min_points": est.WINDOW_MIN_POINTS,
            "hutchinson_seed": est.HUTCHINSON_SEED,
            "eigsh_seed": est.EIGSH_SEED,
            "kappa_grid": list(KAPPA_GRID),
            "union_top_levels": list(UNION_TOP_LEVELS),
            "single_control_levels": list(SINGLE_CONTROL_LEVELS),
            "level_ceiling": LEVEL_CEILING,
            "float_significant_digits": receipts.FLOAT_SIGNIFICANT_DIGITS,
            "dimension_tolerance": cal.DIMENSION_TOLERANCE,
            "cross_check_tolerances": {
                "d_s_pointwise": CROSS_CHECK_DS_POINTWISE_TOL,
                "d_s_median": CROSS_CHECK_DS_MEDIAN_TOL,
                "p_return_relative": CROSS_CHECK_P_REL_TOL,
            },
            "cross_check_comparison_window": (
                "sigma <= 1/lambda_2, finite d_s on both paths, "
                "dense P >= 4*kernel_dim/N"
            ),
            "cross_check_saturation_guard": CROSS_CHECK_SATURATION_GUARD,
        },
        "conventions": list(CONVENTIONS),
        "citations": CITATIONS,
        "tower": {
            "levels": [
                {"level": graph.level, "cell_count": graph.cell_count, "edge_count": graph.edge_count}
                for graph in level_graphs
            ],
            "refinements": [
                {
                    "coarse_level": incidence.coarse_level,
                    "fine_level": incidence.fine_level,
                    "max_uniform_deviation": incidence.max_uniform_deviation,
                    "max_parent_weight_sum_residual": incidence.max_parent_weight_sum_residual,
                }
                for incidence in incidences
            ],
        },
        "calibration": calibration_rows,
        "cross_checks": cross_check_rows,
        "configurations": configuration_rows,
        "probe_count_reductions": [],
    }
    timings = {
        "label": "unhashed_timing_sidecar",
        "calibration_seconds": {
            row["name"]: row.pop("runtime_seconds") for row in calibration_rows
        },
        "configuration_seconds": {
            row["name"]: row.pop("runtime_seconds") for row in configuration_rows
        },
        "total_seconds": time.perf_counter() - started,
    }
    return document, timings


def summary_lines(document: dict) -> list[str]:
    lines = ["calibration:"]
    for row in document["calibration"]:
        lines.append(
            f"  {row['name']:<16} n={row['node_count']:<6} "
            f"d_s_median={_fmt(row['d_s_median'])} d_weyl={_fmt(row['d_weyl'])}"
        )
    lines.append("configurations:")
    for row in document["configurations"]:
        lines.append(
            f"  {row['name']:<28} n={row['node_count']:<6} "
            f"d_s_median={_fmt(row['d_s_median'])} d_weyl={_fmt(row['d_weyl'])} "
            f"window_pts={row['window_point_count']}"
        )
    return lines


def _fmt(value) -> str:
    return "none" if value is None else f"{value:.4f}"
