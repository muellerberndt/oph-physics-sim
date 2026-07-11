from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.cap_normals import cap_gram_matrix, cap_normal_report
from oph_fpe.bulk.h3_chart import h3_chart_report
from oph_fpe.bulk.lorentz_algebra import lorentz_algebra_report
from oph_fpe.claims import PAPER_THEOREM_3D_BULK_CHART_RECEIPT
from oph_fpe.core.graph import fibonacci_sphere_points


def conformal_h3_spatial_chart_report(caps: list[RoundCap]) -> dict[str, Any]:
    normal_report = cap_normal_report(caps)
    chart_report = h3_chart_report(caps)
    algebra_report = lorentz_algebra_report()
    gram = cap_gram_matrix(caps)
    finite_gram = gram[np.isfinite(gram)] if gram.size else np.zeros(0, dtype=float)
    receipt = bool(
        normal_report.get("unit_normal_receipt", False)
        and chart_report.get("conformal_h3_spatial_chart_receipt", False)
        and algebra_report.get("lorentz_algebra_receipt", False)
    )
    return {
        "mode": "conformal_lorentz_to_h3_spatial_chart",
        "cap_normal_report": normal_report,
        "h3_chart_report": chart_report,
        "lorentz_algebra_report": algebra_report,
        "cross_ratio_proxy": {
            "kind": "minkowski_cap_normal_gram",
            "entry_count": int(finite_gram.size),
            "min": float(np.min(finite_gram)) if finite_gram.size else None,
            "median": float(np.median(finite_gram)) if finite_gram.size else None,
            "max": float(np.max(finite_gram)) if finite_gram.size else None,
        },
        "conformal_h3_spatial_chart_receipt": receipt,
        "lorentz_algebra_receipt": bool(algebra_report.get("lorentz_algebra_receipt", False)),
        "spatial_homogeneous_space": chart_report.get("homogeneous_space", "SO+(3,1)/SO(3)"),
        "spatial_dimension_derivation": chart_report.get(
            "spatial_dimension_derivation",
            "dim SO+(3,1)-dim SO(3)=6-3=3",
        ),
        "record_populated_h3_receipt": False,
        "defect_localized_in_h3_receipt": False,
        "claim_boundary": (
            "constructs the canonical 3D H3 spatial chart from the S2 cap/conformal Lorentz branch. "
            "The receipt tier is legacy chart-level; the issue #309 theorem-aligned receipt is "
            "CAP_NORMAL_H3_CHART_RECEIPT, recomputed from primitive null-section, cap-normal, "
            "Lorentz-equivariance, and H3-sheet fields. Populated spatial-bulk, neutral bulk, "
            "particle, and CMB prediction claims require separate gates."
        ),
    }


def paper_theorem_3d_bulk_chart_report(
    conformal_chart_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    observer_chart_object_report: dict[str, Any] | None = None,
    neutral_report: dict[str, Any] | None = None,
    state_bw_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report the paper-side 3D spatial chart receipt separately from neutral reconstruction.

    OPH's Lorentz/3D branch is a cap-chart theorem surface: round caps on the
    support-visible S2 chart carry BW-normalized modular flow and Conf+(S2)
    gives SO+(3,1). The spatial chart is H3 = SO+(3,1)/SO(3), whose dimension
    is the rank of the boost orbit, not a fitted dimension of a finite observer
    point cloud.
    """
    h3_chart = conformal_chart_report.get("h3_chart_report", {})
    algebra = conformal_chart_report.get("lorentz_algebra_report", {})
    chart_receipt = bool(conformal_chart_report.get("conformal_h3_spatial_chart_receipt", False))
    lorentz_receipt = bool(algebra.get("lorentz_algebra_receipt", False))
    assumed_bw_2pi_receipt = bool(
        transition_selection_report.get("scope") == "visualization_only"
        and transition_selection_report.get("SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT") is True
        and transition_selection_report.get("computed_theorem_receipts_unchanged") is True
    )
    # A declared 2pi target is an assumption, never a computed theorem receipt.
    declared_bw_2pi_receipt = False
    state_bw = state_bw_report or {}
    inferred_clock = state_bw.get("inferred_modular_clock_fit") or {}
    finite_lorentz_modular_clock_receipt = bool(
        (
            state_bw.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT", False)
            or state_bw.get("endogenous_modular_generator_receipt", False)
        )
        and (
            state_bw.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT", False)
            or state_bw.get("kms_geometric_clock_fit_receipt", False)
            or inferred_clock.get("receipt", False)
        )
    )
    bw_2pi_receipt = finite_lorentz_modular_clock_receipt
    spatial_dimension = int(algebra.get("h3_spatial_dimension_from_boost_orbit", 0) or 0)
    chart_dimension = int(h3_chart.get("spatial_dimension", 0) or 0)
    group_dimension = int(algebra.get("group_dimension", 0) or 0)
    spatial_dimension_derivation = str(
        algebra.get(
            "spatial_dimension_derivation",
            h3_chart.get("spatial_dimension_derivation", "dim SO+(3,1)-dim SO(3)=6-3=3"),
        )
    )
    object_report = observer_chart_object_report or {}
    neutral = neutral_report or {}
    localized_object_precursor = bool(
        object_report.get("localized_object_precursor_receipt", False)
        or object_report.get("localized_nonboundary_object_precursor_receipt", False)
        or object_report.get("localized_h3_object_precursor_receipt", False)
    )
    h3_object_precursor = bool(
        object_report.get("observer_chart_object_h3_receipt", False)
        or object_report.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT", False)
        or object_report.get("modular_response_h3_control_separation_receipt", False)
        or object_report.get("H3_RESPONSE_CONTROL_SEPARATION_RECEIPT", False)
        or object_report.get("h3_control_separation_receipt", False)
    )
    object_precursor = bool(
        localized_object_precursor
        and h3_object_precursor
    )
    populated_bulk = bool(
        object_report.get("localized_nonboundary_bulk_population_receipt", False)
        and neutral.get("bulk_3d_established", False)
    )
    chart_receipt_pass = bool(
        chart_receipt
        and lorentz_receipt
        and bw_2pi_receipt
        and group_dimension == 6
        and spatial_dimension == 3
        and chart_dimension == 3
    )
    assumed_chart_receipt = bool(
        chart_receipt
        and lorentz_receipt
        and assumed_bw_2pi_receipt
        and group_dimension == 6
        and spatial_dimension == 3
        and chart_dimension == 3
    )
    return {
        "mode": "paper_theorem_3d_bulk_chart",
        PAPER_THEOREM_3D_BULK_CHART_RECEIPT: chart_receipt_pass,
        "paper_theorem_3d_bulk_chart_receipt": chart_receipt_pass,
        "paper_theorem_object_populated_chart_precursor_receipt": bool(chart_receipt_pass and object_precursor),
        "paper_theorem_neutral_populated_bulk_receipt": bool(chart_receipt_pass and populated_bulk),
        "chart_level_conformal_lorentz_receipt": chart_receipt,
        "bw_2pi_cap_flow_receipt": bw_2pi_receipt,
        "declared_bw_2pi_cap_flow_receipt": declared_bw_2pi_receipt,
        "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": assumed_bw_2pi_receipt,
        "simulation_assumed_bw_2pi_geometric_branch_receipt": assumed_bw_2pi_receipt,
        "SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT": assumed_chart_receipt,
        "simulation_assumed_3d_h3_chart_receipt": assumed_chart_receipt,
        "finite_lorentz_modular_clock_receipt": finite_lorentz_modular_clock_receipt,
        "bw_2pi_cap_flow_source": (
            "finite_endogenous_l2_l3_modular_clock"
            if finite_lorentz_modular_clock_receipt
            else None
        ),
        "lorentz_algebra_receipt": lorentz_receipt,
        "conformal_boundary_group": algebra.get("conformal_boundary_group", "Conf+(S2) ~= PSL(2,C)"),
        "lorentz_group": algebra.get("group", "SO+(3,1)"),
        "spatial_homogeneous_space": algebra.get("spatial_homogeneous_space", "H3 = SO+(3,1)/SO(3)"),
        "lorentz_group_dimension": group_dimension,
        "h3_spatial_dimension_from_boost_orbit": spatial_dimension,
        "h3_chart_spatial_dimension": chart_dimension,
        "spatial_dimension_derivation": spatial_dimension_derivation,
        "finite_point_cloud_dimension_estimator_used": False,
        "neutral_reconstruction_bulk_3d_established": bool(neutral.get("bulk_3d_established", False)),
        "observer_object_precursor_available": object_precursor,
        "observer_object_precursor_components": {
            "localized_object_precursor_receipt": localized_object_precursor,
            "h3_object_precursor_receipt": h3_object_precursor,
            "strict_object_h3_receipt": bool(object_report.get("observer_chart_object_h3_receipt", False)),
            "h3_control_separation_receipt": bool(
                object_report.get("modular_response_h3_control_separation_receipt", False)
                or object_report.get("H3_RESPONSE_CONTROL_SEPARATION_RECEIPT", False)
                or object_report.get("h3_control_separation_receipt", False)
            ),
        },
        "source_alignment": {
            "screen_net": "Axiom Screen Net: physical data is organized on a horizon screen S2 carrying local algebras.",
            "bw_normalization": (
                "The issue #308 theorem target is sigma_t = alpha_{lambda_C_hat(2*pi*t)} on a "
                "BW-framed support-visible cap pair. This chart report records the paper route; BW3 "
                "requires the separate finite cap-normal BWRec audit."
            ),
            "lorentz_group": "Conf+(S2) ~= PSL(2,C) ~= SO+(3,1).",
            "spatial_chart": "The 3D spatial chart is H3 = SO+(3,1)/SO(3), with dimension 6-3=3.",
            "cap_normal_h3_chart": (
                "Issue #309 chart certification requires CAP_NORMAL_H3_CHART_RECEIPT: q(Omega)=(1,Omega), "
                "n_C=(cot(alpha),csc(alpha)c), signed cap incidence, n_gC=Lambda_g n_C, and H3 future-sheet checks."
            ),
            "finite_regulator_boundary": (
                "The finite cellulated carrier is the regulator side; the theorem concerns the "
                "support-visible scaling-limit spherical cap chart."
            ),
        },
        "claim_boundary": (
            "This is the paper-side 3D spatial chart receipt: S2 cap/conformal geometry plus "
            "a computed finite modular-clock receipt yields the SO+(3,1) Lorentz branch and "
            "the H3 spatial chart of dimension 3. The receipt tier is chart-level; issue #309 "
            "primitive-field chart certification uses CAP_NORMAL_H3_CHART_RECEIPT. A declared "
            "2pi target is reported only as SIMULATION_ASSUMED_* visualization data. BW3 finite "
            "cap-net evidence, populated third-person bulk, particle, "
            "and physical CMB claims require separate gates."
        ),
    }


def write_paper_chart_receipts(
    run_dir: Path,
    *,
    point_count: int = 4096,
    cap_count: int = 32,
    theta_values: tuple[float, ...] = (0.35, 0.55, 0.75, 1.0, 1.25),
    seed: int = 20260610,
) -> dict[str, Any]:
    """Write paper-aligned S2 -> Lorentz -> H3 chart receipts for a run folder.

    This is a chart/theorem receipt, not a state-derived finite modular-probe
    receipt. It is useful when a run emits H3/CMB diagnostics and needs
    the paper-side chart proof recorded in the same evidence folder.
    """

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    points = fibonacci_sphere_points(max(8, int(point_count)))
    caps = sample_caps(points, count=int(cap_count), theta_values=theta_values, seed=int(seed))
    chart = conformal_h3_spatial_chart_report(caps)
    transition = {
        "mode": "paper_chart_explicit_2pi_visualization_assumption",
        "primary_source": "explicit_visualization_assumption",
        "scope": "visualization_only",
        "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": True,
        "computed_theorem_receipts_unchanged": True,
        "two_pi_selected": False,
        "response_degenerate": None,
        "state_derived_finite_modular_probe": False,
        "claim_boundary": (
            "Explicit visualization-only assumption of the BW-normalized cap-flow branch. "
            "This records the target sigma_t = alpha_{lambda_C(2*pi*t)} without changing any "
            "computed theorem receipt; finite state-derived modular runs use their own gate."
        ),
    }
    object_report = _read_json(root / "observer_chart_object_h3_scale_compressed_report.json") or _read_json(
        root / "observer_chart_object_h3_report.json"
    )
    neutral = _read_json(root / "bulk_reconstruction_report.json")
    paper = paper_theorem_3d_bulk_chart_report(chart, transition, object_report, neutral)
    emergence = _read_json(root / "emergence_status_report.json")
    emergence.update(
        {
            "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT": bool(
                paper.get("chart_level_conformal_lorentz_receipt", False)
            ),
            "CHART_LORENTZ_H3_RECEIPT": bool(paper.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)),
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": bool(
                paper.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)
            ),
            "BW_KMS_DIRECT_2PI_RECEIPT": bool(paper.get("bw_2pi_cap_flow_receipt", False)),
            "SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT": bool(
                paper.get("SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT", False)
            ),
            "SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT": bool(
                paper.get("SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT", False)
            ),
            "physical_cmb_prediction": bool(emergence.get("physical_cmb_prediction", False)),
            "bulk_3d_established": bool(emergence.get("bulk_3d_established", False)),
            "chart_receipt_source": "paper_chart_certificate",
            "chart_claim_boundary": (
                "Chart-level Lorentz/H3 receipt from the paper-side S2 cap conformal route. "
                "Does not assert strict neutral third-person bulk or physical CMB."
            ),
        }
    )
    (root / "conformal_h3_spatial_chart_report.json").write_text(
        json.dumps(chart, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "transition_selection_report.json").write_text(
        json.dumps(transition, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "paper_3d_bulk_chart_report.json").write_text(
        json.dumps(paper, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "emergence_status_report.json").write_text(
        json.dumps(emergence, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "mode": "paper_chart_receipt_writer",
        "run_path": str(root),
        "point_count": int(point_count),
        "cap_count": int(cap_count),
        "paper_theorem_3d_bulk_chart_receipt": bool(paper.get(PAPER_THEOREM_3D_BULK_CHART_RECEIPT, False)),
        "chart_level_conformal_lorentz_receipt": bool(paper.get("chart_level_conformal_lorentz_receipt", False)),
        "bw_2pi_cap_flow_receipt": bool(paper.get("bw_2pi_cap_flow_receipt", False)),
        "simulation_assumed_bw_2pi_geometric_branch_receipt": bool(
            paper.get("SIMULATION_ASSUMED_BW_2PI_GEOMETRIC_BRANCH_RECEIPT", False)
        ),
        "simulation_assumed_3d_h3_chart_receipt": bool(
            paper.get("SIMULATION_ASSUMED_3D_H3_CHART_RECEIPT", False)
        ),
        "strict_neutral_third_person_bulk": bool(neutral.get("bulk_3d_established", False)),
        "claim_boundary": (
            "Co-located paper-chart receipt writer. It records the exact chart-level 3+1D Lorentz/H3 "
            "route. Strict neutral finite bulk proof claims require their separate gate."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
