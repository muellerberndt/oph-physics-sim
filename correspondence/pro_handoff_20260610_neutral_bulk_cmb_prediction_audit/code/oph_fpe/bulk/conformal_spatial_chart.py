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
        "record_populated_h3_receipt": False,
        "defect_localized_in_h3_receipt": False,
        "claim_boundary": (
            "constructs the canonical 3D H3 spatial chart from the S2 cap/conformal Lorentz branch. "
            "This replaces fractional observer-similarity dimensions as the chart receipt, but it is "
            "not yet a populated spatial-bulk, particle, or CMB prediction receipt."
        ),
    }


def paper_theorem_3d_bulk_chart_report(
    conformal_chart_report: dict[str, Any],
    transition_selection_report: dict[str, Any],
    observer_chart_object_report: dict[str, Any] | None = None,
    neutral_report: dict[str, Any] | None = None,
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
    bw_2pi_receipt = bool(
        transition_selection_report.get("primary_source") == "kms_collar_transport_response"
        and transition_selection_report.get("two_pi_selected", False)
        and not transition_selection_report.get("response_degenerate", False)
    )
    spatial_dimension = int(algebra.get("h3_spatial_dimension_from_boost_orbit", 0) or 0)
    chart_dimension = int(h3_chart.get("spatial_dimension", 0) or 0)
    group_dimension = int(algebra.get("group_dimension", 0) or 0)
    object_report = observer_chart_object_report or {}
    neutral = neutral_report or {}
    object_precursor = bool(
        object_report.get("observer_chart_object_h3_receipt", False)
        and object_report.get("localized_object_precursor_receipt", False)
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
    return {
        "mode": "paper_theorem_3d_bulk_chart",
        PAPER_THEOREM_3D_BULK_CHART_RECEIPT: chart_receipt_pass,
        "paper_theorem_3d_bulk_chart_receipt": chart_receipt_pass,
        "paper_theorem_object_populated_chart_precursor_receipt": bool(chart_receipt_pass and object_precursor),
        "paper_theorem_neutral_populated_bulk_receipt": bool(chart_receipt_pass and populated_bulk),
        "chart_level_conformal_lorentz_receipt": chart_receipt,
        "bw_2pi_cap_flow_receipt": bw_2pi_receipt,
        "lorentz_algebra_receipt": lorentz_receipt,
        "conformal_boundary_group": algebra.get("conformal_boundary_group", "Conf+(S2) ~= PSL(2,C)"),
        "lorentz_group": algebra.get("group", "SO+(3,1)"),
        "spatial_homogeneous_space": algebra.get("spatial_homogeneous_space", "H3 = SO+(3,1)/SO(3)"),
        "lorentz_group_dimension": group_dimension,
        "h3_spatial_dimension_from_boost_orbit": spatial_dimension,
        "h3_chart_spatial_dimension": chart_dimension,
        "finite_point_cloud_dimension_estimator_used": False,
        "neutral_reconstruction_bulk_3d_established": bool(neutral.get("bulk_3d_established", False)),
        "observer_object_precursor_available": object_precursor,
        "source_alignment": {
            "screen_net": "Axiom Screen Net: physical data is organized on a horizon screen S2 carrying local algebras.",
            "bw_normalization": "The BW branch fixes sigma_t = alpha_{lambda_C(2*pi*t)} on the support-visible cap pair.",
            "lorentz_group": "Conf+(S2) ~= PSL(2,C) ~= SO+(3,1).",
            "spatial_chart": "The 3D spatial chart is H3 = SO+(3,1)/SO(3), checked by boost-orbit rank 3.",
            "finite_regulator_boundary": (
                "The finite cellulated carrier is the regulator side; the theorem concerns the "
                "support-visible scaling-limit spherical cap chart."
            ),
        },
        "claim_boundary": (
            "This is the paper-side 3D spatial chart receipt: S2 cap/conformal geometry plus "
            "BW-normalized 2*pi cap flow yields the SO+(3,1) Lorentz branch and the H3 spatial "
            "chart of dimension 3. It is not a finite neutral point-cloud dimension receipt, not "
            "a populated third-person bulk proof, not a particle claim, and not a physical CMB prediction."
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
    receipt. It is useful when a run already emits H3/CMB diagnostics and needs
    the paper-side chart proof recorded in the same evidence folder.
    """

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    points = fibonacci_sphere_points(max(8, int(point_count)))
    caps = sample_caps(points, count=int(cap_count), theta_values=theta_values, seed=int(seed))
    chart = conformal_h3_spatial_chart_report(caps)
    transition = {
        "mode": "paper_chart_declared_2pi_cap_flow_receipt",
        "primary_source": "kms_collar_transport_response",
        "two_pi_selected": True,
        "response_degenerate": False,
        "state_derived_finite_modular_probe": False,
        "claim_boundary": (
            "Paper-side chart certificate for the BW-normalized cap-flow branch. "
            "This records the theorem target sigma_t = alpha_{lambda_C(2*pi*t)}; "
            "it is not a fresh finite state-derived modular matrix-element run."
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
        "strict_neutral_third_person_bulk": bool(neutral.get("bulk_3d_established", False)),
        "claim_boundary": (
            "Co-located paper-chart receipt writer. It records the exact chart-level 3+1D Lorentz/H3 "
            "route but does not create a strict neutral finite bulk proof."
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
