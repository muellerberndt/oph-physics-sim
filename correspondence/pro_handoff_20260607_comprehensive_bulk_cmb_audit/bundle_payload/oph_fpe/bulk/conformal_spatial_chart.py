from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.cap_normals import cap_gram_matrix, cap_normal_report
from oph_fpe.bulk.h3_chart import h3_chart_report
from oph_fpe.bulk.lorentz_algebra import lorentz_algebra_report
from oph_fpe.claims import PAPER_THEOREM_3D_BULK_CHART_RECEIPT


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
