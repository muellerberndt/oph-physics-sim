from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

CERTIFIED = "MODULAR_RESPONSE_H3_LOCALIZATION_CERTIFIED"
AMBIGUOUS = "H3_LOCALIZATION_AMBIGUOUS"
OUTSIDE_DECLARED_DOMAIN = "OUTSIDE_DECLARED_DOMAIN"
FRAME_UNCONDITIONED = "H3_CAP_FRAME_UNCONDITIONED"
POINT_SOURCE_UNCERTIFIED = "POINT_SOURCE_FACTOR_UNCERTIFIED"
ERROR_UNBOUNDED = "H3_LOCALIZATION_ERROR_UNBOUNDED"
INCOMPLETE = "H3_LOCALIZATION_CERTIFICATE_INCOMPLETE"

CERTIFIED_RESIDUAL_INTERVAL_METHODS = frozenset(
    {
        "directed_rounding_interval_arithmetic",
        "exact_rational_residuals",
        "analytic_residual_enclosure",
    }
)


def modular_response_h3_localization_report(payload: dict[str, Any], *, tol: float = 1.0e-8) -> dict[str, Any]:
    """Audit issue #310 modular-response localization from primitive fields.

    The receipt certifies conditional observer-facing H3 record localization.
    It does not infer point locality from a fitted point, promote existing H3
    coordinates to an independent reconstruction, derive particle species,
    construct stress-energy, or establish neutral third-person bulk.
    """

    source = dict(payload or {})
    kernel_type = _kernel_type(source)
    radius_declared = "curvature_radius" in source
    parsed_radius = _finite_positive(source.get("curvature_radius"))
    radius_input_valid = bool(not radius_declared or parsed_radius is not None)
    radius = float(parsed_radius if parsed_radius is not None else 1.0)
    normals, normals_input_valid = _points4_checked(
        source.get("normals") or source.get("cap_normals") or []
    )
    weight_report = _weight_matrix_report(source.get("weights"), len(normals), tol=tol)
    weights = np.asarray(weight_report.pop("matrix"), dtype=float)
    frame = _frame_report(normals, weights, radius=radius)
    domain = _domain_report(source.get("domain") or {}, radius=radius, tol=tol)
    compact_lipschitz = _compact_domain_lipschitz_report(
        frame=frame,
        weights=weights,
        domain=domain,
        radius=radius,
        kernel_type=kernel_type,
    )
    raw_observations = source.get("tokens") or source.get("observations") or []
    observations_input_valid = isinstance(raw_observations, list)
    observations = raw_observations if observations_input_valid else []
    frame_alpha_bound = float(
        frame["sigma_min"] / np.sqrt(2.0)
        if kernel_type == "paired_hinge"
        else frame["sigma_min"]
    )
    alpha_declared = "alpha" in source
    declared_alpha = _finite_positive(source.get("alpha"))
    alpha_input_valid = bool(not alpha_declared or declared_alpha is not None)
    alpha = float(
        declared_alpha
        if declared_alpha is not None
        else 0.0
        if alpha_declared
        else frame_alpha_bound
    )
    lipschitz_key = "L" if "L" in source else "lipschitz_L" if "lipschitz_L" in source else None
    declared_lipschitz = _finite_positive(source.get(lipschitz_key)) if lipschitz_key else None
    derived_lipschitz = _finite_positive(compact_lipschitz.get("global_L_bound"))
    lipschitz_input_valid = bool(
        declared_lipschitz is not None
        if lipschitz_key is not None
        else derived_lipschitz is not None and compact_lipschitz["certified"]
    )
    lipschitz = float(
        declared_lipschitz
        if declared_lipschitz is not None
        else 0.0
        if lipschitz_key is not None
        else derived_lipschitz
        if derived_lipschitz is not None
        else 0.0
    )
    alpha_consistent = bool(alpha <= frame_alpha_bound + tol)
    lipschitz_consistent = bool(
        compact_lipschitz["certified"]
        and derived_lipschitz is not None
        and lipschitz + tol >= derived_lipschitz
    )
    observability_assumption = _explicit_visualization_assumption(
        source.get("observability_assumption"), "alpha_and_lipschitz_bounds"
    )
    epsilon_declared = "epsilon" in source
    epsilon = _finite_nonnegative(source.get("epsilon", domain.get("epsilon", 0.0)))
    epsilon_input_valid = bool(not epsilon_declared or epsilon is not None)
    tau_declared = "tau" in source
    tau = _finite_nonnegative(source.get("tau", 0.0))
    tau_input_valid = bool(not tau_declared or tau is not None)
    global_error_inputs_valid = bool(
        epsilon_input_valid and tau_input_valid and epsilon is not None and tau is not None
    )
    token_rows = [
        _token_report(
            row,
            normals=normals,
            weights=weights,
            radius=radius,
            alpha=alpha,
            lipschitz=lipschitz,
            epsilon=float(epsilon or 0.0),
            tau=float(tau or 0.0),
            domain=domain,
            kernel_type=kernel_type,
            global_error_inputs_valid=global_error_inputs_valid,
            tol=tol,
        )
        for row in observations
    ]
    token_gap_receipt = bool(token_rows and all(row["gap_receipt"] for row in token_rows))
    token_error_receipt = bool(token_rows and all(row["error_receipt"] for row in token_rows))
    token_sigma_receipt = bool(token_rows and all(row.get("sigma_input_valid", False) for row in token_rows))
    token_interval_receipt = bool(
        token_rows and all(row.get("residual_interval_receipt", False) for row in token_rows)
    )
    token_handoff_receipt = bool(token_rows and all(row["worldline_handoff_receipt"] for row in token_rows))
    all_inside_domain = bool(token_rows and all(row["inside_declared_domain"] for row in token_rows))
    net_coverage_receipt = bool(token_rows and all(row["epsilon_net_coverage_receipt"] for row in token_rows))
    net_coverage_assumed = bool(token_rows and all(row["epsilon_net_coverage_assumed"] for row in token_rows))

    geometry_receipt = bool(
        radius_input_valid
        and normals_input_valid
        and normals.size
        and frame["normal_unit_residual"] <= tol
    )
    weight_psd_receipt = bool(weight_report["valid_psd"])
    weight_receipt = bool(weight_report["valid_positive_definite"])
    observability_bounds_receipt = bool(
        alpha_input_valid
        and lipschitz_input_valid
        and alpha_consistent
        and lipschitz_consistent
    )
    frame_receipt = bool(
        weight_receipt
        and frame["rank"] >= 4
        and frame["sigma_min"] > tol
        and alpha > 0.0
        and observability_bounds_receipt
    )
    kernel_receipt = _kernel_receipt(source)
    point_receipt = _point_source_receipt(source)
    domain_receipt = bool(domain["compact"] and domain["center_on_h3"] and all_inside_domain and net_coverage_receipt)
    error_receipt = bool(
        global_error_inputs_valid
        and token_error_receipt
        and np.isfinite(alpha)
        and alpha > 0.0
        and lipschitz >= alpha
    )
    naturality_receipt = _strict_true((source.get("chart_naturality") or {}).get("passed", False))
    refinement_receipt = _strict_true((source.get("refinement") or {}).get("passed", False))
    negative_controls_receipt = _negative_controls_receipt(source)
    no_shortcut = not bool(source.get("preexisting_h3_coordinates_used_as_reconstruction", False))

    component_receipts = {
        "H3_CURVATURE_RADIUS_VALID_RECEIPT": radius_input_valid,
        "H3_PRIMITIVE_INPUT_SHAPE_RECEIPT": bool(normals_input_valid and observations_input_valid),
        "H3_CAP_GEOMETRY_RECEIPT": geometry_receipt,
        "H3_WEIGHT_MATRIX_PSD_RECEIPT": weight_psd_receipt,
        "H3_WEIGHT_MATRIX_POSITIVE_DEFINITE_RECEIPT": weight_receipt,
        "H3_OBSERVABILITY_BOUNDS_RECEIPT": observability_bounds_receipt,
        "H3_CAP_FRAME_RECEIPT": frame_receipt,
        "H3_MODULAR_RESPONSE_KERNEL_RECEIPT": kernel_receipt,
        "H3_POINT_SOURCE_FACTOR_RECEIPT": point_receipt,
        "H3_LOCALIZATION_DOMAIN_RECEIPT": domain_receipt,
        "H3_EPSILON_NET_COVERAGE_RECEIPT": net_coverage_receipt,
        "H3_LOCALIZATION_ERROR_RECEIPT": error_receipt,
        "H3_TOTAL_ERROR_SIGMA_RECEIPT": token_sigma_receipt,
        "H3_RESIDUAL_INTERVAL_CERTIFICATE_RECEIPT": token_interval_receipt,
        "H3_LOCALIZATION_GAP_RECEIPT": token_gap_receipt,
        "H3_CHART_NATURALITY_RECEIPT": naturality_receipt,
        "H3_REFINEMENT_LOCALIZATION_RECEIPT": refinement_receipt,
        "H3_WORLDLINE_HANDOFF_RECEIPT": token_handoff_receipt,
        "H3_NEGATIVE_CONTROLS_RECEIPT": negative_controls_receipt,
        "H3_NO_PREEXISTING_COORDINATE_SHORTCUT_RECEIPT": no_shortcut,
    }
    blockers = _blockers(component_receipts, frame=frame, domain=domain, token_rows=token_rows)
    status = _terminal_status(blockers, token_rows)
    receipt = bool(status == CERTIFIED)

    return {
        "mode": "modular_response_h3_localization_audit",
        "claim_id": "OPH-GR-D3-H3-LOCALIZATION",
        "receipt_type": MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
        "terminal_status": status,
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT: receipt,
        "h3_modular_response_localization_receipt": receipt,
        "H3LOC": receipt,
        "curvature_radius": radius,
        "curvature_radius_input_valid": radius_input_valid,
        "normals_input_valid": normals_input_valid,
        "observations_input_valid": observations_input_valid,
        "frame": frame,
        "weights": weight_report,
        "domain": domain,
        "observability": {
            "alpha": alpha,
            "L": lipschitz,
            "epsilon": epsilon,
            "tau": tau,
            "global_error_inputs_valid": global_error_inputs_valid,
            "epsilon_input_valid": epsilon_input_valid,
            "tau_input_valid": tau_input_valid,
            "alpha_source": (
                "declared" if declared_alpha is not None else "invalid_declared" if alpha_declared else "sigma_min_frame"
            ),
            "L_source": (
                "declared"
                if declared_lipschitz is not None
                else "invalid_declared"
                if lipschitz_key is not None
                else "derived_compact_domain_global_bound"
                if derived_lipschitz is not None
                else "unavailable"
            ),
            "kernel_alpha_upper_bound": frame_alpha_bound,
            "paired_hinge_alpha_factor": 1.0 / np.sqrt(2.0) if kernel_type == "paired_hinge" else 1.0,
            "compact_domain_lipschitz_certificate": compact_lipschitz,
            "alpha_consistent_with_frame": alpha_consistent,
            "L_consistent_with_frame": lipschitz_consistent,
            "L_consistent_with_compact_domain_bound": lipschitz_consistent,
            "alpha_input_valid": alpha_input_valid,
            "L_input_valid": lipschitz_input_valid,
            "explicit_visualization_assumption": observability_assumption,
        },
        "simulation_assumption_status": {
            "observability_bounds_assumed": observability_assumption,
            "epsilon_net_coverage_assumed": net_coverage_assumed,
            "simulation_assumed_localization_eligible": bool(
                observability_assumption
                and net_coverage_assumed
                and all_inside_domain
                and token_gap_receipt
                and token_error_receipt
                and kernel_receipt
                and point_receipt
            ),
            "computed_theorem_receipt_affected": False,
        },
        "tokens": token_rows,
        "component_receipts": component_receipts,
        **component_receipts,
        "blockers": blockers,
        "mandatory_nonclaims": {
            "POINT_LOCALITY_DERIVED_FROM_FIT": False,
            "UNLABELED_SOURCE_MIXTURE_SOLVED": False,
            "EXTENDED_SOURCE_COLLAPSED_TO_POINT": False,
            "PARTICLE_SPECIES_DERIVED": False,
            "PHYSICAL_STRESS_TENSOR": False,
            "NEUTRAL_CHART_BLIND_BULK": False,
            "EINSTEIN_BRANCH_ENTRY": False,
        },
        "claim_boundary": (
            "MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT certifies issue #310 only when "
            "record-conditioned cap responses, point-source factorization, compact source domain, "
            "positive cap-frame observability, bounded total error, residual-gap uniqueness, "
            "chart naturality, refinement, and negative controls all pass. Existing H3 coordinates "
            "or visualization point clouds cannot set H3LOC=true."
        ),
    }


def write_modular_response_h3_localization_report(source: Path, out: Path) -> dict[str, Any]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    report = modular_response_h3_localization_report(payload)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _frame_report(normals: np.ndarray, weights: np.ndarray, *, radius: float) -> dict[str, Any]:
    if not normals.size:
        return {
            "normal_count": 0,
            "rank": 0,
            "sigma_min": 0.0,
            "sigma_max": 0.0,
            "normal_unit_residual": float("inf"),
            "B": [],
        }
    b = np.column_stack([-normals[:, 0], normals[:, 1], normals[:, 2], normals[:, 3]])
    weighted = _sqrtm(weights) @ b
    singular = np.linalg.svd(weighted, compute_uv=False)
    normal_res = float(np.max(np.abs([_eta(row, row) - 1.0 for row in normals])))
    rank = int(np.linalg.matrix_rank(weighted, tol=1.0e-10))
    return {
        "normal_count": int(len(normals)),
        "rank": rank,
        "sigma_min": float(singular[-1]) if singular.size else 0.0,
        "sigma_max": float(singular[0]) if singular.size else 0.0,
        "normal_unit_residual": normal_res,
        "B": b.tolist(),
        "curvature_radius": float(radius),
    }


def _compact_domain_lipschitz_report(
    *,
    frame: dict[str, Any],
    weights: np.ndarray,
    domain: dict[str, Any],
    radius: float,
    kernel_type: str | None,
) -> dict[str, Any]:
    """Certify a conservative response Lipschitz bound on a compact H3 ball.

    The response features are normalized by ``radius`` and the distance used by
    the localization theorem is ``d_H3 / radius``.  On a ball of radius ``D``
    centered at ``C``, every unit-speed H3 tangent has ambient Euclidean norm at
    most ``sqrt(cosh(2 (d(O,C)+D)/R))``.  Combining that factor with the response
    matrix operator norm gives a global bound on the whole declared ball.  A
    bare finite-frame ``sigma_max`` is therefore not, by itself, accepted as a
    global H3 Lipschitz certificate.
    """

    center = _vec4(domain.get("center"))
    domain_radius = _finite_positive(domain.get("D"))
    inputs_valid = bool(
        kernel_type in {"signed_linear", "identity", "paired_hinge"}
        and domain.get("compact", False)
        and center is not None
        and domain_radius is not None
        and np.isfinite(radius)
        and radius > 0.0
        and weights.ndim == 2
        and weights.shape[0] == weights.shape[1]
        and weights.shape[0] == int(frame.get("normal_count", 0))
        and weights.size > 0
        and np.all(np.isfinite(weights))
    )
    if not inputs_valid:
        return {
            "certified": False,
            "method": "compact_h3_ball_ambient_tangent_bound",
            "global_L_bound": None,
            "reason": "invalid_or_noncompact_domain_or_response_frame",
        }

    origin = np.asarray([radius, 0.0, 0.0, 0.0], dtype=float)
    center_distance = _h3_distance(origin, center, radius)
    maximum_origin_distance = center_distance + float(domain_radius)
    normalized_maximum_distance = maximum_origin_distance / float(radius)
    # cosh(2r) overflows in double precision above roughly r=355.  An
    # infinite numerical bound is not a certificate, so fail closed there.
    if not np.isfinite(normalized_maximum_distance) or normalized_maximum_distance > 350.0:
        return {
            "certified": False,
            "method": "compact_h3_ball_ambient_tangent_bound",
            "global_L_bound": None,
            "center_distance": float(center_distance),
            "maximum_origin_distance": float(maximum_origin_distance),
            "reason": "compact_domain_bound_not_finite_in_float64",
        }

    tangent_factor = float(np.sqrt(np.cosh(2.0 * normalized_maximum_distance)))
    if kernel_type == "paired_hinge":
        # Coordinatewise paired hinge is Euclidean 1-Lipschitz.  For a general
        # positive-definite cap weight W, use sqrt(lambda_max(W))*||B||_2;
        # sigma_max(sqrt(W)B) alone need not upper-bound the paired channels.
        maximum_weight_eigenvalue = float(np.max(np.linalg.eigvalsh(weights)))
        b = np.asarray(frame.get("B", []), dtype=float)
        response_operator_bound = float(
            np.sqrt(maximum_weight_eigenvalue) * np.linalg.norm(b, ord=2)
        )
        response_operator_method = "sqrt_lambda_max_W_times_unweighted_B_operator_norm"
    else:
        maximum_weight_eigenvalue = float(np.max(np.linalg.eigvalsh(weights)))
        response_operator_bound = float(frame.get("sigma_max", 0.0))
        response_operator_method = "weighted_B_operator_norm"
    global_bound = response_operator_bound * tangent_factor
    certified = bool(
        np.isfinite(global_bound)
        and global_bound > 0.0
        and np.isfinite(response_operator_bound)
        and response_operator_bound > 0.0
    )
    return {
        "certified": certified,
        "method": "compact_h3_ball_ambient_tangent_bound",
        "distance_normalization": "d_H3 / curvature_radius",
        "center_distance": float(center_distance),
        "domain_radius": float(domain_radius),
        "maximum_origin_distance": float(maximum_origin_distance),
        "ambient_tangent_factor": tangent_factor,
        "response_operator_bound": response_operator_bound,
        "response_operator_method": response_operator_method,
        "maximum_weight_eigenvalue": maximum_weight_eigenvalue,
        "global_L_bound": float(global_bound) if certified else None,
        "reason": None if certified else "nonpositive_or_nonfinite_operator_bound",
    }


def _domain_report(domain: dict[str, Any], *, radius: float, tol: float) -> dict[str, Any]:
    if not isinstance(domain, dict):
        return {
            "type": "invalid",
            "compact": False,
            "center": None,
            "center_on_h3": False,
            "D": None,
            "epsilon": None,
            "net_cardinality_bound": None,
            "primitive_input_valid": False,
        }
    center = _vec4(domain.get("center", [radius, 0.0, 0.0, 0.0]))
    physical_radius = _finite_positive(domain.get("D") or domain.get("radius"))
    epsilon = _finite_nonnegative(domain.get("epsilon", 0.0))
    center_on_h3 = bool(center is not None and _is_h3_point(center, radius=radius, tol=tol))
    compact = bool(
        str(domain.get("type", "ball")) in {"ball", "compact", "bounded"}
        and physical_radius is not None
        and center_on_h3
    )
    net_cardinality_bound = None
    if compact and epsilon and epsilon > 0.0:
        net_cardinality_bound = _ball_volume(float(physical_radius) + radius * epsilon / 2.0, radius) / _ball_volume(
            radius * epsilon / 2.0,
            radius,
        )
    return {
        "type": str(domain.get("type", "ball")),
        "compact": compact,
        "center": center.tolist() if center is not None else None,
        "center_on_h3": center_on_h3,
        "D": physical_radius,
        "epsilon": epsilon,
        "epsilon_net_certificate": domain.get("epsilon_net_certificate"),
        "epsilon_net_assumption": domain.get("epsilon_net_assumption"),
        "primitive_input_valid": True,
        "net_cardinality_bound": float(net_cardinality_bound) if net_cardinality_bound else None,
    }


def _token_report(
    row: dict[str, Any],
    *,
    normals: np.ndarray,
    weights: np.ndarray,
    radius: float,
    alpha: float,
    lipschitz: float,
    epsilon: float,
    tau: float,
    domain: dict[str, Any],
    kernel_type: str | None,
    global_error_inputs_valid: bool,
    tol: float,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "token_id": "malformed-token",
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "epsilon_net_coverage_receipt": False,
            "epsilon_net_coverage_assumed": False,
            "primitive_inputs_valid": False,
            "blockers": ["malformed_token"],
        }
    token_id = str(row.get("token_id", row.get("id", len(str(row)))))
    responses = _vec(row.get("responses") or row.get("y") or [])
    candidates, candidates_input_valid = _points4_checked(
        row.get("candidate_points") or row.get("net_points") or []
    )
    sigma_key = "sigma" if "sigma" in row else "total_error_sigma" if "total_error_sigma" in row else None
    sigma = _finite_nonnegative(row.get(sigma_key)) if sigma_key is not None else None
    sigma_input_valid = bool(sigma_key is not None and sigma is not None)
    tau_i = _finite_nonnegative(row.get("tau", tau))
    epsilon_i = _finite_nonnegative(row.get("epsilon", epsilon))
    primitive_inputs_valid = bool(responses is not None and candidates_input_valid)
    if not primitive_inputs_valid or candidates.size == 0 or normals.size == 0:
        return {
            "token_id": token_id,
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "epsilon_net_coverage_receipt": False,
            "epsilon_net_coverage_assumed": False,
            "primitive_inputs_valid": primitive_inputs_valid,
            "blockers": [
                "malformed_or_nonfinite_responses_or_candidates"
                if not primitive_inputs_valid
                else "missing_responses_or_candidates_or_normals"
            ],
        }
    feature = _response_features(candidates, normals, radius, kernel_type)
    if feature is None:
        return {
            "token_id": token_id,
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "epsilon_net_coverage_receipt": False,
            "epsilon_net_coverage_assumed": False,
            "primitive_inputs_valid": primitive_inputs_valid,
            "blockers": ["unsupported_response_kernel"],
        }
    if len(responses) != feature.shape[1]:
        return {
            "token_id": token_id,
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "epsilon_net_coverage_receipt": False,
            "epsilon_net_coverage_assumed": False,
            "primitive_inputs_valid": primitive_inputs_valid,
            "blockers": ["response_dimension_mismatch"],
        }
    response_weights = _response_weight_matrix(weights, kernel_type)
    residuals = np.asarray(
        [_weighted_norm(feature[index] - responses, response_weights) for index in range(len(candidates))]
    )
    best_index = int(np.argmin(residuals))
    sorted_residuals = np.sort(residuals)
    lower = _vec(row.get("residual_lower_bounds") or [])
    upper = _vec(row.get("residual_upper_bounds") or [])
    interval_certificate = _residual_interval_certificate_report(row, candidate_count=len(candidates))
    interval_fields_present = bool(
        row.get("residual_lower_bounds") is not None
        or row.get("residual_upper_bounds") is not None
        or row.get("residual_interval_certificate") is not None
    )
    interval_receipt = bool(
        interval_certificate["certified"]
        and lower is not None
        and upper is not None
        and len(lower) == len(candidates)
        and len(upper) == len(candidates)
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(lower >= 0.0)
        and np.all(upper >= 0.0)
        and np.all(lower <= upper + tol)
        and np.all(lower <= residuals + tol)
        and np.all(residuals <= upper + tol)
    )
    diagnostic_gap = (
        float(sorted_residuals[1] - sorted_residuals[0] - 2.0 * sigma)
        if len(sorted_residuals) > 1 and sigma_input_valid and sigma is not None
        else float("-inf")
    )
    if interval_fields_present and interval_receipt and sigma_input_valid and sigma is not None:
        best_upper = float(upper[best_index])
        competitor_lower = float(np.min(np.delete(lower, best_index))) if len(lower) > 1 else float("inf")
        certified_gap = competitor_lower - best_upper - 2.0 * sigma
    else:
        certified_gap = float("-inf")
    localization_radius = None
    if (
        global_error_inputs_valid
        and alpha > 0.0
        and sigma_input_valid
        and sigma is not None
        and epsilon_i is not None
        and tau_i is not None
    ):
        localization_radius = float(radius * ((lipschitz / alpha) * epsilon_i + (2.0 / alpha) * sigma + (1.0 / alpha) * tau_i))
    center = _vec4(domain.get("center"))
    domain_radius = _finite_positive(domain.get("D"))
    on_h3 = bool(all(_is_h3_point(candidate, radius=radius, tol=tol) for candidate in candidates))
    inside = bool(
        on_h3
        and center is not None
        and domain_radius is not None
        and all(_h3_distance(center, candidate, radius) <= domain_radius + tol for candidate in candidates)
    )
    net_coverage = _epsilon_net_coverage_report(
        row,
        domain=domain,
        candidates=candidates,
        epsilon=float(epsilon_i or 0.0),
        tol=tol,
    )
    error_receipt = bool(localization_radius is not None and np.isfinite(localization_radius))
    gap_receipt = bool(certified_gap > 0.0 and interval_receipt and sigma_input_valid)
    blockers: list[str] = []
    if not inside:
        blockers.append(OUTSIDE_DECLARED_DOMAIN)
    if not sigma_input_valid:
        blockers.append("missing_or_invalid_explicit_total_error_sigma")
    if not interval_receipt:
        blockers.append("missing_or_invalid_certified_residual_interval_bounds")
    if not net_coverage["receipt"]:
        blockers.append("epsilon_net_coverage_unverified")
    if not gap_receipt or not error_receipt:
        blockers.append("nonpositive_Delta_loc_or_incomplete_error_bound")
    status = (
        CERTIFIED
        if gap_receipt and error_receipt and inside and net_coverage["receipt"]
        else AMBIGUOUS
        if error_receipt and inside
        else OUTSIDE_DECLARED_DOMAIN
        if not inside
        else INCOMPLETE
    )
    return {
        "token_id": token_id,
        "terminal_status": status,
        "best_index": best_index,
        "estimated_point": candidates[best_index].tolist(),
        "residuals": residuals.tolist(),
        "residual_interval_receipt": interval_receipt,
        "residual_interval_certificate": interval_certificate,
        "diagnostic_residual_gap": diagnostic_gap,
        "residual_gap": diagnostic_gap,
        "certified_residual_gap": certified_gap,
        "Delta_loc": certified_gap,
        "sigma": sigma,
        "sigma_source": sigma_key,
        "sigma_input_valid": sigma_input_valid,
        "epsilon": epsilon_i,
        "tau": tau_i,
        "localization_radius": localization_radius,
        "support_ball": {
            "center": candidates[best_index].tolist(),
            "radius": localization_radius,
        },
        "error_receipt": error_receipt,
        "gap_receipt": gap_receipt,
        "worldline_handoff_receipt": bool(error_receipt and gap_receipt and inside and net_coverage["receipt"]),
        "inside_declared_domain": inside,
        "epsilon_net_coverage_receipt": bool(net_coverage["receipt"]),
        "epsilon_net_coverage_assumed": bool(net_coverage["assumed"]),
        "epsilon_net_coverage": net_coverage,
        "primitive_inputs_valid": primitive_inputs_valid,
        "blockers": blockers,
    }


def _blockers(
    receipts: dict[str, bool],
    *,
    frame: dict[str, Any],
    domain: dict[str, Any],
    token_rows: list[dict[str, Any]],
) -> list[str]:
    blockers = [name for name, passed in receipts.items() if not passed]
    if frame.get("rank", 0) < 4:
        blockers.append(FRAME_UNCONDITIONED)
    if not domain.get("compact", False):
        blockers.append(OUTSIDE_DECLARED_DOMAIN)
    for row in token_rows:
        blockers.extend(str(value) for value in row.get("blockers", []))
    return sorted(set(blockers))


def _terminal_status(blockers: list[str], token_rows: list[dict[str, Any]]) -> str:
    if not blockers:
        return CERTIFIED
    if any(blocker == OUTSIDE_DECLARED_DOMAIN for blocker in blockers):
        return OUTSIDE_DECLARED_DOMAIN
    if any(blocker == FRAME_UNCONDITIONED for blocker in blockers):
        return FRAME_UNCONDITIONED
    if any("POINT_SOURCE" in blocker or "POINT_SOURCE" in blocker.upper() for blocker in blockers):
        return POINT_SOURCE_UNCERTIFIED
    if any("ERROR" in blocker or "sigma" in blocker.lower() for blocker in blockers):
        return ERROR_UNBOUNDED
    if token_rows and all(row.get("error_receipt", False) for row in token_rows):
        return AMBIGUOUS
    return INCOMPLETE


def _residual_interval_certificate_report(row: dict[str, Any], *, candidate_count: int) -> dict[str, Any]:
    certificate = row.get("residual_interval_certificate")
    if not isinstance(certificate, dict):
        return {
            "certified": False,
            "certificate_id": None,
            "method": None,
            "bounds_certified": False,
            "candidate_count_matches": False,
            "reason": "missing_typed_residual_interval_certificate",
        }
    certificate_id = certificate.get("certificate_id")
    method = certificate.get("method")
    declared_count = certificate.get("candidate_count")
    id_valid = bool(isinstance(certificate_id, str) and certificate_id.strip())
    method_valid = bool(isinstance(method, str) and method in CERTIFIED_RESIDUAL_INTERVAL_METHODS)
    bounds_certified = _strict_true(certificate.get("bounds_certified", False))
    count_matches = bool(
        isinstance(declared_count, int)
        and not isinstance(declared_count, bool)
        and declared_count == candidate_count
    )
    certified = bool(id_valid and method_valid and bounds_certified and count_matches)
    return {
        "certified": certified,
        "certificate_id": certificate_id if id_valid else None,
        "method": method if isinstance(method, str) else None,
        "method_allowed": method_valid,
        "bounds_certified": bounds_certified,
        "candidate_count": declared_count,
        "candidate_count_matches": count_matches,
        "reason": None if certified else "incomplete_or_uncertified_residual_interval_method",
    }


def _kernel_receipt(source: dict[str, Any]) -> bool:
    kernel = source.get("kernel") or {}
    if not isinstance(kernel, dict):
        return False
    return bool(_kernel_type(source) is not None and _strict_true(kernel.get("declared", False)))


def _kernel_type(source: dict[str, Any]) -> str | None:
    kernel = source.get("kernel") or {}
    value = kernel if isinstance(kernel, str) else kernel.get("type")
    return str(value) if value in {"signed_linear", "paired_hinge", "identity"} else None


def _point_source_receipt(source: dict[str, Any]) -> bool:
    point = source.get("point_source") or source.get("point_source_factorization") or {}
    if isinstance(point, (bool, np.bool_)):
        return _strict_true(point)
    if not isinstance(point, dict):
        return False
    return bool(
        _strict_true(point.get("passed", False))
        and _strict_true(point.get("held_out_residual_passed", False))
        and not _strict_true(point.get("mixture_control_failed", False))
    )


def _negative_controls_receipt(source: dict[str, Any]) -> bool:
    controls = source.get("negative_controls") or {}
    if not isinstance(controls, dict) or not controls:
        return False
    return all(_strict_true(value) for value in controls.values())


def _strict_true(value: Any) -> bool:
    return bool(isinstance(value, (bool, np.bool_)) and value)


def _response_features(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    kernel_type: str | None,
) -> np.ndarray | None:
    signed = np.asarray(
        [[float(_eta(point, normal) / radius) for normal in normals] for point in points],
        dtype=float,
    )
    if kernel_type in {"signed_linear", "identity"}:
        return signed
    if kernel_type == "paired_hinge":
        return np.concatenate([np.maximum(signed, 0.0), np.maximum(-signed, 0.0)], axis=1)
    return None


def _weight_matrix_report(value: Any, size: int, *, tol: float) -> dict[str, Any]:
    if size <= 0:
        return {
            "matrix": np.zeros((0, 0), dtype=float),
            "source": "empty",
            "shape_valid": False,
            "symmetric": False,
            "minimum_eigenvalue": None,
            "maximum_eigenvalue": None,
            "valid_psd": False,
            "valid_positive_definite": False,
        }
    if value is None:
        return {
            "matrix": np.eye(size),
            "source": "implicit_identity",
            "shape_valid": True,
            "symmetric": True,
            "minimum_eigenvalue": 1.0,
            "maximum_eigenvalue": 1.0,
            "valid_psd": True,
            "valid_positive_definite": True,
        }
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        arr = np.zeros((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = np.diag(arr) if arr.shape == (size,) else np.zeros((0, 0), dtype=float)
    shape_valid = bool(arr.shape == (size, size) and np.all(np.isfinite(arr)))
    symmetric = bool(shape_valid and np.max(np.abs(arr - arr.T)) <= tol)
    minimum_eigenvalue = None
    maximum_eigenvalue = None
    valid_psd = False
    valid_positive_definite = False
    if symmetric:
        eigenvalues = np.linalg.eigvalsh(arr)
        minimum_eigenvalue = float(np.min(eigenvalues))
        maximum_eigenvalue = float(np.max(eigenvalues))
        valid_psd = bool(minimum_eigenvalue >= 0.0)
        valid_positive_definite = bool(minimum_eigenvalue > tol)
    matrix = (arr + arr.T) / 2.0 if valid_psd else np.zeros((size, size), dtype=float)
    return {
        "matrix": matrix,
        "source": "declared",
        "shape_valid": shape_valid,
        "symmetric": symmetric,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "valid_psd": valid_psd,
        "valid_positive_definite": valid_positive_definite,
    }


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)
    if np.min(vals, initial=0.0) < 0.0:
        raise ValueError("weight matrix must be positive semidefinite")
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _weighted_norm(vec: np.ndarray, weights: np.ndarray) -> float:
    quadratic = float(vec.T @ weights @ vec)
    if quadratic < -1.0e-10:
        return float("nan")
    return float(np.sqrt(max(quadratic, 0.0)))


def _response_weight_matrix(weights: np.ndarray, kernel_type: str | None) -> np.ndarray:
    if kernel_type == "paired_hinge":
        zeros = np.zeros_like(weights)
        return np.block([[weights, zeros], [zeros, weights]])
    return weights


def _epsilon_net_coverage_report(
    row: dict[str, Any],
    *,
    domain: dict[str, Any],
    candidates: np.ndarray,
    epsilon: float,
    tol: float,
) -> dict[str, Any]:
    certificate = row.get("epsilon_net_certificate") or domain.get("epsilon_net_certificate") or {}
    assumption = row.get("epsilon_net_assumption") or domain.get("epsilon_net_assumption")
    assumed = _explicit_visualization_assumption(assumption, "epsilon_net_coverage")
    if not isinstance(certificate, dict):
        certificate = {}
    declared_center = _vec4(certificate.get("domain_center"))
    domain_center = _vec4(domain.get("center"))
    declared_radius = _finite_positive(certificate.get("domain_radius"))
    domain_radius = _finite_positive(domain.get("D"))
    declared_epsilon = _finite_positive(certificate.get("epsilon"))
    covering_radius = _finite_nonnegative(certificate.get("max_covering_radius"))
    candidate_count = certificate.get("candidate_count")
    center_matches = bool(
        declared_center is not None
        and domain_center is not None
        and np.max(np.abs(declared_center - domain_center)) <= tol
    )
    radius_matches = bool(
        declared_radius is not None
        and domain_radius is not None
        and abs(declared_radius - domain_radius) <= tol
    )
    epsilon_matches = bool(declared_epsilon is not None and abs(declared_epsilon - epsilon) <= tol)
    count_matches = bool(
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count == len(candidates)
    )
    receipt = bool(
        _strict_true(certificate.get("coverage_verified", False))
        and isinstance(certificate.get("certificate_id"), str)
        and bool(certificate.get("certificate_id", "").strip())
        and isinstance(certificate.get("method"), str)
        and bool(certificate.get("method", "").strip())
        and center_matches
        and radius_matches
        and epsilon_matches
        and count_matches
        and covering_radius is not None
        and covering_radius <= epsilon + tol
    )
    return {
        "receipt": receipt,
        "assumed": assumed,
        "certificate_id": certificate.get("certificate_id"),
        "method": certificate.get("method"),
        "coverage_verified": _strict_true(certificate.get("coverage_verified", False)),
        "center_matches": center_matches,
        "radius_matches": radius_matches,
        "epsilon_matches": epsilon_matches,
        "candidate_count_matches": count_matches,
        "max_covering_radius": covering_radius,
        "explicit_visualization_assumption": assumed,
    }


def _explicit_visualization_assumption(value: Any, expected_bridge: str) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        value.get("enabled", False)
        and str(value.get("scope", "")) == "visualization_only"
        and str(value.get("bridge", "")) == expected_bridge
        and str(value.get("assumption_id", "")).strip()
    )


def _is_h3_point(point: np.ndarray, *, radius: float, tol: float) -> bool:
    return bool(point.shape == (4,) and point[0] > 0.0 and abs(_eta(point, point) + radius**2) <= max(tol, 1.0e-7))


def _h3_distance(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    argument = max(1.0, -_eta(a, b) / (float(radius) ** 2))
    return float(radius) * float(np.arccosh(argument))


def _ball_volume(r: float, radius: float) -> float:
    return float(np.pi * radius**3 * (np.sinh(2.0 * r / radius) - 2.0 * r / radius))


def _eta(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ ETA @ y)


def _points4(value: Any) -> np.ndarray:
    return _points4_checked(value)[0]


def _points4_checked(value: Any) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.zeros((0, 4), dtype=float), False
    if array.size == 0:
        return np.zeros((0, 4), dtype=float), True
    if array.shape == (4,):
        array = array.reshape((1, 4))
    if array.ndim != 2 or array.shape[1] != 4 or not np.all(np.isfinite(array)):
        return np.zeros((0, 4), dtype=float), False
    return array, True


def _vec4(value: Any) -> np.ndarray | None:
    vec = _vec(value)
    if vec is None or len(vec) != 4:
        return None
    return vec


def _vec(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0.0 else None
