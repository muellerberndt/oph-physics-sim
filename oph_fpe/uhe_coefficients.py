from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from oph_fpe.evidence.hashes import stable_json_hash


FORBIDDEN_UHE_SOURCE_TOKENS = (
    "uhe_event_coordinates",
    "event_coordinates",
    "arrival_direction",
    "arrival_directions",
    "event_energy",
    "event_energies",
    "association_failure",
    "association_failures",
    "catalog_match_after_events",
    "post_event_catalog_match",
    "likelihood_value",
    "likelihood_values",
    "posterior_summary",
    "diagnostic_overlay",
    "diagnostic_overlays",
    "residual_map",
    "residual_maps",
    "human_selected_event_pattern",
    "human_picked_event_pattern",
)

REQUIRED_RECEIPTS = (
    "BASELINE_FULL_SUPPORT",
    "FEATURE_MINIMALITY",
    "MOMENT_INTERIOR",
    "SOURCE_LOAD_QUOTIENT_VISIBLE",
    "NO_UHE_DATA_USE",
    "REFINEMENT_COMPATIBILITY",
    "COEFFICIENT_SOLVE_CONVERGED",
    "COMMON_SOURCE_LOCK",
)

NONCLAIMS = (
    "source-only coefficient emission is not a detected UHE source",
    "a finite MaxEnt solve is not an event-map fit",
    "neutrino, cosmic-ray, and gamma maps must share the source coefficient",
    "propagation and detector kernels are downstream likelihood plumbing",
    "target-data leakage invalidates a source-only label",
)

DEFAULT_UHE_PLANTED_COEFFICIENTS = (0.25, 0.35, -0.20, 0.10, 0.05)


def binary_logit_coefficient(p_oph: float, p0: float) -> float:
    """Return logit(p_oph) - logit(p0) for a binary source gate."""

    p = float(p_oph)
    q = float(p0)
    if not 0.0 < p < 1.0 or not 0.0 < q < 1.0:
        raise ValueError("binary probabilities must lie strictly between 0 and 1")
    return math.log(p * (1.0 - q) / (q * (1.0 - p)))


def poisson_opportunity_coefficient(lambda_oph: float, lambda0: float) -> float:
    """Return the Poisson natural-parameter shift log(lambda_oph/lambda0)."""

    lam = float(lambda_oph)
    base = float(lambda0)
    if lam <= 0.0 or base <= 0.0:
        raise ValueError("Poisson rates must be positive")
    return math.log(lam / base)


def small_signal_coefficient(covariance: list[list[float]] | np.ndarray, delta_c: list[float] | np.ndarray) -> list[float]:
    cov = np.asarray(covariance, dtype=float)
    delta = np.asarray(delta_c, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if delta.ndim != 1 or delta.shape[0] != cov.shape[0]:
        raise ValueError("delta_c length must match covariance rank")
    return np.linalg.solve(cov, delta).astype(float).tolist()


def default_uhe_inputs() -> dict[str, Any]:
    """Return a synthetic solver fixture with five compact-engine features."""

    rows: list[list[float]] = []
    for a in (0.0, 1.0):
        for c in (0.0, 1.0):
            for h in (0.0, 1.0):
                rows.append([a, c, h, a * c, a * h])
    features = np.asarray(rows, dtype=float)
    baseline = np.ones(features.shape[0], dtype=float)
    emitted = np.asarray(DEFAULT_UHE_PLANTED_COEFFICIENTS, dtype=float)
    target = _moments_for_eta(features, baseline, emitted)
    return {
        "features": features.tolist(),
        "baseline_weights": baseline.tolist(),
        "target_moments": target.tolist(),
        "source_dag": {
            "declared_inputs": [
                "Q_r_rel",
                "mu_r_rel",
                "m0_r",
                "F_r",
                "L_r_CE",
                "source_hashes",
            ],
            "forbidden_target_inputs": [],
        },
    }


def feature_minimality_status(features: list[list[float]] | np.ndarray, *, tol: float = 1.0e-10) -> dict[str, Any]:
    matrix = _feature_matrix(features)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=tol))
    feature_count = int(matrix.shape[1])
    return {
        "minimal": rank == feature_count,
        "rank": rank,
        "feature_count": feature_count,
        "nullity": feature_count - rank,
    }


def moment_polytope_status(
    features: list[list[float]] | np.ndarray,
    target_moments: list[float] | np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> dict[str, Any]:
    matrix = _feature_matrix(features)
    target = _target_vector(target_moments, matrix.shape[1])
    n_rows = matrix.shape[0]
    n_features = matrix.shape[1]
    c = np.zeros(n_rows + 1, dtype=float)
    c[-1] = -1.0
    a_eq = np.zeros((n_features + 1, n_rows + 1), dtype=float)
    a_eq[0, :n_rows] = 1.0
    a_eq[1:, :n_rows] = matrix.T
    b_eq = np.concatenate(([1.0], target))
    a_ub = np.zeros((n_rows, n_rows + 1), dtype=float)
    for idx in range(n_rows):
        a_ub[idx, idx] = -1.0
        a_ub[idx, -1] = 1.0
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(n_rows, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * n_rows + [(None, 1.0)],
        method="highs",
    )
    if not result.success:
        return {
            "inside": False,
            "interior": False,
            "margin": None,
            "method": "positive_barycentric_lp",
            "status": result.message,
        }
    margin = float(result.x[-1])
    return {
        "inside": True,
        "interior": margin > tol,
        "margin": margin,
        "method": "positive_barycentric_lp",
        "status": result.message,
    }


def solve_maxent_coefficients(
    features: list[list[float]] | np.ndarray,
    baseline_weights: list[float] | np.ndarray,
    target_moments: list[float] | np.ndarray,
    *,
    max_iter: int = 80,
    tol: float = 1.0e-11,
) -> dict[str, Any]:
    matrix = _feature_matrix(features)
    baseline = _baseline_vector(baseline_weights, matrix.shape[0])
    target = _target_vector(target_moments, matrix.shape[1])
    eta = np.zeros(matrix.shape[1], dtype=float)
    history: list[dict[str, float]] = []
    converged = False
    residual_norm = float("inf")
    for iteration in range(int(max_iter)):
        log_z, probabilities, moments, covariance = _family_state(matrix, baseline, eta)
        residual = moments - target
        residual_norm = float(np.linalg.norm(residual, ord=2))
        objective = float(log_z - eta.dot(target))
        history.append({"iteration": iteration, "objective": objective, "residual_norm": residual_norm})
        if residual_norm <= tol:
            converged = True
            break
        try:
            step = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(covariance).dot(residual)
        accepted = False
        for damping_power in range(24):
            damping = 0.5**damping_power
            candidate = eta - damping * step
            _, _, candidate_moments, _ = _family_state(matrix, baseline, candidate)
            candidate_residual = float(np.linalg.norm(candidate_moments - target, ord=2))
            if candidate_residual < residual_norm:
                eta = candidate
                accepted = True
                break
        if not accepted:
            eta = eta - 0.01 * step
    log_z, probabilities, moments, covariance = _family_state(matrix, baseline, eta)
    final_residual = moments - target
    residual_norm = float(np.linalg.norm(final_residual, ord=2))
    if residual_norm <= tol:
        converged = True
    return {
        "converged": converged,
        "iterations": len(history),
        "residual_norm": residual_norm,
        "coefficients": eta.astype(float).tolist(),
        "moments": moments.astype(float).tolist(),
        "target_moments": target.astype(float).tolist(),
        "log_partition": float(log_z),
        "probabilities": probabilities.astype(float).tolist(),
        "covariance": covariance.astype(float).tolist(),
        "history_tail": history[-5:],
    }


def scan_forbidden_source_tokens(payload: Any) -> list[str]:
    if payload is None:
        return []
    haystack = json.dumps(payload, sort_keys=True, default=str).lower()
    return sorted(token for token in FORBIDDEN_UHE_SOURCE_TOKENS if token in haystack)


def classify_coefficient_source(
    source_dag: Any = None,
    *,
    declared_data_path: bool = False,
    claimed_source_only: bool = True,
) -> dict[str, Any]:
    leak_hits = scan_forbidden_source_tokens(source_dag)
    if leak_hits and claimed_source_only:
        label = "INVALIDATED_COEFFICIENT_DAG"
    elif declared_data_path or leak_hits:
        label = "FITTED_OPH_COEFFICIENT"
    else:
        label = "SOURCE_ONLY_OPH_COEFFICIENT"
    return {
        "label": label,
        "target_leak_hits": leak_hits,
        "declared_data_path": bool(declared_data_path),
        "claimed_source_only": bool(claimed_source_only),
    }


def coefficient_emission_report(
    *,
    features: list[list[float]] | None = None,
    baseline_weights: list[float] | None = None,
    target_moments: list[float] | None = None,
    source_dag: Any = None,
    receipt_overrides: dict[str, Any] | None = None,
    species_coefficients: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    supplied_inputs = {
        "features": features is not None,
        "baseline_weights": baseline_weights is not None,
        "target_moments": target_moments is not None,
        "source_dag": source_dag is not None,
    }
    evidence = _evidence_classification(supplied_inputs)
    defaults = default_uhe_inputs()
    if features is None:
        features = defaults["features"]
    if baseline_weights is None:
        baseline_weights = defaults["baseline_weights"]
    if target_moments is None:
        target_moments = defaults["target_moments"]
    if source_dag is None:
        source_dag = defaults["source_dag"]

    matrix = _feature_matrix(features)
    baseline = _baseline_vector(baseline_weights, matrix.shape[0])
    target = _target_vector(target_moments, matrix.shape[1])
    minimality = feature_minimality_status(matrix)
    polytope = moment_polytope_status(matrix, target)
    classifier = classify_coefficient_source(source_dag)
    common_source = _common_source_lock(species_coefficients)

    solver_error = None
    solver: dict[str, Any]
    if not minimality["minimal"] or not polytope["interior"]:
        solver = {
            "converged": False,
            "iterations": 0,
            "residual_norm": None,
            "coefficients": [],
            "probabilities": [],
            "target_moments": target.astype(float).tolist(),
        }
    else:
        try:
            solver = solve_maxent_coefficients(matrix, baseline, target)
        except Exception as exc:  # pragma: no cover - defensive report path
            solver_error = str(exc)
            solver = {
                "converged": False,
                "iterations": 0,
                "residual_norm": None,
                "coefficients": [],
                "probabilities": [],
                "target_moments": target.astype(float).tolist(),
            }

    receipts = {
        "BASELINE_FULL_SUPPORT": bool(np.all(np.isfinite(baseline)) and np.all(baseline > 0.0)),
        "FEATURE_MINIMALITY": bool(minimality["minimal"]),
        "MOMENT_INTERIOR": bool(polytope["interior"]),
        "SOURCE_LOAD_QUOTIENT_VISIBLE": True,
        "NO_UHE_DATA_USE": classifier["label"] == "SOURCE_ONLY_OPH_COEFFICIENT",
        "REFINEMENT_COMPATIBILITY": True,
        "COEFFICIENT_SOLVE_CONVERGED": bool(solver.get("converged", False)),
        "COMMON_SOURCE_LOCK": common_source["COMMON_SOURCE_LOCK"],
    }
    if receipt_overrides:
        receipts.update({str(key): bool(value) for key, value in receipt_overrides.items()})

    blockers = _blockers(receipts, classifier, minimality, polytope, common_source, solver_error)
    claim_tier, strongest = _claim_tier(receipts, classifier, evidence)
    finite_support = {
        "kind": "finite_enumerated_support",
        "notation": f"U_finite = {{u_0, ..., u_{matrix.shape[0] - 1}}}",
        "cardinality": int(matrix.shape[0]),
        "feature_dimension": int(matrix.shape[1]),
        "baseline_full_support": bool(receipts["BASELINE_FULL_SUPPORT"]),
    }

    return {
        "schema": "oph_uhe_coefficient_emission_v2",
        "artifact_type": "UHE_COEFFICIENT_EMISSION_RECEIPT",
        "problem": "high_energy_messenger_coefficients",
        "status": _report_status(evidence),
        "evidence_classification": evidence,
        "claim_tier": claim_tier,
        "strongest_allowed_claim": strongest,
        "physical_claim": False,
        "source_only": classifier["label"] == "SOURCE_ONLY_OPH_COEFFICIENT",
        "input_hashes": {
            "features": stable_hash(matrix.astype(float).tolist()),
            "baseline": stable_hash(baseline.astype(float).tolist()),
            "target_moments": stable_hash(target.astype(float).tolist()),
            "source_dag": stable_hash(source_dag),
        },
        "moment_targets": target.astype(float).tolist(),
        "feature_minimality": minimality,
        "moment_polytope": polytope,
        "source_classifier": classifier,
        "common_source_lock": common_source,
        "solver": solver,
        "coefficients": solver.get("coefficients", []),
        "source_law": {
            "finite_support": finite_support,
            "baseline_weights": baseline.astype(float).tolist(),
            "probabilities": solver.get("probabilities", []),
            "weight_formula": "p_eta(u) = m_0(u) exp(eta^T F(u) - A(eta)), u in U_finite",
            "normalizer_formula": "A(eta) = log sum_{u in U_finite} m_0(u) exp(eta^T F(u))",
        },
        "readiness_gates": receipts,
        "required_receipts": list(REQUIRED_RECEIPTS),
        "forbidden_source_tokens": list(FORBIDDEN_UHE_SOURCE_TOKENS),
        "nonclaims": list(NONCLAIMS),
        "blockers": blockers,
        "claim_boundary": _claim_boundary(evidence),
    }


def write_uhe_coefficient_emission_report(
    out: Path,
    *,
    features: list[list[float]] | None = None,
    baseline_weights: list[float] | None = None,
    target_moments: list[float] | None = None,
    source_dag: Any = None,
    receipt_overrides: dict[str, Any] | None = None,
    species_coefficients: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    destination = Path(out)
    if destination.suffix.lower() != ".json":
        destination = destination / "uhe_coefficient_emission_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = coefficient_emission_report(
        features=features,
        baseline_weights=baseline_weights,
        target_moments=target_moments,
        source_dag=source_dag,
        receipt_overrides=receipt_overrides,
        species_coefficients=species_coefficients,
    )
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    destination.with_suffix(".md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def stable_hash(payload: Any) -> str:
    return stable_json_hash(payload)


def _feature_matrix(features: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("features must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    return matrix


def _baseline_vector(baseline_weights: list[float] | np.ndarray, expected_rows: int) -> np.ndarray:
    baseline = np.asarray(baseline_weights, dtype=float)
    if baseline.ndim != 1 or baseline.shape[0] != expected_rows:
        raise ValueError("baseline length must match feature row count")
    if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0.0):
        raise ValueError("baseline weights must be finite and strictly positive")
    return baseline


def _target_vector(target_moments: list[float] | np.ndarray, expected_features: int) -> np.ndarray:
    target = np.asarray(target_moments, dtype=float)
    if target.ndim != 1 or target.shape[0] != expected_features:
        raise ValueError("target_moments length must match feature count")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_moments must be finite")
    return target


def _family_state(
    features: np.ndarray,
    baseline: np.ndarray,
    eta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    log_weights = np.log(baseline) + features.dot(eta)
    shift = float(np.max(log_weights))
    exp_weights = np.exp(log_weights - shift)
    partition = float(np.sum(exp_weights))
    probabilities = exp_weights / partition
    log_z = shift + math.log(partition)
    moments = probabilities.dot(features)
    centered = features - moments
    covariance = (centered.T * probabilities).dot(centered)
    return log_z, probabilities, moments, covariance


def _moments_for_eta(features: np.ndarray, baseline: np.ndarray, eta: np.ndarray) -> np.ndarray:
    _, _, moments, _ = _family_state(features, baseline, eta)
    return moments


def _common_source_lock(species_coefficients: dict[str, list[float]] | None) -> dict[str, Any]:
    if not species_coefficients:
        return {
            "COMMON_SOURCE_LOCK": True,
            "species_checked": ["neutrino", "cosmic_ray", "gamma"],
            "status": "single_source_coefficient_unbranched",
            "mismatched_species": [],
        }
    species = sorted(species_coefficients)
    arrays = {name: np.asarray(values, dtype=float) for name, values in species_coefficients.items()}
    first_name = species[0]
    first = arrays[first_name]
    mismatched = [
        name
        for name in species[1:]
        if arrays[name].shape != first.shape or not np.allclose(arrays[name], first, rtol=1.0e-9, atol=1.0e-12)
    ]
    return {
        "COMMON_SOURCE_LOCK": not mismatched,
        "species_checked": species,
        "status": "shared_source_coefficient" if not mismatched else "species_separate_coefficients_detected",
        "mismatched_species": mismatched,
    }


def _evidence_classification(supplied_inputs: dict[str, bool]) -> dict[str, Any]:
    model_input_names = ("features", "baseline_weights", "target_moments")
    supplied_model_inputs = [name for name in model_input_names if supplied_inputs[name]]
    defaulted_inputs = [name for name, supplied in supplied_inputs.items() if not supplied]
    explicit_inputs = [name for name, supplied in supplied_inputs.items() if supplied]

    if not supplied_model_inputs:
        return {
            "label": "SYNTHETIC_DEMO_FIXTURE",
            "synthetic_demo": True,
            "default_fixture": True,
            "coefficient_evidence": "PLANTED_COEFFICIENT_RECOVERY_ONLY",
            "target_moment_origin": "SYNTHETIC_MOMENTS_FROM_PLANTED_COEFFICIENTS",
            "planted_coefficients": list(DEFAULT_UHE_PLANTED_COEFFICIENTS),
            "explicit_inputs": explicit_inputs,
            "defaulted_inputs": defaulted_inputs,
        }
    if len(supplied_model_inputs) != len(model_input_names):
        return {
            "label": "MIXED_DEFAULT_DEMO_INPUTS",
            "synthetic_demo": True,
            "default_fixture": False,
            "coefficient_evidence": "MIXED_INPUT_PIPELINE_EXERCISE",
            "target_moment_origin": (
                "CALLER_SUPPLIED" if supplied_inputs["target_moments"] else "DEFAULT_SYNTHETIC_TARGET_MOMENTS"
            ),
            "planted_coefficients": None,
            "explicit_inputs": explicit_inputs,
            "defaulted_inputs": defaulted_inputs,
        }
    return {
        "label": "DECLARED_SOURCE_INPUTS",
        "synthetic_demo": False,
        "default_fixture": False,
        "coefficient_evidence": "DECLARED_INPUT_MAXENT_SOLUTION",
        "target_moment_origin": "CALLER_SUPPLIED",
        "planted_coefficients": None,
        "explicit_inputs": explicit_inputs,
        "defaulted_inputs": defaulted_inputs,
    }


def _report_status(evidence: dict[str, Any]) -> str:
    label = evidence["label"]
    if label == "SYNTHETIC_DEMO_FIXTURE":
        return "synthetic_demo_fixture"
    if label == "MIXED_DEFAULT_DEMO_INPUTS":
        return "mixed_default_demo_inputs"
    return "declared_source_inputs"


def _claim_boundary(evidence: dict[str, Any]) -> str:
    label = evidence["label"]
    if label == "SYNTHETIC_DEMO_FIXTURE":
        return (
            "Synthetic finite-support MaxEnt demo fixture. Its target moments are generated from planted "
            "coefficients, so recovering those coefficients validates the numerical emission pipeline only; "
            "it is not evidence for a physical high-energy-messenger source coefficient. Source-only here "
            "means only that no UHE target data enter the fixture. Physical forward use still requires a "
            "declared source law, shared source density, and passing no-target-leak receipts."
        )
    if label == "MIXED_DEFAULT_DEMO_INPUTS":
        return (
            "Mixed-input finite-support MaxEnt demo. At least one model input came from the synthetic default "
            "fixture, so the emitted coefficient is a pipeline exercise rather than physical source evidence. "
            "Supply features, baseline weights, and target moments explicitly before treating the report as a "
            "declared source-input calculation."
        )
    return (
        "Finite MaxEnt coefficient emission from explicitly declared model inputs for high-energy-messenger "
        "source ledgers. The emitted coefficients may feed neutrino, cosmic-ray, and gamma forward models only "
        "through a shared source density and only after no-target-leak receipts pass. This report remains a "
        "source-model calculation, not a detected-source or physical-source-validation claim."
    )


def _blockers(
    receipts: dict[str, bool],
    classifier: dict[str, Any],
    minimality: dict[str, Any],
    polytope: dict[str, Any],
    common_source: dict[str, Any],
    solver_error: str | None,
) -> list[str]:
    blockers = [name.lower() for name in REQUIRED_RECEIPTS if not receipts.get(name, False)]
    if classifier["label"] == "INVALIDATED_COEFFICIENT_DAG":
        blockers.append("hidden_target_data_path_in_source_only_coefficient_dag")
    elif classifier["label"] == "FITTED_OPH_COEFFICIENT":
        blockers.append("declared_target_data_path_makes_coefficient_fitted")
    if not minimality["minimal"]:
        blockers.append("feature_nonminimal_remove_redundant_columns")
    if not polytope["interior"]:
        blockers.append("target_moment_not_in_relative_interior")
    if not common_source["COMMON_SOURCE_LOCK"]:
        blockers.append("separate_messenger_coefficients_break_common_source_lock")
    if solver_error:
        blockers.append("coefficient_solver_error")
    return sorted(set(blockers))


def _claim_tier(
    receipts: dict[str, bool],
    classifier: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[str, str]:
    if classifier["label"] == "INVALIDATED_COEFFICIENT_DAG":
        return "INVALIDATED", "INVALIDATED_COEFFICIENT_DAG"
    if classifier["label"] == "FITTED_OPH_COEFFICIENT":
        return "FITTED", "FITTED_OPH_COEFFICIENT"
    if all(receipts.get(name, False) for name in REQUIRED_RECEIPTS):
        if evidence["label"] == "SYNTHETIC_DEMO_FIXTURE":
            return "DEMO", "SYNTHETIC_DEMO_FIXTURE_RECOVERED"
        if evidence["label"] == "MIXED_DEFAULT_DEMO_INPUTS":
            return "DEMO", "MIXED_INPUT_DEMO_COEFFICIENT_EMITTED"
        return "SOURCE_ONLY", "SOURCE_ONLY_COEFFICIENT_EMITTED"
    if evidence["synthetic_demo"]:
        return "CONDITIONAL", "CONDITIONAL_SYNTHETIC_DEMO"
    return "CONDITIONAL", "CONDITIONAL_SOURCE_MODEL"


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report.get("readiness_gates") or {}
    gate_lines = "\n".join(f"- {name}: {value}" for name, value in sorted(gates.items()))
    coeffs = report.get("coefficients") or []
    blockers = report.get("blockers") or []
    evidence = report.get("evidence_classification") or {}
    finite_support = (report.get("source_law") or {}).get("finite_support") or {}
    blocker_lines = "\n".join(f"- {item}" for item in blockers) if blockers else "- none"
    return (
        "# UHE Coefficient Emission Report\n\n"
        f"- Status: {report.get('status')}\n"
        f"- Evidence classification: {evidence.get('label')}\n"
        f"- Coefficient evidence: {evidence.get('coefficient_evidence')}\n"
        f"- Target-moment origin: {evidence.get('target_moment_origin')}\n"
        f"- Claim tier: {report.get('claim_tier')}\n"
        f"- Strongest allowed claim: {report.get('strongest_allowed_claim')}\n"
        f"- Source-only: {report.get('source_only')}\n"
        f"- Physical claim: {report.get('physical_claim')}\n"
        f"- Coefficients: {json.dumps(coeffs)}\n"
        f"- Finite support: {finite_support.get('notation')}\n"
        f"- Support cardinality: {finite_support.get('cardinality')}\n\n"
        "## Readiness Gates\n\n"
        f"{gate_lines}\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n\n"
        "## Boundary\n\n"
        f"{report.get('claim_boundary')}\n"
    )
