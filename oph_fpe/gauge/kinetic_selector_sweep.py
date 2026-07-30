"""Finite source-side gauge-response selector sweep.

This module asks a deliberately narrow question.  Do positivity, locality on
one twelve-port carrier, and covariance under its 60 proper rotations select a
unique quadratic response law?

The answer is computed from a frozen grammar of response Hessians.  Each law
is constructed only from the icosahedral graph Laplacian or its canonical
constant/nonconstant split.  No measured coupling, particle mass, or
laboratory datum enters the grammar.  Sector injections are performed after
the laws have been constructed.

The resulting certificate is source-side only.  Its finite block-isotropy
check is a diagnostic for a direct isometric port-to-ideal identification, not
a continuum Ward identity.  A non-isometric current map can change that
diagnostic.  The report therefore keeps every physical promotion receipt false
even when the finite checks pass.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oph_fpe.core.echosahedral_dynamics import reference_icosahedral_coupling
from oph_fpe.core.icosahedral import (
    icosahedral_a5_equivariance_report,
    icosahedral_a5_port_permutations,
)


REPORT_SCHEMA = "oph.finite_gauge_kinetic_selector_sweep.v1"
VERIFICATION_SCHEMA = "oph.finite_gauge_kinetic_selector_sweep_verification.v1"
FINITE_SOURCE_SWEEP_RECEIPT = "FINITE_GAUGE_KINETIC_SOURCE_SWEEP_RECEIPT"
UNIQUE_SOURCE_RAY_RECEIPT = "UNIQUE_GAUGE_KINETIC_SOURCE_RAY_RECEIPT"
PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT = (
    "PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT"
)
_DEFAULT_TOLERANCE = 2.0e-10
_MAX_TOLERANCE = 1.0e-8

_BAND_ORDER = (
    "constant_singlet",
    "lowest_positive_triplet",
    "five_band",
    "highest_triplet",
)


@dataclass(frozen=True)
class _ResponseLaw:
    name: str
    construction: str
    hessian: np.ndarray
    input_grammar: str


def gauge_kinetic_selector_sweep(
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Run the frozen finite source-response grammar.

    The tested filters are intentionally weaker than a complete OPH dynamics
    theorem.  A non-unique result proves that these filters do not select a
    kinetic ray.  It does not prove that no richer source dynamics can do so.
    """

    tol = float(tolerance)
    if not math.isfinite(tol) or not 0.0 < tol <= _MAX_TOLERANCE:
        raise ValueError(
            f"tolerance must be finite and in (0, {_MAX_TOLERANCE:.0e}]"
        )

    laplacian = np.asarray(reference_icosahedral_coupling(), dtype=np.float64)
    identity = np.eye(12, dtype=np.float64)
    adjacency = 5.0 * identity - laplacian
    graph_distances = _graph_distances(adjacency)
    edge_mask = np.abs(adjacency) > 0.5
    np.fill_diagonal(edge_mask, False)
    carrier_degrees = np.count_nonzero(edge_mask, axis=1)
    carrier_edge_count = int(np.count_nonzero(np.triu(edge_mask, 1)))
    carrier_diameter = int(np.max(graph_distances))
    a5_geometry_report = icosahedral_a5_equivariance_report(0)
    projectors = _spectral_projectors(adjacency)
    projector_audit = _projector_audit(projectors, tolerance=tol)
    projector_ranks = {
        name: int(round(float(np.trace(projectors[name]))))
        for name in _BAND_ORDER
    }
    carrier_audit = {
        "vertex_count": int(laplacian.shape[0]),
        "edge_count": carrier_edge_count,
        "degree_sequence": [int(value) for value in carrier_degrees],
        "diameter": carrier_diameter,
        "spectral_band_ranks_constant_low_five_high": [
            projector_ranks[name] for name in _BAND_ORDER
        ],
        "a5_equivariance_report_schema": a5_geometry_report.get("schema"),
        "a5_base_rotation_count": a5_geometry_report.get("base_rotation_count"),
        "a5_integer_permutation_group_closed": a5_geometry_report.get(
            "integer_permutation_group_closed"
        ),
        "a5_integer_permutation_inverses_present": a5_geometry_report.get(
            "integer_permutation_inverses_present"
        ),
        "a5_faithful_base_vertex_action": a5_geometry_report.get(
            "faithful_base_vertex_action"
        ),
        "a5_rotation_group_order_60_receipt": a5_geometry_report.get(
            "A5_ROTATION_GROUP_ORDER_60_RECEIPT"
        ),
        "reference_a5_geometry_receipt": a5_geometry_report.get(
            "REFERENCE_ICOSAHEDRAL_A5_GEOMETRY_RECEIPT"
        ),
    }
    laws = _frozen_response_grammar(
        laplacian=laplacian,
        projectors=projectors,
    )
    permutation_matrices = tuple(
        _permutation_matrix(row) for row in icosahedral_a5_port_permutations()
    )

    law_rows: list[dict[str, Any]] = []
    for law in laws:
        law_rows.append(
            _audit_response_law(
                law,
                projectors=projectors,
                permutation_matrices=permutation_matrices,
                graph_distances=graph_distances,
                tolerance=tol,
            )
        )

    assignment_rows = _assignment_summary(law_rows, tolerance=tol)
    port_response_summary = _port_response_summary(
        law_rows,
        tolerance=tol,
    )
    finite_checks = {
        "twelve_port_carrier_recovered": laplacian.shape == (12, 12),
        "thirty_edge_carrier_recovered": carrier_edge_count == 30,
        "five_regular_carrier_recovered": bool(np.all(carrier_degrees == 5)),
        "diameter_three_carrier_recovered": carrier_diameter == 3,
        "sixty_proper_rotation_actions_recovered": len(permutation_matrices) == 60,
        "reference_a5_geometry_receipt_consumed": bool(
            a5_geometry_report["A5_ROTATION_GROUP_ORDER_60_RECEIPT"]
            and a5_geometry_report["REFERENCE_ICOSAHEDRAL_A5_GEOMETRY_RECEIPT"]
            and a5_geometry_report["integer_permutation_group_closed"]
            and a5_geometry_report["integer_permutation_inverses_present"]
            and a5_geometry_report["faithful_base_vertex_action"]
        ),
        "spectral_band_ranks_are_1_3_5_3": [
            projector_ranks[name] for name in _BAND_ORDER
        ]
        == [1, 3, 5, 3],
        "spectral_projectors_complete_and_orthogonal": bool(
            projector_audit["complete_and_orthogonal"]
        ),
        "all_frozen_laws_positive": all(row["positive_definite"] for row in law_rows),
        "all_frozen_laws_a5_covariant": all(
            row["a5_covariance_passes"] for row in law_rows
        ),
        "all_frozen_laws_carrier_local": all(
            row["carrier_locality"]["within_one_carrier"] for row in law_rows
        ),
        "all_sector_injections_recovered": all(
            row["sector_injection_audit"]["all_band_responses_recovered"]
            for row in law_rows
        ),
        "strict_nearest_edge_port_response_rays_are_nonunique": not bool(
            port_response_summary["strict_nearest_edge_nontrivial"][
                "unique_after_common_scale_quotient"
            ]
        ),
        "two_step_finite_block_proxy_rays_are_nonunique": all(
            row["finite_two_step_proxy_ray_count"] >= 2
            and not row["unique_two_step_proxy_ray"]
            for row in assignment_rows
        ),
        "nontrivial_two_step_finite_block_proxy_rays_are_nonunique": all(
            row["nontrivial_two_step_proxy_ray_count"] >= 2
            and not row["unique_nontrivial_two_step_proxy_ray"]
            for row in assignment_rows
        ),
        "at_least_two_finite_ward_admissible_rays": all(
            row["finite_ward_admissible_ray_count"] >= 2
            for row in assignment_rows
        ),
        "finite_ward_admissible_rays_are_nonunique": all(
            not row["unique_after_common_scale_quotient"]
            for row in assignment_rows
        ),
    }
    receipt = all(finite_checks.values())
    nonunique = bool(
        receipt
        and all(
            not row["unique_after_common_scale_quotient"]
            for row in assignment_rows
        )
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": (
            "FINITE_SOURCE_FILTERS_DO_NOT_SELECT_A_UNIQUE_PORT_RESPONSE_RAY"
            if nonunique
            else "INCONCLUSIVE_OR_INVALID_FINITE_SWEEP"
        ),
        "scope": "finite_twelve_port_source_response_only",
        "tolerance": tol,
        "source_inputs": {
            "carrier": "regular_icosahedral_12_port_graph",
            "port_count": 12,
            "edge_count": carrier_edge_count,
            "diameter": carrier_diameter,
            "proper_rotation_action_count": len(permutation_matrices),
            "measured_standard_model_couplings_used": False,
            "laboratory_data_used": False,
            "cosmology_data_used": False,
            "downstream_fit_or_target_used_to_construct_laws": False,
            "empirical_targets_used_to_engineer_equalizer_laws": False,
        },
        "carrier_audit": carrier_audit,
        "frozen_grammar": {
            "version": "finite_positive_a5_carrier_response_grammar_v1",
            "selection_filters": [
                "real_symmetric_quadratic_hessian",
                "strictly_positive_definite",
                "commutes_with_all_60_proper_icosahedral_port_actions",
                "support_confined_to_one_diameter_three_carrier",
            ],
            "law_names": [law.name for law in laws],
            "law_count": len(laws),
            "exhaustive_over_all_admissible_source_dynamics": False,
            "purpose": (
                "constructive countergrammar for the sufficiency of the stated "
                "finite filters"
            ),
        },
        "spectral_bands": {
            "order": list(_BAND_ORDER),
            "adjacency_eigenvalues": {
                "constant_singlet": "5",
                "lowest_positive_triplet": "sqrt(5)",
                "five_band": "-1",
                "highest_triplet": "-sqrt(5)",
            },
            "laplacian_costs": {
                "constant_singlet": "0",
                "lowest_positive_triplet": "5-sqrt(5)",
                "five_band": "6",
                "highest_triplet": "5+sqrt(5)",
            },
            "ranks": projector_ranks,
            "projector_audit": projector_audit,
            "triplet_assignment_boundary": (
                "The source graph orders the two Galois-conjugate triplets by "
                "Laplacian cost.  The finite A5 character match permits either "
                "opposite color/weak assignment, so both are audited."
            ),
        },
        "response_laws": law_rows,
        "port_response_nonuniqueness": port_response_summary,
        "opposite_triplet_assignment_audits": assignment_rows,
        "static_hilbert_schmidt_audits": _static_hilbert_schmidt_audits(),
        "checks": finite_checks,
        FINITE_SOURCE_SWEEP_RECEIPT: receipt,
        UNIQUE_SOURCE_RAY_RECEIPT: False,
        PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT: False,
        "result": {
            "unique_finite_port_response_ray_selected": False,
            "physical_gauge_kinetic_ray_selected": False,
            "constructive_counterexample_found": nonunique,
            "counterexample_pair": [
                "triplet_five_equalizer_for_the_audited_assignment",
                "double_triplet_five_equalizer_for_the_audited_assignment",
            ],
            "counterexample_reason": (
                "For each opposite triplet assignment, the corresponding I+Q "
                "and I+2Q radius-two triplet/five equalizers are positive, "
                "A5-covariant, and block-isotropic under the tested direct "
                "grouping.  They yield different grouped port-response rays "
                "after quotienting a common scale, without using the radius-zero "
                "onsite law.  The H=I+L and H=I+2L laws separately prove "
                "nonuniqueness for genuinely edge-coupled nearest-neighbour "
                "port responses without using that grouping."
            ),
        },
        "blockers_to_physical_prediction": [
            "no_source_law_selector_within_the_tested_admissibility_filters",
            "no_physical_port_current_to_continuum_gauge_field_identification",
            "no_derived_physical_gauge_kinetic_action",
            "no_refinement_and_renormalization_transport_for_the_measured_response",
        ],
        "claim_boundary": (
            "This is an exact-small numerical audit of a frozen, target-free "
            "finite response grammar.  Target-free means that no empirical "
            "coupling or downstream fitted value was used.  The equalizers were "
            "engineered algebraically for the declared finite block grouping, "
            "rather than emitted by source dynamics.  The sweep proves by "
            "constructive counterexample "
            "that positivity, one-carrier locality, A5 covariance, and finite "
            "sector block isotropy do not select a unique finite port-response "
            "ray in this grammar.  The finite block-isotropy test applies only "
            "to a direct isometric port-to-ideal grouping and is not a continuum "
            "Ward identity.  Differing port-band coefficients do not imply that "
            "an independently normalized ideal current trace is non-invariant. "
            "The result neither falsifies OPH nor supplies physical gauge "
            "couplings.  A richer source dynamics may select a law, but that "
            "selector and the source-to-field attachment require separate "
            "receipts."
        ),
        "hash_boundary": (
            "Matrix hashes serialize little-endian float64 runtime witnesses. "
            "The payload hash binds the complete same-runtime JSON report. "
            "Neither hash asserts cross-platform byte identity; the explicit "
            "tolerance-bounded semantic checks are the portability boundary."
        ),
    }
    report["certificate_payload_sha256"] = _payload_sha256(report)
    return report


def verify_gauge_kinetic_selector_sweep(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the report and fail closed on malformed or mutated payloads."""

    reasons: list[str] = []
    if not isinstance(report, Mapping):
        return {
            "schema": VERIFICATION_SCHEMA,
            "receipt": False,
            "reasons": ["report_is_not_a_mapping"],
        }
    if report.get("schema") != REPORT_SCHEMA:
        reasons.append("schema_mismatch")
    tolerance = report.get("tolerance")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        reasons.append("tolerance_missing_or_not_numeric")
        tolerance = _DEFAULT_TOLERANCE
    elif (
        not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance) <= _MAX_TOLERANCE
    ):
        reasons.append("tolerance_not_finite_or_out_of_bounds")
        tolerance = _DEFAULT_TOLERANCE

    stated_hash = report.get("certificate_payload_sha256")
    try:
        computed_hash = _payload_sha256(report)
    except (TypeError, ValueError, OverflowError, RecursionError):
        computed_hash = None
        reasons.append("payload_is_not_finite_canonical_json")
    if not isinstance(stated_hash, str) or stated_hash != computed_hash:
        reasons.append("payload_hash_mismatch")

    expected = gauge_kinetic_selector_sweep(tolerance=float(tolerance))
    try:
        submitted_json = _canonical_json(dict(report))
    except (TypeError, ValueError, OverflowError, RecursionError):
        submitted_json = None
    if submitted_json is None or submitted_json != _canonical_json(expected):
        reasons.append("independent_recomputation_mismatch")

    required_false = (
        UNIQUE_SOURCE_RAY_RECEIPT,
        PHYSICAL_GAUGE_KINETIC_ACTION_RECEIPT,
    )
    if any(report.get(name) is not False for name in required_false):
        reasons.append("forbidden_physical_or_uniqueness_promotion")

    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": not reasons,
        "reasons": reasons,
        "recomputed_payload_sha256": expected["certificate_payload_sha256"],
        "scope": "byte_semantic_recomputation_of_finite_source_only_report",
    }


def write_gauge_kinetic_selector_sweep(
    output_path: str | Path,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Write the deterministic source-only report and return it."""

    report = gauge_kinetic_selector_sweep(tolerance=tolerance)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def _spectral_projectors(adjacency: np.ndarray) -> dict[str, np.ndarray]:
    identity = np.eye(12, dtype=np.float64)
    sqrt5 = math.sqrt(5.0)
    eigenvalue_by_band = {
        "constant_singlet": 5.0,
        "lowest_positive_triplet": sqrt5,
        "five_band": -1.0,
        "highest_triplet": -sqrt5,
    }
    projectors: dict[str, np.ndarray] = {}
    for band, eigenvalue in eigenvalue_by_band.items():
        projector = identity.copy()
        denominator = 1.0
        for other in eigenvalue_by_band.values():
            if other == eigenvalue:
                continue
            projector = projector @ (adjacency - other * identity)
            denominator *= eigenvalue - other
        projector = projector / denominator
        projectors[band] = (projector + projector.T) / 2.0
    return projectors


def _projector_audit(
    projectors: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> dict[str, Any]:
    identity = np.eye(12, dtype=np.float64)
    sum_projectors = sum(
        (np.asarray(projectors[name]) for name in _BAND_ORDER),
        start=np.zeros((12, 12), dtype=np.float64),
    )
    idempotence = {
        name: float(np.linalg.norm(matrix @ matrix - matrix, ord="fro"))
        for name, matrix in projectors.items()
    }
    cross = {
        f"{left}__{right}": float(
            np.linalg.norm(projectors[left] @ projectors[right], ord="fro")
        )
        for left_index, left in enumerate(_BAND_ORDER)
        for right in _BAND_ORDER[left_index + 1 :]
    }
    completeness = float(np.linalg.norm(sum_projectors - identity, ord="fro"))
    maximum = max(
        [completeness, *idempotence.values(), *cross.values()],
        default=math.inf,
    )
    return {
        "idempotence_residuals": idempotence,
        "cross_orthogonality_residuals": cross,
        "completeness_residual": completeness,
        "maximum_projector_residual": maximum,
        "complete_and_orthogonal": maximum <= tolerance,
    }


def _frozen_response_grammar(
    *,
    laplacian: np.ndarray,
    projectors: Mapping[str, np.ndarray],
) -> tuple[_ResponseLaw, ...]:
    identity = np.eye(12, dtype=np.float64)
    constant = projectors["constant_singlet"]
    disagreement = identity - constant
    lowest_cost = 5.0 - math.sqrt(5.0)
    highest_cost = 5.0 + math.sqrt(5.0)
    five_cost = 6.0
    lowest_equalizer = (
        (laplacian - lowest_cost * identity)
        @ (laplacian - five_cost * identity)
    ) / 12.0
    highest_equalizer = (
        (laplacian - highest_cost * identity)
        @ (laplacian - five_cost * identity)
    ) / 12.0
    return (
        _ResponseLaw(
            name="onsite_unit",
            construction="H=I",
            hessian=identity,
            input_grammar="unit_on_site_cost",
        ),
        _ResponseLaw(
            name="edge_repair",
            construction="H=I+L",
            hessian=identity + laplacian,
            input_grammar="unit_on_site_plus_unit_edge_difference_cost",
        ),
        _ResponseLaw(
            name="double_edge_repair",
            construction="H=I+2*L",
            hessian=identity + 2.0 * laplacian,
            input_grammar="unit_on_site_plus_two_unit_edge_difference_costs",
        ),
        _ResponseLaw(
            name="squared_edge_repair",
            construction="H=I+L^2",
            hessian=identity + laplacian @ laplacian,
            input_grammar="unit_on_site_plus_two_step_repair_cost",
        ),
        _ResponseLaw(
            name="edge_and_squared_repair",
            construction="H=I+L+L^2",
            hessian=identity + laplacian + laplacian @ laplacian,
            input_grammar="unit_on_site_plus_one_and_two_step_repair_cost",
        ),
        _ResponseLaw(
            name="consensus_mode_penalty",
            construction="H=I+P_constant",
            hessian=identity + constant,
            input_grammar="unit_on_site_plus_unit_consensus_readback_cost",
        ),
        _ResponseLaw(
            name="disagreement_mode_penalty",
            construction="H=I+(I-P_constant)",
            hessian=identity + disagreement,
            input_grammar="unit_on_site_plus_unit_disagreement_readback_cost",
        ),
        _ResponseLaw(
            name="lowest_positive_band_penalty",
            construction="H=I+P_lowest_positive_triplet",
            hessian=identity + projectors["lowest_positive_triplet"],
            input_grammar="unit_on_site_plus_unit_lowest_nonconstant_band_cost",
        ),
        _ResponseLaw(
            name="lowest_triplet_five_equalizer",
            construction="Q=(L-(5-sqrt(5))*I)*(L-6*I)/12; H=I+Q",
            hessian=identity + lowest_equalizer,
            input_grammar=(
                "empirical_target_free_engineered_low_triplet_five_equalizer"
            ),
        ),
        _ResponseLaw(
            name="double_lowest_triplet_five_equalizer",
            construction="Q=(L-(5-sqrt(5))*I)*(L-6*I)/12; H=I+2*Q",
            hessian=identity + 2.0 * lowest_equalizer,
            input_grammar=(
                "empirical_target_free_engineered_double_low_triplet_five_equalizer"
            ),
        ),
        _ResponseLaw(
            name="highest_triplet_five_equalizer",
            construction="Q=(L-(5+sqrt(5))*I)*(L-6*I)/12; H=I+Q",
            hessian=identity + highest_equalizer,
            input_grammar=(
                "empirical_target_free_engineered_high_triplet_five_equalizer"
            ),
        ),
        _ResponseLaw(
            name="double_highest_triplet_five_equalizer",
            construction="Q=(L-(5+sqrt(5))*I)*(L-6*I)/12; H=I+2*Q",
            hessian=identity + 2.0 * highest_equalizer,
            input_grammar=(
                "empirical_target_free_engineered_double_high_triplet_five_equalizer"
            ),
        ),
    )


def _audit_response_law(
    law: _ResponseLaw,
    *,
    projectors: Mapping[str, np.ndarray],
    permutation_matrices: Sequence[np.ndarray],
    graph_distances: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    hessian = np.asarray(law.hessian, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(hessian)
    inverse = np.linalg.inv(hessian)
    covariance_residuals = [
        float(np.linalg.norm(matrix @ hessian - hessian @ matrix, ord="fro"))
        for matrix in permutation_matrices
    ]
    maximum_covariance = max(covariance_residuals, default=math.inf)
    nonzero = np.argwhere(np.abs(hessian) > tolerance)
    locality_radius = max(
        (int(graph_distances[left, right]) for left, right in nonzero),
        default=0,
    )

    band_rows: dict[str, dict[str, Any]] = {}
    for name in _BAND_ORDER:
        projector = np.asarray(projectors[name], dtype=np.float64)
        rank = int(round(float(np.trace(projector))))
        stiffness = float(np.trace(projector @ hessian) / rank)
        susceptibility = float(np.trace(projector @ inverse) / rank)
        reconstructed = 1.0 / susceptibility
        hessian_leakage = float(
            np.linalg.norm((np.eye(12) - projector) @ hessian @ projector, ord="fro")
        )
        response_leakage = float(
            np.linalg.norm((np.eye(12) - projector) @ inverse @ projector, ord="fro")
        )
        isotropy = float(
            np.linalg.norm(
                projector @ hessian @ projector - stiffness * projector,
                ord="fro",
            )
        )
        band_rows[name] = {
            "rank": rank,
            "injected_orthonormal_probe_count": rank,
            "mean_injected_energy_stiffness": stiffness,
            "mean_linear_susceptibility": susceptibility,
            "stiffness_reconstructed_from_susceptibility": reconstructed,
            "stiffness_reconstruction_residual": abs(reconstructed - stiffness),
            "hessian_cross_band_leakage": hessian_leakage,
            "response_cross_band_leakage": response_leakage,
            "within_band_isotropy_residual": isotropy,
            "band_response_recovered": bool(
                abs(reconstructed - stiffness) <= tolerance
                and hessian_leakage <= tolerance
                and response_leakage <= tolerance
                and isotropy <= tolerance
            ),
        }

    assignment_a = _gauge_assignment_audit(
        band_rows,
        weak_triplet="lowest_positive_triplet",
        color_triplet="highest_triplet",
        tolerance=tolerance,
    )
    assignment_b = _gauge_assignment_audit(
        band_rows,
        weak_triplet="highest_triplet",
        color_triplet="lowest_positive_triplet",
        tolerance=tolerance,
    )
    return {
        "name": law.name,
        "construction": law.construction,
        "input_grammar": law.input_grammar,
        "matrix_sha256": "sha256:"
        + hashlib.sha256(np.asarray(hessian, dtype="<f8").tobytes()).hexdigest(),
        "matrix_sha256_scope": (
            "little_endian_float64_runtime_witness_not_cross_platform_identity"
        ),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "positive_definite": bool(np.min(eigenvalues) > tolerance),
        "maximum_a5_commutator_residual": maximum_covariance,
        "a5_covariance_passes": maximum_covariance <= tolerance,
        "carrier_locality": {
            "graph_support_radius": locality_radius,
            "carrier_graph_diameter": int(np.max(graph_distances)),
            "nearest_edge_only": locality_radius <= 1,
            "within_one_carrier": locality_radius <= int(np.max(graph_distances)),
        },
        "sector_injection_audit": {
            "bands": band_rows,
            "common_scale_quotient_band_ray_constant_low_five_high": [
                _stable_float(
                    float(
                        band_rows[name]["mean_injected_energy_stiffness"]
                    )
                    / float(
                        band_rows["lowest_positive_triplet"][
                            "mean_injected_energy_stiffness"
                        ]
                    )
                )
                for name in _BAND_ORDER
            ],
            "all_band_responses_recovered": all(
                row["band_response_recovered"] for row in band_rows.values()
            ),
        },
        "gauge_block_audits": {
            "weak_is_lowest_positive_triplet": assignment_a,
            "weak_is_highest_triplet": assignment_b,
        },
    }


def _gauge_assignment_audit(
    band_rows: Mapping[str, Mapping[str, Any]],
    *,
    weak_triplet: str,
    color_triplet: str,
    tolerance: float,
) -> dict[str, Any]:
    u1 = float(band_rows["constant_singlet"]["mean_injected_energy_stiffness"])
    su2 = float(band_rows[weak_triplet]["mean_injected_energy_stiffness"])
    color_three = float(
        band_rows[color_triplet]["mean_injected_energy_stiffness"]
    )
    color_five = float(band_rows["five_band"]["mean_injected_energy_stiffness"])
    su3_average = (3.0 * color_three + 5.0 * color_five) / 8.0
    su3_isotropy_residual = abs(color_three - color_five)
    finite_ward = su3_isotropy_residual <= tolerance
    normalized = [
        _stable_float(u1 / su2),
        1.0,
        _stable_float(su3_average / su2),
    ]
    return {
        "u1_band": "constant_singlet",
        "su2_band": weak_triplet,
        "su3_bands": [color_triplet, "five_band"],
        "stiffnesses": {
            "u1": u1,
            "su2": su2,
            "su3_dimension_weighted_average": su3_average,
            "su3_triplet_component": color_three,
            "su3_five_component": color_five,
        },
        "finite_su3_block_isotropy_residual": su3_isotropy_residual,
        "finite_sector_ward_proxy_passes": finite_ward,
        "proxy_scope": (
            "direct_isometric_port_band_to_ideal_block_grouping_only; "
            "not_a_general_or_continuum_Ward_identity"
        ),
        "common_scale_quotient_u1_su2_su3": normalized,
        "ray_eligible_under_finite_proxy": finite_ward,
        "continuum_ward_identity_receipt": False,
    }


def _port_response_summary(
    law_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        reported = [
            {
                "law": row["name"],
                "common_scale_quotient_band_ray_constant_low_five_high": row[
                    "sector_injection_audit"
                ]["common_scale_quotient_band_ray_constant_low_five_high"],
            }
            for row in rows
        ]
        unique: list[list[float]] = []
        for row in reported:
            ray = [
                float(value)
                for value in row[
                    "common_scale_quotient_band_ray_constant_low_five_high"
                ]
            ]
            if not any(
                max(
                    abs(left - right)
                    for left, right in zip(ray, prior, strict=True)
                )
                <= tolerance
                for prior in unique
            ):
                unique.append(ray)
        return {
            "laws": reported,
            "distinct_ray_count": len(unique),
            "distinct_rays": unique,
            "unique_after_common_scale_quotient": len(unique) == 1,
        }

    strict_nearest_edge = [
        row for row in law_rows if row["carrier_locality"]["nearest_edge_only"]
    ]
    strict_nearest_edge_nontrivial = [
        row
        for row in law_rows
        if row["carrier_locality"]["graph_support_radius"] == 1
    ]
    return {
        "all_frozen_laws": summarize(law_rows),
        "strict_nearest_edge": summarize(strict_nearest_edge),
        "strict_nearest_edge_nontrivial": summarize(
            strict_nearest_edge_nontrivial
        ),
        "interpretation": (
            "These are four-band port-response rays.  A physical gauge kinetic "
            "interpretation requires a separately normalized current map."
        ),
    }


def _assignment_summary(
    law_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    def distinct_rays(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        unique: list[list[float]] = []
        for row in rows:
            ray = [
                float(value)
                for value in row["common_scale_quotient_u1_su2_su3"]
            ]
            if not any(
                max(
                    abs(left - right)
                    for left, right in zip(ray, prior, strict=True)
                )
                <= tolerance
                for prior in unique
            ):
                unique.append(ray)
        return unique

    summaries: list[dict[str, Any]] = []
    for assignment in (
        "weak_is_lowest_positive_triplet",
        "weak_is_highest_triplet",
    ):
        eligible: list[dict[str, Any]] = []
        for law in law_rows:
            audit = law["gauge_block_audits"][assignment]
            if not audit["finite_sector_ward_proxy_passes"]:
                continue
            eligible.append(
                {
                    "law": law["name"],
                    "graph_support_radius": law["carrier_locality"][
                        "graph_support_radius"
                    ],
                    "common_scale_quotient_u1_su2_su3": audit[
                        "common_scale_quotient_u1_su2_su3"
                    ],
                }
            )
        unique_rays = distinct_rays(eligible)
        two_step = [
            row for row in eligible if int(row["graph_support_radius"]) <= 2
        ]
        nontrivial_two_step = [
            row
            for row in two_step
            if 0 < int(row["graph_support_radius"]) <= 2
        ]
        two_step_rays = distinct_rays(two_step)
        nontrivial_two_step_rays = distinct_rays(nontrivial_two_step)
        summaries.append(
            {
                "assignment": assignment,
                "finite_ward_admissible_laws": eligible,
                "finite_ward_admissible_ray_count": len(unique_rays),
                "distinct_common_scale_quotient_rays": unique_rays,
                "unique_after_common_scale_quotient": len(unique_rays) == 1,
                "finite_two_step_proxy_laws": two_step,
                "finite_two_step_proxy_ray_count": len(two_step_rays),
                "unique_two_step_proxy_ray": len(two_step_rays) == 1,
                "nontrivial_two_step_proxy_laws": nontrivial_two_step,
                "nontrivial_two_step_proxy_ray_count": len(
                    nontrivial_two_step_rays
                ),
                "distinct_nontrivial_two_step_proxy_rays": (
                    nontrivial_two_step_rays
                ),
                "unique_nontrivial_two_step_proxy_ray": (
                    len(nontrivial_two_step_rays) == 1
                ),
            }
        )
    return summaries


def _static_hilbert_schmidt_audits() -> dict[str, Any]:
    projector_ranks = {
        "u1": 1,
        "su2": 3,
        "su3": 8,
    }
    fields = (
        ("Q", 6, Fraction(1, 6), Fraction(1), Fraction(3, 2)),
        ("u_c", 3, Fraction(-2, 3), Fraction(1, 2), Fraction(0)),
        ("d_c", 3, Fraction(1, 3), Fraction(1, 2), Fraction(0)),
        ("L", 2, Fraction(-1, 2), Fraction(0), Fraction(1, 2)),
        ("e_c", 1, Fraction(1), Fraction(0), Fraction(0)),
    )
    u1_index = sum(
        Fraction(multiplicity) * hypercharge * hypercharge
        for _, multiplicity, hypercharge, _, _ in fields
    )
    su3_index = sum(row[3] for row in fields)
    su2_index = sum(row[4] for row in fields)
    return {
        "raw_twelve_port_projector_pairing": {
            "hilbert_schmidt_squared_norms": projector_ranks,
            "per_generator_normalized_norms": {
                name: "1" for name in ("u1", "su2", "su3")
            },
            "physical_kinetic_selector": False,
            "reason": (
                "Raw block ranks and per-generator normalization are distinct "
                "conventions; neither is selected as a physical action."
            ),
        },
        "conditional_one_generation_representation_trace": {
            "field_rows": [
                {
                    "field": name,
                    "state_multiplicity": multiplicity,
                    "hypercharge": str(hypercharge),
                    "su3_quadratic_index_with_spectator_multiplicity": str(su3),
                    "su2_quadratic_index_with_spectator_multiplicity": str(su2),
                }
                for name, multiplicity, hypercharge, su3, su2 in fields
            ],
            "trace_indices_u1_su2_su3": [
                str(u1_index),
                str(su2_index),
                str(su3_index),
            ],
            "su2_normalized_ray_u1_su2_su3": [
                str(u1_index / su2_index),
                "1",
                str(su3_index / su2_index),
            ],
            "expected_exact_ray_recovered": (
                u1_index,
                su2_index,
                su3_index,
            )
            == (Fraction(10, 3), Fraction(2), Fraction(2)),
            "conditional_on_declared_one_generation_representation": True,
            "used_to_construct_or_select_response_laws": False,
            "physical_kinetic_selector": False,
            "reason": (
                "This is static representation-trace arithmetic.  It does not "
                "identify the source Hessian with the continuum gauge action."
            ),
        },
    }


def _graph_distances(adjacency: np.ndarray) -> np.ndarray:
    graph = np.abs(np.asarray(adjacency, dtype=np.float64)) > 0.5
    np.fill_diagonal(graph, False)
    count = graph.shape[0]
    distances = np.full((count, count), count + 1, dtype=np.int64)
    for source in range(count):
        distances[source, source] = 0
        frontier = [source]
        while frontier:
            vertex = frontier.pop(0)
            for neighbor in np.flatnonzero(graph[vertex]):
                neighbor = int(neighbor)
                if distances[source, neighbor] > distances[source, vertex] + 1:
                    distances[source, neighbor] = distances[source, vertex] + 1
                    frontier.append(neighbor)
    return distances


def _permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    row = np.asarray(tuple(int(value) for value in permutation), dtype=np.int64)
    matrix = np.zeros((row.size, row.size), dtype=np.float64)
    matrix[row, np.arange(row.size)] = 1.0
    return matrix


def _payload_sha256(report: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(report))
    payload.pop("certificate_payload_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_float(value: float) -> float:
    """Remove eigensolver-scale noise from reported dimensionless rays."""

    return float(round(float(value), 14))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the finite source-only gauge kinetic selector sweep."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON report to this path; stdout is used when omitted",
    )
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    report = gauge_kinetic_selector_sweep(tolerance=args.tolerance)
    verification = verify_gauge_kinetic_selector_sweep(report)
    if not verification["receipt"]:
        raise SystemExit(
            "internal verification failed: " + ",".join(verification["reasons"])
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
