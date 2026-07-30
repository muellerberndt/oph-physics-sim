"""Exact finite certificate for the canonical seam-repair branch.

The certificate separates three statements which are easy to conflate.

* On one scalar working-readback coordinate at each port of a twelve-port
  carrier, the
  uniform-trace conditional expectation onto one seam-agreement equalizer is
  unique.  It replaces the two endpoint readings by their arithmetic mean.
* Under a declared uniform, presentation-natural schedule reference on the
  transitive thirty-seam simplex, the A3 optimizer gives the expected
  one-step operator ``T = I - L/60``.  Among linear, conservative,
  nearest-seam Markov generators which are covariant under the proper carrier
  rotations, the generator ray is uniquely ``-L``; a common rate is a choice
  of time unit.
* Bare A1--A3 do not state that every repair instrument is this conditional
  expectation.  Nonlinear atomic repair, partial averaging, and larger-radius
  covariant laws remain distinct policies.  Full refinement-semigroup
  compatibility and a physical repair-law identification also remain open.

After the carrier topology and rotation action have been constructed, the
reported matrix, spectrum, energy, refinement, and classification decisions
use integers or ``fractions.Fraction``.  The rotation helper's numerical
coordinate-matching provenance is disclosed in the payload.  The entropy
conclusion is obtained from the exactly checked doubly-stochastic matrices
and the standard finite Jensen/majorization theorem; no floating entropy
comparison is promoted as evidence.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


REPORT_SCHEMA = "oph.canonical_seam_repair_certificate.v1"
VERIFICATION_SCHEMA = "oph.canonical_seam_repair_verification.v1"

CANONICAL_SEAM_EXPECTATION_RECEIPT = "CANONICAL_SEAM_EXPECTATION_RECEIPT"
CONDITIONAL_UNIFORM_SEAM_SCHEDULE_RECEIPT = (
    "CONDITIONAL_UNIFORM_SEAM_SCHEDULE_RECEIPT"
)
LAPLACIAN_REPAIR_GENERATOR_RECEIPT = "LAPLACIAN_REPAIR_GENERATOR_RECEIPT"
FINITE_REPAIR_CONVERGENCE_RECEIPT = "FINITE_REPAIR_CONVERGENCE_RECEIPT"
GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT = (
    "GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT"
)
COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT = (
    "COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT"
)
FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT = (
    "FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT"
)
FULL_REFINEMENT_SEMIGROUP_RECEIPT = "FULL_REFINEMENT_SEMIGROUP_RECEIPT"
PHYSICAL_REPAIR_LAW_RECEIPT = "PHYSICAL_REPAIR_LAW_RECEIPT"

PORT_COUNT = 12
SEAM_COUNT = 30
PORT_DEGREE = 5

FMatrix = tuple[tuple[Fraction, ...], ...]
IMatrix = tuple[tuple[int, ...], ...]
Edge = tuple[int, int]


def reference_edges() -> tuple[Edge, ...]:
    """Return the thirty unoriented seams of the base carrier."""

    level = build_geodesic_icosahedral_tower(0).levels[0]
    edges = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in level.edges
        )
    )
    if len(edges) != SEAM_COUNT or len(set(edges)) != SEAM_COUNT:
        raise ValueError("the reference carrier does not have thirty distinct seams")
    degrees = [0] * PORT_COUNT
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    if degrees != [PORT_DEGREE] * PORT_COUNT:
        raise ValueError("the reference carrier is not five-regular")
    return edges


def graph_laplacian(
    vertex_count: int,
    edges: Sequence[Edge],
) -> IMatrix:
    """Build the exact combinatorial graph Laplacian."""

    matrix = [[0] * vertex_count for _ in range(vertex_count)]
    for left, right in edges:
        if left == right or not (0 <= left < vertex_count and 0 <= right < vertex_count):
            raise ValueError("invalid graph edge")
        matrix[left][left] += 1
        matrix[right][right] += 1
        matrix[left][right] -= 1
        matrix[right][left] -= 1
    return tuple(tuple(row) for row in matrix)


def edge_conditional_expectation(edge: Edge) -> FMatrix:
    """Uniform-trace expectation onto the endpoint-agreement equalizer.

    For ``edge = {i,j}``, this is ``E_e = I - b_e b_e^T/2`` with
    ``b_e = e_i - e_j``.
    """

    left, right = edge
    if left == right or not (
        0 <= left < PORT_COUNT and 0 <= right < PORT_COUNT
    ):
        raise ValueError("a seam needs two distinct carrier ports")
    matrix = [
        [Fraction(int(row == column)) for column in range(PORT_COUNT)]
        for row in range(PORT_COUNT)
    ]
    matrix[left][left] = Fraction(1, 2)
    matrix[left][right] = Fraction(1, 2)
    matrix[right][left] = Fraction(1, 2)
    matrix[right][right] = Fraction(1, 2)
    return tuple(tuple(row) for row in matrix)


def apply_fraction_matrix(
    matrix: Sequence[Sequence[Fraction]],
    vector: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    if not matrix or len(matrix[0]) != len(vector):
        raise ValueError("matrix/vector shape mismatch")
    return tuple(
        sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction())
        for row in matrix
    )


def atomic_integer_expectation_lift(
    left_load: int,
    right_load: int,
) -> tuple[tuple[Fraction, tuple[int, int]], ...]:
    """Integral, total-preserving lift of endpoint averaging.

    Even pair totals have one exact outcome.  Odd totals have the two balanced
    integer outcomes with equal probability.  The expectation is the rational
    conditional expectation, although pathwise exact agreement is obstructed
    by parity.  This Markov kernel is not the nonlinear one-unit rule of the
    issue-628 record mechanism.
    """

    total = int(left_load) + int(right_load)
    lower = total // 2
    upper = total - lower
    if lower == upper:
        return ((Fraction(1), (lower, upper)),)
    return (
        (Fraction(1, 2), (lower, upper)),
        (Fraction(1, 2), (upper, lower)),
    )


def canonical_seam_repair_certificate() -> dict[str, Any]:
    """Recompute the bounded exact certificate and attach its payload hash."""

    payload = _certificate_payload()
    report = copy.deepcopy(payload)
    report["certificate_payload_sha256"] = _payload_sha256(payload)
    return report


def verify_canonical_seam_repair_certificate(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed by deterministic producer replay of the complete payload."""

    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        received_hash = received.pop("certificate_payload_sha256", None)
        if report.get("schema") != REPORT_SCHEMA:
            reasons.append("schema_mismatch")
        if received_hash != _payload_sha256(received):
            reasons.append("payload_hash_mismatch")

        expected = _certificate_payload()
        if received != expected:
            reasons.append("producer_replay_mismatch")

        forbidden_promotions = {
            GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT,
            COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT,
            FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT,
            FULL_REFINEMENT_SEMIGROUP_RECEIPT,
            PHYSICAL_REPAIR_LAW_RECEIPT,
        }
        if any(report.get(name) is not False for name in forbidden_promotions):
            reasons.append("forbidden_scope_promotion")
    except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("malformed_or_noncanonical_payload")

    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "claim_boundary": (
            "Verification covers the exact finite scalar working-reading "
            "certificate by replaying the producer implementation. "
            "It cannot promote global A1--A3 policy uniqueness, coupled "
            "state-generator uniqueness, full self-reading universe closure, "
            "a full refinement semigroup, or a physical repair law."
        ),
    }


def _certificate_payload() -> dict[str, Any]:
    edges = reference_edges()
    edge_set = frozenset(edges)
    laplacian = graph_laplacian(PORT_COUNT, edges)
    identity = _identity_fraction(PORT_COUNT)
    expectations = {
        edge: edge_conditional_expectation(edge)
        for edge in edges
    }

    expectation_checks = _expectation_checks(expectations)
    classification = _linear_grammar_classification()
    symmetry = _symmetry_and_schedule_checks(edges, edge_set, expectations)
    mean_operator = _mean_operator(expectations)
    expected_from_laplacian = _fraction_subtract(
        identity,
        _fraction_scale(_as_fraction_matrix(laplacian), Fraction(1, 60)),
    )
    if mean_operator != expected_from_laplacian:
        raise ValueError("uniform edge averaging is not I - L/60")

    spectrum = _spectrum_certificate(laplacian)
    energy = _energy_certificate(edges, expectations, laplacian)
    entropy = _entropy_certificate(expectation_checks)
    atomic = _atomic_lift_certificate()
    countermodels = _countermodel_certificate(edges, laplacian)
    refinement = _refinement_certificate()

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "scope": "finite_twelve_port_scalar_working_readback_repair",
        "source_inputs": {
            "carrier": "oriented_twelve_port_icosahedral_boundary_incidence",
            "port_count": PORT_COUNT,
            "seam_count": len(edges),
            "degree": PORT_DEGREE,
            "proper_rotation_count": symmetry["proper_rotation_count"],
            "rotation_action_provenance": (
                "producer helper recovers the 60 permutations by numerical "
                "coordinate matching; every resulting permutation and "
                "incidence action is checked discretely"
            ),
            "strictly_combinatorial_rotation_enumeration_certified": False,
            "laboratory_data_used": False,
            "measured_particle_or_cosmology_values_used": False,
            "fit_objective_used": False,
        },
        "declared_complete_grammar": {
            "name": "linear_scalar_seam_conditional_expectation_grammar_v1",
            "state_space": "one real scalar repairable working reading per port",
            "clauses": [
                "linear on the declared scalar working-readback coordinate",
                "one chosen seam is the complete write support",
                "the endpoint-sum functional is preserved",
                "the endpoint swap is a presentation symmetry",
                "the map is Markov: positive and unital",
                "an accepted seam repair is an idempotent retraction",
                "the retraction range is the A2 endpoint-agreement equalizer",
            ],
            "complete_only_within_declared_grammar": True,
            "not_a_complete_grammar_of_every_a1_a3_repair_instrument": True,
        },
        "conditional_expectation": {
            "formula": "E_e = I - (1/2) b_e b_e^T",
            "endpoint_action": "(x_i, x_j) -> ((x_i+x_j)/2, (x_i+x_j)/2)",
            "agreement_equalizer": "x_i = x_j",
            "uniform_trace_inner_product": "(1/12) sum_i x_i y_i",
            "edges_checked": expectation_checks["edges_checked"],
            "exact_checks": expectation_checks,
            "classification": classification,
        },
        "conditional_schedule": symmetry,
        "laplacian_generator": {
            "laplacian_definition": "L = D - A = 5 I - A",
            "uniform_expected_step": "T = (1/30) sum_e E_e = I - L/60",
            "generator_per_total_attempt": "G = T - I = -L/60",
            "poisson_rate_form": "G_nu = -(nu/60) L",
            "physical_time_scale_selected": False,
            "nearest_seam_covariant_generator_ray": "-L",
            "generator_ray_unique_up_to_positive_time_scale_in_declared_linear_grammar": True,
            "generator_orbit_classification": symmetry[
                "generator_orbit_classification"
            ],
            "exact_matrix_identity_verified": True,
            "spectrum": spectrum,
        },
        "convergence_and_conservation": {
            "quadratic_disagreement": energy,
            "entropy": entropy,
            "atomic_integer_lift": atomic,
            "almost_sure_convergence": {
                "status": "exact_finite_branch_consequence",
                "hypotheses": (
                    "independent uniform seam draws and the declared "
                    "conditional-expectation update"
                ),
                "argument": (
                    "the disagreement energy is nonincreasing pathwise and "
                    "its expectation is at most q^t times its initial value "
                    "with q=(55+sqrt(5))/60<1; its limit is therefore zero "
                    "almost surely"
                ),
                "general_implementation_liveness_claimed": False,
            },
        },
        "countermodels_and_boundaries": countermodels,
        "refinement_boundary": refinement,
        CANONICAL_SEAM_EXPECTATION_RECEIPT: True,
        CONDITIONAL_UNIFORM_SEAM_SCHEDULE_RECEIPT: True,
        LAPLACIAN_REPAIR_GENERATOR_RECEIPT: True,
        FINITE_REPAIR_CONVERGENCE_RECEIPT: True,
        GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT: False,
        COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT: False,
        FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT: False,
        FULL_REFINEMENT_SEMIGROUP_RECEIPT: False,
        PHYSICAL_REPAIR_LAW_RECEIPT: False,
        "status": (
            "UNIQUE_CONDITIONAL_EXPECTATION_BRANCH_WITH_OPEN_GLOBAL_POLICY_"
            "REFINEMENT_AND_PHYSICAL_IDENTIFICATION"
        ),
        "claim_boundary": (
            "The scalar working-readback conditional expectation and its "
            "uniform-schedule Laplacian generator are exact on one finite "
            "carrier. The uniqueness theorem is complete inside the declared "
            "linear seam-retraction grammar. A1 exposes repair instruments "
            "and A2 constrains accepted meaning. Uniform scheduling follows "
            "only under the declared presentation-natural reference on the "
            "fixed transitive move simplex. These statements do not prove "
            "that every repair policy belongs to this grammar or that the "
            "coupled state-generator closure or full self-reading model "
            "closure has one solution. The result supplies no "
            "physical clock, continuum repair semigroup, gauge kinetic action, "
            "particle parameter, or laboratory identification."
        ),
    }
    _require_no_floats(payload)
    return payload


def _expectation_checks(
    expectations: Mapping[Edge, FMatrix],
) -> dict[str, Any]:
    identity = _identity_fraction(PORT_COUNT)
    checked_entries = 0
    for edge, expectation in expectations.items():
        left, right = edge
        if _fraction_multiply(expectation, expectation) != expectation:
            raise ValueError("a seam expectation is not idempotent")
        if _fraction_transpose(expectation) != expectation:
            raise ValueError("a seam expectation is not self-adjoint")
        if any(sum(row, Fraction()) != 1 for row in expectation):
            raise ValueError("a seam expectation is not unital")
        if any(
            sum(expectation[row][column] for row in range(PORT_COUNT))
            != 1
            for column in range(PORT_COUNT)
        ):
            raise ValueError("a seam expectation does not preserve the uniform trace")
        if any(entry < 0 for row in expectation for entry in row):
            raise ValueError("a seam expectation is not positive")
        if expectation[left] != expectation[right]:
            raise ValueError("a seam expectation does not land in the equalizer")
        for port in range(PORT_COUNT):
            if port in edge:
                continue
            if expectation[port] != identity[port]:
                raise ValueError(
                    "a seam expectation changes an off-seam working reading"
                )
        incidence = [Fraction()] * PORT_COUNT
        incidence[left] = Fraction(1)
        incidence[right] = Fraction(-1)
        half_outer = tuple(
            tuple(Fraction(1, 2) * row * column for column in incidence)
            for row in incidence
        )
        if _fraction_subtract(identity, expectation) != half_outer:
            raise ValueError("the expectation is not I - b b^T/2")
        checked_entries += PORT_COUNT * PORT_COUNT
    return {
        "edges_checked": len(expectations),
        "matrix_entries_checked": checked_entries,
        "positive": True,
        "unital": True,
        "uniform_trace_preserving": True,
        "self_adjoint": True,
        "idempotent": True,
        "off_seam_identity": True,
        "range_is_endpoint_agreement_equalizer": True,
        "orthogonal_projection_identity": "I - E_e = (1/2) b_e b_e^T",
    }


def _linear_grammar_classification() -> dict[str, Any]:
    """Classify the two-by-two endpoint block with exact rational roots."""

    candidates = (Fraction(1), Fraction(1, 2))
    rows: list[dict[str, Any]] = []
    for diagonal in candidates:
        off_diagonal = 1 - diagonal
        mismatch_eigenvalue = 2 * diagonal - 1
        idempotent = mismatch_eigenvalue * mismatch_eigenvalue == mismatch_eigenvalue
        exact_agreement = mismatch_eigenvalue == 0
        rows.append(
            {
                "diagonal": str(diagonal),
                "off_diagonal": str(off_diagonal),
                "mismatch_eigenvalue": str(mismatch_eigenvalue),
                "idempotent": idempotent,
                "range_is_agreement_equalizer": exact_agreement,
                "disposition": (
                    "selected_nontrivial_retraction"
                    if exact_agreement
                    else "identity_no_repair"
                ),
            }
        )
    if [row["diagonal"] for row in rows] != ["1", "1/2"]:
        raise ValueError("the exact endpoint classification drifted")
    return {
        "general_endpoint_block": "[[a, 1-a], [1-a, a]]",
        "derivation": (
            "swap covariance and endpoint-sum preservation give the one-"
            "parameter block; idempotence makes the mismatch eigenvalue "
            "2a-1 satisfy r^2=r"
        ),
        "idempotent_solutions": rows,
        "unique_nonidentity_solution": "a=1/2",
        "unique_nonidentity_solution_is_conditional_expectation": True,
        "exact_a2_acceptance_rejects_identity": True,
    }


def _symmetry_and_schedule_checks(
    edges: Sequence[Edge],
    edge_set: frozenset[Edge],
    expectations: Mapping[Edge, FMatrix],
) -> dict[str, Any]:
    rotations = tuple(
        tuple(int(value) for value in permutation)
        for permutation in icosahedral_a5_port_permutations()
    )
    if len(rotations) != 60 or len(set(rotations)) != 60:
        raise ValueError("the proper carrier action does not have sixty elements")
    if any(sorted(permutation) != list(range(PORT_COUNT)) for permutation in rotations):
        raise ValueError("a carrier action row is not a port permutation")

    first_edge = edges[0]
    orbit = {
        tuple(sorted((permutation[first_edge[0]], permutation[first_edge[1]])))
        for permutation in rotations
    }
    if orbit != set(edges):
        raise ValueError("the proper rotations are not transitive on seams")
    vertex_orbit = {permutation[0] for permutation in rotations}
    directed_edge_orbit = {
        (permutation[first_edge[0]], permutation[first_edge[1]])
        for permutation in rotations
    }
    expected_directed_edges = {
        directed
        for left, right in edges
        for directed in ((left, right), (right, left))
    }
    if vertex_orbit != set(range(PORT_COUNT)):
        raise ValueError("the proper rotations are not transitive on ports")
    if directed_edge_orbit != expected_directed_edges:
        raise ValueError("the proper rotations are not transitive on directed seams")

    covariance_squares = 0
    for permutation in rotations:
        for edge in edges:
            mapped = tuple(
                sorted((permutation[edge[0]], permutation[edge[1]]))
            )
            if mapped not in edge_set:
                raise ValueError("a proper rotation does not preserve seams")
            transported = _transport_matrix(expectations[edge], permutation)
            if transported != expectations[mapped]:
                raise ValueError("seam expectation covariance failed")
            covariance_squares += 1

    probability = Fraction(1, SEAM_COUNT)
    if probability * len(edges) != 1:
        raise ValueError("the uniform seam schedule is not normalized")
    return {
        "move_simplex": "the thirty unoriented carrier seams",
        "reference": "uniform_on_the_fixed_move_simplex",
        "reference_status": (
            "declared presentation-natural reference on the complete "
            "thirty-seam grammar"
        ),
        "uniform_reference_derived_from_bare_a1_alone": False,
        "complete_move_simplex_required": True,
        "selected_probability_per_seam": str(probability),
        "full_support": True,
        "minimum_probability": str(probability),
        "proper_rotation_count": len(rotations),
        "port_orbit_size": len(vertex_orbit),
        "first_seam_orbit_size": len(orbit),
        "first_directed_seam_orbit_size": len(directed_edge_orbit),
        "edge_transitive": True,
        "conditional_expectation_covariance_squares_checked": covariance_squares,
        "covariance_formula": "P_g E_e P_g^-1 = E_{g(e)}",
        "schedule_status": (
            "conditional_A3_optimizer_for_declared_uniform_reference_on_"
            "fixed_transitive_move_simplex"
        ),
        "general_repair_policy_selected_by_schedule": False,
        "generator_orbit_classification": {
            "diagonal_support_orbits": 1,
            "directed_seam_support_orbits": 1,
            "off_seam_support_allowed": False,
            "general_covariant_supported_matrix": "a I + b A",
            "conservation_equation": "a + 5 b = 0",
            "markov_sign": "b >= 0",
            "conclusion": "G = -b L, unique up to the common rate b",
        },
    }


def _mean_operator(expectations: Mapping[Edge, FMatrix]) -> FMatrix:
    total = [[Fraction()] * PORT_COUNT for _ in range(PORT_COUNT)]
    for expectation in expectations.values():
        for row in range(PORT_COUNT):
            for column in range(PORT_COUNT):
                total[row][column] += expectation[row][column]
    return tuple(
        tuple(entry / len(expectations) for entry in row)
        for row in total
    )


def _spectrum_certificate(laplacian: IMatrix) -> dict[str, Any]:
    identity = _identity_int(PORT_COUNT)
    shifted_six = _int_subtract(laplacian, _int_scale(identity, 6))
    shifted_five = _int_subtract(laplacian, _int_scale(identity, 5))
    quadratic = _int_subtract(
        _int_multiply(shifted_five, shifted_five),
        _int_scale(identity, 5),
    )
    polynomial = _int_multiply(
        _int_multiply(laplacian, shifted_six),
        quadratic,
    )
    if any(entry != 0 for row in polynomial for entry in row):
        raise ValueError("the exact Laplacian polynomial identity failed")

    adjacency = _int_subtract(_int_scale(identity, PORT_DEGREE), laplacian)
    traces = [
        PORT_COUNT,
        _int_trace(adjacency),
        _int_trace(_int_power(adjacency, 2)),
        _int_trace(_int_power(adjacency, 3)),
    ]
    if traces != [12, 0, 60, 120]:
        raise ValueError("the exact adjacency trace moments drifted")

    multiplicities = (1, 3, 3, 5)
    # Values are pairs a+b*sqrt(5), ordered as 5, +sqrt(5),
    # -sqrt(5), -1.
    roots = ((5, 0), (0, 1), (0, -1), (-1, 0))
    for power, expected_trace in enumerate(traces):
        total = (Fraction(), Fraction())
        for multiplicity, root in zip(multiplicities, roots, strict=True):
            contribution = _q5_power(
                (Fraction(root[0]), Fraction(root[1])),
                power,
            )
            total = _q5_add(
                total,
                _q5_scale(contribution, Fraction(multiplicity)),
            )
        if total != (Fraction(expected_trace), Fraction()):
            raise ValueError("the exact spectral multiplicity trace system failed")
    vandermonde_product = (Fraction(1), Fraction())
    for left in range(len(roots)):
        for right in range(left + 1, len(roots)):
            difference = _q5_add(
                (Fraction(roots[right][0]), Fraction(roots[right][1])),
                _q5_scale(
                    (Fraction(roots[left][0]), Fraction(roots[left][1])),
                    Fraction(-1),
                ),
            )
            vandermonde_product = _q5_multiply(vandermonde_product, difference)
    if vandermonde_product == (Fraction(), Fraction()):
        raise ValueError("the exact trace system is not unique")

    return {
        "minimal_polynomial_identity": (
            "L (L - 6 I) (((L - 5 I)^2) - 5 I) = 0"
        ),
        "verified_entrywise_over_integers": True,
        "adjacency_trace_powers_zero_through_three": traces,
        "trace_system_vandermonde_nonzero": True,
        "laplacian_bands": [
            {"eigenvalue": "0", "multiplicity": 1},
            {"eigenvalue": "5-sqrt(5)", "multiplicity": 3},
            {"eigenvalue": "6", "multiplicity": 5},
            {"eigenvalue": "5+sqrt(5)", "multiplicity": 3},
        ],
        "nonconstant_decay_rate_ratio": [
            "5-sqrt(5)",
            "6",
            "5+sqrt(5)",
        ],
        "expected_step_eigenvalues": [
            "1",
            "(55+sqrt(5))/60",
            "9/10",
            "(55-sqrt(5))/60",
        ],
        "largest_nonconstant_expected_step_eigenvalue": "(55+sqrt(5))/60",
    }


def _energy_certificate(
    edges: Sequence[Edge],
    expectations: Mapping[Edge, FMatrix],
    laplacian: IMatrix,
) -> dict[str, Any]:
    probes_checked = 0
    for edge in edges:
        left, right = edge
        expectation = expectations[edge]
        for left_value in range(-3, 4):
            for right_value in range(-3, 4):
                vector = [Fraction()] * PORT_COUNT
                vector[left] = Fraction(left_value)
                vector[right] = Fraction(right_value)
                before = sum((value * value for value in vector), Fraction())
                after_vector = apply_fraction_matrix(expectation, vector)
                after = sum(
                    (value * value for value in after_vector),
                    Fraction(),
                )
                expected_drop = Fraction(
                    (left_value - right_value) ** 2,
                    2,
                )
                if before - after != expected_drop:
                    raise ValueError("the exact edge energy-drop identity failed")
                if sum(vector, Fraction()) != sum(after_vector, Fraction()):
                    raise ValueError("a seam repair changed the protected total")
                probes_checked += 1

    if any(sum(row) != 0 for row in laplacian):
        raise ValueError("the graph Laplacian does not preserve the total")
    return {
        "protected_conserved_functional": "sum_i x_i",
        "conserved_total_preserved": True,
        "pathwise_energy": "V(x) = sum_i (x_i - mean(x))^2",
        "one_seam_exact_drop": (
            "V(x) - V(E_e x) = (x_i - x_j)^2/2"
        ),
        "exact_integer_probe_count": probes_checked,
        "uniform_conditional_expectation": (
            "E[V(x_next) | x] = V(x) - x^T L x / 60"
        ),
        "spectral_bound": (
            "E[V(x_next) | x] <= ((55+sqrt(5))/60) V(x)"
        ),
        "strict_unless_consensus": True,
        "raw_history_preservation_note": (
            "the state map preserves the endpoint-sum functional, not every "
            "endpoint value; no append-only record or checkpoint instrument "
            "is certified by this scalar map"
        ),
        "external_record_or_checkpoint_instrument_certified": False,
    }


def _entropy_certificate(
    expectation_checks: Mapping[str, Any],
) -> dict[str, Any]:
    if not (
        expectation_checks["positive"]
        and expectation_checks["unital"]
        and expectation_checks["uniform_trace_preserving"]
    ):
        raise ValueError("the entropy theorem lacks a doubly-stochastic map")
    return {
        "domain": "nonnegative normalized scalar port weights",
        "exact_machine_premise": (
            "each E_e has nonnegative rational entries and every row and "
            "column sums exactly to one"
        ),
        "majorization": "E_e p is majorized by p",
        "shannon_entropy": "H(E_e p) >= H(p)",
        "relative_entropy_to_uniform": (
            "D(E_e p || uniform_12) <= D(p || uniform_12)"
        ),
        "strictness": "strict exactly when the two endpoint weights differ",
        "proof_rule": (
            "finite doubly-stochastic majorization plus strict concavity of "
            "-x log(x), equivalently the two-point Jensen inequality"
        ),
        "signed_load_entropy_claimed": False,
    }


def _atomic_lift_certificate() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left in range(-4, 5):
        for right in range(-4, 5):
            outcomes = atomic_integer_expectation_lift(left, right)
            if sum((weight for weight, _ in outcomes), Fraction()) != 1:
                raise ValueError("an atomic lift is not normalized")
            if any(sum(outcome) != left + right for _, outcome in outcomes):
                raise ValueError("an atomic lift changes the total signed load")
            expected_left = sum(
                (weight * outcome[0] for weight, outcome in outcomes),
                Fraction(),
            )
            expected_right = sum(
                (weight * outcome[1] for weight, outcome in outcomes),
                Fraction(),
            )
            target = Fraction(left + right, 2)
            if (expected_left, expected_right) != (target, target):
                raise ValueError("the atomic lift does not average in expectation")
            once: dict[tuple[int, int], Fraction] = {}
            twice: dict[tuple[int, int], Fraction] = {}
            for weight, outcome in outcomes:
                once[outcome] = once.get(outcome, Fraction()) + weight
                for second_weight, second_outcome in atomic_integer_expectation_lift(
                    *outcome
                ):
                    twice[second_outcome] = (
                        twice.get(second_outcome, Fraction())
                        + weight * second_weight
                    )
            if twice != once:
                raise ValueError("the atomic expectation kernel is not idempotent")
            rows.append(
                {
                    "input": [left, right],
                    "outcome_count": len(outcomes),
                    "maximum_pathwise_residual": max(
                        abs(outcome[0] - outcome[1])
                        for _, outcome in outcomes
                    ),
                }
            )
    return {
        "integer_pairs_checked": len(rows),
        "even_total": "one exactly equal integer outcome",
        "odd_total": (
            "two endpoint-swapped balanced integer outcomes with probability 1/2"
        ),
        "expectation_equals_rational_conditional_expectation": True,
        "pathwise_total_preserved": True,
        "odd_total_pathwise_exact_agreement": False,
        "odd_total_maximum_disagreement": 1,
        "range": "nearest_balanced_integer_shell",
        "exact_a2_agreement_retraction": False,
        "kernel_idempotent_in_distribution": True,
        "issue_628_relation": (
            "issue 628 uses a nonlinear one-unit transfer with a settled "
            "difference threshold; it is an alternative integer mechanism, "
            "not this rational conditional expectation"
        ),
    }


def _countermodel_certificate(
    edges: Sequence[Edge],
    laplacian: IMatrix,
) -> dict[str, Any]:
    first_edge = edges[0]
    partial = _partial_expectation(first_edge, Fraction(1, 4))
    if _fraction_multiply(partial, partial) == partial:
        raise ValueError("the partial-average control unexpectedly became idempotent")

    rotations = tuple(icosahedral_a5_port_permutations())
    moved_edge: Edge | None = None
    for permutation in rotations:
        candidate = tuple(
            sorted(
                (
                    int(permutation[first_edge[0]]),
                    int(permutation[first_edge[1]]),
                )
            )
        )
        if candidate != first_edge:
            moved_edge = candidate
            break
    if moved_edge is None:
        raise ValueError("no nontrivial seam rotation was found")
    biased_weights = {
        edge: Fraction(2, 31) if edge == first_edge else Fraction(1, 31)
        for edge in edges
    }
    if sum(biased_weights.values(), Fraction()) != 1:
        raise ValueError("the biased schedule control is not normalized")
    if biased_weights[first_edge] == biased_weights[moved_edge]:
        raise ValueError("the biased schedule does not break covariance")

    distance_two_edges = _distance_edges(PORT_COUNT, edges, distance=2)
    if len(distance_two_edges) != SEAM_COUNT:
        raise ValueError("the distance-two carrier graph does not have thirty edges")
    distance_two_laplacian = graph_laplacian(PORT_COUNT, distance_two_edges)
    if distance_two_laplacian == laplacian:
        raise ValueError("the radius-two control collapsed to the seam Laplacian")
    radius_relation = _distance_two_polynomial_check(edges, distance_two_edges)

    one_unit_at_two = _nonlinear_unit_transfer((2, 0))
    one_unit_at_four = _nonlinear_unit_transfer((4, 0))
    doubled_two = tuple(2 * value for value in one_unit_at_two)
    if one_unit_at_four == doubled_two:
        raise ValueError("the issue-628-style control unexpectedly became linear")

    return {
        "partial_average": {
            "alpha": "1/4",
            "endpoint_map": "I - alpha b_e b_e^T",
            "sum_preserving": True,
            "swap_covariant": True,
            "positive_unital": True,
            "idempotent": False,
            "one_step_exact_agreement": False,
            "uniform_generator": "-L/120",
            "lesson": (
                "without idempotence or one-step accepted agreement, a "
                "continuum of partial repairs survives; all common alpha "
                "values retain the same Laplacian ray up to time scale"
            ),
        },
        "biased_schedule": {
            "first_seam": list(first_edge),
            "rotated_seam": list(moved_edge),
            "first_probability": str(biased_weights[first_edge]),
            "rotated_probability": str(biased_weights[moved_edge]),
            "normalized": True,
            "a5_covariant": False,
            "generator": "weighted_graph_laplacian",
            "lesson": "A3 uniformity and edge transitivity are load-bearing",
        },
        "radius_two_average": {
            "distance_two_edge_count": len(distance_two_edges),
            "a5_covariant": True,
            "sum_preserving": True,
            "doubly_stochastic": True,
            "strict_seam_local": False,
            "generator": "-L_distance2/60",
            "band_action": {
                "constant": "0",
                "adjacency_plus_sqrt5_triplet": "5+sqrt(5)",
                "five_band": "6",
                "adjacency_minus_sqrt5_triplet": "5-sqrt(5)",
            },
            "exact_polynomial_relation": radius_relation,
            "lesson": (
                "A5 covariance and conservation alone permit a different "
                "generator; direct-seam support is load-bearing"
            ),
        },
        "nonlinear_atomic_unit_transfer": {
            "rule": (
                "move one integer unit from the high endpoint to the low "
                "endpoint when their difference is at least two"
            ),
            "F_2_0": list(one_unit_at_two),
            "F_4_0": list(one_unit_at_four),
            "two_times_F_2_0": list(doubled_two),
            "linear": False,
            "sum_preserving": True,
            "issue_628_style": True,
            "lesson": (
                "the certified issue-628 integer mechanism is not identified "
                "with the linear conditional-expectation branch"
            ),
        },
    }


def _refinement_certificate() -> dict[str, Any]:
    tower = build_geodesic_icosahedral_tower(2)
    rows: list[dict[str, Any]] = []
    for coarse_index in (0, 1):
        coarse = tower.levels[coarse_index]
        fine = tower.levels[coarse_index + 1]
        coarse_edge_count = len(coarse.edges)
        fine_edge_count = len(fine.edges)
        if fine_edge_count != 4 * coarse_edge_count:
            raise ValueError("midpoint refinement did not quadruple the seam count")
        coarse_laplacian = graph_laplacian(
            coarse.vertex_count,
            tuple(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in coarse.edges
            ),
        )
        fine_laplacian = graph_laplacian(
            fine.vertex_count,
            tuple(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in fine.edges
            ),
        )
        interpolation = _vertex_interpolation(fine, coarse.vertex_count)
        restriction = _inherited_vertex_restriction(
            coarse.vertex_count,
            fine.vertex_count,
        )
        if _fraction_multiply(restriction, interpolation) != _identity_fraction(
            coarse.vertex_count
        ):
            raise ValueError("refinement interpolation is not split by restriction")

        first_order_left = _fraction_multiply(
            restriction,
            _fraction_multiply(
                _as_fraction_matrix(fine_laplacian),
                interpolation,
            ),
        )
        first_order_right = _fraction_scale(
            _as_fraction_matrix(coarse_laplacian),
            Fraction(1, 2),
        )
        if first_order_left != first_order_right:
            raise ValueError("the inherited readback first-order identity failed")

        coarse_attempt_generator = _fraction_scale(
            _as_fraction_matrix(coarse_laplacian),
            Fraction(1, 2 * coarse_edge_count),
        )
        fine_attempt_generator = _fraction_scale(
            _as_fraction_matrix(fine_laplacian),
            Fraction(1, 2 * fine_edge_count),
        )
        normalized_first_left = _fraction_multiply(
            restriction,
            _fraction_multiply(fine_attempt_generator, interpolation),
        )
        normalized_first_right = _fraction_scale(
            coarse_attempt_generator,
            Fraction(1, 8),
        )
        if normalized_first_left != normalized_first_right:
            raise ValueError(
                "the normalized per-attempt first-order identity failed"
            )

        strong_left = _fraction_multiply(
            fine_attempt_generator,
            interpolation,
        )
        strong_right = _fraction_scale(
            _fraction_multiply(
                interpolation,
                coarse_attempt_generator,
            ),
            Fraction(1, 8),
        )
        strong_witness = _nonzero_residual_witness(
            _fraction_subtract(strong_left, strong_right)
        )
        if strong_witness is None:
            raise ValueError("the strong refinement control unexpectedly commuted")

        fine_attempt_squared = _fraction_multiply(
            fine_attempt_generator,
            fine_attempt_generator,
        )
        coarse_attempt_squared = _fraction_multiply(
            coarse_attempt_generator,
            coarse_attempt_generator,
        )
        second_left = _fraction_multiply(
            restriction,
            _fraction_multiply(
                fine_attempt_squared,
                interpolation,
            ),
        )
        second_right = _fraction_scale(
            coarse_attempt_squared,
            Fraction(1, 64),
        )
        second_witness = _nonzero_residual_witness(
            _fraction_subtract(second_left, second_right)
        )
        if second_witness is None:
            raise ValueError("the second-order refinement control unexpectedly commuted")

        rows.append(
            {
                "coarse_level": coarse_index,
                "fine_level": coarse_index + 1,
                "coarse_vertex_count": coarse.vertex_count,
                "fine_vertex_count": fine.vertex_count,
                "coarse_seam_count": coarse_edge_count,
                "fine_seam_count": fine_edge_count,
                "fine_to_coarse_seam_count_ratio": "4",
                "restriction_after_interpolation_is_identity": True,
                "raw_laplacian_first_order_inherited_readback": (
                    "Q Delta_f J = (1/2) Delta_c"
                ),
                "raw_laplacian_first_order_identity_exact": True,
                "per_total_attempt_generator_definition": (
                    "K_r = Delta_r / (2 |E_r|)"
                ),
                "per_total_attempt_first_order_inherited_readback": (
                    "Q K_f J = (1/8) K_c"
                ),
                "per_total_attempt_first_order_identity_exact": True,
                "per_total_attempt_strong_intertwiner": False,
                "per_total_attempt_strong_intertwiner_witness": strong_witness,
                "per_total_attempt_second_order_semigroup_condition": False,
                "per_total_attempt_second_order_witness": second_witness,
            }
        )
    return {
        "rows": rows,
        "positive_result": (
            "midpoint interpolation followed by inherited-vertex readback "
            "gives Q Delta_f J=(1/2) Delta_c for raw graph Laplacians and "
            "Q K_f J=(1/8) K_c for per-total-attempt generators"
        ),
        "negative_result": (
            "the per-total-attempt fine generator does not preserve the "
            "embedded coarse subspace, and readback fails at second order; "
            "the semigroups therefore do not commute"
        ),
        "clock_normalization_note": (
            "a per-sweep clock would absorb seam-count growth, but no "
            "cross-regulator physical clock has been derived"
        ),
        "cell_conditional_expectation_identified_with_vertex_restriction": False,
        "full_refinement_dynamics_promoted": False,
    }


def _partial_expectation(edge: Edge, alpha: Fraction) -> FMatrix:
    identity = _identity_fraction(PORT_COUNT)
    left, right = edge
    incidence = [Fraction()] * PORT_COUNT
    incidence[left] = Fraction(1)
    incidence[right] = Fraction(-1)
    outer = tuple(
        tuple(row * column for column in incidence)
        for row in incidence
    )
    return _fraction_subtract(identity, _fraction_scale(outer, alpha))


def _distance_edges(
    vertex_count: int,
    edges: Sequence[Edge],
    *,
    distance: int,
) -> tuple[Edge, ...]:
    adjacency = [set() for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    rows: list[Edge] = []
    for source in range(vertex_count):
        distances = [-1] * vertex_count
        distances[source] = 0
        frontier = [source]
        while frontier:
            vertex = frontier.pop(0)
            for neighbor in adjacency[vertex]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[vertex] + 1
                    frontier.append(neighbor)
        rows.extend(
            (source, target)
            for target in range(source + 1, vertex_count)
            if distances[target] == distance
        )
    return tuple(rows)


def _distance_two_polynomial_check(
    seam_edges: Sequence[Edge],
    distance_two_edges: Sequence[Edge],
) -> str:
    seam_laplacian = graph_laplacian(PORT_COUNT, seam_edges)
    identity = _identity_int(PORT_COUNT)
    adjacency = _int_subtract(_int_scale(identity, 5), seam_laplacian)
    distance_two_laplacian = graph_laplacian(PORT_COUNT, distance_two_edges)
    distance_two_adjacency = _int_subtract(
        _int_scale(identity, 5),
        distance_two_laplacian,
    )
    left = _int_scale(distance_two_adjacency, 2)
    right = _int_subtract(
        _int_subtract(
            _int_multiply(adjacency, adjacency),
            _int_scale(adjacency, 2),
        ),
        _int_scale(identity, 5),
    )
    if left != right:
        raise ValueError("the exact distance-two adjacency relation failed")
    return "2 A_distance2 = A^2 - 2 A - 5 I"


def _nonlinear_unit_transfer(pair: tuple[int, int]) -> tuple[int, int]:
    left, right = pair
    if left - right >= 2:
        return left - 1, right + 1
    if right - left >= 2:
        return left + 1, right - 1
    return pair


def _vertex_interpolation(
    fine_level: Any,
    coarse_vertex_count: int,
) -> FMatrix:
    rows: list[tuple[Fraction, ...]] = []
    for support in fine_level.vertex_parent_support:
        row = [Fraction()] * coarse_vertex_count
        parent_count = len(support)
        if parent_count not in {1, 2}:
            raise ValueError("unexpected vertex-parent support arity")
        value = Fraction(1, parent_count)
        for parent, _stored_weight in support:
            row[int(parent)] += value
        if sum(row, Fraction()) != 1:
            raise ValueError("a refinement interpolation row is not normalized")
        rows.append(tuple(row))
    return tuple(rows)


def _inherited_vertex_restriction(
    coarse_vertex_count: int,
    fine_vertex_count: int,
) -> FMatrix:
    return tuple(
        tuple(
            Fraction(int(row == column))
            for column in range(fine_vertex_count)
        )
        for row in range(coarse_vertex_count)
    )


def _nonzero_residual_witness(
    matrix: FMatrix,
) -> dict[str, Any] | None:
    nonzero: list[tuple[int, int, Fraction]] = []
    for row, entries in enumerate(matrix):
        for column, entry in enumerate(entries):
            if entry:
                nonzero.append((row, column, entry))
    if not nonzero:
        return None
    maximum = max(abs(entry) for _, _, entry in nonzero)
    row, column, entry = nonzero[0]
    return {
        "row": row,
        "column": column,
        "value": str(entry),
        "nonzero_entry_count": len(nonzero),
        "maximum_absolute_entry": str(maximum),
    }


def _transport_matrix(matrix: FMatrix, permutation: Sequence[int]) -> FMatrix:
    size = len(matrix)
    transported = [[Fraction()] * size for _ in range(size)]
    for old_row in range(size):
        for old_column in range(size):
            transported[permutation[old_row]][permutation[old_column]] = matrix[
                old_row
            ][old_column]
    return tuple(tuple(row) for row in transported)


def _identity_fraction(size: int) -> FMatrix:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def _identity_int(size: int) -> IMatrix:
    return tuple(
        tuple(int(row == column) for column in range(size))
        for row in range(size)
    )


def _as_fraction_matrix(matrix: Sequence[Sequence[int]]) -> FMatrix:
    return tuple(tuple(Fraction(entry) for entry in row) for row in matrix)


def _fraction_transpose(matrix: Sequence[Sequence[Fraction]]) -> FMatrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _fraction_multiply(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> FMatrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("fraction matrix shape mismatch")
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    result = [[Fraction()] * columns for _ in range(rows)]
    for row in range(rows):
        for pivot in range(inner):
            coefficient = left[row][pivot]
            if not coefficient:
                continue
            for column in range(columns):
                value = right[pivot][column]
                if value:
                    result[row][column] += coefficient * value
    return tuple(tuple(row) for row in result)


def _fraction_subtract(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> FMatrix:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    ):
        raise ValueError("fraction matrix shape mismatch")
    return tuple(
        tuple(
            left_entry - right_entry
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _fraction_scale(
    matrix: Sequence[Sequence[Fraction]],
    scalar: Fraction,
) -> FMatrix:
    return tuple(
        tuple(scalar * entry for entry in row)
        for row in matrix
    )


def _int_multiply(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> IMatrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("integer matrix shape mismatch")
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    result = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        for pivot in range(inner):
            coefficient = left[row][pivot]
            if not coefficient:
                continue
            for column in range(columns):
                value = right[pivot][column]
                if value:
                    result[row][column] += coefficient * value
    return tuple(tuple(row) for row in result)


def _int_subtract(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> IMatrix:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    ):
        raise ValueError("integer matrix shape mismatch")
    return tuple(
        tuple(
            left_entry - right_entry
            for left_entry, right_entry in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _int_scale(
    matrix: Sequence[Sequence[int]],
    scalar: int,
) -> IMatrix:
    return tuple(
        tuple(scalar * entry for entry in row)
        for row in matrix
    )


def _int_power(matrix: IMatrix, power: int) -> IMatrix:
    if power < 0:
        raise ValueError("matrix power must be nonnegative")
    result = _identity_int(len(matrix))
    for _ in range(power):
        result = _int_multiply(result, matrix)
    return result


def _int_trace(matrix: Sequence[Sequence[int]]) -> int:
    return sum(matrix[index][index] for index in range(len(matrix)))


def _q5_add(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def _q5_multiply(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return (
        left[0] * right[0] + 5 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _q5_scale(
    value: tuple[Fraction, Fraction],
    scalar: Fraction,
) -> tuple[Fraction, Fraction]:
    return scalar * value[0], scalar * value[1]


def _q5_power(
    value: tuple[Fraction, Fraction],
    power: int,
) -> tuple[Fraction, Fraction]:
    result = (Fraction(1), Fraction())
    for _ in range(power):
        result = _q5_multiply(result, value)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _require_no_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise ValueError(f"certificate payload contains a float at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_no_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_no_floats(item, f"{path}[{index}]")


def _write_report(report: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the exact finite canonical seam-repair certificate."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)

    if args.verify is not None:
        loaded = json.loads(args.verify.read_text(encoding="utf-8"))
        verification = verify_canonical_seam_repair_certificate(loaded)
        _write_report(verification, args.output)
        return 0 if verification["receipt"] else 1

    report = canonical_seam_repair_certificate()
    verification = verify_canonical_seam_repair_certificate(report)
    if not verification["receipt"]:
        _write_report(verification, args.output)
        return 1
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
