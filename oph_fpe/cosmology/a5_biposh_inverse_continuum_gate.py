"""Full-operator inverse gate for the conditional A5 BipoSH lane.

The finite equal-seam stiffness packet and its continuum-tail child establish
useful finite and blockwise facts.  They do not by themselves justify taking
an inverse after the continuum limit.  Inversion is continuous only on a
uniformly coercive set of operators.

This target-clean packet performs the strongest inexpensive check available
on the registered harmonic band:

* it bounds every raw stiffness block with ``ell,ell'=2,...,8`` and combines
  those bounds into a full-matrix Frobenius tail;
* it measures the complete level-seven raw and projected stiffness spectra;
* it applies the Neumann/coercivity admission gate without replacing a
  missing lower bound by observed spectral stability;
* it records exact counterexamples showing why selected-block convergence,
  or even full positive-matrix convergence, does not imply inverse
  convergence; and
* it separates one-copy scalar transfer cancellation from copy-space/radial
  mixing, which remains free under rotation equivariance.

The packet also records the exact operational alternative.  If a complete
equal-seam repair is source-selected, its one-tick response determines the
stiffness directly through ``L = 2|E|(I-R)``.  This avoids an unjustified
MaxEnt inverse, but it still needs a physical intervention/readout bridge.
No public comparison data are read.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.cosmology.a5_biposh_refinement import (
    ELL_MAX,
    ELL_MIN,
    LOW_MODE_DIMENSION,
    _biposh_rows,
    _biposh_summary,
    _equal_seam_graph_stiffness,
    _harmonic_design,
    _vertex_area_weights,
    _weighted_low_mode_removal,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/refinement/a5_biposh_inverse_continuum_gate_receipt.json"
)
TAIL_PARENT = ROOT / "data/refinement/a5_biposh_continuum_tail_receipt.json"
SOURCE_GATE_PARENT = (
    ROOT / "data/refinement/refined_equal_seam_source_gate_receipt.json"
)
DECLARED_LADDER_PARENT = (
    ROOT / "data/refinement/all_level_primitive_seam_source_receipt.json"
)

SCHEMA = "oph.a5-biposh-inverse-continuum-gate.v1"
STATUS = (
    "FULL_RAW_STIFFNESS_CAUCHY_TAIL_ATTAINED__UNIFORM_COERCIVITY_"
    "PROJECTED_QUOTIENT_AND_PHYSICAL_RESPONSE_OPEN"
)
ANCHOR_LEVEL = 7
EXPECTED_TAIL_STATUS = (
    "CONDITIONAL_EQUAL_SEAM_CONTINUUM_L6_NONZERO_UNDER_DECLARED_NUMERICAL_"
    "ENVELOPE__SOURCE_SELECTION_AND_PHYSICAL_TRANSFER_OPEN"
)
EXPECTED_SOURCE_GATE_STATUS = (
    "BASE_EQUAL_SEAM_GENERATOR_EXACT__REGISTERED_MESH_A5_EDGE_ORBITS_"
    "CLASSIFIED_WITH_RESIDUAL_GATE__SOURCE_COUNTING_EMITTER_OPEN"
)
EXPECTED_DECLARED_LADDER_STATUS = (
    "TARGET_CLEAN_REGISTERED_LADDER_PRIMITIVE_SEAM_ALPHABET_AND_UNIT_COUNTING_"
    "ATTAINED__EXPECTED_A2_RECONCILIATION_FIRST_ORDER_REFINEMENT_ONLY__INFINITE_"
    "TOWER_CANONICAL_DERIVATION_ATOMIC_RECORD_AND_FULL_SEMIGROUP_OPEN"
)
LEAN_BOUNDARY_DECLARATIONS = [
    "stiffness_recovered_from_uniform_repair",
    "collapsing_stiffness_positive",
    "collapsing_inverse_exact",
    "collapsing_stiffness_arbitrarily_small",
    "inverse_nuisance_unbounded",
    "copy_readout_first",
    "copy_readout_mixed",
    "copy_mixing_changes_statistic",
    "inverseTailRadius_pos",
]


class InverseContinuumGateError(ValueError):
    """Raised when a parent or an admission boundary fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_pin(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _load_parents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tail = json.loads(TAIL_PARENT.read_text(encoding="utf-8"))
    source_gate = json.loads(SOURCE_GATE_PARENT.read_text(encoding="utf-8"))
    declared_ladder = json.loads(
        DECLARED_LADDER_PARENT.read_text(encoding="utf-8")
    )
    if (
        tail.get("schema") != "oph.a5-biposh-continuum-tail.v1"
        or tail.get("status") != EXPECTED_TAIL_STATUS
        or tail.get("selection_decision", {}).get("physical_prediction") is not False
    ):
        raise InverseContinuumGateError("continuum-tail parent boundary drift")
    if (
        source_gate.get("schema")
        != "oph.refined-equal-seam-source-selection-gate.v1"
        or source_gate.get("status") != EXPECTED_SOURCE_GATE_STATUS
        or source_gate.get("selection_decision", {}).get(
            "all_level_complete_atomic_counting_law_source_emitted"
        )
        is not False
    ):
        raise InverseContinuumGateError("equal-seam source gate boundary drift")
    ladder_decision = declared_ladder.get("selection_decision", {})
    if (
        declared_ladder.get("schema")
        != "oph.registered-ladder-primitive-seam-source.v1"
        or declared_ladder.get("status") != EXPECTED_DECLARED_LADDER_STATUS
        or ladder_decision.get(
            "registered_ladder_complete_primitive_attempt_alphabet_source_emitted"
        )
        is not True
        or ladder_decision.get(
            "registered_ladder_exact_unit_counting_source_emitted_on_declared_branch"
        )
        is not True
        or ladder_decision.get(
            "infinite_tower_complete_primitive_attempt_alphabet_source_emitted"
        )
        is not False
        or ladder_decision.get("full_refinement_commuting_diagram_discharged")
        is not False
        or ladder_decision.get("physical_repair_law_selected") is not False
    ):
        raise InverseContinuumGateError("declared ladder source boundary drift")
    return tail, source_gate, declared_ladder


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    """Serialize an exact nonnegative rational and an upward binary64 view."""

    approximate = float(value)
    if Fraction.from_float(approximate) < value:
        approximate = math.nextafter(approximate, math.inf)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "binary64_upper": approximate,
    }


def _rational_edge_bound_at(level: int) -> tuple[Fraction, Fraction]:
    """Return exact rational upper bounds for ``h_level`` and its next ratio.

    The base icosahedral edge is ``acos(1/sqrt(5)) < 10/9``.  One elementary
    proof uses the alternating cosine upper polynomial at ``10/9``:

    ``1-x^2/2+x^4/24 = 8783/19683`` and its square is below ``1/5``.

    For ``0 <= h <= H < pi``, spherical midpoint refinement gives
    ``h_next <= h/(2 cos(h/2))``.  The global inequality
    ``cos x >= 1-x^2/2`` therefore gives the rational recurrence below.
    Since the bounds decrease, the returned ratio also bounds every later
    refinement ratio.
    """

    if level < 0:
        raise ValueError("level must be nonnegative")
    bound = Fraction(10, 9)
    for _ in range(level):
        ratio = 1 / (2 * (1 - bound * bound / 8))
        bound *= ratio
    ratio = 1 / (2 * (1 - bound * bound / 8))
    return bound, ratio


def full_raw_stiffness_tail(start_level: int = ANCHOR_LEVEL) -> dict[str, Any]:
    """Bound the complete raw ell=2..8 stiffness tail.

    The rational local refinement identity in the parent packet gives, for
    each parent face,

      3(M_l D_l' + D_l M_l') + 15 M_l M_l'.

    There are ``20*4^n`` parent faces at level ``n``.  The generator bounds
    scale as ``D(qh)<=qD(h)`` and ``M(qh)<=q^2M(h)``.  Therefore the two
    contributions have geometric ratios ``4q^3`` and ``4q^4``.  The result
    applies to every block and hence to the full raw 77-dimensional form.
    """

    if start_level < 0:
        raise ValueError("start_level must be nonnegative")
    edge, contraction = _rational_edge_bound_at(start_level)
    ratio_md = 4 * contraction**3
    ratio_mm = 4 * contraction**4
    if not (ratio_md < 1 and ratio_mm < 1):
        raise InverseContinuumGateError("geometric tail does not contract")

    blocks: list[dict[str, Any]] = []
    scalar_bounds: list[list[Fraction]] = [
        [Fraction(0) for _ in range(ELL_MAX - ELL_MIN + 1)]
        for _ in range(ELL_MAX - ELL_MIN + 1)
    ]
    face_count = 20 * 4**start_level
    # Addition-theorem norm c_l=sqrt((2l+1)/(4*pi)) is below 6/5 for
    # l<=8 because pi>3 and 17/12<36/25.  The rotation-generator norm is
    # l, hence sqrt(l(l+1))<l+1 is a rational upper bound.  Along the
    # midpoint geodesic, the second generator has norm at most l^2;
    # l(l+1) is the registered rational upper bound used here.
    harmonic_norm_bound = Fraction(6, 5)
    for left_index, ell in enumerate(range(ELL_MIN, ELL_MAX + 1)):
        for right_index, ell_prime in enumerate(range(ELL_MIN, ELL_MAX + 1)):
            d_left = (ell + 1) * harmonic_norm_bound * edge
            d_right = (ell_prime + 1) * harmonic_norm_bound * edge
            m_left = (
                ell * (ell + 1) * harmonic_norm_bound * edge**2 / 8
            )
            m_right = (
                ell_prime
                * (ell_prime + 1)
                * harmonic_norm_bound
                * edge**2
                / 8
            )
            first_md = face_count * 3 * (
                m_left * d_right + d_left * m_right
            )
            first_mm = face_count * 15 * m_left * m_right
            raw_tail = first_md / (1 - ratio_md) + first_mm / (
                1 - ratio_mm
            )
            scalar_bounds[left_index][right_index] = raw_tail
            blocks.append(
                {
                    "ell": ell,
                    "ell_prime": ell_prime,
                    "first_md_increment_upper_bound": _fraction_payload(first_md),
                    "first_mm_increment_upper_bound": _fraction_payload(first_mm),
                    "block_frobenius_tail_upper_bound": _fraction_payload(raw_tail),
                }
            )
    row_sums = [sum(row, Fraction(0)) for row in scalar_bounds]
    maximum_row_sum = max(row_sums)
    return {
        "start_level": start_level,
        "harmonic_band": {"ell_min": ELL_MIN, "ell_max": ELL_MAX},
        "raw_harmonic_dimension": sum(2 * ell + 1 for ell in range(2, 9)),
        "base_edge_rational_proof": {
            "angle_upper_bound": _fraction_payload(Fraction(10, 9)),
            "cosine_alternating_upper_at_angle_bound": _fraction_payload(
                Fraction(8783, 19683)
            ),
            "upper_polynomial_square": _fraction_payload(
                Fraction(8783, 19683) ** 2
            ),
            "comparison_target_one_fifth": _fraction_payload(Fraction(1, 5)),
            "upper_polynomial_square_below_one_fifth": (
                Fraction(8783, 19683) ** 2 < Fraction(1, 5)
            ),
            "angle_bound_below_pi_over_two_from_pi_gt_three": True,
            "conclusion": "acos(1/sqrt(5)) < 10/9",
        },
        "maximum_edge_upper_bound_radians": _fraction_payload(edge),
        "future_edge_contraction_upper_bound": _fraction_payload(contraction),
        "md_geometric_ratio_upper_bound": _fraction_payload(ratio_md),
        "mm_geometric_ratio_upper_bound": _fraction_payload(ratio_mm),
        "addition_theorem_vector_norm_upper_bound": "6/5 for every ell<=8",
        "difference_generator_norm_upper_bound": "(ell+1)*(6/5)*h",
        "midpoint_remainder_norm_upper_bound": (
            "ell*(ell+1)*(6/5)*h^2/8"
        ),
        "midpoint_remainder_derivation": (
            "On the centered minimizing geodesic, Y_l(s)=D_l(exp(sJ))Y_l(0). "
            "The generator norm is at most ell, so the symmetric midpoint "
            "remainder is at most h^2 sup||Y_l''||/8; ell*(ell+1) is a "
            "rational upper bound for ell^2."
        ),
        "local_refinement_bound": (
            "Each parent-face increment has Frobenius norm at most "
            "3*(M_l*D_lprime+D_l*M_lprime)+15*M_l*M_lprime. The harmonic "
            "vector norms run over every m, so no extra multiplicity factor "
            "is omitted. The 1/2 face-edge convention sums to one copy per "
            "closed-mesh edge."
        ),
        "block_rows": blocks,
        "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum": (
            _fraction_payload(maximum_row_sum)
        ),
        "operator_tail_bound_is_exact_rational_upper_arithmetic": True,
        "full_raw_stiffness_cauchy_limit_exists": True,
        "scope": (
            "Raw canonical ell=2..8 spherical-harmonic values under the ideal "
            "geodesic-midpoint equal-seam tower. The bound does not include the "
            "level-dependent weighted ell=0,1 quotient projection."
        ),
    }


@lru_cache(maxsize=1)
def _anchor_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mesh = build_geodesic_icosahedral_tower(ANCHOR_LEVEL).levels[ANCHOR_LEVEL]
    design = _harmonic_design(mesh.vertices)
    raw = design[:, LOW_MODE_DIMENSION:]
    weights = _vertex_area_weights(mesh)
    projected, removal = _weighted_low_mode_removal(design, weights)
    stiffness = _equal_seam_graph_stiffness(mesh)
    raw_form = raw.conj().T @ (stiffness @ raw)
    projected_form = projected.conj().T @ (stiffness @ projected)
    raw_form = 0.5 * (raw_form + raw_form.conj().T)
    projected_form = 0.5 * (projected_form + projected_form.conj().T)
    low = design[:, :LOW_MODE_DIMENSION]
    low_gram = low.conj().T @ (weights[:, None] * low)
    overlap = low.conj().T @ (weights[:, None] * raw)
    coefficients = np.linalg.solve(low_gram, overlap)
    diagnostics = {
        **removal,
        "low_to_active_projection_coefficient_operator_norm": float(
            np.linalg.norm(coefficients, 2)
        ),
        "low_to_active_projection_coefficient_frobenius_norm": float(
            np.linalg.norm(coefficients)
        ),
        "raw_minus_projected_stiffness_operator_norm": float(
            np.linalg.norm(raw_form - projected_form, 2)
        ),
        "raw_minus_projected_stiffness_frobenius_norm": float(
            np.linalg.norm(raw_form - projected_form)
        ),
    }
    return raw_form, projected_form, coefficients, diagnostics


def finite_anchor_report() -> dict[str, Any]:
    raw, projected, coefficients, diagnostics = _anchor_matrices()
    raw_values = np.linalg.eigvalsh(raw)
    projected_values = np.linalg.eigvalsh(projected)
    inverse = np.linalg.inv(raw)
    inverse_rows, _ = _biposh_rows(inverse)
    inverse_summary = _biposh_summary(inverse_rows)
    stiffness_rows, _ = _biposh_rows(raw)
    stiffness_summary = _biposh_summary(stiffness_rows)

    offsets: dict[int, tuple[int, int]] = {}
    start = 0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        stop = start + 2 * ell + 1
        offsets[ell] = (start, stop)
        start = stop
    per_band = []
    for ell, (left, right) in offsets.items():
        block = coefficients[:, left:right]
        per_band.append(
            {
                "ell": ell,
                "projection_coefficient_operator_norm": float(
                    np.linalg.norm(block, 2)
                ),
                "projection_coefficient_max_abs": float(np.max(np.abs(block))),
            }
        )

    return {
        "level": ANCHOR_LEVEL,
        "raw_stiffness_minimum_eigenvalue": float(raw_values[0]),
        "raw_stiffness_maximum_eigenvalue": float(raw_values[-1]),
        "raw_stiffness_positive": bool(raw_values[0] > 0.0),
        "raw_inverse_operator_norm": float(1.0 / raw_values[0]),
        "projected_stiffness_minimum_eigenvalue": float(projected_values[0]),
        "projected_stiffness_maximum_eigenvalue": float(projected_values[-1]),
        "projected_stiffness_positive": bool(projected_values[0] > 0.0),
        "raw_stiffness_primary": stiffness_summary[
            "primary_amplitude_free_statistic"
        ],
        "raw_inverse_primary": inverse_summary[
            "primary_amplitude_free_statistic"
        ],
        "projection_diagnostics": diagnostics,
        "projection_rows": per_band,
        "finite_eigensolver_is_interval_certified": False,
        "scope": (
            "Finite binary64 diagnostic. It does not supply a lower bound for "
            "the continuum limit or an interval proof of the eigensolver."
        ),
    }


def inverse_admission_report(
    tail: Mapping[str, Any], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    epsilon = float(
        tail[
            "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
        ]["binary64_upper"]
    )
    gamma_anchor = float(anchor["raw_stiffness_minimum_eigenvalue"])
    ratio = epsilon / gamma_anchor
    gate = bool(epsilon < gamma_anchor)
    return {
        "required_theorem": (
            "a source-compatible full operator K_infinity with a certified "
            "coercivity margin gamma>0 and ||K_infinity-K_n||<=epsilon<gamma"
        ),
        "anchor_level": int(anchor["level"]),
        "declared_full_raw_tail_epsilon": epsilon,
        "declared_full_raw_tail_epsilon_exact": tail[
            "full_matrix_operator_tail_upper_bound_via_symmetric_block_row_sum"
        ],
        "finite_anchor_minimum_eigenvalue_diagnostic": gamma_anchor,
        "epsilon_divided_by_finite_anchor_gap": ratio,
        "finite_anchor_neumann_gate_epsilon_lt_gap": gate,
        "finite_anchor_gap_is_continuum_coercivity_certificate": False,
        "projected_quotient_tail_certified": False,
        "full_raw_inverse_continuum_tail_certified": False,
        "projected_inverse_continuum_tail_certified": False,
        "inverse_covariance_continuum_limit_promoted": False,
        "reason": (
            "The full raw form is Cauchy, but the conservative tail is larger "
            "than the measured finite gap and the parent inverse uses a "
            "level-dependent low-mode projection. Positive finite anchors do "
            "not replace a uniform coercivity theorem."
        ),
        "advance_condition": (
            "Provide a target-clean uniform coercivity proof for the limiting "
            "raw or quotient form, plus a tail for the same level-dependent "
            "quotient construction. Then apply the standard resolvent bound "
            "||K_n^-1-K_infinity^-1|| <= epsilon/(gamma*(gamma-epsilon))."
        ),
    }


def _cotangent_graph_stiffness(mesh: Any) -> Any:
    """Return the chordal piecewise-linear cotangent stiffness diagnostic."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.vertices, dtype=float)[faces]
    cotangents = []
    for vertex in range(3):
        first = points[:, (vertex + 1) % 3] - points[:, vertex]
        second = points[:, (vertex + 2) % 3] - points[:, vertex]
        cotangents.append(
            np.sum(first * second, axis=1)
            / np.linalg.norm(np.cross(first, second), axis=1)
        )
    edge_weights: dict[tuple[int, int], float] = {}
    for opposite, (left_slot, right_slot) in enumerate(
        ((1, 2), (2, 0), (0, 1))
    ):
        for left, right, weight in zip(
            faces[:, left_slot],
            faces[:, right_slot],
            0.5 * cotangents[opposite],
            strict=True,
        ):
            edge = (min(int(left), int(right)), max(int(left), int(right)))
            edge_weights[edge] = edge_weights.get(edge, 0.0) + float(weight)
    diagonal = np.zeros(mesh.vertex_count, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), weight in edge_weights.items():
        diagonal[left] += weight
        diagonal[right] += weight
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((-weight, -weight))
    rows.extend(range(mesh.vertex_count))
    columns.extend(range(mesh.vertex_count))
    values.extend(float(value) for value in diagonal)
    return coo_matrix(
        (values, (rows, columns)),
        shape=(mesh.vertex_count, mesh.vertex_count),
    ).tocsr()


def continuum_geometry_assessment(parent_tail: Mapping[str, Any]) -> dict[str, Any]:
    """Separate the viable coercivity route from the isotropic FEM limit."""

    tower = build_geodesic_icosahedral_tower(ANCHOR_LEVEL)
    shape_rows = []
    for mesh in tower.levels:
        points = np.asarray(mesh.vertices, dtype=float)[mesh.faces]
        a = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
        b = np.linalg.norm(points[:, 2] - points[:, 0], axis=1)
        c = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
        angles = np.stack(
            (
                np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)),
                np.arccos(np.clip((c * c + a * a - b * b) / (2 * c * a), -1, 1)),
            ),
            axis=1,
        )
        third = math.pi - angles[:, 0] - angles[:, 1]
        angles = np.column_stack((angles, third))
        shape_rows.append(
            {
                "level": mesh.level,
                "minimum_chord_triangle_angle_degrees": float(
                    np.degrees(np.min(angles))
                ),
                "maximum_chord_triangle_angle_degrees": float(
                    np.degrees(np.max(angles))
                ),
                "maximum_to_minimum_chord_edge_ratio": float(
                    np.max(np.maximum.reduce((a, b, c)) / np.minimum.reduce((a, b, c)))
                ),
            }
        )

    comparison_rows = []
    for level in range(2, 7):
        mesh = tower.levels[level]
        active = _harmonic_design(mesh.vertices)[:, LOW_MODE_DIMENSION:]
        equal_form = active.conj().T @ (
            _equal_seam_graph_stiffness(mesh) @ active
        )
        cotangent_form = active.conj().T @ (
            _cotangent_graph_stiffness(mesh) @ active
        )
        equal_rows, _ = _biposh_rows(equal_form)
        cotangent_rows, _ = _biposh_rows(cotangent_form)
        comparison_rows.append(
            {
                "level": level,
                "equal_counting_primary_statistic": _biposh_summary(equal_rows)[
                    "primary_amplitude_free_statistic"
                ],
                "cotangent_fem_primary_statistic": _biposh_summary(cotangent_rows)[
                    "primary_amplitude_free_statistic"
                ],
            }
        )

    interval = parent_tail["conditional_continuum_interval"]
    return {
        "finite_shape_regular_diagnostic": {
            "rows": shape_rows,
            "all_level_uniform_shape_regularity_proved": False,
            "binary64_rows_are_a_theorem": False,
        },
        "equal_counting_vs_cotangent_diagnostic": {
            "rows": comparison_rows,
            "equal_counting_is_the_declared_repair_measure": True,
            "cotangent_weights_are_the_declared_repair_measure": False,
            "interpretation": (
                "Cotangent weights approximate the SO(3)-invariant round-sphere "
                "Dirichlet form and suppress the cross-band statistic. Unit edge "
                "counting follows the nested A5 mesh parameterization and need only "
                "be A5 invariant, so an L=6 component is compatible with the limit."
            ),
            "finite_binary64_contrast_is_a_continuum_proof": False,
        },
        "parent_equal_counting_continuum_interval": {
            "primary_amplitude_free_statistic_interval": interval[
                "primary_amplitude_free_statistic_interval"
            ],
            "conditional_interval_excludes_zero": interval[
                "conditional_interval_excludes_zero"
            ],
            "scope": (
                "Conditional equal-counting limit under the parent packet's "
                "declared finite-anchor harmonic evaluation envelope."
            ),
        },
        "shape_regular_coercivity_route": {
            "candidate_route_identified": True,
            "not_excluded_by_current_results": True,
            "closed_here": False,
            "missing_uniform_geometry_theorems": [
                "an explicit all-level angle or bilipschitz bound for nested spherical midpoint refinement",
                "uniform spectral equivalence between unit-edge graph energy and a fixed parameter-domain finite-element energy",
                "an H1-stable band-limited sampling/interpolation lower bound",
            ],
            "precise_next_theorem": (
                "There exist n0 and c_sr>0 such that for every n>=n0 and every "
                "coefficient vector a in the raw ell=2..8 band, a* K_n a >= "
                "c_sr sum_{ell,m} ell(ell+1)|a_{ell m}|^2."
            ),
            "consequence_if_proved": (
                "The raw limit is uniformly coercive and finite-dimensional "
                "inverse convergence follows independently of the loose explicit tail."
            ),
            "does_not_force_so3_isotropy_or_zero_l6": True,
        },
    }


def exact_counterexamples() -> dict[str, Any]:
    sequence = []
    for n in range(1, 9):
        sequence.append(
            {
                "n": n,
                "K_diagonal": [1.0, 1.0 / (n + 1)],
                "K_inverse_diagonal": [1.0, float(n + 1)],
                "selected_first_block": 1.0,
                "minimum_eigenvalue": 1.0 / (n + 1),
            }
        )
    return {
        "inverse_discontinuity_without_uniform_coercivity": {
            "family": "K_n=diag(1,1/(n+1))",
            "every_finite_K_n_positive_definite": True,
            "full_matrix_limit_exists": True,
            "limit": "diag(1,0)",
            "selected_first_block_constant": True,
            "inverse_nuisance_eigenvalue": "n+1",
            "inverse_family_bounded": False,
            "rows": sequence,
            "conclusion": (
                "Selected-block convergence and finite positivity do not imply "
                "a continuum inverse. A uniform lower spectral bound is necessary."
            ),
        },
        "multiplicity_space_radial_mixing": {
            "copy_space_diagonal_blocks": "A=C=I_2",
            "copy_space_cross_block": "B=diag(1,0)",
            "first_readout_vectors": {"u": [1, 0], "v": [1, 0]},
            "first_statistic": 1.0,
            "mixed_readout_vectors": {"u": [1, 1], "v": [1, 1]},
            "mixed_statistic": 0.5,
            "both_copy_space_readouts_commute_with_spatial_rotations": True,
            "rotation_equivariance_forces_scalar_on_copy_space": False,
            "conclusion": (
                "Schur cancellation is automatic only for one copy of each "
                "irreducible band. Radial or copy multiplicity admits mixing "
                "that changes the frozen statistic."
            ),
        },
        "one_copy_scalar_rescaling": {
            "transformation": (
                "N maps to |u2*u4|N, D2 maps to u2^2 D2, "
                "D4 maps to u4^2 D4"
            ),
            "positive_nonzero_scalar_gains_cancel": True,
            "proves_rotation_equivariant_multiplicity_one_transfer": False,
            "proves_absence_of_radial_mixing": False,
        },
    }


def operational_response_report(
    source_gate: Mapping[str, Any], declared_ladder: Mapping[str, Any]
) -> dict[str, Any]:
    base_selected = source_gate["selection_decision"][
        "base_equal_seam_operator_selected_in_bounded_realization"
    ]
    ladder_decision = declared_ladder["selection_decision"]
    registered_levels = declared_ladder["source_scope"][
        "registered_geometry_levels"
    ]
    return {
        "finite_identity": "R = I-L/(2*edge_count)",
        "readback_identity": "L = 2*edge_count*(I-R)",
        "interpretation": (
            "Inject a declared scalar perturbation, measure the one-completed-"
            "tick expected response R, and reconstruct the stiffness bilinear "
            "form without introducing a stochastic covariance or MaxEnt inverse."
        ),
        "base_carrier_response_selected_in_bounded_realization": base_selected,
        "declared_registered_ladder_primitive_alphabet_source_emitted": (
            ladder_decision[
                "registered_ladder_complete_primitive_attempt_alphabet_source_emitted"
            ]
        ),
        "declared_registered_ladder_unit_counting_source_emitted": (
            ladder_decision[
                "registered_ladder_exact_unit_counting_source_emitted_on_declared_branch"
            ]
        ),
        "declared_registered_geometry_levels": registered_levels,
        "inverse_anchor_level": ANCHOR_LEVEL,
        "declared_ladder_reaches_inverse_anchor_level": (
            ANCHOR_LEVEL in registered_levels
        ),
        "first_order_refinement_readback_discharged": ladder_decision[
            "first_order_refinement_readback_discharged"
        ],
        "full_refinement_commuting_diagram_discharged": ladder_decision[
            "full_refinement_commuting_diagram_discharged"
        ],
        "physical_repair_law_selected": ladder_decision[
            "physical_repair_law_selected"
        ],
        "all_level_response_law_source_selected": False,
        "intervention_family_physically_available": False,
        "response_readout_physically_attached": False,
        "screen_to_sky_map_required_for_direct_response_measurement": False,
        "laboratory_or_cosmological_response_attachment_required": True,
        "stiffness_statistic_can_be_an_operational_response_observable": True,
        "stiffness_statistic_is_a_current_physical_prediction": False,
        "recommendation": (
            "Use stiffness as the primary operational target unless a source "
            "ensemble and uniform inverse theorem are independently supplied. "
            "Do not call the stiffness fingerprint a covariance."
        ),
    }


def build_inverse_continuum_gate_packet() -> dict[str, Any]:
    parent_tail, source_gate, declared_ladder = _load_parents()
    full_tail = full_raw_stiffness_tail()
    anchor = finite_anchor_report()
    admission = inverse_admission_report(full_tail, anchor)
    geometry_assessment = continuum_geometry_assessment(parent_tail)
    producer = Path(__file__).resolve()
    verifier = producer.with_name(
        "verify_a5_biposh_inverse_continuum_gate_independent.py"
    )
    tests = ROOT / "tests/test_a5_biposh_inverse_continuum_gate.py"
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": 659,
        "status": STATUS,
        "source_scope": {
            "geometry": "registered ideal geodesic-midpoint icosahedral tower",
            "harmonic_band": {"ell_min": ELL_MIN, "ell_max": ELL_MAX},
            "external_comparison_data_used": False,
            "sky_data_used": False,
            "target_values_used": False,
            "stochastic_ensemble_used": False,
        },
        "parents": {
            "continuum_tail": {
                "path": TAIL_PARENT.relative_to(ROOT).as_posix(),
                "schema": parent_tail["schema"],
                "status": parent_tail["status"],
                "payload_sha256": parent_tail["payload_sha256"],
            },
            "equal_seam_source_gate": {
                "path": SOURCE_GATE_PARENT.relative_to(ROOT).as_posix(),
                "schema": source_gate["schema"],
                "status": source_gate["status"],
                "payload_sha256": source_gate["payload_sha256"],
            },
            "declared_registered_ladder_source": {
                "path": DECLARED_LADDER_PARENT.relative_to(ROOT).as_posix(),
                "schema": declared_ladder["schema"],
                "status": declared_ladder["status"],
                "payload_sha256": declared_ladder["payload_sha256"],
            },
        },
        "full_raw_stiffness_tail": full_tail,
        "finite_anchor": anchor,
        "inverse_admission_gate": admission,
        "continuum_geometry_assessment": geometry_assessment,
        "exact_counterexamples": exact_counterexamples(),
        "operational_stiffness_response": operational_response_report(
            source_gate, declared_ladder
        ),
        "transfer_boundary": {
            "scalar_rescaling_cancellation_proved": True,
            "rotation_equivariant_transfer_proved": False,
            "multiplicity_one_proved": False,
            "radial_copy_mixing_excluded": False,
            "screen_to_observable_map_selected": False,
        },
        "verification_scope": {
            "exact_tail_arithmetic_reimplemented": True,
            "registered_mesh_builder_shared": True,
            "harmonic_design_stiffness_and_biposh_kernels_shared": True,
            "independent_harmonic_implementation": False,
            "classification": (
                "correlated-kernel replay with separately implemented certificate "
                "and boundary checks"
            ),
        },
        "selection_decision": {
            "full_raw_stiffness_cauchy_limit": True,
            "uniform_continuum_coercivity": False,
            "projected_quotient_continuum_tail": False,
            "full_inverse_covariance_continuum_limit": False,
            "source_ensemble_selected": False,
            "all_level_stiffness_response_source_selected": False,
            "physical_response_readout_selected": False,
            "physical_covariance_selected": False,
            "physical_prediction": False,
            "promotion_allowed": False,
        },
        "lean_boundary": {
            "repository": "FloatingPragma/observer-patch-holography",
            "path": "Lean/Screen/BipoSHInverseBoundary.lean",
            "stiffness_readback_theorem": "stiffness_recovered_from_uniform_repair",
            "inverse_counterexample_theorems": [
                "collapsing_stiffness_positive",
                "collapsing_stiffness_arbitrarily_small",
                "inverse_nuisance_unbounded",
            ],
            "copy_mixing_counterexample_theorem": "copy_mixing_changes_statistic",
            "required_declarations": LEAN_BOUNDARY_DECLARATIONS,
            "declaration_presence_gate": (
                "the RER Lean module contains a #check for every required "
                "declaration and is registered in the OPHScreen lake target; the "
                "sim verifier rejects declared-list drift but does not "
                "inspect or hash-pin the cross-repository Lean source"
            ),
            "uses_new_lean_axiom": False,
            "cross_repository_source_hash_pinned_here": False,
        },
        "claim_boundary": (
            "The complete raw ell=2..8 equal-seam stiffness form has a "
            "conditional Cauchy limit on the ideal refinement tower. The "
            "available conservative tail and finite spectral diagnostic do not "
            "certify a uniformly coercive limit, and they do not cover the "
            "parent's level-dependent low-mode projection. Therefore no "
            "inverse-covariance continuum statistic is promoted. "
            "A shape-regular energy-equivalence proof is a viable route, but "
            "its all-level geometry and interpolation bounds are not supplied. "
            "Unit-edge counting retains only A5 invariance, so the parent's "
            "conditional nonzero L=6 limit is compatible with this boundary. "
            "Stiffness can "
            "instead be reconstructed exactly from an operational one-tick "
            "response if a complete tower repair and intervention/readout bridge "
            "are source-selected. Scalar gain cancellation does not remove "
            "copy-space or radial mixing. No physical response or prediction "
            "is claimed. The numerical anchor replay shares the registered mesh, "
            "harmonic-design, stiffness, and BipoSH kernels with the producer; it "
            "is a correlated-kernel replay rather than an independent numerical "
            "implementation."
        ),
        "source_pins": [
            _file_pin(TAIL_PARENT),
            _file_pin(SOURCE_GATE_PARENT),
            _file_pin(DECLARED_LADDER_PARENT),
            _file_pin(ROOT / "oph_fpe/core/icosahedral.py"),
            _file_pin(ROOT / "oph_fpe/cosmology/a5_biposh_refinement.py"),
            _file_pin(producer),
            _file_pin(verifier),
            _file_pin(tests),
        ],
    }
    packet["payload_sha256"] = _sha_bytes(_canonical_bytes(packet))
    return packet


def write_packet(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    packet = build_inverse_continuum_gate_packet()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(packet))
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    packet = write_packet(args.output)
    print(packet["status"])
    print(packet["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
