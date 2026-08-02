"""Exact conditional conservative time lift for the frozen FZ-11 operator.

The frozen primitive-port packet supplies a spatial operator only.  This
module adds one mathematically standard, explicitly conditional completion.
For a finite map ``B : H_q -> H_p`` with declared rational inner products,
let ``B*`` be its weighted adjoint, ``K = B* B``, and

    J_B = [[0, -B*], [B, 0]].

The packet checks exactly that ``J_B`` is skew-adjoint, that its square is
``diag(-K, -B B*)``, and that the continuous auxiliary-parameter flow obeys
``q'' + K q = 0``.  A six-axis, two-site rational fixture specializes the
frozen Laurent symbol at six phases equal to pi and has ``omega^2 = 12``.

This construction does not source-select B or promote the auxiliary flow
parameter to a physical clock.  It supplies no field sector, Lorentz or boost
law, continuum limit, physical scale, readout, or comparison permission.
Discrete leapfrog evolution is audited separately because its phase obeys a
sine-modified dispersion relation rather than ``omega_tilde^2 = K``.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/fz11_conservative_time_lift_receipt.json"
)
FZ11_PARENT = ROOT / "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = (
    ROOT
    / "oph_fpe/dynamics/verify_fz11_conservative_time_lift_independent.py"
)
TEST_PATH = ROOT / "tests/test_fz11_conservative_time_lift.py"

SCHEMA = "oph.fz11-conservative-time-lift.v1"
STATUS = (
    "EXACT_CONSERVATIVE_TIME_LIFT_FOR_DECLARED_FZ11_OPERATOR_ATTAINED__"
    "SOURCE_B_CLOCK_LORENTZ_SECTOR_CONTINUUM_AND_SCALE_OPEN"
)
PARENT_SCHEMA = "oph.fz11-conditional-3d-translation-bridge.v1"
PARENT_STATUS = (
    "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
    "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
    "BOOST_AND_EXCLUSIVITY_OPEN"
)

Matrix = list[list[Fraction]]
Vector = list[Fraction]


class ConservativeTimeLiftError(RuntimeError):
    """Raised when the exact packet fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConservativeTimeLiftError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConservativeTimeLiftError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ConservativeTimeLiftError(f"non-finite JSON constant: {value}")


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConservativeTimeLiftError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConservativeTimeLiftError(f"{path} is not a JSON object")
    return value


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _matrix_json(matrix: Matrix) -> list[list[str]]:
    return [[_fraction_text(value) for value in row] for row in matrix]


def _zeros(rows: int, columns: int) -> Matrix:
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def _identity(size: int) -> Matrix:
    return [
        [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]


def _transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix, strict=True)]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    _require(bool(left) and bool(right), "matrix multiplication received empty input")
    _require(len(left[0]) == len(right), "matrix multiplication shape mismatch")
    transposed = _transpose(right)
    return [
        [
            sum((x * y for x, y in zip(row, column, strict=True)), Fraction(0))
            for column in transposed
        ]
        for row in left
    ]


def _add(left: Matrix, right: Matrix) -> Matrix:
    _require(
        len(left) == len(right)
        and all(len(a) == len(b) for a, b in zip(left, right, strict=True)),
        "matrix addition shape mismatch",
    )
    return [
        [a + b for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _scale(factor: Fraction, matrix: Matrix) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum((x * y for x, y in zip(row, vector, strict=True)), Fraction(0))
        for row in matrix
    ]


def _diagonal(values: Sequence[Fraction]) -> Matrix:
    return [
        [value if row == column else Fraction(0) for column in range(len(values))]
        for row, value in enumerate(values)
    ]


def _block_diagonal(left: Matrix, right: Matrix) -> Matrix:
    result = _zeros(len(left) + len(right), len(left) + len(right))
    for row in range(len(left)):
        for column in range(len(left)):
            result[row][column] = left[row][column]
    for row in range(len(right)):
        for column in range(len(right)):
            result[len(left) + row][len(left) + column] = right[row][column]
    return result


def _weighted_adjoint(
    operator: Matrix,
    domain_weights: Sequence[Fraction],
    codomain_weights: Sequence[Fraction],
) -> Matrix:
    _require(
        len(operator) == len(codomain_weights)
        and all(len(row) == len(domain_weights) for row in operator),
        "weighted-adjoint shape mismatch",
    )
    _require(
        all(value > 0 for value in domain_weights)
        and all(value > 0 for value in codomain_weights),
        "inner-product weights must be positive",
    )
    transposed = _transpose(operator)
    return [
        [
            transposed[row][column]
            * codomain_weights[column]
            / domain_weights[row]
            for column in range(len(codomain_weights))
        ]
        for row in range(len(domain_weights))
    ]


def _time_generator(operator: Matrix, adjoint: Matrix) -> Matrix:
    q_dimension = len(adjoint)
    p_dimension = len(operator)
    result = _zeros(q_dimension + p_dimension, q_dimension + p_dimension)
    for row in range(q_dimension):
        for column in range(p_dimension):
            result[row][q_dimension + column] = -adjoint[row][column]
    for row in range(p_dimension):
        for column in range(q_dimension):
            result[q_dimension + row][column] = operator[row][column]
    return result


def _fixture() -> dict[str, Any]:
    axis_count = 6
    q_dimension = 2
    shift = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    difference = _add(shift, _scale(Fraction(-1), _identity(q_dimension)))
    operator = [row[:] for _ in range(axis_count) for row in difference]
    q_weights = [Fraction(1)] * q_dimension
    p_weights = [Fraction(1, 2)] * (axis_count * q_dimension)
    adjoint = _weighted_adjoint(operator, q_weights, p_weights)
    stiffness = _multiply(adjoint, operator)
    edge_stiffness = _multiply(operator, adjoint)
    generator = _time_generator(operator, adjoint)
    generator_squared = _multiply(generator, generator)
    expected_squared = _block_diagonal(
        _scale(Fraction(-1), stiffness),
        _scale(Fraction(-1), edge_stiffness),
    )
    phase_metric = _diagonal(q_weights + p_weights)
    skew_residual = _add(
        _multiply(_transpose(generator), phase_metric),
        _multiply(phase_metric, generator),
    )
    zero = _zeros(len(generator), len(generator))
    _require(skew_residual == zero, "declared J_B is not skew-adjoint")
    _require(generator_squared == expected_squared, "declared J_B square drifted")

    constant_mode = [Fraction(1), Fraction(1)]
    alternating_mode = [Fraction(1), Fraction(-1)]
    _require(
        _matvec(stiffness, constant_mode) == [Fraction(0), Fraction(0)],
        "constant mode is not in the frozen kernel",
    )
    _require(
        _matvec(stiffness, alternating_mode)
        == [Fraction(12), Fraction(-12)],
        "alternating frozen eigenvalue is not twelve",
    )
    return {
        "fixture_id": "six_axis_two_site_involution_at_a_squared_one",
        "scalar_field": "Q",
        "q_dimension": q_dimension,
        "p_dimension": axis_count * q_dimension,
        "positive_axis_count": axis_count,
        "a_squared": "1",
        "site_shift_T": _matrix_json(shift),
        "forward_difference_per_axis_T_minus_I": _matrix_json(difference),
        "B": _matrix_json(operator),
        "q_inner_product_gram": _matrix_json(_diagonal(q_weights)),
        "p_inner_product_gram": _matrix_json(_diagonal(p_weights)),
        "B_star": _matrix_json(adjoint),
        "K_equals_B_star_B": _matrix_json(stiffness),
        "B_B_star": _matrix_json(edge_stiffness),
        "J_B": _matrix_json(generator),
        "J_B_squared": _matrix_json(generator_squared),
        "phase_space_gram": _matrix_json(phase_metric),
        "skew_adjoint_residual": _matrix_json(skew_residual),
        "identities": {
            "B_star_is_weighted_adjoint": True,
            "K_equals_B_star_B": True,
            "K_positive_semidefinite": True,
            "J_B_skew_adjoint": True,
            "J_B_squared_equals_diag_minus_K_minus_B_B_star": True,
        },
        "exact_modes": {
            "constant": {"vector": ["1", "1"], "eigenvalue": "0"},
            "alternating": {"vector": ["1", "-1"], "eigenvalue": "12"},
        },
        "frozen_symbol_specialization": {
            "six_positive_translation_phases": ["pi"] * axis_count,
            "formula": "a^-2*sum_alpha(1-cos(theta_alpha))",
            "exact_value_at_a_squared_one": "12",
            "continuous_mode_relation": "omega^2=12",
        },
    }


def _validated_parent() -> dict[str, Any]:
    parent = _load_json_strict(FZ11_PARENT)
    payload = copy.deepcopy(parent)
    digest = payload.pop("receipt_sha256", None)
    contract = parent.get("operator_contract", {})
    if (
        digest != _sha(payload)
        or parent.get("schema") != PARENT_SCHEMA
        or parent.get("status") != PARENT_STATUS
        or parent.get("issue") != 655
        or parent.get("comparison_data_read") is not False
        or contract.get("paired_real_space_form")
        != "K_a=(1/(2a^2))*sum_alpha(2I-T_alpha-T_alpha_inverse)"
        or contract.get("laurent_symbol")
        != "(1/(2a^2))*sum_alpha(2-z_alpha-z_alpha^-1)"
        or contract.get("omega_squared_only") is not True
        or contract.get("time_evolution_or_frequency_sign_selected") is not False
    ):
        raise ConservativeTimeLiftError("frozen FZ-11 parent contract drifted")
    return {
        **_raw_pin(FZ11_PARENT),
        "schema": PARENT_SCHEMA,
        "status": PARENT_STATUS,
        "receipt_sha256": digest,
        "operator_projection": {
            "paired_real_space_form": contract["paired_real_space_form"],
            "laurent_symbol": contract["laurent_symbol"],
            "cosine_symbol": contract["cosine_symbol"],
            "omega_squared_only": True,
            "time_evolution_or_frequency_sign_selected": False,
        },
    }


def _laurent_terms_at_unit_scale() -> list[dict[str, Any]]:
    terms: dict[tuple[int, ...], Fraction] = {
        (0, 0, 0, 0, 0, 0): Fraction(6)
    }
    for axis in range(6):
        for orientation in (-1, 1):
            exponent = [0] * 6
            exponent[axis] = orientation
            terms[tuple(exponent)] = Fraction(-1, 2)
    return [
        {
            "exponents": list(exponents),
            "coefficient": _fraction_text(coefficient),
        }
        for exponents, coefficient in sorted(terms.items())
    ]


def _direct_factorization_uniqueness() -> dict[str, Any]:
    """Exact orientation control for the declared direct-incidence class."""

    identity = _identity(3)
    forward_shift = [
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
        [Fraction(1), Fraction(0), Fraction(0)],
    ]
    inverse_shift = _transpose(forward_shift)
    forward = _add(forward_shift, _scale(Fraction(-1), identity))
    reverse = _add(inverse_shift, _scale(Fraction(-1), identity))
    orientation_isometry = _scale(Fraction(-1), inverse_shift)
    _require(
        _multiply(orientation_isometry, forward) == reverse,
        "edge-orientation factorization failed",
    )
    _require(
        _multiply(_transpose(orientation_isometry), orientation_isometry)
        == identity,
        "edge-orientation map is not an isometry",
    )
    weights = [Fraction(1, 2)] * 3
    adjoint_forward = _weighted_adjoint(forward, [Fraction(1)] * 3, weights)
    adjoint_reverse = _weighted_adjoint(reverse, [Fraction(1)] * 3, weights)
    stiffness_forward = _multiply(adjoint_forward, forward)
    stiffness_reverse = _multiply(adjoint_reverse, reverse)
    _require(
        stiffness_forward == stiffness_reverse,
        "edge orientation changed the stiffness operator",
    )
    generator_forward = _time_generator(forward, adjoint_forward)
    generator_reverse = _time_generator(reverse, adjoint_reverse)
    phase_isometry = _block_diagonal(identity, orientation_isometry)
    conjugated = _multiply(
        _multiply(phase_isometry, generator_forward),
        _transpose(phase_isometry),
    )
    _require(
        conjugated == generator_reverse,
        "edge orientation did not conjugate the time generator",
    )
    return {
        "class": (
            "stack each declared T_alpha-I or T_alpha^-1-I exactly once, "
            "with no additional coefficient or internal map and with the "
            "declared common edge weight"
        ),
        "orientation_identity": (
            "T_alpha^-1-I=(-T_alpha^-1)*(T_alpha-I)"
        ),
        "orientation_map_is_edge_isometry": True,
        "axis_permutation_is_edge_isometry": True,
        "B_prime_equals_U_B": True,
        "B_prime_star_B_prime_equals_B_star_B": True,
        "J_B_prime_equals_diag_I_U_J_B_diag_I_U_inverse": True,
        "six_axis_orientation_choices": 64,
        "six_axis_permutation_choices": 720,
        "labeled_presentation_operation_count": 46080,
        "exact_three_site_orientation_fixture": {
            "forward_shift": _matrix_json(forward_shift),
            "inverse_shift": _matrix_json(inverse_shift),
            "forward_difference": _matrix_json(forward),
            "reverse_difference": _matrix_json(reverse),
            "orientation_isometry": _matrix_json(orientation_isometry),
            "common_K": _matrix_json(stiffness_forward),
            "J_conjugacy_exact": True,
        },
        "scope_boundary": (
            "uniqueness holds inside the declared direct one-component-per-"
            "axis incidence class; it does not select the translations, "
            "exclude extra ranges or components, or identify a physical field. "
            "Generic positive-semidefinite square roots and factorizations on "
            "enlarged or internally rotated momentum spaces remain nonunique"
        ),
        "generic_psd_factorizations_remain_nonunique_outside_declared_class": True,
        "enlarged_or_internal_momentum_factorizations_remain_nonunique": True,
    }


def _time_law_uniqueness(stiffness: Matrix) -> dict[str, Any]:
    """Exact finite witness for canonical-energy selection of A = K."""

    mutated = [row[:] for row in stiffness]
    mutated[0][0] += 1
    displacement = [Fraction(1), Fraction(0)]
    velocity = [Fraction(1), Fraction(0)]
    k_minus_a = _add(stiffness, _scale(Fraction(-1), mutated))
    derivative = sum(
        (
            velocity[index] * value
            for index, value in enumerate(_matvec(k_minus_a, displacement))
        ),
        Fraction(0),
    )
    _require(derivative == -1, "mutated energy derivative witness drifted")
    return {
        "declared_hypotheses": [
            "finite-dimensional real configuration space",
            "linear second-order law q_second_derivative+A*q=0",
            "canonical velocity inner product in the kinetic term",
            "self-adjoint frozen stiffness K",
            "conservation of E_K for every initial displacement and velocity",
        ],
        "candidate_second_order_law": "q_second_derivative+A*q=0",
        "canonical_energy": (
            "E_K(q,v)=(1/2)*||v||^2+(1/2)*<q,Kq>"
        ),
        "derivative_identity": "dE_K/dt=<v,(K-A)q>",
        "all_initial_data_consequence": (
            "if dE_K/dt=0 for every initial (q,v), then A=K"
        ),
        "A_equals_K_forced_under_declared_energy_premise": True,
        "exact_mutation_control": {
            "A_not_K": _matrix_json(mutated),
            "initial_q": ["1", "0"],
            "initial_v": ["1", "0"],
            "dE_K_dt_at_initial_state": "-1",
            "conservation_violated": True,
        },
        "complete_response_energy_identified_with_E_K": False,
        "E_K_identified_with_conserved_phase_norm": False,
        "physical_clock_selected": False,
        "scope_boundary": (
            "the implication selects A only after the canonical energy is "
            "identified with the conserved complete-response energy; A1/A2 "
            "do not supply that identification in this packet"
        ),
    }


def _payload() -> dict[str, Any]:
    fixture = _fixture()
    stiffness = [
        [Fraction(value) for value in row]
        for row in fixture["K_equals_B_star_B"]
    ]
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "issue": 655,
        "comparison_data_read": False,
        "target_data_read": False,
        "target_data_paths": [],
        "parent_pin": _validated_parent(),
        "declared_finite_factorization": fixture,
        "frozen_operator_binding": {
            "scope": (
                "finite unitary realizations of the six positive primitive "
                "translations, conditional on a declared scale a"
            ),
            "B_definition": "(Bq)_alpha=(T_alpha-I)q",
            "edge_inner_product": (
                "<p,r>_p=(1/(2a^2))*sum_alpha <p_alpha,r_alpha>"
            ),
            "weighted_adjoint": (
                "(B_star p)=(1/(2a^2))*sum_alpha(T_alpha^-1-I)p_alpha"
            ),
            "factorization": (
                "B_star*B=(1/(2a^2))*sum_alpha(2I-T_alpha-T_alpha^-1)=K_a"
            ),
            "laurent_symbol": (
                "K(z)=(1/(2a^2))*sum_alpha(2-z_alpha-z_alpha^-1)"
            ),
            "laurent_terms_at_a_squared_one": _laurent_terms_at_unit_scale(),
            "plane_wave_symbol": (
                "lambda(k)=a^-2*sum_alpha(1-cos(a*k.u_alpha))"
            ),
            "twelve_port_equivalence": (
                "a^-2*sum_alpha(1-cos(theta_alpha))="
                "(1/(2a^2))*sum_p[1-cos(theta_p)] under antipodal pairing"
            ),
            "matches_parent_frozen_spatial_symbol": True,
            "translation_action_source_selected": False,
            "factorization_arbitrary_within_declared_direct_incidence_class": False,
            "generic_psd_factorizations_remain_nonunique_outside_declared_class": True,
            "enlarged_or_internal_momentum_factorizations_remain_nonunique": True,
            "direct_factorization_canonical_up_to_momentum_frame_isometry": True,
            "finite_fixture_is_physical_spatial_domain": False,
        },
        "direct_factorization_uniqueness": _direct_factorization_uniqueness(),
        "continuous_time_lift": {
            "generator": "J_B=[[0,-B_star],[B,0]]",
            "state_equation": "d/dt(q,p)=(-B_star*p,B*q)",
            "square": "J_B^2=diag(-B_star*B,-B*B_star)",
            "coordinate_equation": "q_second_derivative+K*q=0",
            "mode_equation": "Kq=lambda*q implies q''+lambda*q=0",
            "frozen_relation": "omega^2=lambda",
            "conservation_law": (
                "d/dt(||q||_q^2+||p||_p^2)=0 because J_B is skew-adjoint"
            ),
            "flow": "exp(t*J_B)",
            "flow_parameter": "auxiliary continuous real parameter t",
            "conditional_continuous_evolution_supplied": True,
            "conserved_phase_norm_identified_with_canonical_second_order_energy": False,
            "conserved_phase_norm_identified_with_physical_energy": False,
            "conserved_phase_norm_selects_physical_clock": False,
            "physical_clock_selected": False,
        },
        "time_law_uniqueness_from_energy": _time_law_uniqueness(stiffness),
        "discrete_time_audit": {
            "separate_from_continuous_packet": True,
            "method": "centered_second_difference_or_leapfrog",
            "mode_recurrence": (
                "q_(n+1)-2q_n+q_(n-1)+h^2*lambda*q_n=0"
            ),
            "characteristic_equation": "r^2+(h^2*lambda-2)r+1=0",
            "stable_phase_relation": "4*h^-2*sin(theta/2)^2=lambda",
            "numerical_frequency": "omega_tilde=theta/h",
            "continuous_relation_retained_separately": "omega^2=lambda",
            "omega_tilde_squared_identified_with_lambda": False,
            "leapfrog_map_identified_with_exp_h_J_B": False,
            "stability_condition": "0<=h^2*lambda<=4",
            "exact_fixture": {
                "lambda": "12",
                "h": "1/2",
                "h_squared_lambda": "3",
                "characteristic_polynomial": "r^2+r+1=0",
                "principal_stable_phase": "theta=2*pi/3",
                "sine_modified_relation": (
                    "4*(1/2)^-2*sin((2*pi/3)/2)^2=12"
                ),
                "continuous_frequency": "omega=2*sqrt(3)",
                "leapfrog_numerical_frequency": "omega_tilde=4*pi/3",
                "frequencies_identified": False,
            },
            "physical_discrete_clock_selected": False,
            "repair_tick_supplies_physical_time": False,
            "clock_or_continuum_theorem_required_for_physical_time": True,
        },
        "attainment": {
            "finite_declared_B_exact": True,
            "weighted_adjoint_exact": True,
            "K_equals_B_star_B_exact": True,
            "J_B_skew_adjoint_exact": True,
            "J_B_square_exact": True,
            "continuous_coordinate_equation_exact": True,
            "frozen_spatial_eigenvalue_is_omega_squared_conditionally": True,
            "conditional_auxiliary_time_evolution_supplied": True,
            "B_source_selected": False,
            "phase_norm_identified_with_physical_energy": False,
            "physical_clock_selected": False,
            "lorentz_or_boost_law_derived": False,
            "physical_field_sector_selected": False,
            "continuum_limit_derived": False,
            "physical_scale_selected": False,
            "physical_readout_selected": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
            "issue_655_closure_supported": False,
        },
        "implementation_pins": [
            _raw_pin(PRODUCER_PATH),
            _raw_pin(VERIFIER_PATH),
            _raw_pin(TEST_PATH),
        ],
        "claim_boundary": (
            "For the declared finite rational B, the weighted-adjoint "
            "factorization and conservative continuous flow are exact. The "
            "same construction conditionally factors the frozen primitive "
            "spatial operator whenever its six translations have a finite "
            "unitary realization. Once those translations and the direct "
            "one-component incidence class are supplied, B is canonical up "
            "to a momentum-frame isometry. Conservation of the declared "
            "canonical energy forces the second-order operator A to equal K, "
            "but A1/A2 identification of that energy remains open. The norm "
            "conserved by the auxiliary phase flow is not identified with "
            "that second-order energy, any physical energy, or a clock. The "
            "auxiliary parameter is not a physical clock. The packet does "
            "not source-select the translation action, a Lorentz or boost "
            "law, a field sector, a continuum limit, a physical scale, or a "
            "readout. The leapfrog audit carries its sine-modified frequency "
            "and is not substituted for the continuous frozen relation. A "
            "discrete repair tick cannot supply physical time without a "
            "clock or continuum theorem. No target or comparison data are "
            "read."
        ),
    }


def produce_receipt() -> dict[str, Any]:
    payload = _payload()
    return {**payload, "receipt_sha256": _sha(payload)}


def verify_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        if _canonical_bytes(received) != _canonical_bytes(_payload()):
            reasons.append("producer_replay_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ConservativeTimeLiftError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    return {
        "schema": "oph.fz11-conservative-time-lift-verification.v1",
        "receipt": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "physical_promotion": False,
    }


def load_receipt_strict(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load_json_strict(path)
    result = verify_receipt(report)
    if result["receipt"] is not True:
        raise ConservativeTimeLiftError(
            "strict receipt verification failed: " + ",".join(result["reasons"])
        )
    return report


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        try:
            report = load_receipt_strict(args.verify)
            result = verify_receipt(report)
        except ConservativeTimeLiftError as error:
            result = {
                "schema": "oph.fz11-conservative-time-lift-verification.v1",
                "receipt": False,
                "status": "FAIL",
                "reasons": [str(error)],
            }
        _write_json(result, None if args.output == DEFAULT_RECEIPT else args.output)
        return 0 if result["receipt"] else 1
    receipt = produce_receipt()
    result = verify_receipt(receipt)
    if result["receipt"] is not True:
        _write_json(result, None)
        return 1
    _write_json(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
