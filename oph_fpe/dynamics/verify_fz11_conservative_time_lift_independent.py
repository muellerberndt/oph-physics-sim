"""Independent exact verifier for the conditional FZ-11 time-lift packet.

This module imports neither the producer nor another OPH algebra helper.  It
reconstructs the rational finite factorization, weighted adjoint, conservative
generator, orientation isometry, energy-uniqueness witness, and leapfrog
boundary directly from the serialized contract and its pinned FZ-11 parent.
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
PARENT_RELATIVE = Path(
    "data/repair_closure/fz11_3d_translation_bridge_receipt.json"
)
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
EXPECTED_IMPLEMENTATION_PATHS = {
    "oph_fpe/dynamics/fz11_conservative_time_lift.py",
    "oph_fpe/dynamics/verify_fz11_conservative_time_lift_independent.py",
    "tests/test_fz11_conservative_time_lift.py",
}

Matrix = list[list[Fraction]]
Vector = list[Fraction]


class IndependentVerificationError(RuntimeError):
    """The packet failed independent exact verification."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise IndependentVerificationError(f"non-finite JSON constant: {value}")


def _strict_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(f"cannot load {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


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


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_matrix(value: Any, rows: int, columns: int, label: str) -> Matrix:
    _require(isinstance(value, list) and len(value) == rows, f"{label} row drift")
    parsed: Matrix = []
    for row in value:
        _require(
            isinstance(row, list) and len(row) == columns,
            f"{label} column drift",
        )
        try:
            parsed.append([Fraction(entry) for entry in row])
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise IndependentVerificationError(f"{label} is not rational") from error
    return parsed


def _render(matrix: Matrix) -> list[list[str]]:
    def text(value: Fraction) -> str:
        return (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        )

    return [[text(entry) for entry in row] for row in matrix]


def _zero(rows: int, columns: int) -> Matrix:
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def _eye(size: int) -> Matrix:
    return [
        [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix, strict=True)]


def _product(left: Matrix, right: Matrix) -> Matrix:
    _require(len(left[0]) == len(right), "independent matrix shape mismatch")
    columns = _transpose(right)
    return [
        [
            sum((a * b for a, b in zip(row, column, strict=True)), Fraction(0))
            for column in columns
        ]
        for row in left
    ]


def _sum(left: Matrix, right: Matrix) -> Matrix:
    _require(
        len(left) == len(right)
        and all(len(a) == len(b) for a, b in zip(left, right, strict=True)),
        "independent matrix-sum shape mismatch",
    )
    return [
        [a + b for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _times(factor: Fraction, matrix: Matrix) -> Matrix:
    return [[factor * entry for entry in row] for row in matrix]


def _diag(values: Sequence[Fraction]) -> Matrix:
    return [
        [value if row == column else Fraction(0) for column in range(len(values))]
        for row, value in enumerate(values)
    ]


def _block_diag(left: Matrix, right: Matrix) -> Matrix:
    result = _zero(len(left) + len(right), len(left) + len(right))
    for row, values in enumerate(left):
        result[row][: len(left)] = values
    for row, values in enumerate(right):
        result[len(left) + row][len(left) :] = values
    return result


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum((a * b for a, b in zip(row, vector, strict=True)), Fraction(0))
        for row in matrix
    ]


def _adjoint_from_diagonal_metrics(
    operator: Matrix,
    domain: Sequence[Fraction],
    codomain: Sequence[Fraction],
) -> Matrix:
    transposed = _transpose(operator)
    return [
        [
            transposed[row][column] * codomain[column] / domain[row]
            for column in range(len(codomain))
        ]
        for row in range(len(domain))
    ]


def _generator(operator: Matrix, adjoint: Matrix) -> Matrix:
    n = len(adjoint)
    m = len(operator)
    result = _zero(n + m, n + m)
    for row in range(n):
        for column in range(m):
            result[row][n + column] = -adjoint[row][column]
    for row in range(m):
        for column in range(n):
            result[n + row][column] = operator[row][column]
    return result


def _independent_fixture() -> dict[str, Any]:
    shift = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    difference = _sum(shift, _times(Fraction(-1), _eye(2)))
    operator = [row[:] for _ in range(6) for row in difference]
    q_weights = [Fraction(1), Fraction(1)]
    p_weights = [Fraction(1, 2)] * 12
    adjoint = _adjoint_from_diagonal_metrics(operator, q_weights, p_weights)
    stiffness = _product(adjoint, operator)
    edge_stiffness = _product(operator, adjoint)
    generator = _generator(operator, adjoint)
    phase_metric = _diag(q_weights + p_weights)
    skew_residual = _sum(
        _product(_transpose(generator), phase_metric),
        _product(phase_metric, generator),
    )
    square = _product(generator, generator)
    expected_square = _block_diag(
        _times(Fraction(-1), stiffness),
        _times(Fraction(-1), edge_stiffness),
    )
    _require(skew_residual == _zero(14, 14), "independent skew-adjointness failed")
    _require(square == expected_square, "independent J square failed")
    _require(_apply(stiffness, [Fraction(1), Fraction(1)]) == [0, 0], "kernel drift")
    _require(
        _apply(stiffness, [Fraction(1), Fraction(-1)]) == [12, -12],
        "frozen eigenvalue drift",
    )
    return {
        "shift": shift,
        "difference": difference,
        "operator": operator,
        "q_gram": _diag(q_weights),
        "p_gram": _diag(p_weights),
        "adjoint": adjoint,
        "stiffness": stiffness,
        "edge_stiffness": edge_stiffness,
        "generator": generator,
        "square": square,
        "phase_metric": phase_metric,
        "skew_residual": skew_residual,
    }


def _expected_laurent_terms() -> list[dict[str, Any]]:
    terms: dict[tuple[int, ...], Fraction] = {(0, 0, 0, 0, 0, 0): Fraction(6)}
    for axis in range(6):
        for sign in (-1, 1):
            exponent = [0] * 6
            exponent[axis] = sign
            terms[tuple(exponent)] = Fraction(-1, 2)
    return [
        {"exponents": list(exponents), "coefficient": _render([[coefficient]])[0][0]}
        for exponents, coefficient in sorted(terms.items())
    ]


def _verify_parent(report: Mapping[str, Any], root: Path) -> None:
    pin = report.get("parent_pin")
    _require(isinstance(pin, Mapping), "parent pin missing")
    path = root / PARENT_RELATIVE
    raw = path.read_bytes()
    _require(pin.get("path") == PARENT_RELATIVE.as_posix(), "parent path drift")
    _require(pin.get("bytes") == len(raw), "parent byte count drift")
    _require(pin.get("sha256") == _raw_sha(path), "parent raw hash drift")
    parent = _strict_load(path)
    payload = copy.deepcopy(parent)
    digest = payload.pop("receipt_sha256", None)
    _require(digest == _sha(payload), "parent receipt self-hash drift")
    contract = parent.get("operator_contract", {})
    _require(
        parent.get("schema") == pin.get("schema") == PARENT_SCHEMA
        and parent.get("status") == pin.get("status") == PARENT_STATUS
        and parent.get("issue") == 655
        and parent.get("comparison_data_read") is False
        and pin.get("receipt_sha256") == digest,
        "parent contract header drift",
    )
    expected_projection = {
        "paired_real_space_form": (
            "K_a=(1/(2a^2))*sum_alpha(2I-T_alpha-T_alpha_inverse)"
        ),
        "laurent_symbol": "(1/(2a^2))*sum_alpha(2-z_alpha-z_alpha^-1)",
        "cosine_symbol": (
            "omega^2(k)=(1/(2a^2))*sum_p[1-cos(a*k.u_p)]"
        ),
        "omega_squared_only": True,
        "time_evolution_or_frequency_sign_selected": False,
    }
    _require(
        pin.get("operator_projection") == expected_projection,
        "parent operator projection drift",
    )
    _require(
        contract.get("paired_real_space_form")
        == expected_projection["paired_real_space_form"]
        and contract.get("laurent_symbol") == expected_projection["laurent_symbol"]
        and contract.get("cosine_symbol") == expected_projection["cosine_symbol"]
        and contract.get("omega_squared_only") is True
        and contract.get("time_evolution_or_frequency_sign_selected") is False,
        "parent frozen operator drift",
    )


def _verify_serialized_fixture(report: Mapping[str, Any]) -> None:
    row = report.get("declared_finite_factorization")
    _require(isinstance(row, Mapping), "finite factorization missing")
    expected = _independent_fixture()
    dimensions = (2, 12)
    _require(
        row.get("fixture_id") == "six_axis_two_site_involution_at_a_squared_one"
        and row.get("scalar_field") == "Q"
        and (row.get("q_dimension"), row.get("p_dimension")) == dimensions
        and row.get("positive_axis_count") == 6
        and row.get("a_squared") == "1",
        "finite fixture header drift",
    )
    matrix_fields = {
        "site_shift_T": (expected["shift"], 2, 2),
        "forward_difference_per_axis_T_minus_I": (expected["difference"], 2, 2),
        "B": (expected["operator"], 12, 2),
        "q_inner_product_gram": (expected["q_gram"], 2, 2),
        "p_inner_product_gram": (expected["p_gram"], 12, 12),
        "B_star": (expected["adjoint"], 2, 12),
        "K_equals_B_star_B": (expected["stiffness"], 2, 2),
        "B_B_star": (expected["edge_stiffness"], 12, 12),
        "J_B": (expected["generator"], 14, 14),
        "J_B_squared": (expected["square"], 14, 14),
        "phase_space_gram": (expected["phase_metric"], 14, 14),
        "skew_adjoint_residual": (expected["skew_residual"], 14, 14),
    }
    for name, (matrix, rows, columns) in matrix_fields.items():
        parsed = _parse_matrix(row.get(name), rows, columns, name)
        _require(parsed == matrix and row.get(name) == _render(matrix), f"{name} drift")
    _require(
        row.get("identities")
        == {
            "B_star_is_weighted_adjoint": True,
            "K_equals_B_star_B": True,
            "K_positive_semidefinite": True,
            "J_B_skew_adjoint": True,
            "J_B_squared_equals_diag_minus_K_minus_B_B_star": True,
        },
        "finite identity flags drift",
    )
    _require(
        row.get("exact_modes")
        == {
            "constant": {"vector": ["1", "1"], "eigenvalue": "0"},
            "alternating": {"vector": ["1", "-1"], "eigenvalue": "12"},
        }
        and row.get("frozen_symbol_specialization", {}).get(
            "exact_value_at_a_squared_one"
        )
        == "12"
        and row.get("frozen_symbol_specialization", {}).get(
            "continuous_mode_relation"
        )
        == "omega^2=12",
        "finite eigenmode binding drift",
    )


def _verify_orientation_uniqueness(report: Mapping[str, Any]) -> None:
    row = report.get("direct_factorization_uniqueness")
    _require(isinstance(row, Mapping), "direct-factorization theorem missing")
    fixture = row.get("exact_three_site_orientation_fixture")
    _require(isinstance(fixture, Mapping), "orientation fixture missing")
    identity = _eye(3)
    forward_shift = [
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
        [Fraction(1), Fraction(0), Fraction(0)],
    ]
    inverse_shift = _transpose(forward_shift)
    forward = _sum(forward_shift, _times(Fraction(-1), identity))
    reverse = _sum(inverse_shift, _times(Fraction(-1), identity))
    isometry = _times(Fraction(-1), inverse_shift)
    _require(_product(isometry, forward) == reverse, "orientation identity failed")
    _require(
        _product(_transpose(isometry), isometry) == identity,
        "orientation map is not isometric",
    )
    weights = [Fraction(1, 2)] * 3
    forward_star = _adjoint_from_diagonal_metrics(forward, [Fraction(1)] * 3, weights)
    reverse_star = _adjoint_from_diagonal_metrics(reverse, [Fraction(1)] * 3, weights)
    common_k = _product(forward_star, forward)
    _require(common_k == _product(reverse_star, reverse), "orientation changed K")
    j_forward = _generator(forward, forward_star)
    j_reverse = _generator(reverse, reverse_star)
    phase_isometry = _block_diag(identity, isometry)
    _require(
        _product(_product(phase_isometry, j_forward), _transpose(phase_isometry))
        == j_reverse,
        "orientation did not conjugate J",
    )
    expected_fixture = {
        "forward_shift": _render(forward_shift),
        "inverse_shift": _render(inverse_shift),
        "forward_difference": _render(forward),
        "reverse_difference": _render(reverse),
        "orientation_isometry": _render(isometry),
        "common_K": _render(common_k),
        "J_conjugacy_exact": True,
    }
    _require(fixture == expected_fixture, "orientation fixture serialization drift")
    _require(
        row.get("class")
        == (
            "stack each declared T_alpha-I or T_alpha^-1-I exactly once, "
            "with no additional coefficient or internal map and with the "
            "declared common edge weight"
        )
        and row.get("orientation_identity")
        == "T_alpha^-1-I=(-T_alpha^-1)*(T_alpha-I)"
        and row.get("scope_boundary")
        == (
            "uniqueness holds inside the declared direct one-component-per-"
            "axis incidence class; it does not select the translations, "
            "exclude extra ranges or components, or identify a physical field. "
            "Generic positive-semidefinite square roots and factorizations on "
            "enlarged or internally rotated momentum spaces remain nonunique"
        )
        and row.get(
            "generic_psd_factorizations_remain_nonunique_outside_declared_class"
        )
        is True
        and row.get(
            "enlarged_or_internal_momentum_factorizations_remain_nonunique"
        )
        is True,
        "orientation theorem scope drift",
    )
    for key in (
        "orientation_map_is_edge_isometry",
        "axis_permutation_is_edge_isometry",
        "B_prime_equals_U_B",
        "B_prime_star_B_prime_equals_B_star_B",
        "J_B_prime_equals_diag_I_U_J_B_diag_I_U_inverse",
    ):
        _require(row.get(key) is True, f"orientation theorem flag drift: {key}")
    _require(
        row.get("six_axis_orientation_choices") == 64
        and row.get("six_axis_permutation_choices") == 720
        and row.get("labeled_presentation_operation_count") == 46080,
        "orientation/permutation census drift",
    )


def _verify_energy_uniqueness(report: Mapping[str, Any]) -> None:
    row = report.get("time_law_uniqueness_from_energy")
    _require(isinstance(row, Mapping), "energy uniqueness packet missing")
    _require(
        row.get("declared_hypotheses")
        == [
            "finite-dimensional real configuration space",
            "linear second-order law q_second_derivative+A*q=0",
            "canonical velocity inner product in the kinetic term",
            "self-adjoint frozen stiffness K",
            "conservation of E_K for every initial displacement and velocity",
        ],
        "energy uniqueness hypotheses drift",
    )
    fixture = report["declared_finite_factorization"]
    stiffness = _parse_matrix(fixture["K_equals_B_star_B"], 2, 2, "K")
    control = row.get("exact_mutation_control")
    _require(isinstance(control, Mapping), "energy mutation control missing")
    mutated = _parse_matrix(control.get("A_not_K"), 2, 2, "mutated A")
    q = [Fraction(value) for value in control.get("initial_q", [])]
    velocity = [Fraction(value) for value in control.get("initial_v", [])]
    _require(len(q) == len(velocity) == 2, "energy witness vector drift")
    residual = _sum(stiffness, _times(Fraction(-1), mutated))
    derivative = sum(
        (
            velocity[index] * value
            for index, value in enumerate(_apply(residual, q))
        ),
        Fraction(0),
    )
    _require(
        derivative == -1
        and control.get("dE_K_dt_at_initial_state") == "-1"
        and control.get("conservation_violated") is True,
        "energy uniqueness mutation did not fail",
    )
    _require(
        row.get("derivative_identity") == "dE_K/dt=<v,(K-A)q>"
        and row.get("candidate_second_order_law")
        == "q_second_derivative+A*q=0"
        and row.get("canonical_energy")
        == "E_K(q,v)=(1/2)*||v||^2+(1/2)*<q,Kq>"
        and row.get("all_initial_data_consequence")
        == "if dE_K/dt=0 for every initial (q,v), then A=K"
        and row.get("A_equals_K_forced_under_declared_energy_premise") is True
        and row.get("complete_response_energy_identified_with_E_K") is False
        and row.get("E_K_identified_with_conserved_phase_norm") is False
        and row.get("physical_clock_selected") is False
        and row.get("scope_boundary")
        == (
            "the implication selects A only after the canonical energy is "
            "identified with the conserved complete-response energy; A1/A2 "
            "do not supply that identification in this packet"
        ),
        "energy uniqueness boundary drift",
    )


def _verify_boundaries(report: Mapping[str, Any]) -> None:
    binding = report.get("frozen_operator_binding", {})
    _require(
        binding.get("scope")
        == (
            "finite unitary realizations of the six positive primitive "
            "translations, conditional on a declared scale a"
        )
        and binding.get("B_definition") == "(Bq)_alpha=(T_alpha-I)q"
        and binding.get("edge_inner_product")
        == "<p,r>_p=(1/(2a^2))*sum_alpha <p_alpha,r_alpha>"
        and binding.get("weighted_adjoint")
        == "(B_star p)=(1/(2a^2))*sum_alpha(T_alpha^-1-I)p_alpha"
        and binding.get("factorization")
        == "B_star*B=(1/(2a^2))*sum_alpha(2I-T_alpha-T_alpha^-1)=K_a"
        and binding.get("laurent_symbol")
        == "K(z)=(1/(2a^2))*sum_alpha(2-z_alpha-z_alpha^-1)"
        and binding.get("laurent_terms_at_a_squared_one") == _expected_laurent_terms()
        and binding.get("plane_wave_symbol")
        == "lambda(k)=a^-2*sum_alpha(1-cos(a*k.u_alpha))"
        and binding.get("twelve_port_equivalence")
        == (
            "a^-2*sum_alpha(1-cos(theta_alpha))="
            "(1/(2a^2))*sum_p[1-cos(theta_p)] under antipodal pairing"
        )
        and binding.get("matches_parent_frozen_spatial_symbol") is True
        and binding.get("translation_action_source_selected") is False
        and binding.get(
            "factorization_arbitrary_within_declared_direct_incidence_class"
        )
        is False
        and binding.get(
            "generic_psd_factorizations_remain_nonunique_outside_declared_class"
        )
        is True
        and binding.get(
            "enlarged_or_internal_momentum_factorizations_remain_nonunique"
        )
        is True
        and binding.get(
            "direct_factorization_canonical_up_to_momentum_frame_isometry"
        )
        is True
        and binding.get("finite_fixture_is_physical_spatial_domain") is False,
        "frozen operator/factorization boundary drift",
    )
    continuous = report.get("continuous_time_lift", {})
    _require(
        continuous.get("generator") == "J_B=[[0,-B_star],[B,0]]"
        and continuous.get("state_equation")
        == "d/dt(q,p)=(-B_star*p,B*q)"
        and continuous.get("square")
        == "J_B^2=diag(-B_star*B,-B*B_star)"
        and continuous.get("coordinate_equation")
        == "q_second_derivative+K*q=0"
        and continuous.get("mode_equation")
        == "Kq=lambda*q implies q''+lambda*q=0"
        and continuous.get("frozen_relation") == "omega^2=lambda"
        and continuous.get("conservation_law")
        == "d/dt(||q||_q^2+||p||_p^2)=0 because J_B is skew-adjoint"
        and continuous.get("flow") == "exp(t*J_B)"
        and continuous.get("flow_parameter")
        == "auxiliary continuous real parameter t"
        and continuous.get("conditional_continuous_evolution_supplied") is True
        and continuous.get(
            "conserved_phase_norm_identified_with_canonical_second_order_energy"
        )
        is False
        and continuous.get("conserved_phase_norm_identified_with_physical_energy")
        is False
        and continuous.get("conserved_phase_norm_selects_physical_clock") is False
        and continuous.get("physical_clock_selected") is False,
        "continuous time-lift boundary drift",
    )
    discrete = report.get("discrete_time_audit", {})
    _require(
        discrete.get("separate_from_continuous_packet") is True
        and discrete.get("method") == "centered_second_difference_or_leapfrog"
        and discrete.get("mode_recurrence")
        == "q_(n+1)-2q_n+q_(n-1)+h^2*lambda*q_n=0"
        and discrete.get("characteristic_equation")
        == "r^2+(h^2*lambda-2)r+1=0"
        and discrete.get("stable_phase_relation")
        == "4*h^-2*sin(theta/2)^2=lambda"
        and discrete.get("numerical_frequency") == "omega_tilde=theta/h"
        and discrete.get("continuous_relation_retained_separately")
        == "omega^2=lambda"
        and discrete.get("omega_tilde_squared_identified_with_lambda") is False
        and discrete.get("leapfrog_map_identified_with_exp_h_J_B") is False
        and discrete.get("physical_discrete_clock_selected") is False
        and discrete.get("repair_tick_supplies_physical_time") is False
        and discrete.get("clock_or_continuum_theorem_required_for_physical_time")
        is True
        and discrete.get("stability_condition") == "0<=h^2*lambda<=4",
        "discrete-time boundary drift",
    )
    exact = discrete.get("exact_fixture", {})
    _require(
        exact.get("lambda") == "12"
        and exact.get("h") == "1/2"
        and exact.get("h_squared_lambda") == "3"
        and exact.get("characteristic_polynomial") == "r^2+r+1=0"
        and exact.get("principal_stable_phase") == "theta=2*pi/3"
        and exact.get("sine_modified_relation")
        == "4*(1/2)^-2*sin((2*pi/3)/2)^2=12"
        and exact.get("continuous_frequency") == "omega=2*sqrt(3)"
        and exact.get("leapfrog_numerical_frequency") == "omega_tilde=4*pi/3"
        and exact.get("frequencies_identified") is False,
        "leapfrog exact fixture drift",
    )
    attainment = report.get("attainment", {})
    required_true = {
        "finite_declared_B_exact",
        "weighted_adjoint_exact",
        "K_equals_B_star_B_exact",
        "J_B_skew_adjoint_exact",
        "J_B_square_exact",
        "continuous_coordinate_equation_exact",
        "frozen_spatial_eigenvalue_is_omega_squared_conditionally",
        "conditional_auxiliary_time_evolution_supplied",
    }
    required_false = {
        "B_source_selected",
        "phase_norm_identified_with_physical_energy",
        "physical_clock_selected",
        "lorentz_or_boost_law_derived",
        "physical_field_sector_selected",
        "continuum_limit_derived",
        "physical_scale_selected",
        "physical_readout_selected",
        "physical_prediction_promoted",
        "comparison_permitted",
        "issue_655_closure_supported",
    }
    _require(
        set(attainment) == required_true | required_false
        and all(attainment.get(key) is True for key in required_true)
        and all(attainment.get(key) is False for key in required_false),
        "attainment boundary drift",
    )


def _verify_implementation_pins(report: Mapping[str, Any], root: Path) -> None:
    pins = report.get("implementation_pins")
    _require(isinstance(pins, list) and len(pins) == 3, "implementation pin drift")
    _require(
        {pin.get("path") for pin in pins if isinstance(pin, Mapping)}
        == EXPECTED_IMPLEMENTATION_PATHS,
        "implementation path set drift",
    )
    for pin in pins:
        _require(isinstance(pin, Mapping), "malformed implementation pin")
        relative = pin.get("path")
        _require(isinstance(relative, str), "implementation path is not text")
        path = root / relative
        raw = path.read_bytes()
        _require(pin.get("bytes") == len(raw), f"implementation bytes drift: {relative}")
        _require(pin.get("sha256") == _raw_sha(path), f"implementation hash drift: {relative}")


def verify_receipt(
    path: Path = DEFAULT_RECEIPT,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    report = _strict_load(path)
    expected_top_level = {
        "schema",
        "status",
        "issue",
        "comparison_data_read",
        "target_data_read",
        "target_data_paths",
        "parent_pin",
        "declared_finite_factorization",
        "frozen_operator_binding",
        "direct_factorization_uniqueness",
        "continuous_time_lift",
        "time_law_uniqueness_from_energy",
        "discrete_time_audit",
        "attainment",
        "implementation_pins",
        "claim_boundary",
        "receipt_sha256",
    }
    _require(set(report) == expected_top_level, "top-level schema drift")
    payload = copy.deepcopy(report)
    digest = payload.pop("receipt_sha256", None)
    _require(digest == _sha(payload), "receipt self-hash drift")
    _require(
        report.get("schema") == SCHEMA
        and report.get("status") == STATUS
        and report.get("issue") == 655,
        "receipt header drift",
    )
    _require(
        report.get("comparison_data_read") is False
        and report.get("target_data_read") is False
        and report.get("target_data_paths") == [],
        "target/comparison boundary drift",
    )
    _verify_parent(report, root)
    _verify_serialized_fixture(report)
    _verify_orientation_uniqueness(report)
    _verify_energy_uniqueness(report)
    _verify_boundaries(report)
    _verify_implementation_pins(report, root)
    _require(
        report.get("claim_boundary")
        == (
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
        "claim boundary drift",
    )
    return {
        "schema": "oph.fz11-conservative-time-lift-independent-verification.v1",
        "receipt": True,
        "exact_rational_fixture": True,
        "checked_J_dimension": 14,
        "checked_laurent_terms": 13,
        "declared_orientation_permutation_presentations": 46080,
        "checked_orientation_generator_fixture": True,
        "energy_uniqueness_mutation_rejected": True,
        "continuous_relation": "omega^2=lambda",
        "leapfrog_relation": "4*h^-2*sin(theta/2)^2=lambda",
        "producer_imported": False,
        "comparison_data_read": False,
        "physical_promotion": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.receipt)
    except (IndependentVerificationError, OSError, ValueError) as error:
        result = {
            "schema": "oph.fz11-conservative-time-lift-independent-verification.v1",
            "receipt": False,
            "reasons": [str(error)],
            "comparison_data_read": False,
            "physical_promotion": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
