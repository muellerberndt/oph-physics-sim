"""Decide what the twelve-port geometry fixes about angular transfer.

The experiment stays on the source side.  It constructs the twelve persistent
icosahedral ports from the simulator geometry and compares several linear
extensions of port data to the sphere.  In particular, it gives a smooth
same-codomain counterfamily: every member has the same constant port samples
and the same spherical mean, while its degree-six and degree-ten content
varies continuously.

That counterfamily is a constructive non-identifiability result.  It does not
select a physical sky map, and it does not compare against observational data.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import special as scipy_special

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_equivariance_report,
    icosahedral_a5_port_permutations,
)


_TOLERANCE = 5.0e-11
_COUNTERFAMILY_DEGREES = (6, 10)
_COMB_RAY_DEGREES = (6, 10, 12)
_EPSILON_SAMPLES = (-1.0, -0.25, 0.0, 0.5, 1.0)
_MAX_EXACT_DEGREE = 14
_MIN_PROBE_COUNT = 32
_MAX_PROBE_COUNT = 4096
_MIN_PROBE_SEED = 0
_MAX_PROBE_SEED = 2**32 - 1
REPORT_SCHEMA = "oph.source_angular_transfer_decision.v1"
VERIFICATION_SCHEMA = "oph.source_angular_transfer_decision_verification.v1"


def _fraction_payload(value: Fraction) -> dict[str, int | str]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "text": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
    }


def _legendre_at_inverse_sqrt_five(degree: int) -> tuple[Fraction, Fraction]:
    """Return ``P_degree(1/sqrt(5))`` as ``a + b/sqrt(5)``."""

    if int(degree) != degree or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    p_previous = (Fraction(1), Fraction(0))
    if degree == 0:
        return p_previous
    p_current = (Fraction(0), Fraction(1))
    if degree == 1:
        return p_current
    for order in range(1, int(degree)):
        rational, radical = p_current
        multiplied_by_inverse_sqrt_five = (radical / 5, rational)
        p_next = (
            (
                (2 * order + 1) * multiplied_by_inverse_sqrt_five[0]
                - order * p_previous[0]
            )
            / (order + 1),
            (
                (2 * order + 1) * multiplied_by_inverse_sqrt_five[1]
                - order * p_previous[1]
            )
            / (order + 1),
        )
        p_previous, p_current = p_current, p_next
    return p_current


def exact_equal_port_comb_moment(degree: int) -> Fraction:
    r"""Return the exact normalized equal-port pair moment.

    The normalization is

    .. math::

       I_\ell = {1\over 12^2}\sum_{i,j=1}^{12}
                P_\ell(v_i\mathbin{\cdot}v_j).

    Every port sees one self-pair, one antipodal pair, five pairs with dot
    product ``1/sqrt(5)``, and five with dot product ``-1/sqrt(5)``.
    """

    if int(degree) != degree or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if degree % 2:
        return Fraction(0)
    rational, radical = _legendre_at_inverse_sqrt_five(int(degree))
    if radical:
        raise AssertionError("even Legendre value must be rational")
    return (1 + 5 * rational) / 6


def _harmonic_matrix(points: np.ndarray, maximum_degree: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    polar = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * math.pi)
    columns = [
        _spherical_harmonic(degree, order, polar, azimuth)
        for degree in range(maximum_degree + 1)
        for order in range(-degree, degree + 1)
    ]
    return np.stack(columns, axis=1)


def _spherical_harmonic(
    degree: int,
    order: int,
    polar: np.ndarray,
    azimuth: np.ndarray,
) -> np.ndarray:
    """Use the SciPy 1.15 API with the older argument-order fallback."""

    modern = getattr(scipy_special, "sph_harm_y", None)
    if modern is not None:
        return np.asarray(modern(degree, order, polar, azimuth))
    legacy = getattr(scipy_special, "sph_harm")
    return np.asarray(legacy(order, degree, azimuth, polar))


def _bandlimited_values(
    ports: np.ndarray,
    port_values: np.ndarray,
    query_points: np.ndarray,
    *,
    maximum_degree: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    evaluation = _harmonic_matrix(ports, maximum_degree)
    coefficients = np.linalg.pinv(evaluation, rcond=1.0e-13) @ np.asarray(
        port_values, dtype=float
    )
    values = _harmonic_matrix(query_points, maximum_degree) @ coefficients
    residual = float(
        np.max(np.abs(evaluation @ coefficients - np.asarray(port_values)))
    )
    return values, coefficients, residual, int(np.linalg.matrix_rank(evaluation))


def _zonal_port_sum(
    query_points: np.ndarray,
    ports: np.ndarray,
    degree: int,
) -> np.ndarray:
    dot_products = np.clip(
        np.asarray(query_points, dtype=float) @ np.asarray(ports, dtype=float).T,
        -1.0,
        1.0,
    )
    return np.mean(scipy_special.eval_legendre(int(degree), dot_products), axis=1)


def _counterfunction(query_points: np.ndarray, ports: np.ndarray) -> np.ndarray:
    moment_six = float(exact_equal_port_comb_moment(6))
    moment_ten = float(exact_equal_port_comb_moment(10))
    return (
        _zonal_port_sum(query_points, ports, 6) / moment_six
        - _zonal_port_sum(query_points, ports, 10) / moment_ten
    )


def _rotation_from_port_permutation(
    ports: np.ndarray,
    permutation: Sequence[int],
) -> tuple[np.ndarray, float, float]:
    permutation_array = np.asarray(permutation, dtype=np.int64)
    rotation_transpose, *_ = np.linalg.lstsq(
        ports,
        ports[permutation_array],
        rcond=None,
    )
    rotation = rotation_transpose.T
    coordinate_residual = float(
        np.max(np.abs(ports @ rotation.T - ports[permutation_array]))
    )
    orthogonality_residual = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    return rotation, coordinate_residual, orthogonality_residual


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=float)
    unit_axis /= np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    cross = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(unit_axis, unit_axis)
        + math.sin(angle) * cross
    )


def _unit_probes(seed: int, count: int) -> np.ndarray:
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not _MIN_PROBE_SEED <= seed <= _MAX_PROBE_SEED
    ):
        raise ValueError(
            f"seed must be an integer in [{_MIN_PROBE_SEED}, {_MAX_PROBE_SEED}]"
        )
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not _MIN_PROBE_COUNT <= count <= _MAX_PROBE_COUNT
    ):
        raise ValueError(
            "probe_count must be an integer in "
            f"[{_MIN_PROBE_COUNT}, {_MAX_PROBE_COUNT}]"
        )
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(count, 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    return probes


def angular_transfer_decision_report(
    *,
    seed: int = 643,
    probe_count: int = 512,
) -> dict[str, Any]:
    """Build the deterministic source-side angular-transfer verdict."""

    mesh = build_geodesic_icosahedral_tower(0).levels[0]
    ports = np.asarray(mesh.vertices, dtype=float)
    probes = _unit_probes(seed, probe_count)
    gram = np.clip(ports @ ports.T, -1.0, 1.0)
    inverse_sqrt_five = 1.0 / math.sqrt(5.0)
    dot_classes = (-1.0, -inverse_sqrt_five, inverse_sqrt_five, 1.0)
    nearest_classes = np.argmin(
        np.abs(gram[..., None] - np.asarray(dot_classes)[None, None, :]),
        axis=2,
    )
    maximum_dot_class_residual = float(
        np.max(np.abs(gram - np.asarray(dot_classes, dtype=float)[nearest_classes]))
    )
    dot_class_counts = {
        "-1": int(np.count_nonzero(nearest_classes == 0)),
        "-1/sqrt(5)": int(np.count_nonzero(nearest_classes == 1)),
        "1/sqrt(5)": int(np.count_nonzero(nearest_classes == 2)),
        "1": int(np.count_nonzero(nearest_classes == 3)),
    }
    geometry_valid = bool(
        mesh.vertex_count == 12
        and dot_class_counts
        == {
            "-1": 12,
            "-1/sqrt(5)": 60,
            "1/sqrt(5)": 60,
            "1": 12,
        }
        and maximum_dot_class_residual <= _TOLERANCE
    )

    exact_moments = {
        degree: exact_equal_port_comb_moment(degree)
        for degree in range(_MAX_EXACT_DEGREE + 1)
    }
    numerical_moments = {
        degree: float(np.mean(scipy_special.eval_legendre(degree, gram)))
        for degree in range(_MAX_EXACT_DEGREE + 1)
    }
    moment_residual = float(
        max(
            abs(numerical_moments[degree] - float(exact_moments[degree]))
            for degree in exact_moments
        )
    )
    ray_base = exact_moments[6]
    normalized_comb_ray = {
        degree: exact_moments[degree] / ray_base for degree in _COMB_RAY_DEGREES
    }
    exact_comb_check = bool(
        exact_moments[6] == Fraction(11, 25)
        and exact_moments[10] == Fraction(247, 1875)
        and exact_moments[12] == Fraction(1071, 3125)
        and exact_moments[14] == 0
        and moment_residual <= _TOLERANCE
    )

    constant_ports = np.ones(12, dtype=float)
    _, constant_coefficients, interpolation_residual, interpolation_rank = (
        _bandlimited_values(ports, constant_ports, ports)
    )
    nonconstant_coefficient_norm = float(np.linalg.norm(constant_coefficients[1:]))
    deterministic_port_values = np.linspace(-1.25, 1.5, 12)
    base_bandlimited, _, general_interpolation_residual, _ = _bandlimited_values(
        ports,
        deterministic_port_values,
        probes,
    )

    base_counterfunction = _counterfunction(probes, ports)
    port_counterfunction = _counterfunction(ports, ports)
    counterfunction_port_residual = float(np.max(np.abs(port_counterfunction)))
    base_bandlimited_at_ports, _, _, _ = _bandlimited_values(
        ports,
        deterministic_port_values,
        ports,
    )
    basis_port_values = np.eye(12, dtype=float)
    base_basis_bandlimited, _, basis_interpolation_residual, _ = _bandlimited_values(
        ports,
        basis_port_values,
        probes,
    )
    basis_means = np.mean(basis_port_values, axis=0)

    maximum_rotation_coordinate_residual = 0.0
    maximum_rotation_orthogonality_residual = 0.0
    maximum_rotation_determinant_residual = 0.0
    minimum_rotation_determinant = math.inf
    maximum_bandlimited_equivariance_residual = 0.0
    maximum_operator_family_equivariance_residual = 0.0
    maximum_counterfunction_invariance_residual = 0.0
    maximum_comb_moment_rotation_residual = 0.0
    permutations = icosahedral_a5_port_permutations()
    for permutation_tuple in permutations:
        permutation = np.asarray(permutation_tuple, dtype=np.int64)
        rotation, coordinate_residual, orthogonality_residual = (
            _rotation_from_port_permutation(ports, permutation)
        )
        maximum_rotation_coordinate_residual = max(
            maximum_rotation_coordinate_residual,
            coordinate_residual,
        )
        maximum_rotation_orthogonality_residual = max(
            maximum_rotation_orthogonality_residual,
            orthogonality_residual,
        )
        determinant = float(np.linalg.det(rotation))
        maximum_rotation_determinant_residual = max(
            maximum_rotation_determinant_residual,
            abs(determinant - 1.0),
        )
        minimum_rotation_determinant = min(
            minimum_rotation_determinant,
            determinant,
        )
        rotated_port_values = np.empty_like(deterministic_port_values)
        rotated_port_values[permutation] = deterministic_port_values
        rotated_bandlimited, _, _, _ = _bandlimited_values(
            ports,
            rotated_port_values,
            probes @ rotation.T,
        )
        maximum_bandlimited_equivariance_residual = max(
            maximum_bandlimited_equivariance_residual,
            float(np.max(np.abs(rotated_bandlimited - base_bandlimited))),
        )
        rotated_counterfunction = _counterfunction(
            probes @ rotation.T,
            ports,
        )
        port_action = np.zeros((12, 12), dtype=float)
        port_action[permutation, np.arange(12)] = 1.0
        rotated_basis_bandlimited, _, _, _ = _bandlimited_values(
            ports,
            port_action,
            probes @ rotation.T,
        )
        rotated_basis_means = np.mean(port_action, axis=0)
        for epsilon in _EPSILON_SAMPLES:
            base_operator = base_basis_bandlimited + epsilon * np.outer(
                base_counterfunction,
                basis_means,
            )
            rotated_operator = rotated_basis_bandlimited + epsilon * np.outer(
                rotated_counterfunction,
                rotated_basis_means,
            )
            maximum_operator_family_equivariance_residual = max(
                maximum_operator_family_equivariance_residual,
                float(np.max(np.abs(rotated_operator - base_operator))),
            )
        maximum_counterfunction_invariance_residual = max(
            maximum_counterfunction_invariance_residual,
            float(np.max(np.abs(rotated_counterfunction - base_counterfunction))),
        )
        rotated_gram = gram[np.ix_(permutation, permutation)]
        for degree in _COMB_RAY_DEGREES:
            rotated_moment = float(
                np.mean(scipy_special.eval_legendre(degree, rotated_gram))
            )
            maximum_comb_moment_rotation_residual = max(
                maximum_comb_moment_rotation_residual,
                abs(rotated_moment - numerical_moments[degree]),
            )

    arbitrary_rotation = _axis_angle_rotation(
        np.asarray([1.0, 2.0, 3.0]),
        0.731,
    )
    rotated_ports = ports @ arbitrary_rotation.T
    rotated_probes = probes @ arbitrary_rotation.T
    arbitrary_rotated_bandlimited, _, _, _ = _bandlimited_values(
        rotated_ports,
        deterministic_port_values,
        rotated_probes,
    )
    arbitrary_rotation_bandlimited_residual = float(
        np.max(np.abs(arbitrary_rotated_bandlimited - base_bandlimited))
    )
    arbitrary_rotation_counterfunction_residual = float(
        np.max(
            np.abs(
                _counterfunction(rotated_probes, rotated_ports) - base_counterfunction
            )
        )
    )
    arbitrary_rotation_comb_residual = float(
        max(
            abs(
                float(
                    np.mean(
                        scipy_special.eval_legendre(
                            degree,
                            rotated_ports @ rotated_ports.T,
                        )
                    )
                )
                - numerical_moments[degree]
            )
            for degree in _COMB_RAY_DEGREES
        )
    )

    a5_report = icosahedral_a5_equivariance_report(0)
    equivariance_valid = bool(
        len(permutations) == 60
        and a5_report["A5_ROTATION_GROUP_ORDER_60_RECEIPT"] is True
        and maximum_rotation_coordinate_residual <= _TOLERANCE
        and maximum_rotation_orthogonality_residual <= _TOLERANCE
        and maximum_rotation_determinant_residual <= _TOLERANCE
        and minimum_rotation_determinant > 0.0
        and maximum_bandlimited_equivariance_residual <= _TOLERANCE
        and maximum_operator_family_equivariance_residual <= _TOLERANCE
        and maximum_counterfunction_invariance_residual <= _TOLERANCE
        and maximum_comb_moment_rotation_residual <= _TOLERANCE
        and arbitrary_rotation_bandlimited_residual <= _TOLERANCE
        and arbitrary_rotation_counterfunction_residual <= _TOLERANCE
        and arbitrary_rotation_comb_residual <= _TOLERANCE
    )

    family_rows = []
    for epsilon in _EPSILON_SAMPLES:
        port_extension = 1.0 + epsilon * port_counterfunction
        probe_extension = 1.0 + epsilon * base_counterfunction
        family_rows.append(
            {
                "epsilon": epsilon,
                "maximum_port_sample_residual_from_one": float(
                    np.max(np.abs(port_extension - 1.0))
                ),
                "maximum_probe_residual_from_constant_extension": float(
                    np.max(np.abs(probe_extension - 1.0))
                ),
                "degree_6_zonal_sum_coefficient": epsilon / float(exact_moments[6]),
                "degree_10_zonal_sum_coefficient": -epsilon / float(exact_moments[10]),
            }
        )
    family_is_nontrivial = bool(
        max(
            row["maximum_probe_residual_from_constant_extension"] for row in family_rows
        )
        > 1.0e-4
    )
    operator_rows = []
    port_value_mean = float(np.mean(deterministic_port_values))
    for epsilon in _EPSILON_SAMPLES:
        operator_at_ports = (
            base_bandlimited_at_ports + epsilon * port_value_mean * port_counterfunction
        )
        operator_at_probes = (
            base_bandlimited + epsilon * port_value_mean * base_counterfunction
        )
        operator_rows.append(
            {
                "epsilon": epsilon,
                "maximum_right_inverse_residual": float(
                    np.max(np.abs(operator_at_ports - deterministic_port_values))
                ),
                "maximum_probe_difference_from_epsilon_zero": float(
                    np.max(np.abs(operator_at_probes - base_bandlimited))
                ),
            }
        )
    operator_family_valid = bool(
        all(
            row["maximum_right_inverse_residual"] <= _TOLERANCE for row in operator_rows
        )
        and max(
            row["maximum_probe_difference_from_epsilon_zero"] for row in operator_rows
        )
        > 1.0e-4
    )
    smooth_family_positivity_bound = Fraction(2717, 26800)
    smooth_counterfamily_valid = bool(
        counterfunction_port_residual <= _TOLERANCE
        and family_is_nontrivial
        and operator_family_valid
        and all(
            row["maximum_port_sample_residual_from_one"] <= _TOLERANCE
            for row in family_rows
        )
    )

    # A one-degree control makes the variable l=6 statistic explicit in closed
    # form.  Write H6=sum_i P6(n.v_i) and c=H6(v_j)=12 I6.  The unnormalized
    # family 1+epsilon(H6-c) keeps every port sample equal to one.  Dividing by
    # its mean gives the stated normalized l=6 statistic.  The two-degree
    # counterfamily above is stronger because it preserves both samples and
    # mean without that final normalization.
    h6_port_value = 12 * exact_moments[6]
    h6_mean_square = 144 * exact_moments[6] / 13
    h6_cl_prefactor = h6_mean_square / 13
    positivity_epsilon_bound = Fraction(25, 432)
    one_degree_rows = []
    for epsilon in (
        Fraction(-1, 20),
        Fraction(-1, 100),
        Fraction(0),
        Fraction(1, 100),
        Fraction(1, 20),
    ):
        mean = 1 - h6_port_value * epsilon
        normalized_statistic = h6_cl_prefactor * epsilon * epsilon / (mean * mean)
        one_degree_rows.append(
            {
                "epsilon": _fraction_payload(epsilon),
                "sphere_mean_before_normalization": _fraction_payload(mean),
                "normalized_degree_6_statistic": _fraction_payload(
                    normalized_statistic
                ),
                "strictly_inside_sufficient_positivity_bound": (
                    abs(epsilon) < positivity_epsilon_bound
                ),
            }
        )
    one_degree_control_valid = bool(
        h6_port_value == Fraction(132, 25)
        and h6_mean_square == Fraction(1584, 325)
        and h6_cl_prefactor == Fraction(1584, 4225)
        and max(
            abs(
                float(12 * _zonal_port_sum(ports, ports, 6)[index])
                - float(h6_port_value)
            )
            for index in range(12)
        )
        <= _TOLERANCE
        and all(
            row["strictly_inside_sufficient_positivity_bound"] is True
            for row in one_degree_rows
        )
    )

    instrument_valid = bool(
        geometry_valid
        and exact_comb_check
        and interpolation_rank == 12
        and interpolation_residual <= _TOLERANCE
        and general_interpolation_residual <= _TOLERANCE
        and basis_interpolation_residual <= _TOLERANCE
        and nonconstant_coefficient_norm <= _TOLERANCE
        and equivariance_valid
        and smooth_counterfamily_valid
        and one_degree_control_valid
    )
    nonidentifiability_witness = bool(instrument_valid)
    verdict = (
        "NONIDENTIFIABLE_WITHOUT_DYNAMICAL_TRANSFER_SELECTOR"
        if nonidentifiability_witness
        else "INSTRUMENT_INVALID_NO_SCIENTIFIC_VERDICT"
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "mode": "source_side_static_transfer_decision",
        "verdict": verdict,
        "strongest_allowed_claim": (
            "STATIC_TRANSFER_NONIDENTIFIABILITY"
            if nonidentifiability_witness
            else "NONE"
        ),
        "source_inputs": {
            "external_observational_data_used": False,
            "target_values_used": False,
            "geometry_constructor": (
                "oph_fpe.core.icosahedral.build_geodesic_icosahedral_tower(0)"
            ),
            "geometry_hash": mesh.geometry_hash,
            "port_count": mesh.vertex_count,
            "probe_seed": int(seed),
            "probe_count": int(probe_count),
        },
        "geometry_checks": {
            "dot_product_classes": [
                "-1",
                "-1/sqrt(5)",
                "1/sqrt(5)",
                "1",
            ],
            "ordered_pair_class_counts": dot_class_counts,
            "maximum_dot_class_residual": maximum_dot_class_residual,
            "regular_icosahedral_port_geometry": geometry_valid,
        },
        "equal_port_comb": {
            "definition": ("I_l=(1/144) sum_ij P_l(v_i dot v_j)"),
            "exact_moments": {
                str(degree): _fraction_payload(exact_moments[degree])
                for degree in range(_MAX_EXACT_DEGREE + 1)
            },
            "numerical_moments": {
                str(degree): numerical_moments[degree]
                for degree in range(_MAX_EXACT_DEGREE + 1)
            },
            "maximum_exact_numeric_residual": moment_residual,
            "normalized_degree_6_10_12_ray_relative_to_degree_6": {
                str(degree): _fraction_payload(normalized_comb_ray[degree])
                for degree in _COMB_RAY_DEGREES
            },
            "exact_degree_6_moment_is_11_over_25": (
                exact_moments[6] == Fraction(11, 25)
            ),
            "exact_comb_moment_check": exact_comb_check,
        },
        "minimum_norm_bandlimited_interpolation": {
            "definition": (
                "Moore-Penrose minimum-L2 coefficient extension in the "
                "full spherical-harmonic space l<=3"
            ),
            "evaluation_matrix_shape": [12, 16],
            "evaluation_rank": interpolation_rank,
            "constant_port_interpolation_residual": interpolation_residual,
            "constant_extension_nonconstant_coefficient_norm": (
                nonconstant_coefficient_norm
            ),
            "degree_6_moment": _fraction_payload(Fraction(0)),
            "degree_6_absence_reason": "codomain_is_bandlimited_to_l_at_most_3",
            "canonical_only_relative_to_declared_L2_metric": True,
        },
        "smooth_same_codomain_counterfamily": {
            "definition": (
                "h_l(n)=(1/12) sum_i P_l(n dot v_i); "
                "g=h_6/I_6-h_10/I_10; f_epsilon=1+epsilon*g"
            ),
            "counterfunction_degrees": list(_COUNTERFAMILY_DEGREES),
            "exact_h6_value_at_every_port": _fraction_payload(exact_moments[6]),
            "exact_h10_value_at_every_port": _fraction_payload(exact_moments[10]),
            "maximum_counterfunction_port_residual": (counterfunction_port_residual),
            "sphere_mean_preserved_exactly": True,
            "sphere_mean_reason": ("each zonal P_l term has zero sphere mean for l>0"),
            "A5_invariant_by_equal_orbit_sum": True,
            "sufficient_positivity_bound_on_abs_epsilon": _fraction_payload(
                smooth_family_positivity_bound
            ),
            "positivity_bound_reason": (
                "|g|<=1/I6+1/I10=26800/2717 because |P_l(x)|<=1 on [-1,1]"
            ),
            "strictly_positive_subfamily_exists": True,
            "epsilon_rows": family_rows,
            "linear_transfer_operator_family": {
                "definition": (
                    "T_epsilon(q)=T_0(q)+epsilon*mean(q)*g, where "
                    "T_0 is the declared l<=3 minimum-L2 right inverse"
                ),
                "linearity": True,
                "A5_equivariance": True,
                "same_sphere_mean_as_T0": True,
                "right_inverse_rows": operator_rows,
                "checks_pass": operator_family_valid,
            },
            "continuous_nontrivial_same_sample_family": (smooth_counterfamily_valid),
        },
        "exact_one_degree_control_family": {
            "definition": (
                "H6(n)=sum_i P6(n dot v_i); F_epsilon=1+epsilon*(H6-132/25)"
            ),
            "sphere_measure_convention": "dmu=dOmega/(4*pi), so integral_S2 dmu=1",
            "normalized_degree_6_statistic_definition": (
                "S6(f)=(1/13)*integral_S2 "
                "|Pi_6(f/integral_S2(f*dmu))|^2*dmu"
            ),
            "harmonic_projector_convention": (
                "Pi_6 is the degree-six projector orthogonal for normalized "
                "sphere measure dmu"
            ),
            "exact_H6_value_at_every_port": _fraction_payload(h6_port_value),
            "exact_normalized_sphere_integral_H6_squared": (
                _fraction_payload(h6_mean_square)
            ),
            "exact_degree_6_statistic_prefactor": _fraction_payload(h6_cl_prefactor),
            "normalized_degree_6_statistic_formula": (
                "1584*epsilon^2/(4225*(1-(132/25)*epsilon)^2)"
            ),
            "sufficient_positivity_bound_on_abs_epsilon": (
                _fraction_payload(positivity_epsilon_bound)
            ),
            "positivity_bound_reason": ("|H6-132/25|<=12+132/25=432/25"),
            "epsilon_rows": one_degree_rows,
            "checks_pass": one_degree_control_valid,
        },
        "equivariance_and_rotation_checks": {
            "A5_rotation_count": len(permutations),
            "maximum_A5_coordinate_residual": (maximum_rotation_coordinate_residual),
            "maximum_A5_orthogonality_residual": (
                maximum_rotation_orthogonality_residual
            ),
            "minimum_A5_rotation_determinant": minimum_rotation_determinant,
            "maximum_A5_rotation_determinant_residual_from_one": (
                maximum_rotation_determinant_residual
            ),
            "maximum_bandlimited_A5_equivariance_residual": (
                maximum_bandlimited_equivariance_residual
            ),
            "operator_equivariance_input_basis": (
                "all_12_canonical_port_basis_vectors"
            ),
            "operator_equivariance_epsilon_samples": list(_EPSILON_SAMPLES),
            "operator_equivariance_input_basis_count": 12,
            "maximum_transfer_operator_family_A5_equivariance_residual": (
                maximum_operator_family_equivariance_residual
            ),
            "maximum_counterfunction_A5_invariance_residual": (
                maximum_counterfunction_invariance_residual
            ),
            "maximum_comb_A5_moment_residual": (maximum_comb_moment_rotation_residual),
            "arbitrary_rotation_bandlimited_covariance_residual": (
                arbitrary_rotation_bandlimited_residual
            ),
            "arbitrary_rotation_counterfunction_covariance_residual": (
                arbitrary_rotation_counterfunction_residual
            ),
            "arbitrary_rotation_comb_moment_residual": (
                arbitrary_rotation_comb_residual
            ),
            "checks_pass": equivariance_valid,
        },
        "decision_gates": {
            "instrument_valid": instrument_valid,
            "static_port_geometry_and_equivariance_select_unique_transfer": (False),
            "smooth_same_sample_counterfamily_constructed": (
                nonidentifiability_witness
            ),
            "dynamical_transfer_selector_supplied": False,
            "screen_to_sky_identification_supplied": False,
            "physical_angular_prediction_receipt": False,
        },
        "claim_boundary": (
            "The exact twelve-port geometry, A5 equivariance, fixed constant "
            "port samples, and fixed spherical mean do not select a unique "
            "smooth angular extension. A repair/readback dynamics may select "
            "one map, but no such selector is part of this experiment. The "
            "receipt therefore supplies no CMB or laboratory prediction."
        ),
    }
    report["certificate_payload_sha256"] = _payload_sha256(report)
    return report


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(report: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(report))
    payload.pop("certificate_payload_sha256", None)
    return (
        "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    )


def verify_angular_transfer_decision(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the source experiment and fail closed on malformed payloads."""

    reasons: list[str] = []
    if not isinstance(report, Mapping):
        return {
            "schema": VERIFICATION_SCHEMA,
            "receipt": False,
            "reasons": ["report_is_not_a_mapping"],
        }
    if report.get("schema") != REPORT_SCHEMA:
        reasons.append("schema_mismatch")
    source_inputs = report.get("source_inputs")
    if not isinstance(source_inputs, Mapping):
        reasons.append("source_inputs_missing_or_not_mapping")
        seed = 643
        probe_count = 512
    else:
        seed = source_inputs.get("probe_seed")
        probe_count = source_inputs.get("probe_count")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not _MIN_PROBE_SEED <= seed <= _MAX_PROBE_SEED
        ):
            reasons.append("probe_seed_missing_or_out_of_bounds")
            seed = 643
        if (
            not isinstance(probe_count, int)
            or isinstance(probe_count, bool)
            or not _MIN_PROBE_COUNT <= probe_count <= _MAX_PROBE_COUNT
        ):
            reasons.append("probe_count_missing_or_out_of_bounds")
            probe_count = 512

    stated_hash = report.get("certificate_payload_sha256")
    try:
        computed_hash = _payload_sha256(report)
    except (TypeError, ValueError, OverflowError, RecursionError):
        computed_hash = None
        reasons.append("payload_is_not_finite_canonical_json")
    if not isinstance(stated_hash, str) or stated_hash != computed_hash:
        reasons.append("payload_hash_mismatch")

    expected = angular_transfer_decision_report(
        seed=int(seed),
        probe_count=int(probe_count),
    )
    try:
        submitted_json = _canonical_json(dict(report))
    except (TypeError, ValueError, OverflowError, RecursionError):
        submitted_json = None
    if submitted_json is None or submitted_json != _canonical_json(expected):
        reasons.append("independent_recomputation_mismatch")

    gates = report.get("decision_gates")
    if not isinstance(gates, Mapping):
        reasons.append("decision_gates_missing_or_not_mapping")
    else:
        required_false = (
            "static_port_geometry_and_equivariance_select_unique_transfer",
            "dynamical_transfer_selector_supplied",
            "screen_to_sky_identification_supplied",
            "physical_angular_prediction_receipt",
        )
        if any(gates.get(name) is not False for name in required_false):
            reasons.append("forbidden_uniqueness_or_physical_promotion")
    allowed_claim = report.get("strongest_allowed_claim")
    instrument_valid = (
        gates.get("instrument_valid") if isinstance(gates, Mapping) else None
    )
    expected_claim = (
        "STATIC_TRANSFER_NONIDENTIFIABILITY"
        if instrument_valid is True
        else "NONE"
    )
    if allowed_claim != expected_claim:
        reasons.append("strongest_claim_does_not_fail_closed")

    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": not reasons,
        "reasons": reasons,
        "recomputed_payload_sha256": expected["certificate_payload_sha256"],
        "scope": "semantic_recomputation_of_source_only_angular_report",
    }


def write_angular_transfer_decision(
    path: Path,
    *,
    seed: int = 643,
    probe_count: int = 512,
) -> dict[str, Any]:
    """Write a canonical JSON source-side decision receipt."""

    report = angular_transfer_decision_report(
        seed=seed,
        probe_count=probe_count,
    )
    verification = verify_angular_transfer_decision(report)
    if verification["receipt"] is not True:
        raise RuntimeError(
            "internal angular-transfer verification failed: "
            + ",".join(verification["reasons"])
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether static twelve-port geometry uniquely fixes an "
            "angular transfer."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON receipt to this path; otherwise print it.",
    )
    parser.add_argument("--seed", type=int, default=643)
    parser.add_argument("--probe-count", type=int, default=512)
    arguments = parser.parse_args(argv)
    report = angular_transfer_decision_report(
        seed=arguments.seed,
        probe_count=arguments.probe_count,
    )
    verification = verify_angular_transfer_decision(report)
    if verification["receipt"] is not True:
        raise SystemExit(
            "internal angular-transfer verification failed: "
            + ",".join(verification["reasons"])
        )
    if arguments.output is None:
        print(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    else:
        write_angular_transfer_decision(
            arguments.output,
            seed=arguments.seed,
            probe_count=arguments.probe_count,
        )
    return 0 if report["decision_gates"]["instrument_valid"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
