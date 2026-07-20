"""Exact-rational audit of an interval contraction certificate.

The producer supplies outward-rounded residual and Jacobian intervals.  This
module does not trust a producer rank or Krawczyk Boolean.  It recomputes the
preconditioned interval image and the infinity-norm contraction bound using
exact rational arithmetic.  A separate primitive verifier must still establish
that the supplied intervals enclose the physical evaluator on the whole box.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable


Interval = tuple[Fraction, Fraction]


class IntervalCertificateError(ValueError):
    """Raised when a supplied interval packet is malformed."""


def _fraction(value: Any) -> Fraction:
    if not isinstance(value, str):
        raise IntervalCertificateError("proof-critical numbers must be decimal strings")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise IntervalCertificateError(f"invalid rational value: {value!r}") from exc


def _interval(value: Any) -> Interval:
    if not isinstance(value, dict) or set(value) != {"lower", "upper"}:
        raise IntervalCertificateError("intervals require exactly lower and upper")
    lower = _fraction(value["lower"])
    upper = _fraction(value["upper"])
    if lower > upper:
        raise IntervalCertificateError("interval lower endpoint exceeds upper endpoint")
    return lower, upper


def _add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def _neg(value: Interval) -> Interval:
    return -value[1], -value[0]


def _sub(left: Interval, right: Interval) -> Interval:
    return _add(left, _neg(right))


def _scale(scalar: Fraction, value: Interval) -> Interval:
    if scalar >= 0:
        return scalar * value[0], scalar * value[1]
    return scalar * value[1], scalar * value[0]


def _mul(left: Interval, right: Interval) -> Interval:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _sum(values: Iterable[Interval]) -> Interval:
    total = (Fraction(0), Fraction(0))
    for value in values:
        total = _add(total, value)
    return total


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    size = len(matrix)
    work = [row[:] for row in matrix]
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            factor = work[row][column] / pivot_value
            for index in range(column + 1, size):
                work[row][index] -= factor * work[column][index]
    return determinant


def _text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _interval_payload(value: Interval) -> dict[str, str]:
    return {"lower": _text(value[0]), "upper": _text(value[1])}


def verify_interval_contraction(
    augmented_system: dict[str, Any],
    *,
    physical_dimension: int,
) -> dict[str, Any]:
    """Recompute the strong interval contraction receipt.

    The returned receipt passes only when the selected system is square, the
    rational preconditioner is invertible, its interval fixed-point image lies
    strictly inside the source box, the induced infinity-norm bound is below
    one, and the selected rows have a supplied proof of closure to the full
    physical system.
    """

    blockers: list[str] = []
    if augmented_system.get("status") != "SUPPLIED":
        return {
            "interval_contraction_receipt": False,
            "rank_receipt": False,
            "existence_receipt": False,
            "local_isolation_receipt": False,
            "dimension": physical_dimension,
            "blockers": ["augmented_system_not_supplied"],
        }

    try:
        selected_rows = augmented_system["selected_square_row_ids"]
        all_rows = augmented_system["all_row_ids"]
        if len(selected_rows) != physical_dimension:
            blockers.append("selected_system_not_square")
        if not set(selected_rows).issubset(set(all_rows)):
            blockers.append("selected_rows_not_in_augmented_row_registry")

        raw_box = augmented_system["box"]
        raw_center = augmented_system["center"]
        raw_residual = augmented_system["residual_interval_at_center"]
        raw_jacobian = augmented_system["jacobian_interval"]
        raw_preconditioner = augmented_system["preconditioner"]
        if any(value is None for value in (raw_box, raw_center, raw_residual, raw_jacobian, raw_preconditioner)):
            raise IntervalCertificateError("supplied augmented system has null numerical fields")

        box = [_interval(value) for value in raw_box]
        center = [_fraction(value) for value in raw_center]
        residual = [_interval(value) for value in raw_residual]
        jacobian = [[_interval(value) for value in row] for row in raw_jacobian]
        preconditioner = [[_fraction(value) for value in row] for row in raw_preconditioner]

        if not (
            len(box)
            == len(center)
            == len(residual)
            == len(jacobian)
            == len(preconditioner)
            == physical_dimension
        ):
            raise IntervalCertificateError("interval certificate vector dimension mismatch")
        if any(len(row) != physical_dimension for row in jacobian):
            raise IntervalCertificateError("Jacobian enclosure is not square")
        if any(len(row) != physical_dimension for row in preconditioner):
            raise IntervalCertificateError("preconditioner is not square")
        if any(not (interval[0] <= point <= interval[1]) for interval, point in zip(box, center, strict=True)):
            blockers.append("center_outside_source_box")

        determinant = _determinant(preconditioner)
        if determinant == 0:
            blockers.append("preconditioner_singular")

        identity_minus_preconditioned_jacobian: list[list[Interval]] = []
        for row in range(physical_dimension):
            output_row: list[Interval] = []
            for column in range(physical_dimension):
                product = _sum(
                    _scale(preconditioner[row][inner], jacobian[inner][column])
                    for inner in range(physical_dimension)
                )
                identity = (Fraction(int(row == column)),) * 2
                output_row.append(_sub(identity, product))
            identity_minus_preconditioned_jacobian.append(output_row)

        row_bounds = [
            sum(max(abs(value[0]), abs(value[1])) for value in row)
            for row in identity_minus_preconditioned_jacobian
        ]
        contraction_bound = max(row_bounds, default=Fraction(0))
        if contraction_bound >= 1:
            blockers.append("interval_contraction_bound_not_below_one")

        delta_box = [
            (interval[0] - point, interval[1] - point)
            for interval, point in zip(box, center, strict=True)
        ]
        image: list[Interval] = []
        for row in range(physical_dimension):
            preconditioned_residual = _sum(
                _scale(preconditioner[row][inner], residual[inner])
                for inner in range(physical_dimension)
            )
            constant = _sub((center[row], center[row]), preconditioned_residual)
            linear = _sum(
                _mul(identity_minus_preconditioned_jacobian[row][column], delta_box[column])
                for column in range(physical_dimension)
            )
            image.append(_add(constant, linear))

        strict_interior = all(
            source[0] < target[0] and target[1] < source[1]
            for source, target in zip(box, image, strict=True)
        )
        if not strict_interior:
            blockers.append("interval_image_not_strictly_inside_source_box")

        closure = augmented_system.get("full_system_closure", {})
        closure_method = closure.get("method")
        closure_evidence = closure.get("evidence_artifact_id")
        if closure_method == "not_supplied" or not closure_evidence:
            blockers.append("selected_rows_not_closed_to_full_system")

        passed = not blockers
        return {
            "interval_contraction_receipt": passed,
            "rank_receipt": passed,
            "existence_receipt": passed,
            "local_isolation_receipt": passed,
            "dimension": physical_dimension,
            "preconditioner_determinant": _text(determinant),
            "contraction_bound_infinity_norm": _text(contraction_bound),
            "interval_image": [_interval_payload(value) for value in image],
            "strict_interior_inclusion": strict_interior,
            "full_system_closure_method": closure_method,
            "blockers": blockers,
        }
    except (KeyError, TypeError, IntervalCertificateError) as exc:
        return {
            "interval_contraction_receipt": False,
            "rank_receipt": False,
            "existence_receipt": False,
            "local_isolation_receipt": False,
            "dimension": physical_dimension,
            "blockers": [f"malformed_interval_certificate:{exc}"],
        }
