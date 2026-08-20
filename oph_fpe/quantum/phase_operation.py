"""Declared PR-04 phase-sensitive effect on the simulator side.

Register row PR-04 of the research repository carries a recorded decision
(dated 2026-08-18, lane issue 730, disposition axiomatize): the exact phase
lift ``I/2 - (2*sqrt(3)/3)*i*(Q*P - P*Q)`` is a declared phase-sensitive
effect with exact matrix ``[[1/2, -i/2], [i/2, 1/2]]``, the Pauli +Y
projector.  This module carries that declaration on the simulator side:

* the committed standard two-dimensional S3 representation, rebuilt from the
  simulator's own ``oph_fpe.finite_groups`` tables and demanded equal to the
  committed literal matrices;
* the committed context web (the record projector and its six gauge
  conjugates) and the declared lift, demanded equal to the Pauli +Y
  projector, Hermitian, and idempotent;
* an explicitly exhaustive deterministic count semantics for a static
  semantic-conformance fixture on record-diagonal and committed-core
  matrices: every integer pair is a generated expected-frequency numerator
  for the exact Born weight of the context effect, scaled to the least
  positive integer multiple of the run mass that clears the denominator.
  No sampling, no randomness, and no floating-point arithmetic occurs
  anywhere in this module;
* the committed integer phase receipt window
  ``32041*(a - b)^2 <= 30192*(a + b)^2`` with both outcomes positive, applied
  fail-closed to every emitted phase count pair.  The committed coefficients
  read ``32041 = 179^2`` and ``30192 = 4*111*68``: on the committed core the
  window is the positive-semidefiniteness bound of the state family;
* a fail-closed replay comparison against the committed reference receipt
  ``code/phase_operation_producer/PHASE_OPERATION_RECEIPT.v1.json`` of the
  research repository.

All arithmetic is exact over ``Q(sqrt(3), i)``, represented as pairs of
``Q3 = a + b*sqrt(3)`` elements with ``Fraction`` coefficients, following the
``Q5``/``C5`` pattern of ``icosahedral_chsh_candidate``.

Boundaries: the effect is a declared architecture decision, never derived
from source dynamics; the integer pairs are generated expected-frequency
numerators, never measured.  This fixture is not an operation, instrument,
measurement, or validation.  No instrument is frozen by this module, no run
is evidential, INS-01 remains the controlling OL-A1 verdict, and OL-C5
status is untouched by this lane.  PR-64 owns completely-positive
trace-nonincreasing outcome maps, a trace-preserving summed channel, and
effect/readback compatibility; PR-65 owns common preparation, public
outcomes, provenance, and custody; PR-03 owns cross-context additivity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.charged_response import canonical_sha256
from oph_fpe.finite_groups import S3_ELEMENTS, S3_INDEX, S3_MUL


SCHEMA = "oph.sim.pr04_phase_operation_architecture.v1"
REGISTER_ROW = "PR-04"
REGISTER_DISPOSITION = "axiomatize"
DECISION_DATE = "2026-08-18"
LANE_ISSUE = 730

REFERENCE_RECEIPT_SCHEMA = "oph.receipt.pr04_phase_operation.v1"
REFERENCE_PAYLOAD_SHA256 = (
    "71a06f1c15192123cd09feb2386da702b572c8ac57c9b7633f5aa60c5d404e22"
)
REFERENCE_RUN_COMMIT = "b39b78faf894894ebe573571e0902ccfaaeac32a"
REFERENCE_RUN_ID = "b12_prereg_16k_20260806"
REFERENCE_SEED = 20260806

RUN_MASS = 179
RUN_COUNTS = (111, 68)

WINDOW_LHS_COEFFICIENT = 32041
WINDOW_RHS_COEFFICIENT = 30192

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_RECEIPT = (
    REPOSITORY_ROOT.parent
    / "reverse-engineering-reality"
    / "code"
    / "phase_operation_producer"
    / "PHASE_OPERATION_RECEIPT.v1.json"
)

DECISION_STATEMENT = (
    "Recorded decision on register row PR-04 (disposition axiomatize): the "
    "committed exact lift I/2 - (2*sqrt(3)/3)*i*(Q*P - P*Q) of "
    "code/born_context_phase_lift is a declared phase-sensitive effect. The "
    "information cost is one declared non-real effect with the exact "
    "matrix [[1/2, -i/2], [i/2, 1/2]], equal to the Pauli +Y projector."
)

SEMANTICS_DECLARATION = (
    "Explicitly exhaustive deterministic semantics for a static conformance "
    "fixture: every integer pair is a generated expected-frequency numerator "
    "for the exact Born weight of the context effect under the declared "
    "diagonal matrix diag(111/179, 68/179), computed in exact "
    "arithmetic over Q(sqrt(3), i) and scaled to the least positive integer "
    "multiple of the committed run mass 179 that makes both counts integers. "
    "No sampling, no randomness, no statistical estimate, and no "
    "floating-point arithmetic enters at any step."
)

PROVENANCE_STATEMENT = (
    "The phase effect is a declared architecture decision recorded on "
    "register row PR-04; it is never derived from source dynamics, repair "
    "records, or run data. The integer pairs are generated "
    "expected-frequency numerators of the declared exhaustive "
    "deterministic semantics, never measured and never sampled."
)


class PhaseOperationError(ValueError):
    """Typed fail-closed phase-effect fixture error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PhaseOperationError(code, message)


@dataclass(frozen=True)
class Q3:
    """Exact element ``a + b*sqrt(3)`` of the real field Q(sqrt(3))."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    @staticmethod
    def of(value: "Q3 | Fraction | int") -> "Q3":
        if isinstance(value, Q3):
            return value
        return Q3(Fraction(value), Fraction(0))

    def __add__(self, other: "Q3") -> "Q3":
        return Q3(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Q3") -> "Q3":
        return Q3(self.a - other.a, self.b - other.b)

    def __neg__(self) -> "Q3":
        return Q3(-self.a, -self.b)

    def __mul__(self, other: "Q3") -> "Q3":
        return Q3(
            self.a * other.a + 3 * self.b * other.b,
            self.a * other.b + self.b * other.a,
        )

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0


Q3_ZERO = Q3()


@dataclass(frozen=True)
class C3:
    """Exact element ``re + i*im`` of Q(sqrt(3), i) with ``re, im`` in Q3."""

    re: Q3 = Q3_ZERO
    im: Q3 = Q3_ZERO

    @staticmethod
    def of(value: "C3 | Q3 | Fraction | int") -> "C3":
        if isinstance(value, C3):
            return value
        return C3(Q3.of(value), Q3_ZERO)

    def __add__(self, other: "C3") -> "C3":
        return C3(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "C3") -> "C3":
        return C3(self.re - other.re, self.im - other.im)

    def __neg__(self) -> "C3":
        return C3(-self.re, -self.im)

    def __mul__(self, other: "C3") -> "C3":
        return C3(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def conjugate(self) -> "C3":
        return C3(self.re, -self.im)

    def is_zero(self) -> bool:
        return self.re.is_zero() and self.im.is_zero()

    def encode(self) -> list[str]:
        """The reference scalar encoding
        ``[re_rational, re_sqrt3, im_rational, im_sqrt3]`` over Q."""

        return [str(self.re.a), str(self.re.b), str(self.im.a), str(self.im.b)]


Mat = tuple[tuple[C3, C3], tuple[C3, C3]]


def mat(rows: Sequence[Sequence[Any]]) -> Mat:
    require(
        len(rows) == 2 and all(len(row) == 2 for row in rows),
        "MATRIX_SHAPE",
        "expected a 2x2 matrix",
    )
    return tuple(tuple(C3.of(entry) for entry in row) for row in rows)  # type: ignore[return-value]


IDENTITY = mat(((1, 0), (0, 1)))
RECORD_PROJECTOR = mat(((1, 0), (0, 0)))
PAULI_Y_PLUS_PROJECTOR = mat(
    (
        (Fraction(1, 2), C3(Q3_ZERO, Q3.of(Fraction(-1, 2)))),
        (C3(Q3_ZERO, Q3.of(Fraction(1, 2))), Fraction(1, 2)),
    )
)


def madd(x: Mat, y: Mat) -> Mat:
    return mat([[x[i][j] + y[i][j] for j in range(2)] for i in range(2)])


def msub(x: Mat, y: Mat) -> Mat:
    return mat([[x[i][j] - y[i][j] for j in range(2)] for i in range(2)])


def mmul(x: Mat, y: Mat) -> Mat:
    return mat(
        [
            [x[i][0] * y[0][j] + x[i][1] * y[1][j] for j in range(2)]
            for i in range(2)
        ]
    )


def mscale(scalar: C3, x: Mat) -> Mat:
    return mat([[scalar * x[i][j] for j in range(2)] for i in range(2)])


def mdagger(x: Mat) -> Mat:
    return mat([[x[j][i].conjugate() for j in range(2)] for i in range(2)])


def mtranspose(x: Mat) -> Mat:
    return mat([[x[j][i] for j in range(2)] for i in range(2)])


def mtrace(x: Mat) -> C3:
    return x[0][0] + x[1][1]


def encode_matrix(x: Mat) -> list[list[list[str]]]:
    return [[x[i][j].encode() for j in range(2)] for i in range(2)]


def _real_matrix(entries: Sequence[Sequence[tuple[str, str]]]) -> Mat:
    return mat(
        [
            [C3(Q3(Fraction(a), Fraction(b)), Q3_ZERO) for a, b in row]
            for row in entries
        ]
    )


# Committed literal matrices of the standard two-dimensional S3 irrep, in the
# simulator's S3_ELEMENTS order, each entry a pair (rational, sqrt(3)) over Q.
COMMITTED_IRREP_ENTRIES: tuple[tuple[tuple[tuple[str, str], ...], ...], ...] = (
    ((("1", "0"), ("0", "0")), (("0", "0"), ("1", "0"))),
    ((("1", "0"), ("0", "0")), (("0", "0"), ("-1", "0"))),
    ((("-1/2", "0"), ("0", "1/2")), (("0", "1/2"), ("1/2", "0"))),
    ((("-1/2", "0"), ("0", "-1/2")), (("0", "1/2"), ("-1/2", "0"))),
    ((("-1/2", "0"), ("0", "1/2")), (("0", "-1/2"), ("-1/2", "0"))),
    ((("-1/2", "0"), ("0", "-1/2")), (("0", "-1/2"), ("1/2", "0"))),
)

# Committed generator assignment of the reference payload:
# rho((1, 2, 0)) is the rotation by 120 degrees, rho((0, 2, 1)) is diag(1, -1).
GENERATOR_ASSIGNMENT = {
    (1, 2, 0): _real_matrix(COMMITTED_IRREP_ENTRIES[3]),
    (0, 2, 1): _real_matrix(COMMITTED_IRREP_ENTRIES[1]),
}

COMMITTED_COEFFICIENT = Q3(Fraction(0), Fraction(2, 3))


def standard_irrep() -> tuple[Mat, ...]:
    """The six committed irrep matrices, generated from the committed
    generator assignment over the simulator's own S3 composition table and
    demanded equal to the committed literals, orthogonal, and a
    homomorphism for all 36 pairs."""

    representations: dict[int, Mat] = {S3_INDEX[(0, 1, 2)]: IDENTITY}
    frontier = [S3_INDEX[(0, 1, 2)]]
    generators = {
        S3_INDEX[permutation]: matrix
        for permutation, matrix in GENERATOR_ASSIGNMENT.items()
    }
    while frontier:
        index = frontier.pop()
        for generator_index, generator_matrix in generators.items():
            image = int(S3_MUL[generator_index][index])
            if image not in representations:
                representations[image] = mmul(
                    generator_matrix, representations[index]
                )
                frontier.append(image)
    require(
        len(representations) == len(S3_ELEMENTS),
        "IRREP_CLOSURE",
        "generator closure does not reach every S3 element",
    )
    irrep = tuple(representations[index] for index in range(len(S3_ELEMENTS)))
    committed = tuple(_real_matrix(entries) for entries in COMMITTED_IRREP_ENTRIES)
    require(
        irrep == committed,
        "IRREP_DRIFT",
        "generated irrep differs from the committed literal matrices",
    )
    for left in range(len(S3_ELEMENTS)):
        for right in range(len(S3_ELEMENTS)):
            require(
                mmul(irrep[left], irrep[right])
                == irrep[int(S3_MUL[left][right])],
                "IRREP_HOMOMORPHISM",
                f"homomorphism fails on pair ({left}, {right})",
            )
    for index, representation in enumerate(irrep):
        require(
            mmul(representation, mtranspose(representation)) == IDENTITY,
            "IRREP_ORTHOGONALITY",
            f"element {index} is not orthogonal",
        )
    return irrep


def context_projectors(irrep: Sequence[Mat] | None = None) -> tuple[Mat, ...]:
    """The committed context web: the record projector conjugated by every
    irrep element, ``p_g = rho(g) * P * rho(g)^T``."""

    if irrep is None:
        irrep = standard_irrep()
    projectors = tuple(
        mmul(mmul(representation, RECORD_PROJECTOR), mtranspose(representation))
        for representation in irrep
    )
    require(
        projectors[0] == RECORD_PROJECTOR,
        "CONTEXT_WEB",
        "identity conjugate differs from the record projector",
    )
    return projectors


def record_commutator(projectors: Sequence[Mat] | None = None) -> Mat:
    """The committed commutator ``Q*P - P*Q`` with ``Q`` the threecycle
    conjugate (gauge element index 3) and ``P`` the record projector."""

    if projectors is None:
        projectors = context_projectors()
    rotated = projectors[3]
    return msub(
        mmul(rotated, RECORD_PROJECTOR), mmul(RECORD_PROJECTOR, rotated)
    )


def phase_lift(commutator: Mat, coefficient: Q3 | None = None) -> Mat:
    """The lift ``I/2 - c*i*(Q*P - P*Q)`` with committed
    ``c = 2*sqrt(3)/3``; the coefficient knob exists for mutation tests."""

    if coefficient is None:
        coefficient = COMMITTED_COEFFICIENT
    return msub(
        mscale(C3.of(Fraction(1, 2)), IDENTITY),
        mscale(C3(Q3_ZERO, coefficient), commutator),
    )


def declared_phase_operation(coefficient: Q3 | None = None) -> Mat:
    """The declared phase-sensitive effect, demanded equal to the committed
    Pauli +Y projector, Hermitian, and idempotent; a wrong coefficient
    fails closed."""

    effect = phase_lift(record_commutator(), coefficient)
    require(
        effect == PAULI_Y_PLUS_PROJECTOR,
        "PHASE_PROJECTOR",
        "lift is not the committed Pauli +Y projector",
    )
    require(
        mdagger(effect) == effect,
        "PHASE_HERMITIAN",
        "declared effect is not Hermitian",
    )
    require(
        mmul(effect, effect) == effect,
        "PHASE_IDEMPOTENT",
        "declared effect is not idempotent",
    )
    return effect


def record_diagonal_state(first: int, second: int) -> Mat:
    """The record-diagonal state ``diag(first/mass, second/mass)`` with
    ``mass = first + second``."""

    require(
        isinstance(first, int) and isinstance(second, int),
        "STATE_COUNTS",
        "record counts must be integers",
    )
    require(first >= 0 and second >= 0, "STATE_COUNTS", "record counts are negative")
    mass = first + second
    require(mass > 0, "STATE_COUNTS", "record counts have zero total mass")
    state = mat(
        ((Fraction(first, mass), 0), (0, Fraction(second, mass)))
    )
    require(mtrace(state) == C3.of(1), "STATE_TRACE", "state trace is not one")
    return state


def committed_core_state(x: Fraction | int, y: Fraction | int) -> Mat:
    """A committed-core state: the committed diagonal literals
    ``diag(111/179, 68/179)`` with one off-diagonal coordinate ``x + i*y``,
    Hermitian by construction and demanded positive semidefinite
    (``x^2 + y^2 <= 111*68/179^2``, the bound the committed integer window
    expresses on phase count pairs)."""

    x = Fraction(x)
    y = Fraction(y)
    p = Fraction(RUN_COUNTS[0], RUN_MASS)
    q = Fraction(RUN_COUNTS[1], RUN_MASS)
    require(
        x * x + y * y <= p * q,
        "STATE_POSITIVITY",
        "off-diagonal coordinate leaves the positive-semidefinite core",
    )
    off = C3(Q3.of(x), Q3.of(y))
    return mat(((C3.of(p), off), (off.conjugate(), C3.of(q))))


def born_weight(state: Mat, effect: Mat) -> Fraction:
    """The Born weight ``Tr(state * effect)``, demanded to be a plain
    rational in [0, 1]."""

    weight = mtrace(mmul(state, effect))
    require(
        weight.im.is_zero(), "WEIGHT_IMAGINARY", "Born weight has an imaginary part"
    )
    require(
        weight.re.b == 0, "WEIGHT_SQRT3", "Born weight has a sqrt(3) part"
    )
    value = weight.re.a
    require(0 <= value <= 1, "WEIGHT_RANGE", "Born weight outside [0, 1]")
    return value


def deterministic_counts(
    weight: Fraction, run_mass: int = RUN_MASS
) -> tuple[int, int, int, int]:
    """Counts of the exhaustive deterministic semantics: the least positive
    multiple ``k`` of the run mass with ``weight * k * run_mass`` integral,
    the mass ``k * run_mass``, and the two integer counts."""

    require(run_mass > 0, "RUN_MASS", "run mass must be positive")
    k = 1
    while (weight * (k * run_mass)).denominator != 1:
        k += 1
        require(
            k <= weight.denominator,
            "RUN_MASS_MULTIPLE",
            "no admissible run-mass multiple",
        )
    mass = k * run_mass
    first = int(weight * mass)
    return k, mass, first, mass - first


def check_phase_window(counts: Sequence[int]) -> tuple[int, int]:
    """The committed integer receipt window on a phase count pair:
    ``32041*(a - b)^2 <= 30192*(a + b)^2`` with both outcomes positive
    integers; any violation fails closed."""

    require(len(counts) == 2, "WINDOW_SHAPE", "phase counts must be a pair")
    a, b = counts
    require(
        isinstance(a, int) and isinstance(b, int),
        "WINDOW_SHAPE",
        "phase counts must be integers",
    )
    require(a > 0 and b > 0, "WINDOW_POSITIVITY", "a phase outcome has zero mass")
    lhs = WINDOW_LHS_COEFFICIENT * (a - b) ** 2
    rhs = WINDOW_RHS_COEFFICIENT * (a + b) ** 2
    require(
        lhs <= rhs,
        "WINDOW_VIOLATION",
        "phase counts violate the committed receipt window",
    )
    return lhs, rhs


def named_effects() -> tuple[tuple[str, Mat], ...]:
    """The committed context table: the diagonal record context, the six
    conjugated web contexts, and the declared phase context."""

    projectors = context_projectors()
    phase_effect = declared_phase_operation()
    effects: list[tuple[str, Mat]] = [("web_diagonal", RECORD_PROJECTOR)]
    effects += [
        (f"web_conjugated_{g}", projectors[g]) for g in range(len(projectors))
    ]
    effects.append(("phase", phase_effect))
    return tuple(effects)


def outcome_table(state: Mat, run_mass: int = RUN_MASS) -> list[dict[str, Any]]:
    """The complete exact outcome table of every committed context plus the
    phase context on ``state``: one row per context with the effect matrix,
    the exact Born weight, the minimal run-mass multiple, the mass, and the
    integer count pair.  Every outcome is demanded positive and the phase
    row is demanded inside the committed integer window."""

    rows: list[dict[str, Any]] = []
    for name, effect in named_effects():
        weight = born_weight(state, effect)
        k, mass, first, second = deterministic_counts(weight, run_mass)
        require(
            first > 0 and second > 0,
            "OUTCOME_POSITIVITY",
            f"context {name}: an outcome has zero mass",
        )
        rows.append(
            {
                "context": name,
                "effect": encode_matrix(effect),
                "born_weight": str(weight),
                "run_mass_multiple": k,
                "mass": mass,
                "counts": [first, second],
            }
        )
    check_phase_window(rows[-1]["counts"])
    return rows


def build_receipt() -> dict[str, Any]:
    """The committed-run export: the declared effect under its legacy field
    name, the declared diagonal matrix, the complete deterministic count
    table, and the fail-closed phase window clause, stamped with the
    recorded-decision provenance."""

    effect = declared_phase_operation()
    state = record_diagonal_state(*RUN_COUNTS)
    rows = outcome_table(state, RUN_MASS)
    require(
        rows[0]["counts"] == list(RUN_COUNTS),
        "DIAGONAL_SEMANTICS",
        "diagonal context does not reproduce the committed run counts",
    )
    window_lhs, window_rhs = check_phase_window(rows[-1]["counts"])
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_decision": {
            "register_row": REGISTER_ROW,
            "disposition": REGISTER_DISPOSITION,
            "date": DECISION_DATE,
            "lane_issue": LANE_ISSUE,
            "statement": DECISION_STATEMENT,
        },
        "provenance": {
            "operation_origin": "declared_phase_sensitive_effect",
            "derived_from_source_dynamics": False,
            "statement": PROVENANCE_STATEMENT,
            "register_path": "tracking/premise_register.json",
            "reference_receipt_path": (
                "code/phase_operation_producer/PHASE_OPERATION_RECEIPT.v1.json"
            ),
            "reference_receipt_schema": REFERENCE_RECEIPT_SCHEMA,
            "reference_payload_sha256": REFERENCE_PAYLOAD_SHA256,
            "reference_run_git_commit": REFERENCE_RUN_COMMIT,
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_seed": REFERENCE_SEED,
        },
        "declared_operation": {
            "formula": "I/2 - (2*sqrt(3)/3)*i*(Q*P - P*Q)",
            "matrix": encode_matrix(effect),
            "equals_pauli_y_plus_projector": True,
            "scalar_encoding": (
                "each scalar is [re_rational, re_sqrt3, im_rational, im_sqrt3] over Q"
            ),
        },
        "committed_state": {
            "matrix": encode_matrix(state),
            "reading": (
                "declared record-diagonal matrix built from literals "
                "(111, 68) at reference mass 179; no Fin-2 "
                "source-reachability witness"
            ),
        },
        "semantics": SEMANTICS_DECLARATION,
        "contexts": rows,
        "phase_receipt_window": {
            "inequality": "32041*(a - b)^2 <= 30192*(a + b)^2",
            "lhs": window_lhs,
            "rhs": window_rhs,
            "satisfied": True,
        },
        "boundaries": {
            "operation_scope": (
                "declared phase-sensitive effect carried under a legacy "
                "field name; never derived from source dynamics; not an "
                "operation, instrument, measurement, or validation"
            ),
            "count_scope": (
                "integer pairs are generated expected-frequency numerators "
                "of the declared exhaustive deterministic semantics; never "
                "measured, never sampled"
            ),
            "instrument_scope": (
                "this package is a static semantic-conformance fixture; no "
                "instrument is frozen by this receipt; no run is "
                "evidential; INS-01 remains the controlling OL-A1 verdict"
            ),
            "ol_c5_scope": (
                "OL-C5 status is untouched by this lane; attainment stays "
                "conditional structural under the axiomatized premise"
            ),
            "register_scope": (
                "PR-52 physical attachments stay open; source production of "
                "a phase-sensitive effect remains the open obligation "
                "recorded on register row PR-04; PR-64 owns "
                "completely-positive trace-nonincreasing outcome maps, a "
                "trace-preserving summed channel, and effect/readback "
                "compatibility; PR-65 owns common preparation, public "
                "outcomes, provenance, and custody; PR-03 owns "
                "cross-context additivity"
            ),
            "window_reading": (
                "32041 = 179^2 and 30192 = 4*111*68: the committed integer "
                "window is the positive-semidefiniteness bound of the "
                "committed core on phase count pairs"
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def replay_against_reference(
    receipt: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail-closed cross-repository replay: every context row of the
    simulator receipt (name, effect, Born weight, run-mass multiple, mass,
    counts) is demanded equal to the committed reference receipt row, along
    with the legacy-named declared effect matrix, the declared diagonal
    matrix, the input pins, and the phase window integers."""

    require(
        receipt.get("schema") == SCHEMA,
        "REPLAY_SCHEMA",
        "simulator receipt schema drift",
    )
    require(
        reference.get("schema") == REFERENCE_RECEIPT_SCHEMA,
        "REPLAY_SCHEMA",
        "reference receipt schema drift",
    )

    decision = reference.get("recorded_decision", {})
    ours = receipt.get("recorded_decision", {})
    for key in ("register_row", "disposition", "date", "lane_issue"):
        require(
            decision.get(key) == ours.get(key),
            "REPLAY_DECISION",
            f"recorded decision field {key} differs from the reference",
        )

    inputs = reference.get("inputs", {})
    provenance = receipt.get("provenance", {})
    require(
        inputs.get("payload_sha256") == provenance.get("reference_payload_sha256"),
        "REPLAY_PINS",
        "payload digest differs from the reference",
    )
    require(
        inputs.get("run_git_commit") == provenance.get("reference_run_git_commit"),
        "REPLAY_PINS",
        "run commit differs from the reference",
    )
    require(
        inputs.get("run_mass") == RUN_MASS
        and inputs.get("run_counts") == list(RUN_COUNTS),
        "REPLAY_PINS",
        "run literals differ from the reference",
    )

    require(
        reference.get("declared_operation", {}).get("matrix")
        == receipt.get("declared_operation", {}).get("matrix"),
        "REPLAY_OPERATION",
        "legacy operation-field matrix differs from the reference",
    )
    require(
        reference.get("committed_state", {}).get("matrix")
        == receipt.get("committed_state", {}).get("matrix"),
        "REPLAY_STATE",
        "committed state matrix differs from the reference",
    )

    reference_rows = reference.get("contexts")
    rows = receipt.get("contexts")
    require(
        isinstance(reference_rows, list)
        and isinstance(rows, list)
        and len(reference_rows) == len(rows),
        "REPLAY_TABLE",
        "context table size differs from the reference",
    )
    for row, reference_row in zip(rows, reference_rows):
        name = reference_row.get("context")
        for key in (
            "context",
            "effect",
            "born_weight",
            "run_mass_multiple",
            "mass",
            "counts",
        ):
            require(
                row.get(key) == reference_row.get(key),
                "REPLAY_CONTEXT",
                f"context {name}: field {key} differs from the reference",
            )
    check_phase_window(reference_rows[-1].get("counts"))

    window = reference.get("phase_receipt_window", {})
    ours_window = receipt.get("phase_receipt_window", {})
    for key in ("inequality", "lhs", "rhs", "satisfied"):
        require(
            window.get(key) == ours_window.get(key),
            "REPLAY_WINDOW",
            f"phase window field {key} differs from the reference",
        )

    return {
        "schema": "oph.sim.pr04_phase_operation_replay.v1",
        "reference_schema": REFERENCE_RECEIPT_SCHEMA,
        "reference_payload_sha256": REFERENCE_PAYLOAD_SHA256,
        "contexts_replayed": len(rows),
        "phase_counts": rows[-1]["counts"],
        "phase_window_lhs": ours_window.get("lhs"),
        "phase_window_rhs": ours_window.get("rhs"),
        "verdict": (
            "Cross-repository replay verifies static semantic conformance: "
            "the generated Born table of the declared PR-04 effect on the "
            "declared diagonal matrix equals the committed reference "
            "receipt exactly; no instrument, measurement, or validation is "
            "certified."
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON_OBJECT", f"{path} is not an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="replay the receipt against the committed reference receipt",
    )
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE_RECEIPT
    )
    args = parser.parse_args()
    receipt = build_receipt()
    if args.replay:
        report = replay_against_reference(receipt, load_json(args.reference))
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
        return
    print(text, end="")


if __name__ == "__main__":
    main()
