"""Exact icosahedral spinor CHSH candidate for issue 652.

The twelve-port source packet supplies an oriented icosahedral frame and a
non-split binary-icosahedral lift of its proper deck group.  This module asks
what that source geometry would imply on the associated two-spinor quantum
branch.  It proves three finite statements exactly:

* the diagonal binary-icosahedral action on ``C^2 tensor C^2`` has a unique
  invariant line, represented by the alternating spinor tensor;
* the Pauli map is covariant under every measured proper deck rotation; and
* a declared carrier-covariant family of 120 four-setting designs has
  ``|CHSH| = 1 + 3/sqrt(5) > 2`` on that invariant line.

The setting family is not selected uniquely by the carrier.  It is the union
of two 60-element proper-rotation orbits, and it is a strict subset of the
ambient maximizing quadruples.  The receipt records that non-uniqueness rather
than treating covariance as a source selection rule.

This is not a completed-record Bell witness.  The source does not produce two
spinor wings, prepare the invariant state, implement the Pauli settings, or
bind their outcomes to spacelike completed records.  The receipt therefore
names the first missing producer and refuses physical promotion.  In
particular, it does not upgrade the declared algebra-state representation
tracked by issue 230 or evade the finite classical-completion boundary of
issue 311.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.core.charged_response import (
    Q5,
    ZERO,
    canonical_sha256,
    load_carrier,
    match_vertex_frame,
    q5_dot,
)
from oph_fpe.core.spin_statistics_response import (
    Quat,
    measure_deck_realization,
    measure_lift_group,
    produce_spin_statistics_artifact,
)


SCHEMA = "oph.icosahedral_chsh_candidate.v1"
ISSUE = 652
VERDICT = (
    "EXACT_PROJECTIVE_BRANCH_CANDIDATE__"
    "TWO_WING_COMPLETED_RECORD_SOURCE_PRODUCER_MISSING"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "echosahedral_federation_reference.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT / "data" / "quantum" / "icosahedral_chsh_candidate_receipt.json"
)


class CandidateError(ValueError):
    """Typed fail-closed candidate error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise CandidateError(code, message)


@dataclass(frozen=True)
class C5:
    """Exact complex number with real and imaginary parts in Q(sqrt(5))."""

    re: Q5
    im: Q5

    @staticmethod
    def real(value: Q5 | Fraction | int) -> "C5":
        if not isinstance(value, Q5):
            value = Q5.of(value)
        return C5(value, ZERO)

    def __add__(self, other: "C5") -> "C5":
        return C5(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "C5") -> "C5":
        return C5(self.re - other.re, self.im - other.im)

    def __neg__(self) -> "C5":
        return C5(-self.re, -self.im)

    def __mul__(self, other: "C5") -> "C5":
        return C5(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def conjugate(self) -> "C5":
        return C5(self.re, -self.im)

    def inverse(self) -> "C5":
        norm = self.re * self.re + self.im * self.im
        require(not norm.is_zero(), "C5_DIVISION", "division by zero")
        inverse_norm = norm.inverse()
        return C5(self.re * inverse_norm, -self.im * inverse_norm)

    def is_zero(self) -> bool:
        return self.re.is_zero() and self.im.is_zero()


CZERO = C5.real(0)
CONE = C5.real(1)


Matrix = list[list[C5]]
Vector = list[C5]


def matrix_identity(size: int) -> Matrix:
    return [
        [CONE if row == column else CZERO for column in range(size)]
        for row in range(size)
    ]


def matrix_mul(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    middle = len(right)
    columns = len(right[0])
    require(
        all(len(row) == middle for row in left), "MATRIX_SHAPE", "left matrix shape"
    )
    require(
        all(len(row) == columns for row in right), "MATRIX_SHAPE", "right matrix shape"
    )
    result = [[CZERO for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        for index in range(middle):
            if left[row][index].is_zero():
                continue
            for column in range(columns):
                result[row][column] = (
                    result[row][column] + left[row][index] * right[index][column]
                )
    return result


def matrix_dagger(matrix: Matrix) -> Matrix:
    return [
        [matrix[column][row].conjugate() for column in range(len(matrix))]
        for row in range(len(matrix[0]))
    ]


def matrix_equal(left: Matrix, right: Matrix) -> bool:
    return len(left) == len(right) and all(
        len(left[row]) == len(right[row])
        and all(
            (left[row][column] - right[row][column]).is_zero()
            for column in range(len(left[row]))
        )
        for row in range(len(left))
    )


def matrix_vector_mul(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum(
            (matrix[row][column] * vector[column] for column in range(len(vector))),
            CZERO,
        )
        for row in range(len(matrix))
    ]


def tensor(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            left[left_row][left_column] * right[right_row][right_column]
            for left_column in range(len(left[0]))
            for right_column in range(len(right[0]))
        ]
        for left_row in range(len(left))
        for right_row in range(len(right))
    ]


def matrix_rank(rows: Sequence[Sequence[C5]]) -> int:
    work = [list(row) for row in rows]
    if not work:
        return 0
    active = 0
    columns = len(work[0])
    require(
        all(len(row) == columns for row in work), "MATRIX_SHAPE", "ragged rank matrix"
    )
    for column in range(columns):
        pivot = next(
            (
                row
                for row in range(active, len(work))
                if not work[row][column].is_zero()
            ),
            None,
        )
        if pivot is None:
            continue
        work[active], work[pivot] = work[pivot], work[active]
        scale = work[active][column].inverse()
        work[active] = [entry * scale for entry in work[active]]
        for row in range(len(work)):
            if row == active or work[row][column].is_zero():
                continue
            factor = work[row][column]
            work[row] = [
                work[row][index] - factor * work[active][index]
                for index in range(columns)
            ]
        active += 1
        if active == len(work):
            break
    return active


def quaternion_spinor_matrix(quaternion: Quat) -> Matrix:
    """SU(2) matrix whose adjoint action is the producer's SO(3) rotation."""

    w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
    return [
        [C5(w, -z), C5(-y, -x)],
        [C5(y, -x), C5(w, z)],
    ]


def pauli_matrix(vector: Sequence[Q5]) -> Matrix:
    x, y, z = vector
    return [
        [C5.real(z), C5(x, -y)],
        [C5(x, y), C5.real(-z)],
    ]


def q5_vector_from_matrix_column(
    matrix: Sequence[Sequence[Q5]], column: Sequence[Q5]
) -> list[Q5]:
    return [
        matrix[row][0] * column[0]
        + matrix[row][1] * column[1]
        + matrix[row][2] * column[2]
        for row in range(3)
    ]


def scaled_pauli_covariance_holds(
    spinor: Matrix,
    rotation: Sequence[Sequence[Q5]],
    vector: Sequence[Q5],
) -> bool:
    left = matrix_mul(matrix_mul(spinor, pauli_matrix(vector)), matrix_dagger(spinor))
    right = pauli_matrix(q5_vector_from_matrix_column(rotation, vector))
    return matrix_equal(left, right)


def singlet_expectation(
    left_vector: Sequence[Q5],
    right_vector: Sequence[Q5],
    norm_squared: Q5,
) -> Q5:
    """Exact expectation on (|01>-|10>)/sqrt(2), without radicals."""

    observable = tensor(pauli_matrix(left_vector), pauli_matrix(right_vector))
    singlet = [CZERO, CONE, -CONE, CZERO]
    image = matrix_vector_mul(observable, singlet)
    numerator = sum(
        (singlet[index].conjugate() * image[index] for index in range(4)),
        CZERO,
    )
    require(numerator.im.is_zero(), "SINGLET_EXPECTATION", "expectation is not real")
    return numerator.re * Q5.of(Fraction(1, 2)) * norm_squared.inverse()


def render_q5(value: Q5) -> str:
    return value.render()


def q5_abs(value: Q5) -> Q5:
    return value if value.sign() >= 0 else -value


def full_lift_group(lifts: Mapping[tuple[int, ...], Quat]) -> list[Quat]:
    elements: dict[tuple, Quat] = {}
    for lift in lifts.values():
        elements[lift.key()] = lift
        elements[lift.neg().key()] = lift.neg()
    return [elements[key] for key in sorted(elements)]


def invariant_line_certificate(lifts: Mapping[tuple[int, ...], Quat]) -> dict[str, Any]:
    singlet = [CZERO, CONE, -CONE, CZERO]
    identity = matrix_identity(4)
    constraints: list[list[C5]] = []
    invariant_count = 0
    group = full_lift_group(lifts)
    for quaternion in group:
        spinor = quaternion_spinor_matrix(quaternion)
        diagonal = tensor(spinor, spinor)
        image = matrix_vector_mul(diagonal, singlet)
        if all((image[index] - singlet[index]).is_zero() for index in range(4)):
            invariant_count += 1
        constraints.extend(
            [
                [diagonal[row][column] - identity[row][column] for column in range(4)]
                for row in range(4)
            ]
        )
    rank = matrix_rank(constraints)
    require(len(group) == 120, "LIFT_GROUP", "signed lift group must have order 120")
    require(invariant_count == 120, "SINGLET_INVARIANCE", "singlet is not invariant")
    require(
        rank == 3,
        "SINGLET_UNIQUENESS",
        "diagonal invariant space is not one-dimensional",
    )
    return {
        "signed_lift_count": len(group),
        "singlet_invariant_under_all_lifts": True,
        "common_constraint_rank": rank,
        "tensor_dimension": 4,
        "invariant_dimension": 4 - rank,
        "invariant_ray": "span(|01>-|10>)",
    }


def covariance_certificate(
    deck: Mapping[str, Any],
    lifts: Mapping[tuple[int, ...], Quat],
    frame: Sequence[Sequence[Q5]],
) -> dict[str, Any]:
    checks = 0
    for permutation in deck["rotations"]:
        spinor = quaternion_spinor_matrix(lifts[permutation])
        rotation = deck["matrices"][permutation]
        for vector in frame:
            require(
                scaled_pauli_covariance_holds(spinor, rotation, vector),
                "PAULI_COVARIANCE",
                "spinor adjoint action and measured deck rotation disagree",
            )
            checks += 1
    require(checks == 720, "PAULI_COVARIANCE", "unexpected covariance check count")
    return {
        "proper_rotation_count": 60,
        "port_axis_count": 12,
        "exact_covariance_checks": checks,
        "all_passed": True,
    }


def setting_rows(carrier: Mapping[str, Any]) -> Iterable[tuple[int, int, int, int]]:
    adjacency = carrier["adjacency"]
    antipode = carrier["antipode"]
    for a0 in range(12):
        for a1 in range(12):
            if a1 == a0 or a1 == antipode[a0] or adjacency[a0][a1]:
                continue
            common_neighbors = [
                candidate
                for candidate in range(12)
                if adjacency[a0][candidate] and adjacency[a1][candidate]
            ]
            for b0 in common_neighbors:
                yield a0, a1, b0, antipode[a1]


def local_hidden_variable_bound() -> dict[str, Any]:
    counts: dict[int, int] = {}
    for a0 in (-1, 1):
        for a1 in (-1, 1):
            for b0 in (-1, 1):
                for b1 in (-1, 1):
                    score = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1
                    counts[score] = counts.get(score, 0) + 1
    require(
        counts == {-2: 8, 2: 8}, "CLASSICAL_BOUND", "deterministic CHSH census failed"
    )
    return {
        "deterministic_assignment_count": 16,
        "score_counts": {str(key): value for key, value in sorted(counts.items())},
        "maximum_absolute_score": 2,
    }


def chsh_candidate_certificate(
    carrier: Mapping[str, Any],
    frame: Sequence[Sequence[Q5]],
) -> dict[str, Any]:
    norm_squared = q5_dot(frame[0], frame[0])
    require(
        all((q5_dot(vector, vector) - norm_squared).is_zero() for vector in frame),
        "PORT_NORM",
        "port frame does not have a common norm",
    )
    rows = list(setting_rows(carrier))
    require(len(rows) == 120, "SETTING_FAMILY", "expected 120 covariant setting rows")
    expected_magnitude = Q5.of(1, Fraction(3, 5))
    expected_signed = -expected_magnitude
    representative = None
    for row in rows:
        a0, a1, b0, b1 = row
        correlations = [
            singlet_expectation(frame[a0], frame[b0], norm_squared),
            singlet_expectation(frame[a0], frame[b1], norm_squared),
            singlet_expectation(frame[a1], frame[b0], norm_squared),
            singlet_expectation(frame[a1], frame[b1], norm_squared),
        ]
        score = correlations[0] + correlations[1] + correlations[2] - correlations[3]
        require(
            (score - expected_signed).is_zero(),
            "CHSH_SCORE",
            "carrier-covariant setting rows do not share the exact score",
        )
        if representative is None:
            representative = (row, correlations)
    require(representative is not None, "SETTING_FAMILY", "empty setting family")
    margin = expected_magnitude - Q5.of(2)
    require(margin.sign() > 0, "CHSH_VIOLATION", "candidate does not exceed two")
    row, correlations = representative
    ports = carrier["ports"]
    probability_rows = []
    for setting_index, correlation in enumerate(correlations):
        for left_outcome in (-1, 1):
            for right_outcome in (-1, 1):
                probability = (
                    Q5.of(1) + Q5.of(left_outcome * right_outcome) * correlation
                ) * Q5.of(Fraction(1, 4))
                require(
                    probability.sign() >= 0, "JOINT_LAW", "negative joint probability"
                )
                probability_rows.append(
                    {
                        "setting_pair": [setting_index // 2, setting_index % 2],
                        "outcomes": [left_outcome, right_outcome],
                        "probability": render_q5(probability),
                    }
                )
    return {
        "selection_rule": (
            "ordered nonadjacent, nonantipodal A0/A1 ports; B0 any common "
            "incidence neighbor; B1 the antipode of A1"
        ),
        "setting_quadruple_count": len(rows),
        "all_rows_same_exact_score": True,
        "representative_ports": [ports[index] for index in row],
        "representative_correlations": [render_q5(value) for value in correlations],
        "signed_chsh": render_q5(expected_signed),
        "absolute_chsh": render_q5(expected_magnitude),
        "absolute_chsh_float": expected_magnitude.to_float(),
        "classical_margin": render_q5(margin),
        "classical_margin_positive_exact": True,
        "setting_selection": {
            "status": "declared_combinatorial_construction",
            "source_selected": False,
            "unique_carrier_selected_family": False,
        },
        "joint_law": {
            "formula": "p(a,b|x,y)=(1+a*b*E(x,y))/4",
            "rows": probability_rows,
            "local_marginals": "1/2 on every setting slice",
            "algebraic_no_signalling": True,
        },
    }


def setting_family_symmetry_certificate(
    carrier: Mapping[str, Any],
    frame: Sequence[Sequence[Q5]],
    deck: Mapping[str, Any],
) -> dict[str, Any]:
    """Census the declared setting family and the ambient port quadruples.

    Covariance alone does not select the 120 rows.  This exact census keeps the
    distinction machine-readable: the family is two chiral A5 orbits and only
    one eighth of the 960 ordered port quadruples attaining the same maximal
    absolute score on the declared singlet branch.
    """

    declared_rows = set(setting_rows(carrier))
    remaining = set(declared_rows)
    proper_orbits: list[set[tuple[int, int, int, int]]] = []
    for seed in sorted(declared_rows):
        if seed not in remaining:
            continue
        orbit = {
            tuple(permutation[index] for index in seed)
            for permutation in deck["rotations"]
        }
        require(
            orbit <= declared_rows,
            "SETTING_COVARIANCE",
            "proper deck rotation leaves the declared setting family",
        )
        proper_orbits.append(orbit)
        remaining -= orbit
    orbit_sizes = sorted(len(orbit) for orbit in proper_orbits)
    require(
        orbit_sizes == [60, 60] and not remaining,
        "SETTING_COVARIANCE",
        "declared setting family must split into two proper-rotation orbits",
    )
    first_orbit, second_orbit = proper_orbits
    require(
        all(
            {tuple(permutation[index] for index in row) for row in first_orbit}
            == second_orbit
            for permutation in deck["improper"]
        ),
        "SETTING_COVARIANCE",
        "the improper deck coset does not exchange the two setting orbits",
    )

    norm_squared = q5_dot(frame[0], frame[0])
    inverse_norm = norm_squared.inverse()

    def correlation(left: int, right: int) -> Q5:
        return -(q5_dot(frame[left], frame[right]) * inverse_norm)

    maximum = ZERO
    maximizer_count = 0
    all_distinct_maximizer_count = 0
    for a0 in range(12):
        for a1 in range(12):
            for b0 in range(12):
                for b1 in range(12):
                    score = (
                        correlation(a0, b0)
                        + correlation(a0, b1)
                        + correlation(a1, b0)
                        - correlation(a1, b1)
                    )
                    magnitude = q5_abs(score)
                    comparison = (magnitude - maximum).sign()
                    if comparison > 0:
                        maximum = magnitude
                        maximizer_count = 1
                        all_distinct_maximizer_count = int(len({a0, a1, b0, b1}) == 4)
                    elif comparison == 0:
                        maximizer_count += 1
                        all_distinct_maximizer_count += int(len({a0, a1, b0, b1}) == 4)
    expected_maximum = Q5.of(1, Fraction(3, 5))
    require(
        (maximum - expected_maximum).is_zero(),
        "SETTING_CENSUS",
        "declared CHSH value is not the ambient port-axis maximum",
    )
    require(
        maximizer_count == 960 and all_distinct_maximizer_count == 480,
        "SETTING_CENSUS",
        "unexpected ambient maximizing-quadruple census",
    )
    return {
        "proper_rotation_orbit_count": len(proper_orbits),
        "proper_rotation_orbit_sizes": orbit_sizes,
        "improper_coset_exchanges_orbits": True,
        "ambient_ordered_port_quadruple_count": 12**4,
        "ambient_maximum_absolute_chsh": render_q5(maximum),
        "ambient_maximizer_count": maximizer_count,
        "ambient_all_distinct_maximizer_count": all_distinct_maximizer_count,
        "declared_family_count": len(declared_rows),
        "declared_family_is_unique_maximizer": False,
        "meaning": (
            "the 120 rows form a covariant declared construction, not a unique "
            "setting family selected by the carrier or repair dynamics"
        ),
    }


def build_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    carrier = load_carrier(manifest)
    frame = match_vertex_frame(carrier)
    deck = measure_deck_realization(carrier, frame)
    lift = measure_lift_group(deck)
    spin_artifact = produce_spin_statistics_artifact(manifest)

    invariant = invariant_line_certificate(lift["lifts"])
    covariance = covariance_certificate(deck, lift["lifts"], frame)
    candidate = chsh_candidate_certificate(carrier, frame)
    candidate["setting_family_symmetry"] = setting_family_symmetry_certificate(
        carrier, frame, deck
    )
    classical = local_hidden_variable_bound()

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "verdict": VERDICT,
        "source_parents": {
            "carrier_manifest_sha256": carrier["manifest_sha256"],
            "spin_statistics_artifact_sha256": spin_artifact["artifact_sha256"],
            "finite_spin_packet_gate_passed": spin_artifact["physical_source_gate"][
                "passed"
            ],
            "laboratory_exchange_measurement": spin_artifact["physical_source_gate"][
                "laboratory_exchange_measurement"
            ],
            "continuum_spin_statistics_theorem": spin_artifact["physical_source_gate"][
                "continuum_spin_statistics_theorem"
            ],
            "lift_group_order": spin_artifact["lift_measurement"]["lift_group_order"],
            "lift_group_centre_order": spin_artifact["lift_measurement"][
                "centre_order"
            ],
            "spin_structure_count": spin_artifact["support_homology"][
                "spin_structure_count"
            ],
        },
        "source_derived_content": {
            "carrier_lineage_scope": "declared_echosahedral_carrier_lineage",
            "universal_oph_carrier_derivation": False,
            "oriented_twelve_port_frame": True,
            "proper_deck_rotation_count": 60,
            "binary_icosahedral_spin_lift": True,
            "non_split_central_extension": True,
            "port_incidence_and_antipode": True,
        },
        "exact_projective_adapter": {
            "status": "mathematical construction from the source-derived spin lift",
            "spinor_complex_dimension": 2,
            "diagonal_invariant_line": invariant,
            "pauli_axis_covariance": covariance,
            "meaning": (
                "the binary lift admits its defining two-complex-dimensional spinor "
                "realization and a unique diagonal invariant ray within that declared "
                "tensor representation; this does not make that realization the "
                "physical completed-record algebra"
            ),
        },
        "chsh_candidate": candidate,
        "classical_reference": classical,
        "completed_record_gate": {
            "source_positive_nonclassical_record_witness": False,
            "physical_promotion_allowed": False,
            "first_missing_producer": "two_wing_completed_record_instrument",
            "required_outputs": [
                "two source-produced spinor wings on one declared physical domain",
                "preparation or repair selection of the diagonal invariant ray",
                "independent binary local setting choices with measurement-independence custody",
                "local Pauli-axis readout maps for the covariant port-setting family",
                "binary completed outcomes with no postselection",
                "a spacelike or operational-isolation certificate for the two wings",
                "setting-conditioned joint counts and no-signalling checks",
            ],
            "missing_outputs_in_registered_source": [
                "wing tensor-factor producer",
                "source state preparation",
                "setting intervention channel",
                "setting-to-readout attachment",
                "completed two-wing outcome records",
                "physical locality and measurement-independence certificates",
            ],
        },
        "boundaries": {
            "carrier_scope": (
                "the parent is the declared echosahedral carrier lineage certified "
                "by issue 565; the packet does not derive this carrier from every "
                "possible OPH screen"
            ),
            "issue_230": (
                "the quantum algebra-state and Born representation are not derived "
                "from pre-quantum repair records"
            ),
            "issue_311": (
                "the separate finite spectral interfaces retain their explicit "
                "classical harmonic completions"
            ),
            "central_defect_scope": (
                "the source-derived non-split spin lift supplies projective geometry; "
                "it does not by itself supply quantum operational semantics"
            ),
            "simulation_scope": (
                "the exact candidate is a source-geometry calculation, not a simulated "
                "Bell experiment or a laboratory prediction"
            ),
            "setting_scope": (
                "the 120-row family is a declared covariant construction and is not "
                "uniquely selected by the source geometry or repair dynamics"
            ),
        },
        "claim_boundary": (
            "Exact finite projective-branch candidate on the declared echosahedral "
            "carrier lineage: the binary-icosahedral lift, its defining spinor adapter, "
            "and a declared covariant port-setting family give "
            "|CHSH|=1+3/sqrt(5)>2 on the adapter's unique diagonal invariant ray. The "
            "setting family is not uniquely source-selected. No registered source "
            "producer prepares two physical spinor wings or emits setting-controlled "
            "spacelike completed records, so this is not a nonclassical record witness "
            "on the source surface. The representation lane stops at its named-producer "
            "boundary."
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    require(receipt.get("schema") == SCHEMA, "RECEIPT_SCHEMA", "unexpected schema")
    require(receipt.get("issue") == ISSUE, "RECEIPT_ISSUE", "unexpected issue")
    require(receipt.get("verdict") == VERDICT, "RECEIPT_VERDICT", "unexpected verdict")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    require(
        receipt.get("receipt_sha256") == canonical_sha256(body),
        "RECEIPT_HASH",
        "receipt self-hash failed",
    )
    expected = build_receipt(manifest)
    require(receipt == expected, "RECEIPT_REPLAY", "receipt differs from exact replay")
    gate = receipt["completed_record_gate"]
    require(
        gate["source_positive_nonclassical_record_witness"] is False
        and gate["physical_promotion_allowed"] is False,
        "PROMOTION_BOUNDARY",
        "candidate was promoted without the source instrument",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON_OBJECT", f"{path} is not an object")
    return value


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    if args.validate_only:
        validate_receipt(load_json(args.output), manifest)
        print("ICOSAHEDRAL_CHSH_CANDIDATE_VALID")
        return
    receipt = build_receipt(manifest)
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
