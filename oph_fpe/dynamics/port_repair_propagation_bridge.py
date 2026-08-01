"""Exact classifier for the repair-to-propagation boundary of FZ-11.

The finite OPH sources certify a repair operator on twelve *internal*
working readings.  Its thirty supports are incidence seams of one carrier.  A
spatial finite-difference stencil is a different object: it acts on a field at
translated sites along one of the carrier direction orbits.  A physical
readout which identifies that stencil with a scalar or polarization-independent
sector is a third object.  This module refuses to identify any two of those
domains without an explicit, source-pinned packet.

The classifier has three typed exits:

``PORT_INTERFACE_REPAIR_FORCES_FZ11``
    A complete equal-weight twelve-vertex translation stencil and a
    same-operator physical readout have both been supplied and checked.

``SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11``
    A complete equal-weight thirty-edge translation stencil and its physical
    readout have both been supplied.  Its spin-six ray is distinct from FZ-11.

``NO_SOURCE_NATIVE_TRANSLATION_BRIDGE``
    The translation or readout bridge is absent, malformed, or belongs to a
    different orbit.  This is the exit of the repository's current sources.

The exact orbit calculation uses only the pinned carrier geometry and
``Q(sqrt(5))`` arithmetic.  No measured angular data or comparison target is
read.  In particular, the internal seam count of thirty never selects the
thirty-edge spatial stencil by itself.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.charged_response import (
    ONE,
    Q5,
    ZERO,
    load_carrier,
    match_vertex_frame,
    q5_dot,
)
from oph_fpe.dynamics import canonical_seam_repair


REPORT_SCHEMA = "oph.port_repair_propagation_bridge_receipt.v1"
VERIFICATION_SCHEMA = "oph.port_repair_propagation_bridge_verification.v1"
SOURCE_PACKET_SCHEMA = "oph.port_repair_propagation_source_packet.v1"

PORT_INTERFACE_REPAIR_FORCES_FZ11 = "PORT_INTERFACE_REPAIR_FORCES_FZ11"
SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11 = "SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11"
NO_SOURCE_NATIVE_TRANSLATION_BRIDGE = "NO_SOURCE_NATIVE_TRANSLATION_BRIDGE"

# The live issue's certified exits are kept separate from the narrower
# operator classifier above.  In particular, recognizing a vertex stencil is
# not enough to certify the issue's forced-and-exclusive exit.
FORCED_EXCLUSIVE_PRIMITIVE_PORT_PROPAGATION_BRANCH = (
    "FORCED_EXCLUSIVE_PRIMITIVE_PORT_PROPAGATION_BRANCH"
)
BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION = (
    "BOUNDED_NONSELECTION__FZ11_REMAINS_BRANCH_PREDICTION"
)
ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED = (
    "ADDITIONAL_PHYSICAL_PREMISE_REQUIRED__ALTERNATIVES_CERTIFIED"
)

INTERNAL_SEAM_REPAIR_CERTIFIED = "INTERNAL_SEAM_REPAIR_CERTIFIED"
SPATIAL_PORT_HOP_SOURCE_RECEIPT = "SPATIAL_PORT_HOP_SOURCE_RECEIPT"
SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT = (
    "SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT"
)
FZ11_FORCED_EXCLUSIVE_RECEIPT = "FZ11_FORCED_EXCLUSIVE_RECEIPT"

ORBIT_VERTEX = "vertex12"
ORBIT_FACE = "face20"
ORBIT_EDGE = "edge30"
ORBIT_ORDER = (ORBIT_VERTEX, ORBIT_FACE, ORBIT_EDGE)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARRIER_MANIFEST = (
    REPOSITORY_ROOT / "tests/fixtures/echosahedral_federation_reference.json"
)
DEFAULT_REPAIR_RECEIPT = (
    REPOSITORY_ROOT
    / "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
)
CANONICAL_REPAIR_PRODUCER = (
    REPOSITORY_ROOT / "oph_fpe/dynamics/canonical_seam_repair.py"
)
BRIDGE_PRODUCER = Path(__file__).resolve()
INDEPENDENT_CURRENT_EXIT_VERIFIER = (
    REPOSITORY_ROOT
    / "oph_fpe/dynamics/verify_port_repair_propagation_bridge_independent.py"
)
BRIDGE_TEST = REPOSITORY_ROOT / "tests/test_port_repair_propagation_bridge.py"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data/repair_closure/port_repair_propagation_bridge_receipt.json"
)

Exponent = tuple[int, int, int]
Polynomial = dict[Exponent, Q5]
Vector = tuple[Q5, Q5, Q5]


class BridgePacketError(ValueError):
    """A typed source-packet failure which must select the fail-closed exit."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise BridgePacketError(code, message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _raw_file_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "repository_relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _q5_sum(values: Sequence[Q5]) -> Q5:
    result = ZERO
    for value in values:
        result = result + value
    return result


def _q5_power(value: Q5, power: int) -> Q5:
    if power < 0:
        raise ValueError("Q(sqrt(5)) exponent must be nonnegative")
    result = ONE
    for _ in range(power):
        result = result * value
    return result


def _q5_divide(numerator: Q5, denominator: Q5) -> Q5:
    return numerator * denominator.inverse()


def _q5_rational(value: Q5, label: str) -> Fraction:
    if value.b != 0:
        raise ValueError(f"{label} did not descend from Q(sqrt(5)) to Q")
    return value.a


def _q5_render(value: Q5) -> str:
    return value.render()


def _vector_add(*vectors: Vector) -> Vector:
    return tuple(
        _q5_sum([vector[coordinate] for vector in vectors])
        for coordinate in range(3)
    )  # type: ignore[return-value]


def _vector_negate(vector: Vector) -> Vector:
    return (-vector[0], -vector[1], -vector[2])


def _vector_key(vector: Vector) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple((coordinate.a, coordinate.b) for coordinate in vector)


def _render_vector(vector: Vector) -> list[str]:
    return [_q5_render(coordinate) for coordinate in vector]


def _poly_clean(poly: Polynomial) -> Polynomial:
    return {exponent: value for exponent, value in poly.items() if not value.is_zero()}


def _poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, ZERO) + value
    return _poly_clean(result)


def _poly_scale(poly: Polynomial, scalar: Q5) -> Polynomial:
    return _poly_clean({exponent: scalar * value for exponent, value in poly.items()})


def _poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_value in left.items():
        for right_exponent, right_value in right.items():
            exponent = tuple(
                left_exponent[index] + right_exponent[index]
                for index in range(3)
            )
            result[exponent] = result.get(exponent, ZERO) + left_value * right_value
    return _poly_clean(result)


def _poly_power(poly: Polynomial, power: int) -> Polynomial:
    if power < 0:
        raise ValueError("polynomial exponent must be nonnegative")
    result: Polynomial = {(0, 0, 0): ONE}
    for _ in range(power):
        result = _poly_multiply(result, poly)
    return result


def _linear_power(vector: Vector, power: int) -> Polynomial:
    linear: Polynomial = {
        (1, 0, 0): vector[0],
        (0, 1, 0): vector[1],
        (0, 0, 1): vector[2],
    }
    return _poly_power(linear, power)


def _poly_evaluate(poly: Polynomial, vector: Vector) -> Q5:
    total = ZERO
    for exponent, coefficient in poly.items():
        term = coefficient
        for coordinate, power in zip(vector, exponent, strict=True):
            term = term * _q5_power(coordinate, power)
        total = total + term
    return total


def _normalized_moment(vectors: Sequence[Vector], power: int) -> Polynomial:
    if power % 2:
        raise ValueError("only even antipodal moments are used")
    result: Polynomial = {}
    for vector in vectors:
        norm = q5_dot(vector, vector)
        if norm.sign() <= 0:
            raise ValueError("direction has nonpositive exact squared norm")
        scale = _q5_power(norm, power // 2).inverse()
        result = _poly_add(result, _poly_scale(_linear_power(vector, power), scale))
    return result


def _radial_power(power: int) -> Polynomial:
    if power % 2:
        raise ValueError("radial polynomial degree must be even")
    radius_squared: Polynomial = {
        (2, 0, 0): ONE,
        (0, 2, 0): ONE,
        (0, 0, 2): ONE,
    }
    return _poly_power(radius_squared, power // 2)


def _assert_polynomial_equal(left: Polynomial, right: Polynomial, label: str) -> None:
    if _poly_clean(left) != _poly_clean(right):
        raise ValueError(f"exact polynomial identity failed: {label}")


def _orbit_vectors(carrier: Mapping[str, Any], frame: Sequence[Vector]) -> dict[str, tuple[Vector, ...]]:
    vertex_vectors = tuple(frame)
    face_vectors = tuple(
        _vector_add(frame[face[0]], frame[face[1]], frame[face[2]])
        for face in carrier["faces"]
    )
    edge_vectors = tuple(
        _vector_add(frame[left], frame[right])
        for left in range(12)
        for right in range(left + 1, 12)
        if carrier["adjacency"][left][right]
    )
    result = {
        ORBIT_VERTEX: vertex_vectors,
        ORBIT_FACE: face_vectors,
        ORBIT_EDGE: edge_vectors,
    }
    expected_sizes = {ORBIT_VERTEX: 12, ORBIT_FACE: 20, ORBIT_EDGE: 30}
    for orbit, vectors in result.items():
        if len(vectors) != expected_sizes[orbit]:
            raise ValueError(f"{orbit} has the wrong support size")
        keys = {_vector_key(vector) for vector in vectors}
        if len(keys) != len(vectors):
            raise ValueError(f"{orbit} contains duplicate directions")
        if any(_vector_key(_vector_negate(vector)) not in keys for vector in vectors):
            raise ValueError(f"{orbit} is not closed under direction reversal")
        norms = {q5_dot(vector, vector) for vector in vectors}
        if len(norms) != 1:
            raise ValueError(f"{orbit} does not have a single exact radius")
    return result


def exact_orbit_ray_table(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the three equal-weight orbit rays over exact Q(sqrt(5))."""

    carrier = load_carrier(manifest)
    frame = tuple(match_vertex_frame(carrier))
    orbits = _orbit_vectors(carrier, frame)

    radius2 = _radial_power(2)
    radius4 = _radial_power(4)
    radius6 = _radial_power(6)

    moments: dict[str, dict[int, Polynomial]] = {}
    for orbit, vectors in orbits.items():
        count = len(vectors)
        moments[orbit] = {
            power: _normalized_moment(vectors, power)
            for power in (2, 4, 6)
        }
        _assert_polynomial_equal(
            moments[orbit][2],
            _poly_scale(radius2, Q5.of(Fraction(count, 3))),
            f"{orbit} second moment",
        )
        _assert_polynomial_equal(
            moments[orbit][4],
            _poly_scale(radius4, Q5.of(Fraction(count, 5))),
            f"{orbit} fourth moment",
        )

    vertex_residual = _poly_add(
        moments[ORBIT_VERTEX][6],
        _poly_scale(radius6, Q5.of(Fraction(-12, 7))),
    )
    vertex = orbits[ORBIT_VERTEX][0]
    vertex_norm = q5_dot(vertex, vertex)
    vertex_value = _q5_divide(
        _poly_evaluate(vertex_residual, vertex),
        _q5_power(vertex_norm, 3),
    )
    if vertex_value.is_zero():
        raise ValueError("vertex spin-six invariant has zero normalization")
    invariant_i6 = _poly_scale(vertex_residual, vertex_value.inverse())
    for orbit_vertex in orbits[ORBIT_VERTEX]:
        orbit_vertex_norm = q5_dot(orbit_vertex, orbit_vertex)
        normalized_value = _q5_divide(
            _poly_evaluate(invariant_i6, orbit_vertex),
            _q5_power(orbit_vertex_norm, 3),
        )
        if normalized_value != ONE:
            raise ValueError("spin-six invariant is not one on the full vertex orbit")

    rows: dict[str, Any] = {}
    expected_beta = {
        ORBIT_VERTEX: Fraction(64, 175),
        ORBIT_FACE: Fraction(-64, 189),
        ORBIT_EDGE: Fraction(-2, 7),
    }
    for orbit, vectors in orbits.items():
        count = len(vectors)
        residual = _poly_add(
            moments[orbit][6],
            _poly_scale(radius6, Q5.of(Fraction(-count, 7))),
        )
        pivot = next(
            exponent for exponent, value in invariant_i6.items() if not value.is_zero()
        )
        beta = _q5_divide(residual.get(pivot, ZERO), invariant_i6[pivot])
        _assert_polynomial_equal(
            residual,
            _poly_scale(invariant_i6, beta),
            f"{orbit} spin-six residual",
        )
        beta_q = _q5_rational(beta, f"{orbit} beta")
        if beta_q != expected_beta[orbit]:
            raise ValueError(f"{orbit} exact beta disagrees with the frozen orbit value")

        normalization = Fraction(6, count)
        c4 = -normalization * Fraction(count, 5) / 24
        b0 = normalization * Fraction(count, 7) / 720
        b6 = normalization * beta_q / 720
        rows[orbit] = {
            "support_size": count,
            "exact_squared_radius": _q5_render(q5_dot(vectors[0], vectors[0])),
            "antipodal_pair_count": count // 2,
            "equal_weight_normalization": str(normalization),
            "moment_identities": {
                "M2": f"{count}/3 r^2",
                "M4": f"{count}/5 r^4",
                "M6": f"{count}/7 r^6 + ({beta_q}) I6",
                "I6_normalization": "I6(v)=1 on every normalized vertex direction",
                "identities_checked_coefficientwise_over_Q_sqrt5": True,
            },
            "ray": {
                "C4": str(c4),
                "B0": str(b0),
                "B6": str(b6),
                "B0_over_C4_squared": str(b0 / (c4 * c4)),
                "B6_over_C4_squared": str(b6 / (c4 * c4)),
                "B6_over_B0": str(b6 / b0),
            },
            "directions": [_render_vector(vector) for vector in vectors],
        }

    ray_pairs = [
        (left, right)
        for index, left in enumerate(ORBIT_ORDER)
        for right in ORBIT_ORDER[index + 1 :]
    ]
    if any(rows[left]["ray"] == rows[right]["ray"] for left, right in ray_pairs):
        raise ValueError("distinct carrier orbits produced the same exact ray")
    return {
        "invariant_basis": {
            "isotropic_degree_four": "r^4",
            "isotropic_degree_six": "r^6",
            "spin_six": "I6 normalized to one on the vertex orbit",
            "basis_source": "exact twelve-vertex carrier geometry",
        },
        "rows": rows,
        "pairwise_distinct": True,
        "intrinsic_angular_ranks_one_through_five": "zero",
        "binary_refinement_of_rank_six_coefficient": "B6(a/2) = B6(a)/16",
        "physical_or_cosmological_measurements_read": False,
        "physicalization_claimed": False,
    }


def _internal_repair_block() -> dict[str, Any]:
    report = canonical_seam_repair.canonical_seam_repair_certificate()
    verification = canonical_seam_repair.verify_canonical_seam_repair_certificate(report)
    if not verification["receipt"]:
        raise ValueError("canonical seam-repair producer replay failed")
    bounded = json.loads(DEFAULT_REPAIR_RECEIPT.read_text(encoding="utf-8"))
    required = {
        "schema": "oph.bounded_atomic_self_readback_closure.v1",
        "FINITE_DIRECTED_SEAM_TORSOR_RECEIPT": True,
        "PHYSICAL_REPAIR_LAW_RECEIPT": False,
    }
    for key, expected in required.items():
        if bounded.get(key) != expected:
            raise ValueError(f"bounded repair receipt has unexpected {key}")
    return {
        "domain": "finite_twelve_port_scalar_working_readback",
        "support_kind": "thirty_internal_incidence_seams",
        "port_count": 12,
        "support_count": 30,
        "operator": "T = I - L_icosahedron/60",
        "canonical_certificate_payload_sha256": report[
            "certificate_payload_sha256"
        ],
        "canonical_repair_producer_pin": _raw_file_pin(
            CANONICAL_REPAIR_PRODUCER
        ),
        "bounded_atomic_receipt_pin": _raw_file_pin(DEFAULT_REPAIR_RECEIPT),
        "bounded_atomic_receipt_payload_sha256": bounded.get(
            "certificate_payload_sha256"
        ),
        "spatial_translation_identification": False,
        "same_operator_physical_readout": False,
        "physical_repair_law_receipt": False,
    }


def current_source_packet() -> dict[str, Any]:
    """Return the exact packet represented by the committed source tree."""

    return {
        "schema": SOURCE_PACKET_SCHEMA,
        "carrier_manifest_pin": _raw_file_pin(DEFAULT_CARRIER_MANIFEST),
        "internal_seam_repair": _internal_repair_block(),
        "spatial_hop_operator": None,
        "physical_readout": None,
        "scope_boundary": {
            "internal_seams_are_spatial_hops": False,
            "equal_support_counts_imply_operator_identity": False,
            "physicalization_claimed": False,
        },
    }


def _translation_operator_core(orbit: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": "spatial_scalar_translation_field",
        "symbol_class": "real_reciprocal_finite_range_cosine",
        "support_orbit": orbit,
        "support_size": row["support_size"],
        "direction_coordinates": "unnormalized_exact_carrier_rays",
        "unit_direction_rule": "u = d/sqrt(d dot d)",
        "continuum_normalization": f"6/{row['support_size']}",
        "directions": copy.deepcopy(row["directions"]),
        "terms": [
            {
                "direction_index": index,
                "hop_radius": 1,
                "weight": "1",
            }
            for index in range(row["support_size"])
        ],
        "independent_onsite_term": "0",
        "radius_two_terms": [],
        "additional_directional_terms": [],
        "additional_isotropic_terms_through_k6": [],
    }


def make_candidate_source_packet(
    orbit: str,
    *,
    attach_physical_readout: bool = True,
) -> dict[str, Any]:
    """Build a deterministic synthetic packet for classifier/mutation tests.

    This helper does not claim that the packet is present in OPH sources.  It
    makes each load-bearing field explicit so the three exits can be tested.
    """

    manifest = json.loads(DEFAULT_CARRIER_MANIFEST.read_text(encoding="utf-8"))
    table = exact_orbit_ray_table(manifest)
    if orbit not in table["rows"]:
        raise ValueError(f"unknown carrier orbit: {orbit}")
    core = _translation_operator_core(orbit, table["rows"][orbit])
    digest = _canonical_sha256(core)
    spatial = {
        "operator_core": core,
        "operator_sha256": digest,
        "source_native_translation_receipt": True,
        "complete_support_receipt": True,
        "source_history_replay": [
            {
                "event_index": index,
                "direction": copy.deepcopy(direction),
                "hop_radius": 1,
                "weight": "1",
            }
            for index, direction in enumerate(core["directions"])
        ],
    }
    readout: dict[str, Any] | None = None
    if attach_physical_readout:
        readout_core = {
            "domain": "physical_scalar_or_polarization_independent_sector",
            "sector": "scalar",
            "source_operator_sha256": digest,
            "readback_operator_sha256": digest,
            "frame_transport_receipt": True,
            "same_operator_receipt": True,
            "source_history_replay_receipt": True,
        }
        readout = {
            "readout_core": readout_core,
            "readout_sha256": _canonical_sha256(readout_core),
        }
    packet = current_source_packet()
    packet["spatial_hop_operator"] = spatial
    packet["physical_readout"] = readout
    return packet


def _validate_internal_domain(packet: Mapping[str, Any]) -> None:
    expected = current_source_packet()
    _require(
        packet.get("schema") == SOURCE_PACKET_SCHEMA,
        "SOURCE_PACKET_SCHEMA",
        "source packet schema mismatch",
    )
    _require(
        packet.get("carrier_manifest_pin") == expected["carrier_manifest_pin"],
        "CARRIER_PIN",
        "carrier manifest raw-file pin mismatch",
    )
    _require(
        packet.get("internal_seam_repair") == expected["internal_seam_repair"],
        "INTERNAL_REPAIR_BINDING",
        "internal repair block is not the recomputed source block",
    )
    boundary = packet.get("scope_boundary")
    _require(isinstance(boundary, Mapping), "SCOPE_BOUNDARY", "scope boundary missing")
    _require(
        boundary.get("internal_seams_are_spatial_hops") is False
        and boundary.get("equal_support_counts_imply_operator_identity") is False
        and boundary.get("physicalization_claimed") is False,
        "FORBIDDEN_DOMAIN_COLLAPSE",
        "source packet collapses distinct repair, propagation, or readout domains",
    )


def _validate_spatial_operator(
    spatial: Mapping[str, Any],
    table: Mapping[str, Any],
) -> str:
    core = spatial.get("operator_core")
    _require(isinstance(core, Mapping), "SPATIAL_CORE", "spatial operator core missing")
    orbit = core.get("support_orbit")
    _require(orbit in ORBIT_ORDER, "SPATIAL_ORBIT", "unsupported carrier orbit")
    expected_core = _translation_operator_core(orbit, table["rows"][orbit])
    _require(
        core == expected_core,
        "SPATIAL_GRAMMAR",
        "translation operator has extra, missing, unequal, onsite, or radius-two terms",
    )
    expected_digest = _canonical_sha256(core)
    _require(
        spatial.get("operator_sha256") == expected_digest,
        "SPATIAL_DIGEST",
        "translation operator digest mismatch",
    )
    _require(
        spatial.get("source_native_translation_receipt") is True,
        "SOURCE_NATIVE_TRANSLATION",
        "source-native translation receipt absent",
    )
    _require(
        spatial.get("complete_support_receipt") is True,
        "COMPLETE_SUPPORT",
        "complete support receipt absent",
    )
    expected_history = [
        {
            "event_index": index,
            "direction": copy.deepcopy(direction),
            "hop_radius": 1,
            "weight": "1",
        }
        for index, direction in enumerate(core["directions"])
    ]
    _require(
        spatial.get("source_history_replay") == expected_history,
        "SOURCE_HISTORY",
        "source history does not replay the declared operator",
    )
    return str(orbit)


def _validate_physical_readout(
    readout: Mapping[str, Any],
    spatial: Mapping[str, Any],
) -> None:
    core = readout.get("readout_core")
    _require(isinstance(core, Mapping), "READOUT_CORE", "physical readout core missing")
    operator_digest = spatial["operator_sha256"]
    expected = {
        "domain": "physical_scalar_or_polarization_independent_sector",
        "sector": "scalar",
        "source_operator_sha256": operator_digest,
        "readback_operator_sha256": operator_digest,
        "frame_transport_receipt": True,
        "same_operator_receipt": True,
        "source_history_replay_receipt": True,
    }
    _require(
        core == expected,
        "PHYSICAL_READOUT_BINDING",
        "readout does not identify the exact source operator in the declared sector",
    )
    _require(
        readout.get("readout_sha256") == _canonical_sha256(core),
        "PHYSICAL_READOUT_DIGEST",
        "physical readout digest mismatch",
    )


def classify_source_packet(
    packet: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one packet and preserve every failure as a typed blocker."""

    if manifest is None:
        manifest = json.loads(DEFAULT_CARRIER_MANIFEST.read_text(encoding="utf-8"))
    table = exact_orbit_ray_table(manifest)
    blockers: list[str] = []
    internal_ok = False
    spatial_ok = False
    readout_ok = False
    orbit: str | None = None
    try:
        _validate_internal_domain(packet)
        internal_ok = True
        spatial = packet.get("spatial_hop_operator")
        if spatial is None:
            raise BridgePacketError(
                "SPATIAL_TRANSLATION_ABSENT",
                "no source-native translation operator has been supplied",
            )
        _require(
            isinstance(spatial, Mapping),
            "SPATIAL_OPERATOR_TYPE",
            "spatial operator must be an object",
        )
        orbit = _validate_spatial_operator(spatial, table)
        spatial_ok = True
        readout = packet.get("physical_readout")
        if readout is None:
            raise BridgePacketError(
                "PHYSICAL_READOUT_ABSENT",
                "no same-operator physical readout has been supplied",
            )
        _require(
            isinstance(readout, Mapping),
            "PHYSICAL_READOUT_TYPE",
            "physical readout must be an object",
        )
        _validate_physical_readout(readout, spatial)
        readout_ok = True
    except BridgePacketError as error:
        blockers.append(error.code)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        blockers.append("MALFORMED_NONCANONICAL_PACKET")

    if internal_ok and spatial_ok and readout_ok and orbit == ORBIT_VERTEX:
        operator_exit = PORT_INTERFACE_REPAIR_FORCES_FZ11
    elif internal_ok and spatial_ok and readout_ok and orbit == ORBIT_EDGE:
        operator_exit = SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11
    else:
        operator_exit = NO_SOURCE_NATIVE_TRANSLATION_BRIDGE
        if internal_ok and spatial_ok and readout_ok and orbit == ORBIT_FACE:
            blockers.append("FACE20_ORBIT_IS_A_DISTINCT_NON_FZ11_ALTERNATIVE")

    # This v1 packet can classify exact orbit stencils and same-operator
    # readout bindings.  It does not certify a complete repair/propagation
    # grammar, coherent orientation and boost transport, or an independent
    # implementation of the full implication chain.  Consequently it cannot
    # attain the live issue's forced-and-exclusive exit even when its narrower
    # operator classifier recognizes the vertex stencil.
    if internal_ok and spatial_ok and readout_ok:
        issue_exit = ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED
    else:
        issue_exit = BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION

    return {
        "operator_classifier_exit": operator_exit,
        "issue_certified_exit": issue_exit,
        "blockers": sorted(set(blockers)),
        "internal_seam_repair_certified": internal_ok,
        "spatial_hop_source_certified": spatial_ok,
        "same_operator_physical_readout_certified": readout_ok,
        "selected_spatial_orbit": orbit,
        "selected_ray": (
            copy.deepcopy(table["rows"][orbit]["ray"])
            if orbit is not None and spatial_ok
            else None
        ),
        "internal_support_count_used_as_spatial_selection": False,
        "physicalization_assumed": False,
        "forced_exclusive_issue_exit_attainable_in_v1": False,
    }


def _report_payload() -> dict[str, Any]:
    manifest = json.loads(DEFAULT_CARRIER_MANIFEST.read_text(encoding="utf-8"))
    table = exact_orbit_ray_table(manifest)
    packet = current_source_packet()
    classification = classify_source_packet(packet, manifest=manifest)
    if classification["operator_classifier_exit"] != NO_SOURCE_NATIVE_TRANSLATION_BRIDGE:
        raise ValueError("current source tree unexpectedly promoted a translation bridge")
    payload = {
        "schema": REPORT_SCHEMA,
        "issue": 655,
        "status": classification["issue_certified_exit"],
        "source_packet": packet,
        "implementation_pins": {
            "bridge_producer": _raw_file_pin(BRIDGE_PRODUCER),
            "independent_current_exit_verifier": _raw_file_pin(
                INDEPENDENT_CURRENT_EXIT_VERIFIER
            ),
            "mutation_and_independent_orbit_tests": _raw_file_pin(BRIDGE_TEST),
        },
        "typed_domains": {
            "internal_seam_repair": (
                "twelve working readings and thirty internal incidence equalizers"
            ),
            "spatial_hop_operator": (
                "a field translated along one complete carrier direction orbit"
            ),
            "physical_readout": (
                "a separately attached scalar or polarization-independent sector"
            ),
            "domains_identified_by_support_count_alone": False,
        },
        "exact_orbit_ray_table": table,
        "classification": classification,
        "live_issue_acceptance_audit": {
            "physical_response_generated_by_accepted_repair_on_source_native_patch": False,
            "primitive_orbit_forced_as_sole_support_in_complete_grammar": False,
            "equal_weights_and_absence_of_independent_terms_forced": False,
            "surviving_physical_alternatives_exhaustively_enumerated": False,
            "scalar_or_polarization_independent_same_operator_action": False,
            "coherent_frame_transport_and_orientation_profile": False,
            "declared_boost_law": False,
            "physical_readout_isolation": False,
            "machine_checked_exact_orbit_implications": True,
            "producer_replay_receipt": True,
            "independent_orbit_ray_recomputation": True,
            "independent_current_exit_implementation": True,
            "independent_full_bridge_implementation": False,
            "lean_full_bridge_implication_chain": False,
            "mutation_controls": True,
            "comparison_data_consumed": False,
            "forced_exclusive_exit_supported": False,
            "defensible_exit": (
                BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
            ),
        },
        "dependency_audit": {
            "closed_issue_62_supplies_accepted_physical_repair_law": False,
            "reason": (
                "The current exact repair sources type the scalar and integer "
                "seam laws as candidates and set PHYSICAL_REPAIR_LAW_RECEIPT "
                "and canonical_three_axiom_derivation to false."
            ),
            "geometry_and_orbit_moments_available": True,
            "geometry_implies_physical_translation_operator": False,
        },
        INTERNAL_SEAM_REPAIR_CERTIFIED: True,
        SPATIAL_PORT_HOP_SOURCE_RECEIPT: False,
        SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT: False,
        FZ11_FORCED_EXCLUSIVE_RECEIPT: False,
        "epistemic_boundary": {
            "current_result": (
                "The exact vertex, face, and edge rays are classified, while the "
                "committed sources provide no native translation/readout bridge."
            ),
            "internal_thirty_seams_do_not_select_edge30_spatial_hops": True,
            "comparison_data_read": False,
            "physical_prediction_unsealed": False,
            "reopen_condition": (
                "Supply a source-history-replayed translation operator and a "
                "digest-bound physical readout in the same operator domain."
            ),
        },
    }
    _require_no_floats(payload)
    return payload


def produce_bridge_receipt() -> dict[str, Any]:
    payload = _report_payload()
    report = copy.deepcopy(payload)
    report["receipt_sha256"] = _canonical_sha256(payload)
    return report


def verify_bridge_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        received_digest = received.pop("receipt_sha256", None)
        if report.get("schema") != REPORT_SCHEMA:
            reasons.append("schema_mismatch")
        if received_digest != _canonical_sha256(received):
            reasons.append("receipt_digest_mismatch")
        if received != _report_payload():
            reasons.append("producer_replay_mismatch")
        forbidden_true = (
            SPATIAL_PORT_HOP_SOURCE_RECEIPT,
            SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT,
            FZ11_FORCED_EXCLUSIVE_RECEIPT,
        )
        if any(report.get(name) is not False for name in forbidden_true):
            reasons.append("forbidden_physical_promotion")
        if report.get("status") != (
            BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION
        ):
            reasons.append("wrong_current_source_exit")
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        reasons.append("malformed_or_noncanonical_receipt")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "claim_boundary": (
            "Replay checks the exact orbit arithmetic, source pins, and fail-closed "
            "classification. It does not construct a physical translation/readout bridge."
        ),
    }


def _require_no_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise ValueError(f"receipt contains a float at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_no_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_no_floats(item, f"{path}[{index}]")


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = json.loads(args.verify.read_text(encoding="utf-8"))
        verification = verify_bridge_receipt(report)
        _write_json(verification, args.output)
        return 0 if verification["receipt"] else 1
    report = produce_bridge_receipt()
    verification = verify_bridge_receipt(report)
    if not verification["receipt"]:
        _write_json(verification, args.output)
        return 1
    _write_json(report, args.output)
    return 0


__all__ = [
    "FZ11_FORCED_EXCLUSIVE_RECEIPT",
    "ADDITIONAL_PHYSICAL_PREMISE_REQUIRED_ALTERNATIVES_CERTIFIED",
    "BOUNDED_NONSELECTION_FZ11_REMAINS_BRANCH_PREDICTION",
    "FORCED_EXCLUSIVE_PRIMITIVE_PORT_PROPAGATION_BRANCH",
    "INTERNAL_SEAM_REPAIR_CERTIFIED",
    "NO_SOURCE_NATIVE_TRANSLATION_BRIDGE",
    "ORBIT_EDGE",
    "ORBIT_FACE",
    "ORBIT_VERTEX",
    "PORT_INTERFACE_REPAIR_FORCES_FZ11",
    "SAME_OPERATOR_PHYSICAL_READOUT_RECEIPT",
    "SEAM_REPAIR_SELECTS_EDGE30_NOT_FZ11",
    "SPATIAL_PORT_HOP_SOURCE_RECEIPT",
    "classify_source_packet",
    "current_source_packet",
    "exact_orbit_ray_table",
    "make_candidate_source_packet",
    "produce_bridge_receipt",
    "verify_bridge_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
