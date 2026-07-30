"""Exact A2 holonomy classifier and source-bridge audit.

The twelve-port carrier has real coefficient module

    1 + 3 + 3' + 5.

That module and the inverse-port response do not determine a Lie bracket.
This module keeps two questions separate:

1. What follows if the observer's own reversible response is a faithful,
   compact, commutator-closed twelve-dimensional tangent and every proper
   carrier recharting is implemented internally by closed overlap holonomy?
2. Do the released finite-source artifacts establish those hypotheses?

The first question has an exact positive answer.  Compact classification and
the one-dimensional fixed space force centre dimension one and simple-factor
dimensions three and eight.  The second question fails closed because the
released response artifact contains no ordered current tomography or
same-current overlap words.  This is an open producer boundary rather than a
negative result about OPH.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.charged_response import (
    _oriented_face_set,
    incidence_automorphisms,
    load_carrier,
    produce_charged_response_artifact,
)
from oph_fpe.core.spin_statistics_response import produce_spin_statistics_artifact


SCHEMA = "oph.a2-holonomy-current-selector/1.0.0"
STATUS_OPEN = "OPEN_SOURCE_HOLONOMY_BRIDGE"

REQUIRED_RAW_SOURCE_OBJECTS = (
    "ordered_two_sided_port_response_histories",
    "twelve_infinitesimal_generator_derivatives",
    "exact_commutator_reconstruction",
    "closed_overlap_words_covering_all_proper_rechartings",
    "same_words_response_implementers",
    "response_generated_identity_component_factorizations",
    "port_current_commuting_square",
    "cofinal_refinement_intertwiners",
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode("ascii")
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _partitions_from_simple_dimensions(
    total: int,
    simple_dimensions: Sequence[int],
) -> list[tuple[int, ...]]:
    """All unordered sums of the declared compact-simple dimensions."""

    allowed = tuple(sorted(set(int(value) for value in simple_dimensions)))
    rows: list[tuple[int, ...]] = []

    def visit(remaining: int, minimum_index: int, row: tuple[int, ...]) -> None:
        if remaining == 0:
            rows.append(row)
            return
        for index in range(minimum_index, len(allowed)):
            value = allowed[index]
            if value > remaining:
                break
            visit(remaining - value, index, row + (value,))

    visit(total, 0, ())
    return rows


def compact_inner_action_classification(
    *,
    port_dimension: int = 12,
    fixed_dimension: int = 1,
) -> dict[str, Any]:
    """Recompute the finite arithmetic core of the holonomy theorem.

    The list ``(3, 8, 10)`` is the declared classical classification input for
    compact simple Lie algebras of dimension at most twelve.  No gauge-group
    name, centre dimension, or desired factor split is supplied to the
    enumeration.
    """

    simple_dimensions = (3, 8, 10)
    center_candidates = tuple(range(fixed_dimension + 1))
    branches: list[dict[str, Any]] = []

    for center_dimension in center_candidates:
        semisimple_dimension = port_dimension - center_dimension
        for factors in _partitions_from_simple_dimensions(
            semisimple_dimension, simple_dimensions
        ):
            row: dict[str, Any] = {
                "center_dimension": center_dimension,
                "semisimple_dimension": semisimple_dimension,
                "simple_factor_dimensions": list(factors),
                "survives_compact_dimension_arithmetic": True,
            }
            if center_dimension == 0 and factors == (3, 3, 3, 3):
                possible_fixed_dimensions = sorted(
                    {
                        sum(choice)
                        for choice in itertools.product((0, 3), repeat=4)
                    }
                )
                row["centerless_inner_a5_fixed_dimensions"] = (
                    possible_fixed_dimensions
                )
                row["survives_inner_fixed_space_test"] = (
                    fixed_dimension in possible_fixed_dimensions
                )
                row["exclusion_reason"] = (
                    "an inner A5 action on each su(2) factor is trivial with "
                    "fixed dimension three or icosahedral with fixed dimension "
                    "zero; their sum cannot equal one"
                )
            else:
                row["survives_inner_fixed_space_test"] = True
            branches.append(row)

    survivors = [
        row for row in branches if row["survives_inner_fixed_space_test"]
    ]
    passed = (
        len(survivors) == 1
        and survivors[0]["center_dimension"] == 1
        and survivors[0]["simple_factor_dimensions"] == [3, 8]
    )
    return {
        "port_tangent_dimension": port_dimension,
        "a5_fixed_dimension": fixed_dimension,
        "declared_classical_simple_dimension_input": list(simple_dimensions),
        "center_bound_from_inner_action_and_fixed_space": [0, 1],
        "branches": branches,
        "survivors": survivors,
        "unique_lie_type": (
            "u(1)+su(2)+su(3)" if passed else None
        ),
        "passed": passed,
        "claim_boundary": (
            "This is a conditional compact Lie-type classifier. It neither "
            "constructs the response tangent nor proves holonomy fullness."
        ),
    }


def port_action_fixed_dimension(carrier_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the fixed-space dimension from the proper carrier action."""

    carrier = load_carrier(carrier_manifest)
    oriented_faces = _oriented_face_set(carrier["faces"])
    rotations = []
    for permutation in incidence_automorphisms(carrier["adjacency"]):
        images = _oriented_face_set(
            [
                [permutation[a], permutation[b], permutation[c]]
                for a, b, c in carrier["faces"]
            ]
        )
        if images == oriented_faces:
            rotations.append(permutation)

    unseen = set(range(len(carrier["ports"])))
    orbits: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        orbit = sorted({int(permutation[seed]) for permutation in rotations})
        orbits.append(orbit)
        unseen.difference_update(orbit)
    return {
        "port_dimension": len(carrier["ports"]),
        "proper_rotation_count": len(rotations),
        "port_orbits": orbits,
        "fixed_space_dimension": len(orbits),
        "transitive": len(orbits) == 1,
    }


def audit_released_source_artifacts(
    charged_artifact: Mapping[str, Any],
    spin_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether the released artifacts contain the A2 source bridge.

    Summary booleans and semantic construction labels are deliberately
    insufficient.  The future gate must consume raw histories, words,
    generators, and exact factorization witnesses from one source packet.
    """

    charged_binding = charged_artifact.get("carrier_binding")
    spin_binding = spin_artifact.get("carrier_binding")
    same_carrier = (
        isinstance(charged_binding, Mapping)
        and isinstance(spin_binding, Mapping)
        and charged_binding.get("carrier_manifest_sha256")
        == spin_binding.get("carrier_manifest_sha256")
    )
    lift = spin_artifact.get("lift_measurement")
    binary_deck_available = (
        isinstance(lift, Mapping)
        and lift.get("lift_group_order") == 120
        and lift.get("unique_nontrivial_involution") == "-1"
        and lift.get("centre_order") == 2
    )
    derived = charged_artifact.get("derived")
    lift_status = (
        derived.get("current_lift_status")
        if isinstance(derived, Mapping)
        else None
    )
    response_constraints_typed = (
        isinstance(lift_status, Mapping)
        and lift_status.get("source_selected") is False
        and lift_status.get("commutator_reconstructed_from_ordered_response")
        is False
        and lift_status.get("overlap_holonomy_internality_certified") is False
    )

    source_packet = charged_artifact.get("a2_holonomy_source_packet")
    raw_object_presence = {
        name: isinstance(source_packet, Mapping) and name in source_packet
        for name in REQUIRED_RAW_SOURCE_OBJECTS
    }
    all_raw_objects_present = all(raw_object_presence.values())
    # There is no registered verifier for these raw objects.  Presence,
    # summary booleans, and hashes cannot promote the source bridge.
    registered_source_verifier = False
    source_bridge_passed = False

    return {
        "same_carrier_source_projection": same_carrier,
        "binary_icosahedral_deck_measurement_available": binary_deck_available,
        "inverse_port_response_constraints_correctly_typed": (
            response_constraints_typed
        ),
        "required_raw_source_objects": raw_object_presence,
        "raw_source_object_count": sum(raw_object_presence.values()),
        "required_raw_source_object_count": len(REQUIRED_RAW_SOURCE_OBJECTS),
        "all_raw_objects_present": all_raw_objects_present,
        "registered_source_verifier": registered_source_verifier,
        "a2_holonomy_source_bridge_receipt": source_bridge_passed,
        "status": "ATTAINED" if source_bridge_passed else STATUS_OPEN,
        "missing_objects": [
            name for name, present in raw_object_presence.items() if not present
        ],
        "reason": (
            "the released artifacts share the carrier and supply the inverse-"
            "port response and binary deck lift, but they do not contain raw "
            "ordered current tomography or same-current closed overlap words"
            if not all_raw_objects_present
            else "raw object names are present, but no registered verifier "
            "recomputes their scientific content"
        ),
    }


def adversarial_controls() -> dict[str, Any]:
    """Exact logical controls for the three common shortcut errors."""

    return {
        "abelian_port_records": {
            "same_a5_coefficient_module": True,
            "commutator_closed": True,
            "nontrivial_a5_action_is_inner": False,
            "rejected": True,
        },
        "binary_deck_on_independent_ancilla": {
            "binary_deck_profile_matches": True,
            "same_current_source_identity": False,
            "rejected": True,
        },
        "ambient_normalizer_without_response_generated_path": {
            "covariance_square_can_match": True,
            "implementer_in_response_generated_identity_component": False,
            "rejected": True,
        },
        "semantic_model_label": {
            "construction_name_present": True,
            "raw_ordered_response_present": False,
            "accepted_as_source_derivation": False,
            "rejected": True,
        },
    }


def build_report(carrier_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run the exact classifier and audit both released source producers."""

    charged = produce_charged_response_artifact(carrier_manifest)
    spin = produce_spin_statistics_artifact(carrier_manifest)
    port_action = port_action_fixed_dimension(carrier_manifest)
    classification = compact_inner_action_classification(
        port_dimension=port_action["port_dimension"],
        fixed_dimension=port_action["fixed_space_dimension"],
    )
    source_audit = audit_released_source_artifacts(charged, spin)
    controls = adversarial_controls()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "carrier_manifest_sha256": charged["carrier_binding"][
            "carrier_manifest_sha256"
        ],
        "source_artifacts": {
            "inverse_port_response_sha256": charged["artifact_sha256"],
            "binary_deck_transport_sha256": spin["artifact_sha256"],
        },
        "port_action": port_action,
        "theorem_classifier": classification,
        "source_bridge_audit": source_audit,
        "negative_controls": controls,
        "receipts": {
            "A2_HOLONOMY_CLASSIFIER_RECEIPT": classification["passed"],
            "A2_HOLONOMY_SOURCE_BRIDGE_RECEIPT": source_audit[
                "a2_holonomy_source_bridge_receipt"
            ],
            "PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT": (
                classification["passed"]
                and source_audit["a2_holonomy_source_bridge_receipt"]
            ),
        },
        "status": source_audit["status"],
        "claim_boundary": (
            "The exact conditional classifier forces Standard Model Lie type "
            "from a compact twelve-dimensional port current with internal A5 "
            "holonomy. The released source artifacts do not establish the "
            "ordered-response and same-current holonomy hypotheses. No physical "
            "gauge-current promotion follows from this report."
        ),
    }
    report["report_sha256"] = _sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.carrier_manifest.read_text(encoding="utf-8"))
    report = build_report(manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_RAW_SOURCE_OBJECTS",
    "SCHEMA",
    "STATUS_OPEN",
    "adversarial_controls",
    "audit_released_source_artifacts",
    "build_report",
    "compact_inner_action_classification",
    "port_action_fixed_dimension",
]
