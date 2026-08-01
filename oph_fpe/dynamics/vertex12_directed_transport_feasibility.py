"""Exact feasibility boundary for the issue-655 directed transport ledger.

The pinned source packet emits twelve carrier matchings ``S_p`` through its
repair-event ledger.  This module asks whether an enlarged noncollapsed site
set can turn those same maps into directed transports while retaining a
surjective map back to the source carriers.

There is an elementary exact obstruction.  If ``pi T_p = S_p pi`` for every
port and ``T_-p = T_p^-1``, then surjectivity of ``pi`` forces
``S_-p = S_p^-1``.  Every emitted ``S_p`` is an involution, while each of the
six antipodal source pairs consists of distinct matchings.  No cover or
extension satisfying that semiconjugacy contract can therefore supply the
requested ledger.

This is a boundary on reuse of the present undirected repair matchings.  It is
not a boundary on a source law that emits genuinely oriented transition maps,
and it is not a spatial or physical no-go result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations
from oph_fpe.dynamics import verify_vertex12_atomic_port_transfer_independent


SCHEMA = "oph.vertex12-directed-transport-feasibility.v1"
VERIFICATION_SCHEMA = "oph.vertex12-directed-transport-feasibility-verification.v1"
STATUS = (
    "EXACT_SEMICONJUGATE_COVER_OBSTRUCTION_FOR_CURRENT_SOURCE_MATCHINGS__"
    "ORIENTED_SOURCE_TRANSITION_LAW_OPEN"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
UPSTREAM_PATH = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
PRODUCER_PATH = Path(__file__).resolve()
INDEPENDENT_VERIFIER_PATH = (
    REPOSITORY_ROOT
    / "oph_fpe/dynamics/verify_vertex12_directed_transport_feasibility_independent.py"
)
TEST_PATH = (
    REPOSITORY_ROOT / "tests/test_vertex12_directed_transport_feasibility.py"
)
GEOMETRY_PATH = REPOSITORY_ROOT / "oph_fpe/core/icosahedral.py"

UPSTREAM_SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket.v1"
UPSTREAM_STATUS = (
    "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__"
    "SPATIAL_PHYSICAL_BRIDGE_OPEN"
)


class FeasibilityError(ValueError):
    """Raised when the pinned source packet cannot support the exact audit."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FeasibilityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(value, dict):
        raise FeasibilityError("receipt JSON root is not an object")
    return value


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    size = len(permutation)
    if sorted(permutation) != list(range(size)):
        raise FeasibilityError("source transport row is not a permutation")
    result = [0] * size
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _compose(
    left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    if len(left) != len(right):
        raise FeasibilityError("cannot compose permutations of different sizes")
    return tuple(left[right[index]] for index in range(len(left)))


def _validated_upstream() -> dict[str, Any]:
    report = _load_json(UPSTREAM_PATH)
    verification = (
        verify_vertex12_atomic_port_transfer_independent.verify_report(report)
    )
    if verification.get("receipt") is not True:
        raise FeasibilityError("pinned atomic-transfer packet failed verification")
    if (
        report.get("schema") != UPSTREAM_SCHEMA
        or report.get("status") != UPSTREAM_STATUS
        or type(report.get("issue")) is not int
        or report.get("issue") != 655
    ):
        raise FeasibilityError("pinned atomic-transfer packet contract drifted")
    return report


def _obstruction(upstream: Mapping[str, Any]) -> dict[str, Any]:
    operator = upstream.get("atomic_transfer_operator")
    boundary = upstream.get("quotient_and_spatial_boundary")
    history = upstream.get("source_history_replay")
    if not all(isinstance(item, Mapping) for item in (operator, boundary, history)):
        raise FeasibilityError("upstream source blocks are missing")

    carrier_ids = operator.get("carrier_ids")
    port_rows = operator.get("port_rows")
    antipodes = boundary.get("carrier_antipode_map")
    event_rows = history.get("event_rows")
    if (
        not isinstance(carrier_ids, list)
        or len(carrier_ids) != 8
        or len(set(carrier_ids)) != 8
        or not isinstance(port_rows, list)
        or len(port_rows) != 12
        or not isinstance(antipodes, list)
        or len(antipodes) != 12
        or not isinstance(event_rows, list)
        or len(event_rows) != 48
    ):
        raise FeasibilityError("upstream carrier, port, or event census drifted")

    by_port: dict[int, tuple[int, ...]] = {}
    events_by_port: dict[int, list[str]] = {port: [] for port in range(12)}
    for row in port_rows:
        if not isinstance(row, Mapping) or type(row.get("port")) is not int:
            raise FeasibilityError("upstream port row is malformed")
        port = row["port"]
        values = row.get("carrier_partner_permutation")
        if port in by_port or not isinstance(values, list) or len(values) != 8:
            raise FeasibilityError("upstream port permutation is missing or duplicated")
        permutation = tuple(values)
        if any(type(item) is not int for item in permutation):
            raise FeasibilityError("upstream port permutation has a noninteger entry")
        inverse = _inverse(permutation)
        if inverse != permutation or any(permutation[index] == index for index in range(8)):
            raise FeasibilityError("upstream port map is not a fixed-point-free involution")
        by_port[port] = permutation
    if set(by_port) != set(range(12)):
        raise FeasibilityError("upstream port rows do not cover 0 through 11")

    for row in event_rows:
        if (
            not isinstance(row, Mapping)
            or type(row.get("port")) is not int
            or row["port"] not in events_by_port
            or type(row.get("event_id")) is not str
            or not row["event_id"].startswith("sha256:")
        ):
            raise FeasibilityError("upstream repair-event provenance row is malformed")
        events_by_port[row["port"]].append(row["event_id"])
    if any(len(rows) != 4 or len(set(rows)) != 4 for rows in events_by_port.values()):
        raise FeasibilityError("each source matching must be backed by four repair events")

    normalized_antipodes = tuple(antipodes)
    if (
        any(type(item) is not int or not 0 <= item < 12 for item in normalized_antipodes)
        or any(normalized_antipodes[normalized_antipodes[p]] != p for p in range(12))
        or any(normalized_antipodes[p] == p for p in range(12))
    ):
        raise FeasibilityError("upstream antipode map is malformed")

    pair_rows: list[dict[str, Any]] = []
    visited: set[int] = set()
    for port in range(12):
        antipode = normalized_antipodes[port]
        if port in visited:
            continue
        visited.update((port, antipode))
        forward = by_port[port]
        backward = by_port[antipode]
        inverse = _inverse(forward)
        differing = [
            carrier for carrier in range(8) if backward[carrier] != inverse[carrier]
        ]
        if not differing:
            raise FeasibilityError("an antipodal source pair unexpectedly satisfies inversion")
        witness = differing[0]
        pair_rows.append(
            {
                "port": port,
                "antipodal_port": antipode,
                "S_p": list(forward),
                "S_p_sha256": _sha(list(forward)),
                "S_antipode_p": list(backward),
                "S_antipode_p_sha256": _sha(list(backward)),
                "inverse_S_p": list(inverse),
                "inverse_S_p_sha256": _sha(list(inverse)),
                "first_differing_carrier_index": witness,
                "S_antipode_p_at_witness": backward[witness],
                "inverse_S_p_at_witness": inverse[witness],
                "antipodal_source_matching_differs_from_inverse": True,
                "source_repair_event_ids_for_p": sorted(events_by_port[port]),
                "source_repair_event_ids_for_antipode_p": sorted(
                    events_by_port[antipode]
                ),
            }
        )
    pair_rows.sort(key=lambda row: row["port"])
    if len(pair_rows) != 6:
        raise FeasibilityError("source antipode census does not contain six pairs")

    return {
        "site_extension_class": (
            "nonempty site sets X with a surjective carrier projection pi, twelve "
            "bijective maps T_p, and exact semiconjugacy pi_after_T_p="
            "S_p_after_pi for every emitted carrier matching"
        ),
        "exact_implication": (
            "T_antipode_p=T_p_inverse and pi_after_T_p=S_p_after_pi imply "
            "S_antipode_p=S_p_inverse by surjectivity of pi"
        ),
        "proof_steps": [
            "pi_after_T_antipode_p=S_antipode_p_after_pi",
            "T_antipode_p=T_p_inverse",
            "pi_after_T_p_inverse=S_p_inverse_after_pi",
            "therefore_S_antipode_p_after_pi=S_p_inverse_after_pi",
            "surjective_pi_implies_S_antipode_p=S_p_inverse",
        ],
        "carrier_count": 8,
        "port_count": 12,
        "antipodal_pair_count": 6,
        "source_repair_event_count": 48,
        "source_repair_events_per_port": 4,
        "pair_rows": pair_rows,
        "pair_rows_sha256": _sha(pair_rows),
        "all_source_matchings_are_fixed_point_free_involutions": True,
        "all_six_antipodal_source_matchings_differ_from_required_inverse": True,
        "semiconjugate_noncollapsed_site_cover_can_satisfy_inverse_law": False,
        "finite_or_infinite_site_cardinality_changes_the_obstruction": False,
    }


def _free_abelian_positive_control(
    antipodes: Sequence[int],
) -> dict[str, Any]:
    """Construct the universal six-axis grammar and one executable quotient.

    The universal object is the free abelian group on the ports modulo
    ``e_-p + e_p = 0``, hence ``Z^6``.  Reduction modulo three is its smallest
    finite cyclic quotient in which ``+e_i`` and ``-e_i`` remain distinct.  It
    checks the inverse and covariance grammar without pretending that the
    source emitted or selected this site completion.
    """

    normalized_antipodes = tuple(antipodes)
    pairs = tuple(
        sorted(
            (min(port, normalized_antipodes[port]), max(port, normalized_antipodes[port]))
            for port in range(12)
            if port < normalized_antipodes[port]
        )
    )
    if len(pairs) != 6 or len({item for pair in pairs for item in pair}) != 12:
        raise FeasibilityError("positive control requires six antipodal port pairs")
    axis_for_port: dict[int, tuple[int, int]] = {}
    for axis, (positive, negative) in enumerate(pairs):
        axis_for_port[positive] = (axis, 1)
        axis_for_port[negative] = (axis, -1)

    actions = tuple(icosahedral_a5_port_permutations())
    if (
        len(actions) != 60
        or len(set(actions)) != 60
        or any(sorted(action) != list(range(12)) for action in actions)
    ):
        raise FeasibilityError("proper icosahedral port action is not an exact order-60 family")
    action_set = set(actions)
    identity12 = tuple(range(12))
    if identity12 not in action_set or any(
        _compose(left, right) not in action_set
        for left in actions
        for right in actions
    ):
        raise FeasibilityError("proper icosahedral port action is not closed")
    if any(
        action[normalized_antipodes[port]]
        != normalized_antipodes[action[port]]
        for action in actions
        for port in range(12)
    ):
        raise FeasibilityError("proper A5 action does not preserve port antipodes")

    modulus = 3
    sites = tuple(itertools.product(range(modulus), repeat=6))
    site_index = {site: index for index, site in enumerate(sites)}
    if len(sites) != 729 or len(site_index) != 729:
        raise FeasibilityError("free-abelian control site census drifted")

    directions: list[tuple[int, ...]] = []
    transport_permutations: list[tuple[int, ...]] = []
    for port in range(12):
        axis, sign = axis_for_port[port]
        direction = tuple(sign if index == axis else 0 for index in range(6))
        directions.append(direction)
        transport_permutations.append(
            tuple(
                site_index[
                    tuple(
                        (coordinate + direction[index]) % modulus
                        for index, coordinate in enumerate(site)
                    )
                ]
                for site in sites
            )
        )
    if any(
        sorted(permutation) != list(range(len(sites)))
        for permutation in transport_permutations
    ):
        raise FeasibilityError("free-abelian transport row is not a site permutation")
    if any(
        transport_permutations[normalized_antipodes[port]]
        != _inverse(transport_permutations[port])
        for port in range(12)
    ):
        raise FeasibilityError("free-abelian control fails antipodal inversion")

    site_actions: list[tuple[int, ...]] = []
    signed_action_rows: list[dict[str, Any]] = []
    for group_index, action in enumerate(actions):
        targets: list[int] = []
        signs: list[int] = []
        for axis, (positive, _) in enumerate(pairs):
            target_axis, target_sign = axis_for_port[action[positive]]
            targets.append(target_axis)
            signs.append(target_sign)
        if sorted(targets) != list(range(6)):
            raise FeasibilityError("A5 port action does not induce a signed axis permutation")

        def act(site: Sequence[int]) -> tuple[int, ...]:
            result = [0] * 6
            for source_axis, coordinate in enumerate(site):
                result[targets[source_axis]] = (
                    signs[source_axis] * coordinate
                ) % modulus
            return tuple(result)

        site_action = tuple(site_index[act(site)] for site in sites)
        if sorted(site_action) != list(range(len(sites))):
            raise FeasibilityError("induced A5 site action is not a permutation")
        site_actions.append(site_action)
        signed_action_rows.append(
            {
                "group_element_index": group_index,
                "port_permutation_sha256": _sha(list(action)),
                "axis_targets": targets,
                "axis_signs": signs,
                "site_permutation_sha256": _sha(list(site_action)),
            }
        )
    if len(set(site_actions)) != 60:
        raise FeasibilityError("induced A5 action on the control sites is not faithful")
    site_action_set = set(site_actions)
    if any(
        _compose(left, right) not in site_action_set
        for left in site_actions
        for right in site_actions
    ):
        raise FeasibilityError("induced A5 site-action family is not closed")
    covariance_checks = 0
    for group_index, action in enumerate(actions):
        site_action = site_actions[group_index]
        inverse_site_action = _inverse(site_action)
        for port in range(12):
            conjugated = _compose(
                site_action,
                _compose(transport_permutations[port], inverse_site_action),
            )
            if conjugated != transport_permutations[action[port]]:
                raise FeasibilityError("free-abelian control fails exact A5 covariance")
            covariance_checks += 1

    # Every independent reversal of the six displayed axis representatives is
    # a diagonal basis conjugation.  Recheck the signed-vector covariance for
    # all 2^6 conventions so the serialized lower-index convention cannot be
    # mistaken for physical content.
    presentation_covariance_checks = 0
    presentation_inverse_checks = 0
    for flips in itertools.product((-1, 1), repeat=6):
        transformed_directions = [
            tuple(flips[index] * value for index, value in enumerate(direction))
            for direction in directions
        ]
        for port in range(12):
            antipode = normalized_antipodes[port]
            if transformed_directions[antipode] != tuple(
                -value for value in transformed_directions[port]
            ):
                raise FeasibilityError("axis-sign presentation changed antipodal inversion")
            presentation_inverse_checks += 1
        for group_index, action in enumerate(actions):
            targets = signed_action_rows[group_index]["axis_targets"]
            signs = signed_action_rows[group_index]["axis_signs"]
            transformed_signs = [
                flips[source_axis]
                * signs[source_axis]
                * flips[targets[source_axis]]
                for source_axis in range(6)
            ]
            for port in range(12):
                source = transformed_directions[port]
                result = [0] * 6
                for source_axis, value in enumerate(source):
                    result[targets[source_axis]] = (
                        transformed_signs[source_axis] * value
                    )
                if tuple(result) != transformed_directions[action[port]]:
                    raise FeasibilityError("axis-sign presentation changed A5 covariance")
                presentation_covariance_checks += 1

    port_rows = [
        {
            "port": port,
            "antipodal_port": normalized_antipodes[port],
            "axis": axis_for_port[port][0],
            "orientation_sign": axis_for_port[port][1],
            "direction_vector_in_Z_power_6_basis": list(directions[port]),
            "site_permutation_sha256": _sha(list(transport_permutations[port])),
            "antipodal_map_is_exact_inverse": True,
        }
        for port in range(12)
    ]
    return {
        "schema": "oph.vertex12-free-abelian-translation-grammar-control.v1",
        "status": (
            "EXACT_INVERSE_AND_A5_COVARIANCE_CONTROL_ATTAINED__"
            "SOURCE_EVENT_AND_PHYSICAL_BINDING_ABSENT"
        ),
        "universal_site_object": "Z[P]/(e_antipode_p+e_p) isomorphic to Z^6",
        "universal_transport_formula": "T_p(x)=x+e_p with e_antipode_p=-e_p",
        "executable_control": (
            "reduction of the universal free-abelian grammar to its smallest "
            "direction-separating cyclic quotient (Z/3Z)^6"
        ),
        "site_domain": "(Z/3Z)^6",
        "site_count": len(sites),
        "site_ids_sha256": _sha([list(site) for site in sites]),
        "modulus": modulus,
        "modulus_scope": (
            "three is the smallest finite modulus distinguishing +e_i from -e_i; "
            "it is a control choice, not a source-selected physical period"
        ),
        "antipodal_axis_pairs": [list(pair) for pair in pairs],
        "presentation_convention": (
            "axis pairs are lexicographically ordered and the lower numbered port "
            "is the positive representative"
        ),
        "presentation_scope": (
            "serialized vectors and digests use this convention; inverse and A5 "
            "covariance are checked under every independent sign reversal; an "
            "axis permutation only reindexes the six serialized coordinates"
        ),
        "independent_axis_sign_choice_count": 64,
        "axis_sign_presentation_changes_algebraic_equivalence_class": False,
        "axis_sign_presentation_inverse_checks": presentation_inverse_checks,
        "axis_sign_presentation_covariance_checks": presentation_covariance_checks,
        "all_axis_sign_presentations_preserve_inverse_and_covariance": True,
        "port_rows": port_rows,
        "port_rows_sha256": _sha(port_rows),
        "transport_permutation_family_sha256": _sha(
            [list(row) for row in transport_permutations]
        ),
        "proper_A5_port_action_order": len(actions),
        "proper_A5_port_action_sha256": _sha([list(row) for row in actions]),
        "signed_axis_action_rows": signed_action_rows,
        "signed_axis_action_rows_sha256": _sha(signed_action_rows),
        "site_A5_action_family_sha256": _sha([list(row) for row in site_actions]),
        "site_A5_action_is_faithful": True,
        "site_A5_action_is_closed": True,
        "all_twelve_site_maps_are_bijections": True,
        "all_six_antipodal_pairs_are_exact_inverses": True,
        "exact_covariance_formula": "U_g T_p U_g_inverse = T_g(p)",
        "exact_covariance_checks": covariance_checks,
        "exact_covariance_attained": True,
        "port_geometry_derived_after_declared_free_abelian_completion": True,
        "source_transition_event_emitted": False,
        "repair_generated": False,
        "source_selected_site_completion": False,
        "spatial_translation": False,
        "physical_readout": False,
        "physical_prediction": False,
    }


def _payload() -> dict[str, Any]:
    upstream = _validated_upstream()
    obstruction = _obstruction(upstream)
    positive_control = _free_abelian_positive_control(
        upstream["quotient_and_spatial_boundary"]["carrier_antipode_map"]
    )
    payload = {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "upstream_source_packet": {
            "pin": _raw_pin(UPSTREAM_PATH),
            "schema": upstream["schema"],
            "status": upstream["status"],
            "receipt_sha256": upstream["receipt_sha256"],
            "source_state_root_sha256": upstream["source_capture_binding"][
                "source_state_root_sha256"
            ],
            "repair_log_sha256": upstream["source_capture_binding"][
                "repair_log_sha256"
            ],
            "packet_independently_verified": True,
            "source_engine_independently_reimplemented": False,
        },
        "exact_semiconjugacy_obstruction": obstruction,
        "algebraic_transport_positive_control": positive_control,
        "requested_directed_transport_ledger": {
            "schema": "oph.vertex12-directed-transport-ledger.v1",
            "attained_from_current_source_emissions": False,
            "twelve_event_emitted_directed_maps_attained": False,
            "exact_T_antipode_p_equals_inverse_T_p_attained": False,
            "site_A5_action_and_exact_covariance_attained": False,
            "reason": (
                "the only complete twelve-port maps in the pinned source packet "
                "are the undirected carrier matchings covered by the exact "
                "semiconjugacy obstruction"
            ),
            "inverse_and_A5_covariance_equations_algebraically_satisfiable": True,
            "positive_control_is_source_transport_ledger": False,
        },
        "minimal_source_producer_contract": {
            "required_emission_point": (
                "inside the source transition producer before capture hashing, not "
                "as a post-capture port assignment"
            ),
            "required_captured_artifact_schema": (
                "oph.vertex12-directed-transport-ledger.v1"
            ),
            "required_fields": [
                "source_capture_root_sha256_covering_the_transport_ledger",
                "nonempty_noncollapsed_site_ids",
                "twelve_port_rows_with_complete_site_permutations",
                "one_or_more_source_transition_event_ids_for_each_port_row",
                "event_payload_digest_equal_to_each_transport_digest",
                "icosahedral_antipode_map",
                "exact_T_antipode_p_equals_inverse_T_p_receipt",
                "sixty_element_proper_A5_site_action",
                "exact_U_g_T_p_U_g_inverse_equals_T_g_p_covariance_receipt",
            ],
            "must_not_claim": [
                "spatial_translation_without_a_site_to_spacetime_map",
                "physical_readout_without_a_sector_and_measurement_attachment",
                "frame_or_boost_transport_from_finite_port covariance alone",
                "a physical prediction from the transport ledger alone",
            ],
            "reuse_current_matching_maps_via_a_surjective_semiconjugate_cover": (
                "exactly_obstructed"
            ),
            "new_or_non_semiconjugate_source_emitted_"
            "oriented_law_required_for_issue_contract": True,
            "inverse_and_A5_covariance_are_jointly_consistent_in_declared_control": True,
            "source_selection_of_a_site_completion_and_event_provenance_remain_required": True,
        },
        "scope_boundary": {
            "rules_out_all_carrier_set_quotients": True,
            "rules_out_surjective_semiconjugate_covers_or_"
            "extensions_of_the_current_matchings": True,
            "rules_out_a_new_source_emitted_oriented_transport_law": False,
            "rules_out_non_semiconjugate_linear_or_state_space_transport": False,
            "rules_out_other_seeds_or_carrier_counts": False,
            "spatial_translation_attained": False,
            "physical_sector_readout_attained": False,
            "physical_prediction_unsealed": False,
        },
        "comparison_data_read": False,
        "implementation_pins": {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_verifier": _raw_pin(INDEPENDENT_VERIFIER_PATH),
            "serialized_mutation_tests": _raw_pin(TEST_PATH),
            "proper_icosahedral_action_source": _raw_pin(GEOMETRY_PATH),
        },
        "claim_boundary": (
            "The exact theorem excludes only site transports that project "
            "surjectively and port by port to all twelve undirected matching maps "
            "emitted by the pinned source repair ledger. It applies to finite and "
            "infinite covers. A genuinely oriented source transition law, a "
            "non-semiconjugate linear or enlarged state-space construction, other "
            "source configurations, and every physical attachment remain open. An "
            "exact free-abelian translation-grammar control on (Z/3Z)^6 proves that the "
            "antipodal inverse and A5 covariance equations are mutually consistent. "
            "That control is geometry-derived and declared; it is not emitted or "
            "selected by source repair events. "
            "This receipt supplies no spatial translation, sector readout, frame, "
            "boost, or physical prediction."
        ),
    }
    return payload


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
        if report.get("status") != STATUS:
            reasons.append("status_mismatch")
        requested = report.get("requested_directed_transport_ledger")
        scope = report.get("scope_boundary")
        control = report.get("algebraic_transport_positive_control")
        if not isinstance(requested, Mapping) or any(
            requested.get(key) is not False
            for key in (
                "attained_from_current_source_emissions",
                "twelve_event_emitted_directed_maps_attained",
                "exact_T_antipode_p_equals_inverse_T_p_attained",
                "site_A5_action_and_exact_covariance_attained",
            )
        ):
            reasons.append("unsupported_transport_promotion")
        if not isinstance(scope, Mapping) or any(
            scope.get(key) is not False
            for key in (
                "spatial_translation_attained",
                "physical_sector_readout_attained",
                "physical_prediction_unsealed",
            )
        ):
            reasons.append("unsupported_physical_promotion")
        if not isinstance(control, Mapping) or any(
            control.get(key) is not False
            for key in (
                "source_transition_event_emitted",
                "repair_generated",
                "source_selected_site_completion",
                "spatial_translation",
                "physical_readout",
                "physical_prediction",
            )
        ):
            reasons.append("positive_control_scope_promotion")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        FeasibilityError,
        RecursionError,
    ):
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
            "Producer replay checks the exact source-matching obstruction. It "
            "does not supply the missing directed source law or a physical bridge."
        ),
    }


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = _load_json(args.verify)
        verification = verify_receipt(report)
        _write_json(verification, args.output)
        return 0 if verification["receipt"] else 1
    report = produce_receipt()
    verification = verify_receipt(report)
    if not verification["receipt"]:
        _write_json(verification, args.output)
        return 1
    _write_json(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
