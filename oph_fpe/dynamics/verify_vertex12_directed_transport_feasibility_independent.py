"""Independent verifier for the issue-655 transport feasibility packet.

This module does not import the producer.  It rechecks the pinned upstream
packet, the semiconjugacy obstruction, and the free-abelian positive control
from the exact twelve-port action.
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
VERIFICATION_SCHEMA = (
    "oph.vertex12-directed-transport-feasibility-independent-verification.v1"
)
STATUS = (
    "EXACT_SEMICONJUGATE_COVER_OBSTRUCTION_FOR_CURRENT_SOURCE_MATCHINGS__"
    "ORIENTED_SOURCE_TRANSITION_LAW_OPEN"
)
UPSTREAM_SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket.v1"
UPSTREAM_STATUS = (
    "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__"
    "SPATIAL_PHYSICAL_BRIDGE_OPEN"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
UPSTREAM_PATH = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
PRODUCER_PATH = (
    REPOSITORY_ROOT
    / "oph_fpe/dynamics/vertex12_directed_transport_feasibility.py"
)
VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests/test_vertex12_directed_transport_feasibility.py"
GEOMETRY_PATH = REPOSITORY_ROOT / "oph_fpe/core/icosahedral.py"


class VerificationError(ValueError):
    """Raised when exact packet reconstruction fails."""


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
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(value, dict):
        raise VerificationError("receipt JSON root is not an object")
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
        raise VerificationError("row is not a permutation")
    result = [0] * size
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise VerificationError("permutation size mismatch")
    return tuple(left[right[index]] for index in range(len(left)))


def _validated_upstream() -> dict[str, Any]:
    report = _load_json(UPSTREAM_PATH)
    check = verify_vertex12_atomic_port_transfer_independent.verify_report(report)
    if check.get("receipt") is not True:
        raise VerificationError("upstream packet failed independent verification")
    if (
        report.get("schema") != UPSTREAM_SCHEMA
        or report.get("status") != UPSTREAM_STATUS
        or type(report.get("issue")) is not int
        or report.get("issue") != 655
    ):
        raise VerificationError("upstream packet contract drifted")
    return report


def _recompute_obstruction(upstream: Mapping[str, Any]) -> dict[str, Any]:
    operator = upstream["atomic_transfer_operator"]
    boundary = upstream["quotient_and_spatial_boundary"]
    history = upstream["source_history_replay"]
    if not all(isinstance(item, Mapping) for item in (operator, boundary, history)):
        raise VerificationError("upstream blocks are malformed")
    carrier_ids = operator.get("carrier_ids")
    rows = operator.get("port_rows")
    antipodes = boundary.get("carrier_antipode_map")
    events = history.get("event_rows")
    if (
        not isinstance(carrier_ids, list)
        or len(carrier_ids) != 8
        or len(set(carrier_ids)) != 8
        or not isinstance(rows, list)
        or len(rows) != 12
        or not isinstance(antipodes, list)
        or len(antipodes) != 12
        or not isinstance(events, list)
        or len(events) != 48
    ):
        raise VerificationError("upstream census drifted")
    maps: dict[int, tuple[int, ...]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("port")) is not int:
            raise VerificationError("port row malformed")
        port = row["port"]
        values = row.get("carrier_partner_permutation")
        if port in maps or not isinstance(values, list) or len(values) != 8:
            raise VerificationError("port map duplicated or missing")
        permutation = tuple(values)
        if any(type(item) is not int for item in permutation):
            raise VerificationError("port map has noninteger entry")
        if _inverse(permutation) != permutation or any(
            permutation[index] == index for index in range(8)
        ):
            raise VerificationError("port map is not a fixed-point-free involution")
        maps[port] = permutation
    if set(maps) != set(range(12)):
        raise VerificationError("port map census is incomplete")
    normalized_antipodes = tuple(antipodes)
    if (
        any(type(item) is not int or not 0 <= item < 12 for item in normalized_antipodes)
        or any(normalized_antipodes[normalized_antipodes[p]] != p for p in range(12))
        or any(normalized_antipodes[p] == p for p in range(12))
    ):
        raise VerificationError("antipode map malformed")
    events_by_port: dict[int, list[str]] = {port: [] for port in range(12)}
    for row in events:
        if (
            not isinstance(row, Mapping)
            or type(row.get("port")) is not int
            or row["port"] not in events_by_port
            or type(row.get("event_id")) is not str
            or not row["event_id"].startswith("sha256:")
        ):
            raise VerificationError("repair-event row malformed")
        events_by_port[row["port"]].append(row["event_id"])
    if any(len(ids) != 4 or len(set(ids)) != 4 for ids in events_by_port.values()):
        raise VerificationError("source event coverage drifted")

    pair_rows = []
    visited: set[int] = set()
    for port in range(12):
        antipode = normalized_antipodes[port]
        if port in visited:
            continue
        visited.update((port, antipode))
        forward = maps[port]
        backward = maps[antipode]
        inverse = _inverse(forward)
        differing = [index for index in range(8) if backward[index] != inverse[index]]
        if not differing:
            raise VerificationError("antipodal inverse obstruction disappeared")
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
                "source_repair_event_ids_for_antipode_p": sorted(events_by_port[antipode]),
            }
        )
    pair_rows.sort(key=lambda row: row["port"])
    if len(pair_rows) != 6:
        raise VerificationError("antipodal pair count drifted")
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


def _recompute_positive_control(antipodes: Sequence[int]) -> dict[str, Any]:
    normalized_antipodes = tuple(antipodes)
    pairs = tuple(
        sorted(
            (min(port, normalized_antipodes[port]), max(port, normalized_antipodes[port]))
            for port in range(12)
            if port < normalized_antipodes[port]
        )
    )
    if len(pairs) != 6:
        raise VerificationError("positive-control antipode census drifted")
    axis_for_port: dict[int, tuple[int, int]] = {}
    for axis, (positive, negative) in enumerate(pairs):
        axis_for_port[positive] = (axis, 1)
        axis_for_port[negative] = (axis, -1)
    actions = tuple(icosahedral_a5_port_permutations())
    action_set = set(actions)
    if (
        len(actions) != 60
        or len(action_set) != 60
        or tuple(range(12)) not in action_set
        or any(sorted(action) != list(range(12)) for action in actions)
        or any(_compose(left, right) not in action_set for left in actions for right in actions)
        or any(
            action[normalized_antipodes[port]] != normalized_antipodes[action[port]]
            for action in actions
            for port in range(12)
        )
    ):
        raise VerificationError("proper A5 port-action audit failed")
    modulus = 3
    sites = tuple(itertools.product(range(modulus), repeat=6))
    index = {site: position for position, site in enumerate(sites)}
    directions = []
    transports = []
    for port in range(12):
        axis, sign = axis_for_port[port]
        direction = tuple(sign if item == axis else 0 for item in range(6))
        directions.append(direction)
        transports.append(
            tuple(
                index[tuple((value + direction[item]) % modulus for item, value in enumerate(site))]
                for site in sites
            )
        )
    if any(
        transports[normalized_antipodes[port]] != _inverse(transports[port])
        for port in range(12)
    ):
        raise VerificationError("positive control inverse audit failed")
    site_actions = []
    signed_rows = []
    for group_index, action in enumerate(actions):
        targets = []
        signs = []
        for positive, _ in pairs:
            target_axis, target_sign = axis_for_port[action[positive]]
            targets.append(target_axis)
            signs.append(target_sign)
        if sorted(targets) != list(range(6)):
            raise VerificationError("signed axis action malformed")

        def act(site: Sequence[int]) -> tuple[int, ...]:
            result = [0] * 6
            for source_axis, coordinate in enumerate(site):
                result[targets[source_axis]] = signs[source_axis] * coordinate % modulus
            return tuple(result)

        site_action = tuple(index[act(site)] for site in sites)
        site_actions.append(site_action)
        signed_rows.append(
            {
                "group_element_index": group_index,
                "port_permutation_sha256": _sha(list(action)),
                "axis_targets": targets,
                "axis_signs": signs,
                "site_permutation_sha256": _sha(list(site_action)),
            }
        )
    site_action_set = set(site_actions)
    if len(site_action_set) != 60 or any(
        _compose(left, right) not in site_action_set
        for left in site_actions
        for right in site_actions
    ):
        raise VerificationError("positive-control A5 site action failed")
    covariance_checks = 0
    for group_index, action in enumerate(actions):
        site_action = site_actions[group_index]
        inverse_action = _inverse(site_action)
        for port in range(12):
            if _compose(
                site_action, _compose(transports[port], inverse_action)
            ) != transports[action[port]]:
                raise VerificationError("positive-control covariance failed")
            covariance_checks += 1
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
                raise VerificationError("presentation inverse audit failed")
            presentation_inverse_checks += 1
        for group_index, action in enumerate(actions):
            targets = signed_rows[group_index]["axis_targets"]
            signs = signed_rows[group_index]["axis_signs"]
            transformed_signs = [
                flips[source_axis] * signs[source_axis] * flips[targets[source_axis]]
                for source_axis in range(6)
            ]
            for port in range(12):
                result = [0] * 6
                for source_axis, value in enumerate(transformed_directions[port]):
                    result[targets[source_axis]] = transformed_signs[source_axis] * value
                if tuple(result) != transformed_directions[action[port]]:
                    raise VerificationError("presentation covariance audit failed")
                presentation_covariance_checks += 1
    port_rows = [
        {
            "port": port,
            "antipodal_port": normalized_antipodes[port],
            "axis": axis_for_port[port][0],
            "orientation_sign": axis_for_port[port][1],
            "direction_vector_in_Z_power_6_basis": list(directions[port]),
            "site_permutation_sha256": _sha(list(transports[port])),
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
        "transport_permutation_family_sha256": _sha([list(row) for row in transports]),
        "proper_A5_port_action_order": len(actions),
        "proper_A5_port_action_sha256": _sha([list(row) for row in actions]),
        "signed_axis_action_rows": signed_rows,
        "signed_axis_action_rows_sha256": _sha(signed_rows),
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


def _expected_payload() -> dict[str, Any]:
    upstream = _validated_upstream()
    obstruction = _recompute_obstruction(upstream)
    control = _recompute_positive_control(
        upstream["quotient_and_spatial_boundary"]["carrier_antipode_map"]
    )
    return {
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
            "repair_log_sha256": upstream["source_capture_binding"]["repair_log_sha256"],
            "packet_independently_verified": True,
            "source_engine_independently_reimplemented": False,
        },
        "exact_semiconjugacy_obstruction": obstruction,
        "algebraic_transport_positive_control": control,
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
            "required_captured_artifact_schema": "oph.vertex12-directed-transport-ledger.v1",
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
            "reuse_current_matching_maps_via_a_surjective_"
            "semiconjugate_cover": "exactly_obstructed",
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
            "independent_verifier": _raw_pin(VERIFIER_PATH),
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
            "selected by source repair events. This receipt supplies no spatial "
            "translation, sector readout, frame, boost, or physical prediction."
        ),
    }


def verify_report(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        received = copy.deepcopy(dict(report))
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        if _canonical_bytes(received) != _canonical_bytes(_expected_payload()):
            reasons.append("independent_reconstruction_mismatch")
        if report.get("status") != STATUS:
            reasons.append("status_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        VerificationError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "packet_analysis_independently_reimplemented": True,
        "source_engine_independently_reimplemented": False,
        "claim_boundary": (
            "The verifier independently reconstructs the finite algebra and exact "
            "obstruction. It does not provide source event provenance or physics."
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = _load_json(args.input)
    verification = verify_report(report)
    _write_json(verification, args.output)
    return 0 if verification["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
