"""Construct a target-free source law for the issue-655 quotient packet.

The canonical simulator capture does not emit the directed reversible maps
required by issue 655.  This module therefore does not retrofit those maps
onto that capture.  It builds a separate, minimal source-law control:

* the raw state is the integer six-axis record module ``Z^6``;
* the twelve primitive source events add the signed basis vectors fixed by
  the canonical antipodal port geometry;
* the public observer meaning is the coordinatewise quotient ``(Z/3Z)^6``,
  using the smallest modulus that separates a basis step from its inverse;
* A2 naturality is componentwise reduction modulo three;
* the six inverse identities, fifteen all-state endpoint diamonds, and the
  same-quotient A5 covariance are emitted and checked exactly.

The construction is an executable consistency and producer-feasibility
control.  Canonical A1--A3 do not select this source law, and no spatial or
physical interpretation is attached.
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/vertex12_constructive_source_law_receipt.json"
)
FEASIBILITY_PATH = (
    ROOT / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
ENDPOINT_PATH = (
    ROOT / "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json"
)
VERIFIER_PATH = (
    ROOT / "oph_fpe/dynamics/verify_vertex12_constructive_source_law_independent.py"
)
TEST_PATH = ROOT / "tests/test_vertex12_constructive_source_law.py"

SCHEMA = "oph.vertex12-constructive-source-law-control.v1"
STATUS = (
    "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ATTAINED__"
    "CANONICAL_A1_A2_A3_SELECTION_AND_PHYSICAL_ATTACHMENT_OPEN"
)
FEASIBILITY_SCHEMA = "oph.vertex12-directed-transport-feasibility.v1"
FEASIBILITY_STATUS = (
    "EXACT_SEMICONJUGATE_COVER_OBSTRUCTION_FOR_CURRENT_SOURCE_MATCHINGS__"
    "ORIENTED_SOURCE_TRANSITION_LAW_OPEN"
)
ENDPOINT_SCHEMA = "oph.vertex12-a2-endpoint-commutator-boundary.v1"
ENDPOINT_STATUS = (
    "A2_ENDPOINT_TO_QUOTIENT_COMMUTATOR_THEOREM_ATTAINED__"
    "SOURCE_NATURALITY_INVERSES_DIAMONDS_AND_PHYSICAL_ACTION_OPEN"
)
MODULUS = 3
AXIS_COUNT = 6
PORT_COUNT = 12


class ConstructiveSourceLawError(RuntimeError):
    """Raised when the constructive source-law packet fails closed."""


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConstructiveSourceLawError(f"{path} is not a JSON object")
    return value


def _compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ConstructiveSourceLawError("cannot compose different state spaces")
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    if sorted(permutation) != list(range(len(permutation))):
        raise ConstructiveSourceLawError("row is not a permutation")
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _validated_parents() -> tuple[dict[str, Any], dict[str, Any]]:
    feasibility = _load_json(FEASIBILITY_PATH)
    endpoint = _load_json(ENDPOINT_PATH)
    if (
        feasibility.get("schema") != FEASIBILITY_SCHEMA
        or feasibility.get("status") != FEASIBILITY_STATUS
        or feasibility.get("issue") != 655
    ):
        raise ConstructiveSourceLawError("feasibility parent contract drifted")
    if (
        endpoint.get("schema") != ENDPOINT_SCHEMA
        or endpoint.get("status") != ENDPOINT_STATUS
        or endpoint.get("issue") != 655
    ):
        raise ConstructiveSourceLawError("endpoint parent contract drifted")
    attainment = endpoint.get("attainment")
    if not isinstance(attainment, Mapping) or any(
        attainment.get(key) is not False
        for key in (
            "current_source_emitted_endpoint_diamond_ledger",
            "faithful_physical_z_power_6_action",
            "spatial_translation",
            "physical_prediction",
        )
    ):
        raise ConstructiveSourceLawError("canonical-source boundary was promoted")
    return feasibility, endpoint


def _geometry(
    feasibility: Mapping[str, Any],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    control = feasibility.get("algebraic_transport_positive_control")
    if not isinstance(control, Mapping):
        raise ConstructiveSourceLawError("positive geometry control is missing")
    raw_pairs = control.get("antipodal_axis_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != AXIS_COUNT:
        raise ConstructiveSourceLawError("antipodal axis pairs are missing")
    pairs = tuple(tuple(int(value) for value in row) for row in raw_pairs)
    flattened = [port for pair in pairs for port in pair]
    if sorted(flattened) != list(range(PORT_COUNT)) or any(
        len(pair) != 2 or pair[0] >= pair[1] for pair in pairs
    ):
        raise ConstructiveSourceLawError("antipodal pairs do not partition ports")
    antipodes = [0] * PORT_COUNT
    for left, right in pairs:
        antipodes[left] = right
        antipodes[right] = left
    actions = tuple(
        tuple(int(target) for target in row)
        for row in icosahedral_a5_port_permutations()
    )
    if len(actions) != 60 or len(set(actions)) != 60:
        raise ConstructiveSourceLawError("canonical proper port action is not A5")
    if _sha([list(row) for row in actions]) != control.get(
        "proper_A5_port_action_sha256"
    ):
        raise ConstructiveSourceLawError("canonical A5 port action pin drifted")
    if any(
        action[antipodes[port]] != antipodes[action[port]]
        for action in actions
        for port in range(PORT_COUNT)
    ):
        raise ConstructiveSourceLawError("A5 action does not preserve antipodes")
    return pairs, tuple(antipodes), actions


def _sites() -> tuple[tuple[int, ...], ...]:
    sites = tuple(itertools.product(range(MODULUS), repeat=AXIS_COUNT))
    if len(sites) != MODULUS**AXIS_COUNT:
        raise ConstructiveSourceLawError("quotient site census drifted")
    return sites


def _directions(
    pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...] | None] = [None] * PORT_COUNT
    for axis, pair in enumerate(pairs):
        for port, sign in ((int(pair[0]), 1), (int(pair[1]), -1)):
            rows[port] = tuple(sign if index == axis else 0 for index in range(6))
    if any(row is None for row in rows):
        raise ConstructiveSourceLawError("signed direction family is incomplete")
    return tuple(row for row in rows if row is not None)


def _meaning_steps(
    sites: Sequence[Sequence[int]], directions: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    site_index = {tuple(site): index for index, site in enumerate(sites)}
    output = []
    for direction in directions:
        output.append(
            tuple(
                site_index[
                    tuple(
                        (coordinate + direction[axis]) % MODULUS
                        for axis, coordinate in enumerate(site)
                    )
                ]
                for site in sites
            )
        )
    if any(sorted(row) != list(range(len(sites))) for row in output):
        raise ConstructiveSourceLawError("meaning-side step is not bijective")
    return tuple(output)


def _signed_axis_action(
    action: Sequence[int],
    pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    port_to_axis: dict[int, tuple[int, int]] = {}
    for axis, pair in enumerate(pairs):
        port_to_axis[int(pair[0])] = (axis, 1)
        port_to_axis[int(pair[1])] = (axis, -1)
    targets = []
    signs = []
    for positive, _negative in pairs:
        axis, sign = port_to_axis[int(action[int(positive)])]
        targets.append(axis)
        signs.append(sign)
    if sorted(targets) != list(range(AXIS_COUNT)):
        raise ConstructiveSourceLawError("port action does not induce signed axes")
    return tuple(targets), tuple(signs)


def _meaning_a5_actions(
    sites: Sequence[Sequence[int]],
    actions: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    site_index = {tuple(site): index for index, site in enumerate(sites)}
    output = []
    for action in actions:
        targets, signs = _signed_axis_action(action, pairs)
        permutation = []
        for site in sites:
            result = [0] * AXIS_COUNT
            for source_axis, coordinate in enumerate(site):
                result[targets[source_axis]] = (
                    signs[source_axis] * int(coordinate)
                ) % MODULUS
            permutation.append(site_index[tuple(result)])
        output.append(tuple(permutation))
    if len(set(output)) != 60 or any(
        sorted(row) != list(range(len(sites))) for row in output
    ):
        raise ConstructiveSourceLawError("meaning-side A5 action is not faithful")
    return tuple(output)


def _source_event_rows(
    directions: Sequence[Sequence[int]],
    antipodes: Sequence[int],
    meaning_steps: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    rows = []
    for port in range(PORT_COUNT):
        payload = {
            "schema": "oph.vertex12-signed-axis-source-event.v1",
            "event_kind": "accepted_signed_axis_unit_record",
            "port": port,
            "antipodal_port": int(antipodes[port]),
            "raw_direction_in_Z_power_6": list(directions[port]),
            "raw_step": "T_p(n)=n+v_p on Z^6",
            "meaning_step": "tau_p([n]_3)=[n+v_p]_3",
            "meaning_step_permutation_sha256": _sha(list(meaning_steps[port])),
            "emission_phase": "inside_constructive_source_law_before_capture_hash",
            "comparison_or_target_input_used": False,
        }
        rows.append({**payload, "event_id": _sha(payload)})
    if len({row["event_id"] for row in rows}) != PORT_COUNT:
        raise ConstructiveSourceLawError("source event identifiers are not unique")
    return rows


def _capture(
    feasibility: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    pairs: Sequence[Sequence[int]],
) -> dict[str, Any]:
    payload = {
        "schema": "oph.vertex12-constructive-source-capture.v1",
        "source_law_id": "signed-six-axis-record-translation-control",
        "raw_state": "Z^6",
        "observer_meaning": "(Z/3Z)^6",
        "quotient_formula": "q(n_0,...,n_5)=(n_0 mod 3,...,n_5 mod 3)",
        "antipodal_axis_pairs": [list(pair) for pair in pairs],
        "event_rows": list(events),
        "a1_contract": (
            "the complete accepted primitive alphabet is one signed unit-record "
            "event for each of the twelve canonical ports"
        ),
        "a2_contract": (
            "observer meaning is reduction modulo three and every accepted step "
            "must descend through that interpretation"
        ),
        "a3_contract": (
            "the declared exact counting reference is uniform on the complete "
            "twelve-event alphabet; no event is selected by downstream data"
        ),
        "parent_receipts": {
            "feasibility_receipt_sha256": feasibility["receipt_sha256"],
            "endpoint_receipt_sha256": endpoint["receipt_sha256"],
        },
    }
    return {
        "payload": payload,
        "source_capture_root_sha256": _sha(payload),
        "event_count": len(events),
        "event_ids_emitted_before_capture_hash": True,
        "capture_hash_binds_every_event_payload": True,
    }


def _payload() -> dict[str, Any]:
    feasibility, endpoint = _validated_parents()
    pairs, antipodes, actions = _geometry(feasibility)
    sites = _sites()
    directions = _directions(pairs)
    meaning_steps = _meaning_steps(sites, directions)
    meaning_a5_actions = _meaning_a5_actions(sites, actions, pairs)
    events = _source_event_rows(directions, antipodes, meaning_steps)
    capture = _capture(feasibility, endpoint, events, pairs)
    identity = tuple(range(len(sites)))

    q_contract = {
        "raw_state_domain": "Z^6",
        "meaning_state_domain": "(Z/3Z)^6",
        "meaning_state_count": len(sites),
        "meaning_states": [list(site) for site in sites],
        "q_formula": "componentwise_reduction_modulo_3",
        "surjective": True,
        "surjectivity_witnesses": [
            {"meaning_state_index": index, "raw_integer_representative": list(site)}
            for index, site in enumerate(sites)
        ],
        "q_is_spatial_readout": False,
        "q_is_physical_readout": False,
    }
    raw_step_rows = [
        {
            "port": port,
            "event_id": events[port]["event_id"],
            "direction": list(directions[port]),
            "formula": "T_p(n)=n+direction",
            "bijective_on_Z_power_6": True,
            "inverse_port": int(antipodes[port]),
        }
        for port in range(PORT_COUNT)
    ]
    meaning_step_rows = [
        {
            "port": port,
            "event_id": events[port]["event_id"],
            "permutation": list(meaning_steps[port]),
            "permutation_sha256": _sha(list(meaning_steps[port])),
            "bijective": True,
        }
        for port in range(PORT_COUNT)
    ]
    descent_rows = [
        {
            "port": port,
            "event_id": events[port]["event_id"],
            "identity": "q(T_p(n))=tau_p(q(n)) for every n in Z^6",
            "proof": "componentwise integer congruence modulo three",
            "complete_residue_representative_count": len(sites),
            "complete_residue_endpoint_indices": list(meaning_steps[port]),
            "exact_A2_naturality_square": True,
        }
        for port in range(PORT_COUNT)
    ]

    inverse_rows = []
    for positive, negative in pairs:
        forward_reverse = _compose(meaning_steps[negative], meaning_steps[positive])
        reverse_forward = _compose(meaning_steps[positive], meaning_steps[negative])
        if forward_reverse != identity or reverse_forward != identity:
            raise ConstructiveSourceLawError("quotient inverse identity failed")
        inverse_rows.append(
            {
                "positive_port": int(positive),
                "negative_port": int(negative),
                "tau_negative_equals_tau_positive_inverse": True,
                "forward_then_reverse_endpoints": list(forward_reverse),
                "reverse_then_forward_endpoints": list(reverse_forward),
            }
        )

    positive_ports = [int(pair[0]) for pair in pairs]
    diamond_rows = []
    for left, right in itertools.combinations(positive_ports, 2):
        left_after_right = _compose(meaning_steps[left], meaning_steps[right])
        right_after_left = _compose(meaning_steps[right], meaning_steps[left])
        if left_after_right != right_after_left:
            raise ConstructiveSourceLawError("positive-axis endpoint diamond failed")
        diamond_rows.append(
            {
                "left_port": left,
                "right_port": right,
                "all_state_count": len(sites),
                "left_after_right_endpoints": list(left_after_right),
                "right_after_left_endpoints": list(right_after_left),
                "all_state_endpoints_equal": True,
            }
        )

    group_rows = []
    covariance_rows = []
    for group_index, (port_action, q_action) in enumerate(
        zip(actions, meaning_a5_actions, strict=True)
    ):
        axis_targets, axis_signs = _signed_axis_action(port_action, pairs)
        group_rows.append(
            {
                "group_element_index": group_index,
                "port_permutation": list(port_action),
                "axis_targets": list(axis_targets),
                "axis_signs": list(axis_signs),
                "meaning_permutation": list(q_action),
                "meaning_permutation_sha256": _sha(list(q_action)),
            }
        )
        q_inverse = _inverse(q_action)
        for port in range(PORT_COUNT):
            conjugated = _compose(
                q_action,
                _compose(meaning_steps[port], q_inverse),
            )
            target = meaning_steps[port_action[port]]
            if conjugated != target:
                raise ConstructiveSourceLawError("same-Q A5 covariance failed")
            covariance_rows.append(
                {
                    "group_element_index": group_index,
                    "port": port,
                    "target_port": int(port_action[port]),
                    "conjugated_permutation_sha256": _sha(list(conjugated)),
                    "target_permutation_sha256": _sha(list(target)),
                    "exact": True,
                }
            )

    implementation_files = [Path(__file__).resolve(), VERIFIER_PATH, TEST_PATH]
    return {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "comparison_data_read": False,
        "canonical_parent_pins": {
            "feasibility": {
                "schema": feasibility["schema"],
                "status": feasibility["status"],
                "receipt_sha256": feasibility["receipt_sha256"],
                "raw_pin": _raw_pin(FEASIBILITY_PATH),
            },
            "endpoint": {
                "schema": endpoint["schema"],
                "status": endpoint["status"],
                "receipt_sha256": endpoint["receipt_sha256"],
                "raw_pin": _raw_pin(ENDPOINT_PATH),
            },
        },
        "canonical_source_audit": {
            "canonical_source_emits_requested_packet": False,
            "canonical_source_emits_accepted_observer_quotient": False,
            "canonical_source_emits_A2_naturality_rows": False,
            "canonical_source_emits_endpoint_diamonds": False,
            "reason": (
                "the canonical packet emits noninvertible disjoint-port repair "
                "projectors and fails the directed inverse-compatible source gate"
            ),
            "current_source_retrofit_performed": False,
        },
        "constructive_source_law": {
            "source_capture": capture,
            "a1_complete_event_alphabet": {
                "event_count": len(events),
                "event_rows": events,
                "every_event_accepted_before_capture_hash": True,
                "complete_signed_port_orbit": True,
            },
            "a2_interpretation": q_contract,
            "a3_counting_reference": {
                "event_count": len(events),
                "weight_per_event": "1/12",
                "reference_fixed_before_any_downstream_comparison": True,
                "unique_information_projection_only_under_declared_counting_reference": True,
                "canonical_A3_alone_selects_this_source_law": False,
            },
            "raw_step_rows": raw_step_rows,
            "meaning_step_rows": meaning_step_rows,
            "a2_descent_rows": descent_rows,
            "antipodal_inverse_rows": inverse_rows,
            "positive_axis_endpoint_diamond_rows": diamond_rows,
            "same_Q_A5_action": {
                "group_order": len(actions),
                "group_rows": group_rows,
                "action_is_faithful_on_Q": True,
                "covariance_formula": "U_g tau_p U_g_inverse=tau_g(p)",
                "covariance_rows": covariance_rows,
                "exact_covariance_check_count": len(covariance_rows),
            },
        },
        "attainment": {
            "constructive_source_law_capture_root": True,
            "accepted_surjective_quotient": True,
            "twelve_raw_steps": True,
            "twelve_meaning_steps": True,
            "twelve_A2_descent_squares": True,
            "six_quotient_inverse_identities": True,
            "fifteen_complete_all_state_endpoint_diamonds": True,
            "same_Q_A5_covariance": True,
            "universal_z_power_6_factorization": True,
            "canonical_source_selection": False,
            "spatial_translation": False,
            "physical_readout": False,
            "physical_prediction": False,
        },
        "issue_655_disposition": {
            "advances_algebraic_producer_feasibility": True,
            "advances_canonical_source_bridge": False,
            "advances_physical_bridge": False,
            "issue_closure_supported": False,
            "classification": "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ONLY",
            "next_gate": (
                "derive or source-select the signed six-axis record law inside "
                "the canonical simulator, then attach the digest-identical action "
                "to a physical sector, readout, frame, and boost law"
            ),
        },
        "provenance_boundary": {
            "new_constructive_source_law": True,
            "post_capture_assignment_to_canonical_source": False,
            "canonical_A1_A2_A3_derivation_claimed": False,
            "full_canonical_A1_typed_object_instantiated": False,
            "full_A2_observer_federation_functor_instantiated": False,
            "canonical_A3_maximum_entropy_selection_instantiated": False,
            "additional_branch_contract_required": True,
            "modulus_three_scope": (
                "three is the smallest coordinate modulus that distinguishes "
                "+e_i from -e_i; this does not assert a globally smallest finite "
                "A5-equivariant quotient or a source-selected physical period"
            ),
            "spatial_identification_claimed": False,
            "physical_attachment_claimed": False,
            "comparison_or_target_data_used": False,
        },
        "implementation_pins": [_raw_pin(path) for path in implementation_files],
        "claim_boundary": (
            "The requested X,Q,q,T_p,tau_p algebraic packet is attained on a "
            "separate target-free constructive source law. The canonical source "
            "still does not emit or select that law, and this control does not "
            "instantiate the full canonical A1 object, A2 federation functor, or "
            "A3 selector. The finite observer meaning is a quotient of integer "
            "record coordinates, not a spatial or physical readout. This result "
            "is a producer-feasibility control and does not close issue 655 or "
            "promote FZ-11."
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
        if report.get("status") != STATUS or report.get("issue") != 655:
            reasons.append("identity_or_status_mismatch")
        attainment = report.get("attainment")
        if not isinstance(attainment, Mapping) or any(
            attainment.get(key) is not False
            for key in (
                "canonical_source_selection",
                "spatial_translation",
                "physical_readout",
                "physical_prediction",
            )
        ):
            reasons.append("unsupported_promotion")
        if report.get("comparison_data_read") is not False:
            reasons.append("comparison_boundary_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ConstructiveSourceLawError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    return {
        "schema": "oph.vertex12-constructive-source-law-control-verification.v1",
        "receipt": not reasons,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_replay": True,
        "independent_implementation": False,
        "claim_boundary": (
            "Producer replay checks a constructive control and does not select "
            "it inside the canonical source or attach it physically."
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
        result = verify_receipt(_load_json(args.verify))
        _write_json(result, args.output)
        return 0 if result["receipt"] else 1
    receipt = produce_receipt()
    result = verify_receipt(receipt)
    if not result["receipt"]:
        _write_json(result, args.output)
        return 1
    _write_json(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
