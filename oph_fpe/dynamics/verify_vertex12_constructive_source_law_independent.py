"""Independent verifier for the issue-655 constructive source-law control.

This verifier does not import the producer.  It reconstructs the finite
quotient, all twelve maps, the naturality representatives, inverse rows,
endpoint diamonds, and A5 covariance directly from the canonical port action
and the serialized parent receipts.
"""

from __future__ import annotations

import argparse
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
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/vertex12_constructive_source_law.py"
INDEPENDENT_VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_vertex12_constructive_source_law.py"
FEASIBILITY_PATH = (
    ROOT / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
ENDPOINT_PATH = (
    ROOT / "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json"
)

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
AXES = 6
PORTS = 12


class IndependentVerificationError(RuntimeError):
    """Raised when the independent source-law audit fails closed."""


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"{path} is not an object")
    return value


def _same(left: Any, right: Any) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    _fail(set(value) == expected, f"{label} key set")


def _compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    _fail(len(left) == len(right), "composition domain mismatch")
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def _inverse(row: Sequence[int]) -> tuple[int, ...]:
    _fail(
        sorted(int(value) for value in row) == list(range(len(row))),
        "not a permutation",
    )
    result = [0] * len(row)
    for source, target in enumerate(row):
        result[int(target)] = source
    return tuple(result)


def _parent_contract(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    feasibility = _load(FEASIBILITY_PATH)
    endpoint = _load(ENDPOINT_PATH)
    _fail(
        feasibility.get("schema") == FEASIBILITY_SCHEMA
        and feasibility.get("status") == FEASIBILITY_STATUS
        and feasibility.get("issue") == 655,
        "feasibility parent drift",
    )
    _fail(
        endpoint.get("schema") == ENDPOINT_SCHEMA
        and endpoint.get("status") == ENDPOINT_STATUS
        and endpoint.get("issue") == 655,
        "endpoint parent drift",
    )
    parent_pins = report.get("canonical_parent_pins")
    _fail(isinstance(parent_pins, Mapping), "parent pins missing")
    _exact_keys(parent_pins, {"feasibility", "endpoint"}, "parent pins")
    for key, path, parent in (
        ("feasibility", FEASIBILITY_PATH, feasibility),
        ("endpoint", ENDPOINT_PATH, endpoint),
    ):
        pin = parent_pins.get(key)
        _fail(isinstance(pin, Mapping), f"{key} pin missing")
        _exact_keys(
            pin,
            {"schema", "status", "receipt_sha256", "raw_pin"},
            f"{key} parent pin",
        )
        _fail(pin.get("schema") == parent["schema"], f"{key} schema pin")
        _fail(pin.get("status") == parent["status"], f"{key} status pin")
        _fail(
            pin.get("receipt_sha256") == parent["receipt_sha256"],
            f"{key} receipt pin",
        )
        raw_pin = pin.get("raw_pin")
        _fail(isinstance(raw_pin, Mapping), f"{key} raw pin missing")
        _exact_keys(raw_pin, {"path", "bytes", "sha256"}, f"{key} raw pin")
        _fail(raw_pin.get("path") == path.relative_to(ROOT).as_posix(), f"{key} path")
        _fail(raw_pin.get("bytes") == len(path.read_bytes()), f"{key} bytes")
        _fail(raw_pin.get("sha256") == _raw_sha(path), f"{key} raw hash")
    return feasibility, endpoint


def _geometry(
    feasibility: Mapping[str, Any],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    control = feasibility["algebraic_transport_positive_control"]
    pairs = tuple((int(row[0]), int(row[1])) for row in control["antipodal_axis_pairs"])
    _fail(len(pairs) == AXES, "axis pair count")
    _fail(
        sorted(port for pair in pairs for port in pair) == list(range(PORTS)),
        "axis pair partition",
    )
    antipodes = [0] * PORTS
    directions: list[tuple[int, ...] | None] = [None] * PORTS
    for axis, (positive, negative) in enumerate(pairs):
        _fail(positive < negative, "axis presentation convention")
        antipodes[positive] = negative
        antipodes[negative] = positive
        directions[positive] = tuple(1 if i == axis else 0 for i in range(AXES))
        directions[negative] = tuple(-1 if i == axis else 0 for i in range(AXES))
    _fail(all(row is not None for row in directions), "direction family incomplete")
    actions = tuple(
        tuple(int(value) for value in row) for row in icosahedral_a5_port_permutations()
    )
    _fail(len(actions) == 60 and len(set(actions)) == 60, "A5 port action order")
    _fail(
        _sha([list(row) for row in actions]) == control["proper_A5_port_action_sha256"],
        "A5 port action source pin",
    )
    _fail(
        all(
            action[antipodes[port]] == antipodes[action[port]]
            for action in actions
            for port in range(PORTS)
        ),
        "A5 antipode preservation",
    )
    return (
        pairs,
        tuple(antipodes),
        tuple(row for row in directions if row is not None),
        actions,
    )


def _finite_model(
    directions: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    sites = tuple(itertools.product(range(MODULUS), repeat=AXES))
    lookup = {site: index for index, site in enumerate(sites)}
    steps = []
    for direction in directions:
        steps.append(
            tuple(
                lookup[
                    tuple(
                        (int(site[axis]) + int(direction[axis])) % MODULUS
                        for axis in range(AXES)
                    )
                ]
                for site in sites
            )
        )
    _fail(
        all(sorted(row) == list(range(len(sites))) for row in steps),
        "meaning map bijectivity",
    )
    return sites, tuple(steps)


def _axis_action(
    action: Sequence[int], pairs: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lookup: dict[int, tuple[int, int]] = {}
    for axis, (positive, negative) in enumerate(pairs):
        lookup[int(positive)] = (axis, 1)
        lookup[int(negative)] = (axis, -1)
    targets = []
    signs = []
    for positive, _negative in pairs:
        target, sign = lookup[int(action[int(positive)])]
        targets.append(target)
        signs.append(sign)
    _fail(sorted(targets) == list(range(AXES)), "signed axis action")
    return tuple(targets), tuple(signs)


def _q_action(
    sites: Sequence[Sequence[int]],
    action: Sequence[int],
    pairs: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    targets, signs = _axis_action(action, pairs)
    lookup = {tuple(site): index for index, site in enumerate(sites)}
    row = []
    for site in sites:
        target = [0] * AXES
        for source_axis, coordinate in enumerate(site):
            target[targets[source_axis]] = (
                signs[source_axis] * int(coordinate)
            ) % MODULUS
        row.append(lookup[tuple(target)])
    _fail(sorted(row) == list(range(len(sites))), "Q action bijectivity")
    return tuple(row)


def verify_receipt(receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = _load(receipt_path)
    stored_digest = report.pop("receipt_sha256", None)
    _fail(stored_digest == _sha(report), "receipt digest")
    report["receipt_sha256"] = stored_digest
    _exact_keys(
        report,
        {
            "schema",
            "issue",
            "status",
            "comparison_data_read",
            "canonical_parent_pins",
            "canonical_source_audit",
            "constructive_source_law",
            "attainment",
            "issue_655_disposition",
            "provenance_boundary",
            "implementation_pins",
            "claim_boundary",
            "receipt_sha256",
        },
        "top-level receipt",
    )
    _fail(report.get("schema") == SCHEMA, "schema")
    _fail(report.get("status") == STATUS, "status")
    _fail(report.get("issue") == 655, "issue")
    _fail(report.get("comparison_data_read") is False, "comparison boundary")

    feasibility, endpoint = _parent_contract(report)
    canonical = report["canonical_source_audit"]
    _exact_keys(
        canonical,
        {
            "canonical_source_emits_requested_packet",
            "canonical_source_emits_accepted_observer_quotient",
            "canonical_source_emits_A2_naturality_rows",
            "canonical_source_emits_endpoint_diamonds",
            "reason",
            "current_source_retrofit_performed",
        },
        "canonical source audit",
    )
    _fail(
        all(
            canonical[key] is False
            for key in (
                "canonical_source_emits_requested_packet",
                "canonical_source_emits_accepted_observer_quotient",
                "canonical_source_emits_A2_naturality_rows",
                "canonical_source_emits_endpoint_diamonds",
                "current_source_retrofit_performed",
            )
        ),
        "canonical source promotion",
    )
    _fail(
        canonical["reason"]
        == (
            "the canonical packet emits noninvertible disjoint-port repair "
            "projectors and fails the directed inverse-compatible source gate"
        ),
        "canonical source audit reason",
    )
    endpoint_attainment = endpoint["attainment"]
    _fail(
        endpoint_attainment["current_source_emitted_endpoint_diamond_ledger"] is False,
        "endpoint parent no longer reports the source gap",
    )

    pairs, antipodes, directions, actions = _geometry(feasibility)
    sites, steps = _finite_model(directions)
    law = report["constructive_source_law"]
    _exact_keys(
        law,
        {
            "source_capture",
            "a1_complete_event_alphabet",
            "a2_interpretation",
            "a3_counting_reference",
            "raw_step_rows",
            "meaning_step_rows",
            "a2_descent_rows",
            "antipodal_inverse_rows",
            "positive_axis_endpoint_diamond_rows",
            "same_Q_A5_action",
        },
        "constructive law",
    )
    alphabet = law["a1_complete_event_alphabet"]
    _exact_keys(
        alphabet,
        {
            "event_count",
            "event_rows",
            "every_event_accepted_before_capture_hash",
            "complete_signed_port_orbit",
        },
        "event alphabet",
    )
    events = alphabet["event_rows"]
    _fail(alphabet["event_count"] == PORTS and len(events) == PORTS, "event count")
    _fail(alphabet["every_event_accepted_before_capture_hash"] is True, "event phase")
    _fail(alphabet["complete_signed_port_orbit"] is True, "event completeness")
    event_ids = []
    for port, row in enumerate(events):
        _exact_keys(
            row,
            {
                "schema",
                "event_kind",
                "port",
                "antipodal_port",
                "raw_direction_in_Z_power_6",
                "raw_step",
                "meaning_step",
                "meaning_step_permutation_sha256",
                "emission_phase",
                "comparison_or_target_input_used",
                "event_id",
            },
            f"event {port}",
        )
        payload = dict(row)
        event_id = payload.pop("event_id", None)
        _fail(event_id == _sha(payload), f"event {port} digest")
        _fail(payload["port"] == port, f"event {port} ordering")
        _fail(payload["antipodal_port"] == antipodes[port], f"event {port} antipode")
        _fail(
            payload["raw_direction_in_Z_power_6"] == list(directions[port]),
            f"event {port} direction",
        )
        _fail(
            payload["meaning_step_permutation_sha256"] == _sha(list(steps[port])),
            f"event {port} meaning pin",
        )
        _fail(
            {
                key: payload[key]
                for key in ("schema", "event_kind", "raw_step", "meaning_step")
            }
            == {
                "schema": "oph.vertex12-signed-axis-source-event.v1",
                "event_kind": "accepted_signed_axis_unit_record",
                "raw_step": "T_p(n)=n+v_p on Z^6",
                "meaning_step": "tau_p([n]_3)=[n+v_p]_3",
            },
            f"event {port} contract",
        )
        _fail(
            payload["emission_phase"]
            == "inside_constructive_source_law_before_capture_hash",
            f"event {port} capture phase",
        )
        _fail(
            payload["comparison_or_target_input_used"] is False,
            f"event {port} target boundary",
        )
        event_ids.append(event_id)
    _fail(len(set(event_ids)) == PORTS, "event id uniqueness")

    capture = law["source_capture"]
    _exact_keys(
        capture,
        {
            "payload",
            "source_capture_root_sha256",
            "event_count",
            "event_ids_emitted_before_capture_hash",
            "capture_hash_binds_every_event_payload",
        },
        "source capture",
    )
    _fail(capture["event_ids_emitted_before_capture_hash"] is True, "capture order")
    _fail(capture["capture_hash_binds_every_event_payload"] is True, "capture binding")
    _fail(capture["event_count"] == PORTS, "capture event count")
    capture_payload = capture["payload"]
    _exact_keys(
        capture_payload,
        {
            "schema",
            "source_law_id",
            "raw_state",
            "observer_meaning",
            "quotient_formula",
            "antipodal_axis_pairs",
            "event_rows",
            "a1_contract",
            "a2_contract",
            "a3_contract",
            "parent_receipts",
        },
        "capture payload",
    )
    _fail(capture_payload["event_rows"] == events, "capture event payload")
    _fail(
        {
            key: capture_payload[key]
            for key in (
                "schema",
                "source_law_id",
                "raw_state",
                "observer_meaning",
                "quotient_formula",
                "antipodal_axis_pairs",
                "a1_contract",
                "a2_contract",
                "a3_contract",
            )
        }
        == {
            "schema": "oph.vertex12-constructive-source-capture.v1",
            "source_law_id": "signed-six-axis-record-translation-control",
            "raw_state": "Z^6",
            "observer_meaning": "(Z/3Z)^6",
            "quotient_formula": "q(n_0,...,n_5)=(n_0 mod 3,...,n_5 mod 3)",
            "antipodal_axis_pairs": [list(pair) for pair in pairs],
            "a1_contract": (
                "the complete accepted primitive alphabet is one signed "
                "unit-record event for each of the twelve canonical ports"
            ),
            "a2_contract": (
                "observer meaning is reduction modulo three and every accepted "
                "step must descend through that interpretation"
            ),
            "a3_contract": (
                "the declared exact counting reference is uniform on the complete "
                "twelve-event alphabet; no event is selected by downstream data"
            ),
        },
        "capture contract",
    )
    _fail(
        capture["source_capture_root_sha256"] == _sha(capture_payload),
        "capture root",
    )
    _fail(
        capture_payload["parent_receipts"]
        == {
            "feasibility_receipt_sha256": feasibility["receipt_sha256"],
            "endpoint_receipt_sha256": endpoint["receipt_sha256"],
        },
        "capture ancestry",
    )

    q = law["a2_interpretation"]
    _exact_keys(
        q,
        {
            "raw_state_domain",
            "meaning_state_domain",
            "meaning_state_count",
            "meaning_states",
            "q_formula",
            "surjective",
            "surjectivity_witnesses",
            "q_is_spatial_readout",
            "q_is_physical_readout",
        },
        "A2 interpretation",
    )
    _fail(q["raw_state_domain"] == "Z^6", "raw domain")
    _fail(q["meaning_state_domain"] == "(Z/3Z)^6", "meaning domain")
    _fail(q["meaning_state_count"] == len(sites), "meaning count")
    _fail(q["meaning_states"] == [list(site) for site in sites], "meaning states")
    _fail(q["q_formula"] == "componentwise_reduction_modulo_3", "q formula")
    _fail(q["surjective"] is True, "q surjectivity")
    expected_witnesses = [
        {"meaning_state_index": index, "raw_integer_representative": list(site)}
        for index, site in enumerate(sites)
    ]
    _fail(q["surjectivity_witnesses"] == expected_witnesses, "q witnesses")
    _fail(q["q_is_spatial_readout"] is False, "spatial q promotion")
    _fail(q["q_is_physical_readout"] is False, "physical q promotion")

    raw_rows = law["raw_step_rows"]
    meaning_rows = law["meaning_step_rows"]
    descent_rows = law["a2_descent_rows"]
    _fail(
        len(raw_rows) == len(meaning_rows) == len(descent_rows) == PORTS,
        "step row count",
    )
    for port in range(PORTS):
        raw = raw_rows[port]
        meaning = meaning_rows[port]
        descent = descent_rows[port]
        _exact_keys(
            raw,
            {
                "port",
                "event_id",
                "direction",
                "formula",
                "bijective_on_Z_power_6",
                "inverse_port",
            },
            f"raw step {port}",
        )
        _exact_keys(
            meaning,
            {
                "port",
                "event_id",
                "permutation",
                "permutation_sha256",
                "bijective",
            },
            f"meaning step {port}",
        )
        _exact_keys(
            descent,
            {
                "port",
                "event_id",
                "identity",
                "proof",
                "complete_residue_representative_count",
                "complete_residue_endpoint_indices",
                "exact_A2_naturality_square",
            },
            f"descent square {port}",
        )
        _fail(
            raw["port"] == meaning["port"] == descent["port"] == port,
            f"step {port} index",
        )
        _fail(
            raw["event_id"]
            == meaning["event_id"]
            == descent["event_id"]
            == event_ids[port],
            f"step {port} event",
        )
        _fail(raw["direction"] == list(directions[port]), f"raw {port} direction")
        _fail(raw["inverse_port"] == antipodes[port], f"raw {port} inverse")
        _fail(raw["bijective_on_Z_power_6"] is True, f"raw {port} bijectivity")
        _fail(raw["formula"] == "T_p(n)=n+direction", f"raw {port} formula")
        _fail(meaning["permutation"] == list(steps[port]), f"meaning {port} table")
        _fail(
            meaning["permutation_sha256"] == _sha(list(steps[port])),
            f"meaning {port} digest",
        )
        _fail(meaning["bijective"] is True, f"meaning {port} bijectivity")
        _fail(
            descent["complete_residue_representative_count"] == len(sites),
            f"descent {port} count",
        )
        _fail(
            descent["complete_residue_endpoint_indices"] == list(steps[port]),
            f"descent {port} endpoints",
        )
        _fail(
            descent["identity"] == "q(T_p(n))=tau_p(q(n)) for every n in Z^6"
            and descent["proof"] == "componentwise integer congruence modulo three",
            f"descent {port} contract",
        )
        _fail(descent["exact_A2_naturality_square"] is True, f"descent {port} gate")

    identity = tuple(range(len(sites)))
    inverse_rows = law["antipodal_inverse_rows"]
    _fail(len(inverse_rows) == AXES, "inverse row count")
    for axis, ((positive, negative), row) in enumerate(
        zip(pairs, inverse_rows, strict=True)
    ):
        _exact_keys(
            row,
            {
                "positive_port",
                "negative_port",
                "tau_negative_equals_tau_positive_inverse",
                "forward_then_reverse_endpoints",
                "reverse_then_forward_endpoints",
            },
            f"inverse row {axis}",
        )
        forward_reverse = _compose(steps[negative], steps[positive])
        reverse_forward = _compose(steps[positive], steps[negative])
        _fail(forward_reverse == reverse_forward == identity, f"inverse algebra {axis}")
        _fail(
            row["positive_port"] == positive and row["negative_port"] == negative,
            f"inverse ports {axis}",
        )
        _fail(
            row["tau_negative_equals_tau_positive_inverse"] is True,
            f"inverse verdict {axis}",
        )
        _fail(
            row["forward_then_reverse_endpoints"] == list(identity),
            f"inverse endpoints {axis}",
        )
        _fail(
            row["reverse_then_forward_endpoints"] == list(identity),
            f"reverse endpoints {axis}",
        )

    diamonds = law["positive_axis_endpoint_diamond_rows"]
    expected_pairs = list(itertools.combinations([pair[0] for pair in pairs], 2))
    _fail(len(diamonds) == len(expected_pairs) == 15, "diamond row count")
    for index, ((left, right), row) in enumerate(
        zip(expected_pairs, diamonds, strict=True)
    ):
        _exact_keys(
            row,
            {
                "left_port",
                "right_port",
                "all_state_count",
                "left_after_right_endpoints",
                "right_after_left_endpoints",
                "all_state_endpoints_equal",
            },
            f"diamond row {index}",
        )
        ab = _compose(steps[left], steps[right])
        ba = _compose(steps[right], steps[left])
        _fail(ab == ba, f"diamond algebra {index}")
        _fail(
            row["left_port"] == left and row["right_port"] == right,
            f"diamond ports {index}",
        )
        _fail(row["all_state_count"] == len(sites), f"diamond count {index}")
        _fail(row["left_after_right_endpoints"] == list(ab), f"diamond AB {index}")
        _fail(row["right_after_left_endpoints"] == list(ba), f"diamond BA {index}")
        _fail(row["all_state_endpoints_equal"] is True, f"diamond verdict {index}")

    a5 = law["same_Q_A5_action"]
    _exact_keys(
        a5,
        {
            "group_order",
            "group_rows",
            "action_is_faithful_on_Q",
            "covariance_formula",
            "covariance_rows",
            "exact_covariance_check_count",
        },
        "same-Q A5 action",
    )
    group_rows = a5["group_rows"]
    covariance_rows = a5["covariance_rows"]
    _fail(a5["group_order"] == len(group_rows) == 60, "A5 group count")
    _fail(a5["action_is_faithful_on_Q"] is True, "A5 faithfulness")
    _fail(
        a5["covariance_formula"] == "U_g tau_p U_g_inverse=tau_g(p)",
        "A5 covariance formula",
    )
    q_actions = []
    for group_index, (port_action, row) in enumerate(
        zip(actions, group_rows, strict=True)
    ):
        _exact_keys(
            row,
            {
                "group_element_index",
                "port_permutation",
                "axis_targets",
                "axis_signs",
                "meaning_permutation",
                "meaning_permutation_sha256",
            },
            f"A5 group row {group_index}",
        )
        targets, signs = _axis_action(port_action, pairs)
        q_action = _q_action(sites, port_action, pairs)
        q_actions.append(q_action)
        _fail(row["group_element_index"] == group_index, f"A5 index {group_index}")
        _fail(
            row["port_permutation"] == list(port_action), f"A5 port row {group_index}"
        )
        _fail(row["axis_targets"] == list(targets), f"A5 targets {group_index}")
        _fail(row["axis_signs"] == list(signs), f"A5 signs {group_index}")
        _fail(row["meaning_permutation"] == list(q_action), f"A5 Q row {group_index}")
        _fail(
            row["meaning_permutation_sha256"] == _sha(list(q_action)),
            f"A5 Q hash {group_index}",
        )
    _fail(len(set(q_actions)) == 60, "A5 Q action not faithful")
    q_action_set = set(q_actions)
    _fail(
        all(
            _compose(left, right) in q_action_set
            for left in q_actions
            for right in q_actions
        ),
        "A5 Q action not closed",
    )
    _fail(
        a5["exact_covariance_check_count"] == len(covariance_rows) == 720,
        "covariance count",
    )
    expected_index = 0
    for group_index, (port_action, q_action) in enumerate(
        zip(actions, q_actions, strict=True)
    ):
        q_inverse = _inverse(q_action)
        for port in range(PORTS):
            row = covariance_rows[expected_index]
            _exact_keys(
                row,
                {
                    "group_element_index",
                    "port",
                    "target_port",
                    "conjugated_permutation_sha256",
                    "target_permutation_sha256",
                    "exact",
                },
                f"covariance row {group_index}:{port}",
            )
            conjugated = _compose(q_action, _compose(steps[port], q_inverse))
            target = steps[port_action[port]]
            _fail(conjugated == target, f"covariance algebra {group_index}:{port}")
            _fail(
                row
                == {
                    "group_element_index": group_index,
                    "port": port,
                    "target_port": port_action[port],
                    "conjugated_permutation_sha256": _sha(list(conjugated)),
                    "target_permutation_sha256": _sha(list(target)),
                    "exact": True,
                },
                f"covariance row {group_index}:{port}",
            )
            expected_index += 1

    a3 = law["a3_counting_reference"]
    _exact_keys(
        a3,
        {
            "event_count",
            "weight_per_event",
            "reference_fixed_before_any_downstream_comparison",
            "unique_information_projection_only_under_declared_counting_reference",
            "canonical_A3_alone_selects_this_source_law",
        },
        "A3 counting reference",
    )
    _fail(
        a3["event_count"] == PORTS and a3["weight_per_event"] == "1/12", "A3 reference"
    )
    _fail(
        a3["reference_fixed_before_any_downstream_comparison"] is True,
        "A3 target boundary",
    )
    _fail(
        a3["unique_information_projection_only_under_declared_counting_reference"]
        is True,
        "A3 conditionality",
    )
    _fail(a3["canonical_A3_alone_selects_this_source_law"] is False, "A3 overclaim")

    attainment = report["attainment"]
    _exact_keys(
        attainment,
        {
            "constructive_source_law_capture_root",
            "accepted_surjective_quotient",
            "twelve_raw_steps",
            "twelve_meaning_steps",
            "twelve_A2_descent_squares",
            "six_quotient_inverse_identities",
            "fifteen_complete_all_state_endpoint_diamonds",
            "same_Q_A5_covariance",
            "universal_z_power_6_factorization",
            "canonical_source_selection",
            "spatial_translation",
            "physical_readout",
            "physical_prediction",
        },
        "attainment",
    )
    for key in (
        "constructive_source_law_capture_root",
        "accepted_surjective_quotient",
        "twelve_raw_steps",
        "twelve_meaning_steps",
        "twelve_A2_descent_squares",
        "six_quotient_inverse_identities",
        "fifteen_complete_all_state_endpoint_diamonds",
        "same_Q_A5_covariance",
        "universal_z_power_6_factorization",
    ):
        _fail(attainment[key] is True, f"missing attainment {key}")
    for key in (
        "canonical_source_selection",
        "spatial_translation",
        "physical_readout",
        "physical_prediction",
    ):
        _fail(attainment[key] is False, f"unsupported attainment {key}")
    disposition = report["issue_655_disposition"]
    _exact_keys(
        disposition,
        {
            "advances_algebraic_producer_feasibility",
            "advances_canonical_source_bridge",
            "advances_physical_bridge",
            "issue_closure_supported",
            "classification",
            "next_gate",
        },
        "issue disposition",
    )
    _fail(
        disposition["classification"] == "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ONLY",
        "disposition",
    )
    _fail(
        disposition["advances_algebraic_producer_feasibility"] is True,
        "feasibility progress",
    )
    _fail(
        disposition["advances_canonical_source_bridge"] is False,
        "canonical bridge promotion",
    )
    _fail(disposition["advances_physical_bridge"] is False, "physical bridge promotion")
    _fail(disposition["issue_closure_supported"] is False, "issue closure promotion")
    _fail(
        disposition["next_gate"]
        == (
            "derive or source-select the signed six-axis record law inside the "
            "canonical simulator, then attach the digest-identical action to a "
            "physical sector, readout, frame, and boost law"
        ),
        "next gate",
    )
    boundary = report["provenance_boundary"]
    _exact_keys(
        boundary,
        {
            "new_constructive_source_law",
            "post_capture_assignment_to_canonical_source",
            "canonical_A1_A2_A3_derivation_claimed",
            "full_canonical_A1_typed_object_instantiated",
            "full_A2_observer_federation_functor_instantiated",
            "canonical_A3_maximum_entropy_selection_instantiated",
            "additional_branch_contract_required",
            "modulus_three_scope",
            "spatial_identification_claimed",
            "physical_attachment_claimed",
            "comparison_or_target_data_used",
        },
        "provenance boundary",
    )
    _fail(boundary["new_constructive_source_law"] is True, "new law boundary")
    for key in (
        "post_capture_assignment_to_canonical_source",
        "canonical_A1_A2_A3_derivation_claimed",
        "full_canonical_A1_typed_object_instantiated",
        "full_A2_observer_federation_functor_instantiated",
        "canonical_A3_maximum_entropy_selection_instantiated",
        "spatial_identification_claimed",
        "physical_attachment_claimed",
        "comparison_or_target_data_used",
    ):
        _fail(boundary[key] is False, f"provenance promotion {key}")
    _fail(
        boundary["additional_branch_contract_required"] is True,
        "branch premise boundary",
    )
    _fail(
        boundary["modulus_three_scope"]
        == (
            "three is the smallest coordinate modulus that distinguishes "
            "+e_i from -e_i; this does not assert a globally smallest finite "
            "A5-equivariant quotient or a source-selected physical period"
        ),
        "modulus scope",
    )

    expected_paths = {
        PRODUCER_PATH.relative_to(ROOT).as_posix(),
        INDEPENDENT_VERIFIER_PATH.relative_to(ROOT).as_posix(),
        TEST_PATH.relative_to(ROOT).as_posix(),
    }
    pins = report["implementation_pins"]
    _fail(
        isinstance(pins, list)
        and {pin.get("path") for pin in pins if isinstance(pin, Mapping)}
        == expected_paths
        and len(pins) == len(expected_paths),
        "implementation pin set",
    )
    for pin in pins:
        _exact_keys(pin, {"path", "bytes", "sha256"}, "implementation pin")
        path = ROOT / pin["path"]
        _fail(path.is_file(), f"implementation path {path}")
        _fail(pin["bytes"] == len(path.read_bytes()), f"implementation size {path}")
        _fail(pin["sha256"] == _raw_sha(path), f"implementation hash {path}")

    _fail(
        report["claim_boundary"]
        == (
            "The requested X,Q,q,T_p,tau_p algebraic packet is attained on a "
            "separate target-free constructive source law. The canonical source "
            "still does not emit or select that law, and this control does not "
            "instantiate the full canonical A1 object, A2 federation functor, or "
            "A3 selector. The finite observer meaning is a quotient of integer "
            "record coordinates, not a spatial or physical readout. This result "
            "is a producer-feasibility control and does not close issue 655 or "
            "promote FZ-11."
        ),
        "claim boundary",
    )

    return {
        "schema": "oph.vertex12-constructive-source-law-control-independent-verification.v1",
        "status": "PASS",
        "receipt": True,
        "producer_imported": False,
        "source_engine_independently_reimplemented": False,
        "quotient_algebra_independently_reimplemented": True,
        "checked_meaning_states": len(sites),
        "checked_source_events": len(events),
        "checked_descent_squares": len(descent_rows),
        "checked_inverse_rows": len(inverse_rows),
        "checked_endpoint_diamond_rows": len(diamonds),
        "checked_A5_covariance_rows": len(covariance_rows),
        "claim_boundary": (
            "The verifier confirms a separate constructive source-law control. "
            "It does not find that law in the canonical capture or attach it to "
            "space, a physical sector, or a comparison."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.receipt)
    except (KeyError, TypeError, ValueError, IndependentVerificationError) as exc:
        result = {
            "schema": "oph.vertex12-constructive-source-law-control-independent-verification.v1",
            "status": "FAIL",
            "receipt": False,
            "producer_imported": False,
            "source_engine_independently_reimplemented": False,
            "quotient_algebra_independently_reimplemented": True,
            "reasons": [str(exc)],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
