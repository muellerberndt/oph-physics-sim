"""Exact A2 endpoint/commutator boundary for the twelve-port source lane.

A2 makes accepted data maps natural after they have been typed as part of the
observer-accessible interface.  It does not create a transport map and it does
not assert confluence.  This module separates three statements that can look
similar in informal prose:

* equality of two visible endpoints at one starting state;
* equality of the two ordered visible endpoints at every starting state; and
* equality of the underlying ordered histories.

Only the second statement, together with descent of each primitive step to the
observer-meaning quotient, gives a commuting pair of quotient maps.  If this
holds for the fifteen pairs of six antipodal-axis representatives and the six
antipodal maps are inverses, the quotient action factors uniquely through the
universal abelian port group ``Z^6``.  The physical image may still be a proper
quotient of ``Z^6``.  A finite state ledger cannot certify a faithful ``Z^6``
action without a symbolic or cofinal no-extra-relations certificate.

The finite controls below are exact.  They distinguish local terminal
coincidence, quotient terminal confluence, path equality, and well-defined
quotient descent.  They do not promote the existing algebraic ``(Z/3Z)^6``
control to a source-emitted or spatial transport.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.dynamics import (
    verify_vertex12_atomic_port_transfer_independent,
    verify_vertex12_directed_transport_feasibility_independent,
)


SCHEMA = "oph.vertex12-a2-endpoint-commutator-boundary.v1"
VERIFICATION_SCHEMA = "oph.vertex12-a2-endpoint-commutator-boundary-verification.v1"
STATUS = (
    "A2_ENDPOINT_TO_QUOTIENT_COMMUTATOR_THEOREM_ATTAINED__"
    "SOURCE_NATURALITY_INVERSES_DIAMONDS_AND_PHYSICAL_ACTION_OPEN"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json"
)
UPSTREAM_PATH = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
ATOMIC_PATH = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
PRODUCER_PATH = Path(__file__).resolve()
INDEPENDENT_VERIFIER_PATH = (
    REPOSITORY_ROOT
    / "oph_fpe/dynamics/verify_vertex12_a2_endpoint_commutator_independent.py"
)
TEST_PATH = REPOSITORY_ROOT / "tests/test_vertex12_a2_endpoint_commutator.py"

UPSTREAM_SCHEMA = "oph.vertex12-directed-transport-feasibility.v1"
UPSTREAM_STATUS = (
    "EXACT_SEMICONJUGATE_COVER_OBSTRUCTION_FOR_CURRENT_SOURCE_MATCHINGS__"
    "ORIENTED_SOURCE_TRANSITION_LAW_OPEN"
)
ATOMIC_SCHEMA = "oph.vertex12-atomic-port-transfer-subpacket.v1"
ATOMIC_STATUS = (
    "INTERNAL_VERTEX12_SEAM_MATCHING_PROJECTORS_AND_IN_PROCESS_SNAPSHOT_REREAD_ATTAINED__"
    "SPATIAL_PHYSICAL_BRIDGE_OPEN"
)


class EndpointCommutatorError(ValueError):
    """Raised when the endpoint/commutator packet cannot be reconstructed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EndpointCommutatorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(value, dict):
        raise EndpointCommutatorError("receipt JSON root is not an object")
    return value


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _compose(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise EndpointCommutatorError("cannot compose maps of different sizes")
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    size = len(permutation)
    if sorted(permutation) != list(range(size)):
        raise EndpointCommutatorError("map is not a permutation")
    inverse = [0] * size
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)


def _exact_rank(rows: Sequence[Sequence[int]]) -> int:
    work = [list(map(Fraction, row)) for row in rows]
    if not work:
        return 0
    width = len(work[0])
    if any(len(row) != width for row in work):
        raise EndpointCommutatorError("relation matrix is ragged")
    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(work)) if work[index][column]),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for index in range(len(work)):
            if index == rank or not work[index][column]:
                continue
            factor = work[index][column]
            work[index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[index], work[rank])
            ]
        rank += 1
    return rank


def _validated_upstream() -> dict[str, Any]:
    report = _load_json(UPSTREAM_PATH)
    verification = (
        verify_vertex12_directed_transport_feasibility_independent.verify_report(report)
    )
    if verification.get("receipt") is not True:
        raise EndpointCommutatorError("upstream feasibility packet failed verification")
    if (
        report.get("schema") != UPSTREAM_SCHEMA
        or report.get("status") != UPSTREAM_STATUS
        or type(report.get("issue")) is not int
        or report.get("issue") != 655
    ):
        raise EndpointCommutatorError("upstream feasibility contract drifted")
    return report


def _axis_data(upstream: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    control = upstream.get("algebraic_transport_positive_control")
    if not isinstance(control, Mapping):
        raise EndpointCommutatorError("upstream algebraic control is missing")
    pairs = control.get("antipodal_axis_pairs")
    if (
        not isinstance(pairs, list)
        or len(pairs) != 6
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(port) is not int for port in pair)
            for pair in pairs
        )
    ):
        raise EndpointCommutatorError("upstream antipodal pairs are malformed")
    normalized = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
    flattened = [port for pair in normalized for port in pair]
    if sorted(flattened) != list(range(12)) or any(
        left >= right for left, right in normalized
    ):
        raise EndpointCommutatorError(
            "upstream antipodal pairs do not partition the ports"
        )
    return normalized


def _validated_atomic(upstream: Mapping[str, Any]) -> dict[str, Any]:
    atomic = _load_json(ATOMIC_PATH)
    verification = verify_vertex12_atomic_port_transfer_independent.verify_report(
        atomic
    )
    if verification.get("receipt") is not True:
        raise EndpointCommutatorError("atomic source packet failed verification")
    if (
        atomic.get("schema") != ATOMIC_SCHEMA
        or atomic.get("status") != ATOMIC_STATUS
        or type(atomic.get("issue")) is not int
        or atomic.get("issue") != 655
    ):
        raise EndpointCommutatorError("atomic source contract drifted")
    binding = upstream.get("upstream_source_packet")
    if not isinstance(binding, Mapping):
        raise EndpointCommutatorError("feasibility packet lacks its atomic binding")
    if _canonical_bytes(binding.get("pin")) != _canonical_bytes(
        _raw_pin(ATOMIC_PATH)
    ) or binding.get("receipt_sha256") != atomic.get("receipt_sha256"):
        raise EndpointCommutatorError(
            "atomic packet does not match feasibility ancestry"
        )
    return atomic


def _current_source_repair_control(atomic: Mapping[str, Any]) -> dict[str, Any]:
    """Separate exact repair confluence from a directed common-site action."""

    operator = atomic.get("atomic_transfer_operator")
    history = atomic.get("source_history_replay")
    if not isinstance(operator, Mapping) or not isinstance(history, Mapping):
        raise EndpointCommutatorError("atomic repair blocks are missing")
    rows = operator.get("port_rows")
    events = history.get("event_rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise EndpointCommutatorError("atomic port row census drifted")
    if not isinstance(events, list) or len(events) != 48:
        raise EndpointCommutatorError("atomic event row census drifted")

    permutations: dict[int, tuple[int, ...]] = {}
    event_ids: dict[int, list[str]] = {port: [] for port in range(12)}
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("port")) is not int:
            raise EndpointCommutatorError("atomic port row is malformed")
        port = row["port"]
        permutation = row.get("carrier_partner_permutation")
        seam_ids = row.get("seam_ids")
        if (
            port in permutations
            or not isinstance(permutation, list)
            or len(permutation) != 8
            or any(type(value) is not int for value in permutation)
            or not isinstance(seam_ids, list)
            or len(seam_ids) != 4
        ):
            raise EndpointCommutatorError("atomic matching row is incomplete")
        normalized = tuple(permutation)
        if (
            sorted(normalized) != list(range(8))
            or _inverse(normalized) != normalized
            or any(normalized[index] == index for index in range(8))
            or row.get("repair_projector_idempotent") is not True
        ):
            raise EndpointCommutatorError("atomic matching/projector contract drifted")
        permutations[port] = normalized
    if set(permutations) != set(range(12)):
        raise EndpointCommutatorError("atomic port rows do not cover all ports")
    for event in events:
        if (
            not isinstance(event, Mapping)
            or type(event.get("port")) is not int
            or event["port"] not in event_ids
            or type(event.get("event_id")) is not str
            or not event["event_id"].startswith("sha256:")
        ):
            raise EndpointCommutatorError("atomic event provenance row is malformed")
        event_ids[event["port"]].append(event["event_id"])
    if any(len(ids) != 4 or len(set(ids)) != 4 for ids in event_ids.values()):
        raise EndpointCommutatorError("atomic event coverage is not four per port")
    event_ids = {port: sorted(ids) for port, ids in event_ids.items()}

    shared_carrier_commuting_pairs = []
    shared_carrier_noncommuting_pairs = []
    for left, right in itertools.combinations(range(12), 2):
        if _compose(permutations[left], permutations[right]) == _compose(
            permutations[right], permutations[left]
        ):
            shared_carrier_commuting_pairs.append([left, right])
        else:
            shared_carrier_noncommuting_pairs.append([left, right])
    if (
        len(shared_carrier_commuting_pairs) != 23
        or len(shared_carrier_noncommuting_pairs) != 43
    ):
        raise EndpointCommutatorError("shared-carrier commutator census drifted")

    witness_left, witness_right = 0, 1
    word_left_right = event_ids[witness_left] + event_ids[witness_right]
    word_right_left = event_ids[witness_right] + event_ids[witness_left]
    if word_left_right == word_right_left:
        raise EndpointCommutatorError("event-word order witness collapsed")
    return {
        "schema": "oph.vertex12-source-repair-confluence-control.v1",
        "source_data_domain": "Q^(eight_carriers_times_twelve_port_coordinates)",
        "source_data_dimension": 96,
        "repair_event_count": 48,
        "repair_events_per_port": 4,
        "every_carrier_port_coordinate_written_once": (
            history.get("terminal_write_coordinate_count") == 96
        ),
        "twelve_port_block_repair_maps": (
            "Ahat_p acts by A_p=(I+S_p)/2 on port column p and by identity on the other eleven columns"
        ),
        "port_block_map_count": 12,
        "unordered_port_block_pair_count": 66,
        "commuting_port_block_pair_count": 66,
        "reason_all_port_block_maps_commute": (
            "distinct port blocks have disjoint carrier-port coordinate support"
        ),
        "each_eight_coordinate_matching_projector_rank": 4,
        "each_ninety_six_coordinate_port_block_map_rank": 92,
        "each_port_block_map_kernel_dimension": 4,
        "each_port_block_map_idempotent": True,
        "each_port_block_map_bijective": False,
        "full_twelve_port_repair_projector_rank": 48,
        "full_twelve_port_repair_projector_kernel_dimension": 48,
        "full_repair_terminal_operator_independent_of_port_block_order": True,
        "source_history_order_replay_exact": history.get("order_replay_exact") is True,
        "source_history_idempotence_replay_exact": (
            history.get("idempotence_replay_exact") is True
        ),
        "distinct_event_words_same_terminal_operator_witness": {
            "first_port": witness_left,
            "second_port": witness_right,
            "first_then_second_event_ids": word_left_right,
            "second_then_first_event_ids": word_right_left,
            "event_words_equal": False,
            "terminal_state_operators_equal": True,
        },
        "shared_eight_carrier_reinterpretation": {
            "object": (
                "the twelve S_p matchings acting on one shared eight-carrier set after the port coordinate is discarded"
            ),
            "unordered_pair_count": 66,
            "commuting_pair_count": len(shared_carrier_commuting_pairs),
            "noncommuting_pair_count": len(shared_carrier_noncommuting_pairs),
            "commuting_pairs": shared_carrier_commuting_pairs,
            "noncommuting_pairs": shared_carrier_noncommuting_pairs,
            "discarding_the_port_coordinate_preserves_source_operator_semantics": False,
        },
        "terminal_confluence_on_actual_source_field_attained": True,
        "underlying_event_word_equality_attained": False,
        "oriented_bijective_port_step_ledger_attained": False,
        "antipodal_inverse_translation_law_attained": False,
        "universal_z_power_6_translation_action_attained": False,
        "spatial_translation_attained": False,
        "claim_boundary": (
            "The current source supplies exact commuting idempotent repair blocks because port columns are disjoint. These are noninvertible projections on separate coordinate blocks, not six reversible directions on a common site object. Treating the twelve carrier matchings as common-site steps changes the source domain and exposes 43 noncommuting pairs."
        ),
    }


def _local_endpoint_control() -> dict[str, Any]:
    """One visible endpoint coincidence does not imply map commutation."""

    first = (1, 0, 2, 3)  # (0 1)
    second = (0, 2, 1, 3)  # (1 2)
    first_then_second = _compose(second, first)
    second_then_first = _compose(first, second)
    equal = [
        state
        for state in range(4)
        if first_then_second[state] == second_then_first[state]
    ]
    unequal = [state for state in range(4) if state not in equal]
    if equal != [3] or unequal != [0, 1, 2]:
        raise EndpointCommutatorError("local endpoint countercontrol drifted")
    return {
        "schema": "oph.local-endpoint-coincidence-control.v1",
        "state_count": 4,
        "observer_readout": "identity",
        "first_step_permutation": list(first),
        "second_step_permutation": list(second),
        "both_steps_bijective": True,
        "equal_ordered_endpoint_start_states": equal,
        "unequal_ordered_endpoint_start_states": unequal,
        "one_state_endpoint_coincidence_attained": True,
        "global_commutation_attained": False,
        "implication_refuted": (
            "one_start_state_visible_endpoint_equality_implies_global_quotient_commutation"
        ),
    }


def _nondescent_control() -> dict[str, Any]:
    """Commuting raw maps need not define maps on a declared quotient."""

    quotient = (0, 0, 1, 1)
    first = (0, 2, 1, 3)  # swaps representatives 1 and 2 across the classes
    second = tuple(range(4))
    first_then_second = _compose(second, first)
    second_then_first = _compose(first, second)
    if first_then_second != second_then_first:
        raise EndpointCommutatorError(
            "nondescent control unexpectedly fails raw commutation"
        )
    violation_pairs = []
    for left in range(4):
        for right in range(left + 1, 4):
            if (
                quotient[left] == quotient[right]
                and quotient[first[left]] != quotient[first[right]]
            ):
                violation_pairs.append([left, right])
    if violation_pairs != [[0, 1], [2, 3]]:
        raise EndpointCommutatorError("nondescent witness census drifted")
    return {
        "schema": "oph.quotient-descent-necessity-control.v1",
        "state_count": 4,
        "observer_class_ids": list(quotient),
        "observer_class_count": 2,
        "first_step_permutation": list(first),
        "second_step_permutation": list(second),
        "raw_ordered_maps_equal_on_every_state": True,
        "first_step_quotient_congruence_violation_pairs": violation_pairs,
        "first_step_descends_to_observer_quotient": False,
        "quotient_port_action_defined": False,
        "implication_refuted": (
            "raw_path_commutation_without_A2_accepted_map_naturality_defines_a_quotient_action"
        ),
    }


def _heisenberg_product(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    a, b, c = left
    x, y, z = right
    return ((a + x) % 3, (b + y) % 3, (c + z + a * y) % 3)


def _heisenberg_control() -> dict[str, Any]:
    """Visible terminal confluence is compatible with hidden path holonomy."""

    states = tuple(itertools.product(range(3), repeat=3))
    index = {state: position for position, state in enumerate(states)}
    first_generator = (1, 0, 0)
    second_generator = (0, 1, 0)
    first = tuple(
        index[_heisenberg_product(state, first_generator)] for state in states
    )
    second = tuple(
        index[_heisenberg_product(state, second_generator)] for state in states
    )
    first_then_second = _compose(second, first)
    second_then_first = _compose(first, second)
    quotient = tuple((state[0], state[1]) for state in states)
    raw_equal = [
        state_index
        for state_index in range(len(states))
        if first_then_second[state_index] == second_then_first[state_index]
    ]
    quotient_equal = [
        state_index
        for state_index in range(len(states))
        if quotient[first_then_second[state_index]]
        == quotient[second_then_first[state_index]]
    ]
    central_shifts = sorted(
        {
            (
                states[first_then_second[state_index]][2]
                - states[second_then_first[state_index]][2]
            )
            % 3
            for state_index in range(len(states))
        }
    )
    if raw_equal or quotient_equal != list(range(27)) or central_shifts != [1]:
        raise EndpointCommutatorError("Heisenberg quotient control drifted")

    class_members: dict[tuple[int, int], list[int]] = {}
    for state_index, class_id in enumerate(quotient):
        class_members.setdefault(class_id, []).append(state_index)
    for permutation in (first, second):
        for members in class_members.values():
            targets = {quotient[permutation[state_index]] for state_index in members}
            if len(targets) != 1:
                raise EndpointCommutatorError("Heisenberg step does not descend")

    identity_index = index[(0, 0, 0)]
    return {
        "schema": "oph.heisenberg-visible-confluence-control.v1",
        "raw_state_object": "H_3(F_3)",
        "raw_state_count": len(states),
        "observer_quotient_object": "F_3^2 via (a,b,c) maps to (a,b)",
        "observer_quotient_class_count": len(class_members),
        "members_per_observer_class": sorted(
            {len(row) for row in class_members.values()}
        ),
        "first_step_permutation_sha256": _sha(list(first)),
        "second_step_permutation_sha256": _sha(list(second)),
        "both_steps_bijective": (
            sorted(first) == list(range(27)) and sorted(second) == list(range(27))
        ),
        "both_steps_descend_to_observer_quotient": True,
        "visible_ordered_endpoint_equality_count": len(quotient_equal),
        "raw_ordered_endpoint_equality_count": len(raw_equal),
        "visible_ordered_endpoints_equal_for_every_state": True,
        "raw_ordered_paths_equal_for_any_state": False,
        "central_commutator_shift_modulo_three": central_shifts[0],
        "identity_start_witness": {
            "first_then_second_raw_endpoint": list(
                states[first_then_second[identity_index]]
            ),
            "second_then_first_raw_endpoint": list(
                states[second_then_first[identity_index]]
            ),
            "shared_visible_endpoint": list(
                quotient[first_then_second[identity_index]]
            ),
        },
        "implication_refuted": (
            "observer_quotient_terminal_confluence_implies_underlying_path_equality_or_trivial_holonomy"
        ),
    }


def _universal_factorization(axis_pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    positive_ports = tuple(pair[0] for pair in axis_pairs)
    axis_for_port: dict[int, tuple[int, int]] = {}
    for axis, (positive, negative) in enumerate(axis_pairs):
        axis_for_port[positive] = (axis, 1)
        axis_for_port[negative] = (axis, -1)

    relation_rows = []
    normal_form_rows = []
    for positive, negative in axis_pairs:
        relation = [0] * 12
        relation[positive] = 1
        relation[negative] = 1
        relation_rows.append(relation)
    relation_rank = _exact_rank(relation_rows)
    for port in range(12):
        axis, sign = axis_for_port[port]
        vector = [0] * 6
        vector[axis] = sign
        normal_form_rows.append(
            {
                "port": port,
                "antipodal_port": axis_pairs[axis][1 if sign == 1 else 0],
                "axis": axis,
                "orientation_sign": sign,
                "normal_form_vector": vector,
            }
        )

    diamond_pairs = list(itertools.combinations(range(6), 2))
    oriented_nonantipodal_pairs = [
        [left, right]
        for left in range(12)
        for right in range(left + 1, 12)
        if axis_for_port[left][0] != axis_for_port[right][0]
    ]
    if len(diamond_pairs) != 15 or len(oriented_nonantipodal_pairs) != 60:
        raise EndpointCommutatorError("commutator-pair census drifted")

    # Each of the fifteen positive-axis commutator relations is independent of
    # the other fourteen: map just that pair to the two generators of H_3(F_3)
    # and every other generator to the identity.  The omitted commutator is the
    # nontrivial central element while all retained commutators vanish.
    commutator_irredundancy = [
        {
            "omitted_axis_pair": [left, right],
            "finite_witness_group": "H_3(F_3)",
            "finite_witness_group_order": 27,
            "retained_commutator_relations_satisfied": 14,
            "all_six_antipodal_inverse_relations_satisfied": True,
            "omitted_commutator_central_shift_modulo_three": 1,
            "omitted_relation_not_implied_by_other_fourteen": True,
        }
        for left, right in diamond_pairs
    ]
    inverse_irredundancy = [
        {
            "omitted_axis": axis,
            "finite_witness_group": "Z/3Z",
            "positive_and_antipodal_generators_both_map_to": 1,
            "required_inverse_image": 2,
            "all_commutator_relations_satisfied": True,
            "retained_five_inverse_relations_satisfied": True,
            "omitted_inverse_relation_not_implied_by_commutation": True,
        }
        for axis in range(6)
    ]

    return {
        "schema": "oph.universal-abelian-port-factorization.v1",
        "oriented_port_count": 12,
        "antipodal_axis_count": 6,
        "positive_axis_representatives": list(positive_ports),
        "antipodal_relation_matrix": relation_rows,
        "antipodal_relation_matrix_rank_over_Q": relation_rank,
        "free_abelian_rank_after_antipodal_relations": 12 - relation_rank,
        "port_normal_form_rows": normal_form_rows,
        "positive_axis_commutator_diamond_count": len(diamond_pairs),
        "positive_axis_commutator_diamond_pairs": [
            list(pair) for pair in diamond_pairs
        ],
        "all_oriented_nonantipodal_pair_count": len(oriented_nonantipodal_pairs),
        "all_oriented_nonantipodal_pairs": oriented_nonantipodal_pairs,
        "fifteen_positive_diamonds_plus_inverse_law_imply_all_sixty_oriented_diamonds": True,
        "commutator_relation_irredundancy_witnesses": commutator_irredundancy,
        "every_positive_commutator_relation_is_individually_necessary": True,
        "inverse_relation_irredundancy_witnesses": inverse_irredundancy,
        "every_antipodal_inverse_relation_is_individually_necessary": True,
        "conditional_theorem": {
            "hypotheses": [
                "a nonempty source state object X and a surjective accepted-data interpretation q:X_to_Q",
                "twelve primitive bijections T_p on X typed as accepted A1 seam translations",
                "A2 naturality q_after_T_p=tau_p_after_q for twelve meaning-side maps tau_p on Q",
                "six quotient inverse identities tau_antipode_p=tau_p_inverse",
                "fifteen all-state endpoint diamonds tau_i_tau_j=tau_j_tau_i for the positive axis representatives",
            ],
            "conclusion": (
                "there is a unique action homomorphism rho: Z^6 -> Sym(Q) sending each signed basis vector to the corresponding tau_p"
            ),
            "proof_steps": [
                "A2_naturality_makes_each_tau_p_a_well_defined_quotient_map",
                "surjectivity_of_q_turns_all_state_visible_endpoint_equalities_into_equalities_of_meaning_side_composites",
                "the_six_inverse_relations_eliminate_the_six_antipodal_generators",
                "the_fifteen_diamond_relations_make_the_six_remaining_generators_pairwise_commute",
                "the_universal_property_of_the_free_abelian_group_gives_the_unique_z_power_6_action",
            ],
            "a2_alone_supplies_the_fifteen_endpoint_diamonds": False,
            "independence_of_port_labels_alone_supplies_endpoint_equality": False,
            "accepted_meaning_action_factors_through_z_power_6": True,
            "physical_quotient_action_factors_through_z_power_6_if_Q_is_separately_identified_as_physical": True,
            "Q_identified_with_physical_support_by_this_theorem": False,
            "accepted_meaning_action_is_forced_to_be_faithful": False,
        },
        "universal_object": "Z^6",
        "universal_object_role": (
            "initial abelian group carrying six oriented generators; a realized quotient action is an image and may obey additional relations"
        ),
        "finite_ledger_faithfulness_boundary": {
            "finite_Q_can_carry_a_faithful_z_power_6_action": False,
            "reason": (
                "the image of Z^6 in the finite permutation group Sym(Q) is finite, so the homomorphism has a nonzero kernel"
            ),
            "finite_z3_power_6_control_is_faithful_universal_z_power_6_action": False,
            "finite_Z3_power_6_control_obeys_extra_relations": "3e_i=0 for every axis",
            "exact_faithfulness_requires": (
                "a symbolic integer-coordinate cocycle with a separating orbit, or a compatible cofinal family whose kernels have trivial intersection"
            ),
        },
    }


def _minimum_source_ledger(axis_pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    return {
        "schema": "oph.vertex12-minimum-source-endpoint-diamond-ledger.v1",
        "goal": "source-instantiated quotient action factoring through the universal abelian port group",
        "logical_minimum_not_byte_encoding_minimum": True,
        "minimum_scope": (
            "necessary executable fields under issue 655's source-event provenance contract; equivalent symbolic encodings are allowed"
        ),
        "source_capture_fields": [
            "source_capture_root_sha256_covering_every_state_step_and_readout_row",
            "nonempty_source_state_ids_or_a_symbolic_source_state_schema",
            "accepted_observer_meaning_class_id_for_every_finite_source_state",
            "twelve_total_bijective_primitive_port_step_maps",
            "at_least_one_source_transition_event_id_and_payload_digest_binding_each_port_map",
            "icosahedral_antipode_map",
        ],
        "a1_a2_typing_fields": [
            "all_twelve_steps_declared_as_accepted_A1_seam_translation_maps",
            "twelve_meaning_side_step_maps",
            "twelve_exact_A2_naturality_or_quotient_congruence_receipts",
        ],
        "factorization_checks": {
            "antipodal_inverse_family_count": len(axis_pairs),
            "positive_axis_endpoint_diamond_family_count": 15,
            "endpoint_diamond_scope": "every accepted observer-meaning class",
            "negative_orientation_diamonds_need_separate_rows": False,
            "reason_negative_rows_are_redundant": (
                "inverse identities turn the fifteen positive-positive commutators into all sixty nonantipodal oriented commutators"
            ),
        },
        "faithful_universal_z_power_6_addendum": [
            "source_emitted_symbolic_integer_coordinate_readout_kappa_on_Q",
            "exact_cocycle_identity_kappa_after_tau_p_equals_kappa_plus_signed_basis_vector",
            "one_orbit_separation_receipt_or_a_cofinal_kernel_intersection_theorem",
        ],
        "issue_655_geometry_addendum": [
            "sixty_element_proper_A5_action_on_the_same_accepted_quotient",
            "exact_A5_equivariance_of_the_interpretation_map",
            "seven_hundred_twenty_exact_U_g_tau_p_U_g_inverse_equals_tau_g_p_checks_or_a_symbolic_proof",
        ],
        "physicalization_addendum": [
            "source_selected_identification_of_the_quotient_sites_with_spatial_or_other_physical_support",
            "sector_readout_and_measurement_attachment",
            "frame_and_boost_transport_if_the_claim_uses_spacetime_translation",
        ],
        "current_source_packet_supplies_complete_port_step_maps": False,
        "current_source_packet_supplies_accepted_observer_quotient": False,
        "current_source_packet_supplies_A2_naturality_rows": False,
        "current_source_packet_supplies_endpoint_diamonds": False,
        "current_source_packet_supplies_faithful_z_power_6_action": False,
        "algebraic_Z3_power_6_control_counts_as_source_ledger": False,
        "spatial_or_physical_promotion_allowed": False,
    }


def _payload() -> dict[str, Any]:
    upstream = _validated_upstream()
    atomic = _validated_atomic(upstream)
    axis_pairs = _axis_data(upstream)
    endpoint_theorem = {
        "schema": "oph.a2-endpoint-descent-commutator-theorem.v1",
        "a2_scope": (
            "A2 constrains operational meaning on accepted shared data and makes declared data-access diagrams commute after interpretation; it does not create the accepted domain, primitive step maps, termination, or confluence"
        ),
        "single_start_state_statement": {
            "premise": "q(T_p(T_r(x)))=q(T_r(T_p(x))) for one x",
            "conclusion": "the two paths have one equal terminal meaning at x",
            "global_quotient_commutation_follows": False,
        },
        "all_state_descent_statement": {
            "premises": [
                "q:X_to_Q is surjective",
                "q_after_T_p=tau_p_after_q and q_after_T_r=tau_r_after_q",
                "q_after_T_p_after_T_r=q_after_T_r_after_T_p on every x in X",
            ],
            "conclusion": "tau_p_after_tau_r=tau_r_after_tau_p on Q",
            "proof": (
                "compose the two naturality squares, use the all-state endpoint equality, then cancel the surjective interpretation q"
            ),
            "exact_conditional_implication": True,
        },
        "underlying_path_equality_statement": {
            "visible_terminal_equality_implies_T_p_T_r_equals_T_r_T_p_on_X": False,
            "hidden_kernel_or_holonomy_can_remain": True,
        },
        "a2_forces_pairwise_endpoint_diamonds_without_a_source_or_repair_premise": False,
        "a2_can_supply_quotient_descent_after_A1_accepts_and_types_each_step": True,
        "same_geometric_or_base_endpoint_label_forces_equal_internal_holonomy": False,
    }
    payload = {
        "schema": SCHEMA,
        "issue": 655,
        "status": STATUS,
        "upstream_feasibility_packet": {
            "pin": _raw_pin(UPSTREAM_PATH),
            "schema": upstream["schema"],
            "status": upstream["status"],
            "receipt_sha256": upstream["receipt_sha256"],
            "independently_verified": True,
        },
        "atomic_source_packet": {
            "pin": _raw_pin(ATOMIC_PATH),
            "schema": atomic["schema"],
            "status": atomic["status"],
            "receipt_sha256": atomic["receipt_sha256"],
            "independently_verified": True,
        },
        "a2_endpoint_commutator_theorem": endpoint_theorem,
        "exact_current_source_repair_control": _current_source_repair_control(atomic),
        "exact_finite_controls": {
            "local_endpoint_coincidence_without_global_commutation": _local_endpoint_control(),
            "raw_commutation_without_quotient_descent": _nondescent_control(),
            "visible_terminal_confluence_without_path_equality": _heisenberg_control(),
        },
        "universal_abelian_port_factorization": _universal_factorization(axis_pairs),
        "minimum_source_emitted_port_step_ledger": _minimum_source_ledger(axis_pairs),
        "attainment": {
            "conditional_A2_endpoint_descent_commutator_lemma": True,
            "universal_z_power_6_factorization": True,
            "fifteen_commutator_relations_proved_individually_necessary": True,
            "six_antipodal_inverse_relations_proved_individually_necessary": True,
            "current_source_emitted_endpoint_diamond_ledger": False,
            "faithful_physical_z_power_6_action": False,
            "spatial_translation": False,
            "physical_prediction": False,
        },
        "issue_655_disposition": {
            "negative_closure_supported": False,
            "source_producer_narrowed": True,
            "current_source_positive_result": (
                "exact terminal confluence of twelve noninvertible port-block repair projectors on the actual carrier-port field"
            ),
            "minimum_next_source_packet": [
                "twelve accepted all-state visible port-step tables with source-event bindings",
                "six quotient inverse identities",
                "fifteen complete positive-axis AB/BA endpoint-diamond tables",
            ],
            "faithful_z_power_6_and_physical_support_are_later_separate_gates": True,
            "reason": (
                "the endpoint theorem converts a precise finite producer output into the abelian factorization, while the present source contains repair confluence rather than oriented reversible steps"
            ),
        },
        "comparison_data_read": False,
        "implementation_pins": {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_verifier": _raw_pin(INDEPENDENT_VERIFIER_PATH),
            "serialized_mutation_tests": _raw_pin(TEST_PATH),
        },
        "claim_boundary": (
            "A2 turns an all-state visible endpoint diamond into a commuting pair of meaning-side maps only after the primitive steps are accepted maps and their naturality squares exist. A2 does not supply the endpoint diamond or erase hidden path holonomy. Six quotient inverse identities plus fifteen complete positive-axis diamonds make the accepted meaning action factor uniquely through Z^6. A physical interpretation and faithfulness remain separate. The current source attains exact terminal confluence for twelve noninvertible repair blocks on disjoint port columns, which narrows the missing producer to accepted reversible step tables, inverse rows, and complete AB/BA diamonds. It does not support a negative closure of issue 655. The declared (Z/3Z)^6 control remains non-source and nonphysical. No spatial translation or prediction is promoted."
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
        attainment = report.get("attainment")
        ledger = report.get("minimum_source_emitted_port_step_ledger")
        if not isinstance(attainment, Mapping) or any(
            attainment.get(key) is not False
            for key in (
                "current_source_emitted_endpoint_diamond_ledger",
                "faithful_physical_z_power_6_action",
                "spatial_translation",
                "physical_prediction",
            )
        ):
            reasons.append("unsupported_attainment_promotion")
        if not isinstance(ledger, Mapping) or any(
            ledger.get(key) is not False
            for key in (
                "current_source_packet_supplies_complete_port_step_maps",
                "current_source_packet_supplies_accepted_observer_quotient",
                "current_source_packet_supplies_A2_naturality_rows",
                "current_source_packet_supplies_endpoint_diamonds",
                "current_source_packet_supplies_faithful_z_power_6_action",
                "algebraic_Z3_power_6_control_counts_as_source_ledger",
                "spatial_or_physical_promotion_allowed",
            )
        ):
            reasons.append("unsupported_source_or_physical_promotion")
        if type(report.get("issue")) is not int or report.get("issue") != 655:
            reasons.append("issue_type_or_value_mismatch")
        if type(report.get("comparison_data_read")) is not bool or report.get(
            "comparison_data_read"
        ):
            reasons.append("comparison_boundary_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        EndpointCommutatorError,
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
            "Producer replay checks the endpoint distinctions and conditional factorization theorem. It does not emit the missing source ledger or a physical quotient."
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
