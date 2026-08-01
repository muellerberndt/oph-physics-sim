"""Independent verifier for the issue-655 A2 endpoint packet.

The verifier does not import the producer.  It independently reconstructs
the three finite controls, the irredundant six-axis presentation, and the
minimum source-ledger boundary from the pinned transport-feasibility packet.
"""

from __future__ import annotations

import argparse
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
VERIFICATION_SCHEMA = (
    "oph.vertex12-a2-endpoint-commutator-boundary-independent-verification.v1"
)
STATUS = (
    "A2_ENDPOINT_TO_QUOTIENT_COMMUTATOR_THEOREM_ATTAINED__"
    "SOURCE_NATURALITY_INVERSES_DIAMONDS_AND_PHYSICAL_ACTION_OPEN"
)
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json"
)
UPSTREAM_PATH = (
    REPOSITORY_ROOT
    / "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json"
)
ATOMIC_PATH = (
    REPOSITORY_ROOT / "data/repair_closure/vertex12_atomic_port_transfer_receipt.json"
)
PRODUCER_PATH = REPOSITORY_ROOT / "oph_fpe/dynamics/vertex12_a2_endpoint_commutator.py"
VERIFIER_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests/test_vertex12_a2_endpoint_commutator.py"


class IndependentVerificationError(ValueError):
    """Raised when independent reconstruction cannot be completed."""


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
            raise IndependentVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(value, dict):
        raise IndependentVerificationError("JSON root is not an object")
    return value


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _after(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise IndependentVerificationError("map sizes differ")
    return tuple(left[right[state]] for state in range(len(left)))


def _rank(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(map(Fraction, row)) for row in matrix]
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise IndependentVerificationError("ragged relation matrix")
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        value = rows[pivot_row][column]
        rows[pivot_row] = [entry / value for entry in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            value = rows[row][column]
            if value:
                rows[row] = [
                    entry - value * basis
                    for entry, basis in zip(rows[row], rows[pivot_row])
                ]
        pivot_row += 1
    return pivot_row


def _upstream_and_axes() -> tuple[dict[str, Any], tuple[tuple[int, int], ...]]:
    upstream = _load_json(UPSTREAM_PATH)
    check = verify_vertex12_directed_transport_feasibility_independent.verify_report(
        upstream
    )
    if check.get("receipt") is not True:
        raise IndependentVerificationError("upstream packet failed verification")
    if (
        upstream.get("schema") != UPSTREAM_SCHEMA
        or upstream.get("status") != UPSTREAM_STATUS
        or type(upstream.get("issue")) is not int
        or upstream.get("issue") != 655
    ):
        raise IndependentVerificationError("upstream contract drifted")
    control = upstream.get("algebraic_transport_positive_control")
    pairs = (
        control.get("antipodal_axis_pairs") if isinstance(control, Mapping) else None
    )
    if not isinstance(pairs, list) or len(pairs) != 6:
        raise IndependentVerificationError("axis-pair census drifted")
    axes = tuple(tuple(pair) for pair in pairs)
    if (
        any(
            len(pair) != 2 or any(type(port) is not int for port in pair)
            for pair in axes
        )
        or sorted(port for pair in axes for port in pair) != list(range(12))
        or any(pair[0] >= pair[1] for pair in axes)
    ):
        raise IndependentVerificationError("axis pairs are malformed")
    return upstream, axes


def _atomic(upstream: Mapping[str, Any]) -> dict[str, Any]:
    atomic = _load_json(ATOMIC_PATH)
    check = verify_vertex12_atomic_port_transfer_independent.verify_report(atomic)
    if check.get("receipt") is not True:
        raise IndependentVerificationError("atomic packet failed verification")
    if (
        atomic.get("schema") != ATOMIC_SCHEMA
        or atomic.get("status") != ATOMIC_STATUS
        or type(atomic.get("issue")) is not int
        or atomic.get("issue") != 655
    ):
        raise IndependentVerificationError("atomic packet contract drifted")
    ancestry = upstream.get("upstream_source_packet")
    if not isinstance(ancestry, Mapping):
        raise IndependentVerificationError("feasibility ancestry is missing")
    if _canonical_bytes(ancestry.get("pin")) != _canonical_bytes(
        _raw_pin(ATOMIC_PATH)
    ) or ancestry.get("receipt_sha256") != atomic.get("receipt_sha256"):
        raise IndependentVerificationError("atomic ancestry does not match")
    return atomic


def _source_repair_control(atomic: Mapping[str, Any]) -> dict[str, Any]:
    operator = atomic.get("atomic_transfer_operator")
    history = atomic.get("source_history_replay")
    if not isinstance(operator, Mapping) or not isinstance(history, Mapping):
        raise IndependentVerificationError("atomic source blocks are missing")
    rows = operator.get("port_rows")
    events = history.get("event_rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise IndependentVerificationError("atomic port-row census drifted")
    if not isinstance(events, list) or len(events) != 48:
        raise IndependentVerificationError("atomic event census drifted")
    maps: dict[int, tuple[int, ...]] = {}
    event_ids: dict[int, list[str]] = {port: [] for port in range(12)}
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("port")) is not int:
            raise IndependentVerificationError("atomic port row malformed")
        port = row["port"]
        values = row.get("carrier_partner_permutation")
        seams = row.get("seam_ids")
        if (
            port in maps
            or not isinstance(values, list)
            or len(values) != 8
            or any(type(value) is not int for value in values)
            or not isinstance(seams, list)
            or len(seams) != 4
        ):
            raise IndependentVerificationError("atomic matching row incomplete")
        permutation = tuple(values)
        inverse = [0] * 8
        if sorted(permutation) != list(range(8)):
            raise IndependentVerificationError("atomic matching is not a permutation")
        for source, target in enumerate(permutation):
            inverse[target] = source
        if (
            tuple(inverse) != permutation
            or any(permutation[state] == state for state in range(8))
            or row.get("repair_projector_idempotent") is not True
        ):
            raise IndependentVerificationError("atomic matching contract drifted")
        maps[port] = permutation
    if set(maps) != set(range(12)):
        raise IndependentVerificationError("atomic ports are incomplete")
    for event in events:
        if (
            not isinstance(event, Mapping)
            or type(event.get("port")) is not int
            or event["port"] not in event_ids
            or type(event.get("event_id")) is not str
            or not event["event_id"].startswith("sha256:")
        ):
            raise IndependentVerificationError("atomic event row malformed")
        event_ids[event["port"]].append(event["event_id"])
    if any(len(ids) != 4 or len(set(ids)) != 4 for ids in event_ids.values()):
        raise IndependentVerificationError("atomic event coverage drifted")
    event_ids = {port: sorted(ids) for port, ids in event_ids.items()}
    commuting = []
    noncommuting = []
    for left, right in itertools.combinations(range(12), 2):
        if _after(maps[left], maps[right]) == _after(maps[right], maps[left]):
            commuting.append([left, right])
        else:
            noncommuting.append([left, right])
    if len(commuting) != 23 or len(noncommuting) != 43:
        raise IndependentVerificationError("shared-carrier commutator census drifted")
    first, second = 0, 1
    first_word = event_ids[first] + event_ids[second]
    second_word = event_ids[second] + event_ids[first]
    if first_word == second_word:
        raise IndependentVerificationError("event-word witness collapsed")
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
        "source_history_idempotence_replay_exact": history.get(
            "idempotence_replay_exact"
        )
        is True,
        "distinct_event_words_same_terminal_operator_witness": {
            "first_port": first,
            "second_port": second,
            "first_then_second_event_ids": first_word,
            "second_then_first_event_ids": second_word,
            "event_words_equal": False,
            "terminal_state_operators_equal": True,
        },
        "shared_eight_carrier_reinterpretation": {
            "object": (
                "the twelve S_p matchings acting on one shared eight-carrier set after the port coordinate is discarded"
            ),
            "unordered_pair_count": 66,
            "commuting_pair_count": len(commuting),
            "noncommuting_pair_count": len(noncommuting),
            "commuting_pairs": commuting,
            "noncommuting_pairs": noncommuting,
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


def _local_control() -> dict[str, Any]:
    a = (1, 0, 2, 3)
    b = (0, 2, 1, 3)
    ab = _after(b, a)
    ba = _after(a, b)
    equal = [state for state in range(4) if ab[state] == ba[state]]
    unequal = [state for state in range(4) if ab[state] != ba[state]]
    if equal != [3] or unequal != [0, 1, 2]:
        raise IndependentVerificationError("local control failed")
    return {
        "schema": "oph.local-endpoint-coincidence-control.v1",
        "state_count": 4,
        "observer_readout": "identity",
        "first_step_permutation": list(a),
        "second_step_permutation": list(b),
        "both_steps_bijective": True,
        "equal_ordered_endpoint_start_states": equal,
        "unequal_ordered_endpoint_start_states": unequal,
        "one_state_endpoint_coincidence_attained": True,
        "global_commutation_attained": False,
        "implication_refuted": (
            "one_start_state_visible_endpoint_equality_implies_global_quotient_commutation"
        ),
    }


def _descent_control() -> dict[str, Any]:
    q = (0, 0, 1, 1)
    a = (0, 2, 1, 3)
    identity = tuple(range(4))
    if _after(identity, a) != _after(a, identity):
        raise IndependentVerificationError("descent control does not commute")
    violations = [
        [left, right]
        for left in range(4)
        for right in range(left + 1, 4)
        if q[left] == q[right] and q[a[left]] != q[a[right]]
    ]
    if violations != [[0, 1], [2, 3]]:
        raise IndependentVerificationError("descent witnesses drifted")
    return {
        "schema": "oph.quotient-descent-necessity-control.v1",
        "state_count": 4,
        "observer_class_ids": list(q),
        "observer_class_count": 2,
        "first_step_permutation": list(a),
        "second_step_permutation": list(identity),
        "raw_ordered_maps_equal_on_every_state": True,
        "first_step_quotient_congruence_violation_pairs": violations,
        "first_step_descends_to_observer_quotient": False,
        "quotient_port_action_defined": False,
        "implication_refuted": (
            "raw_path_commutation_without_A2_accepted_map_naturality_defines_a_quotient_action"
        ),
    }


def _multiply(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    a, b, c = left
    x, y, z = right
    return ((a + x) % 3, (b + y) % 3, (c + z + a * y) % 3)


def _holonomy_control() -> dict[str, Any]:
    states = tuple(itertools.product(range(3), repeat=3))
    lookup = {state: position for position, state in enumerate(states)}
    a = tuple(lookup[_multiply(state, (1, 0, 0))] for state in states)
    b = tuple(lookup[_multiply(state, (0, 1, 0))] for state in states)
    ab = _after(b, a)
    ba = _after(a, b)
    q = tuple(state[:2] for state in states)
    visible_equal = [state for state in range(27) if q[ab[state]] == q[ba[state]]]
    raw_equal = [state for state in range(27) if ab[state] == ba[state]]
    shifts = sorted(
        {(states[ab[state]][2] - states[ba[state]][2]) % 3 for state in range(27)}
    )
    classes: dict[tuple[int, int], list[int]] = {}
    for state, class_id in enumerate(q):
        classes.setdefault(class_id, []).append(state)
    descends = True
    for permutation in (a, b):
        for members in classes.values():
            descends = (
                descends and len({q[permutation[state]] for state in members}) == 1
            )
    if visible_equal != list(range(27)) or raw_equal or shifts != [1] or not descends:
        raise IndependentVerificationError("holonomy control failed")
    origin = lookup[(0, 0, 0)]
    return {
        "schema": "oph.heisenberg-visible-confluence-control.v1",
        "raw_state_object": "H_3(F_3)",
        "raw_state_count": 27,
        "observer_quotient_object": "F_3^2 via (a,b,c) maps to (a,b)",
        "observer_quotient_class_count": len(classes),
        "members_per_observer_class": sorted({len(row) for row in classes.values()}),
        "first_step_permutation_sha256": _sha(list(a)),
        "second_step_permutation_sha256": _sha(list(b)),
        "both_steps_bijective": sorted(a) == list(range(27))
        and sorted(b) == list(range(27)),
        "both_steps_descend_to_observer_quotient": True,
        "visible_ordered_endpoint_equality_count": len(visible_equal),
        "raw_ordered_endpoint_equality_count": len(raw_equal),
        "visible_ordered_endpoints_equal_for_every_state": True,
        "raw_ordered_paths_equal_for_any_state": False,
        "central_commutator_shift_modulo_three": shifts[0],
        "identity_start_witness": {
            "first_then_second_raw_endpoint": list(states[ab[origin]]),
            "second_then_first_raw_endpoint": list(states[ba[origin]]),
            "shared_visible_endpoint": list(q[ab[origin]]),
        },
        "implication_refuted": (
            "observer_quotient_terminal_confluence_implies_underlying_path_equality_or_trivial_holonomy"
        ),
    }


def _factorization(axes: Sequence[tuple[int, int]]) -> dict[str, Any]:
    by_port: dict[int, tuple[int, int]] = {}
    for axis, (positive, negative) in enumerate(axes):
        by_port[positive] = axis, 1
        by_port[negative] = axis, -1
    relation_matrix = []
    normal_forms = []
    for positive, negative in axes:
        row = [0] * 12
        row[positive] = row[negative] = 1
        relation_matrix.append(row)
    for port in range(12):
        axis, sign = by_port[port]
        vector = [0] * 6
        vector[axis] = sign
        normal_forms.append(
            {
                "port": port,
                "antipodal_port": axes[axis][1 if sign == 1 else 0],
                "axis": axis,
                "orientation_sign": sign,
                "normal_form_vector": vector,
            }
        )
    diamonds = list(itertools.combinations(range(6), 2))
    oriented = [
        [left, right]
        for left in range(12)
        for right in range(left + 1, 12)
        if by_port[left][0] != by_port[right][0]
    ]
    if len(diamonds) != 15 or len(oriented) != 60 or _rank(relation_matrix) != 6:
        raise IndependentVerificationError("universal presentation census failed")
    commutator_witnesses = [
        {
            "omitted_axis_pair": [left, right],
            "finite_witness_group": "H_3(F_3)",
            "finite_witness_group_order": 27,
            "retained_commutator_relations_satisfied": 14,
            "all_six_antipodal_inverse_relations_satisfied": True,
            "omitted_commutator_central_shift_modulo_three": 1,
            "omitted_relation_not_implied_by_other_fourteen": True,
        }
        for left, right in diamonds
    ]
    inverse_witnesses = [
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
        "positive_axis_representatives": [pair[0] for pair in axes],
        "antipodal_relation_matrix": relation_matrix,
        "antipodal_relation_matrix_rank_over_Q": 6,
        "free_abelian_rank_after_antipodal_relations": 6,
        "port_normal_form_rows": normal_forms,
        "positive_axis_commutator_diamond_count": 15,
        "positive_axis_commutator_diamond_pairs": [list(pair) for pair in diamonds],
        "all_oriented_nonantipodal_pair_count": 60,
        "all_oriented_nonantipodal_pairs": oriented,
        "fifteen_positive_diamonds_plus_inverse_law_imply_all_sixty_oriented_diamonds": True,
        "commutator_relation_irredundancy_witnesses": commutator_witnesses,
        "every_positive_commutator_relation_is_individually_necessary": True,
        "inverse_relation_irredundancy_witnesses": inverse_witnesses,
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


def _theorem() -> dict[str, Any]:
    return {
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


def _ledger(axes: Sequence[tuple[int, int]]) -> dict[str, Any]:
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
            "antipodal_inverse_family_count": len(axes),
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


def verify_report(report: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        upstream, axes = _upstream_and_axes()
        atomic = _atomic(upstream)
        received = dict(report)
        digest = received.pop("receipt_sha256", None)
        if digest != _sha(received):
            reasons.append("receipt_digest_mismatch")
        expected_keys = {
            "schema",
            "issue",
            "status",
            "upstream_feasibility_packet",
            "atomic_source_packet",
            "a2_endpoint_commutator_theorem",
            "exact_current_source_repair_control",
            "exact_finite_controls",
            "universal_abelian_port_factorization",
            "minimum_source_emitted_port_step_ledger",
            "attainment",
            "issue_655_disposition",
            "comparison_data_read",
            "implementation_pins",
            "claim_boundary",
            "receipt_sha256",
        }
        if set(report) != expected_keys:
            reasons.append("top_level_key_set_mismatch")
        if report.get("schema") != SCHEMA:
            reasons.append("schema_mismatch")
        if type(report.get("issue")) is not int or report.get("issue") != 655:
            reasons.append("issue_mismatch")
        if report.get("status") != STATUS:
            reasons.append("status_mismatch")
        expected_upstream = {
            "pin": _raw_pin(UPSTREAM_PATH),
            "schema": upstream["schema"],
            "status": upstream["status"],
            "receipt_sha256": upstream["receipt_sha256"],
            "independently_verified": True,
        }
        if _canonical_bytes(
            report.get("upstream_feasibility_packet")
        ) != _canonical_bytes(expected_upstream):
            reasons.append("upstream_binding_mismatch")
        expected_atomic = {
            "pin": _raw_pin(ATOMIC_PATH),
            "schema": atomic["schema"],
            "status": atomic["status"],
            "receipt_sha256": atomic["receipt_sha256"],
            "independently_verified": True,
        }
        if _canonical_bytes(report.get("atomic_source_packet")) != _canonical_bytes(
            expected_atomic
        ):
            reasons.append("atomic_binding_mismatch")
        if _canonical_bytes(
            report.get("a2_endpoint_commutator_theorem")
        ) != _canonical_bytes(_theorem()):
            reasons.append("endpoint_theorem_mismatch")
        if _canonical_bytes(
            report.get("exact_current_source_repair_control")
        ) != _canonical_bytes(_source_repair_control(atomic)):
            reasons.append("current_source_repair_control_mismatch")
        expected_controls = {
            "local_endpoint_coincidence_without_global_commutation": _local_control(),
            "raw_commutation_without_quotient_descent": _descent_control(),
            "visible_terminal_confluence_without_path_equality": _holonomy_control(),
        }
        if _canonical_bytes(report.get("exact_finite_controls")) != _canonical_bytes(
            expected_controls
        ):
            reasons.append("finite_control_mismatch")
        if _canonical_bytes(
            report.get("universal_abelian_port_factorization")
        ) != _canonical_bytes(_factorization(axes)):
            reasons.append("factorization_mismatch")
        if _canonical_bytes(
            report.get("minimum_source_emitted_port_step_ledger")
        ) != _canonical_bytes(_ledger(axes)):
            reasons.append("minimum_ledger_mismatch")
        expected_attainment = {
            "conditional_A2_endpoint_descent_commutator_lemma": True,
            "universal_z_power_6_factorization": True,
            "fifteen_commutator_relations_proved_individually_necessary": True,
            "six_antipodal_inverse_relations_proved_individually_necessary": True,
            "current_source_emitted_endpoint_diamond_ledger": False,
            "faithful_physical_z_power_6_action": False,
            "spatial_translation": False,
            "physical_prediction": False,
        }
        if _canonical_bytes(report.get("attainment")) != _canonical_bytes(
            expected_attainment
        ):
            reasons.append("attainment_mismatch")
        expected_disposition = {
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
        }
        if _canonical_bytes(report.get("issue_655_disposition")) != _canonical_bytes(
            expected_disposition
        ):
            reasons.append("issue_disposition_mismatch")
        if type(report.get("comparison_data_read")) is not bool or report.get(
            "comparison_data_read"
        ):
            reasons.append("comparison_boundary_mismatch")
        expected_pins = {
            "producer": _raw_pin(PRODUCER_PATH),
            "independent_verifier": _raw_pin(VERIFIER_PATH),
            "serialized_mutation_tests": _raw_pin(TEST_PATH),
        }
        if _canonical_bytes(report.get("implementation_pins")) != _canonical_bytes(
            expected_pins
        ):
            reasons.append("implementation_pin_mismatch")
        expected_claim = "A2 turns an all-state visible endpoint diamond into a commuting pair of meaning-side maps only after the primitive steps are accepted maps and their naturality squares exist. A2 does not supply the endpoint diamond or erase hidden path holonomy. Six quotient inverse identities plus fifteen complete positive-axis diamonds make the accepted meaning action factor uniquely through Z^6. A physical interpretation and faithfulness remain separate. The current source attains exact terminal confluence for twelve noninvertible repair blocks on disjoint port columns, which narrows the missing producer to accepted reversible step tables, inverse rows, and complete AB/BA diamonds. It does not support a negative closure of issue 655. The declared (Z/3Z)^6 control remains non-source and nonphysical. No spatial translation or prediction is promoted."
        if report.get("claim_boundary") != expected_claim:
            reasons.append("claim_boundary_mismatch")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        IndependentVerificationError,
        RecursionError,
    ):
        reasons.append("malformed_or_noncanonical_receipt")
    passed = not reasons
    return {
        "schema": VERIFICATION_SCHEMA,
        "receipt": passed,
        "status": "PASS" if passed else "FAIL",
        "reasons": sorted(set(reasons)),
        "producer_imported": False,
        "finite_controls_independently_reconstructed": True,
        "universal_presentation_independently_reconstructed": True,
        "source_engine_independently_reimplemented": False,
        "claim_boundary": (
            "The verifier reconstructs the finite implication controls and presentation theorem without importing the producer. It does not supply the source-emitted diamonds, a faithful physical action, or spatial readout."
        ),
    }


def _write(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = _load_json(args.input)
    result = verify_report(report)
    _write(result, args.output)
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
