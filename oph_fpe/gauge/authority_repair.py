"""Protected-authority repair for gauge-coupled port pairs.

The legacy array kernel chooses which endpoint to overwrite from a random
stream.  That choice is schedule dependent and is therefore unsuitable for a
finite-consensus receipt on a graph with shared patch frames.  This module
implements the narrower repair law in which a protected, source-bound node
authority decides the endpoint before scheduling begins.

Authority values are immutable, pairwise distinct signed 64-bit integers;
the larger value has higher authority.  On an inconsistent oriented seam
``i -> j`` the rule preserves the higher-authority endpoint and transports
that label to the lower-authority endpoint through the fixed link ``g_ij``.
Thus the link, and hence every link-only holonomy, is unchanged.  Authorities
are metadata, not gauge labels, and must travel with nodes under a graph
relabeling.

This is an exact edge-slot kernel.  It does not by itself construct the
protected authority source, prove that authority is physical, or discharge
patch-local constraints coupling several port slots.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from oph_fpe.finite_groups import S3_MUL
from oph_fpe.gauge.covariant_overlap import (
    GAUGE_COVARIANT_OVERLAP_SCHEMA,
    covariant_mismatch_mask,
    repair_covariant_port_pairs,
)


AUTHORITY_REPAIR_SCHEMA = "oph_protected_node_authority_port_repair_v1"
AUTHORITY_SCHEMA = "oph_protected_distinct_node_authority_v1"
PROTECTED_AUTHORITY_SOURCE_HASH_SCHEMA = (
    "oph_protected_authority_coupled_source_hash_v1"
)
PROTECTED_AUTHORITY_TERMINAL_HASH_SCHEMA = (
    "oph_protected_authority_terminal_quotient_hash_v1"
)
ENDPOINT_SYMMETRIC_S3_MERGE_NO_GO_RECEIPT = (
    "ENDPOINT_SYMMETRIC_S3_REGULAR_ACTION_MERGE_NO_GO_RECEIPT"
)


def validate_node_authorities(
    authorities: Sequence[int] | np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Return an immutable exact copy of one distinct authority per node.

    Larger integers have higher authority.  Booleans, floats, collisions, and
    values outside signed 64-bit range are rejected rather than coerced.
    """

    if isinstance(node_count, (bool, np.bool_)) or not isinstance(
        node_count, (int, np.integer)
    ):
        raise TypeError("node_count must be an integer")
    count = int(node_count)
    if count < 0:
        raise ValueError("node_count must be nonnegative")

    raw = np.asarray(authorities)
    if raw.ndim != 1 or raw.shape != (count,):
        raise ValueError("authorities must contain exactly one value per node")
    values: list[int] = []
    for raw_value in raw.tolist():
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, np.integer)
        ):
            raise TypeError("node authorities must be exact integers, not booleans or floats")
        value = int(raw_value)
        if value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max:
            raise ValueError("node authority lies outside signed 64-bit range")
        values.append(value)
    if len(set(values)) != count:
        raise ValueError("node authorities must be pairwise distinct")

    # Back with immutable ``bytes`` rather than merely clearing WRITEABLE on
    # an owned allocation (whose flag a caller could set again).
    encoded = np.asarray(values, dtype="<i8").tobytes()
    result = np.frombuffer(encoded, dtype="<i8")
    return result


def authority_repair_directions(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edges: np.ndarray,
    authorities: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return endpoint writes selected by protected node authority.

    The result follows :func:`repair_covariant_port_pairs`: ``True`` means
    rewrite the left endpoint.  This occurs exactly when the right endpoint
    has higher authority and must therefore be preserved.
    """

    left, right = _validate_graph_edges(edge_left, edge_right)
    selected = _validate_selected_edges(edges, left.size)
    raw_authorities = np.asarray(authorities)
    if raw_authorities.ndim != 1:
        raise ValueError("authorities must be one-dimensional")
    authority = validate_node_authorities(authorities, int(raw_authorities.size))
    if left.size:
        highest_node = int(max(int(np.max(left)), int(np.max(right))))
        if highest_node >= authority.size:
            raise ValueError("edge endpoint is not covered by node authorities")
    if selected.size and np.any(left[selected] == right[selected]):
        raise ValueError(
            "node authority cannot distinguish the two endpoints of a self-loop"
        )
    directions = authority[right[selected]] > authority[left[selected]]
    return np.asarray(directions, dtype=bool)


def authority_sha256(
    authorities: Sequence[int] | np.ndarray,
) -> str:
    """Return the canonical, schema-separated hash of an authority order."""

    raw = np.asarray(authorities)
    if raw.ndim != 1:
        raise ValueError("authorities must be one-dimensional")
    authority = validate_node_authorities(authorities, int(raw.size))
    values = np.ascontiguousarray(authority, dtype="<i8")
    hasher = hashlib.sha256()
    hasher.update((AUTHORITY_SCHEMA + "\0").encode("ascii"))
    hasher.update(int(values.size).to_bytes(8, "little", signed=False))
    hasher.update(values.tobytes())
    return "sha256:" + hasher.hexdigest()


def protected_authority_source_sha256(
    coupled_source_sha256: str,
    authorities: Sequence[int] | np.ndarray,
) -> str:
    """Bind a coupled port/link source hash to its protected authority input."""

    return _authority_bound_sha256(
        coupled_source_sha256,
        authorities,
        schema=PROTECTED_AUTHORITY_SOURCE_HASH_SCHEMA,
    )


def protected_authority_terminal_sha256(
    terminal_quotient_sha256: str,
    authorities: Sequence[int] | np.ndarray,
) -> str:
    """Bind a terminal gauge-quotient hash to its protected authority input."""

    return _authority_bound_sha256(
        terminal_quotient_sha256,
        authorities,
        schema=PROTECTED_AUTHORITY_TERMINAL_HASH_SCHEMA,
    )


def repair_authority_directed_port_pairs(
    port_left: np.ndarray,
    port_right: np.ndarray,
    gauge: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edges: np.ndarray,
    authorities: Sequence[int] | np.ndarray,
    *,
    group_name: str,
    group_order: int,
) -> dict[str, Any]:
    """Repair selected mismatches in place and return an exact audit receipt.

    Consistent selected seams are left untouched.  Every selected mismatch is
    eliminated, the higher-authority endpoint value is preserved exactly, and
    no gauge link is written.  Since an edge slot belongs to one seam only,
    the global mismatch count drops by precisely the number of enabled
    selected seams, independent of selection order or batch partition.
    """

    left_values, right_values, gauge_values = _validate_port_state(
        port_left, port_right, gauge
    )
    graph_left, graph_right = _validate_graph_edges(edge_left, edge_right)
    if graph_left.shape != left_values.shape:
        raise ValueError("graph endpoint arrays must match port-pair arrays")
    selected = _validate_selected_edges(edges, left_values.size)
    directions = authority_repair_directions(
        graph_left,
        graph_right,
        selected,
        authorities,
    )
    raw_authorities = np.asarray(authorities)
    authority = validate_node_authorities(authorities, int(raw_authorities.size))

    mismatch_before = covariant_mismatch_mask(
        left_values,
        right_values,
        gauge_values,
        group_name=group_name,
        group_order=group_order,
    )
    active_mask = mismatch_before[selected]
    active = selected[active_mask]
    active_directions = directions[active_mask]
    phi_before = int(np.count_nonzero(mismatch_before))

    gauge_before = np.asarray(gauge_values).copy()
    preserved_left_edges = active[~active_directions]
    preserved_right_edges = active[active_directions]
    preserved_left_values = left_values[preserved_left_edges].copy()
    preserved_right_values = right_values[preserved_right_edges].copy()

    if active.size:
        repair_covariant_port_pairs(
            left_values,
            right_values,
            gauge_values,
            active,
            active_directions,
            group_name=group_name,
            group_order=group_order,
        )

    mismatch_after = covariant_mismatch_mask(
        left_values,
        right_values,
        gauge_values,
        group_name=group_name,
        group_order=group_order,
    )
    phi_after = int(np.count_nonzero(mismatch_after))
    gauge_fixed = bool(np.array_equal(gauge_values, gauge_before))
    higher_authority_endpoints_preserved = bool(
        np.array_equal(left_values[preserved_left_edges], preserved_left_values)
        and np.array_equal(right_values[preserved_right_edges], preserved_right_values)
    )
    selected_mismatches_eliminated = bool(
        not np.any(mismatch_after[active]) if active.size else True
    )
    exact_delta = int(phi_after - phi_before)
    exact_strict_descent = bool(
        active.size > 0
        and exact_delta == -int(active.size)
        and selected_mismatches_eliminated
    )
    invariant_pass = bool(
        gauge_fixed
        and higher_authority_endpoints_preserved
        and selected_mismatches_eliminated
        and exact_delta == -int(active.size)
    )
    if not invariant_pass:
        raise RuntimeError("protected-authority repair invariant failed")

    return {
        "schema": AUTHORITY_REPAIR_SCHEMA,
        "authority_schema": AUTHORITY_SCHEMA,
        "authority_sha256": authority_sha256(authority),
        "authority_order": "larger_signed_integer_has_higher_authority",
        "mismatch_definition": GAUGE_COVARIANT_OVERLAP_SCHEMA,
        "edge_count": int(left_values.size),
        "selected_edge_count": int(selected.size),
        "enabled_selected_edge_count": int(active.size),
        "rewrite_left_count": int(np.count_nonzero(active_directions)),
        "rewrite_right_count": int(active.size - np.count_nonzero(active_directions)),
        "phi_before": phi_before,
        "phi_after": phi_after,
        "delta_phi": exact_delta,
        "expected_delta_phi": -int(active.size),
        "gauge_links_unchanged": gauge_fixed,
        "higher_authority_endpoints_preserved": higher_authority_endpoints_preserved,
        "selected_mismatches_eliminated": selected_mismatches_eliminated,
        "exact_strict_descent": exact_strict_descent,
        "receipt": exact_strict_descent,
        "claim_boundary": (
            "Exact finite edge-slot repair conditional on protected, distinct node "
            "authorities. The receipt proves fixed-link covariant mismatch descent for "
            "this selected batch; it does not source the authorities or prove coupled "
            "patch-local, record, refinement, or physical dynamics."
        ),
    }


def endpoint_symmetric_s3_merge_no_go_report() -> dict[str, Any]:
    """Return an exact obstruction to a symmetric deterministic S3 merge.

    For a regular left S3 action, equivariance and endpoint symmetry imply for
    any involution ``s != e`` and ``a = m(e,s)`` that
    ``a = m(s,e) = s*m(e,s) = s*a``.  The regular action is free, so no such
    ``a`` exists.  A deterministic repair must therefore carry an asymmetry
    such as protected authority, change the state space, or abandon endpoint
    symmetry.
    """

    identity = 0
    order = int(S3_MUL.shape[0])
    table_square = bool(S3_MUL.shape == (order, order) and order == 6)
    identity_verified = bool(
        table_square
        and np.array_equal(S3_MUL[identity], np.arange(order))
        and np.array_equal(S3_MUL[:, identity], np.arange(order))
    )
    regular_left_action_verified = bool(
        table_square
        and all(sorted(int(value) for value in row) == list(range(order)) for row in S3_MUL)
    )
    involutions = [
        element
        for element in range(order)
        if element != identity and int(S3_MUL[element, element]) == identity
    ]
    witness = involutions[0] if involutions else None
    fixed_points = (
        [
            value
            for value in range(order)
            if int(S3_MUL[int(witness), value]) == value
        ]
        if witness is not None
        else []
    )
    passed = bool(
        identity_verified
        and regular_left_action_verified
        and witness is not None
        and not fixed_points
    )
    return {
        "schema": "oph_endpoint_symmetric_s3_merge_no_go_v1",
        "group": "S3_regular_left_action",
        "group_order": order,
        "identity_verified": identity_verified,
        "regular_left_action_verified": regular_left_action_verified,
        "nonidentity_involution": witness,
        "involution_fixed_points": fixed_points,
        "equations": [
            "a = m(e,s)",
            "m(e,s) = m(s,e)",
            "m(s,e) = s*m(e,s)",
            "therefore a = s*a",
        ],
        ENDPOINT_SYMMETRIC_S3_MERGE_NO_GO_RECEIPT: passed,
        "receipt": passed,
        "claim_boundary": (
            "No-go only for deterministic endpoint-symmetric equivariant merges on "
            "the regular S3 label space. It does not exclude authority-ordered rules, "
            "stochastic kernels, or merges on a different working coordinate."
        ),
    }


def _validate_port_state(
    port_left: np.ndarray,
    port_right: np.ndarray,
    gauge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(port_left)
    right = np.asarray(port_right)
    links = np.asarray(gauge)
    if left.ndim != 1 or right.shape != left.shape or links.shape != left.shape:
        raise ValueError("port and gauge arrays must be matching one-dimensional arrays")
    if not left.flags.writeable or not right.flags.writeable:
        raise ValueError("port arrays must be writable")
    if (
        np.shares_memory(left, right)
        or np.shares_memory(left, links)
        or np.shares_memory(right, links)
    ):
        raise ValueError("port and gauge arrays must not share memory")
    return left, right, links


def _validate_graph_edges(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(edge_left)
    right = np.asarray(edge_right)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("edge endpoint arrays must be matching one-dimensional arrays")
    if left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
        raise TypeError("edge endpoint arrays must contain integers")
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("edge endpoints must be nonnegative")
    return left, right


def _validate_selected_edges(edges: np.ndarray, edge_count: int) -> np.ndarray:
    selected = np.asarray(edges)
    if selected.ndim != 1:
        raise ValueError("selected edges must be one-dimensional")
    if selected.dtype.kind not in "iu":
        raise TypeError("selected edges must contain integers")
    selected = selected.astype(np.int64, copy=False)
    if np.any((selected < 0) | (selected >= int(edge_count))):
        raise ValueError("selected edge index out of bounds")
    if np.unique(selected).size != selected.size:
        raise ValueError("selected edges must not contain duplicates")
    return selected


def _authority_bound_sha256(
    state_sha256: str,
    authorities: Sequence[int] | np.ndarray,
    *,
    schema: str,
) -> str:
    if not isinstance(state_sha256, str):
        raise TypeError("state hash must be a sha256 string")
    digest = state_sha256.removeprefix("sha256:")
    if (
        not state_sha256.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("state hash must be a lowercase sha256 receipt")
    authority_hash = authority_sha256(authorities)
    hasher = hashlib.sha256()
    hasher.update((schema + "\0").encode("ascii"))
    hasher.update(state_sha256.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(authority_hash.encode("ascii"))
    return "sha256:" + hasher.hexdigest()
