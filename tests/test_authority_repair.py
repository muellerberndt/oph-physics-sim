from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from oph_fpe.gauge.authority_repair import (
    ENDPOINT_SYMMETRIC_S3_MERGE_NO_GO_RECEIPT,
    authority_repair_directions,
    endpoint_symmetric_s3_merge_no_go_report,
    repair_authority_directed_port_pairs,
    validate_node_authorities,
)
from oph_fpe.gauge.covariant_overlap import (
    covariant_mismatch_mask,
    gauge_quotient_state_hash,
    group_inverse_indices,
    transform_local_frames,
)


def _path_state() -> tuple[np.ndarray, ...]:
    return (
        np.asarray([1, 4], dtype=np.int16),
        np.asarray([3, 2], dtype=np.int16),
        np.asarray([5, 2], dtype=np.int16),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([10, 30, 20], dtype=np.int64),
    )


def _repair(
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edges: np.ndarray,
    authorities: np.ndarray,
) -> dict:
    return repair_authority_directed_port_pairs(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        edges,
        authorities,
        group_name="S3",
        group_order=6,
    )


def test_authorities_are_exact_distinct_and_returned_immutable() -> None:
    source = np.asarray([9, 2, 5], dtype=np.int64)
    authority = validate_node_authorities(source, 3)

    source[0] = 100
    assert authority.tolist() == [9, 2, 5]
    assert authority.flags.writeable is False
    with pytest.raises(ValueError):
        authority[0] = 7
    with pytest.raises(ValueError):
        authority.setflags(write=True)
    with pytest.raises(ValueError, match="pairwise distinct"):
        validate_node_authorities([1, 1], 2)
    with pytest.raises(TypeError, match="exact integers"):
        validate_node_authorities([1.0, 2.0], 2)
    with pytest.raises(TypeError, match="exact integers"):
        validate_node_authorities([True, False], 2)


def test_direction_rewrites_lower_authority_endpoint() -> None:
    _, _, _, edge_left, edge_right, authorities = _path_state()
    directions = authority_repair_directions(
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    # edge 0 preserves right/node 1 (authority 30), so left is rewritten;
    # edge 1 preserves left/node 1, so right is rewritten.
    assert directions.tolist() == [True, False]


def test_authority_repair_has_exact_descent_and_fixed_links() -> None:
    left, right, gauge, edge_left, edge_right, authorities = _path_state()
    gauge_before = gauge.copy()
    left_winner = int(left[1])
    right_winner = int(right[0])

    report = _repair(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    assert report["receipt"] is True
    assert report["exact_strict_descent"] is True
    assert report["phi_before"] == 2
    assert report["phi_after"] == 0
    assert report["delta_phi"] == -2
    assert report["rewrite_left_count"] == 1
    assert report["rewrite_right_count"] == 1
    assert report["gauge_links_unchanged"] is True
    assert report["higher_authority_endpoints_preserved"] is True
    assert int(right[0]) == right_winner
    assert int(left[1]) == left_winner
    assert np.array_equal(gauge, gauge_before)
    assert not covariant_mismatch_mask(
        left,
        right,
        gauge,
        group_name="S3",
        group_order=6,
    ).any()


def test_authority_repair_is_schedule_and_batch_independent() -> None:
    source = _path_state()
    hashes: set[str] = set()
    representatives: set[tuple[tuple[int, ...], ...]] = set()
    for order in permutations((0, 1)):
        left, right, gauge, edge_left, edge_right, authorities = (
            values.copy() for values in source
        )
        for edge in order:
            report = _repair(
                left,
                right,
                gauge,
                edge_left,
                edge_right,
                np.asarray([edge], dtype=np.int64),
                authorities,
            )
            assert report["exact_strict_descent"] is True
        representatives.add(
            (tuple(left.tolist()), tuple(right.tolist()), tuple(gauge.tolist()))
        )
        hashes.add(
            gauge_quotient_state_hash(
                left,
                right,
                gauge,
                edge_left=edge_left,
                edge_right=edge_right,
                group_name="S3",
                group_order=6,
            )
        )

    assert len(representatives) == 1
    assert len(hashes) == 1


def test_authority_repair_is_covariant_under_local_frames() -> None:
    left0, right0, gauge0, edge_left, edge_right, authorities = _path_state()
    frames = np.asarray([3, 4, 2], dtype=np.int16)
    transformed = transform_local_frames(
        left0,
        right0,
        gauge0,
        edge_left,
        edge_right,
        frames,
        group_name="S3",
        group_order=6,
    )

    left, right, gauge = left0.copy(), right0.copy(), gauge0.copy()
    _repair(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )
    expected = transform_local_frames(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        frames,
        group_name="S3",
        group_order=6,
    )

    transformed_left, transformed_right, transformed_gauge = (
        values.copy() for values in transformed
    )
    _repair(
        transformed_left,
        transformed_right,
        transformed_gauge,
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    assert all(
        np.array_equal(observed, target)
        for observed, target in zip(
            (transformed_left, transformed_right, transformed_gauge),
            expected,
            strict=True,
        )
    )


def test_authority_repair_is_invariant_under_stored_edge_reversal() -> None:
    left0, right0, gauge0, edge_left, edge_right, authorities = _path_state()
    left, right, gauge = left0.copy(), right0.copy(), gauge0.copy()
    _repair(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    reversed_left = right0.copy()
    reversed_right = left0.copy()
    reversed_gauge = group_inverse_indices(
        gauge0,
        group_name="S3",
        group_order=6,
    ).copy()
    _repair(
        reversed_left,
        reversed_right,
        reversed_gauge,
        edge_right,
        edge_left,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    assert np.array_equal(reversed_left, right)
    assert np.array_equal(reversed_right, left)
    assert np.array_equal(
        reversed_gauge,
        group_inverse_indices(gauge, group_name="S3", group_order=6),
    )


def test_authority_repair_is_invariant_under_node_relabeling() -> None:
    left0, right0, gauge0, edge_left, edge_right, authorities = _path_state()
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    relabeled_authorities = np.empty_like(authorities)
    relabeled_authorities[permutation] = authorities

    left, right, gauge = left0.copy(), right0.copy(), gauge0.copy()
    _repair(
        left,
        right,
        gauge,
        edge_left,
        edge_right,
        np.asarray([0, 1], dtype=np.int64),
        authorities,
    )

    relabeled_left = left0.copy()
    relabeled_right = right0.copy()
    relabeled_gauge = gauge0.copy()
    _repair(
        relabeled_left,
        relabeled_right,
        relabeled_gauge,
        permutation[edge_left],
        permutation[edge_right],
        np.asarray([0, 1], dtype=np.int64),
        relabeled_authorities,
    )

    assert np.array_equal(relabeled_left, left)
    assert np.array_equal(relabeled_right, right)
    assert np.array_equal(relabeled_gauge, gauge)


def test_authority_repair_rejects_unoriented_authority_ties_and_bad_edges() -> None:
    left, right, gauge, edge_left, edge_right, authorities = _path_state()
    with pytest.raises(ValueError, match="pairwise distinct"):
        _repair(
            left,
            right,
            gauge,
            edge_left,
            edge_right,
            np.asarray([0], dtype=np.int64),
            np.asarray([1, 1, 2], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="duplicates"):
        _repair(
            left,
            right,
            gauge,
            edge_left,
            edge_right,
            np.asarray([0, 0], dtype=np.int64),
            authorities,
        )
    with pytest.raises(ValueError, match="self-loop"):
        _repair(
            left[:1].copy(),
            right[:1].copy(),
            gauge[:1].copy(),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            authorities,
        )


def test_endpoint_symmetric_s3_merge_no_go_is_exact() -> None:
    report = endpoint_symmetric_s3_merge_no_go_report()

    assert report[ENDPOINT_SYMMETRIC_S3_MERGE_NO_GO_RECEIPT] is True
    assert report["receipt"] is True
    assert report["identity_verified"] is True
    assert report["regular_left_action_verified"] is True
    assert report["nonidentity_involution"] is not None
    assert report["involution_fixed_points"] == []
    assert "authority-ordered" in report["claim_boundary"]
