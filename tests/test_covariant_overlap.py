import numpy as np

from oph_fpe.gauge.covariant_overlap import (
    absorb_discrepancy_into_gauge,
    covariant_discrepancy,
    covariant_mismatch_mask,
    gauge_quotient_state_hash,
    repair_covariant_port_pairs,
    transform_local_frames,
)


def test_s3_mismatch_and_quotient_hash_are_invariant_under_local_frames():
    port_left = np.asarray([0, 2, 4, 5], dtype=np.int16)
    port_right = np.asarray([1, 3, 0, 2], dtype=np.int16)
    gauge = np.asarray([2, 0, 5, 1], dtype=np.int16)
    edge_left = np.asarray([0, 0, 1, 2], dtype=np.int64)
    edge_right = np.asarray([1, 2, 2, 3], dtype=np.int64)
    frames = np.asarray([4, 1, 5, 2], dtype=np.int16)

    transformed = transform_local_frames(
        port_left,
        port_right,
        gauge,
        edge_left,
        edge_right,
        frames,
        group_name="S3",
        group_order=6,
    )

    original_mask = covariant_mismatch_mask(
        port_left,
        port_right,
        gauge,
        group_name="S3",
        group_order=6,
    )
    transformed_mask = covariant_mismatch_mask(
        *transformed,
        group_name="S3",
        group_order=6,
    )
    assert np.array_equal(original_mask, transformed_mask)
    assert gauge_quotient_state_hash(
        port_left,
        port_right,
        gauge,
        edge_left=edge_left,
        edge_right=edge_right,
        group_name="S3",
        group_order=6,
    ) == gauge_quotient_state_hash(
        *transformed,
        edge_left=edge_left,
        edge_right=edge_right,
        group_name="S3",
        group_order=6,
    )


def test_covariant_port_repair_handles_both_endpoints_for_nonabelian_s3():
    port_left = np.asarray([1, 4], dtype=np.int16)
    port_right = np.asarray([3, 2], dtype=np.int16)
    gauge = np.asarray([5, 2], dtype=np.int16)
    assert covariant_mismatch_mask(
        port_left,
        port_right,
        gauge,
        group_name="S3",
        group_order=6,
    ).tolist() == [True, True]

    repair_covariant_port_pairs(
        port_left,
        port_right,
        gauge,
        np.asarray([0, 1]),
        np.asarray([True, False]),
        group_name="S3",
        group_order=6,
    )

    assert not np.any(
        covariant_mismatch_mask(
            port_left,
            port_right,
            gauge,
            group_name="S3",
            group_order=6,
        )
    )


def test_discrepancy_left_composition_absorbs_link_mismatch_exactly():
    port_left = np.asarray([1, 4, 5], dtype=np.int16)
    port_right = np.asarray([3, 2, 0], dtype=np.int16)
    gauge = np.asarray([5, 1, 2], dtype=np.int16)
    discrepancy = covariant_discrepancy(
        port_left,
        port_right,
        gauge,
        group_name="S3",
        group_order=6,
    )

    absorb_discrepancy_into_gauge(
        gauge,
        np.arange(3),
        discrepancy,
        group_name="S3",
        group_order=6,
    )

    assert not np.any(
        covariant_mismatch_mask(
            port_left,
            port_right,
            gauge,
            group_name="S3",
            group_order=6,
        )
    )


def test_clock_group_transport_is_additive_modulo_group_order():
    port_left = np.asarray([0, 2, 3], dtype=np.int16)
    port_right = np.asarray([3, 0, 1], dtype=np.int16)
    gauge = np.asarray([1, 2, 1], dtype=np.int16)

    assert covariant_mismatch_mask(
        port_left,
        port_right,
        gauge,
        group_name="C4",
        group_order=4,
    ).tolist() == [False, False, True]
