import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.collar_state import (
    cap_collar_partition,
    classical_cmi,
    classical_diagonal_cmi_nats,
    fawzi_renner_bound,
    fawzi_renner_trace_bound_nats,
    visible_packet_encoding_report,
    visible_packets,
)
from oph_fpe.core.graph import fibonacci_sphere_points


def test_cap_collar_partition_disjoint_and_complete():
    points = fibonacci_sphere_points(256)
    cap = RoundCap(axis=np.array([0.0, 0.0, 1.0]), theta0=0.75, tangent=np.array([1.0, 0.0, 0.0]), collar_width=0.08)

    partition = cap_collar_partition(points, cap)

    combined = partition.inside_mask.astype(int) + partition.collar_mask.astype(int) + partition.outside_mask.astype(int)
    assert np.all(combined == 1)
    assert np.sum(partition.inside_mask) > 0
    assert np.sum(partition.collar_mask) > 0
    assert np.sum(partition.outside_mask) > 0


def test_visible_packets_has_stable_integer_ids():
    state = {
        "record_signature": np.array([10, 10, 11, 12]),
        "committed_mask": np.array([1, 1, 0, 0]),
        "stable_count": np.array([3, 4, 1, 2]),
        "repair_load": np.array([0.1, 0.2, 0.8, 0.9]),
        "s3_class_density": np.array([0.0, 0.0, 0.5, 1.0]),
    }

    packets = visible_packets(state, {"stable_count": 2, "repair_load": 2, "s3_class_density": 3})

    assert packets.dtype.kind in {"i", "u"}
    assert packets.shape == (4,)
    assert len(set(packets.tolist())) >= 2


def test_visible_packets_reuses_cached_encoding_for_same_state_and_bins():
    state = {
        "record_signature": np.array([10, 10, 11, 12]),
        "stable_count": np.array([3, 4, 1, 2]),
        "repair_load": np.array([0.1, 0.2, 0.8, 0.9]),
    }

    first = visible_packets(state, {"stable_count": 2, "repair_load": 2})
    second = visible_packets(state, {"stable_count": 2, "repair_load": 2})
    different_bins = visible_packets(state, {"stable_count": 3, "repair_load": 2})

    assert second is first
    assert different_bins is not first
    assert different_bins.shape == first.shape


def test_classical_cmi_zero_for_independent_ad_given_b():
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    assert classical_cmi(a, b, d) < 1e-12
    assert classical_diagonal_cmi_nats(a, b, d) < 1e-12


def test_classical_cmi_positive_for_coupled_ad():
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = a.copy()
    b = np.zeros_like(a)

    assert classical_cmi(a, b, d) > 0.1


def test_fawzi_renner_bound_monotone():
    assert fawzi_renner_bound(0.0) == 0.0
    assert fawzi_renner_bound(0.25) > fawzi_renner_bound(0.01)
    assert fawzi_renner_trace_bound_nats(100.0) == 2.0
    assert np.isclose(fawzi_renner_trace_bound_nats(0.25), 2.0 * np.sqrt(1.0 - np.exp(-0.25)))


def test_visible_packet_encoding_report_exposes_diagnostic_semantics():
    state = {
        "record_signature": np.array([10, 10, 11, 12]),
        "stable_count": np.array([3.0, 4.0, 1.0, 2.0]),
    }

    report = visible_packet_encoding_report(state, {"stable_count": 2})

    assert report["state_semantics"] == "classical_commuting"
    assert report["log_unit"] == "nat"
    assert report["fields"][0]["collision_count"] == 0
    assert report["THEOREM_SAFE_PACKET_ENCODING_RECEIPT"] is False
