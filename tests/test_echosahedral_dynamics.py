from __future__ import annotations

import math

import numpy as np
import pytest

from oph_fpe.core.echosahedral_dynamics import (
    initialize_local_recurrent_carriers,
    local_a5_dynamics_report,
    local_port_statistics,
    propagate_local_recurrent_carriers,
    reference_icosahedral_coupling,
)


def test_reference_coupling_is_exact_twelve_port_icosahedral_laplacian() -> None:
    coupling = reference_icosahedral_coupling()

    assert coupling.shape == (12, 12)
    assert np.allclose(coupling, coupling.T)
    assert np.allclose(np.diag(coupling), 5.0)
    assert np.count_nonzero(np.triu(coupling < 0.0, k=1)) == 30
    assert np.allclose(np.sum(coupling, axis=1), 0.0)


def test_local_propagation_preserves_norm_and_intrinsic_phase_mod_one() -> None:
    source = initialize_local_recurrent_carriers(7, seed=17)
    propagated = propagate_local_recurrent_carriers(
        source,
        intrinsic_step=0.375,
    )

    assert np.allclose(np.linalg.norm(source.amplitudes, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(propagated.amplitudes, axis=1), 1.0)
    assert np.all((propagated.intrinsic_phase >= 0.0) & (propagated.intrinsic_phase < 1.0))
    assert np.allclose(
        propagated.intrinsic_phase,
        np.mod(source.intrinsic_phase + 0.375, 1.0),
    )
    assert np.allclose(np.sum(local_port_statistics(propagated), axis=1), 1.0)
    assert source.amplitudes.flags.writeable is False
    assert source.intrinsic_phase.flags.writeable is False


def test_local_dynamics_is_a5_equivariant_but_does_not_select_2pi() -> None:
    report = local_a5_dynamics_report()

    assert report["a5_action_count"] == 60
    assert report["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"] is True
    assert report["LOCAL_NONTRIVIAL_REVERSIBLE_PROPAGATION_RECEIPT"] is True
    assert report["LOCAL_PHASE_REGISTER_MOD_ONE_RECEIPT"] is True
    assert report["LOCAL_DYNAMICS_DESCENDS_TO_R_MOD_Z_RECEIPT"] is False
    assert report["LOCAL_INTRINSIC_R_MOD_Z_RECURRENCE_RECEIPT"] is False
    assert report["hidden_xyz_coordinates_used_by_dynamics"] is False
    assert report["global_support_chart_used_by_dynamics"] is False
    assert report["candidate_clock_scale_used_by_dynamics"] is False
    assert report["PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT"] is False
    assert report["BW_KMS_CLOCK_RECEIPT"] is False
    assert report["PHYSICAL_H3_KMS_EMERGENCE_RECEIPT"] is False
    assert sorted(
        row["multiplicity"] for row in report["coupling_spectrum"]
    ) == [1, 3, 3, 5]


def test_recurrence_step_is_not_preloaded_with_a_clock_candidate() -> None:
    one = local_a5_dynamics_report(intrinsic_step=0.125)
    two = local_a5_dynamics_report(intrinsic_step=math.sqrt(2.0) / 10.0)

    assert one["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"] is True
    assert two["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"] is True
    assert one["PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT"] is False
    assert two["PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT"] is False
    assert one["dynamics_sha256"] != two["dynamics_sha256"]


def test_trivial_identity_channel_does_not_pass_nontrivial_propagation() -> None:
    report = local_a5_dynamics_report(intrinsic_step=0.0)

    assert report["LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT"] is True
    assert report["LOCAL_NONTRIVIAL_REVERSIBLE_PROPAGATION_RECEIPT"] is False
    assert report["propagation_nontriviality_norm"] == 0.0


def test_invalid_carrier_state_shapes_fail_closed() -> None:
    source = initialize_local_recurrent_carriers(2, seed=1)
    with pytest.raises(ValueError, match="shape"):
        type(source)(
            amplitudes=np.zeros((2, 11), dtype=np.complex128),
            intrinsic_phase=source.intrinsic_phase,
        )
