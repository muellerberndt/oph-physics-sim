from __future__ import annotations

from dataclasses import replace

from oph_fpe.consensus.incidence_nerve_bridge import (
    incidence_nerve_bridge_report,
)
from oph_fpe.core.echosahedral_federation import (
    controlled_oriented_s2_limit_report,
    reference_incidence_nerve_federation,
)


def test_incidence_nerve_bridge_closes_every_finite_596_gate() -> None:
    report = incidence_nerve_bridge_report()

    assert report["NONVACUOUS_PHYSICAL_FEDERATION_RECEIPT"] is True
    assert (
        report["PHASE_TO_ACCEPTED_CONFLUENT_REPAIR_ON_FEDERATION_RECEIPT"]
        is True
    )
    assert report["OPERATIONAL_OBSERVER_ON_FEDERATION_RECEIPT"] is True
    assert report["CONTROLLED_ORIENTED_S2_LIMIT_ON_SAME_TOWER_RECEIPT"] is True
    assert report["PHYSICAL_FEDERATION_SUPPORT_BRIDGE_RECEIPT"] is True
    federation = report["federation_verification"]
    assert federation["sewing"]["declared_triple_overlap_count"] == 20
    assert federation["sewing"]["blockers"] == []
    assert report["phase_to_repair"]["induced_repairs"]["candidate_pair_source"] == (
        "declared_federation_seams"
    )
    assert report["phase_to_repair"]["confluence"][
        "pairwise_disjoint_transactions"
    ] is True
    observer = report["operational_observer"]
    assert observer["phase_readback_complete"] is True
    assert observer["feedback_ablation_strictly_better"] is True
    assert observer["all_predictions_replayed_exactly"] is True
    assert observer["checkpoint_continuation_replay_exact"] is True
    assert report["H3_FRAME_EMERGENCE_RECEIPT"] is False
    assert report["BW_KMS_CLOCK_RECEIPT"] is False


def test_support_binding_corruption_cannot_survive_the_limit_gate() -> None:
    federation = reference_incidence_nerve_federation()
    binding = federation.support_tower_binding
    assert binding is not None
    rows = list(binding.carrier_to_base_vertex)
    rows[0] = (rows[0][0], rows[1][1])
    corrupted = replace(
        federation,
        support_tower_binding=replace(
            binding, carrier_to_base_vertex=tuple(rows)
        ),
    )

    report = controlled_oriented_s2_limit_report(corrupted)
    assert report["carrier_to_base_vertex_bijection"] is False
    assert report["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"] is False


def test_incidence_bridge_replays_deterministically() -> None:
    first = incidence_nerve_bridge_report()
    second = incidence_nerve_bridge_report()

    assert first["federation_bundle_sha256"] == second["federation_bundle_sha256"]
    assert first["phase_to_repair"]["terminal_state_hash"] == (
        second["phase_to_repair"]["terminal_state_hash"]
    )
    assert first["operational_observer"]["record_commit_sha256"] == (
        second["operational_observer"]["record_commit_sha256"]
    )
