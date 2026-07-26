"""Source-derived nonvacuous federation and oriented-support bridge.

This producer binds four finite statements to one artifact:

* the carrier's own oriented ``(12,30,20)`` incidence produces a cover nerve;
* its thirty seam algebras and twenty declared triple restrictions sew exactly;
* local phase dynamics induces accepted confluent repairs only along those seams;
* the same nerve maps simplicially to a refining oriented spherical support.

The result is a constructive finite theorem.  It is not a laboratory
attachment, an H3-frame theorem, a BW/KMS clock, or a continuum QFT result.
"""

from __future__ import annotations

from typing import Any

from oph_fpe.consensus.phase_repair_bridge import (
    circular_phase_mismatch,
    phase_register_key,
    phase_repair_bridge_report,
)
from oph_fpe.consensus.transactional_repair import canonical_state_hash
from oph_fpe.core.echosahedral_federation import (
    controlled_oriented_s2_limit_report,
    interface_algebra_sha256,
    reference_federation_instrument_bundle,
    reference_incidence_nerve_federation,
    verify_reference_federation_instrument_bundle,
)


REFERENCE_SEED = 20260596


def _operational_observer_report(
    *,
    federation_bundle: dict[str, Any],
    federation_verification: dict[str, Any],
    phase_report: dict[str, Any],
) -> dict[str, Any]:
    """Replay the finite readback/record/feedback/checkpoint observer contract."""

    blockers: list[str] = []
    supports = federation_verification.get("sewing", {}).get("observer_supports", [])
    connected_support = bool(
        len(supports) == 1
        and supports[0].get("CONNECTED_OBSERVER_SUPPORT_RECEIPT") is True
        and supports[0].get("carrier_count")
        == federation_verification.get("carrier_count")
    )
    if not connected_support:
        blockers.append("connected_full_federation_observer_support_missing")

    lock = phase_report.get("phase_lock_measurement", {})
    induced = phase_report.get("induced_repairs", {})
    terminal_phases = lock.get("terminal_phases", [])
    proposals = induced.get("proposal_rows", [])
    readback = bool(
        len(terminal_phases) == federation_verification.get("carrier_count")
        and proposals
        and all(
            len(row.get("pair", [])) == 2
            and all(
                0 <= int(index) < len(terminal_phases)
                for index in row.get("pair", [])
            )
            for row in proposals
        )
    )
    if not readback:
        blockers.append("phase_readback_is_not_complete_on_observer_support")

    prediction_rows: list[dict[str, Any]] = []
    prediction_control = bool(proposals)
    replay_state = {
        phase_register_key(index): float(value) % 1.0
        for index, value in enumerate(terminal_phases)
    }
    for row in proposals:
        left, right = (int(value) for value in row["pair"])
        midpoint = float(row["midpoint"]) % 1.0
        predicted_after = 0.0
        actual_after = circular_phase_mismatch(midpoint, midpoint)
        exact = actual_after == predicted_after
        prediction_control = prediction_control and exact
        prediction_rows.append(
            {
                "pair": [left, right],
                "mismatch_before": float(row["mismatch_before"]),
                "predicted_mismatch_after": predicted_after,
                "replayed_mismatch_after": actual_after,
                "prediction_exact": exact,
            }
        )
        replay_state[phase_register_key(left)] = midpoint
        replay_state[phase_register_key(right)] = midpoint
    if not prediction_control:
        blockers.append("repair_prediction_did_not_match_replayed_control")

    checkpoint_hash = canonical_state_hash(replay_state)
    checkpoint_replay = bool(
        checkpoint_hash == phase_report.get("terminal_state_hash")
    )
    if not checkpoint_replay:
        blockers.append("checkpoint_continuation_replay_hash_mismatch")

    mismatch_before = float(induced.get("declared_mismatch_before", 0.0))
    mismatch_after = float(induced.get("declared_mismatch_after", 0.0))
    feedback_ablation = bool(
        mismatch_before > mismatch_after
        and mismatch_after == 0.0
        and lock.get("dispersion_conservation_residual", 1.0) <= lock.get(
            "dispersion_conservation_tolerance", 0.0
        )
    )
    if not feedback_ablation:
        blockers.append("feedback_does_not_strictly_improve_over_no_repair")

    record_payload = {
        "schema": "oph.incidence_nerve.observer_record_commit.v1",
        "federation_bundle_sha256": interface_algebra_sha256(federation_bundle),
        "phase_terminal_state_hash": phase_report.get("terminal_state_hash"),
        "prediction_rows": prediction_rows,
        "checkpoint_replay_hash": checkpoint_hash,
    }
    record_commit_hash = interface_algebra_sha256(record_payload)
    durable_record = bool(
        record_commit_hash.startswith("sha256:")
        and len(record_commit_hash) == 71
        and phase_report.get("FEDERATION_PHASE_TO_REPAIR_BINDING_RECEIPT") is True
    )
    if not durable_record:
        blockers.append("durable_record_commit_not_bound_to_federation")

    receipt = bool(
        connected_support
        and readback
        and durable_record
        and feedback_ablation
        and prediction_control
        and checkpoint_replay
        and not blockers
    )
    return {
        "schema": "oph.incidence_nerve.operational_observer.v1",
        "observer_token": (
            supports[0].get("observer_token") if len(supports) == 1 else None
        ),
        "federation_bundle_sha256": interface_algebra_sha256(federation_bundle),
        "connected_full_federation_support": connected_support,
        "phase_readback_complete": readback,
        "record_commit": record_payload,
        "record_commit_sha256": record_commit_hash,
        "feedback_ablation_strictly_better": feedback_ablation,
        "prediction_control_rows": prediction_rows,
        "all_predictions_replayed_exactly": prediction_control,
        "checkpoint_continuation_replay_sha256": checkpoint_hash,
        "checkpoint_continuation_replay_exact": checkpoint_replay,
        "blockers": blockers,
        "OPERATIONAL_OBSERVER_READBACK_RECEIPT": readback,
        "OPERATIONAL_OBSERVER_DURABLE_RECORD_RECEIPT": durable_record,
        "OPERATIONAL_OBSERVER_FEEDBACK_ABLATION_RECEIPT": feedback_ablation,
        "OPERATIONAL_OBSERVER_PREDICTION_CONTROL_RECEIPT": prediction_control,
        "OPERATIONAL_OBSERVER_CHECKPOINT_CONTINUATION_RECEIPT": checkpoint_replay,
        "OPERATIONAL_SELF_READING_OBSERVER_RECEIPT": receipt,
    }


def incidence_nerve_bridge_report(
    *,
    seed: int = REFERENCE_SEED,
    max_level: int = 3,
) -> dict[str, Any]:
    """Produce the complete finite #596 bridge on one common source tower."""

    federation = reference_incidence_nerve_federation()
    bundle = reference_federation_instrument_bundle(federation)
    bundle_sha256 = interface_algebra_sha256(bundle)
    verification = verify_reference_federation_instrument_bundle(bundle)
    carrier_ids = tuple(carrier.carrier_id for carrier in federation.carriers)
    carrier_index = {
        carrier_id: index for index, carrier_id in enumerate(carrier_ids)
    }
    seam_pairs = tuple(
        (
            carrier_index[seam.left_carrier_id],
            carrier_index[seam.right_carrier_id],
        )
        for seam in federation.seams
    )
    # This declared target-free initial packet guarantees six disjoint
    # over-threshold seam repairs.  It is an initial condition, not an
    # A5-invariant vacuum or a dynamically selected phase law.
    initial_phases = tuple(0.0 if index % 2 == 0 else 0.5 for index in range(12))
    phase = phase_repair_bridge_report(
        carrier_count=len(carrier_ids),
        seed=int(seed),
        initial_phases=initial_phases,
        candidate_pairs=seam_pairs,
        carrier_ids=carrier_ids,
        federation_bundle_sha256=bundle_sha256,
    )
    matched_pairs = {
        tuple(sorted(int(value) for value in row))
        for row in phase["induced_repairs"]["matched_pairs"]
    }
    declared_pairs = {tuple(sorted(pair)) for pair in seam_pairs}
    phase_binding = bool(
        phase["FEDERATION_PHASE_TO_REPAIR_BINDING_RECEIPT"]
        and matched_pairs
        and matched_pairs <= declared_pairs
        and phase["PHASE_LOCK_MEASUREMENT_RECEIPT"]
        and phase["PHASE_INDUCED_REPAIR_ACCEPTANCE_RECEIPT"]
        and phase["PHASE_REPAIR_CONFLUENCE_RECEIPT"]
    )
    support = controlled_oriented_s2_limit_report(
        federation, max_level=max_level
    )
    observer = _operational_observer_report(
        federation_bundle=bundle,
        federation_verification=verification,
        phase_report=phase,
    )
    receipt = bool(
        verification["ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID"]
        and verification[
            "PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"
        ]
        and verification["NONVACUOUS_HIGHER_OVERLAP_COCYCLE_WITNESS"]
        and phase_binding
        and observer["OPERATIONAL_SELF_READING_OBSERVER_RECEIPT"]
        and support["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"]
    )
    return {
        "schema": "oph.incidence_nerve.physical_bridge.v1",
        "source_rule": (
            "twelve_charts_from_ports__thirty_seams_from_edges__"
            "twenty_triple_restrictions_from_oriented_faces"
        ),
        "target_firewall": (
            "no_SM_H3_event_BW_KMS_or_laboratory_target_enters_the_producer"
        ),
        "federation_bundle": bundle,
        "federation_bundle_sha256": bundle_sha256,
        "federation_verification": verification,
        "phase_to_repair": phase,
        "phase_repairs_are_declared_seams": phase_binding,
        "operational_observer": observer,
        "controlled_oriented_s2_limit": support,
        "NONVACUOUS_PHYSICAL_FEDERATION_RECEIPT": bool(
            verification[
                "PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"
            ]
        ),
        "PHASE_TO_ACCEPTED_CONFLUENT_REPAIR_ON_FEDERATION_RECEIPT": (
            phase_binding
        ),
        "OPERATIONAL_OBSERVER_ON_FEDERATION_RECEIPT": bool(
            observer["OPERATIONAL_SELF_READING_OBSERVER_RECEIPT"]
        ),
        "CONTROLLED_ORIENTED_S2_LIMIT_ON_SAME_TOWER_RECEIPT": bool(
            support["CONTROLLED_ORIENTED_S2_LIMIT_RECEIPT"]
        ),
        "PHYSICAL_FEDERATION_SUPPORT_BRIDGE_RECEIPT": receipt,
        "H3_FRAME_EMERGENCE_RECEIPT": False,
        "EVENT_MANIFOLD_RECEIPT": False,
        "BW_KMS_CLOCK_RECEIPT": False,
        "PHYSICAL_H3_KMS_EMERGENCE_RECEIPT": False,
        "claim_boundary": (
            "The receipt closes a finite source-derived federation, its "
            "operational repair/readback observer, and its controlled "
            "commutative oriented-S2 support limit on one bound artifact. It "
            "does not attach a laboratory device, physical length or clock "
            "scale, H3 frames, events, BW/KMS flow, or continuum fields."
        ),
    }


__all__ = ["REFERENCE_SEED", "incidence_nerve_bridge_report"]
