from oph_fpe.dynamics.consensus_certificate import finite_consensus_theorem_certificate


def test_finite_consensus_certificate_fails_closed_without_theorem_replay():
    report = finite_consensus_theorem_certificate(
        [
            {"cycle": 0, "phase": "exploration", "phi_before": 3, "phi": 1, "delta_phi": -2},
            {"cycle": 1, "phase": "exploration", "phi_before": 1, "phi": 0, "delta_phi": -1},
        ]
    )

    assert report["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert report["finite_consensus_theorem_receipt"] is False
    assert "theorem_phase_repair_events" in report["missing_evidence"]
    assert "schedule_replay_count" in report["missing_evidence"]


def test_finite_consensus_certificate_accepts_strict_theorem_evidence():
    report = finite_consensus_theorem_certificate(
        [
            {
                "cycle": 0,
                "phase": "theorem",
                "node": 4,
                "accepted": True,
                "delta_touched_phi": -1.0,
                "delta_global_phi": -1.0,
            },
            {
                "cycle": 1,
                "phase": "theorem",
                "node": 7,
                "accepted": False,
                "delta_touched_phi": 0.0,
                "delta_global_phi": 0.0,
            },
        ],
        evidence={
            "disjoint_commutation_violation_count": 0,
            "local_diamond_violation_count": 0,
            "repair_completeness_violation_count": 0,
            "unique_terminal_quotient_hash_count": 1,
            "schedule_replay_count": 16,
            "requested_schedule_replays": 16,
        },
    )

    assert report["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert report["strict_descent_violation_count"] == 0
    assert report["missing_evidence"] == []


def test_finite_consensus_certificate_accepts_counted_replay_evidence_without_full_trace():
    report = finite_consensus_theorem_certificate(
        [],
        evidence={
            "theorem_phase_event_count": 12,
            "accepted_theorem_move_count": 12,
            "strict_descent_violation_count": 0,
            "accepted_phi_increase_violation_count": 0,
            "disjoint_commutation_violation_count": 0,
            "local_diamond_violation_count": 0,
            "repair_completeness_violation_count": 0,
            "unique_terminal_quotient_hash_count": 1,
            "schedule_replay_count": 16,
            "requested_schedule_replays": 16,
        },
    )

    assert report["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert report["theorem_phase_event_count"] == 12
    assert report["missing_evidence"] == []


def test_finite_consensus_certificate_rejects_equal_score_theorem_acceptance():
    report = finite_consensus_theorem_certificate(
        [
            {
                "cycle": 0,
                "phase": "theorem",
                "node": 4,
                "accepted": True,
                "delta_touched_phi": 0.0,
                "delta_global_phi": 0.0,
            }
        ],
        evidence={
            "disjoint_commutation_violation_count": 0,
            "local_diamond_violation_count": 0,
            "repair_completeness_violation_count": 0,
            "unique_terminal_quotient_hash_count": 1,
            "schedule_replay_count": 16,
            "requested_schedule_replays": 16,
        },
    )

    assert report["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert report["strict_descent_violation_count"] == 1
