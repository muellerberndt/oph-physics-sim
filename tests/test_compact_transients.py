from __future__ import annotations

import json

from oph_fpe.compact_transients import (
    CompactHistory,
    DetectionThinner,
    GenealogyDAG,
    MarkedCatalogProcess,
    PromotionGate,
    compact_transient_audit_report,
    generation_prior_leakage_audit,
    generation_prior_score,
    heldout_gain,
    host_mixture_identifiability,
    linear_repair_tail_template,
    repair_reload_waiting_time_shift,
    simulator_accuracy_receipt,
    write_compact_transient_audit_report,
)


def test_compact_transient_default_is_cr2_not_physical_promotion(tmp_path):
    report = compact_transient_audit_report()

    assert report["claim"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert report["promotion_allowed"] is False
    assert report["first_blocked_gate"] == "CONTROLS"
    assert report["readiness_gates"]["COMPACT_SOURCE_LAW_RECEIPT"] is True
    assert report["readiness_gates"]["CONTROL_MODEL_RECEIPT"] is False
    assert report["receipt_payloads"]["CONTROL_MODEL_RECEIPT"]["status"] == "FAIL"
    assert report["receipt_payloads"]["SIMULATOR_ACCURACY_RECEIPT"]["status"] == "OPEN_GATE"
    expected_receipts = {
        "COMPACT_HISTORY_RECEIPT",
        "COMPACT_QUOTIENT_RECEIPT",
        "COMPACT_SOURCE_LAW_RECEIPT",
        "PACKETIZED_KERNEL_RECEIPT",
        "PHYSICAL_CLOCK_RECEIPT",
        "FINITE_PACKET_PARENT_RECEIPT",
        "PACKET_CONSERVATION_RECEIPT",
        "PROPAGATION_RECEIPT",
        "DETECTION_THINNING_RECEIPT",
        "CENSORING_AND_UPPER_LIMIT_RECEIPT",
        "POINT_PROCESS_LIKELIHOOD_RECEIPT",
        "REPEATER_HISTORY_LIKELIHOOD_RECEIPT",
        "FRB_SOURCE_IDENTITY_RECEIPT",
        "FRB_CADENCE_EXPOSURE_RECEIPT",
        "BH_GENEALOGY_DAG_RECEIPT",
        "NO_GENERATION_LEAKAGE_RECEIPT",
        "CONTROL_MODEL_RECEIPT",
        "REFINEMENT_STABILITY_RECEIPT",
        "SIMULATOR_ACCURACY_RECEIPT",
        "FROZEN_HASHES_RECEIPT",
        "HELDOUT_LIKELIHOOD_RECEIPT",
        "PROMOTION_AUDIT_RECEIPT",
    }
    assert expected_receipts <= set(report["receipt_payloads"])
    assert report["implementation_targets"]["frb_control_family"]["M2"] == "young_plus_old_gc_repair_reload_timing"

    written = write_compact_transient_audit_report(tmp_path)
    payload = json.loads((tmp_path / "compact_transient_audit_report.json").read_text(encoding="utf-8"))
    assert written["claim"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert payload["promotion_audit"]["allowed_claim_label"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert (tmp_path / "compact_transient_audit_report.md").exists()


def test_compact_transient_can_promote_to_cr3_but_not_cr4():
    report = compact_transient_audit_report(
        {
            "CONTROL_MODEL_RECEIPT": True,
            "REFINEMENT_STABILITY_RECEIPT": True,
            "FROZEN_HASHES_RECEIPT": True,
        }
    )

    assert report["claim"] == "CR3_FROZEN_PHYSICAL_PREDICTION"
    assert report["promotion_allowed"] is True
    assert report["first_blocked_gate"] == "COMPACT_SOURCE_ACTION_DERIVED_RECEIPT"


def test_compact_history_and_detection_helpers():
    history = CompactHistory("frb-old-host")
    history.append_step(
        "q0",
        "q1",
        {"fluence": 2.0},
        {"conservation_residual": 0.125},
        3.0,
    )

    visible = history.visible_history()
    assert visible["step_count"] == 1
    assert visible["conservation_residual_abs_sum"] == 0.125

    thinner = DetectionThinner()
    assert thinner.detection_probability({"fluence": 2.0}, {"fluence_threshold": 4.0, "duty_cycle": 0.5}) == 0.25


def test_marked_catalog_process_and_heldout_gain():
    process = MarkedCatalogProcess()
    loglike = process.loglike(
        [{"weight": 2.0}, {"weight": 1.0}],
        {"duration": 10.0, "exposure_fraction": 0.5},
        {"base_intensity": 0.2},
    )
    gain = heldout_gain(loglike, [loglike - 2.0, loglike - 0.5], delta_min=0.25)

    assert gain["CONTROL_MODEL_RECEIPT"] is True
    assert abs(gain["heldout_gain"] - 0.5) < 1.0e-12


def test_frb_host_rank_and_repair_reload_helpers():
    rank = host_mixture_identifiability(
        [
            {"SFR": 1.0, "M_star_old": 0.0, "M_GC": 0.0, "exposure": 1.0},
            {"SFR": 0.0, "M_star_old": 1.0, "M_GC": 0.0, "exposure": 1.0},
            {"SFR": 0.0, "M_star_old": 0.0, "M_GC": 1.0, "exposure": 1.0},
        ]
    )
    assert rank["HOST_MIXTURE_IDENTIFIABILITY_RECEIPT"] is True

    collinear = host_mixture_identifiability(
        [
            {"SFR": 1.0, "M_star_old": 2.0, "M_GC": 3.0, "exposure": 1.0},
            {"SFR": 2.0, "M_star_old": 4.0, "M_GC": 6.0, "exposure": 1.0},
        ]
    )
    assert collinear["HOST_MIXTURE_IDENTIFIABILITY_RECEIPT"] is False

    short = repair_reload_waiting_time_shift(
        1.0,
        reservoir_before=4.0,
        fluence_to_discharge=0.5,
        reload_rate=0.25,
        threshold=5.0,
    )
    long = repair_reload_waiting_time_shift(
        3.0,
        reservoir_before=4.0,
        fluence_to_discharge=0.5,
        reload_rate=0.25,
        threshold=5.0,
    )
    assert long["waiting_time_to_threshold"] > short["waiting_time_to_threshold"]


def test_bh_recycling_generation_prior_and_tail_helpers():
    audit = generation_prior_leakage_audit({"M1": 80.0, "ringdown_residual": 0.1})
    assert audit["NO_GENERATION_LEAKAGE_RECEIPT"] is False
    assert "ringdown_residual" in audit["forbidden_hits"]

    prior = generation_prior_score({"M1": 60.0, "M2": 55.0, "chi1": 0.8, "chi2": 0.6, "q": 0.8})
    assert prior["NO_GENERATION_LEAKAGE_RECEIPT"] is True
    assert 0.0 < prior["p_generation_ge_2"] < 1.0

    dag = GenealogyDAG()
    dag.add_seed("bh1", generation=1)
    dag.add_seed("bh2", generation=2)
    receipt = dag.recycle("bh1", "bh2", "bh3", "gw-packet")
    assert receipt["BH_GENEALOGY_DAG_RECEIPT"] is True
    assert receipt["generation"] == 3

    tail = linear_repair_tail_template(
        [0.0, 1.0],
        generation_probability=0.5,
        recycled_mismatch=0.2,
        gamma_rep=0.1,
        omega_rep=1.0,
    )
    assert tail["LINEAR_REPAIR_TAIL_RECEIPT"] is True
    assert tail["amplitude"] == 0.1


def test_promotion_gate_and_accuracy_receipt():
    gate = PromotionGate()
    receipts = {"COMPACT_QUOTIENT_RECEIPT": True}
    assert gate.claim_tier(receipts) == "CR1_QUOTIENT_DIAGNOSTIC"
    assert gate.first_blocked_gate(receipts) == "SOURCE"

    accuracy = simulator_accuracy_receipt(
        {
            "epsilon_mu": 0.01,
            "epsilon_K": 0.02,
            "expected_path_length": 3.0,
            "epsilon_E": 0.01,
            "epsilon_prop": 0.01,
            "epsilon_detector": 0.01,
            "epsilon_canon": 0.01,
            "epsilon_clock": 0.01,
            "epsilon_mc": 0.01,
        }
    )
    assert accuracy["status"] == "PASS"
    assert abs(accuracy["tv_bound"] - 0.13) < 1.0e-12
