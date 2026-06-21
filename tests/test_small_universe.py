from pathlib import Path

from tools.verify_small_universe import run_from_config


def test_small_universe_exact_consensus_and_holonomy_control(tmp_path: Path):
    out = tmp_path / "sou"
    bundle = run_from_config(Path("configs/sou_v1_icosa12.yml"), out_dir=out)

    exact = bundle["exact_consensus"]
    control = bundle["frustrated_control"]
    assert bundle["bundle_receipt"] is True
    assert exact["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert exact["state_count_exhaustively_enumerated"] == 2048
    assert exact["enabled_repair_events_checked"] == 11264
    assert exact["strict_descent_violation_count"] == 0
    assert exact["accepted_phi_increase_violation_count"] == 0
    assert exact["disjoint_commutation_violation_count"] == 0
    assert exact["local_diamond_or_schedule_confluence_violation_count"] == 0
    assert exact["repair_completeness_violation_count"] == 0
    assert exact["global_consistent_state_count"] == 1
    assert exact["unique_terminal_normal_form_count"] == 1
    assert exact["terminal_phi"] == 0
    assert control["HOLONOMY_OBSTRUCTION_RECEIPT"] is True
    assert control["global_consistent_state_count"] == 0
    assert control["nonzero_holonomy_cycle_count"] > 0
    assert control["terminal_phi"] > 0


def test_small_universe_writes_complete_evidence_bundle(tmp_path: Path):
    out = tmp_path / "sou"
    run_from_config(Path("configs/sou_v1_icosa12.yml"), out_dir=out)

    expected = {
        "MANIFEST.json",
        "config.yml",
        "source_hashes.json",
        "all_states.jsonl",
        "repair_transition_table.jsonl",
        "schedule_replays.jsonl",
        "cycle_holonomy.json",
        "exact_consensus_receipt.json",
        "frustrated_control_receipt.json",
        "small_oph_universe_evidence.json",
        "NON_CLAIMS.md",
        "SHA256SUMS",
        "SHA256.txt",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert sum(1 for _ in (out / "all_states.jsonl").open(encoding="utf-8")) == 4096
    assert sum(1 for _ in (out / "repair_transition_table.jsonl").open(encoding="utf-8")) == 22528
    assert sum(1 for _ in (out / "schedule_replays.jsonl").open(encoding="utf-8")) == 65536
    non_claims = (out / "NON_CLAIMS.md").read_text(encoding="utf-8")
    assert "physical CMB" in non_claims
    assert "inferred BW/KMS 2pi" in non_claims


def test_small_universe_verifier_does_not_import_repair_generator():
    source = Path("tools/verify_small_universe.py").read_text(encoding="utf-8")
    assert "oph_fpe.small_universe.oph_repair import" not in source
