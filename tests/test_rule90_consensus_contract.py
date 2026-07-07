from __future__ import annotations

from oph_fpe.consensus import (
    RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT,
    rule90_boundary_bad,
    rule90_boundary_good,
    rule90_consistent,
    rule90_gauge_equiv,
    rule90_gauge_nontrivial_witness,
    rule90_hfib_failure_witness,
    rule90_hfib_holds,
    rule90_lean_consensus_fixture_report,
    rule90_no_frustration_free_local_repair_witness,
    rule90_obs_map,
    rule90_phi,
    rule90_records,
    rule90t,
)
from oph_fpe.cli import main


def test_rule90_carrier_phi_zero_matches_edge_consistency() -> None:
    assert rule90t((0, 0, 0)) == (0, 0, 0)
    assert rule90t((0, 1, 0)) == (1, 0, 1)
    assert rule90t((1, 0, 1)) == (0, 0, 0)
    assert len(rule90_records()) == 64

    for record in rule90_records():
        assert (rule90_phi(record) == 0) == rule90_consistent(record)
        assert (rule90_phi(record) == 0) == (rule90_obs_map(record)[0] == rule90_obs_map(record)[1])


def test_rule90_hfib_good_boundary_holds_exhaustively() -> None:
    assert rule90_hfib_holds(rule90_boundary_good) is True
    assert rule90_hfib_failure_witness(rule90_boundary_good) is None


def test_rule90_hfib_bad_boundary_fails_with_witness() -> None:
    assert rule90_hfib_holds(rule90_boundary_bad) is False
    witness = rule90_hfib_failure_witness(rule90_boundary_bad)

    assert witness is not None
    assert witness["boundary"] == [0, 0]
    assert witness["left"]["bottom"] == [0, 0, 0]
    assert witness["right"]["bottom"] == [0, 1, 0]
    left = (tuple(witness["left"]["seed"]), tuple(witness["left"]["bottom"]))
    right = (tuple(witness["right"]["seed"]), tuple(witness["right"]["bottom"]))
    assert rule90_consistent(left) is True
    assert rule90_consistent(right) is True
    assert rule90_gauge_equiv(left, right) is False


def test_rule90_gauge_nontrivial_witness_has_different_seeds_same_observable() -> None:
    witness = rule90_gauge_nontrivial_witness()

    assert witness is not None
    assert witness["leftSeedDiffersFromRightSeed"] is True
    left = (tuple(witness["left"]["seed"]), tuple(witness["left"]["bottom"]))
    right = (tuple(witness["right"]["seed"]), tuple(witness["right"]["bottom"]))
    assert left[0] != right[0]
    assert rule90_consistent(left) is True
    assert rule90_consistent(right) is True
    assert rule90_gauge_equiv(left, right) is True


def test_rule90_no_frustration_free_local_repair_witness() -> None:
    witness = rule90_no_frustration_free_local_repair_witness()

    assert witness["receipt"] is True
    assert witness["outOfImageBottom"] == [0, 0, 1]
    assert witness["rule90PreimageCount"] == 0
    assert witness["edgeBrokenForEverySeed"] is True
    assert witness["h2ForcesSeedPatchMove"] is True
    assert witness["h1PinsBottomRow"] is True
    assert witness["h3RequiresRule90Preimage"] is True


def test_rule90_lean_consensus_fixture_report_is_fail_closed_receipt() -> None:
    report = rule90_lean_consensus_fixture_report()

    assert report[RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT] is True
    assert report["receipt"] is True
    assert report["physical_claim"] is False
    assert report["claim_level"] == "demo"
    assert report["recordCount"] == 64
    assert report["consistentRecordCount"] == 8
    assert report["phiZeroIffEdgeConsistencyReceipt"] is True
    assert report["rule90GoodBoundaryHfibReceipt"] is True
    assert report["rule90BadBoundaryFailureReceipt"] is True
    assert report["rule90GaugeNontrivialReceipt"] is True
    assert report["rule90NoFrustrationFreeLocalRepairReceipt"] is True
    assert report["blockers"] == []
    assert "rule90_Hfib_good" in report["leanSource"]["theorems"]


def test_rule90_consensus_fixture_cli_writes_report(tmp_path) -> None:
    out = tmp_path / "rule90_consensus_fixture_report.json"

    assert main(["rule90-consensus-fixture", "--out", str(out)]) == 0
    assert out.exists()
    assert RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT in out.read_text(encoding="utf-8")
