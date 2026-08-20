"""Tests of the observer-frame quantum statistics probe (lane D3).

Exploratory, non-evidential scope. Positive receipts: count-ratio
identities on all eight committed contexts, record persistence,
projection statistics against the Lueders update, the exact interference
gap, export round-trip, receipt determinism. Mutation guards: a dropped
micro-configuration breaks the count-ratio identity, a counts-from-
amplitudes mutant is detected by the import-graph check, a tampered
branch-table entry is detected by the trace verification, a classical
reveal-only mock fails the interference receipt, and conditioning on a
never-realized outcome fails closed.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import pytest

from oph_fpe.core.charged_response import canonical_sha256
from oph_fpe.qm_observer import amplitudes, ensemble, receipt, tables
from oph_fpe.qm_observer.ensemble import QMObserverError
from oph_fpe.quantum import phase_operation as po


EXPECTED_BASE = (
    ("web_diagonal", (111, 68), 179, Fraction(111, 179)),
    ("web_conjugated_0", (111, 68), 179, Fraction(111, 179)),
    ("web_conjugated_1", (111, 68), 179, Fraction(111, 179)),
    ("web_conjugated_2", (315, 401), 716, Fraction(315, 716)),
    ("web_conjugated_3", (315, 401), 716, Fraction(315, 716)),
    ("web_conjugated_4", (315, 401), 716, Fraction(315, 716)),
    ("web_conjugated_5", (315, 401), 716, Fraction(315, 716)),
    ("phase", (179, 179), 358, Fraction(1, 2)),
)

cross_repo = pytest.mark.skipif(
    not po.DEFAULT_REFERENCE_RECEIPT.is_file(),
    reason="the committed reference receipt is not checked out",
)


def test_base_count_ratio_identities() -> None:
    for name, counts, mass, weight in EXPECTED_BASE:
        row = receipt.base_context_identity(name)
        assert row["counts"] == list(counts)
        assert row["mass"] == mass
        pair = [weight.numerator, weight.denominator]
        assert row["count_ratio"] == pair
        assert row["amplitude_weight"] == pair
        assert row["committed_weight"] == pair


@cross_repo
def test_reference_cross_check_replays() -> None:
    result = receipt.reference_cross_check()
    assert result["available"] is True
    assert result["contexts_replayed"] == 8


def test_branch_tables_verified_by_trace() -> None:
    report = receipt.verify_tables()
    assert report["entries_checked"] == 32


def test_mutated_table_entry_detected(monkeypatch) -> None:
    tampered = copy.deepcopy(tables.BRANCH_TABLES)
    tampered["rec0"]["E_A"] = (1, 3)
    monkeypatch.setattr(tables, "BRANCH_TABLES", tampered)
    with pytest.raises(QMObserverError, match="TABLE_TRACE"):
        receipt.verify_tables()


def test_dropped_micro_configuration_breaks_identity() -> None:
    base = ensemble.base_ensemble()
    mutant = ensemble.Ensemble(configs=base.configs[1:])
    with pytest.raises(QMObserverError, match="COUNT_RATIO"):
        receipt.base_context_identity("web_diagonal", ens=mutant)
    with pytest.raises(QMObserverError, match="COUNT_RATIO"):
        receipt.base_context_identity("web_conjugated_3", ens=mutant)


def test_import_graph_independence() -> None:
    report = receipt.check_independence()
    assert set(report["modules"]) == {
        "tables.py",
        "ensemble.py",
        "amplitudes.py",
    }
    ensemble_imports = report["modules"]["ensemble.py"]["imports"]
    assert not any("amplitudes" in name for name in ensemble_imports)
    assert not any("quantum" in name for name in ensemble_imports)
    amplitude_imports = report["modules"]["amplitudes.py"]["imports"]
    assert not any("ensemble" in name for name in amplitude_imports)
    assert not any("tables" in name for name in amplitude_imports)


def test_counts_from_amplitudes_mutant_detected() -> None:
    with pytest.raises(QMObserverError, match="IMPORT_INDEPENDENCE"):
        receipt.check_source_independence(
            "ensemble.py", "from oph_fpe.qm_observer import amplitudes\n"
        )
    with pytest.raises(QMObserverError, match="IMPORT_INDEPENDENCE"):
        receipt.check_source_independence(
            "ensemble.py", "from oph_fpe.quantum import phase_operation\n"
        )
    with pytest.raises(QMObserverError, match="IMPORT_INDEPENDENCE"):
        receipt.check_source_independence(
            "amplitudes.py", "from oph_fpe.qm_observer import ensemble\n"
        )


def test_record_persistence_all_chains() -> None:
    for name, _ in tables.CONTEXTS:
        chain = receipt.collapse_chain(name)
        assert len(chain["conditionings"]) == 2
        for conditioning in chain["conditionings"]:
            assert conditioning["repeat"]["fraction"] == [1, 1]


def test_projection_statistics_exact() -> None:
    chain = receipt.collapse_chain("web_conjugated_3")
    first, second = chain["conditionings"]
    assert first["conditioned_class"] == "A0"
    probes = {p["probe"]: p for p in first["probes"]}
    assert probes["web_diagonal"]["count_ratio"] == [1, 4]
    assert probes["web_conjugated_3"]["count_ratio"] == [1, 1]
    assert probes["phase"]["count_ratio"] == [1, 2]
    assert second["conditioned_class"] == "A1"
    probes = {p["probe"]: p for p in second["probes"]}
    assert probes["web_diagonal"]["count_ratio"] == [3, 4]
    assert probes["web_conjugated_3"]["count_ratio"] == [0, 1]
    assert probes["phase"]["count_ratio"] == [1, 2]
    phase_chain = receipt.collapse_chain("phase")
    y_zero = phase_chain["conditionings"][0]
    assert y_zero["conditioned_class"] == "Y0"
    probes = {p["probe"]: p for p in y_zero["probes"]}
    assert probes["web_diagonal"]["count_ratio"] == [1, 2]
    assert probes["web_conjugated_3"]["count_ratio"] == [1, 2]
    assert probes["phase"]["count_ratio"] == [1, 1]
    for chain_name, _ in tables.CONTEXTS:
        chain = receipt.collapse_chain(chain_name)
        for conditioning in chain["conditionings"]:
            for probe in conditioning["probes"]:
                assert probe["count_ratio"] == probe["lueders_weight"]


def test_conditioning_on_never_realized_outcome_fails_closed() -> None:
    applied = ensemble.apply_context(
        ensemble.base_ensemble(), "web_diagonal"
    )
    sub = ensemble.condition(applied, "web_diagonal", 0)
    repeat = ensemble.apply_context(sub, "web_diagonal")
    with pytest.raises(QMObserverError, match="CONDITION_NULL"):
        ensemble.condition(repeat, "web_diagonal", 1)


def test_conditioning_context_mismatch_fails_closed() -> None:
    applied = ensemble.apply_context(
        ensemble.base_ensemble(), "web_diagonal"
    )
    with pytest.raises(QMObserverError, match="CONDITION_CONTEXT"):
        ensemble.condition(applied, "phase", 0)


def test_lueders_weight_zero_fails_closed() -> None:
    post = amplitudes.lueders(
        amplitudes.base_state(), "web_conjugated_3", 1
    )
    with pytest.raises(po.PhaseOperationError, match="LUEDERS_NULL"):
        amplitudes.lueders(post, "web_conjugated_3", 0)


def test_interference_gap_exact() -> None:
    report = receipt.interference_receipt()
    assert report["direct"] == {"counts": [111, 68], "mass": 179}
    assert report["first_measurement"] == {
        "counts": [315, 401],
        "mass": 716,
    }
    assert report["mediated"] == {"counts": [1518, 1346], "mass": 2864}
    assert report["common_mass"] == 2864
    assert report["direct_at_common_mass"] == [1776, 1088]
    assert report["mediated_at_common_mass"] == [1518, 1346]
    assert report["gap"] == [129, 1432]
    branch_counts = [branch["counts"] for branch in report["branches"]]
    assert branch_counts == [[315, 945], [1203, 401]]


def test_classical_mixture_mock_fails_interference() -> None:
    joint = {(0, 0): 30, (0, 1): 81, (1, 0): 51, (1, 1): 17}
    total = sum(joint.values())
    direct_zero = sum(
        mass for (rec, _), mass in joint.items() if rec == 0
    )
    sewn_zero = 0
    for rot_value in (0, 1):
        sub = {
            pair: mass for pair, mass in joint.items() if pair[1] == rot_value
        }
        sewn_zero += sum(mass for (rec, _), mass in sub.items() if rec == 0)
    assert sewn_zero == direct_zero
    gap = Fraction(direct_zero, total) - Fraction(sewn_zero, total)
    assert gap == 0
    with pytest.raises(QMObserverError, match="INTERFERENCE_NULL"):
        receipt.require_interference(gap)


def test_export_round_trip_exact() -> None:
    export = receipt.build_export()
    text = json.dumps(export, indent=2, sort_keys=True)
    parsed = json.loads(text)
    assert parsed == export
    scenario = next(
        s
        for s in parsed["scenarios"]
        if s["scenario_id"] == "base_context/web_conjugated_3"
    )
    node = scenario["tree"]["children"][0]
    assert node["counts"] == [315, 401]
    assert [Fraction(*pair) for pair in node["weights"]] == [
        Fraction(315, 716),
        Fraction(401, 716),
    ]
    interference = parsed["scenarios"][-1]
    assert interference["kind"] == "interference"
    assert Fraction(*interference["comparison"]["gap"]) == Fraction(129, 1432)
    assert interference["collapse_events"][0]["after"]["class"] == "A0"


def _assert_floats_confined(value, path=()) -> None:
    if isinstance(value, float):
        assert "display" in path, f"float outside display block at {path}"
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_floats_confined(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_floats_confined(item, path + (index,))


def test_floats_confined_to_display_blocks() -> None:
    _assert_floats_confined(receipt.build_export())
    body = receipt.build_receipt()
    _assert_floats_confined({"receipt": body})


def test_receipt_deterministic_and_hashed() -> None:
    first = receipt.build_receipt()
    second = receipt.build_receipt()
    assert first == second
    body = {
        key: value
        for key, value in first.items()
        if key != "receipt_sha256"
    }
    assert first["receipt_sha256"] == canonical_sha256(body)
    assert first["export"]["body_sha256"] == canonical_sha256(
        receipt.build_export()
    )


def test_receipt_labels_window_and_boundary() -> None:
    body = receipt.build_receipt()
    assert body["labels"]["exploratory"] is True
    assert body["labels"]["evidential"] is False
    assert "declared architecture operation" in body["boundary"]
    assert "does not derive the Hilbert-space structure" in body["boundary"]
    assert "no probability postulate" in body["semantics"].lower()
    assert body["phase_receipt_window"]["counts"] == [179, 179]
    assert body["phase_receipt_window"]["lhs"] == 0
    assert body["phase_receipt_window"]["rhs"] == 3869527488
    assert len(body["count_ratio_identities"]) == 8
    assert len(body["collapse_chains"]) == 8
    export = receipt.build_export()
    assert export["labels"] == body["labels"]
    assert len(export["scenarios"]) == 17
