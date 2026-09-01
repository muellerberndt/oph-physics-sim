"""Fail-closed tests for the source-causal history extension family."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.bulk import source_causal_history_family as family_module
from oph_fpe.bulk import verify_source_causal_history_family_independent as verifier_module
from oph_fpe.bulk.source_causal_history_family import (
    DEFAULT_OUTPUT,
    DEFAULT_PUBLICATION_OUTPUT,
    SourceCausalHistoryFamilyError,
    _canonical_bytes,
    _sha,
    produce_source_causal_history_family_report,
    publication_projection,
)
from oph_fpe.bulk.verify_source_causal_history_family_independent import (
    IndependentHistoryFamilyVerificationError,
    _summary_line,
    verify_publication_projection,
    verify_receipt,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return produce_source_causal_history_family_report()


def test_complete_round_prefixes_are_exact_induced_orders(report: dict) -> None:
    assert report["INFORMATIONAL_HISTORY_EXTENSION_FAMILY_RECEIPT"] is True
    assert report["INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT"] is True
    assert report["INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT"] is True
    assert report["all_cutoffs_independently_generated"] is True
    assert report["all_induced_order_embeddings_certified"] is True
    assert [row["complete_round_cutoff"] for row in report["levels"]] == [
        4,
        8,
        16,
        32,
        64,
    ]
    assert [row["event_count"] for row in report["levels"]] == [
        24,
        48,
        96,
        192,
        384,
    ]
    assert [row["direct_edge_count"] for row in report["levels"]] == [
        38,
        78,
        158,
        318,
        638,
    ]
    assert all(
        row["proper_carrier_inclusion"]
        and row["direct_order_is_induced_restriction"]
        and row["transitive_order_is_induced_restriction"]
        for row in report["induced_order_embeddings"]
    )
    binding = report["canonical_four_round_source_binding"]
    assert binding["four_round_semantic_events_byte_identical"] is True
    assert binding["four_round_direct_order_byte_identical"] is True
    evidence = report["cutoff_run_evidence"]
    assert [row["complete_round_cutoff"] for row in evidence] == [4, 8, 16, 32, 64]
    assert len(
        {row["source_capture_binding"]["capture_sha256"] for row in evidence}
    ) == 5
    assert all(
        level["generated_from_own_cutoff_capture"] is True
        and level["independent_cutoff_run_evidence_sha256"]
        == row["cutoff_run_evidence_sha256"]
        for level, row in zip(report["levels"], evidence, strict=True)
    )


def test_family_is_exactly_fixed_width_and_becomes_more_ordered(report: dict) -> None:
    scaling = report["scaling_diagnostic"]
    assert scaling["widths"] == [2, 2, 2, 2, 2]
    assert scaling["heights"] == [12, 24, 48, 96, 192]
    assert scaling["width_constant_at_observer_count"] is True
    assert scaling["height_strictly_increases"] is True
    assert scaling["ordering_fraction_strictly_increases"] is True
    assert scaling["ordering_fractions"][0] == pytest.approx(
        0.8840579710144928
    )
    assert scaling["ordering_fractions"][-1] == pytest.approx(
        0.992221496953873
    )
    assert all(row["exact_width_certificate"] for row in report["levels"])
    assert all(
        row["observer_chain_cover"]["all_chain_successors_comparable"]
        and row["matching_antichain"]["pairwise_incomparable"]
        for row in report["levels"]
    )


def test_prescribed_shared_frame_rank3_ansatz_fails_without_a_no_go_claim(
    report: dict,
) -> None:
    cone = report["prescribed_single_frame_source_port_placement"]
    assert cone["event_count"] == 24
    assert cone["comparable_pair_count"] == 244
    assert cone["incomparable_pair_count"] == 32
    assert cone["adjustable_parameter_count"] == 1
    assert cone["causal_lower_time_scale_bound"] == pytest.approx(
        1.129775730952839
    )
    assert cone["spacelike_upper_time_scale_bound"] == 0.0
    assert cone["coincident_same_rank_incomparable_pair_count"] == 8
    assert cone["incomparable_zero_spatial_separation_pair_count"] == 8
    assert cone["source_sequence_time_orientation_compatible"] is True
    assert cone["all_precedence_pairs_future_causal_at_lower_bound"] is True
    assert cone["all_incomparable_pairs_spacelike_at_lower_bound"] is False
    assert cone["injective_four_coordinate_map"] is False
    assert cone["global_time_scale_interval_nonempty"] is False
    assert cone["precedence_iff_future_causal"] is False
    assert cone["FINITE_FAITHFUL_RANK3_CONE_PLACEMENT_RECEIPT"] is False
    assert cone["inter_carrier_frame_gluing_source_derived"] is False
    assert cone["consumed_record_barycentre_rule_source_derived"] is False
    assert cone["other_source_selected_placements_excluded"] is False
    assert cone["physical_no_go_for_other_source_selected_placements"] is False
    assert "not a no-go" in cone["interpretation"]


def test_producer_executes_every_cutoff_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_producer = family_module.produce_source_derived_causal_order_report

    def recording_producer(config: dict) -> dict:
        calls.append(int(config["observer_samples"]))
        return real_producer(config)

    monkeypatch.setattr(
        family_module,
        "produce_source_derived_causal_order_report",
        recording_producer,
    )
    rebuilt = family_module.produce_source_causal_history_family_report()
    assert calls == [4, 8, 16, 32, 64]
    assert rebuilt["all_cutoffs_independently_generated"] is True


def test_no_spacetime_refinement_or_manifold_promotion(report: dict) -> None:
    for key in (
        "PHYSICAL_CAUSAL_ATTACHMENT_RECEIPT",
        "SOURCE_SELECTED_SPACETIME_REFINEMENT_FAMILY_RECEIPT",
        "CAUSET_FAITHFUL_EMBEDDING_RECEIPT",
        "CAUSET_MANIFOLDLIKE_REFINEMENT_RECEIPT",
        "CAUSET_DIMENSION_3P1_RECEIPT",
        "CAUSET_COUNT_VOLUME_DENSITY_RECEIPT",
        "SOURCE_LORENTZ_CONE_COMPATIBILITY_RECEIPT",
        "SOURCE_CAUSAL_STABLE_TIME_FUNCTION_RECEIPT",
        "PHYSICAL_SOURCE_CAUSAL_REFINEMENT_COMPATIBILITY_RECEIPT",
        "EVENT_TOPOLOGY_ATLAS_LIMIT_RECEIPT",
        "SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT",
    ):
        assert report[key] is False
    assert report["physical_promotion_allowed"] is False
    assert report["controls_fail_closed"] is True
    assert all(report["negative_controls"].values())


def test_report_is_deterministic_and_matches_frozen_bytes(report: dict) -> None:
    assert _canonical_bytes(produce_source_causal_history_family_report()) == (
        _canonical_bytes(report)
    )
    assert DEFAULT_OUTPUT.read_bytes() == _canonical_bytes(report)


def test_compact_publication_projection_is_exact_and_independently_bound(
    tmp_path: Path, report: dict
) -> None:
    projection = publication_projection(report)
    assert DEFAULT_PUBLICATION_OUTPUT.read_bytes() == _canonical_bytes(projection)
    assert len(DEFAULT_PUBLICATION_OUTPUT.read_bytes()) < 20_000
    assert projection["levels"][-1]["ordering_fraction"] == pytest.approx(
        0.992221496953873
    )
    assert projection[
        "INFORMATIONAL_INDUCED_PREFIX_REFINEMENT_RECEIPT"
    ] is True
    assert projection[
        "INFORMATIONAL_INDEPENDENT_CUTOFF_GENERATION_RECEIPT"
    ] is True
    assert projection["all_cutoffs_independently_generated"] is True
    assert len(projection["cutoff_run_evidence_sha256s"]) == 5
    assert projection["promotion_and_nonclaim_flags"][
        "PHYSICAL_SOURCE_CAUSAL_REFINEMENT_COMPATIBILITY_RECEIPT"
    ] is False
    result = verify_publication_projection(DEFAULT_OUTPUT, DEFAULT_PUBLICATION_OUTPUT)
    assert result["verified"] is True
    assert result["projection_bytes"] < 20_000

    mutated = json.loads(DEFAULT_PUBLICATION_OUTPUT.read_text(encoding="ascii"))
    mutated["levels"][-1]["ordering_fraction"] = 0.5
    body = {key: value for key, value in mutated.items() if key != "projection_sha256"}
    mutated["projection_sha256"] = _sha(body)
    path = tmp_path / "mutated-publication.json"
    path.write_bytes(_canonical_bytes(mutated))
    with pytest.raises(IndependentHistoryFamilyVerificationError):
        verify_publication_projection(DEFAULT_OUTPUT, path)


def test_independent_verifier_replays_and_rejects_false_promotion(
    tmp_path: Path, report: dict
) -> None:
    result = verify_receipt(DEFAULT_OUTPUT)
    assert result["verified"] is True
    assert result["event_counts"] == [24, 48, 96, 192, 384]
    assert result["widths"] == [2, 2, 2, 2, 2]
    assert "spacetime_refinement=False" in _summary_line(result)

    candidate = json.loads(_canonical_bytes(report).decode("ascii"))
    candidate["levels"][-1]["width"] = 3
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    candidate["report_sha256"] = _sha(body)
    mutated = tmp_path / "mutated-width.json"
    mutated.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(IndependentHistoryFamilyVerificationError):
        verify_receipt(mutated)

    promoted = json.loads(_canonical_bytes(report).decode("ascii"))
    promoted["SOURCE_DERIVED_CAUSAL_3P1_MANIFOLD_LIMIT_RECEIPT"] = True
    promoted["prescribed_single_frame_source_port_placement"][
        "FINITE_FAITHFUL_RANK3_CONE_PLACEMENT_RECEIPT"
    ] = True
    body = {key: value for key, value in promoted.items() if key != "report_sha256"}
    promoted["report_sha256"] = _sha(body)
    promoted_path = tmp_path / "false-promotion.json"
    promoted_path.write_bytes(_canonical_bytes(promoted))
    with pytest.raises(IndependentHistoryFamilyVerificationError):
        verify_receipt(promoted_path)


def test_independent_verifier_rejects_maximum_log_substitution(
    tmp_path: Path, report: dict
) -> None:
    candidate = json.loads(_canonical_bytes(report).decode("ascii"))
    target = candidate["cutoff_run_evidence"][1]
    maximum = candidate["cutoff_run_evidence"][-1]
    for key in (
        "source_order_report_sha256",
        "source_capture_binding",
        "observer_event_log_sha256",
        "observer_log_material_sha256",
        "observer_log_material",
        "semantic_events_sha256",
        "generated_edges_sha256",
        "declared_edges_sha256",
        "generated_edge_count",
        "declared_edge_count",
    ):
        target[key] = maximum[key]
    evidence_body = {
        key: value
        for key, value in target.items()
        if key != "cutoff_run_evidence_sha256"
    }
    target["cutoff_run_evidence_sha256"] = _sha(evidence_body)
    candidate["levels"][1]["independent_cutoff_run_evidence_sha256"] = target[
        "cutoff_run_evidence_sha256"
    ]
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    candidate["report_sha256"] = _sha(body)
    path = tmp_path / "maximum-log-substitution.json"
    path.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(IndependentHistoryFamilyVerificationError):
        verify_receipt(path)


def test_independent_verifier_has_no_producer_or_capture_import() -> None:
    source = Path(verifier_module.__file__).read_text(encoding="utf-8")
    assert "source_causal_history_family import" not in source
    assert "source_derived_causal_order import" not in source
    assert "physical_h3_kms_source_capture" not in source


def test_producer_rejects_mutated_canonical_source_binding(tmp_path: Path) -> None:
    source = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data/causal_order/source_derived_causal_order_receipt.json"
        ).read_text(encoding="ascii")
    )
    source["generated_edges_sha256"] = "sha256:" + "0" * 64
    body = {key: value for key, value in source.items() if key != "report_sha256"}
    source["report_sha256"] = _sha(body)
    path = tmp_path / "mutated-source.json"
    path.write_bytes(_canonical_bytes(source))
    with pytest.raises(SourceCausalHistoryFamilyError):
        produce_source_causal_history_family_report(source_receipt_path=path)
