from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from oph_fpe.ontology import (
    AggregationContract,
    AntecedentRequirement,
    CanonicalValueError,
    CapabilityReceipt,
    Checkpoint,
    ClaimTier,
    ContinuationArrow,
    ExecutionClockReading,
    ExecutionLogEntry,
    FiberStatus,
    FrozenMap,
    NormalFormState,
    ObserverKind,
    ObserverToken,
    OperationalClockReading,
    PhysicalPromotionEvidence,
    PresentationState,
    ProjectorDiagnostic,
    QuotientState,
    ReceiptVerdict,
    RecordAlgebra,
    RepairOrderReading,
    SemanticCarrierState,
    SemanticEvent,
    SemanticOrderReading,
    SourceFirewallViolation,
    aggregate_capability_receipts,
    audit_antecedent_deletions,
    audit_source_packet,
    canonical_hash,
    canonical_json,
    require_source_packet_safe,
)


def _hash(label: str) -> str:
    return canonical_hash({"label": label}, domain="oph.ontology-test")


def _observer(label: str = "observer") -> ObserverToken:
    return ObserverToken(
        kind=ObserverKind.COMPOSITE_FEDERATION,
        birth_event=_hash(f"{label}:birth"),
        lineage_root=_hash(f"{label}:lineage"),
        registry_namespace="test-registry",
    )


def _receipt(
    label: str,
    *,
    receipt_type: str | None = None,
    proves: tuple[str, ...] = ("finite_capability",),
    verdict: ReceiptVerdict = ReceiptVerdict.VALID_PASS,
) -> CapabilityReceipt:
    return CapabilityReceipt(
        receipt_type=receipt_type or f"{label.upper()}_RECEIPT",
        contract_version="test-contract-v1",
        producer="tests.test_simulator_ontology",
        producer_commit="test-commit",
        input_manifest_hash=_hash(f"{label}:manifest"),
        primitive_payload_hash=_hash(f"{label}:payload"),
        verifier_hash=_hash(f"{label}:verifier"),
        claim_tier=ClaimTier.FINITE_THEOREM,
        verdict=verdict,
        scope=f"test:{label}",
        proves=proves if verdict is ReceiptVerdict.VALID_PASS else (),
        nonclaims=("physical_promotion",),
    )


def test_canonical_values_are_deeply_immutable_and_order_independent() -> None:
    mutable = {"z": [1, {"b": 2}], "a": b"payload"}
    frozen = FrozenMap.from_mapping(mutable)
    before = canonical_json(frozen)
    mutable["z"][1]["b"] = 999

    assert canonical_json(frozen) == before
    assert canonical_hash({"a": 1, "b": 2}, domain="test") == canonical_hash(
        {"b": 2, "a": 1}, domain="test"
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        frozen.items_tuple = ()  # type: ignore[misc]
    with pytest.raises(CanonicalValueError):
        FrozenMap.from_mapping({"bad": {1, 2}})
    with pytest.raises(CanonicalValueError):
        FrozenMap.from_mapping({"bad": math.nan})


def test_state_strata_are_distinct_and_never_self_promote() -> None:
    presentation = PresentationState(
        carrier_states={"c0": {"local_coordinates": [0, 1, 2]}},
        seam_states={},
        scheduler_state={"queue_position": 1},
        worker_state={"worker_id": "w0"},
        rng_state={"counter": 7},
        provenance={"commit": "abc"},
    )
    semantic = SemanticCarrierState(
        carrier_id="c0",
        accessible_state={"bit": 1},
        interface_states={"p0": {"response": 0}},
        record_state={"protected": True},
        checkpoint_state={"cut": 2},
        sector_state={"sector": "even"},
    )
    quotient = QuotientState(
        canonical_interface_data=b"interface",
        protected_records=b"records",
        sector_invariants=b"sector",
        semantic_history_root=_hash("history"),
    )
    normal = NormalFormState(
        quotient_state=quotient,
        fiber_status=FiberStatus.UNIQUE,
        normalizer_contract_id="normalizer-v1",
        normalizer_receipt_hash=_hash("normalizer"),
    )

    assert type(presentation) is not type(semantic)
    assert type(semantic) is not type(quotient)
    assert type(quotient) is not type(normal)
    assert not presentation.physical_promotion_receipt
    assert not semantic.physical_promotion_receipt
    assert not quotient.physical_promotion_receipt
    assert not normal.physical_promotion_receipt
    assert "physical_vacuum" in normal.to_jsonable()["nonclaims"]


def test_semantic_state_rejects_nested_presentation_leakage() -> None:
    with pytest.raises(SourceFirewallViolation, match="presentation-only"):
        SemanticCarrierState(
            carrier_id="c0",
            accessible_state={"nested": [{"workerId": "worker-7"}]},
            interface_states={},
            record_state={},
            checkpoint_state={},
            sector_state={},
        )


def test_quotient_and_normal_form_hashes_reject_tampering() -> None:
    with pytest.raises(ValueError, match="quotient_hash"):
        QuotientState(
            canonical_interface_data=b"a",
            protected_records=b"b",
            sector_invariants=b"c",
            semantic_history_root=_hash("history"),
            quotient_hash=_hash("forged"),
        )
    quotient = QuotientState(
        canonical_interface_data=b"a",
        protected_records=b"b",
        sector_invariants=b"c",
        semantic_history_root=_hash("history"),
    )
    with pytest.raises(ValueError, match="normal_form_hash"):
        NormalFormState(
            quotient_state=quotient,
            fiber_status=FiberStatus.UNIQUE,
            normalizer_contract_id="normalizer-v1",
            normalizer_receipt_hash=_hash("normalizer"),
            normal_form_hash=_hash("forged-normal-form"),
        )


def test_semantic_event_id_excludes_executor_metadata() -> None:
    observer = _observer()
    event_a = SemanticEvent(
        canonical_payload={"repair_class": "EXACT_SPLICE", "delta": -1},
        observer_token=observer,
        visible_footprint=("seam-b", "seam-a"),
        semantic_parents=(_hash("parent-b"), _hash("parent-a")),
    )
    log_a = ExecutionLogEntry(
        worker_id="worker-a",
        queue_index=1,
        retry_count=0,
        wall_clock_ns=10,
        process_id="process-a",
        trace_uuid="uuid-a",
        message="committed",
    )
    event_b = SemanticEvent(
        canonical_payload={"delta": -1, "repair_class": "EXACT_SPLICE"},
        observer_token=observer,
        visible_footprint=("seam-a", "seam-b"),
        semantic_parents=(_hash("parent-a"), _hash("parent-b")),
    )
    log_b = ExecutionLogEntry(
        worker_id="worker-z",
        queue_index=99,
        retry_count=8,
        wall_clock_ns=999_999,
        process_id="process-z",
        trace_uuid="uuid-z",
        message="replayed",
    )

    assert event_a.event_id == event_b.event_id
    assert log_a.execution_log_hash != log_b.execution_log_hash
    serialized = event_a.to_json()
    for forbidden in ("worker_id", "queue_index", "retry_count", "wall_clock", "uuid"):
        assert forbidden not in serialized


def test_semantic_event_rejects_executor_fields_even_when_deeply_nested() -> None:
    with pytest.raises(SourceFirewallViolation):
        SemanticEvent(
            canonical_payload={"visible": [{"detail": {"retry_counter": 2}}]},
            observer_token=_observer(),
            visible_footprint=("seam",),
            semantic_parents=(),
        )


def test_record_algebra_diagnostic_and_checkpoint_are_separate() -> None:
    algebra = RecordAlgebra(
        algebra_id="records-v1",
        projector_ids=("p1", "p0", "p1"),
        central_observables={"readback": [0, 1]},
        protected_record_hashes=(_hash("record"),),
    )
    diagnostic = ProjectorDiagnostic(
        projector_id="p0",
        idempotence_error=1.0e-12,
        hermiticity_error=0.0,
        centrality_error=1.0e-13,
        idempotence_tolerance=1.0e-10,
        hermiticity_tolerance=1.0e-10,
        centrality_tolerance=1.0e-10,
        primitive_payload_hash=_hash("projector-payload"),
        verifier_hash=_hash("projector-verifier"),
    )
    checkpoint = Checkpoint(
        observer_token=_observer(),
        quotient_hash=_hash("quotient"),
        semantic_history_root=_hash("history"),
        continuation_data={"frontier": ["event-a", "event-b"]},
    )

    assert diagnostic.passed
    assert not diagnostic.physical_record_receipt
    assert algebra.algebra_hash != checkpoint.checkpoint_hash
    with pytest.raises(SourceFirewallViolation):
        Checkpoint(
            observer_token=_observer(),
            quotient_hash=_hash("quotient"),
            semantic_history_root=_hash("history"),
            continuation_data={"rng_state": {"counter": 9}},
        )


def test_observer_continuation_is_registry_evidence_not_uuid_identity() -> None:
    source = _observer("source")
    target = _observer("target")
    arrow = ContinuationArrow(
        source_observer=source,
        target_observer=target,
        source_checkpoint=_hash("source-checkpoint"),
        target_checkpoint=_hash("target-checkpoint"),
        continuation_error=0.01,
        evidence_hash=_hash("continuation-evidence"),
    )

    assert source.token_hash != target.token_hash
    assert not arrow.physical_continuation_receipt
    assert "worker" not in arrow.to_json()


def test_four_clock_domains_are_nominally_and_semantically_separate() -> None:
    observer = _observer()
    execution = ExecutionClockReading(
        process_id="process-1",
        wall_clock_ns=100,
        cpu_time_ns=80,
        queue_delay_ns=20,
    )
    repair = RepairOrderReading(
        commit_id=_hash("commit"),
        parent_commit_ids=(_hash("parent-commit"),),
        causal_depth=4,
    )
    semantic = SemanticOrderReading(
        observer_token=observer,
        event_id=_hash("event"),
        parent_event_ids=(_hash("parent-event"),),
        observer_causal_depth=3,
    )
    operational = OperationalClockReading(
        observer_token=observer,
        checkpoint_hash=_hash("checkpoint"),
        distribution={"support": [0.0, 1.0], "weights": [0.5, 0.5]},
        calibration_id="independent-calibration-v1",
        calibration_receipt_hash=_hash("calibration"),
        affine_scale=1.0,
        affine_offset=0.0,
        residual_bound=0.02,
    )

    assert len({type(execution), type(repair), type(semantic), type(operational)}) == 4
    assert execution.to_jsonable()["clock_domain"] == "EXECUTION"
    assert repair.to_jsonable()["clock_domain"] == "REPAIR_ORDER"
    assert semantic.to_jsonable()["clock_domain"] == "SEMANTIC_ORDER"
    assert operational.to_jsonable()["clock_domain"] == "OPERATIONAL"
    assert not operational.physical_clock_receipt


def test_recursive_source_firewall_rejects_hidden_and_target_data() -> None:
    packet = {
        "source": {
            "layers": [
                {"visible_algebra": "a"},
                {"workerOwnership": "worker-1"},
                {"physics": {"target_beta": 2.0 * math.pi}},
            ]
        }
    }
    report = audit_source_packet(packet)

    assert not report.passed
    assert {finding.category for finding in report.findings} >= {
        "HIDDEN_PRESENTATION",
        "DOWNSTREAM_TARGET",
    }
    assert not report.physical_promotion_receipt
    with pytest.raises(SourceFirewallViolation):
        require_source_packet_safe(packet)


def test_source_firewall_rejects_presentation_objects_and_schema_extras() -> None:
    presentation = PresentationState(
        carrier_states={},
        seam_states={},
        scheduler_state={},
        worker_state={},
        rng_state={},
        provenance={},
    )
    assert not audit_source_packet({"payload": presentation}).passed
    report = audit_source_packet(
        {"state": {"visible": 1}, "surprise": 2},
        allowed_top_level_fields=frozenset({"state"}),
    )
    assert any(row.category == "SOURCE_SCHEMA_EXTRA_FIELD" for row in report.findings)


def test_physical_claim_tier_does_not_promote_without_typed_evidence() -> None:
    receipt = CapabilityReceipt(
        receipt_type="PHYSICAL_EXAMPLE",
        contract_version="physical-contract-v1",
        producer="test-producer",
        producer_commit="test-commit",
        input_manifest_hash=_hash("manifest"),
        primitive_payload_hash=_hash("payload"),
        verifier_hash=_hash("verifier"),
        claim_tier=ClaimTier.PHYSICAL_RECEIPT,
        verdict=ReceiptVerdict.VALID_PASS,
        scope="physical-example",
        proves=("example_fit",),
        nonclaims=("event_manifold",),
    )

    assert not receipt.physical_promotion_receipt
    assert receipt.to_jsonable()["physical_promotion_receipt"] is False


def test_typed_physical_evidence_must_be_bound_to_antecedents() -> None:
    parents = tuple(_receipt(f"physical-parent-{index}") for index in range(5))
    evidence = PhysicalPromotionEvidence(
        source_contract_receipt_hash=parents[0].receipt_hash,
        source_firewall_receipt_hash=parents[1].receipt_hash,
        no_target_path_receipt_hash=parents[2].receipt_hash,
        independent_evaluator_receipt_hash=parents[3].receipt_hash,
        negative_control_receipt_hashes=(parents[4].receipt_hash,),
        evidence_scope="physical-example",
    )
    promoted = CapabilityReceipt(
        receipt_type="PHYSICAL_EXAMPLE",
        contract_version="physical-contract-v1",
        producer="test-producer",
        producer_commit="test-commit",
        input_manifest_hash=_hash("manifest"),
        primitive_payload_hash=_hash("payload"),
        verifier_hash=_hash("verifier"),
        claim_tier=ClaimTier.PHYSICAL_RECEIPT,
        verdict=ReceiptVerdict.VALID_PASS,
        scope="physical-example",
        proves=("source_separated_physical_result",),
        nonclaims=("cosmology",),
        antecedent_receipts=tuple(parent.receipt_hash for parent in parents),
        physical_evidence=evidence,
    )

    assert promoted.physical_evidence_structurally_bound
    assert not promoted.physical_promotion_receipt
    with pytest.raises(ValueError, match="absent from antecedents"):
        CapabilityReceipt(
            receipt_type="FORGED_PHYSICAL_EXAMPLE",
            contract_version="physical-contract-v1",
            producer="test-producer",
            producer_commit="test-commit",
            input_manifest_hash=_hash("manifest"),
            primitive_payload_hash=_hash("payload"),
            verifier_hash=_hash("verifier"),
            claim_tier=ClaimTier.PHYSICAL_RECEIPT,
            verdict=ReceiptVerdict.VALID_PASS,
            scope="physical-example",
            proves=("source_separated_physical_result",),
            nonclaims=("cosmology",),
            antecedent_receipts=tuple(parent.receipt_hash for parent in parents[:-1]),
            physical_evidence=evidence,
        )


def test_capability_proves_and_nonclaims_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="both proved and nonclaimed"):
        CapabilityReceipt(
            receipt_type="CONTRADICTORY",
            contract_version="contract-v1",
            producer="producer",
            producer_commit="commit",
            input_manifest_hash=_hash("manifest"),
            primitive_payload_hash=_hash("payload"),
            verifier_hash=_hash("verifier"),
            claim_tier=ClaimTier.FINITE_THEOREM,
            verdict=ReceiptVerdict.VALID_PASS,
            scope="test",
            proves=("h3_geometry",),
            nonclaims=("h3_geometry",),
        )


def test_antecedent_deletion_makes_aggregate_fail_closed() -> None:
    source = _receipt(
        "source",
        receipt_type="SOURCE_CONTRACT",
        proves=("source_contract_verified",),
    )
    repair = _receipt(
        "repair",
        receipt_type="REPAIR_CONTRACT",
        proves=("transactional_repair_verified",),
    )
    contract = AggregationContract(
        contract_id="aggregate-contract-v1",
        receipt_type="ONTOLOGY_AGGREGATE",
        claim_tier=ClaimTier.FINITE_THEOREM,
        required_antecedents=(
            AntecedentRequirement(
                requirement_id="source",
                receipt_hash=source.receipt_hash,
                receipt_type="SOURCE_CONTRACT",
                required_proves=("source_contract_verified",),
            ),
            AntecedentRequirement(
                requirement_id="repair",
                receipt_hash=repair.receipt_hash,
                receipt_type="REPAIR_CONTRACT",
                required_proves=("transactional_repair_verified",),
            ),
        ),
        proves=("ontology_chain_complete",),
        nonclaims=("physical_clock", "physical_geometry"),
        scope="finite-ontology-chain",
    )
    kwargs = {
        "producer": "aggregate-verifier",
        "producer_commit": "test-commit",
        "input_manifest_hash": _hash("aggregate-manifest"),
        "primitive_payload_hash": _hash("aggregate-payload"),
        "verifier_hash": _hash("aggregate-verifier"),
    }
    passed = aggregate_capability_receipts(contract, (source, repair), **kwargs)
    deleted = aggregate_capability_receipts(contract, (source,), **kwargs)
    deletion_report = audit_antecedent_deletions(contract, (source, repair))

    assert passed.passed
    assert passed.proves == ("ontology_chain_complete",)
    assert deletion_report.passed
    assert deleted.verdict is ReceiptVerdict.NO_GO
    assert not deleted.proves
    assert "aggregate_contract_not_discharged" in deleted.nonclaims


def test_failed_exact_antecedent_cannot_be_replaced_by_same_type() -> None:
    required = _receipt(
        "required",
        receipt_type="EXACT_PARENT",
        proves=("required_capability",),
    )
    substitute = _receipt(
        "substitute",
        receipt_type="EXACT_PARENT",
        proves=("required_capability",),
    )
    contract = AggregationContract(
        contract_id="exact-parent-contract-v1",
        receipt_type="EXACT_AGGREGATE",
        claim_tier=ClaimTier.FINITE_THEOREM,
        required_antecedents=(
            AntecedentRequirement(
                requirement_id="frozen-required-parent",
                receipt_hash=required.receipt_hash,
                receipt_type=required.receipt_type,
                required_proves=("required_capability",),
            ),
        ),
        proves=("exact_parent_bound",),
        nonclaims=("physical_result",),
        scope="exact-parent-test",
    )
    aggregate = aggregate_capability_receipts(
        contract,
        (substitute,),
        producer="aggregate-verifier",
        producer_commit="test-commit",
        input_manifest_hash=_hash("manifest"),
        primitive_payload_hash=_hash("payload"),
        verifier_hash=_hash("verifier"),
    )

    assert aggregate.verdict is ReceiptVerdict.NO_GO
    assert aggregate.to_jsonable()["derived_values"]["missing_requirements"] == (
        "frozen-required-parent",
    )


def test_deleting_physical_evidence_parent_drops_promotion_without_exception() -> None:
    parents = tuple(
        _receipt(
            f"promotion-parent-{index}",
            receipt_type=f"PROMOTION_PARENT_{index}",
            proves=(f"promotion_capability_{index}",),
        )
        for index in range(5)
    )
    contract = AggregationContract(
        contract_id="physical-aggregate-contract-v1",
        receipt_type="PHYSICAL_AGGREGATE",
        claim_tier=ClaimTier.PHYSICAL_RECEIPT,
        required_antecedents=tuple(
            AntecedentRequirement(
                requirement_id=f"parent-{index}",
                receipt_hash=parent.receipt_hash,
                receipt_type=parent.receipt_type,
                required_proves=(f"promotion_capability_{index}",),
            )
            for index, parent in enumerate(parents)
        ),
        proves=("physical_aggregate_capability",),
        nonclaims=("unregistered_physical_scope",),
        scope="frozen-physical-cell",
    )
    evidence = PhysicalPromotionEvidence(
        source_contract_receipt_hash=parents[0].receipt_hash,
        source_firewall_receipt_hash=parents[1].receipt_hash,
        no_target_path_receipt_hash=parents[2].receipt_hash,
        independent_evaluator_receipt_hash=parents[3].receipt_hash,
        negative_control_receipt_hashes=(parents[4].receipt_hash,),
        evidence_scope="frozen-physical-cell",
    )
    kwargs = {
        "producer": "aggregate-verifier",
        "producer_commit": "test-commit",
        "input_manifest_hash": _hash("physical-aggregate-manifest"),
        "primitive_payload_hash": _hash("physical-aggregate-payload"),
        "verifier_hash": _hash("physical-aggregate-verifier"),
        "physical_evidence": evidence,
    }

    promoted = aggregate_capability_receipts(contract, parents, **kwargs)
    deleted = aggregate_capability_receipts(contract, parents[:-1], **kwargs)

    assert promoted.physical_evidence_structurally_bound
    assert not promoted.physical_promotion_receipt
    assert deleted.verdict is ReceiptVerdict.NO_GO
    assert not deleted.physical_evidence_structurally_bound
    assert not deleted.physical_promotion_receipt
