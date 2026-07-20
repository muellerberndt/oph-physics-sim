"""Proof-carrying capability receipts and deletion-sensitive aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._canonical import (
    FrozenMap,
    canonical_hash,
    canonical_json,
    require_nonempty,
    require_sha256,
)


RECEIPT_SCHEMA = "oph.capability-receipt.v2"


class ClaimTier(str, Enum):
    IMPLEMENTATION = "IMPLEMENTATION"
    FINITE_THEOREM = "FINITE_THEOREM"
    REGULATOR_DIAGNOSTIC = "REGULATOR_DIAGNOSTIC"
    CONDITIONAL_PHYSICAL = "CONDITIONAL_PHYSICAL"
    PHYSICAL_RECEIPT = "PHYSICAL_RECEIPT"


class ReceiptVerdict(str, Enum):
    VALID_PASS = "VALID_PASS"
    VALID_FAIL = "VALID_FAIL"
    NO_GO = "NO_GO"
    INVALID_INSTRUMENT = "INVALID_INSTRUMENT"


@dataclass(frozen=True, slots=True)
class PhysicalPromotionEvidence:
    """Typed references required before physical evidence is structurally bound.

    These are content-addressed parent receipts, not caller-authored booleans.
    The aggregate receipt also requires every reference to occur in its frozen
    antecedent list, so deleting any one invalidates structural binding.  This
    object is not an artifact verifier and cannot promote a physical claim.
    """

    source_contract_receipt_hash: str
    source_firewall_receipt_hash: str
    no_target_path_receipt_hash: str
    independent_evaluator_receipt_hash: str
    negative_control_receipt_hashes: tuple[str, ...]
    evidence_scope: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_contract_receipt_hash",
            "source_firewall_receipt_hash",
            "no_target_path_receipt_hash",
            "independent_evaluator_receipt_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if isinstance(self.negative_control_receipt_hashes, (str, bytes)):
            raise TypeError(
                "negative_control_receipt_hashes must be a sequence, not text"
            )
        controls = tuple(sorted(set(self.negative_control_receipt_hashes)))
        if not controls:
            raise ValueError("physical promotion requires a negative-control receipt")
        for receipt_hash in controls:
            require_sha256(receipt_hash, field_name="negative_control_receipt_hash")
        object.__setattr__(self, "negative_control_receipt_hashes", controls)
        require_nonempty(self.evidence_scope, field_name="evidence_scope")

    @property
    def referenced_receipts(self) -> frozenset[str]:
        return frozenset(
            {
                self.source_contract_receipt_hash,
                self.source_firewall_receipt_hash,
                self.no_target_path_receipt_hash,
                self.independent_evaluator_receipt_hash,
                *self.negative_control_receipt_hashes,
            }
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source_contract_receipt_hash": self.source_contract_receipt_hash,
            "source_firewall_receipt_hash": self.source_firewall_receipt_hash,
            "no_target_path_receipt_hash": self.no_target_path_receipt_hash,
            "independent_evaluator_receipt_hash": self.independent_evaluator_receipt_hash,
            "negative_control_receipt_hashes": list(
                self.negative_control_receipt_hashes
            ),
            "evidence_scope": self.evidence_scope,
        }


@dataclass(frozen=True, slots=True)
class CapabilityReceipt:
    """Common immutable receipt schema with explicit capabilities/nonclaims."""

    receipt_type: str
    contract_version: str
    producer: str
    producer_commit: str
    input_manifest_hash: str
    primitive_payload_hash: str
    verifier_hash: str
    claim_tier: ClaimTier
    verdict: ReceiptVerdict
    scope: str
    proves: tuple[str, ...]
    nonclaims: tuple[str, ...]
    antecedent_receipts: tuple[str, ...] = ()
    derived_values: FrozenMap | Mapping[str, Any] = FrozenMap()
    thresholds: FrozenMap | Mapping[str, Any] = FrozenMap()
    uncertainty: FrozenMap | Mapping[str, Any] = FrozenMap()
    controls: FrozenMap | Mapping[str, Any] = FrozenMap()
    physical_evidence: PhysicalPromotionEvidence | None = None
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_type",
            "contract_version",
            "producer",
            "producer_commit",
            "scope",
        ):
            require_nonempty(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "input_manifest_hash",
            "primitive_payload_hash",
            "verifier_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.claim_tier, ClaimTier):
            raise TypeError("claim_tier must be a ClaimTier")
        if not isinstance(self.verdict, ReceiptVerdict):
            raise TypeError("verdict must be a ReceiptVerdict")
        proves = _normalize_capabilities(self.proves, field_name="proves")
        nonclaims = _normalize_capabilities(self.nonclaims, field_name="nonclaims")
        if not nonclaims:
            raise ValueError(
                "every capability receipt must state at least one nonclaim"
            )
        overlap = set(proves) & set(nonclaims)
        if overlap:
            raise ValueError(
                f"capabilities cannot be both proved and nonclaimed: {sorted(overlap)}"
            )
        object.__setattr__(self, "proves", proves)
        object.__setattr__(self, "nonclaims", nonclaims)
        if isinstance(self.antecedent_receipts, (str, bytes)):
            raise TypeError("antecedent_receipts must be a sequence, not text")
        antecedents = tuple(sorted(set(self.antecedent_receipts)))
        for receipt_hash in antecedents:
            require_sha256(receipt_hash, field_name="antecedent_receipt")
        object.__setattr__(self, "antecedent_receipts", antecedents)
        for field_name in ("derived_values", "thresholds", "uncertainty", "controls"):
            object.__setattr__(
                self,
                field_name,
                FrozenMap.from_mapping(getattr(self, field_name)),
            )
        if self.physical_evidence is not None:
            if not isinstance(self.physical_evidence, PhysicalPromotionEvidence):
                raise TypeError("physical_evidence must be PhysicalPromotionEvidence")
            if self.claim_tier is not ClaimTier.PHYSICAL_RECEIPT:
                raise ValueError(
                    "physical_evidence is only valid at PHYSICAL_RECEIPT tier"
                )
            if self.physical_evidence.evidence_scope != self.scope:
                raise ValueError("physical evidence scope does not match receipt scope")
            missing = self.physical_evidence.referenced_receipts - set(antecedents)
            if missing:
                raise ValueError(
                    "physical evidence references are absent from antecedents: "
                    + ", ".join(sorted(missing))
                )
        expected = canonical_hash(
            self._hash_material(), domain="oph.capability-receipt.v2"
        )
        if self.receipt_hash:
            require_sha256(self.receipt_hash, field_name="receipt_hash")
            if self.receipt_hash != expected:
                raise ValueError(
                    "receipt_hash does not match canonical receipt material"
                )
        else:
            object.__setattr__(self, "receipt_hash", expected)

    @property
    def passed(self) -> bool:
        return self.verdict is ReceiptVerdict.VALID_PASS

    @property
    def physical_evidence_structurally_bound(self) -> bool:
        return (
            self.passed
            and self.claim_tier is ClaimTier.PHYSICAL_RECEIPT
            and self.physical_evidence is not None
        )

    @property
    def physical_promotion_receipt(self) -> bool:
        """Fail closed until a primitive-artifact verifier exists.

        A caller can construct the dataclasses in this module, so canonical
        typing and hash binding alone are not an admission-grade physical
        verifier.  A future verifier must expose a separate trusted result.
        """

        return False

    def _hash_material(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "receipt_type": self.receipt_type,
            "contract_version": self.contract_version,
            "producer": self.producer,
            "producer_commit": self.producer_commit,
            "input_manifest_hash": self.input_manifest_hash,
            "antecedent_receipts": self.antecedent_receipts,
            "primitive_payload_hash": self.primitive_payload_hash,
            "derived_values": self.derived_values,
            "thresholds": self.thresholds,
            "uncertainty": self.uncertainty,
            "controls": self.controls,
            "verdict": self.verdict.value,
            "scope": self.scope,
            "claim_tier": self.claim_tier.value,
            "proves": self.proves,
            "nonclaims": self.nonclaims,
            "verifier_hash": self.verifier_hash,
            "physical_evidence": (
                None
                if self.physical_evidence is None
                else self.physical_evidence.to_jsonable()
            ),
        }

    def to_jsonable(self) -> dict[str, Any]:
        return {
            **self._hash_material(),
            "receipt_hash": self.receipt_hash,
            "physical_evidence_structurally_bound": self.physical_evidence_structurally_bound,
            "physical_promotion_receipt": self.physical_promotion_receipt,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class AntecedentRequirement:
    requirement_id: str
    receipt_hash: str
    receipt_type: str
    required_proves: tuple[str, ...]

    def __post_init__(self) -> None:
        require_nonempty(self.requirement_id, field_name="requirement_id")
        require_sha256(self.receipt_hash, field_name="receipt_hash")
        require_nonempty(self.receipt_type, field_name="receipt_type")
        object.__setattr__(
            self,
            "required_proves",
            _normalize_capabilities(self.required_proves, field_name="required_proves"),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "receipt_hash": self.receipt_hash,
            "receipt_type": self.receipt_type,
            "required_proves": list(self.required_proves),
        }


@dataclass(frozen=True, slots=True)
class AggregationContract:
    contract_id: str
    receipt_type: str
    claim_tier: ClaimTier
    required_antecedents: tuple[AntecedentRequirement, ...]
    proves: tuple[str, ...]
    nonclaims: tuple[str, ...]
    scope: str

    def __post_init__(self) -> None:
        require_nonempty(self.contract_id, field_name="contract_id")
        require_nonempty(self.receipt_type, field_name="receipt_type")
        require_nonempty(self.scope, field_name="scope")
        if not isinstance(self.claim_tier, ClaimTier):
            raise TypeError("claim_tier must be a ClaimTier")
        if not self.required_antecedents:
            raise ValueError("aggregation contracts require frozen antecedents")
        ids = [requirement.requirement_id for requirement in self.required_antecedents]
        hashes = [requirement.receipt_hash for requirement in self.required_antecedents]
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise ValueError("antecedent requirement IDs and hashes must be unique")
        object.__setattr__(
            self,
            "required_antecedents",
            tuple(
                sorted(self.required_antecedents, key=lambda row: row.requirement_id)
            ),
        )
        proves = _normalize_capabilities(self.proves, field_name="proves")
        nonclaims = _normalize_capabilities(self.nonclaims, field_name="nonclaims")
        if not nonclaims:
            raise ValueError("aggregation contracts require explicit nonclaims")
        if set(proves) & set(nonclaims):
            raise ValueError("aggregation proves and nonclaims must be disjoint")
        object.__setattr__(self, "proves", proves)
        object.__setattr__(self, "nonclaims", nonclaims)


@dataclass(frozen=True, slots=True)
class AntecedentDeletionReport:
    baseline_passed: bool
    deletion_outcomes: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return self.baseline_passed and all(
            not passed for _, passed in self.deletion_outcomes
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "baseline_passed": self.baseline_passed,
            "deletion_outcomes": [
                {"removed_receipt_hash": receipt_hash, "aggregate_passed": passed}
                for receipt_hash, passed in self.deletion_outcomes
            ],
            "mandatory_parent_deletion_test_passed": self.passed,
        }


def _normalize_capabilities(
    values: Sequence[str], *, field_name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{field_name} must be a sequence of capability names, not text"
        )
    result = tuple(sorted(set(values)))
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{field_name} must contain nonempty strings")
    return result


def _missing_requirements(
    contract: AggregationContract,
    antecedents: Sequence[CapabilityReceipt],
) -> tuple[str, ...]:
    by_hash = {receipt.receipt_hash: receipt for receipt in antecedents}
    missing: list[str] = []
    for requirement in contract.required_antecedents:
        receipt = by_hash.get(requirement.receipt_hash)
        if (
            receipt is None
            or receipt.receipt_type != requirement.receipt_type
            or not receipt.passed
            or not set(requirement.required_proves).issubset(receipt.proves)
        ):
            missing.append(requirement.requirement_id)
    return tuple(sorted(missing))


def audit_antecedent_deletions(
    contract: AggregationContract,
    antecedents: Sequence[CapabilityReceipt],
) -> AntecedentDeletionReport:
    unique = {receipt.receipt_hash: receipt for receipt in antecedents}
    baseline = not _missing_requirements(contract, tuple(unique.values()))
    outcomes: list[tuple[str, bool]] = []
    for requirement in contract.required_antecedents:
        reduced = tuple(
            receipt
            for receipt_hash, receipt in unique.items()
            if receipt_hash != requirement.receipt_hash
        )
        outcomes.append(
            (requirement.receipt_hash, not _missing_requirements(contract, reduced))
        )
    return AntecedentDeletionReport(
        baseline_passed=baseline,
        deletion_outcomes=tuple(outcomes),
    )


def aggregate_capability_receipts(
    contract: AggregationContract,
    antecedents: Sequence[CapabilityReceipt],
    *,
    producer: str,
    producer_commit: str,
    input_manifest_hash: str,
    primitive_payload_hash: str,
    verifier_hash: str,
    physical_evidence: PhysicalPromotionEvidence | None = None,
) -> CapabilityReceipt:
    """Aggregate only if every frozen parent survives the deletion test."""

    unique = tuple(
        sorted(
            {receipt.receipt_hash: receipt for receipt in antecedents}.values(),
            key=lambda receipt: receipt.receipt_hash,
        )
    )
    missing = _missing_requirements(contract, unique)
    deletion = audit_antecedent_deletions(contract, unique)
    passed = not missing and deletion.passed
    verdict = ReceiptVerdict.VALID_PASS if passed else ReceiptVerdict.NO_GO
    nonclaims = contract.nonclaims
    if not passed:
        nonclaims = tuple(sorted({*nonclaims, "aggregate_contract_not_discharged"}))
    antecedent_hashes = frozenset(receipt.receipt_hash for receipt in unique)
    bound_physical_evidence = physical_evidence
    if (
        physical_evidence is not None
        and not physical_evidence.referenced_receipts.issubset(antecedent_hashes)
    ):
        # Deleting a physical-evidence parent must yield a false aggregate, not
        # an exception and never a still-promoted receipt.
        bound_physical_evidence = None
    return CapabilityReceipt(
        receipt_type=contract.receipt_type,
        contract_version=contract.contract_id,
        producer=producer,
        producer_commit=producer_commit,
        input_manifest_hash=input_manifest_hash,
        primitive_payload_hash=primitive_payload_hash,
        verifier_hash=verifier_hash,
        claim_tier=contract.claim_tier,
        verdict=verdict,
        scope=contract.scope,
        proves=contract.proves if passed else (),
        nonclaims=nonclaims,
        antecedent_receipts=tuple(receipt.receipt_hash for receipt in unique),
        derived_values={"missing_requirements": missing},
        controls={"antecedent_deletion": deletion.to_jsonable()},
        physical_evidence=bound_physical_evidence,
    )
