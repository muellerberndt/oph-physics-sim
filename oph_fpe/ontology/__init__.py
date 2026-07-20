"""Canonical simulator ontology and fail-closed promotion contracts."""

from ._canonical import (
    CanonicalValueError,
    FrozenMap,
    canonical_hash,
    canonical_json,
)
from .clocks import (
    ExecutionClockReading,
    OperationalClockReading,
    RepairOrderReading,
    SemanticOrderReading,
)
from .firewall import (
    DOWNSTREAM_TARGET_FIELDS,
    PRESENTATION_ONLY_FIELDS,
    SourceFirewallReport,
    SourceFirewallViolation,
    audit_source_packet,
    require_source_packet_safe,
)
from .observers import (
    ContinuationArrow,
    ContinuationKind,
    ObserverKind,
    ObserverToken,
)
from .receipts import (
    AggregationContract,
    AntecedentDeletionReport,
    AntecedentRequirement,
    CapabilityReceipt,
    ClaimTier,
    PhysicalPromotionEvidence,
    ReceiptVerdict,
    aggregate_capability_receipts,
    audit_antecedent_deletions,
)
from .records import (
    Checkpoint,
    ExecutionLogEntry,
    ProjectorDiagnostic,
    RecordAlgebra,
    SemanticEvent,
)
from .states import (
    FiberStatus,
    NormalFormState,
    PresentationState,
    QuotientState,
    SemanticCarrierState,
)

__all__ = [
    "DOWNSTREAM_TARGET_FIELDS",
    "PRESENTATION_ONLY_FIELDS",
    "AggregationContract",
    "AntecedentDeletionReport",
    "AntecedentRequirement",
    "CanonicalValueError",
    "CapabilityReceipt",
    "Checkpoint",
    "ClaimTier",
    "ContinuationArrow",
    "ContinuationKind",
    "ExecutionClockReading",
    "ExecutionLogEntry",
    "FiberStatus",
    "FrozenMap",
    "NormalFormState",
    "ObserverKind",
    "ObserverToken",
    "OperationalClockReading",
    "PhysicalPromotionEvidence",
    "PresentationState",
    "ProjectorDiagnostic",
    "QuotientState",
    "ReceiptVerdict",
    "RecordAlgebra",
    "RepairOrderReading",
    "SemanticCarrierState",
    "SemanticEvent",
    "SemanticOrderReading",
    "SourceFirewallReport",
    "SourceFirewallViolation",
    "aggregate_capability_receipts",
    "audit_antecedent_deletions",
    "audit_source_packet",
    "canonical_hash",
    "canonical_json",
    "require_source_packet_safe",
]
