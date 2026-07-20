"""Exact finite quotient and obstruction diagnostics.

These utilities certify statements only about the finite model explicitly
enumerated by a caller.  They never promote a simulator run to a physical
emergence claim.
"""

from .obstruction import (
    AbelianEdge,
    FiberClassification,
    FiberStatus,
    FundamentalCycle,
    HolonomyAudit,
    KernelKind,
    LumpabilityAudit,
    LumpabilityDefect,
    RepairInitialAudit,
    RepairPathAudit,
    audit_abelian_cycle_holonomy,
    classify_boundary_fiber,
    exhaust_accepted_repair_paths,
    verify_quotient_lumpability,
)
from .observable_normal_form import (
    BoundaryIdentificationDefect,
    ConditionalResamplingAudit,
    CrossSourceEndpointDefect,
    KernelEntryDefect,
    KernelRowDefect,
    ObservableNormalFormAudit,
    ObservationLeak,
    recognize_conditional_resampling_kernel,
    verify_observation_determined_normal_form,
)

__all__ = [
    "AbelianEdge",
    "BoundaryIdentificationDefect",
    "ConditionalResamplingAudit",
    "CrossSourceEndpointDefect",
    "FiberClassification",
    "FiberStatus",
    "FundamentalCycle",
    "HolonomyAudit",
    "KernelKind",
    "KernelEntryDefect",
    "KernelRowDefect",
    "LumpabilityAudit",
    "LumpabilityDefect",
    "ObservableNormalFormAudit",
    "ObservationLeak",
    "RepairInitialAudit",
    "RepairPathAudit",
    "audit_abelian_cycle_holonomy",
    "classify_boundary_fiber",
    "exhaust_accepted_repair_paths",
    "recognize_conditional_resampling_kernel",
    "verify_observation_determined_normal_form",
    "verify_quotient_lumpability",
]
