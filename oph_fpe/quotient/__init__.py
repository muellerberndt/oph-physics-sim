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

__all__ = [
    "AbelianEdge",
    "FiberClassification",
    "FiberStatus",
    "FundamentalCycle",
    "HolonomyAudit",
    "KernelKind",
    "LumpabilityAudit",
    "LumpabilityDefect",
    "RepairInitialAudit",
    "RepairPathAudit",
    "audit_abelian_cycle_holonomy",
    "classify_boundary_fiber",
    "exhaust_accepted_repair_paths",
    "verify_quotient_lumpability",
]
