"""Proof-carrying finite repair transactions.

The package is deliberately independent of the legacy simulator engines.  A
caller must cross the contracts in :mod:`oph_fpe.repair.transaction` before a
state update can carry a transactional-repair receipt.
"""

from .transaction import (
    MismatchLedger,
    ProposalFootprint,
    ProposalClass,
    ProposalValidation,
    RegisterSnapshot,
    RepairCollar,
    RepairCommitReceipt,
    RepairProposal,
    Snapshot,
    TransactionKind,
    TransactionalRepairEngine,
    TransitionKind,
    VerificationCheck,
    conflict_components,
    prepare_proposal,
    proposals_conflict,
    verify_repair_receipt_artifact,
)

__all__ = [
    "MismatchLedger",
    "ProposalFootprint",
    "ProposalClass",
    "ProposalValidation",
    "RegisterSnapshot",
    "RepairCollar",
    "RepairCommitReceipt",
    "RepairProposal",
    "Snapshot",
    "TransactionKind",
    "TransactionalRepairEngine",
    "TransitionKind",
    "VerificationCheck",
    "conflict_components",
    "prepare_proposal",
    "proposals_conflict",
    "verify_repair_receipt_artifact",
]
