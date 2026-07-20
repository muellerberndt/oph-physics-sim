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
    REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE,
    REPAIR_REPLAY_ENVELOPE_SCHEMA,
    REPAIR_REPLAY_VERIFIER_VERSION,
    RegisterSnapshot,
    RepairCollar,
    RepairCommitReceipt,
    RepairProposal,
    Snapshot,
    TransactionKind,
    TransactionalRepairEngine,
    TransitionKind,
    VerificationCheck,
    build_repair_replay_envelope,
    conflict_components,
    prepare_proposal,
    proposals_conflict,
    verify_repair_receipt_artifact,
    verify_repair_replay_envelope,
)

__all__ = [
    "MismatchLedger",
    "ProposalFootprint",
    "ProposalClass",
    "ProposalValidation",
    "REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE",
    "REPAIR_REPLAY_ENVELOPE_SCHEMA",
    "REPAIR_REPLAY_VERIFIER_VERSION",
    "RegisterSnapshot",
    "RepairCollar",
    "RepairCommitReceipt",
    "RepairProposal",
    "Snapshot",
    "TransactionKind",
    "TransactionalRepairEngine",
    "TransitionKind",
    "VerificationCheck",
    "build_repair_replay_envelope",
    "conflict_components",
    "prepare_proposal",
    "proposals_conflict",
    "verify_repair_receipt_artifact",
    "verify_repair_replay_envelope",
]
