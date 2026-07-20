from importlib import import_module

from oph_fpe.observers.objects import (
    RecordFamily,
    assign_counterfactual_stability_from_records,
    counterfactual_stability,
    extract_record_families,
    object_consensus_score,
    observer_object_report,
    transition_affinity_packet_fields,
    visible_object_packets,
)
from oph_fpe.observers.operational_verifier import (
    OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT,
    OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT,
    TRANSACTION_PARENT_ENVELOPE_SCHEMA,
    compute_observer_contract_binding,
    compute_outcome_generator_precommitment,
    compute_outcome_secret_commitment,
    compute_prediction_phase_commitment,
    frozen_shuffle_permutation,
    frozen_source_outcomes,
    semantic_observer_event_id,
    verify_operational_observer_manifest,
    write_operational_observer_report,
)
from oph_fpe.observers.semantic_clock import (
    OBSERVER_KINDS,
    affine_clock_residual_report,
    distributed_observer_uid,
    normalize_observer_frame,
    observer_registry_audit,
    semantic_event_key,
    semantic_history_digest,
    semantic_history_invariance_report,
)
from oph_fpe.observers.sampling import deterministic_observer_analysis_indices
# ``subjective`` imports cap geometry from ``oph_fpe.bulk`` while legacy bulk
# initialisation imports ``neutral_bulk``, which in turn imports one subjective
# helper.  Eagerly importing those names here therefore makes importing an
# independent observer verifier depend on import order.  Keep the public API,
# but resolve the three legacy analysis helpers lazily.
_LAZY_SUBJECTIVE_EXPORTS = frozenset(
    {
        "observer_consensus_report",
        "observer_view_rows",
    }
)


def __getattr__(name: str):
    if name in _LAZY_SUBJECTIVE_EXPORTS:
        subjective = import_module("oph_fpe.observers.subjective")
        value = getattr(subjective, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "OBSERVER_KINDS",
    "OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT",
    "OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT",
    "RecordFamily",
    "TRANSACTION_PARENT_ENVELOPE_SCHEMA",
    "affine_clock_residual_report",
    "assign_counterfactual_stability_from_records",
    "counterfactual_stability",
    "compute_observer_contract_binding",
    "compute_outcome_generator_precommitment",
    "compute_outcome_secret_commitment",
    "compute_prediction_phase_commitment",
    "deterministic_observer_analysis_indices",
    "distributed_observer_uid",
    "extract_record_families",
    "frozen_shuffle_permutation",
    "frozen_source_outcomes",
    "normalize_observer_frame",
    "object_consensus_score",
    "observer_consensus_report",
    "observer_object_report",
    "observer_registry_audit",
    "observer_view_rows",
    "semantic_event_key",
    "semantic_observer_event_id",
    "semantic_history_digest",
    "semantic_history_invariance_report",
    "transition_affinity_packet_fields",
    "verify_operational_observer_manifest",
    "visible_object_packets",
    "write_operational_observer_report",
]
