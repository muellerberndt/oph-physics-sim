"""Fail-closed audit of the OPH source-to-observer-to-physics ladder.

The simulator has many independently produced reports.  This module does not
infer a theorem from a suggestive diagnostic and does not rerun any numerical
fit.  It admits evidence only through a registry of independent artifact
verifiers, then evaluates a typed dependency DAG.  A caller-authored JSON
boolean is never ladder evidence, even when its key exactly matches a canonical
receipt name.  Missing, verifier-rejected, false, or contradictory evidence
cannot promote a downstream claim.

Three separation rules are encoded in the graph itself:

* an A5/icosahedral refinement witness is necessary but not sufficient for the
  Standard Model spine;
* finite consensus is not a geometric or gravitational receipt;
* an H3 observer-frame fiber is not a 3+1-dimensional event-position manifold.

The canonical receipt names below intentionally err on the side of requiring a
new, narrow receipt when an older report has a broader or ambiguous meaning.
This makes the auditor suitable as a campaign preflight and as a regression
guard against accidental receipt promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oph_fpe.common_source_tower import (
    C0_RECEIPT_KEYS,
    REPORT_ARTIFACT_TYPE as COMMON_SOURCE_REPORT_ARTIFACT_TYPE,
    verify_common_source_tower_report_file,
)
from oph_fpe.core.echosahedral_federation import (
    verify_reference_federation_instrument_bundle,
)
from oph_fpe.observers.operational_verifier import (
    FINITE_RECEIPT_KEYS as OPERATIONAL_OBSERVER_RECEIPT_KEYS,
    OBSERVER_ARTIFACT_INTEGRITY_RECEIPT,
    REPORT_ARTIFACT_TYPE as OPERATIONAL_OBSERVER_REPORT_ARTIFACT_TYPE,
    verify_operational_observer_manifest,
)
from oph_fpe.repair.transaction import (
    REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE,
    verify_repair_replay_envelope,
)


EMERGENCE_LADDER_SCHEMA_VERSION = "oph.emergence-ladder/2.0.0"
EMERGENCE_LADDER_SCHEMA_URI = (
    "https://floatingpragma.io/schemas/oph-emergence-ladder-2.0.0.schema.json"
)
EMERGENCE_LADDER_ARTIFACT_TYPE = "OPH_EMERGENCE_LADDER_AUDIT"
DEFAULT_REPORT_NAME = "emergence_ladder_report.json"
DEFAULT_SCHEMA_NAME = "emergence_ladder.schema.json"

CLAIM_STATUSES = frozenset({"computed", "conditional", "missing"})


@dataclass(frozen=True)
class EvidenceRequirement:
    """One semantic requirement discharged by equivalent, narrowly typed keys."""

    requirement_id: str
    receipt_keys: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class StageSpec:
    """Static definition of one typed node in the emergence DAG."""

    stage_id: str
    title: str
    spine: str
    stage_type: str
    dependencies: tuple[str, ...]
    requirements: tuple[EvidenceRequirement, ...]
    claim_boundary: str


def _requirement(
    requirement_id: str,
    receipt_key: str,
    description: str,
    *equivalent_keys: str,
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        receipt_keys=(receipt_key, *equivalent_keys),
        description=description,
    )


# The order is topological and is part of the serialized contract.
STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        "A0",
        "Typed bounded echosahedral federation",
        "observer_repair",
        "source_architecture",
        (),
        (
            _requirement(
                "repair_primitive_replay_envelope",
                "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT",
                "the repair is reconstructed from its exact pre-state, versions, evaluator, collar, and proposals",
            ),
            _requirement(
                "canonical_federation_bundle",
                "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
                "the bounded federation is reconstructed from the canonical shared-template instrument bundle",
            ),
            _requirement(
                "carrier_conformance",
                "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
                "every instantiated carrier has the exact 12/30/20 antipode and A5 incidence contract",
            ),
            _requirement(
                "federation_sewing",
                "FEDERATION_SEWING_RECEIPT",
                "typed seams, external boundaries, and connected observer supports sew consistently",
            ),
            _requirement(
                "local_state",
                "PATCH_LOCAL_STATE_RECEIPT",
                "each patch has finite local state",
            ),
            _requirement(
                "readback",
                "PATCH_READBACK_RECEIPT",
                "the source architecture exposes a local readback operation",
            ),
        ),
        "Architecture only; it establishes no settling, geometry, gravity, or particle claim.",
    ),
    StageSpec(
        "A1",
        "Gauge-covariant overlap mismatch",
        "observer_repair",
        "overlap_constraint",
        ("A0",),
        (
            _requirement(
                "gauge_covariant_mismatch",
                "GAUGE_COVARIANT_OVERLAP_MISMATCH_RECEIPT",
                "mismatch is computed after transporting endpoint data into a common frame",
            ),
        ),
        "A covariant discrepancy observable is not yet a repair or consensus theorem.",
    ),
    StageSpec(
        "A2",
        "Proof-carrying local transactional repair",
        "observer_repair",
        "repair_dynamics",
        ("A1",),
        (
            _requirement(
                "repair_artifact_integrity",
                "REPAIR_ARTIFACT_INTEGRITY_RECEIPT",
                "the canonical repair artifact is independently reconstructed from primitive fields",
            ),
            _requirement(
                "complete_read_set",
                "COMPLETE_READ_SET_RECEIPT",
                "proposal and mismatch evaluation reads are completely traced to one snapshot",
            ),
            _requirement(
                "conflict_component_support",
                "CONFLICT_COMPONENT_SUPPORT_RECEIPT",
                "the complete read/write conflict component is reconstructed",
            ),
            _requirement(
                "atomic_union_revalidation",
                "ATOMIC_UNION_REVALIDATION_RECEIPT",
                "the union update is revalidated atomically against the frozen mismatch ledger",
            ),
            _requirement(
                "transactional_repair",
                "TRANSACTIONAL_REPAIR_RECEIPT",
                "a nonheuristic strict repair transaction commits under the exact transition contract",
            ),
        ),
        "One verified finite repair transaction does not establish obstruction freedom, confluence, records, observers, or geometry.",
    ),
    StageSpec(
        "A2O",
        "Boundary-fiber obstruction and ambiguity audit",
        "observer_repair",
        "obstruction_classification",
        ("A2",),
        (
            _requirement(
                "complete_boundary_fiber_classification",
                "COMPLETE_BOUNDARY_FIBER_CLASSIFICATION_RECEIPT",
                "every in-scope boundary fiber is classified as unrealizable, unique, ambiguous, or unknown by a complete exact exhaustor",
            ),
            _requirement(
                "higher_overlap_holonomy",
                "HIGHER_OVERLAP_HOLONOMY_AUDIT_RECEIPT",
                "cycle and higher-overlap holonomy are audited before repair",
            ),
            _requirement(
                "no_unresolved_obstruction",
                "NO_UNRESOLVED_REPAIR_OBSTRUCTION_RECEIPT",
                "no nonzero physical obstruction is hidden by a preferred representative",
            ),
            _requirement(
                "no_ambiguous_or_unknown_fibers",
                "NO_AMBIGUOUS_OR_UNKNOWN_BOUNDARY_FIBERS_RECEIPT",
                "every physically promoted boundary fiber has a unique quotient extension",
            ),
        ),
        "AMBIGUOUS and UNKNOWN are valid scientific outputs but cannot be promoted as unique repaired states.",
    ),
    StageSpec(
        "A2Q",
        "Quotient lumpability and schedule-independent normal form",
        "observer_repair",
        "quotient_dynamics",
        ("A2O",),
        (
            _requirement(
                "carrier_quotient_invariance",
                "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
                "the finite carrier contract is invariant under an independently supplied equivalent presentation",
            ),
            _requirement(
                "full_dynamical_quotient_invariance",
                "FULL_DYNAMICAL_QUOTIENT_INVARIANCE_RECEIPT",
                "repair dynamics and semantic outputs descend to the declared quotient",
            ),
            _requirement(
                "probability_lumpability",
                "EXACT_QUOTIENT_PROBABILITY_LUMPABILITY_RECEIPT",
                "all representatives induce the same exact quotient probability kernel",
            ),
            _requirement(
                "rate_lumpability",
                "EXACT_QUOTIENT_RATE_LUMPABILITY_RECEIPT",
                "all representatives induce the same exact quotient rate kernel",
            ),
            _requirement(
                "quotient_repair_confluence",
                "QUOTIENT_REPAIR_CONFLUENCE_RECEIPT",
                "complete multistart repair paths terminate in one quotient normal form independently of schedule",
            ),
        ),
        "Presentation-level agreement or one sampled schedule cannot establish quotient dynamics or a physical normal form.",
    ),
    StageSpec(
        "A3",
        "Central causal record layer",
        "observer_repair",
        "record_causality",
        ("A2Q",),
        (
            _requirement(
                "observer_artifact_integrity",
                "OBSERVER_ARTIFACT_INTEGRITY_RECEIPT",
                "all primitive operational-observer artifacts are present and independently rehashed",
            ),
            _requirement(
                "observer_contract_binding",
                "OBSERVER_CONTRACT_BINDING_RECEIPT",
                "source, transaction, evaluator, configuration, seed, and evidence share one replayed contract binding",
            ),
            _requirement(
                "observer_source_firewall",
                "OBSERVER_SOURCE_FIREWALL_RECEIPT",
                "the operational source and evaluator pass the target and runtime-metadata firewall",
            ),
            _requirement(
                "record_commit_replay",
                "A3_RECORD_COMMIT_REPLAY_RECEIPT",
                "every stable observer-facing record commit is replayed from the semantic trace",
            ),
            _requirement(
                "read_after_write_ancestry",
                "A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT",
                "every counted read has a causally prior committed write in the semantic DAG",
            ),
            _requirement(
                "readback_prediction_control",
                "A3_READBACK_PREDICTION_CONTROL_RECEIPT",
                "record readback predicts held-out outcomes above its frozen shuffled control",
            ),
        ),
        "A classical central record surface is not a noncommutative modular cap algebra.",
    ),
    StageSpec(
        "A4",
        "Operational self-reading observer",
        "observer_repair",
        "observer_system",
        ("A3",),
        (
            _requirement(
                "connected_observer_support",
                "A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT",
                "the observer token is supported on a connected carrier subfederation",
            ),
            _requirement(
                "bounded_interface",
                "A4_BOUNDED_INTERFACE_RECEIPT",
                "the observer support exposes a finite bounded interface",
            ),
            _requirement(
                "feedback_ablation",
                "A4_FEEDBACK_ABLATION_RECEIPT",
                "ablating record feedback changes later local actions under the frozen control",
            ),
            _requirement(
                "checkpoint_continuation_replay",
                "A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT",
                "checkpoint restoration exactly replays the committed observer continuation",
            ),
            _requirement(
                "operational_self_reading_observer",
                "OPERATIONAL_SELF_READING_OBSERVER_RECEIPT",
                "all finite operational self-reading clauses pass in one replayed evidence bundle",
            ),
        ),
        "An operational observer does not by bare consensus imply geometry or a physical clock.",
    ),
    StageSpec(
        "C0",
        "Typed common-domain source tower",
        "common_source",
        "common_domain_provenance",
        ("A4",),
        (
            _requirement(
                "common_domain_tower",
                "COMMON_DOMAIN_SOURCE_TOWER_RECEIPT",
                "repair, records, caps, modular data, events, stress, entropy, and scale are typed readouts of one source object",
            ),
            _requirement(
                "carrier_realization",
                "ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT",
                "the concrete federation preserves accessible algebras, port restrictions, records, repairs, checkpoints, event history, and the physical quotient of the abstract patch net",
            ),
            _requirement(
                "source_provenance_graph",
                "SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT",
                "every readout is hash-bound to its arrays, matrices, graph, evaluator, configuration, and seed",
            ),
            _requirement(
                "no_target_path",
                "SOURCE_TOWER_NO_TARGET_PATH_RECEIPT",
                "normalization, signature, coupling, scale, and terminal conclusions have no path into source generation",
            ),
            _requirement(
                "refinement_commutation",
                "SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT",
                "typed source/readout refinement squares commute exactly or under one declared envelope",
            ),
            _requirement(
                "cross_source_splice_rejection",
                "SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT",
                "cap/state and stress/entropy look-alikes from different sources fail the gate",
            ),
        ),
        "A shared run label or directory is not a common-domain proof; this is the live #572 source-typing gate.",
    ),
    StageSpec(
        "A5",
        "Icosahedral A5-equivariant refinement tower",
        "observer_repair",
        "refinement_symmetry",
        ("C0",),
        (
            _requirement(
                "true_icosahedral_tower",
                "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT",
                "the regulator is a genuine geodesic icosahedral subdivision tower",
            ),
            _requirement(
                "nested_lineage",
                "NESTED_ICOSAHEDRAL_LINEAGE_RECEIPT",
                "coarse cells, vertices, edges, and embeddings have explicit fine-level lineage",
            ),
            _requirement(
                "a5_equivariance",
                "A5_EQUIVARIANT_REFINEMENT_RECEIPT",
                "all refinement maps intertwine the exact order-60 orientation-preserving A5 action",
            ),
            _requirement(
                "carrier_refinement_naturality",
                "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
                "carrier embeddings, coarse maps, seam laws, and quotient maps form a natural refinement system",
            ),
        ),
        "A5 is upstream structure for the SM spine, never a Standard Model receipt by itself.",
    ),
    StageSpec(
        "G0",
        "Quotient-intrinsic geometric readout",
        "geometry",
        "quotient_geometry",
        ("C0",),
        (
            _requirement(
                "quotient_intrinsic_geometry",
                "QUOTIENT_INTRINSIC_GEOMETRIC_READOUT_RECEIPT",
                "geometry is read from quotient-invariant source observables",
                "QUOTIENT_GEOMETRY_CONTRACT_RECEIPT",
            ),
            _requirement(
                "geometry_target_free",
                "QUOTIENT_GEOMETRY_NO_TARGET_LEAK_RECEIPT",
                "the source generator and selector contain no target geometry",
            ),
        ),
        "A quotient geometry candidate is not a BW, Lorentz, or event-manifold receipt.",
    ),
    StageSpec(
        "G1",
        "Noncommutative prime-geometric cap state",
        "geometry",
        "operator_algebra",
        ("G0",),
        (
            _requirement(
                "noncommutative_prime_algebra",
                "NONCOMMUTATIVE_PRIME_GEOMETRIC_ALGEBRA_RECEIPT",
                "the cap-interior prime-geometric algebra is demonstrably noncommutative",
            ),
            _requirement(
                "prime_cap_state",
                "PRIME_GEOMETRIC_CAP_STATE_RECEIPT",
                "the faithful cap state is constructed only on the prime geometric algebra",
            ),
            _requirement(
                "cap_modular_hamiltonian",
                "CAP_INTERIOR_MODULAR_HAMILTONIAN_RECEIPT",
                "the modular Hamiltonian belongs to the cap-interior prime-geometric state",
            ),
        ),
        "A classical record Koopman surrogate cannot discharge this noncommutative stage.",
    ),
    StageSpec(
        "G2",
        "Finite BW certificate and independent 2pi clock",
        "geometry",
        "modular_bw_clock",
        ("G1",),
        (
            _requirement(
                "finite_bw_certificate",
                "ISSUE_308_FINITE_CAP_BW_CERTIFICATE_RECEIPT",
                "the finite cap-normal BW certificate is complete",
            ),
            _requirement(
                "independent_geometric_parameter",
                "INDEPENDENT_GEOMETRIC_CLOCK_PARAMETER_RECEIPT",
                "the geometric parameter is derived independently of modular time and pi",
            ),
            _requirement(
                "frozen_candidate_interventions",
                "CANDIDATE_SCALE_INTERVENTION_INVARIANCE_RECEIPT",
                "1x, pi, 2pi, and 4pi score the same frozen intervention rows",
            ),
            _requirement(
                "wrong_clock_separation",
                "WRONG_CLOCK_CONTROL_SEPARATION_RECEIPT",
                "2pi is selected over 1x, pi, and 4pi on held-out rows",
            ),
            _requirement(
                "direct_2pi",
                "BW_KMS_DIRECT_2PI_RECEIPT",
                "the independently parameterized BW/KMS comparison selects 2pi",
                "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT",
            ),
        ),
        "Modular KMS under an internally rescaled clock cannot select 2pi without this independent comparison.",
    ),
    StageSpec(
        "G3",
        "Lorentz structure and H3 observer-frame fiber",
        "geometry",
        "lorentz_frame_fiber",
        ("G2",),
        (
            _requirement(
                "lorentz_contract",
                "OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1",
                "the finite Lorentz contract is discharged by computed receipts",
            ),
            _requirement(
                "h3_frame_fiber",
                "H3_FRAME_FIBER_CHART_RECEIPT",
                "H3 is reconstructed and typed as the future-timelike observer-frame fiber",
            ),
            _requirement(
                "h3_curvature_identifiability",
                "H3_CURVATURE_SCALE_IDENTIFIABILITY_RECEIPT",
                "the H3 curvature scale is independently identifiable against flat controls",
            ),
        ),
        "H3 parametrizes observer frames; it is not, by itself, the event-position base.",
    ),
    StageSpec(
        "G4",
        "Semantic 3+1 event manifold",
        "geometry",
        "event_manifold",
        ("G3",),
        (
            _requirement(
                "semantic_event_identity",
                "SEMANTIC_EVENT_IDENTITY_RECEIPT",
                "event identity is derived only from semantic payload, causal parents, and committed read-after-write ancestry",
            ),
            _requirement(
                "forbidden_runtime_metadata",
                "EVENT_IDENTITY_FORBIDDEN_METADATA_AUDIT_RECEIPT",
                "worker counters, retries, queue positions, wall time, and target geometry are absent from event identity",
            ),
            _requirement(
                "event_read_after_write_ancestry",
                "EVENT_READ_AFTER_WRITE_ANCESTRY_RECEIPT",
                "semantic event ancestry is nontrivial, acyclic, and reconstructed from committed dependencies",
            ),
            _requirement(
                "event_e1_population",
                "EVENT_E1_POPULATION_DENSITY_RECEIPT",
                "E1 proves cofinal population density with shrinking covering radii",
                "E1_EVENT_POPULATION_DENSITY_RECEIPT",
            ),
            _requirement(
                "event_e2_separation",
                "EVENT_E2_SEPARATION_GAPS_RECEIPT",
                "E2 proves uniform positive separation gaps for distinct event germs",
                "E2_EVENT_SEPARATION_GAPS_RECEIPT",
            ),
            _requirement(
                "event_e3_rank_four",
                "EVENT_E3_RANK_FOUR_FRAME_RECEIPT",
                "E3 reconstructs a rank-four event chart with uniform observability constants",
                "E3_EVENT_RANK_FOUR_FRAME_RECEIPT",
            ),
            _requirement(
                "event_e4_poincare_cocycle",
                "EVENT_E4_POINCARE_COCYCLE_RECEIPT",
                "E4 includes Lorentz and translation parts and verifies overlap cocycle closure",
                "E4_EVENT_POINCARE_COCYCLE_RECEIPT",
            ),
            _requirement(
                "heldout_quadratic_cone",
                "EVENT_HELDOUT_QUADRATIC_CONE_RECEIPT",
                "a one-timelike-direction quadratic cone and time orientation are inferred on held-out event relations",
            ),
            _requirement(
                "cone_margin_tail",
                "EVENT_CONE_MARGIN_COFINAL_TAIL_RECEIPT",
                "the normalized Lorentz-cone margin has a certified positive cofinal tail",
            ),
            _requirement(
                "stable_causality",
                "EVENT_STABLE_CAUSALITY_RECEIPT",
                "a source-derived time function certifies stable causality",
            ),
            _requirement(
                "record_cauchy_refinement",
                "EVENT_RECORD_CAUCHY_REFINEMENT_RECEIPT",
                "the declared record family is Cauchy-complete under refinement",
            ),
            _requirement(
                "frame_base_separation",
                "H3_EVENT_BASE_SEPARATION_RECEIPT",
                "the H3 frame fiber and event-position base are constructed separately",
            ),
            _requirement(
                "event_manifold",
                "EVENT_MANIFOLD_3P1D_RECEIPT",
                "the combined semantic event-manifold contract is complete",
            ),
        ),
        "No H3 response, camera embedding, conformal two-sphere, or frame chart can substitute for semantic E1-E4 and a directly inferred quadratic event cone.",
    ),
    StageSpec(
        "GR0",
        "Null-stress charge",
        "gravity",
        "gravity_null_stress",
        ("G4",),
        (
            _requirement(
                "null_stress",
                "EINSTEIN_NULL_STRESS_CHARGE_RECEIPT",
                "null-deformation stress charge is source-derived",
                "E1_NULL_STRESS_CHARGE_RECEIPT",
            ),
        ),
        "This is the first gravity bridge clause, not the Einstein tensor equation.",
    ),
    StageSpec(
        "GR1",
        "Bounded interval kernel",
        "gravity",
        "gravity_interval_kernel",
        ("GR0",),
        (
            _requirement(
                "bounded_interval",
                "EINSTEIN_BOUNDED_INTERVAL_KERNEL_RECEIPT",
                "the stress/entropy response kernel has a certified bounded interval",
            ),
        ),
        "The bounded kernel is one analytic clause of the gravity bridge.",
    ),
    StageSpec(
        "GR2",
        "Fixed-cap entropy stationarity",
        "gravity",
        "gravity_entropy_stationarity",
        ("GR1",),
        (
            _requirement(
                "fixed_cap_stationarity",
                "EINSTEIN_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT",
                "entropy stationarity is evaluated at fixed geometric cap",
                "E2_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT",
            ),
        ),
        "Stationarity at a moving or target-selected cap does not discharge this node.",
    ),
    StageSpec(
        "GR3",
        "Small-ball area bridge",
        "gravity",
        "gravity_small_ball_area",
        ("GR2",),
        (
            _requirement(
                "small_ball_area",
                "EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT",
                "the small-ball entropy/area variation bridge is finite and controlled",
                "E3_SMALL_BALL_AREA_BRIDGE_RECEIPT",
            ),
        ),
        "A fitted area coefficient without the fixed-cap source chain cannot pass this node.",
    ),
    StageSpec(
        "GR4",
        "Diagonal remainder control",
        "gravity",
        "gravity_remainder",
        ("GR3",),
        (
            _requirement(
                "diagonal_remainder",
                "EINSTEIN_DIAGONAL_REMAINDER_RECEIPT",
                "finite-regulator diagonal/remainder terms satisfy the frozen bound",
            ),
            _requirement(
                "einstein_residual",
                "EINSTEIN_RESIDUAL_RECEIPT",
                "the held-out Einstein residual satisfies the declared tolerance",
            ),
        ),
        "A vanishing central fit residual alone is not a tensor upgrade.",
    ),
    StageSpec(
        "GR5",
        "Timelike coverage",
        "gravity",
        "gravity_timelike_coverage",
        ("GR4",),
        (
            _requirement(
                "timelike_coverage",
                "EINSTEIN_TIMELIKE_COVERAGE_RECEIPT",
                "the certified response covers the required timelike directions",
            ),
        ),
        "A single selected observer direction cannot discharge all-timelike coverage.",
    ),
    StageSpec(
        "GR6",
        "Common source and physical-unit binding",
        "gravity",
        "gravity_provenance_binding",
        ("GR5",),
        (
            _requirement(
                "common_source_id",
                "EINSTEIN_COMMON_SOURCE_ID_RECEIPT",
                "all gravity clauses bind to one source artifact identity",
            ),
            _requirement(
                "common_physical_unit_id",
                "EINSTEIN_COMMON_PHYSICAL_UNIT_ID_RECEIPT",
                "all dimensional clauses bind to one independently derived physical-unit identity",
            ),
            _requirement(
                "source_physical_binding",
                "EINSTEIN_SOURCE_PHYSICAL_BINDING_RECEIPT",
                "source and physical-unit identities are jointly hash-bound",
            ),
        ),
        "Numerically compatible clauses from different sources or unit systems cannot be spliced.",
    ),
    StageSpec(
        "GR7",
        "All-timelike tensor upgrade",
        "gravity",
        "gravity_tensor_upgrade",
        ("GR6",),
        (
            _requirement(
                "tensor_upgrade",
                "EINSTEIN_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT",
                "the scalar/timelike relation is upgraded to the tensor statement",
                "E4_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT",
            ),
        ),
        "Only this terminal gravity node promotes the complete finite gravity spine.",
    ),
    StageSpec(
        "SM0",
        "Target-free echosahedral screen producer laws",
        "standard_model",
        "sm_port_source_laws",
        ("C0",),
        (
            _requirement(
                "local_echosahedral_cells",
                "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT",
                "every bounded observer cell has one explicit twelve-port echosahedral interface",
            ),
            _requirement(
                "calibrated_curvature_risk",
                "CALIBRATED_CURVATURE_KL_RISK_RECEIPT",
                "the coordination-defect cost is emitted as a calibrated source readback law",
            ),
            _requirement(
                "complete_settlement",
                "GROUND_STATE_COMPLETE_SETTLEMENT_RECEIPT",
                "every nonminimum state has an escape move and all terminal states are audited",
            ),
            _requirement(
                "atomic_defect_projection",
                "ATOMIC_DEFECT_PROJECTION_RECEIPT",
                "each selected unit defect is exposed as one persistent central record port",
            ),
            _requirement(
                "pairwise_position_risk",
                "PAIRWISE_FISHER_POSITIONAL_RISK_RECEIPT",
                "the strict pair selector is emitted by target-free positional readback risk",
            ),
            _requirement(
                "edgewise_cofinal_refinement",
                "REGULAR_EDGEWISE_COFINAL_REFINEMENT_RECEIPT",
                "the physical refinement family preserves the selected ports and source laws",
            ),
            _requirement(
                "source_target_free",
                "A5_SM_SOURCE_LAWS_TARGET_FREE_RECEIPT",
                "no target group, field list, charges, family count, or 8+3+1 split enters the producer",
            ),
        ),
        "A configured icosahedral mesh or twelve local port labels cannot replace these source laws.",
    ),
    StageSpec(
        "SM1",
        "Derived twelve-port icosahedral A5 orbit",
        "standard_model",
        "sm_o_port_theorem_transform",
        ("SM0", "A5"),
        (
            _requirement(
                "port_theorem_application",
                "A5_PORT_SELECTOR_THEOREM_APPLICATION_RECEIPT",
                "a verified conditional theorem is applied to the exact hash-bound SM0 hypotheses",
            ),
            _requirement(
                "physical_port_output",
                "PHYSICAL_TWELVE_ATOMIC_A5_PORTS_RECEIPT",
                "the derived output contains twelve persistent physical ports with faithful proper A5 action",
            ),
            _requirement(
                "local_global_intertwiner",
                "LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT",
                "the collective twelve-channel screen mode is linked to the local echosahedral federation",
            ),
        ),
        "This is a derived logical-implication node; the theorem may shorten calculation but cannot supply SM0 evidence.",
    ),
    StageSpec(
        "SM2",
        "Physical local reversible-current source laws",
        "standard_model",
        "sm_current_source_laws",
        ("SM1",),
        (
            _requirement(
                "reciprocal_response",
                "MINIMAL_LOSSLESS_RECIPROCAL_RESPONSE_RECEIPT",
                "a reversible response algebra is separated from irreversible repair",
            ),
            _requirement(
                "local_rank_twelve_fibers",
                "LOCAL_RANK12_CURRENT_FIBER_PER_PATCH_RECEIPT",
                "every sufficiently refined patch carries its own rank-twelve current fiber rather than collapsing 12N local variables into twelve global generators",
            ),
            _requirement(
                "compact_connected_current_group",
                "PHYSICAL_COMPACT_CONNECTED_CURRENT_GROUP_RECEIPT",
                "the current fiber is the Lie algebra of a closed connected compact reversible subgroup",
            ),
            _requirement(
                "raw_tomography",
                "RAW_SYMMETRIC_RESPONSE_TOMOGRAPHY_RECEIPT",
                "generators are reconstructed from frozen symmetric plus/minus-time histories",
            ),
            _requirement(
                "full_rank_image_equality",
                "PORT_CURRENT_FULL_RANK_IMAGE_EQUALITY_RECEIPT",
                "the source-derived map J_v,r:P12,v->k_v,r has rank twelve and image equal to the entire current algebra",
            ),
            _requirement(
                "a5_current_equivariance",
                "PORT_CURRENT_A5_EQUIVARIANCE_RECEIPT",
                "the local P12 action and the physical current action share one source-derived A5 intertwiner",
            ),
            _requirement(
                "irrep_conditioning",
                "PORT_CURRENT_IRREP_CONDITIONING_RECEIPT",
                "the 1, 3, 3-prime, and 5 bands all have a preregistered positive minimum singular-value bound",
            ),
            _requirement(
                "bracket_from_group_commutators",
                "PORT_CURRENT_GROUP_COMMUTATOR_BRACKET_RECEIPT",
                "small reversible group commutators reconstruct a closed skew bracket instead of reusing dissipative repair generators",
            ),
            _requirement(
                "repair_current_covariance",
                "REVERSIBLE_CURRENT_REPAIR_COVARIANCE_RECEIPT",
                "dissipative repair commutes covariantly with reversible currents and is not identified with them",
            ),
            _requirement(
                "off_block_center_visibility",
                "OFF_BLOCK_CENTER_VISIBILITY_RECEIPT",
                "public probes detect the relative central current as well as the derived ideals",
            ),
            _requirement(
                "four_scale_calibration",
                "FOUR_IRREP_SCALE_CALIBRATION_RECEIPT",
                "the four A5-equivariant response scales are source calibrated rather than target chosen",
            ),
            _requirement(
                "current_refinement",
                "PORT_CURRENT_REFINEMENT_INTERTWINER_RECEIPT",
                "response generators commute with the declared refinement maps",
            ),
            _requirement(
                "overlap_current_transport",
                "PORT_CURRENT_OVERLAP_TRANSPORT_INTERTWINER_RECEIPT",
                "neighboring local current fibers glue through the same overlap transport as the carrier federation",
            ),
            _requirement(
                "classifier_route",
                "A5_CURRENT_CLASSIFIER_ROUTE_RECEIPT",
                "either physical inner A5 action, or group-level A5 action plus a stable noncentral W5 commutator, is certified",
            ),
        ),
        "Port permutation covariance and dissipative repair alone leave many brackets and cannot pass this stage; the classifier consumes one local full-rank reversible current fiber per patch.",
    ),
    StageSpec(
        "SM3",
        "Derived compact Standard-Model Lie type",
        "standard_model",
        "sm_o_current_theorem_transform",
        ("SM2",),
        (
            _requirement(
                "current_theorem_application",
                "A5_COMPACT_CURRENT_RECOGNITION_THEOREM_APPLICATION_RECEIPT",
                "the exact compact-current recognition theorem is applied to all physical current hypotheses",
            ),
            _requirement(
                "current_algebra_output",
                "PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT",
                "the output identifies the rank-twelve compact current Lie type in every local patch fiber",
                "PHYSICAL_LOCAL_SM_LIE_CURRENT_FIBER_RECEIPT",
            ),
        ),
        "The A5 module identity is not the Lie bracket; only the hypothesis-bound theorem transform may derive the Lie type.",
    ),
    StageSpec(
        "SM4",
        "Physical carrier, clock, category, deck, and line laws",
        "standard_model",
        "sm_global_source_laws",
        ("SM3",),
        (
            _requirement(
                "minimal_carrier",
                "MINIMAL_SUFFICIENT_PUBLIC_CARRIER_RECEIPT",
                "the least faithful complex carrier nontrivial on both simple ideals is source selected",
            ),
            _requirement(
                "oriented_volume_clock",
                "ORIENTED_PRIMITIVE_VOLUME_CLOCK_RECEIPT",
                "a primitive determinant volume and oriented central clock are physically read",
            ),
            _requirement(
                "source_derived_volume_clock",
                "SOURCE_DERIVED_CENTRAL_ORIENTED_VOLUME_CLOCK_RECEIPT",
                "the central oriented volume clock is independently derived from the SM-branch source and is not imported from BW/KMS normalization",
            ),
            _requirement(
                "complete_category",
                "COMPLETE_PUBLIC_TENSOR_CATEGORY_RECEIPT",
                "the public tensor category is operationally complete",
                "COMPACT_RIGID_CSTAR_TENSOR_CATEGORY_RECEIPT",
            ),
            _requirement(
                "physical_deck",
                "PHYSICAL_PORT_LOOP_COCYCLE_COFINAL_DESCENT_RECEIPT",
                "raw port-loop cocycles have exact kernel, surjectivity, and refinement-natural deck descent",
            ),
            _requirement(
                "uv_polarization",
                "UV_DEFECT_POLARIZATION_RECEIPT",
                "source disorder operators select the genuine line dressing with fusion and DSZ locality",
            ),
            _requirement(
                "current_carrier_intertwiner",
                "A5_EQUIVARIANT_CURRENT_CARRIER_INTERTWINER_RECEIPT",
                "the current output integrates into the same physical carrier and refinement system",
            ),
        ),
        "The SM reversible-current/A5 branch is independent of the BW/KMS geometry branch; a later joint measurement may compare their clocks but cannot supply either source law.",
    ),
    StageSpec(
        "SM5",
        "Derived 3+2 carrier and Z6 global form",
        "standard_model",
        "sm_o_global_theorem_transform",
        ("SM4",),
        (
            _requirement(
                "global_theorem_application",
                "A5_Z6_GLOBAL_FORM_THEOREM_APPLICATION_RECEIPT",
                "verified carrier/volume/kernel theorems are applied to exact physical hypotheses",
            ),
            _requirement(
                "global_output",
                "PHYSICAL_Z6_GLOBAL_FORM_AND_LATTICE_RECEIPT",
                "the output includes the physical quotient, character/cocharacter lattices, and selected line completion",
            ),
        ),
        "A common matter kernel or abstract Z6 is not a physical global-form selection without SM4.",
    ),
    StageSpec(
        "SM6",
        "Physical matter, scalar, family, and pole-completeness laws",
        "standard_model",
        "sm_matter_source_laws",
        ("SM5",),
        (
            _requirement(
                "spin_clifford",
                "PHYSICAL_SPIN_CLIFFORD_STATISTICS_RECEIPT",
                "the auxiliary polarized CAR/Clifford packet is bound to physical Spin and exchange",
            ),
            _requirement(
                "pole_completeness",
                "COMPLETE_ELEMENTARY_POLE_SPECTRAL_LEDGER_RECEIPT",
                "the full interpolating algebra, residues, complement gap, and refinement convergence are audited",
            ),
            _requirement(
                "scalar_completeness",
                "COMPLETE_PRIMITIVE_SCALAR_POLE_LEDGER_RECEIPT",
                "the primitive scalar domain is exhaustive before the stabilizer theorem is applied",
            ),
            _requirement(
                "family_attachment",
                "PHYSICAL_FIRST_POSITIVE_A5_BAND_FAMILY_ATTACHMENT_RECEIPT",
                "the rank-three band acts on a genuine matter pole field with zero constant-mode residue",
            ),
            _requirement(
                "three_point_functions",
                "PROJECTED_AMPUTATED_SOURCE_THREE_POINT_RECEIPT",
                "physical coupling tensors are emitted by source three-point functions",
            ),
            _requirement(
                "elementary_image_equality",
                "ELEMENTARY_IMAGE_EQUALITY_RECEIPT",
                "observer-visible elementary poles equal the producer image, excluding invisible direct-sum additions",
                "NO_EXTRA_LIGHT_SECTOR_RECEIPT",
            ),
        ),
        "Exterior rows, a graph multiplicity, or a synthetic Fock module cannot substitute for physical pole completeness.",
    ),
    StageSpec(
        "SM7",
        "Derived finite three-family one-scalar SM core (Q0)",
        "standard_model",
        "sm_o_smcore_q0_theorem_transform",
        ("SM6",),
        (
            _requirement(
                "q0_theorem_application",
                "A5_FINITE_SM_Q0_FORCING_THEOREM_APPLICATION_RECEIPT",
                "the verified finite forcing theorem is applied to the complete physical packet",
            ),
            _requirement(
                "q0_output",
                "PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT",
                "the finite output has the realized three-family one-scalar chiral package, anomalies, hypercharge, and Z6 binding",
            ),
        ),
        "Q0 is a finite physical representation/response core, not a classical or quantum continuum gauge theory.",
    ),
    StageSpec(
        "SM8",
        "Finite classical gauge regulator (Q1)",
        "standard_model",
        "sm_q1_classical_regulator",
        ("SM7",),
        (
            _requirement(
                "q1_regulator",
                "FINITE_CLASSICAL_GAUGE_REGULATOR_Q1_RECEIPT",
                "fields, local action, covariance, gauge invariance, and refinement are constructed on one carrier",
            ),
        ),
        "A finite cell skeleton without fields and an invariant action is not Q1.",
    ),
    StageSpec(
        "SM9",
        "Exact finite chiral quantum measure (Q2)",
        "standard_model",
        "sm_q2_chiral_measure",
        ("SM8",),
        (
            _requirement(
                "q2_measure",
                "EXACT_FINITE_NONABELIAN_CHIRAL_MEASURE_Q2_RECEIPT",
                "the finite nonabelian chiral measure and exact gauge invariance are proved",
            ),
        ),
        "Anomaly cancellation alone does not construct the chiral measure.",
    ),
    StageSpec(
        "SM10",
        "Formal all-orders perturbative transport (Q3)",
        "standard_model",
        "sm_q3_bv_transport",
        ("SM9",),
        (
            _requirement(
                "q3_transport",
                "FORMAL_ALL_ORDERS_BV_BRST_Q3_RECEIPT",
                "the all-orders perturbative BV/BRST transport is checked on the same carrier",
            ),
        ),
        "Q3 is formal perturbative control and does not imply a nonperturbative continuum theory.",
    ),
    StageSpec(
        "SM11",
        "Nonperturbative continuum QFT (Q4)",
        "standard_model",
        "sm_q4_continuum",
        ("SM10",),
        (
            _requirement(
                "q4_continuum",
                "NONPERTURBATIVE_OS_WIGHTMAN_CONTINUUM_Q4_RECEIPT",
                "reflection positivity, OS/Wightman axioms, and the continuum limit are constructed",
            ),
        ),
        "This is an open constructive-QFT tier and is never inferred from Q0-Q3.",
    ),
)


STAGE_BY_ID: dict[str, StageSpec] = {spec.stage_id: spec for spec in STAGE_SPECS}

OVERALL_TERMINALS: dict[str, tuple[str, ...]] = {
    "OPH_SOURCE_TO_OBSERVER_LADDER_RECEIPT": ("A4",),
    "OPH_A5_REFINEMENT_LADDER_RECEIPT": ("A5",),
    "OPH_GEOMETRY_EMERGENCE_LADDER_RECEIPT": ("G4",),
    "OPH_GRAVITY_EMERGENCE_LADDER_RECEIPT": ("GR7",),
    "OPH_FINITE_SM_Q0_EMERGENCE_LADDER_RECEIPT": ("SM7",),
    "OPH_STANDARD_MODEL_EMERGENCE_LADDER_RECEIPT": ("SM7",),
    "OPH_SM_Q1_CLASSICAL_REGULATOR_LADDER_RECEIPT": ("SM8",),
    "OPH_SM_Q2_CHIRAL_QUANTUM_LADDER_RECEIPT": ("SM9",),
    "OPH_SM_Q3_PERTURBATIVE_LADDER_RECEIPT": ("SM10",),
    "OPH_SM_Q4_CONTINUUM_LADDER_RECEIPT": ("SM11",),
    "OPH_FULL_GRAVITY_STANDARD_MODEL_LADDER_RECEIPT": ("GR7", "SM7"),
}


EMERGENCE_LADDER_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": EMERGENCE_LADDER_SCHEMA_URI,
    "title": "OPH emergence ladder audit",
    "type": "object",
    "required": [
        "$schema",
        "schema_version",
        "artifact_type",
        "run_directory",
        "source_inventory",
        "dag",
        "overall_receipts",
        "policy_checks",
        "blockers",
    ],
    "properties": {
        "$schema": {"const": EMERGENCE_LADDER_SCHEMA_URI},
        "schema_version": {"const": EMERGENCE_LADDER_SCHEMA_VERSION},
        "artifact_type": {"const": EMERGENCE_LADDER_ARTIFACT_TYPE},
        "run_directory": {"type": "string"},
        "source_inventory": {
            "type": "object",
            "required": [
                "scanned_report_paths",
                "ignored_self_artifacts",
                "input_errors",
                "registered_artifacts",
                "unadmitted_json_reports",
                "source_commitment_bindings",
            ],
        },
        "dag": {
            "type": "object",
            "required": ["stage_order", "stages", "edges"],
            "properties": {
                "stage_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "stages": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/stage"},
                },
                "edges": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/edge"},
                },
            },
        },
        "overall_receipts": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
        },
        "overall_claim_status": {"enum": sorted(CLAIM_STATUSES)},
        "policy_checks": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
        },
        "blockers": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "edge": {
            "type": "object",
            "required": ["from", "to", "edge_type"],
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "edge_type": {"const": "strict_dependency"},
            },
        },
        "stage": {
            "type": "object",
            "required": [
                "stage_id",
                "spine",
                "stage_type",
                "dependencies",
                "claim_status",
                "passed",
                "blockers",
                "source_report_paths",
                "closure_source_report_paths",
                "artifact_verifier_ids",
                "closure_artifact_verifier_ids",
                "source_bindings",
                "closure_source_bindings",
                "common_source_binding_required_paths",
                "closure_common_source_binding_required_paths",
                "common_source_binding_required",
                "common_source_binding_verified",
                "evidence",
            ],
            "properties": {
                "stage_id": {"type": "string"},
                "spine": {
                    "enum": [
                        "observer_repair",
                        "common_source",
                        "geometry",
                        "gravity",
                        "standard_model",
                    ]
                },
                "stage_type": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
                "claim_status": {"enum": sorted(CLAIM_STATUSES)},
                "passed": {"type": "boolean"},
                "blockers": {"type": "array", "items": {"type": "string"}},
                "source_report_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "closure_source_report_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "artifact_verifier_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "closure_artifact_verifier_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "source_bindings": {"type": "object"},
                "closure_source_bindings": {"type": "object"},
                "common_source_binding_required_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "closure_common_source_binding_required_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "common_source_binding_required": {"type": "boolean"},
                "common_source_binding_verified": {"type": "boolean"},
                "evidence": {"type": "object"},
            },
        },
    },
}


@dataclass(frozen=True)
class _Observation:
    key: str
    value: bool
    report_path: str
    json_pointer: str
    verifier_id: str
    source_commitment: str | None
    requires_common_source_binding: bool


@dataclass(frozen=True)
class ArtifactVerifierSpec:
    """Public description of one admissible artifact-verifier route."""

    verifier_id: str
    marker_field: str
    marker_value: str
    admitted_receipt_keys: frozenset[str]
    integrity_receipt_key: str
    common_source_semantic_roles: frozenset[str]
    requires_common_source_binding: bool


COMMON_SOURCE_VERIFIER_ID = "typed_common_source_tower_replay_v1"
FEDERATION_VERIFIER_ID = "canonical_echosahedral_federation_bundle_v1"
REPAIR_VERIFIER_ID = "proof_carrying_transaction_replay_v1"
OPERATIONAL_OBSERVER_VERIFIER_ID = "operational_observer_report_replay_v1"

FEDERATION_ADMITTED_RECEIPT_KEYS = frozenset(
    {
        "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "FEDERATION_SEWING_RECEIPT",
        "CARRIER_QUOTIENT_INVARIANCE_RECEIPT",
        "CARRIER_REFINEMENT_NATURALITY_RECEIPT",
        "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT",
        "ECHOSAHEDRAL_FEDERATION_SOURCE_INSTRUMENT_VALID",
    }
)
REPAIR_ADMITTED_RECEIPT_KEYS = frozenset(
    {
        "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT",
        "REPAIR_ARTIFACT_INTEGRITY_RECEIPT",
        "COMPLETE_READ_SET_RECEIPT",
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT",
        "ATOMIC_UNION_REVALIDATION_RECEIPT",
        "TRANSACTIONAL_REPAIR_RECEIPT",
    }
)

ARTIFACT_VERIFIER_REGISTRY: tuple[ArtifactVerifierSpec, ...] = (
    ArtifactVerifierSpec(
        verifier_id=COMMON_SOURCE_VERIFIER_ID,
        marker_field="artifact_type",
        marker_value=COMMON_SOURCE_REPORT_ARTIFACT_TYPE,
        admitted_receipt_keys=frozenset(C0_RECEIPT_KEYS),
        integrity_receipt_key="passed",
        common_source_semantic_roles=frozenset(),
        requires_common_source_binding=False,
    ),
    ArtifactVerifierSpec(
        verifier_id=FEDERATION_VERIFIER_ID,
        marker_field="schema",
        marker_value="oph.echosahedral_federation.instrument_bundle.v1",
        admitted_receipt_keys=FEDERATION_ADMITTED_RECEIPT_KEYS,
        integrity_receipt_key="INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
        common_source_semantic_roles=frozenset({"authoritative_presentation"}),
        requires_common_source_binding=False,
    ),
    ArtifactVerifierSpec(
        verifier_id=REPAIR_VERIFIER_ID,
        marker_field="artifact_type",
        marker_value=REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE,
        admitted_receipt_keys=REPAIR_ADMITTED_RECEIPT_KEYS,
        integrity_receipt_key="REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT",
        common_source_semantic_roles=frozenset({"repair_log"}),
        requires_common_source_binding=False,
    ),
    ArtifactVerifierSpec(
        verifier_id=OPERATIONAL_OBSERVER_VERIFIER_ID,
        marker_field="artifact_type",
        marker_value=OPERATIONAL_OBSERVER_REPORT_ARTIFACT_TYPE,
        admitted_receipt_keys=frozenset(OPERATIONAL_OBSERVER_RECEIPT_KEYS),
        integrity_receipt_key=OBSERVER_ARTIFACT_INTEGRITY_RECEIPT,
        common_source_semantic_roles=frozenset(),
        requires_common_source_binding=True,
    ),
)

_VERIFIER_BY_ID = {spec.verifier_id: spec for spec in ARTIFACT_VERIFIER_REGISTRY}


@dataclass(frozen=True)
class _JsonCandidate:
    path: Path
    report_path: str
    raw: bytes
    sha256: str
    payload: Any
    verifier_id: str | None


def canonical_receipt_keys() -> tuple[str, ...]:
    """Return one canonical key for every primitive ladder requirement."""

    return tuple(
        requirement.receipt_keys[0]
        for spec in STAGE_SPECS
        for requirement in spec.requirements
    )


def audit_emergence_ladder(run_dir: str | Path) -> dict[str, Any]:
    """Replay registered artifacts under ``run_dir`` and evaluate the DAG.

    Every JSON file is inventoried, but only an exact marker in
    :data:`ARTIFACT_VERIFIER_REGISTRY` selects a verifier.  Only the explicit
    allowlist of booleans returned by that verifier is admitted.  Nested or
    top-level booleans in all other JSON are retained as unadmitted diagnostics.
    Previously written ladder audits and schema files are ignored so an audit
    can never certify itself on a second pass.
    """

    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist or is not a directory: {root}")

    observations, inventory = _index_run_reports(root)
    stages: dict[str, dict[str, Any]] = {}
    for spec in STAGE_SPECS:
        stages[spec.stage_id] = _evaluate_stage(spec, stages, observations)

    edges = [
        {"from": dependency, "to": spec.stage_id, "edge_type": "strict_dependency"}
        for spec in STAGE_SPECS
        for dependency in spec.dependencies
    ]
    overall_receipts = {
        name: all(stages[stage_id]["passed"] is True for stage_id in terminal_ids)
        for name, terminal_ids in OVERALL_TERMINALS.items()
    }
    inventory_clean = not inventory["input_errors"]
    overall_receipts["OPH_EMERGENCE_LADDER_SOURCE_INTEGRITY_RECEIPT"] = inventory_clean
    overall_receipts["OPH_EMERGENCE_LADDER_RECEIPT"] = bool(
        inventory_clean
        and overall_receipts["OPH_FULL_GRAVITY_STANDARD_MODEL_LADDER_RECEIPT"]
    )

    policy_checks = _policy_checks()
    stage_blockers = [
        f"{stage_id}:{blocker}"
        for stage_id, stage in stages.items()
        for blocker in stage["blockers"]
    ]
    input_blockers = [
        f"source_inventory:{error}" for error in inventory["input_errors"]
    ]
    blockers = sorted(set(stage_blockers + input_blockers))
    if overall_receipts["OPH_EMERGENCE_LADDER_RECEIPT"]:
        overall_status = "computed"
    elif any(stage["claim_status"] == "missing" for stage in stages.values()):
        overall_status = "missing"
    else:
        overall_status = "conditional"

    report = {
        "$schema": EMERGENCE_LADDER_SCHEMA_URI,
        "schema_version": EMERGENCE_LADDER_SCHEMA_VERSION,
        "artifact_type": EMERGENCE_LADDER_ARTIFACT_TYPE,
        "run_directory": str(root.resolve()),
        "source_inventory": inventory,
        "dag": {
            "stage_order": [spec.stage_id for spec in STAGE_SPECS],
            "stages": stages,
            "edges": edges,
        },
        "overall_receipts": overall_receipts,
        "overall_claim_status": overall_status,
        "policy_checks": policy_checks,
        "blockers": blockers,
        "claim_boundary": (
            "This artifact audits finite simulator receipts and their dependency provenance. "
            "It is neither a continuum proof nor an empirical validation of gravity or the "
            "Standard Model. A terminal receipt is true only when every strict upstream stage "
            "has independently computed evidence."
        ),
    }
    validation_errors = validate_emergence_ladder_report(report)
    if validation_errors:
        raise RuntimeError(
            f"internal emergence ladder schema error: {validation_errors}"
        )
    return report


def write_emergence_ladder_report(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit ``run_dir`` and write a deterministic JSON report and optional schema."""

    root = Path(run_dir)
    report = audit_emergence_ladder(root)
    target = (
        Path(output_path) if output_path is not None else root / DEFAULT_REPORT_NAME
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if schema_path is not None:
        write_emergence_ladder_schema(schema_path)
    return report


def write_emergence_ladder_schema(path: str | Path) -> Path:
    """Write the bundled JSON Schema used by ladder consumers."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(EMERGENCE_LADDER_JSON_SCHEMA, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def validate_emergence_ladder_report(report: Mapping[str, Any]) -> list[str]:
    """Perform dependency-aware validation without requiring ``jsonschema``.

    The bundled JSON Schema validates serialization shape.  This validator adds
    the invariants JSON Schema cannot conveniently express: exact stage order,
    dependency closure, status/pass consistency, and terminal recomputation.
    """

    errors: list[str] = []
    if report.get("$schema") != EMERGENCE_LADDER_SCHEMA_URI:
        errors.append("schema_uri_mismatch")
    if report.get("schema_version") != EMERGENCE_LADDER_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if report.get("artifact_type") != EMERGENCE_LADDER_ARTIFACT_TYPE:
        errors.append("artifact_type_mismatch")
    dag = report.get("dag")
    if not isinstance(dag, Mapping):
        return [*errors, "dag_missing"]
    expected_order = [spec.stage_id for spec in STAGE_SPECS]
    if dag.get("stage_order") != expected_order:
        errors.append("stage_order_mismatch")
    stages = dag.get("stages")
    if not isinstance(stages, Mapping):
        return [*errors, "stages_missing"]
    if set(stages) != set(expected_order):
        errors.append("stage_id_set_mismatch")
    for spec in STAGE_SPECS:
        stage = stages.get(spec.stage_id)
        if not isinstance(stage, Mapping):
            errors.append(f"{spec.stage_id}:stage_missing")
            continue
        if stage.get("dependencies") != list(spec.dependencies):
            errors.append(f"{spec.stage_id}:dependency_list_mismatch")
        if stage.get("claim_status") not in CLAIM_STATUSES:
            errors.append(f"{spec.stage_id}:invalid_claim_status")
        if not isinstance(stage.get("passed"), bool):
            errors.append(f"{spec.stage_id}:passed_not_boolean")
        if stage.get("passed") is True and stage.get("claim_status") != "computed":
            errors.append(f"{spec.stage_id}:passed_without_computed_status")
        if not isinstance(stage.get("common_source_binding_required"), bool):
            errors.append(f"{spec.stage_id}:binding_required_not_boolean")
        if not isinstance(stage.get("common_source_binding_verified"), bool):
            errors.append(f"{spec.stage_id}:binding_verified_not_boolean")
        if (
            stage.get("passed") is True
            and stage.get("common_source_binding_verified") is not True
        ):
            errors.append(f"{spec.stage_id}:passed_without_common_source_binding")
        closure_paths = stage.get("closure_source_report_paths")
        closure_bindings = stage.get("closure_source_bindings")
        if not isinstance(closure_paths, list) or not isinstance(
            closure_bindings, Mapping
        ):
            errors.append(f"{spec.stage_id}:closure_source_binding_shape_invalid")
        elif closure_paths != sorted(closure_bindings):
            errors.append(f"{spec.stage_id}:closure_source_binding_paths_mismatch")
        closure_required_paths = stage.get(
            "closure_common_source_binding_required_paths"
        )
        if not isinstance(closure_required_paths, list) or not all(
            isinstance(value, str) for value in closure_required_paths
        ):
            errors.append(f"{spec.stage_id}:required_binding_paths_invalid")
        elif isinstance(closure_paths, list):
            expected_required = bool(len(closure_paths) > 1 or closure_required_paths)
            if stage.get("common_source_binding_required") is not expected_required:
                errors.append(f"{spec.stage_id}:binding_required_recompute_mismatch")
        for dependency in spec.dependencies:
            upstream = stages.get(dependency)
            if stage.get("passed") is True and (
                not isinstance(upstream, Mapping) or upstream.get("passed") is not True
            ):
                errors.append(
                    f"{spec.stage_id}:passed_with_failed_dependency:{dependency}"
                )
    overall = report.get("overall_receipts")
    if not isinstance(overall, Mapping):
        errors.append("overall_receipts_missing")
    else:
        for name, terminal_ids in OVERALL_TERMINALS.items():
            expected = all(
                isinstance(stages.get(stage_id), Mapping)
                and stages[stage_id].get("passed") is True
                for stage_id in terminal_ids
            )
            if overall.get(name) is not expected:
                errors.append(f"overall_receipt_mismatch:{name}")
    if report.get("overall_claim_status") not in CLAIM_STATUSES:
        errors.append("invalid_overall_claim_status")
    policy = report.get("policy_checks")
    if not isinstance(policy, Mapping) or not all(
        value is True for value in policy.values()
    ):
        errors.append("policy_checks_failed")
    return errors


def _index_run_reports(
    root: Path,
) -> tuple[dict[str, list[_Observation]], dict[str, Any]]:
    candidates: list[_JsonCandidate] = []
    scanned: list[str] = []
    ignored: list[str] = []
    errors: list[str] = []
    hashes: dict[str, str] = {}
    unadmitted: list[dict[str, Any]] = []
    registered_rows: list[dict[str, Any]] = []
    common_source_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
            payload = _strict_json_loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{relative}:unreadable_json:{type(exc).__name__}")
            continue
        if _is_self_artifact(payload):
            ignored.append(relative)
            continue
        scanned.append(relative)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        verifier_id = _select_artifact_verifier(payload)
        candidates.append(
            _JsonCandidate(
                path=path,
                report_path=relative,
                raw=raw,
                sha256=hashes[relative],
                payload=payload,
                verifier_id=verifier_id,
            )
        )

    index: dict[str, list[_Observation]] = {}
    # A verified common-source report authenticates the exact byte hashes of
    # its declared artifacts.  This gives later registry entries the only
    # admissible route to a common bundle commitment.
    commitment_bindings_by_artifact_sha: dict[
        str, set[tuple[str, str, str]]
    ] = {}
    verified_common_report_commitments: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.verifier_id != COMMON_SOURCE_VERIFIER_ID:
            continue
        validation = verify_common_source_tower_report_file(candidate.path)
        validation_blockers = list(validation.get("blockers") or [])
        recomputed = validation.get("recomputed_report")
        commitment = (
            recomputed.get("computed_bundle_commitment")
            if isinstance(recomputed, Mapping)
            else None
        )
        if not isinstance(commitment, str) or not commitment:
            commitment = None
        passed = validation.get("passed") is True
        admitted_commitment = commitment if passed else None
        row = {
            "registry_id": COMMON_SOURCE_VERIFIER_ID,
            "report_path": candidate.report_path,
            "integrity_receipt_key": "passed",
            "passed": passed,
            "source_commitment": admitted_commitment,
            "admitted_receipt_keys": sorted(C0_RECEIPT_KEYS) if passed else [],
            "blockers": validation_blockers,
        }
        common_source_rows.append(
            {
                "report_path": candidate.report_path,
                "passed": passed,
                "blockers": validation_blockers,
            }
        )
        registered_rows.append(row)
        if not passed:
            suffixes = validation_blockers or ["unspecified"]
            errors.extend(
                f"{candidate.report_path}:common_source_verification_failed:{item}"
                for item in suffixes
            )
            continue
        if not isinstance(recomputed, Mapping) or commitment is None:
            errors.append(
                f"{candidate.report_path}:common_source_verifier_missing_commitment"
            )
            continue
        verified_common_report_commitments.setdefault(candidate.sha256, set()).add(
            commitment
        )
        for key in sorted(C0_RECEIPT_KEYS):
            value = recomputed.get(key)
            if type(value) is not bool:
                errors.append(
                    f"{candidate.report_path}:verifier_non_boolean_output:{key}"
                )
                continue
            index.setdefault(key, []).append(
                _Observation(
                    key=key,
                    value=value,
                    report_path=candidate.report_path,
                    json_pointer=f"/recomputed_report/{_pointer_escape(key)}",
                    verifier_id=COMMON_SOURCE_VERIFIER_ID,
                    source_commitment=commitment,
                    requires_common_source_binding=False,
                )
            )
        verification = recomputed.get("artifact_verification")
        verification_rows = (
            verification.get("rows") if isinstance(verification, Mapping) else None
        )
        if isinstance(verification_rows, list):
            for artifact_row in verification_rows:
                if not isinstance(artifact_row, Mapping):
                    continue
                actual = _normalize_sha256(artifact_row.get("actual_sha256"))
                semantic_role = artifact_row.get("semantic_role")
                artifact_id = artifact_row.get("artifact_id")
                if (
                    actual is not None
                    and artifact_row.get("passed") is True
                    and isinstance(semantic_role, str)
                    and isinstance(artifact_id, str)
                ):
                    commitment_bindings_by_artifact_sha.setdefault(
                        actual, set()
                    ).add((commitment, semantic_role, artifact_id))

    federation_verifications = {
        candidate.report_path: verify_reference_federation_instrument_bundle(
            candidate.payload
        )
        for candidate in candidates
        if candidate.verifier_id == FEDERATION_VERIFIER_ID
    }
    valid_federation_hashes = {
        candidate.sha256
        for candidate in candidates
        if candidate.verifier_id == FEDERATION_VERIFIER_ID
        and all(
            federation_verifications[candidate.report_path].get(key) is True
            for key in (
                "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
                "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
                "FEDERATION_SEWING_RECEIPT",
            )
        )
    }
    repair_verifications = {
        candidate.report_path: verify_repair_replay_envelope(candidate.payload)
        for candidate in candidates
        if candidate.verifier_id == REPAIR_VERIFIER_ID
    }
    valid_repair_hashes = {
        candidate.sha256
        for candidate in candidates
        if candidate.verifier_id == REPAIR_VERIFIER_ID
        and repair_verifications[candidate.report_path].get(
            "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT"
        )
        is True
    }

    for candidate in candidates:
        if candidate.verifier_id is None:
            unadmitted.append(_unadmitted_json_row(candidate))
            continue
        if candidate.verifier_id == COMMON_SOURCE_VERIFIER_ID:
            continue
        spec = _VERIFIER_BY_ID[candidate.verifier_id]
        if candidate.verifier_id == FEDERATION_VERIFIER_ID:
            verification = federation_verifications[candidate.report_path]
            blocker_values = verification.get("blockers") or []
        elif candidate.verifier_id == REPAIR_VERIFIER_ID:
            verification = repair_verifications[candidate.report_path]
            blocker_values = verification.get("failure_reasons") or []
        elif candidate.verifier_id == OPERATIONAL_OBSERVER_VERIFIER_ID:
            replay = _replay_operational_observer_report(candidate)
            recomputed_observer = replay.get("recomputed_report")
            if not isinstance(recomputed_observer, Mapping):
                recomputed_observer = {}
            verification = {
                key: (
                    recomputed_observer.get(key)
                    if replay.get("passed") is True
                    else False
                )
                for key in spec.admitted_receipt_keys
            }
            blocker_values = [
                *(replay.get("blockers") or []),
                *(recomputed_observer.get("blockers") or []),
            ]
        else:  # pragma: no cover - registry and dispatcher are defined together
            raise RuntimeError(f"unhandled verifier: {candidate.verifier_id}")
        integrity = verification.get(spec.integrity_receipt_key)
        if type(integrity) is not bool:
            errors.append(
                f"{candidate.report_path}:verifier_non_boolean_output:"
                f"{spec.integrity_receipt_key}"
            )
            integrity = False
        observer_parent_binding: dict[str, Any] | None = None
        if candidate.verifier_id == OPERATIONAL_OBSERVER_VERIFIER_ID:
            contract = (
                recomputed_observer.get("contract_binding")
                if isinstance(recomputed_observer, Mapping)
                else None
            )
            source_parent_sha = _normalize_sha256(
                contract.get("source_bundle_receipt_hash")
                if isinstance(contract, Mapping)
                else None
            )
            repair_parent_sha = _normalize_sha256(
                contract.get("canonical_repair_artifact_hash")
                if isinstance(contract, Mapping)
                else None
            )
            federation_parent_sha = _normalize_sha256(
                contract.get("federation_bundle_receipt_hash")
                if isinstance(contract, Mapping)
                else None
            )
            source_parent_commitments = (
                verified_common_report_commitments.get(source_parent_sha, set())
                if source_parent_sha is not None
                else set()
            )
            repair_parent_rows = (
                commitment_bindings_by_artifact_sha.get(
                    repair_parent_sha, set()
                )
                if repair_parent_sha is not None
                else set()
            )
            repair_role_bindings = {
                row for row in repair_parent_rows if row[1] == "repair_log"
            }
            repair_parent_commitments = {row[0] for row in repair_role_bindings}
            federation_parent_rows = (
                commitment_bindings_by_artifact_sha.get(
                    federation_parent_sha, set()
                )
                if federation_parent_sha is not None
                else set()
            )
            federation_role_bindings = {
                row
                for row in federation_parent_rows
                if row[1] == "authoritative_presentation"
            }
            federation_parent_commitments = {
                row[0] for row in federation_role_bindings
            }
            role_bindings = repair_role_bindings | federation_role_bindings
            source_commitment = None
            if replay.get("passed") is not True or integrity is not True:
                binding_status = "observer_report_replay_or_integrity_failed"
            elif source_parent_sha is None or not source_parent_commitments:
                binding_status = "observer_source_parent_is_not_verified_c0_report"
            elif (
                repair_parent_sha is None
                or repair_parent_sha not in valid_repair_hashes
            ):
                binding_status = "observer_repair_parent_is_not_verified_repair"
            elif (
                federation_parent_sha is None
                or federation_parent_sha not in valid_federation_hashes
            ):
                binding_status = "observer_federation_parent_is_not_verified_bundle"
            elif not repair_parent_commitments:
                binding_status = "observer_repair_parent_not_bound_as_repair_log"
            elif not federation_parent_commitments:
                binding_status = (
                    "observer_federation_parent_not_bound_as_authoritative_presentation"
                )
            elif (
                len(source_parent_commitments) != 1
                or len(repair_parent_commitments) != 1
                or len(federation_parent_commitments) != 1
            ):
                binding_status = "observer_parent_commitment_is_ambiguous"
            elif not (
                source_parent_commitments
                == repair_parent_commitments
                == federation_parent_commitments
            ):
                binding_status = "observer_parent_commitments_disagree"
            else:
                source_commitment = next(iter(source_parent_commitments))
                binding_status = "observer_parents_bound_to_verified_common_source"
            observer_parent_binding = {
                "source_bundle_receipt_sha256": source_parent_sha,
                "canonical_repair_artifact_sha256": repair_parent_sha,
                "federation_bundle_receipt_sha256": federation_parent_sha,
                "source_parent_commitments": sorted(source_parent_commitments),
                "repair_parent_commitments": sorted(repair_parent_commitments),
                "federation_parent_commitments": sorted(
                    federation_parent_commitments
                ),
                "repair_parent_is_verified_repair": bool(
                    repair_parent_sha in valid_repair_hashes
                    if repair_parent_sha is not None
                    else False
                ),
                "federation_parent_is_verified_bundle": bool(
                    federation_parent_sha in valid_federation_hashes
                    if federation_parent_sha is not None
                    else False
                ),
            }
        else:
            all_hash_bindings = commitment_bindings_by_artifact_sha.get(
                candidate.sha256, set()
            )
            role_bindings = {
                row
                for row in all_hash_bindings
                if row[1] in spec.common_source_semantic_roles
            }
            commitments = {row[0] for row in role_bindings}
            source_commitment = (
                next(iter(commitments)) if len(commitments) == 1 else None
            )
            if len(commitments) > 1:
                binding_status = "ambiguous_multiple_source_commitments"
            elif all_hash_bindings and not role_bindings:
                binding_status = "hash_bound_under_wrong_semantic_role"
            elif source_commitment is None:
                binding_status = "not_bound_by_verified_common_source_manifest"
            else:
                binding_status = "bound_by_verified_common_source_manifest"
        admitted_keys: list[str] = []
        for key in sorted(spec.admitted_receipt_keys):
            value = verification.get(key)
            if type(value) is not bool:
                errors.append(
                    f"{candidate.report_path}:verifier_non_boolean_output:{key}"
                )
                continue
            admitted_keys.append(key)
            index.setdefault(key, []).append(
                _Observation(
                    key=key,
                    value=value,
                    report_path=candidate.report_path,
                    json_pointer=f"/verifier_output/{_pointer_escape(key)}",
                    verifier_id=spec.verifier_id,
                    source_commitment=source_commitment,
                    requires_common_source_binding=(
                        spec.requires_common_source_binding
                    ),
                )
            )
        blockers = [str(value) for value in blocker_values]
        registered_rows.append(
            {
                "registry_id": spec.verifier_id,
                "report_path": candidate.report_path,
                "integrity_receipt_key": spec.integrity_receipt_key,
                "passed": integrity is True,
                "source_commitment": source_commitment,
                "source_binding_status": binding_status,
                "requires_common_source_binding": (
                    spec.requires_common_source_binding
                ),
                "observer_parent_binding": observer_parent_binding,
                "expected_common_source_semantic_roles": sorted(
                    spec.common_source_semantic_roles
                ),
                "matched_common_source_artifact_ids": sorted(
                    row[2] for row in role_bindings
                ),
                "admitted_receipt_keys": admitted_keys,
                "blockers": blockers,
            }
        )
        if integrity is not True:
            suffixes = blockers or ["integrity_receipt_false"]
            errors.extend(
                f"{candidate.report_path}:{spec.verifier_id}_failed:{item}"
                for item in suffixes
            )
    digest = hashlib.sha256()
    for relative in scanned:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashes[relative].encode("ascii"))
        digest.update(b"\n")
    inventory = {
        "scanned_report_paths": scanned,
        "ignored_self_artifacts": ignored,
        "input_errors": errors,
        "report_sha256": hashes,
        "inventory_sha256": digest.hexdigest(),
        "common_source_tower_reports": common_source_rows,
        "registered_artifacts": registered_rows,
        "unadmitted_json_reports": unadmitted,
        "source_commitment_bindings": {
            row["report_path"]: row.get("source_commitment")
            for row in registered_rows
        },
    }
    return index, inventory


def _select_artifact_verifier(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    matches = [
        spec.verifier_id
        for spec in ARTIFACT_VERIFIER_REGISTRY
        if payload.get(spec.marker_field) == spec.marker_value
    ]
    if len(matches) > 1:
        # No current marker set overlaps.  Treat future overlap as unregistered
        # rather than choosing an order-dependent verifier.
        return None
    return matches[0] if matches else None


def _strict_json_loads(raw: bytes) -> Any:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite_json_constant:{value}")

    return json.loads(raw, object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def _replay_operational_observer_report(
    candidate: _JsonCandidate,
) -> dict[str, Any]:
    """Replay the sibling manifest and type-strictly compare its full report."""

    payload = candidate.payload
    blockers: list[str] = []
    if not isinstance(payload, Mapping):
        return {
            "passed": False,
            "blockers": ["observer_report_root_must_be_object"],
            "recomputed_report": None,
        }
    manifest_name = payload.get("manifest_path")
    if not isinstance(manifest_name, str) or not manifest_name:
        return {
            "passed": False,
            "blockers": ["observer_report_manifest_path_missing"],
            "recomputed_report": None,
        }
    relative = Path(manifest_name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        return {
            "passed": False,
            "blockers": ["observer_report_manifest_path_unsafe"],
            "recomputed_report": None,
        }
    manifest_path = candidate.path.parent / relative
    recomputed = verify_operational_observer_manifest(manifest_path)
    try:
        stored_bytes = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        recomputed_bytes = json.dumps(
            recomputed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        blockers.append(f"observer_report_not_canonical_json:{type(exc).__name__}")
    else:
        if stored_bytes != recomputed_bytes:
            blockers.append("stored_observer_report_not_exact_verifier_output")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "recomputed_report": recomputed,
    }


def _normalize_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.removeprefix("sha256:")
    if len(candidate) != 64 or candidate.lower() != candidate:
        return None
    try:
        int(candidate, 16)
    except ValueError:
        return None
    return candidate


def _unadmitted_json_row(candidate: _JsonCandidate) -> dict[str, Any]:
    receipt_claims = [
        {
            "receipt_key": key,
            "value": value,
            "json_pointer": pointer,
        }
        for key, value, pointer in _walk_named_values(candidate.payload)
        if key.endswith("_RECEIPT") and type(value) is bool
    ]
    payload = candidate.payload
    return {
        "report_path": candidate.report_path,
        "schema": payload.get("schema") if isinstance(payload, Mapping) else None,
        "artifact_type": (
            payload.get("artifact_type") if isinstance(payload, Mapping) else None
        ),
        "claimed_receipt_booleans": receipt_claims,
        "admitted": False,
        "reason": "no_exact_artifact_verifier_registry_match",
    }


def _is_self_artifact(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return bool(
        payload.get("artifact_type") == EMERGENCE_LADDER_ARTIFACT_TYPE
        or payload.get("$id") == EMERGENCE_LADDER_SCHEMA_URI
    )


def _walk_named_values(
    payload: Any, pointer: str = ""
) -> Iterable[tuple[str, Any, str]]:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            child = f"{pointer}/{_pointer_escape(key)}"
            yield key, value, child
            yield from _walk_named_values(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child = f"{pointer}/{index}"
            yield from _walk_named_values(value, child)


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _evaluate_stage(
    spec: StageSpec,
    prior_stages: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Sequence[_Observation]],
) -> dict[str, Any]:
    dependency_blockers = [
        f"dependency_not_computed:{dependency}"
        for dependency in spec.dependencies
        if prior_stages.get(dependency, {}).get("passed") is not True
    ]
    evidence: dict[str, dict[str, Any]] = {}
    evidence_blockers: list[str] = []
    source_paths: set[str] = set()
    verifier_ids: set[str] = set()
    source_bindings: dict[str, str | None] = {}
    required_binding_paths: set[str] = set()
    any_missing = False
    all_evidence_passed = True
    for requirement in spec.requirements:
        result = _evaluate_requirement(requirement, observations)
        evidence[requirement.requirement_id] = result
        source_paths.update(result["source_report_paths"])
        verifier_ids.update(result["artifact_verifier_ids"])
        source_bindings.update(result["source_bindings"])
        required_binding_paths.update(
            result["required_common_source_binding_paths"]
        )
        evidence_blockers.extend(result["blockers"])
        any_missing = any_missing or result["claim_status"] == "missing"
        all_evidence_passed = all_evidence_passed and result["passed"] is True
    closure_bindings = dict(source_bindings)
    closure_verifier_ids = set(verifier_ids)
    closure_required_binding_paths = set(required_binding_paths)
    for dependency in spec.dependencies:
        upstream = prior_stages.get(dependency, {})
        upstream_bindings = upstream.get("closure_source_bindings", {})
        if isinstance(upstream_bindings, Mapping):
            for path, commitment in upstream_bindings.items():
                existing = closure_bindings.get(str(path), commitment)
                closure_bindings[str(path)] = (
                    commitment if existing == commitment else None
                )
        upstream_verifiers = upstream.get("closure_artifact_verifier_ids", [])
        if isinstance(upstream_verifiers, list):
            closure_verifier_ids.update(
                value for value in upstream_verifiers if isinstance(value, str)
            )
        upstream_required_paths = upstream.get(
            "closure_common_source_binding_required_paths", []
        )
        if isinstance(upstream_required_paths, list):
            closure_required_binding_paths.update(
                value for value in upstream_required_paths if isinstance(value, str)
            )
    binding_required = bool(
        len(closure_bindings) > 1 or closure_required_binding_paths
    )
    unbound_paths = sorted(
        path for path, commitment in closure_bindings.items() if commitment is None
    )
    bound_commitments = {
        commitment
        for commitment in closure_bindings.values()
        if isinstance(commitment, str)
    }
    binding_blockers: list[str] = []
    if binding_required and unbound_paths:
        binding_blockers.append(
            "common_source_commitment_unbound:" + ",".join(unbound_paths)
        )
    if binding_required and len(bound_commitments) > 1:
        binding_blockers.append("common_source_commitment_mismatch")
    binding_verified = bool(
        not binding_required
        or (not unbound_paths and len(bound_commitments) == 1)
    )
    passed = bool(
        not dependency_blockers
        and not binding_blockers
        and all_evidence_passed
    )
    if passed:
        claim_status = "computed"
    elif any_missing:
        claim_status = "missing"
    else:
        claim_status = "conditional"
    return {
        "stage_id": spec.stage_id,
        "title": spec.title,
        "spine": spec.spine,
        "stage_type": spec.stage_type,
        "dependencies": list(spec.dependencies),
        "claim_status": claim_status,
        "passed": passed,
        "blockers": sorted(
            set(dependency_blockers + binding_blockers + evidence_blockers)
        ),
        "source_report_paths": sorted(source_paths),
        "closure_source_report_paths": sorted(closure_bindings),
        "artifact_verifier_ids": sorted(verifier_ids),
        "closure_artifact_verifier_ids": sorted(closure_verifier_ids),
        "source_bindings": dict(sorted(source_bindings.items())),
        "closure_source_bindings": dict(sorted(closure_bindings.items())),
        "common_source_binding_required_paths": sorted(required_binding_paths),
        "closure_common_source_binding_required_paths": sorted(
            closure_required_binding_paths
        ),
        "common_source_binding_required": binding_required,
        "common_source_binding_verified": binding_verified,
        "evidence": evidence,
        "claim_boundary": spec.claim_boundary,
    }


def _evaluate_requirement(
    requirement: EvidenceRequirement,
    observations: Mapping[str, Sequence[_Observation]],
) -> dict[str, Any]:
    rows = [
        observation
        for key in requirement.receipt_keys
        for observation in observations.get(key, ())
    ]
    if not rows:
        expected = "|".join(requirement.receipt_keys)
        return {
            "requirement_id": requirement.requirement_id,
            "description": requirement.description,
            "accepted_receipt_keys": list(requirement.receipt_keys),
            "claim_status": "missing",
            "passed": False,
            "observations": [],
            "source_report_paths": [],
            "artifact_verifier_ids": [],
            "source_bindings": {},
            "required_common_source_binding_paths": [],
            "blockers": [
                f"missing_receipt:{requirement.requirement_id}:expected_one_of={expected}"
            ],
        }

    serialized_rows = [
        {
            "receipt_key": row.key,
            "value": row.value,
            "value_type": type(row.value).__name__,
            "report_path": row.report_path,
            "json_pointer": row.json_pointer,
            "artifact_verifier_id": row.verifier_id,
            "source_commitment": row.source_commitment,
            "requires_common_source_binding": (
                row.requires_common_source_binding
            ),
        }
        for row in rows
    ]
    non_boolean = [row for row in rows if type(row.value) is not bool]
    false_rows = [row for row in rows if row.value is False]
    true_rows = [row for row in rows if row.value is True]
    blockers: list[str] = []
    for row in non_boolean:
        blockers.append(
            f"non_boolean_receipt:{requirement.requirement_id}:{row.report_path}:{row.json_pointer}"
        )
    for row in false_rows:
        blockers.append(
            f"false_receipt:{requirement.requirement_id}:{row.report_path}:{row.json_pointer}"
        )
    if true_rows and false_rows:
        blockers.append(f"contradictory_receipts:{requirement.requirement_id}")
    passed = bool(true_rows and not non_boolean and not false_rows)
    return {
        "requirement_id": requirement.requirement_id,
        "description": requirement.description,
        "accepted_receipt_keys": list(requirement.receipt_keys),
        "claim_status": "computed" if passed else "conditional",
        "passed": passed,
        "observations": serialized_rows,
        "source_report_paths": sorted({row.report_path for row in rows}),
        "artifact_verifier_ids": sorted({row.verifier_id for row in rows}),
        "source_bindings": {
            row.report_path: row.source_commitment
            for row in sorted(rows, key=lambda item: item.report_path)
        },
        "required_common_source_binding_paths": sorted(
            {
                row.report_path
                for row in rows
                if row.requires_common_source_binding
            }
        ),
        "blockers": sorted(set(blockers)),
    }


def _policy_checks() -> dict[str, bool]:
    """Recompute structural anti-promotion rules from the static specifications."""

    sm0 = STAGE_BY_ID["SM0"]
    sm1 = STAGE_BY_ID["SM1"]
    sm4 = STAGE_BY_ID["SM4"]
    sm7 = STAGE_BY_ID["SM7"]
    g0 = STAGE_BY_ID["G0"]
    g4 = STAGE_BY_ID["G4"]
    gr0 = STAGE_BY_ID["GR0"]
    sm0_requirement_ids = {item.requirement_id for item in sm0.requirements}
    sm1_requirement_ids = {item.requirement_id for item in sm1.requirements}
    semantic_ids = {requirement.requirement_id for requirement in g4.requirements}
    admitted_registry_keys = {
        key
        for verifier in ARTIFACT_VERIFIER_REGISTRY
        for key in verifier.admitted_receipt_keys
    }
    sm4_receipt_keys = {
        key for requirement in sm4.requirements for key in requirement.receipt_keys
    }
    return {
        "A5_NECESSARY_BUT_NOT_SUFFICIENT_FOR_SM": bool(
            sm0.dependencies == ("C0",)
            and sm1.dependencies == ("SM0", "A5")
            and "physical_port_output" in sm1_requirement_ids
            and len(sm0_requirement_ids) >= 7
            and OVERALL_TERMINALS["OPH_STANDARD_MODEL_EMERGENCE_LADDER_RECEIPT"]
            == ("SM7",)
        ),
        "THEOREM_SHORTCUTS_CANNOT_PROMOTE_SOURCE_LAWS": bool(
            sm0.stage_type == "sm_port_source_laws"
            and "theorem" not in sm0.stage_type
            and sm1.stage_type.endswith("theorem_transform")
            and sm7.stage_type.endswith("theorem_transform")
        ),
        "SM_AND_BW_CLOCK_BRANCHES_ARE_INDEPENDENT": bool(
            sm4.dependencies == ("SM3",)
            and "SOURCE_DERIVED_CENTRAL_ORIENTED_VOLUME_CLOCK_RECEIPT"
            in sm4_receipt_keys
            and "BW_TO_CENTRAL_VOLUME_CLOCK_INTERTWINER_RECEIPT"
            not in sm4_receipt_keys
        ),
        "SM_Q0_Q1_Q2_Q3_Q4_TIERS_DO_NOT_COLLAPSE": bool(
            STAGE_BY_ID["SM8"].dependencies == ("SM7",)
            and STAGE_BY_ID["SM9"].dependencies == ("SM8",)
            and STAGE_BY_ID["SM10"].dependencies == ("SM9",)
            and STAGE_BY_ID["SM11"].dependencies == ("SM10",)
        ),
        "BARE_CONSENSUS_CANNOT_PROMOTE_GEOMETRY": bool(
            g0.dependencies == ("C0",)
            and "FINITE_CONSENSUS_THEOREM_RECEIPT"
            not in {
                key
                for requirement in g0.requirements
                for key in requirement.receipt_keys
            }
        ),
        "BARE_CONSENSUS_CANNOT_PROMOTE_GRAVITY": gr0.dependencies == ("G4",),
        "H3_FRAME_FIBER_CANNOT_PROMOTE_EVENT_MANIFOLD": bool(
            g4.dependencies == ("G3",)
            and {
                "semantic_event_identity",
                "forbidden_runtime_metadata",
                "event_read_after_write_ancestry",
                "event_e1_population",
                "event_e2_separation",
                "event_e3_rank_four",
                "event_e4_poincare_cocycle",
                "heldout_quadratic_cone",
                "frame_base_separation",
                "event_manifold",
            }.issubset(semantic_ids)
        ),
        "TARGET_CONFORMANCE_DIAGNOSTICS_ARE_NOT_PRIMITIVE_RECEIPTS": all(
            "TARGET_CONFORMANCE_DIAGNOSTIC" not in key
            for key in canonical_receipt_keys()
        ),
        "ONLY_REGISTERED_VERIFIER_OUTPUTS_ARE_ADMITTED": bool(
            admitted_registry_keys
            == (
                set(C0_RECEIPT_KEYS)
                | set(FEDERATION_ADMITTED_RECEIPT_KEYS)
                | set(REPAIR_ADMITTED_RECEIPT_KEYS)
                | set(OPERATIONAL_OBSERVER_RECEIPT_KEYS)
            )
            and {item.verifier_id for item in ARTIFACT_VERIFIER_REGISTRY}
            == {
                COMMON_SOURCE_VERIFIER_ID,
                FEDERATION_VERIFIER_ID,
                REPAIR_VERIFIER_ID,
                OPERATIONAL_OBSERVER_VERIFIER_ID,
            }
        ),
        "RECORDS_AND_OBSERVERS_FOLLOW_QUOTIENT_REPAIR": bool(
            STAGE_BY_ID["A2O"].dependencies == ("A2",)
            and STAGE_BY_ID["A2Q"].dependencies == ("A2O",)
            and STAGE_BY_ID["A3"].dependencies == ("A2Q",)
            and STAGE_BY_ID["A4"].dependencies == ("A3",)
        ),
        "A5_REQUIRES_CARRIER_REFINEMENT_NATURALITY": any(
            requirement.requirement_id == "carrier_refinement_naturality"
            and requirement.receipt_keys
            == ("CARRIER_REFINEMENT_NATURALITY_RECEIPT",)
            for requirement in STAGE_BY_ID["A5"].requirements
        ),
        "OPERATIONAL_OBSERVER_REQUIRES_COMMON_SOURCE_BINDING": bool(
            _VERIFIER_BY_ID[
                OPERATIONAL_OBSERVER_VERIFIER_ID
            ].requires_common_source_binding
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--schema-output", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return exit status 0 even when the terminal ladder receipt is false",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = write_emergence_ladder_report(
        args.run_dir,
        args.output,
        schema_path=args.schema_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.allow_incomplete:
        return 0
    return 0 if report["overall_receipts"]["OPH_EMERGENCE_LADDER_RECEIPT"] else 2


if (
    __name__ == "__main__"
):  # pragma: no cover - exercised through the public writer in tests
    raise SystemExit(main())
