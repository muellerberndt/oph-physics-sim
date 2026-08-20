"""Fail-closed contract for simulations that invoke the three OPH axioms.

The kernel does not manufacture a physical result.  It checks that a
simulation packet carries the exact A1 architecture, A2 meaning diagrams, and
A3 information-projection data before the packet can be classified.  Result
specific producers may add stronger premises, but they may not hide them in
the axiom fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
import re
from typing import Mapping


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_SELECTOR_FIELDS = frozenset(
    {
        "mar",
        "minimal_admissible_realization",
        "economy_selector",
        "recovery_axiom",
        "generalized_entropy_axiom",
        "refinement_closure_axiom",
    }
)


class ResultStatus(StrEnum):
    """The canonical epistemic classes accepted by simulator receipts."""

    AXIOM_FORCED = "axiom_forced"
    EXACT_NAMED_REALIZATION = "exact_named_realization"
    DISCOVERY_ONLY = "discovery_only"
    CONDITIONAL_OPEN_INTERFACE = "conditional_open_interface"
    INDEPENDENCE_LIMITED = "independence_limited"
    PHYSICAL_IDENTIFICATION = "physical_identification"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class A1RegulatorContract:
    """Source-bound finite architecture consumed by one simulation."""

    port_count: int
    edge_count: int
    face_count: int
    oriented_two_face_edges: bool
    degree_five_ports: bool
    five_cycle_links: bool
    typed_seams: bool
    coherent_triple_overlaps: bool
    isotone_local_algebra_net: bool
    disjoint_visible_locality: bool
    central_record_algebra: bool
    primitive_central_port_partition: bool
    operational_interfaces: frozenset[str]
    boundary_source_sha256: str
    federation_source_sha256: str
    support_bridge_receipt_sha256: str
    support_bridge_degree_one: bool
    support_bridge_refinement_natural: bool
    refinement_interfaces_natural: bool
    response_source_sha256: str
    response_finite_dimensional: bool
    response_real_linear: bool
    response_injective: bool
    response_commutator_closed: bool
    response_complete_on_public_tangent: bool
    response_pairing_positive_definite: bool
    response_refinement_natural: bool


@dataclass(frozen=True)
class A2MeaningDiagram:
    """One accepted-data naturality square with an evidence binding."""

    diagram_id: str
    data_kind: str
    transport_kind: str
    commutes_after_interpretation: bool
    evidence_sha256: str


@dataclass(frozen=True)
class A2AgreementContract:
    """The complete declared A2 diagram family for a simulation packet."""

    diagrams: tuple[A2MeaningDiagram, ...]
    accepted_data_domain_complete: bool
    pairwise_and_higher_overlaps_covered: bool
    refinement_diagrams_covered: bool
    holonomy_source_sha256: str
    proper_recharting_holonomy_surjective: bool
    projective_implementers_exist: bool
    response_naturality: bool
    same_response_endogenous: bool
    path_composition_projective: bool
    refinement_natural: bool


@dataclass(frozen=True)
class FeasibleStateSignature:
    """Exact cover-visible signature of one feasible finite state family."""

    state_id: str
    local_state_sha256: tuple[tuple[str, str], ...]

    def as_mapping(self) -> dict[str, str]:
        return dict(self.local_state_sha256)


@dataclass(frozen=True)
class A3InformationProjectionContract:
    """Exact finite A3 reference, aggregation, and feasible-family packet."""

    object_type: str
    reference_rule_id: str
    reference_rule_sha256: str
    reference_compatible: bool
    reference_faithful: bool
    reference_presentation_invariant: bool
    observer_cover: tuple[str, ...]
    cover_a1_generated: bool
    weights: tuple[tuple[str, Fraction], ...]
    weight_rule_sha256: str
    weights_from_quotient_visible_a1: bool
    aggregation_presentation_natural: bool
    aggregation_refinement_behavior_declared: bool
    objective: str
    feasible_family_nonempty: bool
    feasible_family_convex: bool
    local_states_normalized_and_compatible: bool
    optimizer_exists: bool
    optimizer_unique: bool
    optimizer_support_and_faithfulness_stated: bool
    feasible_signatures: tuple[FeasibleStateSignature, ...]
    observable_grammar: tuple[str, ...]
    visible_constraints: tuple[str, ...]
    constraint_factorization: tuple[tuple[str, str], ...]
    constraints_target_independent: bool
    constraints_quotient_visible: bool
    omitted_constraint_mutations_pass: bool
    wrong_reference_controls: tuple[str, ...]
    wrong_cover_controls: tuple[str, ...]
    alternative_weight_controls: tuple[str, ...]
    identity_proportional_reference: bool
    entropy_equivalence_evidence_sha256: str | None


@dataclass(frozen=True)
class ThreeAxiomSimulationContract:
    """Complete axiom-facing input to a result-specific simulation."""

    contract_id: str
    a1: A1RegulatorContract
    a2: A2AgreementContract
    a3: A3InformationProjectionContract
    presentation_equivalence_sha256: str
    presentation_equivalence_complete: bool
    source_metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractAudit:
    """Machine-readable result of the fail-closed contract audit."""

    valid: bool
    errors: tuple[str, ...]
    state_determining_cover: bool
    constraint_grammar_complete: bool
    entropy_equivalence_claimable: bool


def _valid_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _cover_is_state_determining(
    cover: tuple[str, ...],
    signatures: tuple[FeasibleStateSignature, ...],
) -> bool:
    projected: set[tuple[str, ...]] = set()
    for signature in signatures:
        local = signature.as_mapping()
        if set(local) != set(cover):
            return False
        value = tuple(local[patch] for patch in cover)
        if value in projected:
            return False
        projected.add(value)
    return True


def audit_three_axiom_contract(contract: ThreeAxiomSimulationContract) -> ContractAudit:
    """Validate a three-axiom simulation packet without inferred defaults."""

    errors: list[str] = []
    a1 = contract.a1
    if (a1.port_count, a1.edge_count, a1.face_count) != (12, 30, 20):
        errors.append("A1 boundary cardinalities must be exactly 12/30/20")
    if not all(
        (
            a1.oriented_two_face_edges,
            a1.degree_five_ports,
            a1.five_cycle_links,
            a1.typed_seams,
            a1.coherent_triple_overlaps,
            a1.isotone_local_algebra_net,
            a1.disjoint_visible_locality,
            a1.central_record_algebra,
            a1.primitive_central_port_partition,
            a1.support_bridge_degree_one,
            a1.support_bridge_refinement_natural,
            a1.refinement_interfaces_natural,
        )
    ):
        errors.append(
            "A1 incidence, algebra-net, seam, triple-overlap, or support-bridge receipt is incomplete"
        )
    required_interfaces = {
        "state",
        "readback",
        "records",
        "repair",
        "prediction",
        "checkpoint",
    }
    if not required_interfaces.issubset(a1.operational_interfaces):
        errors.append("A1 operational interface set is incomplete")
    for label, value in (
        ("A1 boundary source", a1.boundary_source_sha256),
        ("A1 federation source", a1.federation_source_sha256),
        ("A1 support bridge", a1.support_bridge_receipt_sha256),
        ("A1 response source", a1.response_source_sha256),
        ("presentation equivalence", contract.presentation_equivalence_sha256),
    ):
        if not _valid_sha256(value):
            errors.append(f"{label} requires a content-addressed sha256 binding")
    if not contract.presentation_equivalence_complete:
        errors.append("A1 presentation equivalence does not preserve the complete public contract")
    if not all(
        (
            a1.response_finite_dimensional,
            a1.response_real_linear,
            a1.response_injective,
            a1.response_commutator_closed,
            a1.response_complete_on_public_tangent,
            a1.response_pairing_positive_definite,
            a1.response_refinement_natural,
        )
    ):
        errors.append(
            "A1 faithful compact response object or its refinement naturality is incomplete"
        )

    a2 = contract.a2
    if not a2.diagrams:
        errors.append("A2 requires at least one declared accepted-data diagram")
    if not all(
        (
            a2.accepted_data_domain_complete,
            a2.pairwise_and_higher_overlaps_covered,
            a2.refinement_diagrams_covered,
        )
    ):
        errors.append("A2 diagram coverage is incomplete")
    diagram_ids = [diagram.diagram_id for diagram in a2.diagrams]
    if len(diagram_ids) != len(set(diagram_ids)):
        errors.append("A2 diagram identifiers must be unique")
    for diagram in a2.diagrams:
        if not diagram.commutes_after_interpretation:
            errors.append(f"A2 diagram does not commute: {diagram.diagram_id}")
        if not _valid_sha256(diagram.evidence_sha256):
            errors.append(f"A2 diagram lacks an evidence hash: {diagram.diagram_id}")
    required_transport_kinds = {
        "overlap_restriction",
        "recharting",
        "seam_translation",
        "higher_overlap",
        "federation_map",
        "refinement",
    }
    observed_transport_kinds = {diagram.transport_kind for diagram in a2.diagrams}
    missing_transport_kinds = sorted(required_transport_kinds - observed_transport_kinds)
    if missing_transport_kinds:
        errors.append(
            "A2 accepted-data diagram family omits transport kinds: "
            + ", ".join(missing_transport_kinds)
        )
    if not _valid_sha256(a2.holonomy_source_sha256):
        errors.append("A2 holonomy implementation requires a content-addressed sha256 binding")
    if not all(
        (
            a2.proper_recharting_holonomy_surjective,
            a2.projective_implementers_exist,
            a2.response_naturality,
            a2.same_response_endogenous,
            a2.path_composition_projective,
            a2.refinement_natural,
        )
    ):
        errors.append(
            "A2 proper-recharting holonomy, projective implementation, or endogeneity is incomplete"
        )

    a3 = contract.a3
    allowed_object_types = {
        "ontic_state_on_fixed_accessible_algebra",
        "observer_inference_state_on_fixed_evidence_algebra",
        "transition_distribution_on_fixed_finite_move_simplex",
    }
    if a3.object_type not in allowed_object_types:
        errors.append("A3 object type is not one of the three canonical optimizer types")
    if not a3.reference_rule_id or not _valid_sha256(a3.reference_rule_sha256):
        errors.append("A3 exact reference rule and content hash are required")
    if not all(
        (
            a3.reference_compatible,
            a3.reference_faithful,
            a3.reference_presentation_invariant,
        )
    ):
        errors.append("A3 reference must be compatible, faithful, and presentation-invariant")
    if not a3.observer_cover or len(set(a3.observer_cover)) != len(a3.observer_cover):
        errors.append("A3 observer cover must be finite, nonempty, and duplicate-free")
    if not a3.cover_a1_generated:
        errors.append("A3 observer cover must be generated from quotient-visible A1 data")
    weights = dict(a3.weights)
    if set(weights) != set(a3.observer_cover) or len(weights) != len(a3.weights):
        errors.append("A3 weights must bind every cover member exactly once")
    elif any(weight <= 0 for weight in weights.values()):
        errors.append("A3 weights must be strictly positive")
    elif sum(weights.values(), Fraction(0, 1)) != Fraction(1, 1):
        errors.append("A3 exact weights must be normalized to one")
    if not _valid_sha256(a3.weight_rule_sha256):
        errors.append("A3 weight rule requires a content-addressed sha256 binding")
    if not all(
        (
            a3.weights_from_quotient_visible_a1,
            a3.aggregation_presentation_natural,
            a3.aggregation_refinement_behavior_declared,
        )
    ):
        errors.append(
            "A3 aggregation rule must be A1-sourced, presentation-natural, and declare refinement behavior"
        )
    if a3.objective != "weighted_local_umegaki_relative_entropy":
        errors.append("A3 objective must be the weighted local Umegaki relative entropy")
    if not all(
        (
            a3.feasible_family_nonempty,
            a3.feasible_family_convex,
            a3.local_states_normalized_and_compatible,
        )
    ) or not a3.feasible_signatures:
        errors.append(
            "A3 feasible family must be nonempty, convex, locally normalized, compatible, and explicitly represented"
        )
    if not a3.optimizer_exists or not a3.optimizer_unique:
        errors.append("A3 information projection must have stated existence and uniqueness")
    if not a3.optimizer_support_and_faithfulness_stated:
        errors.append("A3 optimizer support and faithfulness conditions must be stated")

    state_determining = _cover_is_state_determining(
        a3.observer_cover, a3.feasible_signatures
    )
    if not state_determining:
        errors.append("A3 observer cover is not state-determining on the feasible family")

    grammar = set(a3.observable_grammar)
    factors = dict(a3.constraint_factorization)
    grammar_complete = (
        bool(grammar)
        and set(factors) == set(a3.visible_constraints)
        and all(target in grammar for target in factors.values())
        and a3.constraints_target_independent
        and a3.constraints_quotient_visible
        and a3.omitted_constraint_mutations_pass
    )
    if not grammar_complete:
        errors.append("A3 visible-constraint factorization or mutation coverage is incomplete")
    if not a3.wrong_reference_controls:
        errors.append("A3 wrong-reference controls are required")
    if not a3.wrong_cover_controls:
        errors.append("A3 wrong-cover controls are required")
    if not a3.alternative_weight_controls:
        errors.append("A3 alternative-weight controls are required")

    forbidden = _FORBIDDEN_SELECTOR_FIELDS.intersection(
        key.lower() for key in contract.source_metadata
    )
    if forbidden:
        errors.append(
            "retired or unclassified selector fields are forbidden: "
            + ", ".join(sorted(forbidden))
        )

    entropy_evidence_valid = bool(
        a3.entropy_equivalence_evidence_sha256
        and _valid_sha256(a3.entropy_equivalence_evidence_sha256)
    )
    if a3.identity_proportional_reference and not entropy_evidence_valid:
        errors.append(
            "A3 entropy-maximization equivalence requires content-addressed reference evidence"
        )
    entropy_equivalence = bool(
        not errors and a3.identity_proportional_reference and entropy_evidence_valid
    )
    return ContractAudit(
        valid=not errors,
        errors=tuple(errors),
        state_determining_cover=state_determining,
        constraint_grammar_complete=grammar_complete,
        entropy_equivalence_claimable=entropy_equivalence,
    )
