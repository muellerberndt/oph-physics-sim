from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from oph_fpe.core.axiom_kernel import (
    A1RegulatorContract,
    A2AgreementContract,
    A2MeaningDiagram,
    A3InformationProjectionContract,
    FeasibleStateSignature,
    ThreeAxiomSimulationContract,
    audit_three_axiom_contract,
)


HASH = "sha256:" + "a" * 64


def valid_contract() -> ThreeAxiomSimulationContract:
    return ThreeAxiomSimulationContract(
        contract_id="test-three-axiom-packet",
        a1=A1RegulatorContract(
            port_count=12,
            edge_count=30,
            face_count=20,
            oriented_two_face_edges=True,
            degree_five_ports=True,
            five_cycle_links=True,
            typed_seams=True,
            coherent_triple_overlaps=True,
            isotone_local_algebra_net=True,
            disjoint_visible_locality=True,
            central_record_algebra=True,
            primitive_central_port_partition=True,
            operational_interfaces=frozenset(
                {"state", "readback", "records", "repair", "prediction", "checkpoint"}
            ),
            boundary_source_sha256=HASH,
            federation_source_sha256=HASH,
            support_bridge_receipt_sha256=HASH,
            support_bridge_degree_one=True,
            support_bridge_refinement_natural=True,
            refinement_interfaces_natural=True,
            response_source_sha256=HASH,
            response_finite_dimensional=True,
            response_real_linear=True,
            response_injective=True,
            response_commutator_closed=True,
            response_complete_on_public_tangent=True,
            response_pairing_positive_definite=True,
            response_refinement_natural=True,
        ),
        a2=A2AgreementContract(
            diagrams=tuple(
                A2MeaningDiagram(
                    diagram_id=f"accepted-{kind}",
                    data_kind="accepted_readback",
                    transport_kind=kind,
                    commutes_after_interpretation=True,
                    evidence_sha256=HASH,
                )
                for kind in (
                    "overlap_restriction",
                    "recharting",
                    "seam_translation",
                    "higher_overlap",
                    "federation_map",
                    "refinement",
                )
            ),
            accepted_data_domain_complete=True,
            pairwise_and_higher_overlaps_covered=True,
            refinement_diagrams_covered=True,
            holonomy_source_sha256=HASH,
            proper_recharting_holonomy_surjective=True,
            projective_implementers_exist=True,
            response_naturality=True,
            same_response_endogenous=True,
            path_composition_projective=True,
            refinement_natural=True,
        ),
        a3=A3InformationProjectionContract(
            object_type="ontic_state_on_fixed_accessible_algebra",
            reference_rule_id="canonical-local-reference",
            reference_rule_sha256=HASH,
            reference_compatible=True,
            reference_faithful=True,
            reference_presentation_invariant=True,
            observer_cover=("p0", "p1"),
            cover_a1_generated=True,
            weights=(("p0", Fraction(1, 3)), ("p1", Fraction(2, 3))),
            weight_rule_sha256=HASH,
            weights_from_quotient_visible_a1=True,
            aggregation_presentation_natural=True,
            aggregation_refinement_behavior_declared=True,
            objective="weighted_local_umegaki_relative_entropy",
            feasible_family_nonempty=True,
            feasible_family_convex=True,
            local_states_normalized_and_compatible=True,
            optimizer_exists=True,
            optimizer_unique=True,
            optimizer_support_and_faithfulness_stated=True,
            feasible_signatures=(
                FeasibleStateSignature(
                    "rho0", (("p0", "sha256:" + "0" * 64), ("p1", "sha256:" + "1" * 64))
                ),
                FeasibleStateSignature(
                    "rho1", (("p0", "sha256:" + "0" * 64), ("p1", "sha256:" + "2" * 64))
                ),
            ),
            observable_grammar=("energy", "charge"),
            visible_constraints=("visible-energy", "visible-charge"),
            constraint_factorization=(
                ("visible-energy", "energy"),
                ("visible-charge", "charge"),
            ),
            constraints_target_independent=True,
            constraints_quotient_visible=True,
            omitted_constraint_mutations_pass=True,
            wrong_reference_controls=("nonuniform-reference",),
            wrong_cover_controls=("drop-p1",),
            alternative_weight_controls=("equal-weights",),
            identity_proportional_reference=True,
            entropy_equivalence_evidence_sha256=HASH,
        ),
        presentation_equivalence_sha256=HASH,
        presentation_equivalence_complete=True,
        source_metadata={},
    )


def test_valid_exact_contract_passes() -> None:
    audit = audit_three_axiom_contract(valid_contract())
    assert audit.valid, audit.errors
    assert audit.state_determining_cover
    assert audit.constraint_grammar_complete
    assert audit.entropy_equivalence_claimable


def test_cover_must_determine_the_feasible_family() -> None:
    contract = valid_contract()
    repeated = replace(
        contract.a3.feasible_signatures[1],
        local_state_sha256=contract.a3.feasible_signatures[0].local_state_sha256,
    )
    audit = audit_three_axiom_contract(
        replace(
            contract,
            a3=replace(
                contract.a3,
                feasible_signatures=(contract.a3.feasible_signatures[0], repeated),
            ),
        )
    )
    assert not audit.valid
    assert any("state-determining" in error for error in audit.errors)


def test_weights_are_exact_positive_and_complete() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(
            contract,
            a3=replace(contract.a3, weights=(("p0", Fraction(1, 1)),)),
        )
    )
    assert not audit.valid
    assert any("weights must bind" in error for error in audit.errors)


def test_controls_are_mandatory() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(
            contract,
            a3=replace(
                contract.a3,
                wrong_reference_controls=(),
                wrong_cover_controls=(),
                alternative_weight_controls=(),
            ),
        )
    )
    assert not audit.valid
    assert sum("controls are required" in error for error in audit.errors) == 3


def test_a1_response_object_is_not_optional() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, a1=replace(contract.a1, response_commutator_closed=False))
    )
    assert not audit.valid
    assert any("response object" in error for error in audit.errors)


def test_a2_holonomy_endogeneity_is_not_reduced_to_pairwise_agreement() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, a2=replace(contract.a2, same_response_endogenous=False))
    )
    assert not audit.valid
    assert any("holonomy" in error for error in audit.errors)


def test_a2_requires_every_canonical_transport_kind() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, a2=replace(contract.a2, diagrams=contract.a2.diagrams[:-1]))
    )
    assert not audit.valid
    assert any("refinement" in error for error in audit.errors)


def test_a3_requires_unique_information_projection() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, a3=replace(contract.a3, optimizer_unique=False))
    )
    assert not audit.valid
    assert any("existence and uniqueness" in error for error in audit.errors)


def test_entropy_equivalence_requires_evidence_and_a_valid_contract() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(
            contract,
            a3=replace(contract.a3, entropy_equivalence_evidence_sha256=None),
        )
    )
    assert not audit.valid
    assert not audit.entropy_equivalence_claimable
    assert any("entropy-maximization" in error for error in audit.errors)


def test_retired_selector_fields_fail_closed() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, source_metadata={"MAR": "legacy-selector"})
    )
    assert not audit.valid
    assert any("retired or unclassified" in error for error in audit.errors)
