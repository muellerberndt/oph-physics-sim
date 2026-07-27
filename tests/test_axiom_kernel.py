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
            operational_interfaces=frozenset(
                {"state", "readback", "records", "repair", "checkpoint"}
            ),
            boundary_source_sha256=HASH,
            federation_source_sha256=HASH,
            support_bridge_receipt_sha256=HASH,
            support_bridge_degree_one=True,
            support_bridge_refinement_natural=True,
        ),
        a2=A2AgreementContract(
            diagrams=(
                A2MeaningDiagram(
                    diagram_id="pairwise-readback",
                    data_kind="accepted_readback",
                    transport_kind="seam_translation",
                    commutes_after_interpretation=True,
                    evidence_sha256=HASH,
                ),
            ),
            accepted_data_domain_complete=True,
            pairwise_and_higher_overlaps_covered=True,
            refinement_diagrams_covered=True,
        ),
        a3=A3InformationProjectionContract(
            object_type="ontic_state_on_fixed_accessible_algebra",
            reference_rule_id="canonical-local-reference",
            reference_rule_sha256=HASH,
            reference_compatible=True,
            reference_presentation_invariant=True,
            observer_cover=("p0", "p1"),
            weights=(("p0", Fraction(1, 3)), ("p1", Fraction(2, 3))),
            weight_rule_sha256=HASH,
            objective="weighted_local_umegaki_relative_entropy",
            feasible_family_nonempty=True,
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
            omitted_constraint_mutations_pass=True,
            wrong_reference_controls=("nonuniform-reference",),
            wrong_cover_controls=("drop-p1",),
            alternative_weight_controls=("equal-weights",),
        ),
        presentation_equivalence_sha256=HASH,
        source_metadata={
            "reference_density_class": "identity_proportional_in_declared_trace"
        },
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


def test_retired_selector_fields_fail_closed() -> None:
    contract = valid_contract()
    audit = audit_three_axiom_contract(
        replace(contract, source_metadata={"MAR": "legacy-selector"})
    )
    assert not audit.valid
    assert any("retired or unclassified" in error for error in audit.errors)
