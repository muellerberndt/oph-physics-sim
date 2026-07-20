from __future__ import annotations

import hashlib
import inspect
from fractions import Fraction

import pytest

from oph_fpe.gauge.source_to_eft import (
    CalculationLane,
    CoherentSourceCommitments,
    EFTIntervalCensus,
    ExactAffineMomentMap,
    ExactAffineTransformation,
    ExactLeftInverseCertificate,
    ExactMatchingStep,
    ExactMatchingTower,
    FiniteSourceLawPacket,
    MomentSemantics,
    PerturbativeMask,
    QUOTIENT_MOMENT_SCHEMA_VERSION,
    QuotientMomentPacket,
    REQUIRED_SOURCE_COMPONENTS,
    SOURCE_LAW_SCHEMA_VERSION,
    SourceComponentEnvelope,
    SourceLawSemantics,
    UntrustedParameterCandidate,
    compose_exact_matching_tower,
    finite_source_law_report,
    matching_field_census_hash,
    matching_scheme_hash,
    matching_threshold_hash,
    source_to_eft_contract_report,
    verify_coherent_source_components,
    verify_common_perturbative_mask,
    verify_exact_affine_identifiability,
    verify_finite_source_law_artifact,
)


def _hash(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _mask() -> PerturbativeMask:
    return PerturbativeMask(
        axes=("loop", "coupling"),
        monomials=((0, 0), (0, 1), (1, 0)),
    )


def _steps(mask: PerturbativeMask) -> tuple[ExactMatchingStep, ...]:
    interval_0 = EFTIntervalCensus("eft0", ("field0",), "scheme0")
    interval_1 = EFTIntervalCensus("eft1", ("field0", "field1"), "scheme1")
    interval_2 = EFTIntervalCensus("eft2", ("field2",), "scheme2")
    step_0 = ExactMatchingStep(
        step_id="threshold0",
        source_census=interval_0,
        target_census=interval_1,
        flow=ExactAffineTransformation(
            input_coordinates=("parameter0", "parameter1"),
            output_coordinates=("q0", "q1"),
            matrix=((2, 0), (0, 3)),
            offset=(1, 0),
            artifact_hash=_hash("flow0"),
        ),
        matching=ExactAffineTransformation(
            input_coordinates=("q0", "q1"),
            output_coordinates=("r0", "r1"),
            matrix=((1, 1), (0, 1)),
            offset=(0, 2),
            artifact_hash=_hash("matching0"),
        ),
        implementation_remainder_bound=Fraction(1, 20),
        perturbative_mask=mask,
    )
    step_1 = ExactMatchingStep(
        step_id="threshold1",
        source_census=interval_1,
        target_census=interval_2,
        flow=ExactAffineTransformation(
            input_coordinates=("r0", "r1"),
            output_coordinates=("s0", "s1"),
            matrix=((1, 0), (0, 1)),
            offset=(0, 1),
            artifact_hash=_hash("flow1"),
        ),
        matching=ExactAffineTransformation(
            input_coordinates=("s0", "s1"),
            output_coordinates=("t0", "t1"),
            matrix=((Fraction(1, 2), 0), (0, 2)),
            offset=(1, 0),
            artifact_hash=_hash("matching1"),
        ),
        implementation_remainder_bound=Fraction(1, 100),
        perturbative_mask=mask,
    )
    return (step_0, step_1)


def _commitments(
    mask: PerturbativeMask, steps: tuple[ExactMatchingStep, ...]
) -> CoherentSourceCommitments:
    return CoherentSourceCommitments(
        source_root_hash=_hash("source-root"),
        branch_hash=_hash("branch-0"),
        field_census_hash=matching_field_census_hash(steps),
        scheme_hash=matching_scheme_hash(steps),
        threshold_hash=matching_threshold_hash(steps),
        fj_convention_hash=_hash("fj-convention"),
        perturbative_mask_hash=mask.commitment,
        analytic_sheet_hash=_hash("analytic-sheet"),
        units_clock_hash=_hash("dimensionless-only"),
    )


@pytest.fixture
def exact_fixture() -> dict[str, object]:
    mask = _mask()
    steps = _steps(mask)
    commitments = _commitments(mask, steps)
    packet = QuotientMomentPacket(
        regulator_id="r0",
        quotient_artifact_hash=_hash("quotient"),
        observable_map_hash=_hash("moment-map"),
        observable_names=("obs0", "obs1", "obs2"),
        moments=(4, -3, 1),
        semantics=MomentSemantics.DETERMINISTIC_SETTLED_STATE,
        admissible_branch_hashes=(commitments.branch_hash,),
        commitments=commitments,
        settled_state_hash=_hash("settled-state"),
        refinement_defects=(Fraction(1, 100),),
    )
    moment_map = ExactAffineMomentMap(
        parameter_names=("parameter0", "parameter1"),
        observable_names=packet.observable_names,
        matrix=((2, 0), (0, 3), (1, 1)),
        offset=(0, 0, 0),
        map_artifact_hash=packet.observable_map_hash,
    )
    candidate = UntrustedParameterCandidate(
        values=(2, -1), candidate_artifact_hash=_hash("authored-candidate")
    )
    left_inverse = ExactLeftInverseCertificate(
        matrix=((Fraction(1, 2), 0, 0), (0, Fraction(1, 3), 0)),
        certificate_artifact_hash=_hash("left-inverse"),
    )
    source_law = FiniteSourceLawPacket(
        semantics=SourceLawSemantics.DETERMINISTIC,
        coordinate_names=("parameter0", "parameter1"),
        admissible_branch_hashes=(commitments.branch_hash,),
        source_law_hash=_hash("source-law"),
        commitments=commitments,
        deterministic_point=(2, -1),
    )
    components = tuple(
        SourceComponentEnvelope(kind, _hash(f"component:{kind}"), commitments)
        for kind in REQUIRED_SOURCE_COMPONENTS
    )
    return {
        "mask": mask,
        "steps": steps,
        "commitments": commitments,
        "packet": packet,
        "moment_map": moment_map,
        "candidate": candidate,
        "left_inverse": left_inverse,
        "source_law": source_law,
        "components": components,
        "tower": ExactMatchingTower(steps, commitments),
    }


def test_exact_affine_identifiability_proves_only_the_math(exact_fixture: dict[str, object]) -> None:
    certificate = verify_exact_affine_identifiability(
        exact_fixture["packet"],
        exact_fixture["moment_map"],
        exact_fixture["candidate"],
        exact_fixture["left_inverse"],
        residual_bound=Fraction(1, 10),
    )
    report = certificate.to_report()

    assert report["left_inverse_exact"] is True
    assert report["lower_inverse_modulus_sigma"] == "2"
    assert report["actual_residual"] == "0"
    assert report["two_candidate_ambiguity_diameter_bound"] == "1/10"
    assert report["receipts"]["EXACT_SOURCE_PARAMETER_IDENTIFIABILITY_MATH_RECEIPT"] is True
    assert report["receipts"]["SOURCE_PARAMETER_CANDIDATE_PRODUCER_RECEIPT"] is False
    assert report["receipts"]["OPH_NATIVE_EFT_REALIZATION_RECEIPT"] is False


def test_authored_theta_cannot_be_injected_into_quotient_packet(
    exact_fixture: dict[str, object]
) -> None:
    artifact = exact_fixture["packet"].to_artifact()
    artifact["theta"] = {"parameter0": "2", "parameter1": "-1"}

    with pytest.raises(ValueError, match="unexpected fields"):
        QuotientMomentPacket.from_artifact(artifact)


def test_exact_residual_and_bad_left_inverse_fail_closed(exact_fixture: dict[str, object]) -> None:
    candidate = UntrustedParameterCandidate(
        values=(Fraction(21, 10), -1), candidate_artifact_hash=_hash("candidate-off-root")
    )
    bad_inverse = ExactLeftInverseCertificate(
        matrix=((Fraction(1, 2), 0, 0), (0, Fraction(1, 2), 0)),
        certificate_artifact_hash=_hash("bad-left-inverse"),
    )
    certificate = verify_exact_affine_identifiability(
        exact_fixture["packet"],
        exact_fixture["moment_map"],
        candidate,
        bad_inverse,
        residual_bound=Fraction(1, 100),
    )

    assert certificate.mathematical_receipt is False
    assert "left_inverse_identity_failed" in certificate.blockers
    assert "candidate_residual_exceeds_declared_bound" in certificate.blockers


def test_quotient_parser_rejects_floating_point_moments(exact_fixture: dict[str, object]) -> None:
    artifact = exact_fixture["packet"].to_artifact()
    artifact["moments"][0] = 4.0

    with pytest.raises(TypeError, match="exact integer"):
        QuotientMomentPacket.from_artifact(artifact)


def test_deterministic_and_stochastic_source_laws_are_separate(
    exact_fixture: dict[str, object]
) -> None:
    deterministic = finite_source_law_report(exact_fixture["source_law"])
    stochastic_packet = FiniteSourceLawPacket(
        semantics=SourceLawSemantics.STOCHASTIC,
        coordinate_names=("z0", "z1"),
        admissible_branch_hashes=(
            exact_fixture["commitments"].branch_hash,
            _hash("branch-1"),
        ),
        source_law_hash=_hash("stochastic-law"),
        commitments=exact_fixture["commitments"],
        support_points=((0, 0), (2, 4)),
        weights=(Fraction(1, 4), Fraction(3, 4)),
    )
    stochastic = finite_source_law_report(stochastic_packet)

    assert deterministic["source_covariance"] == [["0", "0"], ["0", "0"]]
    assert deterministic["receipts"]["DETERMINISTIC_SOURCE_SEMANTICS_RECEIPT"] is True
    assert stochastic["source_mean"] == ["3/2", "3"]
    assert stochastic["source_covariance"] == [["3/4", "3/2"], ["3/2", "3"]]
    assert stochastic["receipts"]["STOCHASTIC_SOURCE_SEMANTICS_RECEIPT"] is True
    assert deterministic["receipts"]["COV1_RECEIPT"] is False
    assert stochastic["receipts"]["COV1_RECEIPT"] is False


def test_branch_ambiguity_forbids_deterministic_zero_covariance(
    exact_fixture: dict[str, object]
) -> None:
    ambiguous = FiniteSourceLawPacket(
        semantics=SourceLawSemantics.DETERMINISTIC,
        coordinate_names=("z0",),
        admissible_branch_hashes=(
            exact_fixture["commitments"].branch_hash,
            _hash("branch-ambiguous"),
        ),
        source_law_hash=_hash("ambiguous-law"),
        commitments=exact_fixture["commitments"],
        deterministic_point=(0,),
    )
    report = finite_source_law_report(ambiguous)

    assert report["source_covariance"] is None
    assert report["receipts"]["FINITE_SOURCE_LAW_STRUCTURAL_RECEIPT"] is False
    assert "deterministic_source_forbidden_by_branch_ambiguity" in report["blockers"]


def test_interval_or_enclosure_cannot_be_authored_as_source_covariance(
    exact_fixture: dict[str, object]
) -> None:
    packet = exact_fixture["source_law"]
    artifact = {
        "schema_version": SOURCE_LAW_SCHEMA_VERSION,
        "semantics": packet.semantics.value,
        "coordinate_names": list(packet.coordinate_names),
        "admissible_branch_hashes": list(packet.admissible_branch_hashes),
        "source_law_hash": packet.source_law_hash,
        "commitments": packet.commitments.to_artifact(),
        "deterministic_point": ["2", "-1"],
        "source_covariance": [["1/100", "0"], ["0", "1/100"]],
    }
    report = verify_finite_source_law_artifact(artifact)

    assert report["receipts"]["FINITE_SOURCE_LAW_STRUCTURAL_RECEIPT"] is False
    assert report["receipts"]["COV1_RECEIPT"] is False
    assert "unexpected fields" in report["blockers"][0]


def test_mask_mismatch_blocks_fj_and_brst_even_with_same_order_label() -> None:
    direct = PerturbativeMask(
        axes=("loop", "coupling"), monomials=((0, 0), (1, 0))
    )
    converted = PerturbativeMask(
        axes=("loop", "coupling"), monomials=((0, 0), (0, 1))
    )
    report = verify_common_perturbative_mask(direct, converted)

    assert report["receipts"]["COMMON_PERTURBATIVE_MASK_EQUALITY_RECEIPT"] is False
    assert report["receipts"]["FJ1_RECEIPT"] is False
    assert report["receipts"]["BRST1_RECEIPT"] is False


def test_non_downward_closed_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="not downward closed"):
        PerturbativeMask(axes=("loop",), monomials=((0,), (2,)))


def test_source_hash_mismatch_invalidates_coherent_packet(
    exact_fixture: dict[str, object]
) -> None:
    original = exact_fixture["commitments"]
    altered = CoherentSourceCommitments(
        **{
            **original.to_artifact(),
            "source_root_hash": _hash("different-source-root"),
        }
    )
    components = list(exact_fixture["components"])
    components[-1] = SourceComponentEnvelope(
        components[-1].component_kind,
        components[-1].artifact_hash,
        altered,
    )
    report = verify_coherent_source_components(components)

    assert report["receipts"]["COHERENT_SOURCE_COMMITMENT_EQUALITY_RECEIPT"] is False
    assert report["receipts"]["COHERENT_PHYSICAL_SOURCE_PACKET_RECEIPT"] is False
    assert any("source_root_hash" in blocker for blocker in report["blockers"])


def test_forged_but_equal_hash_labels_do_not_create_physical_source(
    exact_fixture: dict[str, object]
) -> None:
    report = verify_coherent_source_components(exact_fixture["components"])

    assert report["receipts"]["COHERENT_SOURCE_COMMITMENT_EQUALITY_RECEIPT"] is True
    assert report["receipts"]["COHERENT_PHYSICAL_SOURCE_PACKET_RECEIPT"] is False


def test_missing_field_census_fails_before_matching() -> None:
    with pytest.raises(ValueError, match="complete nonempty unique census"):
        EFTIntervalCensus("empty", (), "scheme0")

    incomplete = {
        "source_root_hash": _hash("root"),
        "branch_hash": _hash("branch"),
        # field_census_hash deliberately absent
        "scheme_hash": _hash("scheme"),
        "threshold_hash": _hash("threshold"),
        "fj_convention_hash": _hash("fj"),
        "perturbative_mask_hash": _hash("mask"),
        "analytic_sheet_hash": _hash("sheet"),
        "units_clock_hash": _hash("clock"),
    }
    with pytest.raises(ValueError, match="field_census_hash"):
        CoherentSourceCommitments.from_artifact(incomplete)


def test_matching_tower_composes_exact_error_and_jacobian(
    exact_fixture: dict[str, object]
) -> None:
    certificate = compose_exact_matching_tower(
        exact_fixture["tower"], initial_error_bound=Fraction(1, 10)
    )
    report = certificate.to_report()

    assert certificate.endpoint_error_bound == Fraction(131, 100)
    assert certificate.endpoint_jacobian == (
        (Fraction(1), Fraction(3, 2)),
        (Fraction(0), Fraction(6)),
    )
    assert certificate.total_offset == (Fraction(3, 2), Fraction(6))
    assert report["receipts"]["MATCHING_TOWER_EXACT_ERROR_COMPOSITION_RECEIPT"] is True
    assert report["receipts"]["MATCHING_TOWER_EXACT_JACOBIAN_COMPOSITION_RECEIPT"] is True
    assert report["receipts"]["MATCH1_RECEIPT"] is False


def test_imported_and_oph_native_lanes_are_disjoint() -> None:
    imported = source_to_eft_contract_report(
        CalculationLane.IMPORTED_SM_STRICT_1L_VALIDATION
    )
    native = source_to_eft_contract_report(CalculationLane.OPH_NATIVE_STRICT_1L)

    assert imported["receipts"]["IMPORTED_SM_VALIDATION_LANE_DECLARATION_RECEIPT"] is True
    assert imported["receipts"]["OPH_NATIVE_LANE_DECLARATION_RECEIPT"] is False
    assert native["receipts"]["IMPORTED_SM_VALIDATION_LANE_DECLARATION_RECEIPT"] is False
    assert native["receipts"]["OPH_NATIVE_LANE_DECLARATION_RECEIPT"] is True
    assert imported["receipts"]["IMPORTED_SM_STRICT_1L_VALIDATION_RECEIPT"] is False
    assert native["receipts"]["OPH_NATIVE_STRICT_1L_PHYSICAL_RECEIPT"] is False


def test_complete_exact_fixture_still_cannot_promote_physics(
    exact_fixture: dict[str, object]
) -> None:
    report = source_to_eft_contract_report(
        CalculationLane.OPH_NATIVE_STRICT_1L,
        moment_packet=exact_fixture["packet"],
        moment_map=exact_fixture["moment_map"],
        candidate=exact_fixture["candidate"],
        left_inverse=exact_fixture["left_inverse"],
        residual_bound=Fraction(1, 10),
        source_law=exact_fixture["source_law"],
        source_components=exact_fixture["components"],
        matching_tower=exact_fixture["tower"],
        initial_matching_error=Fraction(1, 10),
    )

    assert report["receipts"]["EXACT_SOURCE_PARAMETER_IDENTIFIABILITY_MATH_RECEIPT"] is True
    assert report["receipts"]["COHERENT_SOURCE_COMMITMENT_EQUALITY_RECEIPT"] is True
    assert report["receipts"]["SOURCE_TO_MATCHING_COORDINATE_BINDING_RECEIPT"] is True
    assert report["receipts"]["QUOTIENT_MOMENT_SOURCE_LAW_BINDING_RECEIPT"] is True
    assert report["receipts"]["MATCHING_TOWER_EXACT_JACOBIAN_COMPOSITION_RECEIPT"] is True
    for receipt in (
        "COHERENT_PHYSICAL_SOURCE_PACKET_RECEIPT",
        "SOURCE_PARAMETER_CANDIDATE_PRODUCER_RECEIPT",
        "SOURCE_LAW_PRIMITIVE_PRODUCER_RECEIPT",
        "OPH_SOURCE_PRIMITIVE_PRODUCER_RECEIPT",
        "OPH_QUOTIENT_MOMENT_PHYSICAL_RECEIPT",
        "OPH_NATIVE_EFT_REALIZATION_RECEIPT",
        "MATCH1_RECEIPT",
        "FJ1_RECEIPT",
        "COV1_RECEIPT",
        "BRST1_RECEIPT",
        "PHYSICAL_POLE_RECEIPT",
        "PHYSICAL_COHOMOLOGY_POLE_RECEIPT",
        "OPH_NATIVE_STRICT_1L_PHYSICAL_RECEIPT",
    ):
        assert report["receipts"][receipt] is False


def test_caller_boolean_receipts_are_not_an_api_surface(
    exact_fixture: dict[str, object]
) -> None:
    signature = inspect.signature(source_to_eft_contract_report)
    assert "receipts" not in signature.parameters
    assert "primitive_receipts" not in signature.parameters

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        source_to_eft_contract_report(
            CalculationLane.OPH_NATIVE_STRICT_1L,
            MATCH1_RECEIPT=True,
        )


def test_quotient_artifact_schema_is_strict(exact_fixture: dict[str, object]) -> None:
    artifact = exact_fixture["packet"].to_artifact()
    assert artifact["schema_version"] == QUOTIENT_MOMENT_SCHEMA_VERSION
    parsed = QuotientMomentPacket.from_artifact(artifact)
    assert parsed == exact_fixture["packet"]
