from __future__ import annotations

from dataclasses import replace

from oph_fpe.cosmology.anomaly_abundance_selector import (
    ANOMALY_ABUNDANCE_SOURCE_RECEIPT,
    LOAD_NO_CIRCULARITY_RECEIPT,
    LOAD_NO_DATA_USE_RECEIPT,
    LOAD_QUOTIENT_INVARIANCE_RECEIPT,
    LOAD_REFINEMENT_COMPATIBILITY_RECEIPT,
    PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE,
    RHO_A_SOURCE_RECEIPT,
    RHO_A_TRANSPORT_RECEIPT,
    SOURCE_MAXENT_RELEASE_LAW_RECEIPT,
    SOURCE_ONLY_ANOMALY_ABUNDANCE,
    ANOMALY_RELEASE_STATE_RECEIPT,
    AnomalyAbundanceSelectorArtifact,
    AnomalyLoadObservableArtifact,
    AnomalyReleaseStateArtifact,
    LoadRefinementCompatibilityArtifact,
    SourceMaxEntReleaseLawArtifact,
    compute_selector,
    verify_anomaly_abundance_source_receipt,
    verify_anomaly_release_state,
    verify_load_quotient_invariance,
    verify_load_refinement_compatibility,
    verify_source_maxent_release_law,
)


def test_abundance_from_friedmann_residual_fails():
    bundle = _bundle()
    bundle["load_observable"] = replace(
        bundle["load_observable"],
        normalization_source="friedmann_residual_external_omega_A",
    )

    report = verify_anomaly_abundance_source_receipt(bundle)

    assert report[ANOMALY_ABUNDANCE_SOURCE_RECEIPT] is False
    assert "load_normalization_reads_external_abundance" in report["blockers"]


def test_abundance_from_loaded_parent_is_circular():
    load = replace(_load(), computed_from_loaded_parent=True)

    report = verify_anomaly_abundance_source_receipt(_bundle(load_observable=load))

    assert report[ANOMALY_ABUNDANCE_SOURCE_RECEIPT] is False
    assert "load_computed_from_amplitude_loaded_parent" in report["blockers"]


def test_load_changes_under_port_relabel_fails():
    load = replace(_load(), hidden_relabel_residual=1.0e-3)

    report = verify_load_quotient_invariance(load)

    assert report[LOAD_QUOTIENT_INVARIANCE_RECEIPT] is False
    assert "hidden_carrier_relabel_changes_load" in report["blockers"]


def test_source_maxent_with_planck_target_fails():
    law = replace(_release_law(), source_dag_inputs=("Planck Omega_m target",))

    report = verify_source_maxent_release_law(law)

    assert report[SOURCE_MAXENT_RELEASE_LAW_RECEIPT] is False
    assert "release_law_reads_observational_data" in report["blockers"]


def test_source_maxent_with_sparc_target_fails():
    law = replace(_release_law(), source_dag_inputs=("SPARC acceleration residual",))

    report = verify_source_maxent_release_law(law)

    assert report[SOURCE_MAXENT_RELEASE_LAW_RECEIPT] is False
    assert "release_law_reads_observational_data" in report["blockers"]


def test_missing_release_law_fails():
    law = replace(_release_law(), release_law_hash="")

    report = verify_source_maxent_release_law(law)

    assert report[SOURCE_MAXENT_RELEASE_LAW_RECEIPT] is False
    assert "release_law_hash_missing" in report["blockers"]


def test_missing_release_surface_fails():
    release = replace(_release_state(), release_surface_hash="")

    report = verify_anomaly_release_state(release)

    assert report[ANOMALY_RELEASE_STATE_RECEIPT] is False
    assert "release_surface_hash_missing" in report["blockers"]


def test_exact_refinement_passes():
    report = verify_load_refinement_compatibility(_refinement())

    assert report[LOAD_REFINEMENT_COMPATIBILITY_RECEIPT] is True
    assert report["blockers"] == []


def test_approx_refinement_bound_enforced():
    refinement = replace(
        _refinement(),
        passes_exact=False,
        law_tv_defect=0.1,
        load_naturality_defect=0.25,
        load_sup_bound=2.0,
        selected_load_difference_bound=0.65,
    )

    report = verify_load_refinement_compatibility(refinement)

    assert report[LOAD_REFINEMENT_COMPATIBILITY_RECEIPT] is True
    assert report["selected_load_difference_bound"] <= report["computed_bound"]


def test_approx_refinement_bound_fails_when_exceeded():
    refinement = replace(
        _refinement(),
        passes_exact=False,
        law_tv_defect=0.1,
        load_naturality_defect=0.25,
        load_sup_bound=2.0,
        selected_load_difference_bound=0.66,
    )

    report = verify_load_refinement_compatibility(refinement)

    assert report[LOAD_REFINEMENT_COMPATIBILITY_RECEIPT] is False
    assert "load_refinement_bound_exceeded" in report["blockers"]


def test_transport_without_selector_is_conditional():
    bundle = _bundle()
    bundle["load_observable"] = replace(bundle["load_observable"], computed_from_loaded_parent=True)
    bundle[RHO_A_TRANSPORT_RECEIPT] = True

    report = verify_anomaly_abundance_source_receipt(bundle)

    assert report[RHO_A_TRANSPORT_RECEIPT] is True
    assert report[RHO_A_SOURCE_RECEIPT] is False
    assert report["claim_label"] == PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE


def test_selector_promotes_source_only_abundance():
    report = verify_anomaly_abundance_source_receipt(_bundle())

    assert report[ANOMALY_ABUNDANCE_SOURCE_RECEIPT] is True
    assert report[RHO_A_TRANSPORT_RECEIPT] is True
    assert report[RHO_A_SOURCE_RECEIPT] is True
    assert report["claim_label"] == SOURCE_ONLY_ANOMALY_ABUNDANCE


def test_compute_selector_returns_weighted_expectation():
    assert compute_selector([1.0, 3.0], [2.0, 10.0]) == 8.0


def _bundle(**overrides):
    bundle = {
        "release_state": _release_state(),
        "load_observable": _load(),
        "release_law": _release_law(),
        "selector": _selector(),
        "refinement": _refinement(),
        RHO_A_TRANSPORT_RECEIPT: True,
    }
    bundle.update(overrides)
    return bundle


def _release_state() -> AnomalyReleaseStateArtifact:
    return AnomalyReleaseStateArtifact(
        artifact_id="release",
        regulator_id="r1",
        parent_generation_id="gen1",
        release_surface_hash=_hash("a"),
        normal_tetrad_hash=_hash("b"),
        scale_factor_hash=_hash("c"),
        physical_volume_hash=_hash("d"),
        comoving_volume_hash=_hash("e"),
        homogeneous_projector_hash=_hash("f"),
        release_boundary_sector_hash=_hash("1"),
        P_star_hash=_hash("2"),
        N_CRC_hash=_hash("3"),
        generation_manifest_hash=_hash("4"),
        no_data_ledger_hash=_hash("5"),
    )


def _load() -> AnomalyLoadObservableArtifact:
    return AnomalyLoadObservableArtifact(
        artifact_id="load",
        regulator_id="r1",
        parent_generation_id="gen1",
        release_state_artifact_id="release",
        stress_readout_hash=_hash("6"),
        bw_source_readout_hash=_hash("7"),
        tomography_artifact_hash=_hash("8"),
        variational_moment_residual_hash=_hash("9"),
        cell_volume_hash=_hash("a"),
        normal_hash=_hash("b"),
        homogeneous_projector_hash=_hash("c"),
        load_values_hash=_hash("d"),
        load_units="J",
        load_bound_hash=_hash("e"),
        quotient_invariance_residual=0.0,
        circularity_guard_hash=_hash("f"),
    )


def _release_law() -> SourceMaxEntReleaseLawArtifact:
    return SourceMaxEntReleaseLawArtifact(
        artifact_id="law",
        regulator_id="r1",
        quotient_space_hash=_hash("a"),
        base_weight_hash=_hash("b"),
        action_hash=_hash("c"),
        constraint_observable_hashes=[_hash("d")],
        constraint_target_hashes=[_hash("e")],
        constraint_units_hash=_hash("f"),
        zero_mode_policy_hash=_hash("1"),
        sector_policy_hash=_hash("2"),
        lagrange_multiplier_hash=_hash("3"),
        release_law_hash=_hash("4"),
        no_data_ledger_hash=_hash("5"),
        refinement_map_hash=_hash("6"),
    )


def _selector() -> AnomalyAbundanceSelectorArtifact:
    return AnomalyAbundanceSelectorArtifact(
        artifact_id="selector",
        regulator_id="r1",
        parent_generation_id="gen1",
        release_state_artifact_id="release",
        release_law_artifact_id="law",
        load_observable_artifact_id="load",
        quotient_ensemble_hash=_hash("7"),
        P_star_hash=_hash("8"),
        N_CRC_hash=_hash("9"),
        selected_load_energy_hash=_hash("a"),
        rho_A_bar_hash=_hash("b"),
        refinement_residual_hash=_hash("c"),
        no_data_use_receipt_hash=_hash("d"),
        source_claim_label=SOURCE_ONLY_ANOMALY_ABUNDANCE,
    )


def _refinement() -> LoadRefinementCompatibilityArtifact:
    return LoadRefinementCompatibilityArtifact(
        artifact_id="refinement",
        fine_regulator_id="r2",
        coarse_regulator_id="r1",
        coarse_map_hash=_hash("e"),
        fine_release_law_hash=_hash("f"),
        coarse_release_law_hash=_hash("1"),
        law_tv_defect=0.0,
        load_naturality_defect=0.0,
        load_sup_bound=1.0,
        selected_load_difference_bound=0.0,
        passes_exact=True,
    )


def _hash(seed: str) -> str:
    return "sha256:" + seed * 64
