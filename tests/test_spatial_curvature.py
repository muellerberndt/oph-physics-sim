from __future__ import annotations

import math

from oph_fpe.cosmology.spatial_curvature import (
    DensitySource,
    friedmann_curvature_readout,
    s3_holonomy_spatial_curvature_gate,
    spatial_curvature_status_report,
)


HASH = "sha256:" + "1" * 64


def test_default_spatial_curvature_status_is_open_and_degenerate():
    report = spatial_curvature_status_report(
        omega_lambda_oph=0.68,
        omega_b0=0.05,
        omega_nu0=0.002,
        omega_r0=0.0001,
    )

    assert report["status"] == "OPEN_THEOREM"
    assert report["geometry_branch"] == "UNRESOLVED"
    assert report["exact_flat_sector_selection"] is False
    assert report["selected_Omega_K"] is None
    assert report["Omega_A0"] is None
    assert math.isclose(report["Omega_A0_plus_Omega_K0"], 0.2679)
    assert report["friedmann_curvature_independence_receipt"] is False


def test_explicit_flat_branch_is_not_an_exact_theorem():
    report = spatial_curvature_status_report(
        geometry_branch="FLAT_ASSUMED",
        omega_lambda_oph=0.68,
        omega_b0=0.05,
        omega_nu0=0.002,
        omega_r0=0.0001,
    )

    assert report["status"] == "EXPLICIT_ASSUMPTION"
    assert report["geometry_branch"] == "FLAT_ASSUMED"
    assert report["flat_branch_assumption"] is True
    assert report["exact_flat_sector_selection"] is False
    assert report["selected_Omega_K"] == 0.0
    assert report["Omega_K_source"] == DensitySource.EXPLICIT_ASSUMPTION.value
    assert report["Omega_A_source"] == DensitySource.RESIDUAL_DEFINITION.value


def test_conditional_cmh_requires_full_certificate_hashes():
    missing = spatial_curvature_status_report(conditional_cmh=True)
    assert missing["exact_flat_sector_selection"] is False
    assert "clock_boundary_theorem_proof_hashes" in missing["blockers"]

    report = spatial_curvature_status_report(
        conditional_cmh=True,
        flat_extension_exists=True,
        quotient_refinement_naturality=True,
        curvature_functional_certified=True,
        zero_set_unique=True,
        clock_hash=HASH,
        boundary_packet_hash=HASH,
        theorem_hash=HASH,
        proof_certificate_hash=HASH,
        topology_policy="R3_ONLY",
    )
    assert report["status"] == "CONDITIONAL_CMH"
    assert report["geometry_branch"] == "FLAT_EXACT"
    assert report["exact_flat_sector_selection"] is True
    assert report["selected_K"] == 0.0
    assert report["blockers"] == ()


def test_s3_screen_defect_does_not_pass_spatial_curvature_gate():
    report = s3_holonomy_spatial_curvature_gate({"structure_group": "S3", "connection_type": "screen"})

    assert report["S3_SCREEN_DEFECT_IS_SPATIAL_CURVATURE_RECEIPT"] is False
    assert report["spatial_levi_civita_interpretation"] is False
    assert "S3_SCREEN_DEFECT_NOT_SPATIAL_LEVI_CIVITA_HOLONOMY" in report["blockers"]


def test_friedmann_curvature_readout_rejects_residual_circularity():
    report = friedmann_curvature_readout(
        H=70.0,
        rho_total=1.0,
        G=1.0,
        Lambda=0.0,
        ancestry_labels=["OMEGA_A_CLOSURE_RESIDUAL"],
        unit_consistency_receipt=True,
        clock_epoch_alignment_receipt=True,
    )

    assert report["friedmann_curvature_readout_receipt"] is False
    assert "CIRCULAR_FRIEDMANN_ANCESTRY" in report["blockers"]
