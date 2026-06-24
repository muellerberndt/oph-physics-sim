from __future__ import annotations

from oph_fpe.cosmology.cosmological_scale_bridge import (
    imported_flrw_reference_receipts,
    validate_physical_scale_bridge_receipts,
)


def test_imported_flrw_reference_receipt_passes_conditional_scale_bridge() -> None:
    validation = validate_physical_scale_bridge_receipts(imported_flrw_reference_receipts())

    assert validation["PHYSICAL_SCALE_BRIDGE_RECEIPT"] is True
    assert validation["PHYSICAL_K_RECEIPT"] is True
    assert validation["SOURCE_ANGULAR_MODE_RECEIPT"] is True
    assert validation["CALIBRATED_A_EVOLUTION_RECEIPT"] is True
    assert validation["PHYSICAL_FREEZEOUT_SURFACE_RECEIPT"] is True
    assert validation["claim_tier"] == "conditional_physical"
    assert validation["geometry_origin"] == "IMPORTED_FLRW"
    assert validation["oph_native_k_derivation"] is False
    assert validation["blockers"] == []


def test_scale_bridge_rejects_mismatched_claim_tier_and_geometry_origin() -> None:
    receipt = imported_flrw_reference_receipts()
    receipt["claim_tier"] = "oph_native_physical"

    validation = validate_physical_scale_bridge_receipts(receipt)

    assert validation["PHYSICAL_SCALE_BRIDGE_RECEIPT"] is False
    assert "oph_native_geometry_origin_missing" in validation["blockers"]
    assert "claim_tier_geometry_origin_mismatch" in validation["blockers"]


def test_scale_bridge_rejects_bare_boolean_flags_without_hashes() -> None:
    validation = validate_physical_scale_bridge_receipts(
        {
            "claim_tier": "conditional_physical",
            "geometry_origin": "imported_flrw",
            "PHYSICAL_K_RECEIPT": True,
            "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": True,
            "NO_POSTHOC_CALIBRATION_RECEIPT": True,
        }
    )

    assert validation["PHYSICAL_SCALE_BRIDGE_RECEIPT"] is False
    assert "physical_scale_bridge_receipt_missing" not in validation["blockers"]
    assert "geometry_hash_missing" in validation["blockers"]
    assert "no_peak_fit_receipt_hash_missing" in validation["blockers"]
