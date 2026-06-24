from __future__ import annotations

from enum import Enum


class ClaimTier(Enum):
    DIAGNOSTIC_PROXY = "diagnostic_proxy"
    CONDITIONAL_PHYSICAL = "conditional_physical"
    OPH_NATIVE_PHYSICAL = "oph_native_physical"


class GeometryOrigin(Enum):
    UNKNOWN = "UNKNOWN"
    EXTERNAL_FIDUCIAL = "EXTERNAL_FIDUCIAL"
    IMPORTED_FLRW = "IMPORTED_FLRW"
    OPH_NATIVE = "OPH_NATIVE"


def normalize_claim_tier(value: object) -> ClaimTier:
    if isinstance(value, ClaimTier):
        return value
    text = str(value or "").strip()
    text_lower = text.lower()
    for tier in ClaimTier:
        if text in {tier.value, tier.name} or text_lower in {tier.value.lower(), tier.name.lower()}:
            return tier
    return ClaimTier.DIAGNOSTIC_PROXY


def normalize_geometry_origin(value: object) -> GeometryOrigin:
    if isinstance(value, GeometryOrigin):
        return value
    text = str(value or "").strip()
    text_lower = text.lower()
    for origin in GeometryOrigin:
        if text in {origin.value, origin.name} or text_lower in {origin.value.lower(), origin.name.lower()}:
            return origin
    return GeometryOrigin.UNKNOWN
