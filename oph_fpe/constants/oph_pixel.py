from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np


ALPHA_INV_SOURCE_CANDIDATE = 136.9948351646
ALPHA_INV_ENDPOINT_CALIBRATED = 137.035999177
P_SOURCE_CANDIDATE = (1.0 + math.sqrt(5.0)) / 2.0 + math.sqrt(math.pi) / ALPHA_INV_SOURCE_CANDIDATE
P_ENDPOINT_CALIBRATED = 1.6309682094039593
P_STAR = P_ENDPOINT_CALIBRATED

P_EMPIRICAL_HADRON_CLOSURE_CENTRAL = 1.6310415203592243
P_EMPIRICAL_HADRON_CLOSURE_INTERVAL = (1.6310314631270647, 1.6310515775913838)

PIXEL_MODE_SOURCE_CANDIDATE = "source_candidate"
PIXEL_MODE_ENDPOINT_CALIBRATED = "endpoint_calibrated"


class PixelParameterProfile(str, Enum):
    SOURCE_CANDIDATE = "source_candidate"
    HIERARCHY_PUBLIC = "hierarchy_public"
    EMPIRICAL_HADRON_CLOSURE = "empirical_hadron_closure"
    MEASURED_COMPARISON = "measured_comparison"


@dataclass(frozen=True)
class PixelProfileSpec:
    profile: PixelParameterProfile
    P: float | None
    interval: tuple[float, float] | None
    epistemic_class: str
    simulation_role: str
    recovered_core_allowed: bool = False
    public_prediction_allowed: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "P": self.P,
            "interval": list(self.interval) if self.interval is not None else None,
            "epistemic_class": self.epistemic_class,
            "simulation_role": self.simulation_role,
            "recovered_core_allowed": self.recovered_core_allowed,
            "public_prediction_allowed": self.public_prediction_allowed,
        }


PIXEL_PARAMETER_PROFILES: dict[PixelParameterProfile, PixelProfileSpec] = {
    PixelParameterProfile.SOURCE_CANDIDATE: PixelProfileSpec(
        PixelParameterProfile.SOURCE_CANDIDATE,
        P_SOURCE_CANDIDATE,
        None,
        "source_candidate_audit",
        "source-directed diagnostic; no theorem promotion",
    ),
    PixelParameterProfile.HIERARCHY_PUBLIC: PixelProfileSpec(
        PixelParameterProfile.HIERARCHY_PUBLIC,
        P_ENDPOINT_CALIBRATED,
        None,
        "endpoint_calibrated_hierarchy_surface",
        "legacy public hierarchy comparison",
    ),
    PixelParameterProfile.EMPIRICAL_HADRON_CLOSURE: PixelProfileSpec(
        PixelParameterProfile.EMPIRICAL_HADRON_CLOSURE,
        P_EMPIRICAL_HADRON_CLOSURE_CENTRAL,
        P_EMPIRICAL_HADRON_CLOSURE_INTERVAL,
        "external_data_hadron_closure",
        "empirical uncertainty propagation only",
    ),
    PixelParameterProfile.MEASURED_COMPARISON: PixelProfileSpec(
        PixelParameterProfile.MEASURED_COMPARISON,
        P_ENDPOINT_CALIBRATED,
        None,
        "measured_endpoint_comparison",
        "comparison only; forbidden in generative receipts",
    ),
}


@dataclass(frozen=True)
class OPHPixelConstants:
    """OPH local pixel/cell constants.

    P is the local screen-cell area in Planck-area units. It calibrates
    area/capacity weights and later quantitative readouts. It is not used to
    change the BW normalization or force a 3D dimension estimate.
    """

    P: float = P_STAR
    source: str = "endpoint_public"
    pixel_mode: str = PIXEL_MODE_ENDPOINT_CALIBRATED
    epistemic_profile: str = PixelParameterProfile.HIERARCHY_PUBLIC.value
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0
    sqrt_pi: float = math.sqrt(math.pi)

    @property
    def alpha_from_P(self) -> float:
        return (self.P - self.phi) / self.sqrt_pi

    @property
    def alpha_inverse_from_P(self) -> float:
        alpha = self.alpha_from_P
        if abs(alpha) < 1e-30:
            return float("inf")
        return 1.0 / alpha

    @property
    def cell_area_planck(self) -> float:
        return self.P

    @property
    def cell_entropy_capacity(self) -> float:
        return self.P / 4.0

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "P": self.P,
            "P_source": self.source,
            "pixel_mode": self.pixel_mode,
            "epistemic_profile": self.epistemic_profile,
            "profile": pixel_parameter_profile(self.epistemic_profile).as_jsonable(),
            "phi": self.phi,
            "sqrt_pi": self.sqrt_pi,
            "alpha_from_P": self.alpha_from_P,
            "alpha_inverse_from_P": self.alpha_inverse_from_P,
            "source_candidate": {
                "alpha_inverse": ALPHA_INV_SOURCE_CANDIDATE,
                "P": P_SOURCE_CANDIDATE,
            },
            "endpoint_calibrated": {
                "alpha_inverse": ALPHA_INV_ENDPOINT_CALIBRATED,
                "P": P_ENDPOINT_CALIBRATED,
            },
            "role": (
                "local screen-cell area and entropy/capacity normalization; "
                "not a BW normalization factor and not a dimension-forcing input. "
                "endpoint_calibrated values must stay out of recovered-core receipts"
            ),
        }


def pixel_constants_for_mode(pixel_mode: str) -> OPHPixelConstants:
    mode = str(pixel_mode).lower().replace("-", "_")
    if mode == PIXEL_MODE_SOURCE_CANDIDATE:
        return OPHPixelConstants(
            P=P_SOURCE_CANDIDATE,
            source="source_candidate",
            pixel_mode=PIXEL_MODE_SOURCE_CANDIDATE,
            epistemic_profile=PixelParameterProfile.SOURCE_CANDIDATE.value,
        )
    if mode in {PIXEL_MODE_ENDPOINT_CALIBRATED, "endpoint_public"}:
        return OPHPixelConstants(
            P=P_ENDPOINT_CALIBRATED,
            source="endpoint_public",
            pixel_mode=PIXEL_MODE_ENDPOINT_CALIBRATED,
            epistemic_profile=PixelParameterProfile.HIERARCHY_PUBLIC.value,
        )
    raise ValueError(f"unknown OPH pixel mode: {pixel_mode}")


def pixel_parameter_profile(profile: str | PixelParameterProfile) -> PixelProfileSpec:
    try:
        key = profile if isinstance(profile, PixelParameterProfile) else PixelParameterProfile(str(profile))
    except ValueError as exc:
        raise ValueError(f"unknown OPH pixel parameter profile: {profile}") from exc
    return PIXEL_PARAMETER_PROFILES[key]


def pixel_constants_for_profile(profile: str | PixelParameterProfile) -> OPHPixelConstants:
    spec = pixel_parameter_profile(profile)
    if spec.profile is PixelParameterProfile.EMPIRICAL_HADRON_CLOSURE:
        raise ValueError(
            "empirical_hadron_closure is an interval-valued comparison profile; "
            "select a sampled value explicitly and preserve its interval provenance"
        )
    if spec.P is None:
        raise ValueError(f"pixel profile {spec.profile.value} has no scalar P")
    return OPHPixelConstants(
        P=spec.P,
        source=spec.epistemic_class,
        pixel_mode=spec.profile.value,
        epistemic_profile=spec.profile.value,
    )


def equal_cell_area_planck(patch_count: int, P: float = P_STAR) -> np.ndarray:
    return np.full(int(patch_count), float(P), dtype=np.float64)


def equal_cell_entropy(patch_count: int, P: float = P_STAR) -> np.ndarray:
    return np.full(int(patch_count), float(P) / 4.0, dtype=np.float64)


def total_area_planck(patch_count: int, P: float = P_STAR) -> float:
    return float(int(patch_count) * float(P))


def total_entropy_capacity(patch_count: int, P: float = P_STAR) -> float:
    return float(int(patch_count) * float(P) / 4.0)


def screen_radius_planck(patch_count: int, P: float = P_STAR) -> float:
    return math.sqrt((int(patch_count) * float(P)) / (4.0 * math.pi))


def scale_factor_from_patch_count(N_t: int, N_0: int) -> float:
    if int(N_0) <= 0:
        raise ValueError("N_0 must be positive")
    return math.sqrt(int(N_t) / int(N_0))


def cap_area_planck(cap_weights: np.ndarray, cell_area_planck: float | np.ndarray = P_STAR) -> float:
    weights = np.asarray(cap_weights, dtype=np.float64)
    area = np.asarray(cell_area_planck, dtype=np.float64)
    return float(np.sum(area * weights))


def cap_entropy_capacity(cap_weights: np.ndarray, cell_entropy: float | np.ndarray = P_STAR / 4.0) -> float:
    weights = np.asarray(cap_weights, dtype=np.float64)
    entropy = np.asarray(cell_entropy, dtype=np.float64)
    return float(np.sum(entropy * weights))
