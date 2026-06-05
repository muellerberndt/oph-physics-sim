from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ALPHA_INV_SOURCE_CANDIDATE = 136.9948351646
ALPHA_INV_ENDPOINT_CALIBRATED = 137.035999177
P_SOURCE_CANDIDATE = (1.0 + math.sqrt(5.0)) / 2.0 + math.sqrt(math.pi) / ALPHA_INV_SOURCE_CANDIDATE
P_ENDPOINT_CALIBRATED = 1.6309682094039593
P_STAR = P_ENDPOINT_CALIBRATED

PIXEL_MODE_SOURCE_CANDIDATE = "source_candidate"
PIXEL_MODE_ENDPOINT_CALIBRATED = "endpoint_calibrated"


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
        )
    if mode in {PIXEL_MODE_ENDPOINT_CALIBRATED, "endpoint_public"}:
        return OPHPixelConstants(
            P=P_ENDPOINT_CALIBRATED,
            source="endpoint_public",
            pixel_mode=PIXEL_MODE_ENDPOINT_CALIBRATED,
        )
    raise ValueError(f"unknown OPH pixel mode: {pixel_mode}")


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
