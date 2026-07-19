from __future__ import annotations

from dataclasses import dataclass
import math

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.neutrino_status import (
    CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION,
    CONVENTIONAL_CAMB_SUM_MNU_EV,
)

DEFAULT_R_DS_M = 1.66e26
DEFAULT_L_PLANCK_M = 1.616e-35
DEFAULT_OBSERVED_HORIZON_N_COMPARISON = math.pi * (
    DEFAULT_R_DS_M / DEFAULT_L_PLANCK_M
) ** 2
# Compatibility alias used by historical diagnostics.  It is not an OPH
# public-record-capacity producer; physical N must come from the exact
# PUBLIC_CHECKPOINT_PACKET evaluator and its closure receipts.
DEFAULT_N_CRC = DEFAULT_OBSERVED_HORIZON_N_COMPARISON
LAMBDA_COLLAR_EXACT_GATE = "UNIFORM_PRODUCT_THICKENING_EXACT"


@dataclass(frozen=True)
class OPHConstants:
    """Constants shared by OPH cosmology diagnostic lanes.

    These are comparison/readout constants.  In particular, ``N_CRC`` defaults
    to an observed-horizon comparison and cannot define public-record capacity.
    They do not by themselves close the finite-lattice derivation gate for
    B_A(k,a), neutrino masses, physical N, or a physical Boltzmann likelihood.
    """

    P: float = P_STAR
    N_CRC: float = DEFAULT_N_CRC
    N_CRC_source: str = "observed_horizon_comparison"
    N_CRC_producer_eligible: bool = False
    pi_wl: float = 5.0 / 7.0
    conventional_camb_sum_mnu_eV: float = CONVENTIONAL_CAMB_SUM_MNU_EV
    N_eff: float = 3.044
    S8_oph_compressed: float = 0.828924043
    S8_wl_target_reference: float = 0.790
    S8_wl_sigma_reference: float = 0.016

    @property
    def z6_normalized_trace_mean(self) -> float:
        return float(self.P) / 24.0

    @property
    def z6_reciprocal_trace(self) -> float:
        return 24.0 / float(self.P)

    @property
    def lambda_collar_exact_uniform_product_thickening(self) -> float:
        """Exact-uniform/product-thickening diagnostic target.

        Finite-thickness runs must promote this through the
        UNIFORM_PRODUCT_THICKENING_EXACT gate before using it as a physical
        local coefficient.
        """

        return math.exp(-float(self.P) / 24.0)

    @property
    def lambda_collar(self) -> float:
        return self.lambda_collar_exact_uniform_product_thickening

    @property
    def lambda_collar_exact_gate(self) -> str:
        return LAMBDA_COLLAR_EXACT_GATE

    @property
    def lambda_collar_claim_status(self) -> str:
        return "diagnostic_exact_uniform_product_thickening_target_not_unconditional"

    @property
    def lambda_collar_profile_default(self) -> str:
        return "lambda_collar = integral dy w(y) exp[-epsilon_Z6(y)]"

    @property
    def finite_thickness_jensen_band(self) -> list[float]:
        return [self.lambda_collar_exact_uniform_product_thickening, 1.0]

    @property
    def reserve(self) -> float:
        return 1.0 - self.lambda_collar

    @property
    def epsilon_A_wl(self) -> float:
        return self.pi_wl * self.reserve

    @property
    def R_wl(self) -> float:
        return 1.0 - self.epsilon_A_wl

    @property
    def S8_projected_wl(self) -> float:
        return self.S8_oph_compressed * self.R_wl

    @property
    def N_patch_bare_ratio(self) -> float:
        return float(self.N_CRC) / math.pi

    @property
    def Lambda_lP2(self) -> float:
        return 3.0 * math.pi / float(self.N_CRC)

    @property
    def P_cell_count_for_N_CRC(self) -> float:
        return 4.0 * float(self.N_CRC) / float(self.P)

    def as_jsonable(self) -> dict[str, object]:
        return {
            "P": float(self.P),
            "N_CRC": float(self.N_CRC),
            "N_CRC_source": self.N_CRC_source,
            "N_CRC_producer_eligible": bool(self.N_CRC_producer_eligible),
            "N_CRC_contract": (
                "comparison only; physical N requires exact target-free public-record "
                "capacity, complete-fiber scalarization, robust closure, unique "
                "regulator-stable slack zero, and horizon-record saturation"
            ),
            "N_patch_bare_ratio": self.N_patch_bare_ratio,
            "Lambda_lP2": self.Lambda_lP2,
            "P_cell_count_for_N_CRC": self.P_cell_count_for_N_CRC,
            "z6_normalized_trace_mean": self.z6_normalized_trace_mean,
            "z6_reciprocal_trace": self.z6_reciprocal_trace,
            "lambda_collar": self.lambda_collar,
            "lambda_collar_exact_uniform_product_thickening": (
                self.lambda_collar_exact_uniform_product_thickening
            ),
            "lambda_collar_exact_gate": self.lambda_collar_exact_gate,
            "lambda_collar_claim_status": self.lambda_collar_claim_status,
            "lambda_collar_profile_default": self.lambda_collar_profile_default,
            "finite_thickness_jensen_band": self.finite_thickness_jensen_band,
            "reserve": self.reserve,
            "pi_wl": float(self.pi_wl),
            "epsilon_A_wl": self.epsilon_A_wl,
            "R_wl": self.R_wl,
            "oph_neutrino_mass_prediction": {
                "available": False,
                "masses_eV": None,
                "sum_mnu_eV": None,
                "public_promotion_allowed": False,
                "status": "no_source_derived_neutrino_mass_prediction",
            },
            "conventional_camb_neutrino_baseline": {
                "assumption": CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION,
                "sum_mnu_eV": float(self.conventional_camb_sum_mnu_eV),
                "counts_as_oph_prediction": False,
            },
            "N_eff": float(self.N_eff),
            "S8_oph_compressed": float(self.S8_oph_compressed),
            "S8_wl_target_reference": float(self.S8_wl_target_reference),
            "S8_wl_sigma_reference": float(self.S8_wl_sigma_reference),
            "S8_projected_wl": self.S8_projected_wl,
        }
