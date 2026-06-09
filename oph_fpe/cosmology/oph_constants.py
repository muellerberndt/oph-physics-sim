from __future__ import annotations

from dataclasses import dataclass
import math

from oph_fpe.constants.oph_pixel import P_STAR

DEFAULT_R_DS_M = 1.66e26
DEFAULT_L_PLANCK_M = 1.616e-35
DEFAULT_N_CRC = math.pi * (DEFAULT_R_DS_M / DEFAULT_L_PLANCK_M) ** 2


@dataclass(frozen=True)
class OPHConstants:
    """Constants shared by OPH cosmology diagnostic lanes.

    These are target/readout constants. They do not by themselves close the
    finite-lattice derivation gate for B_A(k,a), neutrino masses, or a physical
    Boltzmann likelihood.
    """

    P: float = P_STAR
    N_CRC: float = DEFAULT_N_CRC
    pi_wl: float = 5.0 / 7.0
    mnu_eV: tuple[float, float, float] = (
        0.017454720257976796,
        0.019481987935919015,
        0.05307522145074924,
    )
    N_eff: float = 3.044
    S8_oph_compressed: float = 0.828924043
    S8_wl_target_reference: float = 0.790
    S8_wl_sigma_reference: float = 0.016

    @property
    def lambda_collar(self) -> float:
        return math.exp(-float(self.P) / 24.0)

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

    def as_jsonable(self) -> dict[str, float | list[float]]:
        return {
            "P": float(self.P),
            "N_CRC": float(self.N_CRC),
            "N_patch_bare_ratio": self.N_patch_bare_ratio,
            "Lambda_lP2": self.Lambda_lP2,
            "P_cell_count_for_N_CRC": self.P_cell_count_for_N_CRC,
            "lambda_collar": self.lambda_collar,
            "reserve": self.reserve,
            "pi_wl": float(self.pi_wl),
            "epsilon_A_wl": self.epsilon_A_wl,
            "R_wl": self.R_wl,
            "mnu_eV": [float(value) for value in self.mnu_eV],
            "sum_mnu_eV": float(sum(self.mnu_eV)),
            "N_eff": float(self.N_eff),
            "S8_oph_compressed": float(self.S8_oph_compressed),
            "S8_wl_target_reference": float(self.S8_wl_target_reference),
            "S8_wl_sigma_reference": float(self.S8_wl_sigma_reference),
            "S8_projected_wl": self.S8_projected_wl,
        }
