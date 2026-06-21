from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MeasurementTarget:
    name: str
    lane: str
    source_url: str
    status: str
    claim_boundary: str

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


TARGETS = {
    "sparc": MeasurementTarget(
        name="SPARC",
        lane="galaxy_proxy",
        source_url="https://astroweb.case.edu/SPARC/",
        status="external_table_required",
        claim_boundary="RAR/BTFR target metadata only; no SPARC rows are bundled with this package",
    ),
    "planck2018_low_l_tt": MeasurementTarget(
        name="Planck2018_low_l_TT_shape",
        lane="cosmo_proxy",
        source_url="https://arxiv.org/abs/1807.06209",
        status="external_table_required",
        claim_boundary="shape-only benchmark metadata; use a local binned TT table for comparisons",
    ),
    "planck2018_tt_te_ee_lensing": MeasurementTarget(
        name="Planck2018_TT_TE_EE_lensing",
        lane="oph_cmb_anomaly_stress_adapter",
        source_url="https://arxiv.org/abs/1807.06209",
        status="full_likelihood_required",
        claim_boundary=(
            "physical CMB target metadata only; requires CAMB/CLASS OPH anomaly module, "
            "full covariance likelihoods, and CDM-limit regression"
        ),
    ),
    "planck2018_scalar_tilt": MeasurementTarget(
        name="Planck2018_scalar_tilt_ns",
        lane="oph_screen_power_effective_theory",
        source_url="https://arxiv.org/abs/1807.06209",
        status="screen_eta_R_regression_target",
        claim_boundary=(
            "Target metadata for eta_R = 1 - n_s. This is a screen-theory parameter target, "
            "not a CMB likelihood and not a physical TT/TE/EE prediction."
        ),
    ),
    "planck2018_tt_camb_lcdm_baseline": MeasurementTarget(
        name="Planck2018_TT_CAMB_LambdaCDM_baseline",
        lane="camb_lcdm_baseline",
        source_url="https://arxiv.org/abs/1807.06209",
        status="local_binned_tt_regression_enabled",
        claim_boundary=(
            "standard LambdaCDM CAMB regression target; useful for Boltzmann plumbing only, "
            "not an OPH prediction"
        ),
    ),
    "act_dr6_lensing": MeasurementTarget(
        name="ACT_DR6_CMB_lensing_shape",
        lane="cosmo_proxy",
        source_url="https://arxiv.org/abs/2304.05202",
        status="external_table_required",
        claim_boundary="late-structure target metadata only; no lensing table is bundled",
    ),
    "desi_dr2_bao": MeasurementTarget(
        name="DESI_DR2_BAO",
        lane="background_adapter",
        source_url="https://arxiv.org/abs/2503.14738",
        status="adapter_not_enabled",
        claim_boundary="future background-adapter metadata only",
    ),
    "oph_compressed_cmb_bao_growth": MeasurementTarget(
        name="OPH_compressed_CMB_BAO_growth_S8_reference",
        lane="compressed_likelihood_reference",
        source_url="local:cosmology/correspondence/inflation/inflation.md",
        status="reference_diagnostic_available",
        claim_boundary=(
            "Compressed diagnostic regression target only. Keeps S8/weak-lensing tension visible; "
            "does not replace full Planck/BAO/lensing likelihoods."
        ),
    ),
    "oph_cmb_success_ladder_v04": MeasurementTarget(
        name="OPH_CMB_success_ladder_v0_4",
        lane="oph_inflation_cmb_bridge",
        source_url="local:cosmology/correspondence/cmb/2",
        status="public_data_diagnostic_available",
        claim_boundary=(
            "Planck low-ell TT/CAMB/full-sky-MC diagnostic ladder imported from Pro notes. "
            "This is measured-data comparison, not an official likelihood and not a finite-lattice "
            "derivation of the fitted OPH IR/parity parameters."
        ),
    ),
    "oph_p48_screen_spectrum": MeasurementTarget(
        name="OPH_P_over_48_screen_spectrum",
        lane="oph_inflation_cmb_bridge",
        source_url="local:cosmology/correspondence/inflation/2/comms.md",
        status="conditional_screen_theorem_target_available",
        claim_boundary=(
            "Conditional finite-screen target for the MaxEnt Gaussian screen covariance and "
            "n_s=1-P/48 tilt after scalar-readout and repair-operator receipts. Scalar release "
            "amplitude, screen-to-primordial lift, and physical TT/TE/EE spectra remain separate gates."
        ),
    ),
    "pantheon_plus": MeasurementTarget(
        name="PantheonPlus",
        lane="background_adapter",
        source_url="https://arxiv.org/abs/2202.04077",
        status="adapter_not_enabled",
        claim_boundary="future supernova-distance adapter metadata only",
    ),
}


def measurement_target(name: str) -> MeasurementTarget:
    key = str(name).lower().replace("-", "_")
    if key not in TARGETS:
        raise KeyError(f"unknown measurement target: {name}")
    return TARGETS[key]


def target_registry() -> dict[str, dict[str, Any]]:
    return {name: target.as_jsonable() for name, target in TARGETS.items()}
