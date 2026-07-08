from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from .active_band import ChernBandCertificate, chern_integer_stability
from .hamiltonian import PhaseCertificate, promote_hamiltonian_to_ledger
from .identifiability import identify_optical_sector
from .line_fan import LineFanPeak, fractional_optical_slope_certificate, line_fan_from_module
from .manybody import ManyBodyCertificate, manybody_gap_certificate
from .no_target_leak import no_target_leak_audit
from .optical_module import OpticalModuleLedger, OpticalSector
from .presentation import FractionalMaterialPresentation
from .quotient import (
    QuotientSchema,
    canonicalizer_idempotence,
    no_orbit_size_bias,
    quotient_lumpability,
    representative_invariance,
)
from .receipts import fail
from .refinement import refinement_compatibility
from .source_law import SourceLaw, require_frozen
from .topological_ledger import TopologicalLedger, abelian_k_matrix_readout


def compare_after_freeze(source: SourceLaw, predictions: Mapping[str, float], measurements: Mapping[str, float]) -> dict:
    frozen = require_frozen(source)
    if frozen["status"] != "pass":
        return frozen
    residuals = {
        key: float(predictions[key]) - float(measurements[key])
        for key in predictions
        if key in measurements
    }
    return {
        "status": "pass",
        "SOURCE_HAMILTONIAN_FROZEN": True,
        "residuals": residuals,
        "claim_boundary": "Comparison is allowed because the source law was frozen first.",
    }


def demo_fractional_report() -> dict:
    """Small deterministic fractional sandbox report for viewer and receipt tests."""

    presentation = FractionalMaterialPresentation(
        material_id="twisted_tmd_fractional_sandbox",
        regulator=12,
        representatives=("a0", "a1", "vac"),
        quotient_map={"a0": "anyon_e_over_3", "a1": "anyon_e_over_3", "vac": "vacuum"},
    )
    source = SourceLaw(action={"laughlin_e_over_3": 0.0}, frozen=True)
    schema = QuotientSchema(
        canonical={"a0": "anyon_e_over_3", "a1": "anyon_e_over_3", "vac": "vacuum"},
        orbit_sizes={"anyon_e_over_3": 1, "vacuum": 1},
        transition_kernel={
            "a0": {"a1": 0.5, "vac": 0.5},
            "a1": {"a0": 0.5, "vac": 0.5},
            "vac": {"vac": 1.0},
        },
    )
    ledger = TopologicalLedger(
        sectors=("vacuum", "anyon_e_over_3", "neutral_fractional_exciton"),
        charges={"vacuum": Fraction(0), "anyon_e_over_3": Fraction(1, 3), "neutral_fractional_exciton": Fraction(0)},
        spins={"anyon_e_over_3": Fraction(1, 3), "neutral_fractional_exciton": Fraction(1, 3)},
        fusion={"anyon_e_over_3": ("vacuum",), "neutral_fractional_exciton": ("anyon_e_over_3",)},
        k_matrix=((3,),),
        t_vector=(1,),
        edge_spectrum_certified=True,
    )
    cert = PhaseCertificate(
        source_hamiltonian_frozen=True,
        active_band_projector=True,
        chern_number=True,
        band_geometry=True,
        manybody_gap=True,
        ground_sector_degeneracy=True,
        flux_insertion_pump=True,
        hall_conductance=True,
        edge_spectrum=True,
        topological_sector_ledger=True,
        refinement_stability=True,
        no_target_leak=True,
        certificate_map_injective=True,
    )
    module = OpticalModuleLedger(
        quotient_descended_operators=True,
        sectors=(
            OpticalSector(
                label="neutral_fractional_exciton",
                tau="anyon_e_over_3",
                total_charge=0.0,
                energy=1.61,
                intensity=0.72,
                polarization="sigma+",
                binding_derivative_bound=0.03,
                eta="demo",
            ),
            OpticalSector(
                label="anyon_trion",
                tau="anyon_e_over_3",
                total_charge=1.0 / 3.0,
                energy=1.55,
                intensity=0.41,
                polarization="sigma-",
                binding_derivative_bound=0.04,
                eta="demo",
            ),
        ),
    )
    fan = line_fan_from_module(module)
    peak = LineFanPeak(**fan["peaks"][0])
    peaks = [LineFanPeak(**payload) for payload in fan["peaks"]]
    leak = no_target_leak_audit(
        source_nodes=["source_hamiltonian", "active_band", "interaction"],
        target_nodes=["optical_measurement"],
        edges=[("source_hamiltonian", "phase_certificate"), ("active_band", "phase_certificate")],
    )
    if leak["status"] != "pass":
        return fail("TARGET_LEAK_DETECTED", details=leak)
    return {
        "schema": "oph_fractional_quotient_sim_report_v1",
        "mode": "fractional_quotient_sector_sandbox",
        "claim_boundary": (
            "Diagnostic fractional quotient-sector sandbox. The report demonstrates quotient and "
            "line-fan receipt logic; it is not a material proof for a real sample."
        ),
        "material_presentation": presentation.to_report(),
        "source": require_frozen(source),
        "quotient": {
            "canonicalizer": canonicalizer_idempotence(schema),
            "representative_invariance": representative_invariance(schema, lambda state: schema.canonicalize(state)),
            "lumpability": quotient_lumpability(schema),
            "orbit_size_bias": no_orbit_size_bias(schema, {"anyon_e_over_3": 1.0, "vacuum": 1.0}),
        },
        "active_band": chern_integer_stability(
            ChernBandCertificate(
                active_band_projector=True,
                chern_number=1,
                chern_stable=True,
                band_geometry_bound=0.2,
            )
        ),
        "manybody": manybody_gap_certificate(
            ManyBodyCertificate(
                gap=0.04,
                ground_sector_degeneracy=3,
                expected_degeneracy=3,
                flux_pump_charge=1 / 3,
                hall_conductance=1 / 3,
            )
        ),
        "topological_ledger": ledger.to_report(),
        "abelian_readout": abelian_k_matrix_readout(((3,),), (1,), (1,)),
        "phase_promotion": promote_hamiltonian_to_ledger(cert, ledger),
        "optical_module": module.to_report(),
        "line_fan": fan,
        "slope_certificate": fractional_optical_slope_certificate(
            peak,
            binding_derivative_bound=0.03,
        ),
        "identifiability": identify_optical_sector(peaks),
        "refinement": refinement_compatibility(0.01, max_defect=0.03),
        "no_target_leak": leak,
    }
