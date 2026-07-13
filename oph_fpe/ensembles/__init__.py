"""Reference ensemble baselines for OPH-FPE."""

from oph_fpe.ensembles.reference_vacuum import (
    EnsembleSpec,
    free_scalar_ensemble_spec,
    harmonic_gaussian_reference_report,
    sample_harmonic_coefficients,
    u1_lattice_gauge_reference_report,
    write_reference_vacuum_baseline_report,
)
from oph_fpe.ensembles.quotient_ensemble import (
    QuotientEnsembleManifest,
    canonicalize_representative,
    canonicalizer_receipt,
    claim_tier_gate,
    fail_closed_promotion_receipts,
    quotient_lumpability_receipt,
    representative_lift_firewall_receipt,
    rg_exponential_family_closure_receipt,
    sampler_correctness_receipt,
)

__all__ = [
    "EnsembleSpec",
    "QuotientEnsembleManifest",
    "canonicalize_representative",
    "canonicalizer_receipt",
    "claim_tier_gate",
    "fail_closed_promotion_receipts",
    "free_scalar_ensemble_spec",
    "harmonic_gaussian_reference_report",
    "quotient_lumpability_receipt",
    "representative_lift_firewall_receipt",
    "rg_exponential_family_closure_receipt",
    "sampler_correctness_receipt",
    "sample_harmonic_coefficients",
    "u1_lattice_gauge_reference_report",
    "write_reference_vacuum_baseline_report",
]
