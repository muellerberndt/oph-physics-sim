from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Literal


GammaRoute = Literal["TRANSPORTED_STRESS", "BOUNDARY_RECORD_DIPOLE", "BOTH"]

CLAIM_TIERS = (
    "DIAGNOSTIC_GAMMA_MAP",
    "SOURCE_DERIVED_GAMMA_TEMPLATE",
    "INSTRUMENT_CONVOLVED_GAMMA_TEMPLATE",
    "IDENTIFIABLE_GAMMA_TEMPLATE",
    "LIKELIHOOD_EVALUATED_GAMMA_MORPHOLOGY",
    "CROSS_TRACER_VALIDATED_GAMMA_MORPHOLOGY",
    "OPH_GAMMA_MORPHOLOGY_CANDIDATE",
)

CLAIM_REQUIREMENTS = {
    "SOURCE_DERIVED_GAMMA_TEMPLATE": (
        "GAMMA_SOURCE_ARTIFACT_RECEIPT",
        "GAMMA_ROUTE_DECLARATION_RECEIPT",
        "GAMMA_SOURCE_LAW_RECEIPT",
        "GAMMA_NO_DATA_USE_RECEIPT",
        "GAMMA_TEMPLATE_FREEZE_RECEIPT",
    ),
    "INSTRUMENT_CONVOLVED_GAMMA_TEMPLATE": (
        "PHOTON_RESPONSE_KERNEL_RECEIPT",
        "LINE_OF_SIGHT_PROJECTION_RECEIPT",
        "INSTRUMENT_RESPONSE_RECEIPT",
        "SIGNED_TEMPLATE_POSITIVITY_RECEIPT",
    ),
    "IDENTIFIABLE_GAMMA_TEMPLATE": (
        "FOREGROUND_ALTERNATIVE_SET_RECEIPT",
        "GAMMA_IDENTIFIABILITY_RECEIPT",
    ),
    "LIKELIHOOD_EVALUATED_GAMMA_MORPHOLOGY": (
        "FROZEN_GAMMA_LIKELIHOOD_RECEIPT",
        "HELDOUT_GAMMA_VALIDATION_RECEIPT",
    ),
    "CROSS_TRACER_VALIDATED_GAMMA_MORPHOLOGY": (
        "GAMMA_CROSS_TRACER_RECEIPT",
    ),
    "OPH_GAMMA_MORPHOLOGY_CANDIDATE": (
        "OPH_GAMMA_MORPHOLOGY_PREDICTION_RECEIPT",
    ),
}

RECEIPTS = (
    "GAMMA_SOURCE_ARTIFACT_RECEIPT",
    "GAMMA_ROUTE_DECLARATION_RECEIPT",
    "GAMMA_SOURCE_LAW_RECEIPT",
    "GAMMA_STRESS_PARENT_RECEIPT",
    "ANOMALY_SM_CURRENT_NULL_RECEIPT",
    "STRESS_TO_GAMMA_CONTRACTION_RECEIPT",
    "PHOTON_RESPONSE_KERNEL_RECEIPT",
    "BOUNDARY_RECORD_SOURCE_RECEIPT",
    "BOUNDARY_DIPOLE_AXIS_FREEZE_RECEIPT",
    "ASTRO_BRIDGE_FREEZE_RECEIPT",
    "LINE_OF_SIGHT_PROJECTION_RECEIPT",
    "INSTRUMENT_RESPONSE_RECEIPT",
    "SIGNED_TEMPLATE_POSITIVITY_RECEIPT",
    "GAMMA_TEMPLATE_FREEZE_RECEIPT",
    "GAMMA_NO_DATA_USE_RECEIPT",
    "FOREGROUND_ALTERNATIVE_SET_RECEIPT",
    "GAMMA_IDENTIFIABILITY_RECEIPT",
    "FROZEN_GAMMA_LIKELIHOOD_RECEIPT",
    "HELDOUT_GAMMA_VALIDATION_RECEIPT",
    "GAMMA_CROSS_TRACER_RECEIPT",
    "OPH_GAMMA_MORPHOLOGY_PREDICTION_RECEIPT",
)

FORBIDDEN_SOURCE_TOKENS = (
    "gamma_residual_map",
    "gamma_residual_maps",
    "lat_count_residual",
    "lat_count_residuals",
    "likelihood_value",
    "likelihood_values",
    "posterior_summary",
    "foreground_nuisance_fit",
    "foreground_nuisance_fits",
    "template_covariance_from_target_data",
    "human_picked_axis_after_gamma_inspection",
    "gamma_derived_axis",
    "residual_amplitude",
)

FOREGROUND_ALTERNATIVES = (
    "isotropic",
    "galactic_diffuse",
    "point_sources",
    "extended_sources",
    "gas_rings",
    "dust",
    "inverse_compton",
    "bremsstrahlung",
    "fermi_bubbles_or_bubble_like",
    "loop_i_like",
    "stellar_bulge_or_boxy_bulge",
    "nuclear_bulge",
    "disk_msp",
    "annihilation_rho_squared",
    "decay_rho",
    "flattened_halo",
    "contracted_halo",
    "extragalactic_source_populations",
)

REQUIRED_FILES = (
    "manifest.json",
    "gamma_source_artifact.json",
    "source_dag.json",
    "stress_parent_adapter.json",
    "boundary_record_adapter.json",
    "astro_bridge.json",
    "stress_contraction.json",
    "photon_response.json",
    "los_projection.json",
    "instrument_response.json",
    "template_compiler.json",
    "foreground_registry.json",
    "identifiability.json",
    "likelihood.json",
    "cross_tracer.json",
    "nulls.json",
    "receipts.json",
    "claim_ladder.json",
    "claim.md",
)


@dataclass(frozen=True)
class GammaSourceArtifact:
    artifact_id: str
    regulator_id: str
    route: GammaRoute
    quotient_hash: str
    source_law_hash: str
    stress_parent_hash: str | None
    boundary_record_hash: str | None
    astro_bridge_hash: str
    stress_contraction_hash: str | None
    photon_response_hash: str
    los_projection_hash: str
    instrument_response_hash: str
    baseline_registry_hash: str
    alternative_registry_hash: str
    cross_tracer_registry_hash: str
    source_dag_hash: str
    no_data_use_receipt: bool
    freeze_manifest_hash: str
    units_hash: str
    frame_hash: str
    error_ledger_hash: str


@dataclass(frozen=True)
class GammaMorphologyInputs:
    route: GammaRoute = "TRANSPORTED_STRESS"
    claim: str = "DIAGNOSTIC_GAMMA_MAP"
    source: str = "oph_gamma_morphology_scaffold"
    config: Path | None = None
    direct_anomaly_gamma: bool = False
    anomaly_em_current_receipt: bool = False


def signed_template_amplitude_interval(mu0: list[float], tau: list[float]) -> dict[str, Any]:
    if len(mu0) != len(tau):
        raise ValueError("mu0 and tau must have the same length")
    low = float("-inf")
    high = float("inf")
    blockers: list[str] = []
    for index, (base, signed) in enumerate(zip(mu0, tau, strict=True)):
        if not isfinite(base) or not isfinite(signed):
            blockers.append(f"nonfinite_bin_{index}")
            continue
        if base <= 0:
            blockers.append(f"nonpositive_baseline_bin_{index}")
            continue
        if signed > 0:
            low = max(low, -base / signed)
        elif signed < 0:
            high = min(high, base / abs(signed))
    return {
        "amplitude_min": low,
        "amplitude_max": high,
        "nonempty": not blockers and low < high,
        "blockers": blockers,
    }


def gamma_morphology_report(
    inputs: GammaMorphologyInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = inputs if inputs is not None else GammaMorphologyInputs(**kwargs)
    _validate_inputs(config)
    leak_hits = _target_leak_hits(config.config)
    direct_blocker = config.direct_anomaly_gamma and not config.anomaly_em_current_receipt
    receipts = _default_receipts(config, leak_hits=leak_hits, direct_blocker=direct_blocker)
    strongest, first_blocked, missing = strongest_allowed_claim(receipts)
    blockers = []
    if leak_hits:
        blockers.append("gamma_source_dag_reads_target_data")
    if direct_blocker:
        blockers.append("direct_anomaly_gamma_without_em_current_theorem")
    if strongest != config.claim and config.claim != "DIAGNOSTIC_GAMMA_MAP":
        blockers.append("requested_claim_exceeds_receipts")
    if strongest == "DIAGNOSTIC_GAMMA_MAP":
        blockers.append(first_blocked or "source_artifact_not_promoted")

    return {
        "mode": "oph_gamma_morphology_v1",
        "milestone": "GAMMA_MORPHOLOGY_AUDIT",
        "source": config.source,
        "route": config.route,
        "claim": config.claim,
        "strongest_allowed_claim": strongest,
        "first_blocked_gate": first_blocked,
        "missing_for_next_claim": missing,
        "promotion_allowed": strongest == "OPH_GAMMA_MORPHOLOGY_CANDIDATE",
        "source_open": strongest == "DIAGNOSTIC_GAMMA_MAP",
        "readiness_gates": receipts,
        "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        "target_leak_hits": leak_hits,
        "foreground_alternatives": list(FOREGROUND_ALTERNATIVES),
        "claim_ladder": list(CLAIM_TIERS),
        "required_files": list(REQUIRED_FILES),
        "claim_boundary": (
            "OPH gamma-ray signatures are source-derived morphology claims, not "
            "excess-power claims. The simulator may compile frozen templates and "
            "run count-space likelihood, foreground, null, held-out, and cross-tracer "
            "tests, but it must not discover the OPH template from gamma residuals."
        ),
        "blockers": blockers,
    }


def write_gamma_morphology_bundle(
    out_dir: Path,
    inputs: GammaMorphologyInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = gamma_morphology_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = _bundle_payloads(report)
    file_hashes: dict[str, str] = {}
    for rel_path, payload in payloads.items():
        path = out / rel_path
        _write_payload(path, payload)
        file_hashes[rel_path] = _sha256_bytes(path.read_bytes())
    manifest = {
        "artifact": "gamma_morphology_manifest",
        "generated_utc": _now_utc(),
        "milestone": report["milestone"],
        "route": report["route"],
        "claim": report["claim"],
        "strongest_allowed_claim": report["strongest_allowed_claim"],
        "first_blocked_gate": report["first_blocked_gate"],
        "promotion_allowed": report["promotion_allowed"],
        "required_files": list(REQUIRED_FILES),
        "missing_files": [rel for rel in REQUIRED_FILES if rel != "manifest.json" and not (out / rel).is_file()],
        "file_hashes": file_hashes,
    }
    _write_payload(out / "manifest.json", manifest)
    manifest["file_hashes"]["manifest.json"] = _sha256_bytes((out / "manifest.json").read_bytes())
    _write_payload(out / "manifest.json", manifest)
    full_report = {**report, "manifest": manifest}
    _write_payload(out / "gamma_morphology_report.json", full_report)
    (out / "gamma_morphology_report.md").write_text(_markdown_report(full_report), encoding="utf-8")
    return full_report


def strongest_allowed_claim(receipts: dict[str, bool]) -> tuple[str, str | None, list[str]]:
    strongest = "DIAGNOSTIC_GAMMA_MAP"
    for tier in CLAIM_TIERS[1:]:
        requirements: list[str] = []
        for claim in CLAIM_TIERS[1 : CLAIM_TIERS.index(tier) + 1]:
            requirements.extend(CLAIM_REQUIREMENTS[claim])
        missing = [name for name in requirements if not receipts.get(name, False)]
        if missing:
            return strongest, missing[0], missing
        strongest = tier
    return strongest, None, []


def _default_receipts(
    config: GammaMorphologyInputs,
    *,
    leak_hits: list[str],
    direct_blocker: bool,
) -> dict[str, bool]:
    receipts = {name: False for name in RECEIPTS}
    receipts["GAMMA_ROUTE_DECLARATION_RECEIPT"] = config.route in {"TRANSPORTED_STRESS", "BOUNDARY_RECORD_DIPOLE", "BOTH"}
    receipts["ANOMALY_SM_CURRENT_NULL_RECEIPT"] = not direct_blocker
    receipts["GAMMA_NO_DATA_USE_RECEIPT"] = not leak_hits
    receipts["FOREGROUND_ALTERNATIVE_SET_RECEIPT"] = True
    return receipts


def _bundle_payloads(report: dict[str, Any]) -> dict[str, str | dict[str, Any]]:
    gates = report["readiness_gates"]
    base = {
        "claim": report["claim"],
        "strongest_allowed_claim": report["strongest_allowed_claim"],
        "route": report["route"],
        "promotion_allowed": report["promotion_allowed"],
    }
    return {
        "gamma_source_artifact.json": {
            **base,
            "artifact": "gamma_source_artifact",
            "required_object": "mathfrak_G_gamma_r",
            "status": "SOURCE_ARTIFACT_REQUIRED",
            "readiness_gates": {
                "GAMMA_SOURCE_ARTIFACT_RECEIPT": gates["GAMMA_SOURCE_ARTIFACT_RECEIPT"],
                "GAMMA_SOURCE_LAW_RECEIPT": gates["GAMMA_SOURCE_LAW_RECEIPT"],
                "GAMMA_NO_DATA_USE_RECEIPT": gates["GAMMA_NO_DATA_USE_RECEIPT"],
            },
        },
        "source_dag.json": {
            **base,
            "artifact": "gamma_source_dag",
            "forbidden_source_tokens": report["forbidden_source_tokens"],
            "target_leak_hits": report["target_leak_hits"],
            "status": "PASS_EMPTY_COMPARISON_DAG" if not report["target_leak_hits"] else "FAIL_FORBIDDEN_SOURCE_INPUT",
        },
        "stress_parent_adapter.json": {
            **base,
            "artifact": "stress_parent_adapter",
            "required_receipts": [
                "FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT",
                "PACKET_MASS_SHELL_RECEIPT",
                "TOTAL_STRESS_CLOSURE_RECEIPT",
                "SM_CURRENT_NULL_RECEIPT",
                "RETARDED_RESPONSE_RECEIPT",
                "RESPONSE_STABILITY_RECEIPT",
                "REFINEMENT_CONVERGENCE_RECEIPT",
                "CDM_LIMIT_RECOVERY_RECEIPT",
            ],
            "scalar_only_status": "SCALAR_ONLY_GAMMA_DIAGNOSTIC",
        },
        "boundary_record_adapter.json": {
            **base,
            "artifact": "boundary_record_adapter",
            "required": ["boundary_record_map", "dipole_projector", "axis_hash", "energy_response"],
            "status": "BOUNDARY_RECORD_SOURCE_REQUIRED",
        },
        "astro_bridge.json": {
            **base,
            "artifact": "astro_bridge",
            "ordinary_channels": ["pi0", "inverse_compton", "bremsstrahlung", "baryonic_potential"],
            "status": "ASTRO_BRIDGE_FREEZE_REQUIRED",
        },
        "stress_contraction.json": {
            **base,
            "artifact": "stress_contraction",
            "default_contraction": "sigma_T = T_A^{mu nu} u_mu u_nu",
            "forbidden_shortcut": "sigma_T = abs(rho_A) without parent positivity theorem",
            "status": "STRESS_PARENT_REQUIRED",
        },
        "photon_response.json": {
            **base,
            "artifact": "photon_response",
            "direct_anomaly_gamma_default": 0.0,
            "nonzero_direct_requires": "ANOMALY_EM_CURRENT_RECEIPT",
            "status": "PHOTON_RESPONSE_KERNEL_REQUIRED",
        },
        "los_projection.json": {
            **base,
            "artifact": "line_of_sight_projection",
            "normalization": "fisher_unit_norm_predeclared",
            "status": "LINE_OF_SIGHT_PROJECTION_REQUIRED",
        },
        "instrument_response.json": {
            **base,
            "artifact": "instrument_response",
            "operator": "intensity_to_binned_counts",
            "required": ["exposure", "psf", "energy_dispersion", "mask", "event_class", "energy_bins"],
            "status": "INSTRUMENT_RESPONSE_REQUIRED",
        },
        "template_compiler.json": {
            **base,
            "artifact": "template_compiler",
            "outputs": ["tau_raw_hash", "tau_sky_hash", "tau_convolved_hash"],
            "status": "TEMPLATE_FREEZE_REQUIRED",
        },
        "foreground_registry.json": {
            **base,
            "artifact": "foreground_registry",
            "alternatives": report["foreground_alternatives"],
            "readiness_gates": {"FOREGROUND_ALTERNATIVE_SET_RECEIPT": gates["FOREGROUND_ALTERNATIVE_SET_RECEIPT"]},
        },
        "identifiability.json": {
            **base,
            "artifact": "identifiability",
            "metric": "eta_id = ||(I-P_B) tau_OPH||_W / ||tau_OPH||_W",
            "status": "IDENTIFIABILITY_TEST_REQUIRED",
        },
        "likelihood.json": {
            **base,
            "artifact": "frozen_gamma_likelihood",
            "likelihood": "binned_poisson_counts",
            "required_hashes": [
                "data_hash",
                "catalog_hash",
                "diffuse_model_hash",
                "mask_hash",
                "exposure_hash",
                "psf_hash",
                "energy_dispersion_hash",
                "event_class_hash",
                "energy_bins_hash",
                "nuisance_prior_hash",
                "likelihood_code_hash",
            ],
        },
        "cross_tracer.json": {
            **base,
            "artifact": "cross_tracer",
            "transported_stress_tracers": ["lensing", "baryonic_potential", "gas", "dust", "source_catalogs", "large_scale_structure"],
            "boundary_dipole_tests": ["energy_axis_stability", "time_axis_stability", "mask_axis_stability", "event_class_axis_stability"],
        },
        "nulls.json": {
            **base,
            "artifact": "gamma_nulls",
            "required_nulls": ["random_rotations", "axis_shuffles", "exposure", "ecliptic", "solar", "lunar", "earth_limb"],
        },
        "receipts.json": {**base, "artifact": "gamma_receipts", "readiness_gates": gates},
        "claim_ladder.json": {
            **base,
            "artifact": "gamma_claim_ladder",
            "claim_ladder": list(CLAIM_TIERS),
            "first_blocked_gate": report["first_blocked_gate"],
            "missing_for_next_claim": report["missing_for_next_claim"],
        },
        "claim.md": report["strongest_allowed_claim"] + "\n",
    }


def _validate_inputs(config: GammaMorphologyInputs) -> None:
    if config.route not in {"TRANSPORTED_STRESS", "BOUNDARY_RECORD_DIPOLE", "BOTH"}:
        raise ValueError(f"unknown gamma route: {config.route}")
    if config.claim not in CLAIM_TIERS:
        raise ValueError(f"unknown gamma claim tier: {config.claim}")


def _target_leak_hits(config: Path | None) -> list[str]:
    if config is None or not config.is_file():
        return []
    text = config.read_text(encoding="utf-8")
    haystack = text.lower()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        haystack = json.dumps(parsed, sort_keys=True).lower()
    return sorted(token for token in FORBIDDEN_SOURCE_TOKENS if token in haystack)


def _write_payload(path: Path, payload: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OPH Gamma Morphology",
            "",
            str(report["claim_boundary"]),
            "",
            f"- route: `{report['route']}`",
            f"- strongest allowed claim: `{report['strongest_allowed_claim']}`",
            f"- first blocked gate: `{report['first_blocked_gate']}`",
            f"- promotion allowed: `{str(bool(report['promotion_allowed'])).lower()}`",
            "",
            "## Blockers",
            "",
            *[f"- `{item}`" for item in report["blockers"]],
            "",
        ]
    )


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
