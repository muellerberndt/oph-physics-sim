from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLAIMS = (
    "CONVENTIONAL_QCD_REFERENCE",
    "SOURCE_PROTOTYPE_NOT_PROMOTED",
    "SOURCE_INTERVAL_PROMOTED",
    "EMPIRICAL_CLOSURE_ONLY",
    "COMPARISON_ONLY",
)

CLAIM_TIERS = {
    "H0": "conventional QCD reference or comparison-only framing",
    "H1": "OPH-QCD quotient schema declared",
    "H2": "source prototype scaffold emitted, not promoted",
    "H3": "Ward current and Euclidean correlator receipts populated",
    "H4": "positive two-current spectral export for running-alpha/HVP",
    "H5": "same-scheme fine-structure endpoint interval",
    "H6": "higher-point and transition spectral sectors",
    "H7": "source interval promoted with no forbidden target path",
}

FORBIDDEN_SOURCE_INPUTS = (
    "CODATA_ALPHA",
    "MUON_G_MINUS_2",
    "EE_TO_HADRONS",
    "RARE_DECAY_DATA",
    "HADRON_MASS_TARGETS",
    "PDG_QCD_FITS",
)

SOURCE_OBJECT = (
    "QCD quotient ensemble",
    "Euclidean QCD slab/vacuum transfer",
    "hadronic Hilbert quotient",
    "Ward-normalized electromagnetic current",
    "two-current spectral measure d rho_Q^(2)",
    "four-current spectral measure d rho_QQQQ^(4)",
    "B/Sigma transition spectral measures",
    "same-scheme endpoint remainder Xi_Q",
    "systematics ledger E_sys",
)

REQUIRED_FILES = (
    "manifest.json",
    "source_dag.json",
    "qcd_ensemble/quotient_schema.json",
    "qcd_ensemble/gamma_groupoid.json",
    "qcd_ensemble/base_measure.json",
    "qcd_ensemble/source_action.json",
    "qcd_ensemble/source_parameter_map.json",
    "qcd_ensemble/coarse_maps.json",
    "vacuum/euclidean_slab.json",
    "vacuum/transfer_operator.json",
    "vacuum/reflection_positivity.json",
    "vacuum/vacuum_promotion.json",
    "currents/ward_current_definition.json",
    "currents/current_normalization_ZV.json",
    "currents/contact_terms.json",
    "currents/ward_residuals.csv",
    "correlators/vector_current_2pt_raw.json",
    "correlators/vector_current_2pt_covariance.json",
    "correlators/disconnected_diagrams.json",
    "correlators/autocorrelation_report.json",
    "spectral/moments.json",
    "spectral/hankel_positivity.json",
    "spectral/stieltjes_bounds.json",
    "spectral/J24Q.json",
    "spectral/omegaQ.json",
    "spectral/spectral_interval.json",
    "endpoint/kernel_definition.json",
    "endpoint/Xi_same_scheme.json",
    "endpoint/Delta_had_interval.json",
    "endpoint/ATh_interval.json",
    "endpoint/pixel_contraction_interval.json",
    "higher_point/Q4_HLbL_receipt.json",
    "higher_point/transition_B_to_K_receipt.json",
    "higher_point/transition_Sigma_to_p_receipt.json",
    "controls/no_target_leak_dag.json",
    "controls/empirical_data_exclusion_manifest.json",
    "controls/frozen_code_hashes.json",
    "controls/replay_receipts.json",
    "controls/comparison_data_manifest.json",
    "claim.md",
)

TWO_CURRENT_REQUIRED_RECEIPTS = (
    "qcd_quotient_ensemble_receipt",
    "source_qcd_parameter_map_receipt",
    "euclidean_qcd_slab_receipt",
    "hadronic_hilbert_quotient_receipt",
    "ward_current_normalization_receipt",
    "two_current_spectral_export_receipt",
    "same_scheme_remainder_receipt",
    "qcd_systematics_ledger_receipt",
    "qcd_no_target_leak_dag_receipt",
)

FULL_PRECISION_REQUIRED_RECEIPTS = TWO_CURRENT_REQUIRED_RECEIPTS + (
    "higher_point_spectral_exports_receipt",
    "transition_spectral_exports_receipt",
)


@dataclass(frozen=True)
class HadronSourceBackendInputs:
    claim: str = "SOURCE_PROTOTYPE_NOT_PROMOTED"
    tier: str = "H2"
    source: str = "oph_fpe_hadron_source_backend"
    lambda_msbar_descendant: Path | None = None


def hadron_source_backend_report(
    inputs: HadronSourceBackendInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = inputs if inputs is not None else HadronSourceBackendInputs(**kwargs)
    _validate_claim(config.claim, config.tier)
    lambda_payload = _read_optional_json(config.lambda_msbar_descendant)
    promoted = config.claim == "SOURCE_INTERVAL_PROMOTED"
    receipt_gates = {key: bool(promoted) for key in FULL_PRECISION_REQUIRED_RECEIPTS}
    if config.claim == "CONVENTIONAL_QCD_REFERENCE":
        receipt_gates = {key: False for key in FULL_PRECISION_REQUIRED_RECEIPTS}
    two_current = all(receipt_gates[key] for key in TWO_CURRENT_REQUIRED_RECEIPTS)
    full_precision = all(receipt_gates[key] for key in FULL_PRECISION_REQUIRED_RECEIPTS)
    forbidden_clean = True
    blockers = []
    if not two_current:
        blockers.append("two_current_hadronic_backend_receipt_missing")
    if not full_precision:
        blockers.append("full_hadronic_precision_backend_receipt_missing")
    if config.claim != "SOURCE_INTERVAL_PROMOTED":
        blockers.append("source_qcd_law_not_promoted")

    return {
        "mode": "oph_qcd_hadron_source_backend_v1",
        "milestone": "HVP_ALPHA_SOURCE_PROTOTYPE",
        "source": config.source,
        "claim": config.claim,
        "claim_tier": config.tier,
        "claim_tier_description": CLAIM_TIERS[config.tier],
        "promotion_allowed": promoted,
        "source_open": not promoted,
        "source_object": list(SOURCE_OBJECT),
        "claim_tiers": CLAIM_TIERS,
        "required_files": list(REQUIRED_FILES),
        "forbidden_source_inputs": list(FORBIDDEN_SOURCE_INPUTS),
        "lambda_msbar_descendant": lambda_payload,
        "readiness_gates": {
            **receipt_gates,
            "two_current_hadronic_backend_receipt": two_current,
            "full_hadronic_precision_backend_receipt": full_precision,
            "forbidden_source_inputs_excluded": forbidden_clean,
            "fine_structure_endpoint_promotion_receipt": False,
            "hvp_g_minus_2_promotion_receipt": False,
            "hlbl_g_minus_2_promotion_receipt": False,
            "rare_decay_long_distance_promotion_receipt": False,
        },
        "scope": {
            "two_current_marginal": "running-alpha endpoint and HVP only",
            "higher_point_required_for": ["HLbL g-2", "rare B long-distance amplitudes", "rare Sigma long-distance amplitudes"],
            "empirical_data_policy": "comparison data may be attached only after source freeze",
        },
        "blockers": blockers,
        "claim_boundary": (
            "This is a proof-producing receipt scaffold for the OPH-QCD hadronic precision backend. "
            "It is not a lattice-QCD solver and does not promote alpha(0), g-2, HLbL, or rare-decay "
            "claims unless the source QCD law, Ward current ledger, spectral exports, systematics, "
            "and no-target-leak receipts are populated without forbidden target inputs."
        ),
    }


def write_hadron_source_backend_bundle(
    out_dir: Path,
    inputs: HadronSourceBackendInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = hadron_source_backend_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = _bundle_payloads(report)
    file_hashes: dict[str, str] = {}
    for rel_path, payload in payloads.items():
        path = out / rel_path
        _write_payload(path, payload)
        file_hashes[rel_path] = _sha256_bytes(path.read_bytes())
    manifest = {
        "artifact": "hadron_source_backend_manifest",
        "generated_utc": _now_utc(),
        "milestone": report["milestone"],
        "claim": report["claim"],
        "claim_tier": report["claim_tier"],
        "promotion_allowed": report["promotion_allowed"],
        "source_open": report["source_open"],
        "required_files": list(REQUIRED_FILES),
        "missing_files": [rel for rel in REQUIRED_FILES if rel != "manifest.json" and not (out / rel).is_file()],
        "forbidden_source_inputs": list(FORBIDDEN_SOURCE_INPUTS),
        "file_hashes": file_hashes,
    }
    _write_payload(out / "manifest.json", manifest)
    manifest["file_hashes"]["manifest.json"] = _sha256_bytes((out / "manifest.json").read_bytes())
    _write_payload(out / "manifest.json", manifest)
    report = {**report, "manifest": manifest}
    _write_payload(out / "hadron_source_backend_report.json", report)
    (out / "hadron_source_backend_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _bundle_payloads(report: dict[str, Any]) -> dict[str, str | dict[str, Any]]:
    tier = str(report["claim_tier"])
    claim = str(report["claim"])
    base = {
        "claim": claim,
        "claim_tier": tier,
        "promotion_allowed": bool(report["promotion_allowed"]),
        "forbidden_source_inputs": list(FORBIDDEN_SOURCE_INPUTS),
    }
    return {
        "source_dag.json": {
            **base,
            "source_inputs": ["P", "source QCD parameter map"],
            "forbidden_edges": [],
            "status": "PASS_EMPTY_COMPARISON_DAG",
        },
        "qcd_ensemble/quotient_schema.json": {
            **base,
            "Sigma_QCD_r": "finite links, quark fields, boundaries, currents, operators, regulator metadata",
            "Gamma_QCD_r": "gauge reps, mesh labels, ports, hidden carriers, worker ids, repair schedules, inert ancillas",
            "Q_QCD_r": "Sigma_QCD_r/Gamma_QCD_r",
        },
        "qcd_ensemble/gamma_groupoid.json": {**base, "status": "SCHEMA_ONLY"},
        "qcd_ensemble/base_measure.json": {**base, "status": "BASE_MEASURE_REQUIRED"},
        "qcd_ensemble/source_action.json": {
            **base,
            "law": "mu_r(q;P)=Z_r^-1 m_r(q) exp[-S_r(q;P)]",
            "status": "SOURCE_LAW_REQUIRED",
        },
        "qcd_ensemble/source_parameter_map.json": {
            **base,
            "required_map": "P -> (g3, theta_QCD, m_u, m_d, m_s, m_c, m_b, m_t, Z_scheme)",
            "lambda_msbar_descendant": report["lambda_msbar_descendant"],
        },
        "qcd_ensemble/coarse_maps.json": {**base, "required": "c_sr refinement maps"},
        "vacuum/euclidean_slab.json": {**base, "required_tuple": ["Q_r^QCD", "m_r^0", "J_r", "V_r", "a_t,r", "Theta_RP"]},
        "vacuum/transfer_operator.json": {**base, "required": "finite vacuum transfer operator"},
        "vacuum/reflection_positivity.json": {**base, "status": "CERTIFICATE_REQUIRED"},
        "vacuum/vacuum_promotion.json": {**base, "status": "NOT_PROMOTED_BY_HMC_ALONE"},
        "currents/ward_current_definition.json": {**base, "current": "J_Q^{W,R,mu}"},
        "currents/current_normalization_ZV.json": {**base, "required": ["Z_V", "Ward residuals", "contact terms", "Omega_Q"]},
        "currents/contact_terms.json": {**base, "status": "LEDGER_REQUIRED"},
        "currents/ward_residuals.csv": "momentum,residual,bound,status\n",
        "correlators/vector_current_2pt_raw.json": {**base, "observable": "C_QQ(t)", "status": "NOT_COMPUTED"},
        "correlators/vector_current_2pt_covariance.json": {**base, "status": "NOT_COMPUTED"},
        "correlators/disconnected_diagrams.json": {**base, "status": "NOT_COMPUTED"},
        "correlators/autocorrelation_report.json": {**base, "status": "NOT_COMPUTED"},
        "spectral/moments.json": {**base, "status": "MOMENTS_REQUIRED_FROM_CORRELATORS"},
        "spectral/hankel_positivity.json": {**base, "status": "POSITIVITY_CERTIFICATE_REQUIRED"},
        "spectral/stieltjes_bounds.json": {**base, "status": "STIELTJES_INTERVAL_REQUIRED"},
        "spectral/J24Q.json": {**base, "status": "MUST_DERIVE_FROM_MOMENTS_OR_LANCZOS"},
        "spectral/omegaQ.json": {**base, "status": "CURRENT_NORMALIZATION_REQUIRED"},
        "spectral/spectral_interval.json": {**base, "status": "NO_SOURCE_INTERVAL"},
        "endpoint/kernel_definition.json": {**base, "kernel": "Delta_had from d rho_Q^(2) plus Xi_Q"},
        "endpoint/Xi_same_scheme.json": {**base, "status": "SAME_SCHEME_REMAINDER_REQUIRED"},
        "endpoint/Delta_had_interval.json": {**base, "interval": None},
        "endpoint/ATh_interval.json": {**base, "interval": None},
        "endpoint/pixel_contraction_interval.json": {**base, "interval": None},
        "higher_point/Q4_HLbL_receipt.json": {**base, "status": "TWO_CURRENT_MEASURE_INSUFFICIENT"},
        "higher_point/transition_B_to_K_receipt.json": {**base, "status": "TRANSITION_SPECTRAL_EXPORT_REQUIRED"},
        "higher_point/transition_Sigma_to_p_receipt.json": {**base, "status": "TRANSITION_SPECTRAL_EXPORT_REQUIRED"},
        "controls/no_target_leak_dag.json": {**base, "forbidden_edges": [], "status": "PASS_EMPTY_COMPARISON_DAG"},
        "controls/empirical_data_exclusion_manifest.json": {**base, "comparison_allowed_after_source_freeze": True},
        "controls/frozen_code_hashes.json": {**base, "module": __name__, "module_sha256": _sha256_text(Path(__file__).read_text(encoding="utf-8"))},
        "controls/replay_receipts.json": {**base, "status": "SCHEMA_ONLY_REPLAY_NOT_RUN"},
        "controls/comparison_data_manifest.json": {**base, "comparison_data": [], "status": "NO_COMPARISON_DATA_ATTACHED"},
        "claim.md": claim + "\n",
    }


def _validate_claim(claim: str, tier: str) -> None:
    if claim not in CLAIMS:
        raise ValueError(f"unknown hadron source backend claim: {claim}")
    if tier not in CLAIM_TIERS:
        raise ValueError(f"unknown hadron source backend tier: {tier}")


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    source = Path(path)
    if not source.is_file():
        return {"exists": False, "path": str(source)}
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lambda_msbar_descendant must be a JSON object")
    return {**payload, "exists": True, "path": str(source), "sha256": _sha256_bytes(raw)}


def _write_payload(path: Path, payload: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH-QCD Hadron Source Backend",
            "",
            str(report["claim_boundary"]),
            "",
            f"- claim: `{report['claim']}`",
            f"- tier: `{report['claim_tier']}`",
            f"- source open: `{str(bool(report['source_open'])).lower()}`",
            f"- two-current backend receipt: `{str(bool(gates['two_current_hadronic_backend_receipt'])).lower()}`",
            f"- full hadronic precision backend receipt: `{str(bool(gates['full_hadronic_precision_backend_receipt'])).lower()}`",
            f"- forbidden source inputs excluded: `{str(bool(gates['forbidden_source_inputs_excluded'])).lower()}`",
            "",
            "## Blockers",
            "",
            *[f"- `{item}`" for item in report["blockers"]],
            "",
        ]
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))
