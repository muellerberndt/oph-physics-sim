from __future__ import annotations

import csv
import hashlib
import json
import math
import re
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
    "controls/systematics_ledger.json",
    "claim.md",
)

EVIDENCE_SCHEMA = "oph_qcd_source_promotion_evidence_v1"
EVIDENCE_ARTIFACT = "oph_qcd_source_promotion_evidence"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")

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
    evidence_dir: Path | None = None


def hadron_source_backend_report(
    inputs: HadronSourceBackendInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = inputs if inputs is not None else HadronSourceBackendInputs(**kwargs)
    _validate_claim(config.claim, config.tier)
    lambda_payload = _read_optional_json(config.lambda_msbar_descendant)
    evidence_validation = _validate_promotion_bundle(config.evidence_dir)
    promotion_requested = config.claim == "SOURCE_INTERVAL_PROMOTED"
    receipt_gates = dict(evidence_validation["receipt_gates"])
    if config.claim == "CONVENTIONAL_QCD_REFERENCE":
        receipt_gates = {key: False for key in FULL_PRECISION_REQUIRED_RECEIPTS}
    two_current = all(receipt_gates[key] for key in TWO_CURRENT_REQUIRED_RECEIPTS)
    full_precision = all(receipt_gates[key] for key in FULL_PRECISION_REQUIRED_RECEIPTS)
    forbidden_clean = bool(evidence_validation["no_target_leak_passed"])
    promoted = bool(
        promotion_requested
        and two_current
        and full_precision
        and forbidden_clean
        and evidence_validation["artifact_hashes_passed"]
        and evidence_validation["provenance_passed"]
    )
    blockers = list(evidence_validation["blockers"] if promotion_requested else [])
    if not two_current:
        blockers.append("two_current_hadronic_backend_receipt_missing")
    if not full_precision:
        blockers.append("full_hadronic_precision_backend_receipt_missing")
    if not forbidden_clean:
        blockers.append("qcd_no_target_leak_audit_missing")
    if not promotion_requested:
        blockers.append("source_qcd_law_not_promoted")
    elif not promoted:
        blockers.append("source_interval_promotion_evidence_invalid")

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
        "promotion_evidence": evidence_validation,
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
        "blockers": _unique(blockers),
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
    config = inputs if inputs is not None else HadronSourceBackendInputs(**kwargs)
    out = Path(out_dir)
    if config.evidence_dir is not None and Path(config.evidence_dir).resolve() == out.resolve():
        raise ValueError("evidence_dir must be separate from the generated report/scaffold directory")
    report = hadron_source_backend_report(config)
    out.mkdir(parents=True, exist_ok=True)
    payloads = _bundle_payloads(report)
    file_hashes: dict[str, str] = {}
    for rel_path, payload in payloads.items():
        path = out / rel_path
        _write_payload(path, payload)
        file_hashes[rel_path] = _sha256_bytes(path.read_bytes())
    manifest = {
        "schema": "oph_qcd_source_backend_scaffold_v1",
        "artifact": "hadron_source_backend_manifest",
        "generator": __name__,
        "source_commit": None,
        "source_dirty": True,
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
    # A manifest cannot contain its own final byte hash without a detached
    # signature. Keep the compatibility key explicit, but never use it as a
    # promotion input; all non-manifest evidence files are independently hashed.
    manifest["file_hashes"]["manifest.json"] = None
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
        "controls/systematics_ledger.json": {
            **base,
            "components": {},
            "status": "SYSTEMATICS_LEDGER_REQUIRED",
        },
        "claim.md": claim + "\n",
    }


def _validate_promotion_bundle(evidence_dir: Path | None) -> dict[str, Any]:
    closed_gates = {key: False for key in FULL_PRECISION_REQUIRED_RECEIPTS}
    if evidence_dir is None:
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_dir": None,
            "written": False,
            "provenance_passed": False,
            "artifact_hashes_passed": False,
            "no_target_leak_passed": False,
            "receipt_gates": closed_gates,
            "provenance_checks": {},
            "artifact_hashes": [],
            "semantic_checks": {},
            "blockers": ["promotion_evidence_dir_missing"],
        }

    root = Path(evidence_dir)
    if not root.is_dir():
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_dir": str(root),
            "written": False,
            "provenance_passed": False,
            "artifact_hashes_passed": False,
            "no_target_leak_passed": False,
            "receipt_gates": closed_gates,
            "provenance_checks": {},
            "artifact_hashes": [],
            "semantic_checks": {},
            "blockers": ["promotion_evidence_dir_not_found"],
        }

    manifest = _read_json_file(root / "manifest.json")
    file_hashes = manifest.get("file_hashes") if isinstance(manifest.get("file_hashes"), dict) else {}
    provenance_checks = {
        "schema_exact": manifest.get("schema") == EVIDENCE_SCHEMA,
        "artifact_exact": manifest.get("artifact") == EVIDENCE_ARTIFACT,
        "generator_named": _nonempty_string(manifest.get("generator")),
        "source_commit_typed": bool(
            isinstance(manifest.get("source_commit"), str)
            and _COMMIT_RE.fullmatch(str(manifest.get("source_commit")))
        ),
        "source_tree_clean": manifest.get("source_dirty") is False,
        "generated_utc_named": _nonempty_string(manifest.get("generated_utc")),
    }

    hash_rows = []
    for rel_path in REQUIRED_FILES:
        if rel_path == "manifest.json":
            continue
        path = root / rel_path
        expected = file_hashes.get(rel_path)
        scope_ok = False
        try:
            path.resolve().relative_to(root.resolve())
            scope_ok = not path.is_symlink()
        except (OSError, ValueError):
            scope_ok = False
        actual = _sha256_bytes(path.read_bytes()) if scope_ok and path.is_file() else None
        hash_rows.append(
            {
                "path": rel_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": path.is_file(),
                "scope_ok": scope_ok,
                "passed": bool(
                    scope_ok
                    and path.is_file()
                    and isinstance(expected, str)
                    and _SHA256_RE.fullmatch(expected)
                    and actual == expected
                ),
            }
        )
    artifact_hashes_passed = bool(hash_rows and all(row["passed"] for row in hash_rows))

    quotient = _read_json_file(root / "qcd_ensemble/quotient_schema.json")
    groupoid = _read_json_file(root / "qcd_ensemble/gamma_groupoid.json")
    base_measure = _read_json_file(root / "qcd_ensemble/base_measure.json")
    source_action = _read_json_file(root / "qcd_ensemble/source_action.json")
    parameter_map = _read_json_file(root / "qcd_ensemble/source_parameter_map.json")
    slab = _read_json_file(root / "vacuum/euclidean_slab.json")
    transfer = _read_json_file(root / "vacuum/transfer_operator.json")
    reflection = _read_json_file(root / "vacuum/reflection_positivity.json")
    vacuum = _read_json_file(root / "vacuum/vacuum_promotion.json")
    current = _read_json_file(root / "currents/ward_current_definition.json")
    normalization = _read_json_file(root / "currents/current_normalization_ZV.json")
    contact = _read_json_file(root / "currents/contact_terms.json")
    correlator = _read_json_file(root / "correlators/vector_current_2pt_raw.json")
    covariance = _read_json_file(root / "correlators/vector_current_2pt_covariance.json")
    disconnected = _read_json_file(root / "correlators/disconnected_diagrams.json")
    autocorrelation = _read_json_file(root / "correlators/autocorrelation_report.json")
    moments = _read_json_file(root / "spectral/moments.json")
    hankel = _read_json_file(root / "spectral/hankel_positivity.json")
    stieltjes = _read_json_file(root / "spectral/stieltjes_bounds.json")
    j24 = _read_json_file(root / "spectral/J24Q.json")
    omega = _read_json_file(root / "spectral/omegaQ.json")
    spectral_interval = _read_json_file(root / "spectral/spectral_interval.json")
    xi = _read_json_file(root / "endpoint/Xi_same_scheme.json")
    delta = _read_json_file(root / "endpoint/Delta_had_interval.json")
    ath = _read_json_file(root / "endpoint/ATh_interval.json")
    pixel = _read_json_file(root / "endpoint/pixel_contraction_interval.json")
    q4 = _read_json_file(root / "higher_point/Q4_HLbL_receipt.json")
    b_transition = _read_json_file(root / "higher_point/transition_B_to_K_receipt.json")
    sigma_transition = _read_json_file(root / "higher_point/transition_Sigma_to_p_receipt.json")
    source_dag = _read_json_file(root / "source_dag.json")
    no_target = _read_json_file(root / "controls/no_target_leak_dag.json")
    exclusions = _read_json_file(root / "controls/empirical_data_exclusion_manifest.json")
    frozen_hashes = _read_json_file(root / "controls/frozen_code_hashes.json")
    replay = _read_json_file(root / "controls/replay_receipts.json")
    comparison = _read_json_file(root / "controls/comparison_data_manifest.json")
    systematics = _read_json_file(root / "controls/systematics_ledger.json")

    qcd_quotient = bool(
        _all_literal_true(quotient, "quotient_defined", "finite_regulator")
        and _all_literal_true(groupoid, "groupoid_action_verified", "inert_labels_quotiented")
        and _all_literal_true(base_measure, "base_measure_normalized", "positive_measure")
        and _positive_number(base_measure.get("total_weight"))
        and _all_literal_true(source_action, "source_action_explicit", "action_bounded_below")
        and _finite_number(source_action.get("action_lower_bound"))
    )
    source_parameter_map = bool(
        _literal_true(parameter_map.get("source_parameter_map_complete"))
        and isinstance(parameter_map.get("parameters"), dict)
        and {"g3", "theta_qcd", "quark_masses", "renormalization_scheme"}
        <= set(parameter_map.get("parameters", {}))
        and parameter_map.get("forbidden_target_inputs") == []
    )
    euclidean_slab = bool(
        _literal_true(slab.get("euclidean_slab_constructed"))
        and _int_at_least(slab.get("time_slices"), 2)
        and _literal_true(transfer.get("transfer_operator_constructed"))
        and _nonnegative_number(transfer.get("minimum_eigenvalue"))
    )
    reflection_positive = _positivity_certificate(reflection)
    hadronic_hilbert = bool(
        reflection_positive
        and _all_literal_true(
            vacuum,
            "hadronic_hilbert_quotient_constructed",
            "null_space_quotiented",
            "positive_transfer_operator",
        )
    )
    ward_rows_passed, ward_row_count = _ward_csv_passes(root / "currents/ward_residuals.csv")
    ward_current = bool(
        _literal_true(current.get("ward_current_defined"))
        and _positive_number(normalization.get("z_v"))
        and _literal_true(normalization.get("ward_identity_verified"))
        and _literal_true(contact.get("contact_terms_accounted"))
        and ward_rows_passed
    )
    two_current = bool(
        _literal_true(correlator.get("two_point_correlator_computed"))
        and _positive_int(correlator.get("sample_count"))
        and _literal_true(covariance.get("covariance_psd"))
        and _literal_true(disconnected.get("disconnected_diagrams_included"))
        and _positive_number(autocorrelation.get("effective_sample_size"))
        and _finite_number_list(moments.get("moments"))
        and _positivity_certificate(hankel)
        and _literal_true(stieltjes.get("stieltjes_bounds_verified"))
        and _ordered_interval(stieltjes)
        and _finite_number(j24.get("j24q"))
        and _positive_number(omega.get("omega_q"))
        and _ordered_interval(spectral_interval)
        and _nonnegative_number(spectral_interval.get("lower"))
    )
    same_scheme = bool(
        _literal_true(xi.get("same_scheme_remainder_computed"))
        and _ordered_interval(xi)
        and _ordered_interval(delta)
        and _ordered_interval(ath)
        and _ordered_interval(pixel)
        and all(
            item.get("scheme") == xi.get("scheme") and _nonempty_string(item.get("scheme"))
            for item in (delta, ath, pixel)
        )
    )
    systematics_passed = _systematics_pass(systematics)
    no_target_leak = bool(
        isinstance(source_dag.get("source_inputs"), list)
        and bool(source_dag.get("source_inputs"))
        and source_dag.get("forbidden_edges") == []
        and _all_literal_true(
            no_target,
            "audit_passed",
            "source_frozen_before_comparison",
        )
        and no_target.get("forbidden_edges") == []
        and exclusions.get("forbidden_inputs_present") == []
        and _literal_true(exclusions.get("source_frozen_before_comparison"))
        and _literal_true(comparison.get("attached_after_source_freeze"))
    )
    higher_point = bool(
        _all_literal_true(q4, "spectral_export_receipt", "positive_measure_receipt")
        and _positive_int(q4.get("sample_count"))
    )
    transitions = bool(
        all(
            _all_literal_true(item, "spectral_export_receipt", "positive_measure_receipt")
            and _positive_int(item.get("sample_count"))
            for item in (b_transition, sigma_transition)
        )
    )
    frozen_code_passed = bool(
        isinstance(frozen_hashes.get("code_hashes"), dict)
        and bool(frozen_hashes.get("code_hashes"))
        and all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in frozen_hashes.get("code_hashes", {}).values()
        )
    )
    replay_passed = bool(
        _all_literal_true(replay, "independent_replay_passed", "deterministic_replay_passed")
        and _positive_int(replay.get("replay_count"))
    )
    provenance_passed = bool(
        all(provenance_checks.values()) and frozen_code_passed and replay_passed
    )

    gates = {
        "qcd_quotient_ensemble_receipt": qcd_quotient,
        "source_qcd_parameter_map_receipt": source_parameter_map,
        "euclidean_qcd_slab_receipt": euclidean_slab,
        "hadronic_hilbert_quotient_receipt": hadronic_hilbert,
        "ward_current_normalization_receipt": ward_current,
        "two_current_spectral_export_receipt": two_current,
        "same_scheme_remainder_receipt": same_scheme,
        "qcd_systematics_ledger_receipt": systematics_passed,
        "qcd_no_target_leak_dag_receipt": no_target_leak,
        "higher_point_spectral_exports_receipt": higher_point,
        "transition_spectral_exports_receipt": transitions,
    }
    semantic_checks = {
        **gates,
        "reflection_positivity_certificate": reflection_positive,
        "ward_residual_csv_receipt": ward_rows_passed,
        "ward_residual_row_count": ward_row_count,
        "frozen_code_hashes_receipt": frozen_code_passed,
        "independent_replay_receipt": replay_passed,
    }
    blockers = [
        f"provenance_{key}" for key, value in provenance_checks.items() if not value
    ]
    if not artifact_hashes_passed:
        blockers.extend(
            f"artifact_{row['path']}_{'missing' if not row['exists'] else 'hash_mismatch'}"
            for row in hash_rows
            if not row["passed"]
        )
    blockers.extend(f"semantic_{key}" for key, value in gates.items() if not value)
    if not frozen_code_passed:
        blockers.append("semantic_frozen_code_hashes_receipt")
    if not replay_passed:
        blockers.append("semantic_independent_replay_receipt")

    return {
        "schema": EVIDENCE_SCHEMA,
        "evidence_dir": str(root),
        "written": True,
        "provenance_passed": provenance_passed,
        "artifact_hashes_passed": artifact_hashes_passed,
        "no_target_leak_passed": no_target_leak,
        "receipt_gates": gates,
        "provenance_checks": provenance_checks,
        "artifact_hashes": hash_rows,
        "semantic_checks": semantic_checks,
        "blockers": _unique(blockers),
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ward_csv_passes(path: Path) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False, 0
    passed = bool(rows)
    for row in rows:
        residual = _csv_number(row.get("residual"))
        bound = _csv_number(row.get("bound"))
        passed = bool(
            passed
            and residual is not None
            and bound is not None
            and bound >= 0.0
            and abs(residual) <= bound
        )
    return passed, len(rows)


def _positivity_certificate(data: dict[str, Any]) -> bool:
    minimum = _number(data.get("minimum_eigenvalue"))
    tolerance = _number(data.get("tolerance"))
    return bool(
        _literal_true(data.get("positivity_tested"))
        and minimum is not None
        and tolerance is not None
        and tolerance >= 0.0
        and minimum >= -tolerance
        and _positive_int(data.get("sample_count"))
    )


def _systematics_pass(data: dict[str, Any]) -> bool:
    components = data.get("components")
    if not isinstance(components, dict) or not components:
        return False
    required = {"statistical", "discretization", "finite_volume", "renormalization", "continuum"}
    if not required <= set(components):
        return False
    values = [_number(components.get(key)) for key in required]
    total = _number(data.get("total_uncertainty"))
    if any(value is None or value < 0.0 for value in values) or total is None or total < 0.0:
        return False
    rss = math.sqrt(sum(float(value) ** 2 for value in values if value is not None))
    return bool(
        _literal_true(data.get("all_systematics_bounded"))
        and total + 1.0e-12 >= rss
        and _int_at_least(data.get("refinement_level_count"), 2)
        and _int_at_least(data.get("independent_seed_count"), 2)
    )


def _ordered_interval(data: dict[str, Any]) -> bool:
    lower = _number(data.get("lower"))
    upper = _number(data.get("upper"))
    return lower is not None and upper is not None and lower <= upper


def _all_literal_true(data: dict[str, Any], *keys: str) -> bool:
    return all(_literal_true(data.get(key)) for key in keys)


def _literal_true(value: Any) -> bool:
    return type(value) is bool and value is True


def _number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _csv_number(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_number(value: Any) -> bool:
    return _number(value) is not None


def _positive_number(value: Any) -> bool:
    parsed = _number(value)
    return parsed is not None and parsed > 0.0


def _nonnegative_number(value: Any) -> bool:
    parsed = _number(value)
    return parsed is not None and parsed >= 0.0


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _int_at_least(value: Any, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def _finite_number_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_finite_number(item) for item in value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


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
