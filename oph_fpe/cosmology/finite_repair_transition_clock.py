from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.edge_center_clock import (
    E_DIAGNOSTIC_KAPPA,
    EdgeCenterClockTarget,
    edge_center_clock_target,
    validate_edge_center_clock_evidence,
)
from oph_fpe.cosmology.scalar_repair_semigroup import ScalarRepairSemigroupSpec, scalar_repair_semigroup_report


DEFAULT_PACKET_FIELDS = ("checkpoint_class", "stable_flag", "s3_sector_class", "repair_load_bucket")
DETAILED_BALANCE_TOLERANCE = 1.0e-12
SPECTRAL_GAP_TOLERANCE = 1.0e-12
DEFAULT_PACKET_FIELD_SWEEP = (
    ("checkpoint", ("checkpoint_class",)),
    ("checkpoint_stable", ("checkpoint_class", "stable_flag")),
    ("checkpoint_sector", ("checkpoint_class", "s3_sector_class")),
    ("checkpoint_repair", ("checkpoint_class", "repair_load_bucket")),
    ("checkpoint_sector_repair", ("checkpoint_class", "s3_sector_class", "repair_load_bucket")),
    ("record", ("record_family",)),
    ("record_checkpoint", ("record_family", "checkpoint_class")),
    ("record_sector", ("record_family", "s3_sector_class")),
    ("record_repair", ("record_family", "repair_load_bucket")),
    ("record_checkpoint_sector", ("record_family", "checkpoint_class", "s3_sector_class")),
    (
        "record_checkpoint_sector_repair",
        ("record_family", "checkpoint_class", "s3_sector_class", "repair_load_bucket"),
    ),
    ("default", DEFAULT_PACKET_FIELDS),
    (
        "all_step_fields",
        ("record_family", "checkpoint_class", "stable_flag", "s3_sector_class", "repair_load_bucket"),
    ),
)


@dataclass(frozen=True)
class FiniteRepairTransitionClockConfig:
    packet_fields: tuple[str, ...] = DEFAULT_PACKET_FIELDS
    primary_matrix: str = "raw_empirical"
    repair_step_time: float = 1.0
    clock_normalization_source: str = "declared_cli_value"
    weight_field: str = "transition_history_mean_modal_mass"
    min_transition_count: int = 1
    p_value: float = P_STAR
    clock_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FiniteRepairTransitionSweepConfig:
    packet_fieldsets: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_PACKET_FIELD_SWEEP
    primary_matrices: tuple[str, ...] = ("raw_empirical", "reversible_empirical")
    repair_step_times: tuple[float, ...] = (1.0,)
    clock_normalization_source: str = "sweep_declared_values"
    weight_field: str = "transition_history_mean_modal_mass"
    min_transition_count: int = 1
    p_value: float = P_STAR


def validate_transition_clock_eligibility(report: dict[str, Any] | None) -> dict[str, Any]:
    """Recompute transition-clock eligibility without trusting cached booleans.

    Historical reports produced before the fail-closed gate can contain
    ``finite_transition_matrix_ready=true`` even when their primary chain is
    reducible, periodic, or gapless.  Every consumer must call this validator
    and use ``eligible`` rather than copying those legacy flags.

    A scalar-semigroup sidecar is accepted only when it embeds the raw primary
    metadata in ``transition_matrix_certificate``.  Older sidecars lack that
    evidence and therefore fail closed.
    """

    source = report if isinstance(report, dict) else {}
    evidence = source
    if not isinstance(evidence.get("primary"), dict):
        certificate = evidence.get("transition_matrix_certificate")
        evidence = certificate if isinstance(certificate, dict) else {}

    primary = evidence.get("primary")
    primary = primary if isinstance(primary, dict) else {}
    state_count = _strict_nonnegative_int(evidence.get("state_count"))
    transition_count = _strict_nonnegative_int(evidence.get("transition_count"))
    lambda_2 = _float_or_none(primary.get("lambda_2"))
    detailed_balance_error = _float_or_none(primary.get("detailed_balance_max_abs_error"))

    checks = {
        "state_count_at_least_two": state_count is not None and state_count >= 2,
        "transition_count_positive": transition_count is not None and transition_count > 0,
        "primary_finite": primary.get("finite") is True,
        "primary_irreducible": primary.get("irreducible") is True,
        "primary_aperiodic": primary.get("aperiodic") is True,
        "primary_spectral_gap": (
            lambda_2 is not None
            and 0.0 <= lambda_2 < 1.0 - SPECTRAL_GAP_TOLERANCE
        ),
        "primary_detailed_balance": (
            detailed_balance_error is not None
            and 0.0 <= detailed_balance_error <= DETAILED_BALANCE_TOLERANCE
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "oph_transition_clock_eligibility_v1",
        "eligible": not blockers,
        "state_count": state_count,
        "transition_count": transition_count,
        "lambda_2": lambda_2,
        "detailed_balance_max_abs_error": detailed_balance_error,
        "checks": checks,
        "blockers": blockers,
        "legacy_ready_flags_ignored": {
            "finite_transition_matrix_ready": source.get("finite_transition_matrix_ready"),
            "finite_lattice_derived": source.get("finite_lattice_derived"),
            "physical_cmb_eligible_eta_R_empirical": source.get(
                "physical_cmb_eligible_eta_R_empirical"
            ),
        },
    }


def write_finite_repair_transition_clock_report(
    run_dir: Path,
    out_dir: Path,
    *,
    packet_fields: tuple[str, ...] = DEFAULT_PACKET_FIELDS,
    primary_matrix: str = "raw_empirical",
    repair_step_time: float = 1.0,
    clock_normalization_source: str = "declared_cli_value",
    weight_field: str = "transition_history_mean_modal_mass",
    min_transition_count: int = 1,
    p_value: float = P_STAR,
    clock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report, matrices = finite_repair_transition_clock_report(
        run_dir,
        FiniteRepairTransitionClockConfig(
            packet_fields=tuple(packet_fields),
            primary_matrix=primary_matrix,
            repair_step_time=repair_step_time,
            clock_normalization_source=clock_normalization_source,
            weight_field=weight_field,
            min_transition_count=min_transition_count,
            p_value=p_value,
            clock_evidence=clock_evidence,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "finite_repair_transition_matrix.npz",
        counts=matrices["counts"],
        raw_empirical=matrices["raw_empirical"],
        reversible_empirical=matrices["reversible_empirical"],
        state_labels=np.asarray(report["state_labels"], dtype=object),
    )
    (out_dir / "finite_repair_transition_matrix_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_matrix_rows(out_dir / "finite_repair_transition_rows.csv", report)

    eligibility = validate_transition_clock_eligibility(report)
    scalar = scalar_repair_semigroup_report(
        ScalarRepairSemigroupSpec(
            dimension=max(int(report["state_count"]), 2),
            kappa_rep=float(report["primary"]["kappa_rep_estimate"]),
            source="finite_state_transition_matrix",
            finite_lattice_derived=bool(eligibility["eligible"]),
            matrix_source=str(out_dir / "finite_repair_transition_matrix.npz"),
            p_value=p_value,
            clock_evidence=clock_evidence,
        )
    )
    scalar["transition_matrix_certificate"] = {
        "source_report": str(out_dir / "finite_repair_transition_matrix_report.json"),
        "primary_matrix": report["primary_matrix"],
        "state_count": report["state_count"],
        "transition_count": report["transition_count"],
        "primary": {
            "finite": report["primary"].get("finite"),
            "irreducible": report["primary"].get("irreducible"),
            "aperiodic": report["primary"].get("aperiodic"),
            "lambda_2": report["primary"].get("lambda_2"),
            "detailed_balance_max_abs_error": report["primary"].get(
                "detailed_balance_max_abs_error"
            ),
        },
        "eligibility": eligibility,
        "matrix_ready": bool(eligibility["eligible"]),
        "clock_normalization_certified": bool(report["clock_normalization_certified"]),
        "required_repair_step_time_for_selected_edge_center": report["primary"].get(
            "required_repair_step_time_for_selected_edge_center"
        ),
        "repair_step_time_for_e_diagnostic_control": report["primary"].get(
            "repair_step_time_for_e_diagnostic_control"
        ),
        "clock_normalization_candidates": report.get("clock_normalization_candidates", []),
        "edge_center_clock_evidence": report.get("edge_center_clock_evidence", {}),
        "primary_lambda_2": report["primary"].get("lambda_2"),
        "primary_gamma": report["primary"].get("gamma_continuous"),
    }
    scalar["eligible_for_repair_clock_certificate"] = bool(
        eligibility["eligible"]
        and scalar["eligible_for_repair_clock_certificate"]
        and report["clock_normalization_certified"]
    )
    scalar["repair_clock_certificate"] = bool(
        eligibility["eligible"]
        and scalar["repair_clock_certificate"]
        and report["clock_normalization_certified"]
    )
    (out_dir / "scalar_repair_semigroup_report.json").write_text(
        json.dumps(scalar, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def write_finite_repair_transition_clock_sweep_report(
    run_dir: Path,
    out_dir: Path,
    *,
    packet_fieldsets: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_PACKET_FIELD_SWEEP,
    primary_matrices: tuple[str, ...] = ("raw_empirical", "reversible_empirical"),
    repair_step_times: tuple[float, ...] = (1.0,),
    clock_normalization_source: str = "sweep_declared_values",
    weight_field: str = "transition_history_mean_modal_mass",
    min_transition_count: int = 1,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    report = finite_repair_transition_clock_sweep_report(
        run_dir,
        FiniteRepairTransitionSweepConfig(
            packet_fieldsets=packet_fieldsets,
            primary_matrices=tuple(primary_matrices),
            repair_step_times=tuple(float(value) for value in repair_step_times),
            clock_normalization_source=clock_normalization_source,
            weight_field=weight_field,
            min_transition_count=min_transition_count,
            p_value=p_value,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finite_repair_transition_clock_sweep_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_sweep_rows(out_dir / "finite_repair_transition_clock_sweep_rows.csv", report["rows"])
    return report


def finite_repair_transition_clock_sweep_report(
    run_dir: Path,
    config: FiniteRepairTransitionSweepConfig | None = None,
) -> dict[str, Any]:
    """Sweep observer-visible quotient choices for the repair-clock matrix.

    This is an audit tool, not an optimizer that can certify physics by search.
    It answers a narrow implementation question: whether a different declared
    support-visible packet quotient already present in observer_views.jsonl
    produces a finite transition matrix and clock closer to the OPH target.
    """

    config = config or FiniteRepairTransitionSweepConfig()
    rows: list[dict[str, Any]] = []
    for fieldset_name, fields in config.packet_fieldsets:
        for primary_matrix in config.primary_matrices:
            for repair_step_time in config.repair_step_times:
                try:
                    child, _ = finite_repair_transition_clock_report(
                        run_dir,
                        FiniteRepairTransitionClockConfig(
                            packet_fields=tuple(fields),
                            primary_matrix=str(primary_matrix),
                            repair_step_time=float(repair_step_time),
                            clock_normalization_source=config.clock_normalization_source,
                            weight_field=config.weight_field,
                            min_transition_count=config.min_transition_count,
                            p_value=config.p_value,
                        ),
                    )
                    rows.append(_sweep_row(fieldset_name, child))
                except Exception as exc:  # pragma: no cover - defensive audit path
                    rows.append(
                        {
                            "field_set_name": str(fieldset_name),
                            "packet_fields": list(fields),
                            "primary_matrix": str(primary_matrix),
                            "repair_step_time": float(repair_step_time),
                            "failed": True,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    finite_rows = [
        row
        for row in rows
        if row.get("finite_transition_matrix_ready")
        and _float_or_none(row.get("relative_error_to_selected_edge_center_kappa")) is not None
    ]
    finite_rows.sort(key=lambda row: float(row["relative_error_to_selected_edge_center_kappa"]))
    certified_rows = [row for row in finite_rows if row.get("repair_clock_certificate")]
    target = edge_center_clock_target(float(config.p_value))
    return {
        "mode": "oph_finite_repair_transition_clock_sweep_v1",
        "source_run_dir": str(Path(run_dir)),
        "target": {
            "selected_branch": "edge_center_orientation_half",
            "required_kappa_rep": target.kappa_rep,
            "required_eta_R": target.theta,
            "required_n_s": target.n_s,
            "full_collar_derivative_target": target.full_collar_derivative,
            "orientation_halves": target.orientation_halves,
            "P": float(target.P),
            "phi": float(target.phi),
            "delta_P": target.delta_P,
            "e_diagnostic_control": target.as_jsonable()["diagnostic_controls"]["e"],
        },
        "inputs": {
            "packet_fieldset_count": len(config.packet_fieldsets),
            "primary_matrices": list(config.primary_matrices),
            "repair_step_times": [float(value) for value in config.repair_step_times],
            "clock_normalization_source": config.clock_normalization_source,
            "clock_normalization_source_status": _clock_normalization_source_status(
                config.clock_normalization_source
            ),
            "weight_field": config.weight_field,
            "min_transition_count": int(config.min_transition_count),
        },
        "summary": {
            "row_count": len(rows),
            "finite_ready_count": len(finite_rows),
            "certified_count": len(certified_rows),
            "best_finite_row": finite_rows[0] if finite_rows else None,
            "best_certified_row": certified_rows[0] if certified_rows else None,
            "distinct_state_counts": sorted(
                {
                    int(row["state_count"])
                    for row in rows
                    if _float_or_none(row.get("state_count")) is not None
                }
            ),
        },
        "rows": rows,
        "repair_clock_certificate": bool(certified_rows),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite repair-clock quotient sweep over observer-visible transition-history packet fields. "
            "It may identify a finite diagnostic quotient closer to the selected P/48 edge-center target, "
            "but search cannot supply the full-collar, orientation-half, semigroup/refinement, clock-binding, "
            "or source-DAG evidence receipts required for promotion."
        ),
    }


def finite_repair_transition_clock_report(
    run_dir: Path,
    config: FiniteRepairTransitionClockConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = config or FiniteRepairTransitionClockConfig()
    if config.primary_matrix not in {"raw_empirical", "reversible_empirical"}:
        raise ValueError("primary_matrix must be raw_empirical or reversible_empirical")
    if config.repair_step_time <= 0.0:
        raise ValueError("repair_step_time must be positive")
    observer_path = Path(run_dir) / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)

    counts, labels, observer_count, transition_count, skipped = _transition_counts_from_observer_views(
        observer_path,
        fields=config.packet_fields,
        weight_field=config.weight_field,
        min_transition_count=config.min_transition_count,
    )
    raw = _row_stochastic(counts)
    reversible, reversibilization_method = _reversible_projection(counts)
    pixel = OPHPixelConstants(P=float(config.p_value))
    clock_target = edge_center_clock_target(float(config.p_value))
    clock_evidence = validate_edge_center_clock_evidence(
        config.clock_evidence,
        P=float(config.p_value),
    )
    delta_p = clock_target.delta_P
    target_kappa = clock_target.kappa_rep
    target_eta = clock_target.theta
    raw_summary = _matrix_summary(
        raw,
        target=clock_target,
        repair_step_time=config.repair_step_time,
    )
    reversible_summary = _matrix_summary(
        reversible,
        target=clock_target,
        repair_step_time=config.repair_step_time,
    )
    reversible_summary["reversibilization_method"] = reversibilization_method
    matrix_summaries = {
        "raw_empirical": raw_summary,
        "reversible_empirical": reversible_summary,
    }
    primary = matrix_summaries[config.primary_matrix]
    eligibility = validate_transition_clock_eligibility(
        {
            "state_count": int(len(labels)),
            "transition_count": int(transition_count),
            "primary": primary,
        }
    )
    finite_ready = bool(eligibility["eligible"])
    rel_error = abs(float(primary["kappa_rep_estimate"]) - target_kappa) / target_kappa if finite_ready else None
    rel_error_e_diagnostic = (
        abs(float(primary["kappa_rep_estimate"]) - E_DIAGNOSTIC_KAPPA) / E_DIAGNOSTIC_KAPPA
        if finite_ready
        else None
    )
    source_status = _clock_normalization_source_status(config.clock_normalization_source)
    numeric_clock_match = bool(finite_ready and rel_error is not None and rel_error <= 0.05)
    e_diagnostic_match = bool(
        finite_ready
        and rel_error_e_diagnostic is not None
        and rel_error_e_diagnostic <= 0.05
    )
    clock_certified = bool(
        numeric_clock_match
        and clock_evidence["edge_center_clock_evidence_complete"]
    )
    clock_modes = _clock_mode_reports(
        primary,
        finite_ready=finite_ready,
        target=clock_target,
        clock_certified=clock_certified,
        numeric_clock_match=numeric_clock_match,
    )
    clock_candidates = _clock_normalization_candidates(
        Path(run_dir),
        required_step_time=primary.get(
            "required_repair_step_time_for_selected_edge_center"
        ),
        primary=primary,
        observer_path=observer_path,
    )
    report = {
        "mode": "oph_finite_repair_transition_clock_v1",
        "source_run_dir": str(Path(run_dir)),
        "observer_views_path": str(observer_path),
        "packet_fields": list(config.packet_fields),
        "primary_matrix": config.primary_matrix,
        "repair_step_time": float(config.repair_step_time),
        "clock_normalization_source": config.clock_normalization_source,
        "clock_normalization_source_status": source_status,
        "weight_field": config.weight_field,
        "observer_count": int(observer_count),
        "transition_count": int(transition_count),
        "skipped_observer_count": int(skipped),
        "state_count": int(len(labels)),
        "state_labels": [json.dumps(label, sort_keys=True) for label in labels],
        "target": {
            "selected_branch": "edge_center_orientation_half",
            "formula": "rho_full=P/24; theta=rho_full/2=P/48; kappa_rep=theta/(P-phi)",
            "required_kappa_rep": target_kappa,
            "required_eta_R": target_eta,
            "required_n_s": clock_target.n_s,
            "full_collar_derivative_target": clock_target.full_collar_derivative,
            "orientation_halves": clock_target.orientation_halves,
            "P": float(pixel.P),
            "phi": float(pixel.phi),
            "delta_P": delta_p,
            "e_diagnostic_control": clock_target.as_jsonable()["diagnostic_controls"]["e"],
        },
        "matrices": matrix_summaries,
        "primary": primary,
        "transition_clock_eligibility": eligibility,
        "clock_normalization_candidates": clock_candidates,
        "clock_normalization_candidate_provenance": {
            "config_declared": sum(
                1 for c in clock_candidates if c.get("provenance") == "config_declared"
            ),
            "matrix_derived": sum(
                1 for c in clock_candidates if c.get("provenance") == "matrix_derived"
            ),
            "run_data_derived": sum(
                1 for c in clock_candidates if c.get("provenance") == "run_data_derived"
            ),
            "derived_from_run_data_count": sum(
                1 for c in clock_candidates if c.get("derived_from_run_data")
            ),
            "usable_derived_candidate_count": sum(
                1
                for c in clock_candidates
                if c.get("usable_for_physical_clock_binding")
            ),
            "theorem_bound_certified_count": sum(
                1 for c in clock_candidates if c.get("theorem_bound_certified")
            ),
            "claim_boundary": (
                "Declared/configured quantities and run-derived quantities are diagnostics only. "
                "A derived value becomes usable for physical clock binding only with its stated "
                "assumptions and antecedent evidence, and certification remains exclusively bound "
                "to the complete edge-center evidence receipts."
            ),
        },
        "relative_error_to_selected_edge_center_kappa": rel_error,
        "relative_error_to_e_diagnostic_control": rel_error_e_diagnostic,
        "clock_normalization_numeric_match": numeric_clock_match,
        "edge_center_target_numeric_match": numeric_clock_match,
        "e_diagnostic_numeric_match": e_diagnostic_match,
        "finite_transition_matrix_ready": finite_ready,
        "finite_transition_matrix_derived": finite_ready,
        "finite_step_survival_exponent_derived": finite_ready,
        "finite_lattice_derived": finite_ready,
        "edge_center_clock_evidence": clock_evidence,
        **clock_evidence["receipts"],
        "EDGE_CENTER_CLOCK_RECEIPT": clock_evidence["EDGE_CENTER_CLOCK_RECEIPT"],
        "clock_normalization_certified": clock_certified,
        "repair_clock_certificate": clock_certified,
        "eta_R_finite_lattice_derived": clock_certified,
        "clock_modes": clock_modes,
        "repair_clock_edge_center_certificate": clock_certified,
        "repair_clock_empirical_certificate": False,
        "repair_clock_e_diagnostic_certificate": False,
        "eta_R_empirical_finite_lattice_derived": False,
        "physical_cmb_eligible_eta_R_empirical": False,
        "finite_step_survival": {
            "lambda_2": primary.get("lambda_2"),
            "exponent_per_declared_step": primary.get("finite_step_survival_exponent"),
            "declared_step_time": float(config.repair_step_time),
            "distinct_from_full_collar_derivative": True,
            "satisfies_full_collar_derivative_receipt": False,
            "promoting": False,
        },
        "physical_cmb_prediction": False,
        "blockers": _blockers(
            finite_ready,
            clock_certified,
            numeric_clock_match,
            source_status,
            primary,
            clock_evidence,
        ),
        "claim_boundary": (
            "Finite observer-visible transition-matrix clock diagnostic. Packet paths are read from "
            "observer_views.jsonl and projected to a declared support-visible quotient alphabet. The report "
            "derives a finite-step survival exponent, which remains distinct from the full-collar derivative. "
            "The selected clock is theta=P/48 with kappa_rep=(P/48)/(P-phi), and it can certify only when "
            "all explicit edge-center evidence receipts pass. Euler's number is a named nonpromoting "
            "diagnostic control only."
        ),
    }
    return report, {"counts": counts, "raw_empirical": raw, "reversible_empirical": reversible}


def _sweep_row(fieldset_name: str, report: dict[str, Any]) -> dict[str, Any]:
    primary = report.get("primary", {}) if isinstance(report, dict) else {}
    eligibility = validate_transition_clock_eligibility(report)
    rel_error = _float_or_none(report.get("relative_error_to_selected_edge_center_kappa"))
    required_step = _float_or_none(primary.get("required_repair_step_time_for_selected_edge_center"))
    declared_step = _float_or_none(report.get("repair_step_time"))
    declared_vs_required = (
        abs(declared_step - required_step) / max(abs(required_step), 1.0e-30)
        if declared_step is not None and required_step is not None and required_step > 0.0
        else None
    )
    return {
        "field_set_name": str(fieldset_name),
        "packet_fields": list(report.get("packet_fields", [])),
        "primary_matrix": report.get("primary_matrix"),
        "repair_step_time": declared_step,
        "clock_normalization_source": report.get("clock_normalization_source"),
        "clock_normalization_source_status": report.get("clock_normalization_source_status"),
        "state_count": report.get("state_count"),
        "transition_count": report.get("transition_count"),
        "finite_transition_matrix_ready": bool(eligibility["eligible"]),
        "finite_lattice_derived": bool(eligibility["eligible"]),
        "transition_clock_eligibility": eligibility,
        "clock_normalization_certified": bool(
            eligibility["eligible"] and report.get("clock_normalization_certified", False)
        ),
        "clock_normalization_numeric_match": bool(
            eligibility["eligible"]
            and report.get("clock_normalization_numeric_match", False)
        ),
        "repair_clock_certificate": bool(
            eligibility["eligible"] and report.get("repair_clock_certificate", False)
        ),
        "irreducible": primary.get("irreducible"),
        "aperiodic": primary.get("aperiodic"),
        "lambda_2": primary.get("lambda_2"),
        "gamma_continuous": primary.get("gamma_continuous"),
        "kappa_rep_estimate": primary.get("kappa_rep_estimate"),
        "eta_R_estimate": primary.get("eta_R_estimate"),
        "n_s_estimate": primary.get("n_s_estimate"),
        "relative_error_to_selected_edge_center_kappa": rel_error,
        "relative_error_to_e_diagnostic_control": report.get(
            "relative_error_to_e_diagnostic_control"
        ),
        "required_repair_step_time_for_selected_edge_center": required_step,
        "repair_step_time_for_e_diagnostic_control": primary.get(
            "repair_step_time_for_e_diagnostic_control"
        ),
        "declared_step_relative_error_to_required": declared_vs_required,
        "detailed_balance_max_abs_error": primary.get("detailed_balance_max_abs_error"),
        "top_abs_eigenvalues": primary.get("top_abs_eigenvalues"),
        "blockers": report.get("blockers", []),
    }


def _clock_normalization_source_status(source: str) -> dict[str, Any]:
    value = str(source or "declared_cli_value")
    theorem_grade = value in {
        "paper_theorem_predeclared",
        "finite_theorem_predeclared",
    }
    hypothesis_side = value in {
        "repair_scale_48_hypothesis",
        "crc48_repair_scale_hypothesis",
    }
    return {
        "source": value,
        "theorem_grade": bool(theorem_grade),
        "hypothesis_side": bool(hypothesis_side),
        "posthoc_or_cli_declared": bool(not theorem_grade and not hypothesis_side),
        "sufficient_for_clock_binding_receipt": False,
        "claim_boundary": (
            "A source label alone never closes the physical clock-binding receipt. Even a theorem-grade "
            "label requires a hash-bound evidence bundle and a clean source DAG; hypothesis-side and "
            "CLI-declared sources remain diagnostics."
        ),
    }


def _clock_mode_reports(
    primary: dict[str, Any],
    *,
    finite_ready: bool,
    target: EdgeCenterClockTarget,
    clock_certified: bool,
    numeric_clock_match: bool,
) -> dict[str, dict[str, Any]]:
    kappa = _float_or_none(primary.get("kappa_rep_estimate"))
    eta = _float_or_none(primary.get("eta_R_estimate"))
    return {
        "empirical": {
            "clock_mode": "empirical",
            "clock_normalization_certified": False,
            "eta_R_finite_lattice_derived": False,
            "finite_step_survival_exponent_derived": bool(
                finite_ready and kappa is not None and eta is not None
            ),
            "eta_R_hypothesis": False,
            "kappa_rep_value": kappa,
            "eta_R_value": eta,
            "promoting": False,
            "claim_boundary": (
                "Finite-derived survival-exponent diagnostic. It is not the source full-collar derivative "
                "and cannot promote eta_R without the edge-center evidence bundle."
            ),
        },
        "edge_center_selected": {
            "clock_mode": "edge_center_selected",
            "clock_normalization_certified": bool(clock_certified),
            "eta_R_finite_lattice_derived": bool(clock_certified),
            "eta_R_hypothesis": False,
            "numeric_match": bool(numeric_clock_match),
            "kappa_rep_value": target.kappa_rep,
            "eta_R_value": target.theta,
            "n_s_value": target.n_s,
            "selected_theorem_target": True,
            "claim_boundary": (
                "Selected theorem target theta=P/48. Numerical proximity is insufficient; all full-collar, "
                "orientation-half, semigroup/refinement, clock-binding, and source-DAG receipts must pass."
            ),
        },
        "e_diagnostic": {
            "clock_mode": "e_diagnostic",
            "clock_normalization_certified": False,
            "eta_R_finite_lattice_derived": False,
            "eta_R_hypothesis": False,
            "kappa_rep_value": E_DIAGNOSTIC_KAPPA,
            "eta_R_value": float(E_DIAGNOSTIC_KAPPA * target.delta_P),
            "selected": False,
            "required": False,
            "canonical": False,
            "promoting": False,
            "claim_boundary": "Euler's number is retained only as a named nonpromoting diagnostic control.",
        },
    }


def _transition_counts_from_observer_views(
    path: Path,
    *,
    fields: tuple[str, ...],
    weight_field: str,
    min_transition_count: int,
) -> tuple[np.ndarray, list[tuple[tuple[str, int], ...]], int, int, int]:
    state_to_idx: dict[tuple[tuple[str, int], ...], int] = {}
    counts: Counter[tuple[int, int]] = Counter()
    observer_count = 0
    transition_count = 0
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observer_count += 1
            view = json.loads(line)
            descriptor = view.get("transition_history_descriptor") or {}
            steps = descriptor.get("steps") or []
            if len(steps) < 2:
                skipped += 1
                continue
            weight = _float_or_default(view.get(weight_field), 1.0)
            encoded = []
            for step in steps:
                key = tuple((field, int(step.get(field, 0))) for field in fields)
                encoded.append(state_to_idx.setdefault(key, len(state_to_idx)))
            window = list(zip(encoded, encoded[1:], strict=False))
            if len(window) < min_transition_count:
                # Filtered windows contribute nothing to the count matrix;
                # committing first and skipping afterwards would contaminate
                # the matrix while reporting the observer as skipped.
                skipped += 1
                continue
            for left, right in window:
                counts[(left, right)] += float(weight)
                transition_count += 1
    labels = [None] * len(state_to_idx)
    for key, idx in state_to_idx.items():
        labels[idx] = key
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for (left, right), value in counts.items():
        matrix[left, right] += float(value)
    return matrix, labels, observer_count, transition_count, skipped


def _row_stochastic(counts: np.ndarray) -> np.ndarray:
    if counts.size == 0:
        return counts.copy()
    row_sum = counts.sum(axis=1)
    matrix = np.zeros_like(counts, dtype=np.float64)
    active = row_sum > 0.0
    matrix[active] = counts[active] / row_sum[active, None]
    for idx, is_active in enumerate(active):
        if not is_active:
            matrix[idx, idx] = 1.0
    return matrix


def _reversible_projection(counts: np.ndarray) -> tuple[np.ndarray, str]:
    """Reversibilize the empirical chain.

    When the raw row-stochastic chain is irreducible, the additive
    reversibilization `(P + P*)/2` with `P*` the time reversal under the
    raw chain's stationary law is used; it preserves that stationary law
    and is reversible with respect to it, which is what the declared
    name promises. A reducible raw chain has no unique stationary law,
    so the symmetrized-count estimator is used instead and labeled as
    such; its stationary law is proportional to the symmetrized row
    sums and generally differs from the raw chain's.
    """
    if counts.size == 0:
        return counts.copy(), "empty"
    raw = _row_stochastic(counts)
    if _is_strongly_connected(raw > 1.0e-12):
        stationary = _stationary_distribution(raw)
        if np.all(stationary > 0.0):
            reversal = (stationary[None, :] * raw.T) / stationary[:, None]
            additive = 0.5 * (raw + reversal)
            return additive, "additive_stationary_reversibilization"
    symmetric = 0.5 * (counts + counts.T)
    if np.all(symmetric.sum(axis=1) <= 0.0):
        return np.eye(counts.shape[0], dtype=np.float64), "identity_fallback"
    return _row_stochastic(symmetric), "symmetrized_count_fallback_reducible_raw"


def _matrix_summary(
    matrix: np.ndarray,
    *,
    target: EdgeCenterClockTarget,
    repair_step_time: float,
) -> dict[str, Any]:
    if matrix.size == 0:
        return {"finite": False, "lambda_2": None, "gamma_continuous": None}
    row_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) if matrix.shape[0] else 0.0
    irreducible = _is_strongly_connected(matrix > 1.0e-12)
    aperiodic = bool(irreducible and np.any(np.diag(matrix) > 1.0e-12))
    vals = np.linalg.eigvals(matrix)
    sorted_abs = sorted((float(abs(value)) for value in vals), reverse=True)
    lambda_2 = sorted_abs[1] if len(sorted_abs) > 1 else 0.0
    gamma = -math.log(max(float(lambda_2), 1.0e-12)) / float(repair_step_time)
    kappa = gamma / max(target.delta_P, 1.0e-30)
    required_dt = gamma * repair_step_time / max(target.theta, 1.0e-30)
    e_diagnostic_dt = (
        gamma
        * repair_step_time
        / max(E_DIAGNOSTIC_KAPPA * target.delta_P, 1.0e-30)
    )
    stationary = _stationary_distribution(matrix)
    # A reducible chain has no unique stationary law in general.  Comparing the
    # matrix against whichever unit-eigenvector numpy happens to return can
    # therefore manufacture a zero detailed-balance residual for an absorbing
    # class while the full chain is not a valid reversible repair clock.
    detailed_balance_error = _detailed_balance_error(matrix, stationary) if irreducible else None
    return {
        "finite": bool(np.isfinite(gamma) and np.isfinite(lambda_2)),
        "irreducible": irreducible,
        "aperiodic": aperiodic,
        "row_sum_max_abs_error": row_error,
        "lambda_2": float(lambda_2),
        "gamma_continuous": float(gamma),
        "finite_step_survival_factor": float(lambda_2),
        "finite_step_survival_exponent": float(gamma),
        "finite_step_survival_exponent_is_full_collar_derivative": False,
        "finite_step_survival_exponent_is_promoting": False,
        "gamma_discrete_one_minus_lambda2": float(1.0 - lambda_2),
        "kappa_rep_estimate": float(kappa),
        "eta_R_estimate": float(kappa * target.delta_P),
        "n_s_estimate": float(1.0 - kappa * target.delta_P),
        "required_repair_step_time_for_selected_edge_center": float(required_dt),
        "repair_step_time_for_e_diagnostic_control": float(e_diagnostic_dt),
        "stationary_min": float(np.min(stationary)) if stationary.size else None,
        "stationary_max": float(np.max(stationary)) if stationary.size else None,
        "detailed_balance_max_abs_error": detailed_balance_error,
        "top_abs_eigenvalues": [float(value) for value in sorted_abs[:8]],
    }


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    vals, vecs = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vector = np.real(vecs[:, idx])
    vector = np.abs(vector)
    total = float(np.sum(vector))
    if total <= 1.0e-30:
        return np.full(matrix.shape[0], 1.0 / max(matrix.shape[0], 1), dtype=np.float64)
    return vector / total


def _detailed_balance_error(matrix: np.ndarray, stationary: np.ndarray) -> float | None:
    if matrix.size == 0 or stationary.size == 0:
        return None
    flow = stationary[:, None] * matrix
    return float(np.max(np.abs(flow - flow.T)))


def _blockers(
    finite_ready: bool,
    clock_certified: bool,
    numeric_clock_match: bool,
    source_status: dict[str, Any],
    primary: dict[str, Any],
    clock_evidence: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not finite_ready:
        blockers.append(
            "finite transition matrix is not certificate-ready "
            "(missing, degenerate, nonergodic, or numerically invalid)"
        )
    if primary.get("finite") and not primary.get("irreducible", False):
        blockers.append("primary transition matrix is reducible, so it is not a finite repair-matrix certificate")
    if primary.get("finite") and not primary.get("aperiodic", False):
        blockers.append("primary transition matrix is not certified aperiodic")
    detailed_balance_error = _float_or_none(primary.get("detailed_balance_max_abs_error"))
    if detailed_balance_error is not None and detailed_balance_error > DETAILED_BALANCE_TOLERANCE:
        blockers.append(
            "primary transition matrix is not a reversible/GNS self-adjoint repair operator"
        )
    if finite_ready and not numeric_clock_match:
        blockers.append(
            "finite-step survival exponent does not match the selected edge-center "
            "kappa_rep=(P/48)/(P-phi) under the declared repair-step time"
        )
    if finite_ready and numeric_clock_match and not clock_certified:
        blockers.append(
            "numerical agreement with the selected P/48 target is nonpromoting until the complete "
            "edge-center clock evidence bundle passes"
        )
    for receipt in clock_evidence.get("missing_receipts", []):
        blockers.append(f"missing edge-center clock evidence: {receipt}")
    if source_status.get("hypothesis_side"):
        blockers.append("legacy repair-scale hypothesis label is diagnostic and cannot bind the physical clock")
    return blockers


def _clock_normalization_candidates(
    run_dir: Path,
    *,
    required_step_time: Any,
    primary: dict[str, Any] | None = None,
    observer_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Repair-step-time diagnostics, each tagged with provenance and claim status.

    Three statuses remain distinct:

    * ``config_declared`` -- declared quantities read from ``config.yml``.  These
      are not derived from the run's dynamics and remain declared diagnostics.
    * ``matrix_derived`` / ``run_data_derived`` -- computed from the run itself
      without a CLI-declared step time, but remain run-derived diagnostics here.
    * theorem-bound certification -- not assigned by this helper.  It is decided
      only by the edge-center evidence receipts in the report caller.

    Derived candidates are emitted even when degenerate, with ``degenerate=True``
    and a ``reason``, so the report fails closed *visibly* rather than dropping
    the diagnostic silently.
    """

    required = _float_or_none(required_step_time)

    def _rel_error(value: float | None) -> float | None:
        if value is None or required is None:
            return None
        return abs(float(value) - required) / max(abs(required), 1.0e-30)

    config = _read_yaml(run_dir / "config.yml")
    dynamics = config.get("dynamics", {}) if isinstance(config, dict) else {}
    bw = config.get("bw", {}) if isinstance(config, dict) else {}
    observer_objects = config.get("observer_objects", {}) if isinstance(config, dict) else {}
    cycles = _float_or_none(dynamics.get("cycles"))
    commit = _float_or_none(dynamics.get("record_commit_cycles"))
    history = _float_or_none(observer_objects.get("history_window"))
    times = bw.get("times") if isinstance(bw, dict) else None
    bw_time = _float_or_none(times[0]) if isinstance(times, list) and times else _float_or_none(bw.get("transition_response_time"))
    bw_scale = _float_or_none(bw.get("transition_response_scale")) or (2.0 * math.pi)
    bw_s = bw_time * bw_scale if bw_time is not None and bw_scale is not None else None
    declared: list[tuple[str, float | None, str]] = [
        ("unit_step", 1.0, "one simulator transition-history step"),
        ("record_commit_cycles", commit, "record commit horizon in cycles"),
        ("history_window", history, "observer transition-history window"),
        ("record_commit_cycles_times_2pi", commit * 2.0 * math.pi if commit is not None else None, "KMS 2pi times record commit horizon"),
        ("history_window_times_2pi", history * 2.0 * math.pi if history is not None else None, "KMS 2pi times observer history window"),
        (
            "cycles_times_bw_modular_time",
            cycles * bw_s if cycles is not None and bw_s is not None else None,
            "simulation cycles times BW modular parameter s=2pi*t",
        ),
        (
            "commit_times_history_times_bw_modular_time",
            commit * history * bw_s if commit is not None and history is not None and bw_s is not None else None,
            "record commit horizon times history window times BW modular parameter",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, value, description in declared:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            continue
        rows.append(
            {
                "name": name,
                "value": float(value),
                "relative_error_to_required": _rel_error(value),
                "description": description,
                "provenance": "config_declared",
                "derived_from_run_data": False,
                "claim_status": "declared_diagnostic",
                "usable_for_physical_clock_binding": False,
                "theorem_bound_certified": False,
                "degenerate": False,
            }
        )

    rows.extend(_derived_clock_candidates(primary, observer_path, rel_error=_rel_error))

    # finite (matched) candidates first by relative error; degenerate/None last.
    rows.sort(
        key=lambda row: (
            0 if row.get("relative_error_to_required") is not None else 1,
            float(row["relative_error_to_required"])
            if row.get("relative_error_to_required") is not None
            else float("inf"),
        )
    )
    return rows


def _derived_clock_candidates(
    primary: dict[str, Any] | None,
    observer_path: Path | None,
    *,
    rel_error,
) -> list[dict[str, Any]]:
    """Step-time diagnostics derived from run dynamics (no CLI declaration).

    #12 asks the report to distinguish run-derived diagnostics from usable,
    theorem-bound normalization.  Two diagnostics are unambiguous:

    * ``transition_matrix_spectral_relaxation_time`` -- the spectral diagnostic
      tau = -1/log(abs(lambda_2)).  It is deliberately not labelled a mixing time.
    * ``mean_first_passage_time`` -- observer-averaged mean of the run's
      ``first_passage_time_histogram`` (a measured dynamical timescale).

    Neither diagnostic binds the physical clock or satisfies an edge-center
    receipt by itself.
    """

    rows: list[dict[str, Any]] = []

    primary = primary or {}
    lambda_2 = _float_or_none(primary.get("lambda_2"))
    lambda_2_abs = abs(lambda_2) if lambda_2 is not None else None
    balance_error = _float_or_none(primary.get("detailed_balance_max_abs_error"))
    mixing_assumptions = {
        "finite": primary.get("finite") is True,
        "irreducible": primary.get("irreducible") is True,
        "aperiodic": primary.get("aperiodic") is True,
        "spectral_gap": (
            lambda_2_abs is not None
            and 0.0 <= lambda_2_abs < 1.0 - SPECTRAL_GAP_TOLERANCE
        ),
        "reversible_or_self_adjoint": (
            balance_error is not None
            and 0.0 <= balance_error <= DETAILED_BALANCE_TOLERANCE
        ),
    }
    relaxation: dict[str, Any] = {
        "name": "transition_matrix_spectral_relaxation_time",
        "provenance": "matrix_derived",
        "derived_from_run_data": True,
        "claim_status": "run_derived_diagnostic",
        "usable_for_physical_clock_binding": False,
        "theorem_bound_certified": False,
        "lambda_2": lambda_2,
        "lambda_2_abs": lambda_2_abs,
        "formula": "-1/log(abs(lambda_2))",
        "mixing_time_claimed": False,
        "mixing_assumptions": mixing_assumptions,
        "mixing_assumptions_passed": all(mixing_assumptions.values()),
        "description": (
            "spectral relaxation diagnostic of the finite repair transition matrix; "
            "not a physical clock binding or a claimed mixing time"
        ),
    }
    if (
        lambda_2_abs is not None
        and 0.0 < lambda_2_abs < 1.0 - SPECTRAL_GAP_TOLERANCE
    ):
        tau = -1.0 / math.log(lambda_2_abs)
        relaxation.update(
            value=float(tau),
            relative_error_to_required=rel_error(tau),
            degenerate=False,
        )
    else:
        relaxation.update(
            value=None,
            relative_error_to_required=None,
            degenerate=True,
            reason=(
                "abs(lambda_2) must lie strictly between zero and one for a finite "
                "positive spectral-relaxation diagnostic"
            ),
        )
    rows.append(relaxation)

    mean_fpt = _mean_first_passage_time(observer_path)
    fpt: dict[str, Any] = {
        "name": "mean_first_passage_time",
        "provenance": "run_data_derived",
        "derived_from_run_data": True,
        "claim_status": "run_derived_diagnostic",
        "usable_for_physical_clock_binding": False,
        "theorem_bound_certified": False,
        "description": (
            "observer-averaged mean of first_passage_time_histogram in "
            "observer_views.jsonl; diagnostic only"
        ),
    }
    if mean_fpt is not None and math.isfinite(mean_fpt) and mean_fpt > 0.0:
        fpt.update(value=float(mean_fpt), relative_error_to_required=rel_error(mean_fpt), degenerate=False)
    else:
        fpt.update(
            value=None,
            relative_error_to_required=None,
            degenerate=True,
            reason="first_passage_time_histogram absent or empty in observer_views.jsonl; fails closed",
        )
    rows.append(fpt)
    return rows


def _mean_first_passage_time(observer_path: Path | None) -> float | None:
    """Observer-averaged mean first-passage time from observer_views.jsonl.

    Each observer view may carry a ``first_passage_time_histogram`` mapping an
    integer step bin (as a string key) to a probability.  We normalize each
    observer's histogram, take its mean, and average those observer means.
    Returns None when no usable histogram is present (caller fails closed).
    """

    if observer_path is None or not Path(observer_path).exists():
        return None
    observer_mean_sum = 0.0
    observer_count = 0
    with Path(observer_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                view = json.loads(line)
            except json.JSONDecodeError:
                continue
            hist = view.get("first_passage_time_histogram")
            if not isinstance(hist, dict) or not hist:
                continue
            mass = 0.0
            weighted = 0.0
            for key, prob in hist.items():
                p = _float_or_none(prob)
                t = _float_or_none(key)
                if p is None or t is None or p < 0.0 or t < 0.0:
                    continue
                mass += p
                weighted += t * p
            if mass > 0.0:
                observer_mean_sum += weighted / mass
                observer_count += 1
    if observer_count == 0:
        return None
    return observer_mean_sum / observer_count


def _is_strongly_connected(adjacency: np.ndarray) -> bool:
    if adjacency.size == 0:
        return False
    n = int(adjacency.shape[0])
    if n == 1:
        return True
    return len(_reachable(adjacency, 0)) == n and len(_reachable(adjacency.T, 0)) == n


def _reachable(adjacency: np.ndarray, start: int) -> set[int]:
    seen = {int(start)}
    stack = [int(start)]
    while stack:
        node = stack.pop()
        for nxt in np.flatnonzero(adjacency[node]):
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _write_matrix_rows(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for name, summary in report.get("matrices", {}).items():
        rows.append(
            {
                "matrix": name,
                "lambda_2": summary.get("lambda_2"),
                "gamma_continuous": summary.get("gamma_continuous"),
                "finite_step_survival_exponent": summary.get(
                    "finite_step_survival_exponent"
                ),
                "kappa_rep_estimate": summary.get("kappa_rep_estimate"),
                "eta_R_estimate": summary.get("eta_R_estimate"),
                "required_repair_step_time_for_selected_edge_center": summary.get(
                    "required_repair_step_time_for_selected_edge_center"
                ),
                "repair_step_time_for_e_diagnostic_control": summary.get(
                    "repair_step_time_for_e_diagnostic_control"
                ),
                "detailed_balance_max_abs_error": summary.get("detailed_balance_max_abs_error"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["matrix"])
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "field_set_name",
        "packet_fields",
        "primary_matrix",
        "repair_step_time",
        "state_count",
        "transition_count",
        "finite_transition_matrix_ready",
        "clock_normalization_certified",
        "repair_clock_certificate",
        "irreducible",
        "aperiodic",
        "lambda_2",
        "gamma_continuous",
        "kappa_rep_estimate",
        "eta_R_estimate",
        "n_s_estimate",
        "relative_error_to_selected_edge_center_kappa",
        "relative_error_to_e_diagnostic_control",
        "required_repair_step_time_for_selected_edge_center",
        "repair_step_time_for_e_diagnostic_control",
        "declared_step_relative_error_to_required",
        "detailed_balance_max_abs_error",
        "blockers",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["packet_fields"] = ",".join(str(value) for value in row.get("packet_fields", []))
            out["blockers"] = ";".join(str(value) for value in row.get("blockers", []))
            writer.writerow(out)


def _float_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    return result if math.isfinite(result) else None


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}
