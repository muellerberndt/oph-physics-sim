from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import P_STAR, total_entropy_capacity
from oph_fpe.cosmology.oph_constants import (
    N_CRC_EW_COMPARISON_PIXEL,
    N_CRC_EW_COMPARISON_PIXEL_PROVENANCE,
    N_CRC_EW_FORWARD_CLOSURE_POINT,
    N_CRC_EW_FORWARD_CLOSURE_POINT_PROVENANCE,
    SOURCE_DERIVED_N_CRC_EW,
    SOURCE_DERIVED_N_CRC_EW_DEFAULT_BRANCH,
    SOURCE_DERIVED_N_CRC_EW_SOURCE,
    source_derived_n_crc_ew_jsonable,
)

DEFAULT_R_DS_M = 1.66e26
DEFAULT_L_PLANCK_M = 1.616e-35
DEFAULT_REGULATOR_PATCH_COUNTS = (4_096, 65_536, 262_144, 1_048_576)
DEFAULT_OBSERVED_HORIZON_N_COMPARISON = math.pi * (DEFAULT_R_DS_M / DEFAULT_L_PLANCK_M) ** 2
# Backward-compatible numeric alias.  This value is an observed-horizon
# comparison, never a direct public-record capacity producer.
DEFAULT_N_CRC = DEFAULT_OBSERVED_HORIZON_N_COMPARISON

_SOURCE_DERIVED_EW_BRANCH_VALUES = {
    "comparison_pixel": (N_CRC_EW_COMPARISON_PIXEL, N_CRC_EW_COMPARISON_PIXEL_PROVENANCE),
    "forward_closure_point": (
        N_CRC_EW_FORWARD_CLOSURE_POINT,
        N_CRC_EW_FORWARD_CLOSURE_POINT_PROVENANCE,
    ),
}


def source_derived_screen_capacity_constants(
    branch: str = SOURCE_DERIVED_N_CRC_EW_DEFAULT_BRANCH,
    *,
    p_value: float = P_STAR,
) -> "OPHScreenCapacityConstants":
    """Typed capacity constants on the source-derived electroweak branch.

    The value is the non-circular N_CRC_EW imported from the cross-repo gap
    ledger, branch-tagged as ``comparison_pixel`` or ``forward_closure_point``.
    This is a source-side replay constant, not the quarantined observed-horizon
    comparison and not a finite-simulator capacity fixed point.
    """

    if branch not in _SOURCE_DERIVED_EW_BRANCH_VALUES:
        raise ValueError(
            f"unknown source-derived N_CRC_EW branch: {branch!r}; "
            f"expected one of {sorted(_SOURCE_DERIVED_EW_BRANCH_VALUES)}"
        )
    value, _provenance = _SOURCE_DERIVED_EW_BRANCH_VALUES[branch]
    return OPHScreenCapacityConstants(
        n_crc=value,
        p_value=p_value,
        source=f"{SOURCE_DERIVED_N_CRC_EW_SOURCE}:{branch}",
    )


@dataclass(frozen=True)
class OPHScreenCapacityConstants:
    """Typed capacity comparison, separate from finite regulator patch count."""

    n_crc: float = DEFAULT_N_CRC
    p_value: float = P_STAR
    source: str = "observed_horizon_comparison"

    @property
    def n_patch_bare_ratio(self) -> float:
        return float(self.n_crc) / math.pi

    @property
    def lambda_l_planck2(self) -> float:
        return lambda_planck2_from_capacity(self.n_crc)

    @property
    def radius_planck(self) -> float:
        return math.sqrt(self.n_patch_bare_ratio)

    @property
    def physical_cell_count(self) -> float:
        return physical_cells_for_entropy_capacity(self.n_crc, self.p_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "N_CRC": float(self.n_crc),
            "P": float(self.p_value),
            "source": self.source,
            "N_patch_bare_radius_squared_ratio": self.n_patch_bare_ratio,
            "Lambda_lP2": self.lambda_l_planck2,
            "radius_planck": self.radius_planck,
            "N_cells_if_tiled_by_local_P_cells": self.physical_cell_count,
        }


def bare_horizon_area_ratio(radius_m: float = DEFAULT_R_DS_M, planck_length_m: float = DEFAULT_L_PLANCK_M) -> float:
    return float((float(radius_m) / float(planck_length_m)) ** 2)


def entropy_capacity_from_radius(radius_m: float = DEFAULT_R_DS_M, planck_length_m: float = DEFAULT_L_PLANCK_M) -> float:
    return math.pi * bare_horizon_area_ratio(radius_m, planck_length_m)


def lambda_planck2_from_capacity(n_crc: float) -> float:
    return 3.0 * math.pi / float(n_crc)


def physical_cells_for_entropy_capacity(n_scr: float, p_value: float = P_STAR) -> float:
    """Cells needed when each local cell carries entropy capacity P/4."""

    return 4.0 * float(n_scr) / float(p_value)


def screen_capacity_closure_report(
    *,
    p_value: float = P_STAR,
    n_crc: float | None = None,
    radius_m: float = DEFAULT_R_DS_M,
    planck_length_m: float = DEFAULT_L_PLANCK_M,
    regulator_patch_counts: tuple[int, ...] = DEFAULT_REGULATOR_PATCH_COUNTS,
) -> dict[str, Any]:
    if n_crc is None:
        input_mode = "observed_horizon_comparison"
        n_patch = bare_horizon_area_ratio(radius_m, planck_length_m)
        n_scr = entropy_capacity_from_radius(radius_m, planck_length_m)
        r_ds_m = float(radius_m)
    else:
        input_mode = "declared_N_comparison"
        n_scr = float(n_crc)
        n_patch = n_scr / math.pi
        r_ds_m = math.sqrt(n_patch) * float(planck_length_m)

    capacity = OPHScreenCapacityConstants(
        n_crc=n_scr,
        p_value=p_value,
        source=input_mode,
    )
    lambda_l_planck2 = lambda_planck2_from_capacity(n_scr)
    physical_cells = physical_cells_for_entropy_capacity(n_scr, p_value)
    return {
        "mode": "oph_screen_capacity_closure_v0",
        "contract_version": "pro5-public-record-capacity-v1",
        "source": "typed downstream horizon comparison plus exact finite public-record evaluator",
        "closure_equations": {
            "finite_terminal_capacity": "M_0(q)=alpha(G_q)",
            "set_valued_readback": "F_set,r,0(D)={M_0(q):q in complete terminal fiber Omega_tilde(r,D)}",
            "whole_fiber_scalarization": "N=log M_0(U_N) only when the complete nonempty fiber has one common value",
            "robust_closure": "F_set,r,0(D)={D}",
            "finite_size_slack": "s_r(D)=log(D)-log(M_0,r(D))",
            "unique_physical_selector": "one regulator-stable zero of s(D), not greatest-fixed-point order theory",
            "lambda_readout": "Lambda_CRC * l_P^2 = 3*pi / N_CRC",
            "dimensionless_lambda_readout": "Lambda_CRC * ell_star^2 = 3*pi / N_CRC",
        },
        "producer_firewall": {
            "observed_horizon_radius_may_define_capacity": False,
            "measured_Lambda_may_define_capacity": False,
            "electroweak_bridge_may_define_capacity": False,
            "operational_resolution_may_define_capacity": False,
            "supplied_capacity_label_may_enter_packet_predicate": False,
        },
        "observed_branch_normalization": {
            "input_mode": input_mode,
            "producer_eligible": False,
            "R_dS_m": r_ds_m,
            "planck_length_m": float(planck_length_m),
            "N_CRC": n_scr,
            "N_patch_bare_radius_squared_ratio": n_patch,
            "N_scr_entropy_capacity": n_scr,
            "Lambda_lP2": lambda_l_planck2,
            "Lambda_lP2_is_dimensionless": True,
            "dimensionful_Lambda_m2": None,
            "dimensionful_ell_star_squared_m2": None,
            "dimensionful_G_SI": None,
            "N_cells_if_tiled_by_local_P_cells": physical_cells,
            "cell_entropy_capacity": float(p_value) / 4.0,
            "P": float(p_value),
            "constants": capacity.as_dict(),
        },
        "active_capacity_requirements": {
            "capacity_variable": "D=dim(H_cap), M_0=multiplicative code size, N=log(M_0)",
            "record_atoms": "total local/interface atom readout maps and compatible public global sections",
            "endogenous_reachability": "nonempty semantic histories that do not read a target or supplied D label",
            "joint_checkpoints": "complete source-supplied global kernels; local marginals never determine the joint kernel",
            "decoder": "maximum independent set of the compound confusability graph",
            "whole_fiber": "complete nonempty terminal fiber before scalarization",
            "finite_regulator_status": "exact evaluator implemented; physical source packet and unique-slack law remain open",
        },
        "regulator_scale_comparison": [
            {
                "patch_count": int(count),
                "regulator_entropy_capacity": total_entropy_capacity(int(count), p_value),
                "fraction_of_observed_N_scr": total_entropy_capacity(int(count), p_value) / n_scr,
                "claim_boundary": "finite simulation regulator count, not the cosmic record capacity",
            }
            for count in regulator_patch_counts
        ],
        "readiness_gates": {
            "local_P_cell_capacity_available": True,
            "N_CRC_closure_value_declared": False,
            "observed_branch_N_scr_readout_available": True,
            "observed_branch_is_comparison_only": True,
            "finite_correctable_public_record_evaluator_implemented": True,
            "reference_reversible_12_port_packet_certified": True,
            "active_edge_center_predictive_quotient_implemented": False,
            "observer_supporting_terminal_sector_implemented": False,
            "capacity_readback_map_from_terminal_records_implemented": False,
            "F_N_readback_map_implemented": False,
            "source_derived_complete_terminal_fiber_implemented": False,
            "whole_fiber_scalar_readback_implemented": False,
            "robust_closure_F_set_equals_D_certified": False,
            "unique_regulator_stable_slack_zero_certified": False,
            "horizon_record_saturation_receipt": False,
            "N_CRC_fixed_point_solved_from_finite_simulator": False,
            "Lambda_from_finite_simulator_record_closure": False,
            "independent_scale_bridge_supplied": False,
            "dimensionful_G_SI_eligible": False,
            "finite_simulator_derived_G_SI": False,
        },
        "simulation_relevance": (
            "This closure is globally relevant to cosmology and capacity normalization. It should not be "
            "used to reinterpret ordinary finite run patch counts as physical horizon capacity. Current "
            "64k/256k/1M runs remain numerical regulators. The exact finite evaluator consumes "
            "PUBLIC_CHECKPOINT_PACKET evidence, but no source-derived complete physical packet/fiber or "
            "unique regulator-stable slack zero is currently emitted."
        ),
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Comparison-only horizon normalization plus Pro5 contract status. The observed radius or a "
            "declared N cannot produce public-record capacity. Physical N requires a source-derived complete "
            "packet/fiber, common M_0=alpha(G_q), and a unique regulator-stable slack zero; the horizon and "
            "electroweak identifications are independent downstream receipts."
        ),
    }


def write_screen_capacity_closure_report(out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    report = screen_capacity_closure_report(**kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen_capacity_closure_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "screen_capacity_closure_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def capacity_readback_proxy_report(
    run_dirs: list[Path],
    *,
    p_value: float = P_STAR,
    n_crc: float = DEFAULT_N_CRC,
    max_observer_views: int = 4096,
) -> dict[str, Any]:
    """Summarize legacy finite-run counts beside the exact M_0 evaluator.

    This is deliberately diagnostic. It reads finite regulator outputs and
    estimates counts only.  Counts are not a substitute for a joint checkpoint
    kernel, compound confusability graph, or maximum-independent-set decoder.
    """

    candidates = _capacity_candidate_dirs([Path(path) for path in run_dirs])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = _capacity_proxy_row(
            candidate,
            p_value=p_value,
            n_crc=n_crc,
            max_observer_views=max_observer_views,
        )
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("path", "")))
    gates = {
        "capacity_readback_proxy_written": True,
        "finite_regulator_rows_present": bool(rows),
        "observer_supporting_proxy_present": any(
            bool(row.get("observer_supporting_proxy_present", False)) for row in rows
        ),
        "terminal_sector_proxy_present": any(
            bool(row.get("terminal_sector_proxy_present", False)) for row in rows
        ),
        "active_edge_center_predictive_quotient_implemented": False,
        "observer_supporting_terminal_sector_implemented": False,
        "terminal_normal_form_enumerator_implemented": False,
        "capacity_readback_map_from_terminal_records_implemented": False,
        "F_N_readback_map_implemented": False,
        "finite_correctable_public_record_evaluator_implemented": True,
        "source_derived_complete_terminal_fiber_implemented": False,
        "unique_regulator_stable_slack_zero_certified": False,
        "N_CRC_fixed_point_solved_from_finite_simulator": False,
    }
    return {
        "mode": "oph_capacity_readback_proxy_v0",
        "run_dirs": [str(path) for path in run_dirs],
        "N_CRC": float(n_crc),
        "P": float(p_value),
        "row_count": len(rows),
        "max_observer_count": max((int(row.get("observer_count") or 0) for row in rows), default=0),
        "max_active_record_signature_count": max(
            (int(row.get("active_record_signature_count") or 0) for row in rows),
            default=0,
        ),
        "max_terminal_normal_form_count_proxy": max(
            (int(row.get("terminal_normal_form_count_proxy") or 0) for row in rows),
            default=0,
        ),
        "rows": rows,
        "readiness_gates": gates,
        "physical_cmb_prediction": False,
        "strict_neutral_bulk": False,
        "claim_boundary": (
            "Finite capacity-readback proxy only. Rows summarize emitted finite-regulator observer/object "
            "support and count proxies. They do not provide PUBLIC_CHECKPOINT_PACKET joint kernels, "
            "M_0(q)=alpha(G_q), whole-fiber scalarization, or the unique regulator-stable slack zero."
        ),
    }


def write_capacity_readback_proxy_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    p_value: float = P_STAR,
    n_crc: float = DEFAULT_N_CRC,
    max_observer_views: int = 4096,
) -> dict[str, Any]:
    report = capacity_readback_proxy_report(
        run_dirs,
        p_value=p_value,
        n_crc=n_crc,
        max_observer_views=max_observer_views,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capacity_readback_proxy_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "capacity_readback_proxy_report.md").write_text(
        _markdown_capacity_proxy_report(report),
        encoding="utf-8",
    )
    _write_capacity_proxy_rows(out / "capacity_readback_proxy_rows.csv", report["rows"])
    return report


def _capacity_candidate_dirs(roots: list[Path]) -> list[Path]:
    marker_names = (
        "manifest.json",
        "observer_views.jsonl",
        "observer_chart_object_h3_report.json",
        "observer_chart_object_h3_lineage_report.json",
        "bulk_proof_certificate_report.json",
        "neutral_3d_bulk_audit_report.json",
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidate = root.parent
            resolved = candidate.resolve()
            if resolved not in seen:
                candidates.append(candidate)
                seen.add(resolved)
            continue
        if not root.exists() or not root.is_dir():
            continue
        root_has_marker = any((root / name).exists() for name in marker_names)
        if root_has_marker:
            resolved = root.resolve()
            if resolved not in seen:
                candidates.append(root)
                seen.add(resolved)
        for name in marker_names:
            for marker in sorted(root.glob(f"**/{name}")):
                candidate = marker.parent
                resolved = candidate.resolve()
                if resolved not in seen:
                    candidates.append(candidate)
                    seen.add(resolved)
    return candidates


def _capacity_proxy_row(
    run_dir: Path,
    *,
    p_value: float,
    n_crc: float,
    max_observer_views: int,
) -> dict[str, Any] | None:
    manifest = _read_json(run_dir / "manifest.json")
    h3 = _first_existing_json(
        run_dir,
        "observer_chart_object_h3_lineage_report.json",
        "observer_chart_object_h3_report.json",
        "observer_chart_object_h3_scale_compressed_report.json",
    )
    bulk = _read_json(run_dir / "bulk_proof_certificate_report.json")
    neutral = _read_json(run_dir / "neutral_3d_bulk_audit_report.json")
    observer_views = _observer_view_proxy_counts(run_dir / "observer_views.jsonl", max_observer_views)
    if not any([manifest, h3, bulk, neutral, observer_views["observer_count"]]):
        return None
    patch_count = _patch_count_from_sources(run_dir, manifest, h3, neutral)
    object_count = _int_first(
        h3.get("object_count"),
        h3.get("localized_object_count"),
        h3.get("localized_not_boundary_object_count"),
        ((h3.get("h3_preview") or {}).get("object_count") if isinstance(h3.get("h3_preview"), dict) else None),
    )
    localized_object_count = _int_first(
        h3.get("localized_not_boundary_object_count"),
        h3.get("localized_object_count"),
        h3.get("object_count"),
    )
    observer_count = _int_first(
        observer_views.get("observer_count"),
        manifest.get("observer_count"),
        manifest.get("n_observers"),
        manifest.get("observer_views"),
        object_count,
    )
    active_signature_count = int(observer_views.get("active_record_signature_count") or 0)
    terminal_proxy = max(
        int(active_signature_count),
        int(object_count or 0),
        int(localized_object_count or 0),
        int(observer_count or 0),
    )
    if terminal_proxy <= 0 and patch_count is None:
        return None
    regulator_capacity = total_entropy_capacity(int(patch_count), p_value) if patch_count else None
    log_minus_capacity = (
        math.log(max(terminal_proxy, 1)) - float(regulator_capacity)
        if regulator_capacity is not None
        else None
    )
    source_files = [
        name
        for name in (
            "manifest.json",
            "observer_views.jsonl",
            "observer_chart_object_h3_lineage_report.json",
            "observer_chart_object_h3_report.json",
            "bulk_proof_certificate_report.json",
            "neutral_3d_bulk_audit_report.json",
        )
        if (run_dir / name).exists()
    ]
    return {
        "path": str(run_dir),
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "patch_count": patch_count,
        "observer_count": observer_count,
        "active_record_signature_count": active_signature_count,
        "terminal_normal_form_count_proxy": terminal_proxy,
        "object_count": object_count,
        "localized_object_count": localized_object_count,
        "regulator_entropy_capacity": regulator_capacity,
        "fraction_of_N_CRC": (float(regulator_capacity) / float(n_crc)) if regulator_capacity is not None else None,
        "log_terminal_proxy_count_minus_regulator_capacity": log_minus_capacity,
        "observer_supporting_proxy_present": bool(observer_count and observer_count > 0),
        "terminal_sector_proxy_present": bool(terminal_proxy > 0),
        "theorem_assisted_h3_bulk": bool(
            h3.get("observer_chart_object_h3_receipt", False)
            or bulk.get("bulk_3d_established_theorem_assisted", False)
        ),
        "strict_neutral_bulk": bool(
            neutral.get("strict_neutral_bulk_ready", False)
            or bulk.get("strict_neutral_third_person_bulk_established", False)
        ),
        "source_files": source_files,
        "claim_boundary": "finite regulator diagnostic row; not a cosmic capacity fixed-point solution",
    }


def _observer_view_proxy_counts(path: Path, max_rows: int) -> dict[str, int]:
    if not path.exists():
        return {"observer_count": 0, "active_record_signature_count": 0}
    signatures: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if count >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            count += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                signatures.add(text[:256])
                continue
            if not isinstance(row, dict):
                signatures.add(str(row)[:256])
                continue
            signature = _observer_view_signature(row)
            signatures.add(signature)
    return {"observer_count": count, "active_record_signature_count": len(signatures)}


def _observer_view_signature(row: dict[str, Any]) -> str:
    for key in (
        "record_signature",
        "record_hash",
        "object_signature",
        "transition_signature",
        "terminal_signature",
        "object_id",
    ):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    selected = {
        key: row.get(key)
        for key in (
            "records",
            "record_histogram",
            "record_signature_histogram",
            "object_packet_histogram",
            "transition_history_histograms",
            "observer_packet",
        )
        if key in row
    }
    if not selected:
        selected = {key: row.get(key) for key in sorted(row)[:8]}
    return json.dumps(selected, sort_keys=True, default=str)[:512]


def _patch_count_from_sources(
    run_dir: Path,
    manifest: dict[str, Any],
    h3: dict[str, Any],
    neutral: dict[str, Any],
) -> int | None:
    direct = _int_first(
        manifest.get("patch_count"),
        manifest.get("n_patches"),
        ((manifest.get("config") or {}).get("patch_count") if isinstance(manifest.get("config"), dict) else None),
        h3.get("patch_count"),
        neutral.get("patch_count"),
    )
    if direct:
        return direct
    k_match = re.search(r"(?<!\d)(\d+)[kK](?:_|-|$)", run_dir.name)
    if k_match:
        return int(k_match.group(1)) * 1024
    patch_match = re.search(r"(?<!\d)(\d{3,9})(?:_?patch(?:es)?|_?patch_count)(?![a-zA-Z])", run_dir.name)
    if patch_match:
        return int(patch_match.group(1))
    return None


def _first_existing_json(run_dir: Path, *names: str) -> dict[str, Any]:
    for name in names:
        data = _read_json(run_dir / name)
        if data:
            return data
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _int_first(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _write_capacity_proxy_rows(target: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "path",
        "run_id",
        "patch_count",
        "observer_count",
        "active_record_signature_count",
        "terminal_normal_form_count_proxy",
        "object_count",
        "localized_object_count",
        "regulator_entropy_capacity",
        "fraction_of_N_CRC",
        "log_terminal_proxy_count_minus_regulator_capacity",
        "observer_supporting_proxy_present",
        "terminal_sector_proxy_present",
        "theorem_assisted_h3_bulk",
        "strict_neutral_bulk",
        "source_files",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _markdown_report(report: dict[str, Any]) -> str:
    observed = report["observed_branch_normalization"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH Screen-Capacity Closure",
            "",
            str(report["claim_boundary"]),
            "",
            "## Observed Branch",
            "",
            f"- input mode: `{observed['input_mode']}`",
            f"- N_CRC: `{observed['N_CRC']:.6e}`",
            f"- N_patch bare ratio: `{observed['N_patch_bare_radius_squared_ratio']:.6e}`",
            f"- N_scr entropy capacity: `{observed['N_scr_entropy_capacity']:.6e}`",
            f"- Lambda l_P^2: `{observed['Lambda_lP2']:.6e}`",
            f"- local P-cell count for N_scr: `{observed['N_cells_if_tiled_by_local_P_cells']:.6e}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
            "## Simulation Relevance",
            "",
            str(report["simulation_relevance"]),
            "",
        ]
    )


def _markdown_capacity_proxy_report(report: dict[str, Any]) -> str:
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH Capacity Readback Proxy",
            "",
            str(report["claim_boundary"]),
            "",
            "## Summary",
            "",
            f"- row count: `{report['row_count']}`",
            f"- max observer count: `{report['max_observer_count']}`",
            f"- max active record signatures: `{report['max_active_record_signature_count']}`",
            f"- max terminal normal-form proxy count: `{report['max_terminal_normal_form_count_proxy']}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
        ]
    )
