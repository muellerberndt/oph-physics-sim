from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.constants.oph_pixel import P_STAR


SCALAR_READOUT_FIELDS = (
    "committed_fraction",
    "record_stability_mean",
    "repair_load_mean",
    "mismatch_density_mean",
    "visible_signature_entropy",
    "counterfactual_stability",
)

FORBIDDEN_PRIMARY_FIELDS = (
    "axis",
    "support_nodes",
    "h3_point",
    "cap_axis",
    "cap_normal",
    "radial_depth",
    "modular_depth",
)


def scalar_quotient_report(
    run_dir: Path,
    *,
    target_ell_ir: int = 32,
    bins: int = 8,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    """Build a scalar/geometric quotient certificate from observer-visible records.

    This is the finite-screen lane the CMB notes ask for before any physical
    CMB promotion. It deliberately constructs scalar packets from local
    observer readout numbers only. Screen axes are used later only to remove
    monopole/dipole from the scalar screen field; they are not used to define
    the packets.
    """

    root = Path(run_dir)
    observer_path = root / "observer_views.jsonl"
    if not observer_path.exists():
        raise FileNotFoundError(observer_path)
    manifest = _read_json(root / "manifest.json")
    observer_views = [row for row in _read_jsonl(observer_path) if row.get("view_type") == "patch_observer"]
    if not observer_views:
        raise ValueError(f"{observer_path} contains no patch_observer rows")

    values = _scalar_readout_matrix(observer_views)
    packets = _scalar_packet_tokens(values, bins=int(bins))
    packet_counts = _packet_counts(packets)
    scalar_field = _scalar_field(values)
    center_removed = scalar_field - float(np.mean(scalar_field))
    axes = _unit_axes(observer_views)
    dipole = _remove_monopole_dipole(scalar_field, axes)

    target_level_count = int(target_ell_ir) + 1
    observer_level_proxy = int(math.floor(math.sqrt(len(observer_views))))
    patch_count = _manifest_patch_count(manifest)
    patch_capacity_level_proxy = int(math.floor(math.sqrt(patch_count))) if patch_count else None
    packet_level_proxy = int(math.floor(math.sqrt(max(1, len(packet_counts)))))
    theta_oph = float(p_value) / 48.0
    n_s = 1.0 - theta_oph
    scalar_entropy = _entropy_bits(np.asarray(list(packet_counts.values()), dtype=float))
    no_forbidden_primary = _primary_does_not_depend_on_forbidden(observer_views)
    basic_receipt = bool(
        no_forbidden_primary
        and len(packet_counts) > 0
        and np.isfinite(float(np.var(center_removed)))
        and dipole["center_free_scalar_variance"] is not None
    )
    active_33_level_clause = bool(observer_level_proxy >= target_level_count)
    support_capacity_clause = bool(
        patch_capacity_level_proxy is not None and patch_capacity_level_proxy >= target_level_count
    )
    theorem_grade = False
    finite_ready = bool(basic_receipt and active_33_level_clause and theorem_grade)
    report = {
        "mode": "oph_scalar_geometric_quotient_report_v0",
        "source_run_dir": str(root),
        "observer_views_path": str(observer_path),
        "source_hashes": {
            "observer_views.jsonl": _sha256_file(observer_path),
            "manifest.json": _sha256_file(root / "manifest.json") if (root / "manifest.json").exists() else None,
        },
        "patch_count": patch_count,
        "observer_count": len(observer_views),
        "scalar_readout_fields": list(SCALAR_READOUT_FIELDS),
        "forbidden_primary_fields": list(FORBIDDEN_PRIMARY_FIELDS),
        "primary_uses_forbidden_geometry": not no_forbidden_primary,
        "scalar_packet_alphabet_size": len(packet_counts),
        "scalar_packet_entropy_bits": scalar_entropy,
        "scalar_packet_rows": [
            {"packet": token, "count": int(count), "probability": float(count / len(packets))}
            for token, count in sorted(packet_counts.items(), key=lambda item: (-item[1], item[0]))[:256]
        ],
        "scalar_field_statistics": {
            "mean": float(np.mean(scalar_field)),
            "std": float(np.std(scalar_field)),
            "variance": float(np.var(scalar_field)),
            "centered_variance": float(np.var(center_removed)),
            **dipole,
        },
        "active_angular_levels": {
            "target_ell_IR": int(target_ell_ir),
            "target_level_count_including_monopole": target_level_count,
            "observer_level_proxy_floor_sqrt_observers": observer_level_proxy,
            "patch_capacity_level_proxy_floor_sqrt_patches": patch_capacity_level_proxy,
            "packet_level_proxy_floor_sqrt_alphabet": packet_level_proxy,
            "observer_sampling_supports_33_level_clause": active_33_level_clause,
            "patch_capacity_supports_33_level_clause": support_capacity_clause,
            "active_33_level_freezeout_clause": active_33_level_clause,
            "claim_boundary": (
                "The observer-level proxy is the active finite scalar release support in this report. "
                "Patch capacity may be large enough while the active observer release subspace still "
                "fails the 33-level freezeout clause."
            ),
        },
        "edge_center_readout": {
            "P": float(p_value),
            "theta_OPH_P_over_48": theta_oph,
            "n_s_P_over_48": n_s,
            "source": "OPH P fixed-point branch; theorem-side constant, not fitted to CMB",
        },
        "readiness_gates": {
            "scalar_packets_emitted": bool(len(packet_counts) > 0),
            "center_free_scalar_field_emitted": bool(dipole["center_free_scalar_variance"] is not None),
            "primary_no_forbidden_geometry": no_forbidden_primary,
            "active_33_level_freezeout_clause": active_33_level_clause,
            "patch_capacity_33_level_support": support_capacity_clause,
            "theorem_grade_scalar_release_code": theorem_grade,
            "finite_lattice_cmb_scalar_release_ready": finite_ready,
        },
        "SCALAR_QUOTIENT_RECEIPT": basic_receipt,
        "finite_lattice_cmb_scalar_release_ready": finite_ready,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite scalar/geometric quotient report from observer-visible readouts. It is useful CMB "
            "input instrumentation, but it is not a physical CMB prediction and not a strict 3D bulk "
            "proof. The physical CMB gate stays closed until the theorem-grade scalar release code, "
            "33-level freezeout clause, parent-collar response, repair matrix, and likelihood gates pass."
        ),
    }
    report["blockers"] = _blockers(report)
    return report


def write_scalar_quotient_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    target_ell_ir: int = 32,
    bins: int = 8,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    report = scalar_quotient_report(run_dir, target_ell_ir=target_ell_ir, bins=bins, p_value=p_value)
    destination = Path(out) if out is not None else Path(run_dir) / "scalar_quotient_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_packet_csv(destination.with_name("scalar_quotient_packets.csv"), report["scalar_packet_rows"])
    destination.with_name("scalar_quotient_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _scalar_readout_matrix(observer_views: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for view in observer_views:
        rows.append([_float(view.get(key), 0.0) for key in SCALAR_READOUT_FIELDS])
    matrix = np.asarray(rows, dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return matrix


def _scalar_packet_tokens(values: np.ndarray, *, bins: int) -> list[str]:
    if values.ndim != 2 or values.shape[0] == 0:
        return []
    binned_columns = []
    for column in range(values.shape[1]):
        data = values[:, column]
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi <= lo + 1e-12:
            binned = np.zeros(data.shape[0], dtype=int)
        else:
            scaled = (data - lo) / (hi - lo)
            binned = np.clip(np.floor(scaled * int(bins)), 0, int(bins) - 1).astype(int)
        binned_columns.append(binned)
    binned_matrix = np.vstack(binned_columns).T
    return [".".join(str(int(value)) for value in row) for row in binned_matrix]


def _packet_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _scalar_field(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    z = (values - mean) / np.maximum(std, 1.0e-12)
    signs = np.asarray([1.0, 1.0, -0.5, -0.5, 0.25, 0.75], dtype=float)
    return z @ signs


def _unit_axes(observer_views: list[dict[str, Any]]) -> np.ndarray | None:
    axes = []
    for view in observer_views:
        axis = np.asarray(view.get("axis", []), dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            return None
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            return None
        axes.append(axis / norm)
    return np.vstack(axes) if axes else None


def _remove_monopole_dipole(field: np.ndarray, axes: np.ndarray | None) -> dict[str, Any]:
    if axes is None or axes.shape[0] != field.size or field.size < 4:
        centered = field - float(np.mean(field)) if field.size else field
        return {
            "monopole_dipole_removed": False,
            "monopole": float(np.mean(field)) if field.size else None,
            "dipole_vector": None,
            "center_free_scalar_variance": float(np.var(centered)) if field.size else None,
            "monopole_dipole_residual_variance": None,
            "dipole_removed_fraction": None,
            "reason": "missing_valid_screen_axes_or_too_few_observers",
        }
    design = np.column_stack([np.ones(field.size), axes])
    coeff, *_ = np.linalg.lstsq(design, field, rcond=None)
    fitted = design @ coeff
    residual = field - fitted
    centered = field - float(np.mean(field))
    centered_variance = float(np.var(centered))
    residual_variance = float(np.var(residual))
    removed = 1.0 - residual_variance / max(centered_variance, 1.0e-12)
    return {
        "monopole_dipole_removed": True,
        "monopole": float(coeff[0]),
        "dipole_vector": [float(value) for value in coeff[1:4]],
        "center_free_scalar_variance": centered_variance,
        "monopole_dipole_residual_variance": residual_variance,
        "dipole_removed_fraction": float(np.clip(removed, -1.0e6, 1.0)),
    }


def _entropy_bits(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def _primary_does_not_depend_on_forbidden(observer_views: list[dict[str, Any]]) -> bool:
    # The implementation path above reads only SCALAR_READOUT_FIELDS for packet
    # construction. This runtime check guards against accidental field overlap.
    return bool(not set(SCALAR_READOUT_FIELDS).intersection(FORBIDDEN_PRIMARY_FIELDS))


def _manifest_patch_count(manifest: dict[str, Any]) -> int | None:
    for key in ("patch_count", "screen_patch_count"):
        if key in manifest:
            try:
                value = int(manifest[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def _blockers(report: dict[str, Any]) -> list[str]:
    gates = report.get("readiness_gates", {})
    blockers = []
    if not gates.get("scalar_packets_emitted", False):
        blockers.append("scalar_packets_not_emitted")
    if not gates.get("center_free_scalar_field_emitted", False):
        blockers.append("center_free_scalar_field_missing")
    if not gates.get("primary_no_forbidden_geometry", False):
        blockers.append("scalar_primary_uses_forbidden_geometry")
    if not gates.get("active_33_level_freezeout_clause", False):
        blockers.append("active_33_level_freezeout_clause_not_established")
    if not gates.get("theorem_grade_scalar_release_code", False):
        blockers.append("theorem_grade_scalar_release_code_missing")
    return blockers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _write_packet_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["packet", "count", "probability"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _markdown_report(report: dict[str, Any]) -> str:
    levels = report["active_angular_levels"]
    gates = report["readiness_gates"]
    blockers = report.get("blockers", [])
    lines = [
        "# OPH Scalar/Geometric Quotient Report",
        "",
        report["claim_boundary"],
        "",
        f"- scalar quotient receipt: `{report['SCALAR_QUOTIENT_RECEIPT']}`",
        f"- finite lattice CMB scalar release ready: `{report['finite_lattice_cmb_scalar_release_ready']}`",
        f"- physical CMB prediction: `{report['physical_cmb_prediction']}`",
        f"- observers: `{report['observer_count']}`",
        f"- patches: `{report['patch_count']}`",
        f"- scalar packet alphabet: `{report['scalar_packet_alphabet_size']}`",
        f"- packet entropy bits: `{report['scalar_packet_entropy_bits']}`",
        f"- target ell_IR: `{levels['target_ell_IR']}`",
        f"- target levels including monopole: `{levels['target_level_count_including_monopole']}`",
        f"- observer level proxy: `{levels['observer_level_proxy_floor_sqrt_observers']}`",
        f"- patch capacity level proxy: `{levels['patch_capacity_level_proxy_floor_sqrt_patches']}`",
        f"- n_s from P/48: `{report['edge_center_readout']['n_s_P_over_48']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
