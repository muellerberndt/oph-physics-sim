from __future__ import annotations

from typing import Any

from oph_fpe.claims import PROXY, with_claim_metadata
from oph_fpe.cosmology.data_targets import target_registry
from oph_fpe.cosmology.likelihood_proxy import control_separation_score, shape_only_spectrum_proxy


def cosmo_proxy_receipt(
    cl_report: dict[str, Any],
    *,
    target_spectrum: list[dict[str, float]] | None = None,
    target_label: str | None = None,
) -> dict[str, Any]:
    fields = cl_report.get("fields", {})
    rows: dict[str, Any] = {}
    for name, field_report in fields.items():
        score = control_separation_score(field_report)
        target_comparison = (
            shape_only_spectrum_proxy(field_report.get("spectrum", []), target_spectrum)
            if target_spectrum is not None
            else {"usable": False, "reason": "target_spectrum_not_provided"}
        )
        rows[name] = {
            "control_separation_score": score,
            "peak_ell": field_report.get("peak_ell"),
            "low_ell_power_2_10": field_report.get("low_ell_power_2_10"),
            "target_shape_comparison": target_comparison,
        }
    best_field = _best_field(rows, target_spectrum is not None)
    report = {
        "mode": "OPH_COSMO_PROXY_V0",
        "OPH_COSMO_PROXY_V0": bool(rows),
        "receipt": bool(rows),
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "boltzmann_adapter": False,
        "target_label": target_label,
        "target_registry": target_registry(),
        "best_field": best_field,
        "field_scores": rows,
        "simulator": {
            "ell_max": cl_report.get("ell_max"),
            "estimator": cl_report.get("estimator"),
            "point_count": cl_report.get("point_count"),
            "gate_allowed": bool(cl_report.get("gate_report", {}).get("allowed", False)),
        },
        "claim_boundary": (
            "screen-level freezeout C_l proxy and shape diagnostic. It is measurement-facing, "
            "but not a physical CMB prediction, not a P(k), not CAMB/CLASS input, and not a "
            "populated 3D bulk claim"
        ),
    }
    return with_claim_metadata(report, claim_level=PROXY, receipt="OPH_COSMO_PROXY_V0", physical_claim=False)


def _best_field(rows: dict[str, Any], has_target: bool) -> str | None:
    if not rows:
        return None
    if has_target:
        usable = {
            name: row["target_shape_comparison"]["normalized_rmse"]
            for name, row in rows.items()
            if row["target_shape_comparison"].get("usable")
        }
        if usable:
            return min(usable, key=usable.get)
    return max(rows, key=lambda name: float(rows[name]["control_separation_score"]))
