from __future__ import annotations

from .receipts import fail, pass_report


def refinement_compatibility(total_variation_defect: float, *, max_defect: float) -> dict:
    if total_variation_defect > max_defect:
        return fail(
            "REFINEMENT_DEFECT_TOO_LARGE",
            details={"total_variation_defect": total_variation_defect, "max_defect": max_defect},
        )
    return pass_report(
        receipts={"REFINEMENT_COMPATIBILITY": True, "REFINEMENT_STABILITY": True},
        details={"total_variation_defect": total_variation_defect, "max_defect": max_defect},
    )
