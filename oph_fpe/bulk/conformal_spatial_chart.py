from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.cap_normals import cap_gram_matrix, cap_normal_report
from oph_fpe.bulk.h3_chart import h3_chart_report


def conformal_h3_spatial_chart_report(caps: list[RoundCap]) -> dict[str, Any]:
    normal_report = cap_normal_report(caps)
    chart_report = h3_chart_report(caps)
    gram = cap_gram_matrix(caps)
    finite_gram = gram[np.isfinite(gram)] if gram.size else np.zeros(0, dtype=float)
    receipt = bool(
        normal_report.get("unit_normal_receipt", False)
        and chart_report.get("conformal_h3_spatial_chart_receipt", False)
    )
    return {
        "mode": "conformal_lorentz_to_h3_spatial_chart",
        "cap_normal_report": normal_report,
        "h3_chart_report": chart_report,
        "cross_ratio_proxy": {
            "kind": "minkowski_cap_normal_gram",
            "entry_count": int(finite_gram.size),
            "min": float(np.min(finite_gram)) if finite_gram.size else None,
            "median": float(np.median(finite_gram)) if finite_gram.size else None,
            "max": float(np.max(finite_gram)) if finite_gram.size else None,
        },
        "conformal_h3_spatial_chart_receipt": receipt,
        "record_populated_h3_receipt": False,
        "defect_localized_in_h3_receipt": False,
        "claim_boundary": (
            "constructs the canonical 3D H3 spatial chart from the S2 cap/conformal Lorentz branch. "
            "This replaces fractional observer-similarity dimensions as the chart receipt, but it is "
            "not yet a populated spatial-bulk, particle, or CMB prediction receipt."
        ),
    }

