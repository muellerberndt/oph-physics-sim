from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from oph_fpe.claims import DEMO, with_claim_metadata


def cap_cross_ratio_report(boundary_points: np.ndarray, quadruples: Iterable[tuple[int, int, int, int]]) -> dict[str, Any]:
    points = np.asarray(boundary_points, dtype=float)
    rows: list[dict[str, Any]] = []
    for quad in quadruples:
        i, j, k, l = [int(value) for value in quad]
        ratio = _cross_ratio(points[i], points[j], points[k], points[l])
        rows.append({"quadruple": [i, j, k, l], "cross_ratio": ratio})
    finite = all(np.isfinite(float(row["cross_ratio"])) for row in rows)
    report = {
        "mode": "finite_boundary_cap_cross_ratios",
        "CONFORMAL_H3_CHART_RECEIPT": bool(rows) and finite,
        "receipt": bool(rows) and finite,
        "point_count": int(points.shape[0]),
        "quadruple_count": len(rows),
        "rows": rows,
        "claim_boundary": (
            "finite S2 boundary cross-ratio diagnostic for conformal/H3 chart consistency; "
            "not a populated 3D bulk reconstruction receipt"
        ),
    }
    return with_claim_metadata(report, claim_level=DEMO, receipt="CONFORMAL_H3_CHART_RECEIPT")


def _cross_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    ab = np.linalg.norm(a - b)
    cd = np.linalg.norm(c - d)
    ac = np.linalg.norm(a - c)
    bd = np.linalg.norm(b - d)
    return float((ab * cd) / (ac * bd + 1e-12))
