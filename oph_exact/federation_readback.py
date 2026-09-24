"""Readback of a settled federation: the per-carrier slow-band record field of the terminal states.

The carrier's readback is the rank-three slow-band projection of its twelve loads
(``carrier.slow_band_projector``; the flagship's ``G = 4 P_slow``).  Applied to a settled
terminal state this gives one three-dimensional record per carrier: the part of the
public configuration that the carrier's own repair mean preserves.  This lane reports, for
the initial loads and every terminal state of a run directory:

* the slow-band energy share ``sum |P_slow x_c|^2 / sum |x_c - mean_c|^2`` over carriers,
  against the isolated-carrier value 3/11 for an isotropic load;
* the distribution of the per-carrier slow-band norm (mean, standard deviation, fraction of
  carriers with zero slow component);
* the retention of the slow-band field by the settlement: the correlation and slope of the
  terminal per-carrier slow-band vectors on the initial ones (componentwise), and the same for
  the fast (non-slow, non-constant) components;
* the coarse retention of the slow-band norm by scale, with the same cap binning as the
  texture readout.

Readings of the declared law on the declared gluing; nothing is identified physically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from oph_exact import carrier
from oph_exact import federation_huge as H
from oph_exact.federation_texture import retention

SCHEMA = "oph.exact.federation-readback.v1"


def _sig(x: float, digits: int = 6) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


def fields(x: np.ndarray, n: int, p_slow: np.ndarray, p_fast: np.ndarray) -> dict[str, np.ndarray]:
    blocks = x.reshape(n, 12).astype(np.float64)
    centred = blocks - blocks.mean(axis=1, keepdims=True)
    slow = centred @ p_slow  # (n, 12), rank three per carrier
    fast = centred @ p_fast
    return {"centred": centred, "slow": slow, "fast": fast}


def band_projectors() -> dict[str, np.ndarray]:
    """Spectral projectors of the carrier seam Laplacian: constant (1), slow 5 - sqrt5 (3), middle 6 (5), top 5 + sqrt5 (3)."""

    seams = np.asarray(carrier.seams(), dtype=np.int64)
    lap = np.zeros((12, 12))
    for a, b in seams:
        lap[a, a] += 1; lap[b, b] += 1; lap[a, b] -= 1; lap[b, a] -= 1
    w, u = np.linalg.eigh(lap)
    out = {}
    for name, value in (("constant", 0.0), ("slow_3", 5 - 5**0.5), ("middle_5", 6.0), ("top_3", 5 + 5**0.5)):
        band = np.isclose(w, value)
        out[name] = u[:, band] @ u[:, band].T
    assert [int(np.round(np.trace(v))) for v in out.values()] == [1, 3, 5, 3]
    return out


def summarise(f: dict[str, np.ndarray]) -> dict[str, Any]:
    e_c = float(np.sum(f["centred"] ** 2))
    e_s = float(np.sum(f["slow"] ** 2))
    e_f = float(np.sum(f["fast"] ** 2))
    norms = np.sqrt(np.sum(f["slow"] ** 2, axis=1))
    bands = band_projectors()
    band_shares = {name: _sig(float(np.sum((f["centred"] @ proj) ** 2)) / e_c if e_c else 0.0) for name, proj in bands.items() if name != "constant"}
    return {"slow_band_energy_share": _sig(e_s / e_c if e_c else 0.0), "fast_band_energy_share": _sig(e_f / e_c if e_c else 0.0),
            "band_energy_shares": band_shares, "band_dimensions": {"constant": 1, "slow_3": 3, "middle_5": 5, "top_3": 3},
            "isotropic_band_shares": {"slow_3": _sig(3 / 11), "middle_5": _sig(5 / 11), "top_3": _sig(3 / 11)},
            "isotropic_reference_share": _sig(3.0 / 11.0),
            "slow_norm_mean": _sig(float(norms.mean())), "slow_norm_sd": _sig(float(norms.std())),
            "slow_norm_zero_fraction": _sig(float(np.mean(norms < 1e-12))), "centred_energy_per_carrier": _sig(e_c / f["centred"].shape[0])}


def build(run: Path, cache: Path, *, terminals: int = 4) -> dict[str, Any]:
    receipt = json.loads((run / "receipt.json").read_text())
    level, n, ports = int(receipt["level"]), int(receipt["carriers"]), int(receipt["ports"])
    points = np.load(Path(cache) / f"L{level}" / "cell_points.npy")
    p_slow = carrier.slow_band_projector()
    p_fast = np.eye(12) - p_slow - np.ones((12, 12)) / 12.0
    loads = np.random.default_rng(20260909 + level).integers(0, 6, size=ports).astype(np.int64)
    f0 = fields(loads, n, p_slow, p_fast)
    out = {"schema": SCHEMA, "level": level, "carriers": n, "receipt_sha256": hashlib.sha256((run / "receipt.json").read_bytes()).hexdigest(),
           "definition": "per carrier c: centred loads x_c - mean; slow = P_slow (x_c - mean), fast = (I - P_slow - J/12)(x_c - mean); P_slow the 5 - sqrt5 band projector of the carrier seam Laplacian",
           "initial": summarise(f0), "terminal": [], "used_seeds": []}
    slow0_norm = np.sqrt(np.sum(f0["slow"] ** 2, axis=1))
    for e in receipt["integer_law"]["entries"][:terminals]:
        x = np.load(run / f"integer_{e['seed']}" / e["terminal_state"]["path"]).astype(np.int64)
        if hashlib.sha256(x.astype("<i1").tobytes()).hexdigest() != e["terminal_state"]["sha256"]:
            raise ValueError(f"terminal state of seed {e['seed']} does not match its digest")
        ft = fields(x, n, p_slow, p_fast)
        s = summarise(ft)
        # retention of the slow and fast fields componentwise
        a, b = f0["slow"].ravel(), ft["slow"].ravel()
        s["slow_field_slope_on_initial"] = _sig(float(np.dot(a, b) / np.dot(a, a)))
        s["slow_field_correlation_with_initial"] = _sig(float(np.corrcoef(a, b)[0, 1]))
        a, b = f0["fast"].ravel(), ft["fast"].ravel()
        s["fast_field_slope_on_initial"] = _sig(float(np.dot(a, b) / np.dot(a, a)))
        s["fast_field_correlation_with_initial"] = _sig(float(np.corrcoef(a, b)[0, 1]))
        slow_norm = np.sqrt(np.sum(ft["slow"] ** 2, axis=1))
        rows = retention(points, slow0_norm, [slow_norm], (3, 6, 12, 24, 48, 96))
        s["slow_norm_retention_by_scale"] = [{"cells_per_cap": r["cells_per_cap"], "slope": r["slope_median"], "correlation": r["correlation_median"]} for r in rows]
        # histogram of the per-carrier slow norm squared times 4 (integer-valued for integer loads times 11? keep float bins)
        hist, edges = np.histogram(slow_norm, bins=12, range=(0.0, float(max(slow_norm.max(), 1e-9))))
        s["slow_norm_histogram"] = {"edges": [_sig(v, 5) for v in edges.tolist()], "counts": hist.tolist()}
        out["terminal"].append(s)
        out["used_seeds"].append(int(e["seed"]))
    out["nonclaims"] = ["readings of the declared law on the declared gluing; the slow band is the carrier's own repair-mean invariant, not a physical field",
                        "no continuum limit, no physical identification"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--terminals", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.run, args.cache, terminals=args.terminals)
    out = args.out or (args.run / "readback.json")
    out.write_bytes(H.canonical(result))
    i, t = result["initial"], result["terminal"][0]
    print(f"L{result['level']}: slow-band share initial {i['slow_band_energy_share']} -> terminal {t['slow_band_energy_share']} (isotropic 3/11 = {i['isotropic_reference_share']}); "
          f"slow norm mean {i['slow_norm_mean']} -> {t['slow_norm_mean']}, zero fraction {t['slow_norm_zero_fraction']}; slow field slope/corr on initial {t['slow_field_slope_on_initial']}/{t['slow_field_correlation_with_initial']}, "
          f"fast {t['fast_field_slope_on_initial']}/{t['fast_field_correlation_with_initial']}; slow-norm retention by scale {[(r['cells_per_cap'], r['slope']) for r in t['slow_norm_retention_by_scale']][:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
