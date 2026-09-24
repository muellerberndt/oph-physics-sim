"""Settling readouts of a federation run: descent curve, transport cost, and what the record forgets.

Reads a run directory of ``oph_exact.federation_huge`` (its receipt, per-schedule records and
terminal states) with the level's geometry cache and writes ``texture.json`` with three
finite readings of the integer nearest-agreement law:

* the descent curve: the excess ``V - V_min`` after every sweep as a fraction of the initial
  excess, its median over schedules, and the per-sweep contraction between two declared sweeps;
* the transport cost: unit transfers per port against the trivial lower bound
  ``(1/2) sum_p |x_p - mean|`` (half the L1 distance of the loads to the uniform reading), and
  swaps and waits per port;
* the retention of coarse initial imbalance in the terminal states: cells are binned into caps
  by their direction on the sphere (a cube grid of resolution ``r`` per axis); for every cap the
  mean reading of the initial loads and of a terminal state are compared as excesses over the
  global mean; the slope of the terminal excess on the initial excess and their correlation are
  reported per resolution, with the crossover cap size at which the slope crosses one half.

Every quantity is a reading of the declared law on the declared gluing; nothing here selects a
scale physically or asserts a limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "oph.exact.federation-texture.v1"
RESOLUTIONS = (3, 6, 12, 24, 48, 96, 192, 384)
MIN_CELLS_PER_CAP = 4
CONTRACTION_WINDOW = (8, 40)


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sig(x: float, digits: int = 6) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


def cap_labels(points: np.ndarray, res: int) -> np.ndarray:
    d = points / np.linalg.norm(points, axis=1, keepdims=True)
    key = np.minimum(np.floor((d + 1.0) * res / 2.0).astype(np.int64), res - 1)
    return key[:, 0] * res * res + key[:, 1] * res + key[:, 2]


def retention(points: np.ndarray, cell0: np.ndarray, cellt: Sequence[np.ndarray], resolutions: Sequence[int]) -> list[dict[str, Any]]:
    n = points.shape[0]
    rows = []
    for res in resolutions:
        ids, inv = np.unique(cap_labels(points, res), return_inverse=True)
        counts = np.bincount(inv)
        keep = counts >= MIN_CELLS_PER_CAP  # caps cut by the cube edges are dropped, the rest are compared
        if int(keep.sum()) < 8:
            continue
        m0 = (np.bincount(inv, weights=cell0) / np.maximum(counts, 1))[keep]
        e0 = m0 - float(cell0.mean())
        slopes, corrs, sds = [], [], []
        for ct in cellt:
            mt = (np.bincount(inv, weights=ct) / np.maximum(counts, 1))[keep]
            et = mt - float(ct.mean())
            slopes.append(float(np.dot(e0, et) / np.dot(e0, e0)))
            corrs.append(float(np.corrcoef(e0, et)[0, 1]))
            sds.append(float(et.std()))
        kept = int(keep.sum())
        mean_cells = float(counts[keep].mean())
        rows.append({"resolution": res, "caps": kept, "caps_dropped": int(len(ids) - kept), "cells_per_cap": _sig(mean_cells, 6), "ports_per_cap": _sig(12 * mean_cells, 6),
                     "initial_excess_sd": _sig(float(e0.std())), "terminal_excess_sd_median": _sig(float(np.median(sds))),
                     "slope_median": _sig(float(np.median(slopes))), "slope_min": _sig(min(slopes)), "slope_max": _sig(max(slopes)),
                     "correlation_median": _sig(float(np.median(corrs)))})
    return rows


def crossover(rows: list[dict[str, Any]], level: float = 0.5) -> float | None:
    """Cap size (cells) at which the median slope crosses ``level``, by log-linear interpolation."""

    pts = sorted(((r["cells_per_cap"], r["slope_median"]) for r in rows), key=lambda t: t[0])
    for (c0, s0), (c1, s1) in zip(pts, pts[1:]):
        if (s0 - level) * (s1 - level) <= 0 and s0 != s1:
            t = (level - s0) / (s1 - s0)
            return float(np.exp(np.log(c0) + t * (np.log(c1) - np.log(c0))))
    return None


def build(run: Path, cache: Path, *, resolutions: Sequence[int] = RESOLUTIONS, terminals: int = 4,
          window: tuple[int, int] = CONTRACTION_WINDOW) -> dict[str, Any]:
    receipt = json.loads((run / "receipt.json").read_text())
    level = int(receipt["level"])
    n = int(receipt["carriers"])
    ports = int(receipt["ports"])
    points = np.load(Path(cache) / f"L{level}" / "cell_points.npy")
    if points.shape[0] != n:
        raise ValueError("geometry cache does not match the receipt")
    loads = np.random.default_rng(20260909 + level).integers(0, 6, size=ports).astype(np.int64)
    entries = receipt["integer_law"]["entries"]
    # descent curves
    curves = []
    for e in entries:
        led = np.asarray(e["V_ledger"], dtype=np.float64) - float(e["V_minimum"])
        curves.append(led / led[0])
    longest = max(len(c) for c in curves)
    medians = [float(np.median([c[k] for c in curves if k < len(c)])) for k in range(longest)]
    a, b = window
    contraction = [float((c[b] / c[a]) ** (1.0 / (b - a))) for c in curves if b < len(c) and c[b] > 0]
    # transport
    lower = 0.5 * float(np.abs(loads - loads.mean()).sum())
    transfers = np.array([e["unit_transfers"] for e in entries], dtype=np.float64)
    swaps = np.array([e["swaps"] for e in entries], dtype=np.float64)
    waits = np.array([e["waits"] for e in entries], dtype=np.float64)
    # retention in the terminal states
    cell0 = loads.reshape(n, 12).mean(axis=1)
    terms = []
    used = []
    for e in entries[:terminals]:
        p = run / f"integer_{e['seed']}" / e["terminal_state"]["path"]
        x = np.load(p).astype(np.int64)
        if hashlib.sha256(x.astype("<i1").tobytes()).hexdigest() != e["terminal_state"]["sha256"]:
            raise ValueError(f"terminal state of seed {e['seed']} does not match its digest")
        terms.append(x.reshape(n, 12).mean(axis=1))
        used.append(int(e["seed"]))
    rows = retention(points, cell0, terms, resolutions)
    return {
        "schema": SCHEMA,
        "level": level, "carriers": n, "ports": ports, "schedules": len(entries),
        "receipt_sha256": hashlib.sha256((run / "receipt.json").read_bytes()).hexdigest(),
        "descent_curve": {
            "definition": "(V_k - V_min) / (V_0 - V_min) after sweep k, median over schedules",
            "median_by_sweep": [_sig(m, 6) for m in medians],
            "sweeps": {"min": min(len(c) - 1 for c in curves), "max": max(len(c) - 1 for c in curves), "mean": _sig(float(np.mean([len(c) - 1 for c in curves])), 6)},
            "log2_law_reference": _sig(4.3 * np.log2(n) - 9.0, 6),
            "contraction_per_sweep": {"window": list(window), "median": _sig(float(np.median(contraction))) if contraction else None,
                                      "min": _sig(min(contraction)) if contraction else None, "max": _sig(max(contraction)) if contraction else None},
            "sweeps_below": {str(t): [int(np.sum(c * (np.asarray(e["V_ledger"][0]) - e["V_minimum"]) < t)) for c, e in zip(curves, entries)] for t in (100, 10)},
        },
        "transport": {
            "lower_bound_definition": "(1/2) sum_p |x_p - mean(x)|: half the L1 distance of the loads to the uniform reading",
            "lower_bound_per_port": _sig(lower / ports, 8),
            "unit_transfers_per_port": {"min": _sig(transfers.min() / ports, 8), "max": _sig(transfers.max() / ports, 8)},
            "transfers_over_lower_bound": {"min": _sig(transfers.min() / lower, 8), "max": _sig(transfers.max() / lower, 8), "mean": _sig(float(transfers.mean()) / lower, 8)},
            "swaps_per_port": {"min": _sig(swaps.min() / ports), "max": _sig(swaps.max() / ports)},
            "waits_per_port": {"min": _sig(waits.min() / ports), "max": _sig(waits.max() / ports)},
        },
        "retention": {
            "definition": "cells binned into caps by direction (cube grid, resolution r per axis); per cap the mean reading of the initial loads "
                          "and of a terminal state as excesses over the global mean; slope = <e0, et> / <e0, e0>, correlation = corr(e0, et)",
            "terminal_states": used,
            "rows": rows,
            "crossover_cells_at_slope_one_half": _sig(crossover(rows), 6) if crossover(rows) is not None else None,
            "crossover_ports_at_slope_one_half": _sig(12 * crossover(rows), 6) if crossover(rows) is not None else None,
        },
        "nonclaims": [
            "readings of the declared integer law on the declared gluing at one load seed; no physical scale, no limit",
            "the retention rows describe terminal configurations; the public quotient record is the component multiset, which carries no arrangement",
            "caps are a fixed binning of the sphere, not a source-derived coarse graining",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--terminals", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.run, args.cache, terminals=args.terminals)
    out = args.out or (args.run / "texture.json")
    out.write_bytes(canonical(result))
    d = result["descent_curve"]
    r = result["retention"]
    print(f"L{result['level']}: sweeps {d['sweeps']}, contraction/sweep {d['contraction_per_sweep']['median']}, transfers/lower bound {result['transport']['transfers_over_lower_bound']['mean']}, "
          f"crossover {r['crossover_cells_at_slope_one_half']} cells; slopes {[(row['cells_per_cap'], row['slope_median']) for row in r['rows']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
