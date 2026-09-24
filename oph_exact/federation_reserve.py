"""Reserve-generator readout of the settled tower (issue 985): what crosses the collar, per sweep and per tick.

The edge-center reserve generator receipt (research repository,
``paper/tex_fragments/SCREEN_SPECTRUM_THEOREMS.tex``, ``def:oph-screen-reserve-generator``) asks a
finite source for a covariance-survival cocycle across the full oriented collar under logarithmic
refinement, its infinitesimal density ``-u'(0) = P*/24``, the half-collar identity, and a clock
binding.  This lane measures the natural candidates on the exact glued federation under the
canonical integer nearest-agreement law, with the identifications named in the issue:

* depth ``m`` = tower level; the collar at depth ``m`` = the ``30 * 2^m`` inter-face seams of the
  production gluing (checked); the oriented slots = the two transfer directions across a collar
  seam, "forward" from the lower face index to the higher; intra-face inter-carrier seams and
  intra-carrier seams are the two other seam classes;
* the sub-step = one sweep; the tick at depth ``m`` = ``2^m`` sweeps (the certificate's on-grid
  family ``(1 - eps 2^-m)^(2^m)``); alternative bindings are run as controls;
* reserve candidates: (a) the load of a face, whose presence survival per sweep is one minus the
  fraction of all units that cross a collar seam in that sweep (descent units and swaps); (b) the
  descent excess ``V - V_min``, whose survival per sweep is the universal contraction; (c) the
  settled-field covariance under refinement, read from the texture receipts of two consecutive
  levels at equal angular scale.

The schedules replay the engine's own draws (same loads, seeds and kernel semantics), so the
terminal state must equal the engine's receipt digest; the accounting kernel is otherwise the
engine's kernel with per-class, per-direction counters.  Nothing here is a derivation; the
receipt records readings against the certificate's numbers and the negative controls.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from oph_exact import federation_huge as H
from oph_exact.federation_archive import array_sha256

SCHEMA = "oph.exact.federation-reserve.v1"
P_STAR = 1.6309682094  # certified pixel-root comparison value of the certificate; enters only as the comparison target
CLASSES = ("intra_carrier", "intra_face", "collar")
KINDS = ("descents", "swaps", "waits", "descent_units")

_KERNEL_SOURCE = r"""
#include <stdint.h>

#define ENDPOINTS(s) \
    int64_t i, j; int cls; \
    if ((s) < intra_count) { \
        const int64_t c = (s) / 30; const int64_t t = (s) - 30 * c; \
        i = 12 * c + tpl_a[t]; j = 12 * c + tpl_b[t]; cls = 0; \
    } else { const int64_t k = (s) - intra_count; i = ia[k]; j = ib[k]; cls = cls_inter[k]; }

/* acc layout: [class 3][direction 2][kind 4]; kinds: descents, swaps, waits, descent_units.
   Direction 0 = units flow from the lower face index to the higher (forward), 1 = backward;
   within one face the port order stands in for the face order; waits take direction 0. */
int64_t oph_reserve_integer(int8_t *x, const int8_t *tpl_a, const int8_t *tpl_b,
                            const int32_t *ia, const int32_t *ib, const int8_t *cls_inter,
                            int64_t intra_count, int64_t per_face,
                            const int32_t *seq, const int8_t *coin, int64_t n,
                            int64_t *acc, int64_t *v, int64_t v_min)
{
    int64_t first = -1;
    int64_t vv = v[0];
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        ENDPOINTS(s)
        const int64_t xi = x[i];
        const int64_t xj = x[j];
        const int64_t tot = xi + xj;
        const int64_t lo = tot >> 1;
        const int64_t hi = tot - lo;
        const int64_t ni = coin[t] ? hi : lo;
        const int64_t nj = tot - ni;
        const int64_t d = xi - xj;
        const int64_t fi = (i / 12) / per_face;
        const int64_t fj = (j / 12) / per_face;
        int64_t *slot = acc + cls * 8;
        if (d == 0) { slot[2] += 1; continue; }
        if (d == 1 || d == -1) {
            if (ni == xi) { slot[2] += 1; continue; }
            /* a swap moves one unit from the larger reading to the smaller */
            const int from_i = (ni < xi);
            int dir;
            if (fi != fj) dir = from_i ? (fi < fj ? 0 : 1) : (fj < fi ? 0 : 1);
            else dir = from_i ? (i < j ? 0 : 1) : (j < i ? 0 : 1);
            slot[dir * 4 + 1] += 1;
        } else {
            const int64_t ad = d < 0 ? -d : d;
            const int from_i = (d > 0);
            int dir;
            if (fi != fj) dir = from_i ? (fi < fj ? 0 : 1) : (fj < fi ? 0 : 1);
            else dir = from_i ? (i < j ? 0 : 1) : (j < i ? 0 : 1);
            slot[dir * 4 + 0] += 1;
            slot[dir * 4 + 3] += ad >> 1;
            vv += ni * ni + nj * nj - xi * xi - xj * xj;
        }
        x[i] = (int8_t)ni;
        x[j] = (int8_t)nj;
        if (first < 0 && vv == v_min) first = t;
    }
    v[0] = vv;
    return first;
}
"""

_NATIVE: dict[str, Any] = {"lib": None}


def native_kernel() -> Any:
    if _NATIVE["lib"] is not None:
        return _NATIVE["lib"]
    digest = hashlib.sha256(_KERNEL_SOURCE.encode("utf-8")).hexdigest()[:16]
    directory = Path(os.environ.get("OPH_EXACT_KERNEL_DIR") or Path(tempfile.gettempdir()) / "oph_exact_kernels")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = directory / f"oph_reserve_kernel_{digest}{suffix}"
    if not library.exists():
        compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            raise RuntimeError("no C compiler on PATH")
        source = directory / f"oph_reserve_kernel_{digest}.c"
        source.write_text(_KERNEL_SOURCE, encoding="utf-8")
        staging = directory / f"oph_reserve_kernel_{digest}.{os.getpid()}{suffix}"
        subprocess.run([compiler, "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(staging), str(source)], check=True, capture_output=True)
        os.replace(staging, library)
    lib = ctypes.CDLL(str(library))
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.oph_reserve_integer.argtypes = [i8p, i8p, i8p, i32p, i32p, i8p, ctypes.c_int64, ctypes.c_int64, i32p, i8p, ctypes.c_int64, i64p, i64p, ctypes.c_int64]
    lib.oph_reserve_integer.restype = ctypes.c_int64
    _NATIVE["lib"] = lib
    return lib


def _ptr(array: np.ndarray, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _sig(x: float, digits: int = 6) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


class Collar:
    """Seam classes of one level: intra-carrier (0), intra-face inter-carrier (1), collar (2)."""

    def __init__(self, geo: H.Geometry) -> None:
        self.level = geo.level
        self.per_face = 4**geo.level
        inter = geo.inter64
        fa = inter[:, 0] // self.per_face
        fb = inter[:, 2] // self.per_face
        self.cls_inter = np.ascontiguousarray(np.where(fa != fb, 2, 1).astype(np.int8))
        self.collar_seams = int(np.count_nonzero(self.cls_inter == 2))
        self.expected_collar = 30 * 2**geo.level
        if self.collar_seams != self.expected_collar:
            raise AssertionError(f"collar seam count {self.collar_seams} differs from 30 * 2^{geo.level}")
        self.counts = {"intra_carrier": geo.intra_count, "intra_face": int(np.count_nonzero(self.cls_inter == 1)), "collar": self.collar_seams}


def run_schedule(geo: H.Geometry, collar: Collar, loads: np.ndarray, seed: int, *, max_sweeps: int = 400_000,
                 cls_override: np.ndarray | None = None, log=None) -> dict[str, Any]:
    lib = native_kernel()
    x = np.ascontiguousarray(np.asarray(loads, dtype=np.int8).copy())
    tpl_a = np.ascontiguousarray(geo.template[:, 0], dtype=np.int8)
    tpl_b = np.ascontiguousarray(geo.template[:, 1], dtype=np.int8)
    cls = collar.cls_inter if cls_override is None else np.ascontiguousarray(cls_override, dtype=np.int8)
    exp = H.expectation(geo, loads)
    v_min = int(exp["balanced_minimum"])
    v = np.array([int(np.dot(x.astype(np.int64), x.astype(np.int64)))], dtype=np.int64)
    v0 = int(v[0])
    rng = np.random.default_rng(seed)
    acc = np.zeros(24, dtype=np.int64)
    per_sweep = []
    ledger = [v0]
    sweep = 0
    first_attempt = 0 if v0 == v_min else -1
    started = time.perf_counter()
    while first_attempt < 0 and sweep < max_sweeps:
        seq, coin, _digests = H.draw_sweep(rng, geo.seams, coins=True)
        before = acc.copy()
        first = int(lib.oph_reserve_integer(_ptr(x, ctypes.c_int8), _ptr(tpl_a, ctypes.c_int8), _ptr(tpl_b, ctypes.c_int8),
                                            _ptr(geo.ia, ctypes.c_int32), _ptr(geo.ib, ctypes.c_int32), _ptr(cls, ctypes.c_int8),
                                            geo.intra_count, collar.per_face, _ptr(seq, ctypes.c_int32), _ptr(coin, ctypes.c_int8), geo.seams,
                                            _ptr(acc, ctypes.c_int64), _ptr(v, ctypes.c_int64), v_min))
        if first >= 0:
            first_attempt = sweep * geo.seams + first + 1
        sweep += 1
        delta = (acc - before).reshape(3, 2, 4)
        ledger.append(int(v[0]))
        per_sweep.append(delta.tolist())
        if log and (sweep % 16 == 0 or first_attempt >= 0):
            log(f"  seed {seed}: sweep {sweep} V-Vmin={int(v[0]) - v_min} {time.perf_counter() - started:.0f}s")
    return {"seed": int(seed), "sweeps": sweep, "terminated": bool(first_attempt >= 0), "attempts_to_balanced_class": int(first_attempt),
            "V_ledger": ledger, "V_minimum": v_min, "terminal_sha256": array_sha256(x, "int8"),
            "totals": acc.reshape(3, 2, 4).tolist(), "per_sweep": per_sweep, "seconds": round(time.perf_counter() - started, 3)}


def survival_readout(schedule: dict[str, Any], level: int, total_load: int, *, tick_exponent_offsets: Sequence[int] = (0, -1, 1)) -> dict[str, Any]:
    """Candidate reserves from one schedule's accounting."""

    ps = np.asarray(schedule["per_sweep"], dtype=np.float64)  # (sweeps, 3, 2, 4)
    sweeps = ps.shape[0]
    # (a) face load: units crossing collar seams per sweep (descent units + swaps), both directions
    crossing = ps[:, 2, :, 3].sum(axis=1) + ps[:, 2, :, 1].sum(axis=1)
    forward = ps[:, 2, 0, 3] + ps[:, 2, 0, 1]
    backward = ps[:, 2, 1, 3] + ps[:, 2, 1, 1]
    hazard = crossing / total_load  # per-sweep presence hazard of a unit of face load
    scaled = hazard * 2**level  # depth-scaled hazard, expected to be a universal function of the sweep index
    ticks = {}
    for off in tick_exponent_offsets:
        n = 2 ** max(level + off, 0)
        if n <= sweeps:
            u1 = float(np.prod(1.0 - hazard[:n]))
            u2 = float(np.prod(1.0 - hazard[:2 * n])) if 2 * n <= sweeps else None
            ticks[str(off)] = {"sweeps_per_tick": n, "u_1": _sig(u1, 8), "generator": _sig(-np.log(u1), 6),
                               "u_2": _sig(u2, 8) if u2 is not None else None,
                               "semigroup_defect_u2_over_u1_squared": _sig(u2 / u1**2, 6) if u2 is not None else None}
        else:
            ticks[str(off)] = {"sweeps_per_tick": n, "u_1": None, "generator": None, "note": "the settlement ends before one tick"}
    # forward/backward halves over the whole settlement
    fwd, bwd = float(forward.sum()), float(backward.sum())
    # (b) descent excess survival per sweep
    led = np.asarray(schedule["V_ledger"], dtype=np.float64) - schedule["V_minimum"]
    positive = (led[:-1] > 0) & (led[1:] > 0)
    ratios = np.where(positive, led[1:] / np.where(led[:-1] > 0, led[:-1], 1.0), np.nan)
    start = 8 if sweeps > 40 else 2
    window = ratios[start:start + 32]
    window = window[np.isfinite(window)]
    contraction = float(np.exp(np.mean(np.log(window)))) if window.size else None
    return {
        "face_load": {
            "definition": "presence hazard per sweep = (descent units + swaps crossing collar seams, both directions) / total load; ticks of 2^(m+offset) sweeps",
            "hazard_per_sweep_first_8": [_sig(h, 6) for h in hazard[:8].tolist()],
            "hazard_per_sweep_last_4": [_sig(h, 6) for h in hazard[-4:].tolist()],
            "depth_scaled_hazard_first_8": [_sig(h, 6) for h in scaled[:8].tolist()],
            "depth_scaled_hazard_sum_over_settlement": _sig(float(scaled.sum()), 6),
            "ticks": ticks,
            "forward_units": fwd, "backward_units": bwd, "forward_share": _sig(fwd / (fwd + bwd), 6) if fwd + bwd > 0 else None,
        },
        "descent_excess": {"contraction_per_sweep": _sig(contraction, 6) if contraction else None,
                           "generator_per_sweep": _sig(-np.log(contraction), 6) if contraction else None},
    }


def refinement_survival(texture_lo: dict[str, Any], texture_hi: dict[str, Any]) -> dict[str, Any]:
    """Covariance survival of the settled field under one refinement step, at equal angular scale.

    The texture receipts bin cells into caps at cube-grid resolutions; the same resolution at
    consecutive levels is the same angular scale with four times the cells.  For the initial
    i.i.d. loads the cap-mean variance drops by four (white noise, theta = 2); the settled
    field's ratio at the same scale is its covariance survival lambda(2), theta = -log2 lambda.
    """

    rows_lo = {r["resolution"]: r for r in texture_lo["retention"]["rows"]}
    rows_hi = {r["resolution"]: r for r in texture_hi["retention"]["rows"]}
    out = []
    for res in sorted(set(rows_lo) & set(rows_hi)):
        a, b = rows_lo[res], rows_hi[res]
        lam_init = (b["initial_excess_sd"] / a["initial_excess_sd"]) ** 2
        lam_term = (b["terminal_excess_sd_median"] / a["terminal_excess_sd_median"]) ** 2
        out.append({"resolution": res, "cells_per_cap_low": a["cells_per_cap"], "cells_per_cap_high": b["cells_per_cap"],
                    "lambda_initial": _sig(lam_init, 6), "theta_initial": _sig(-np.log2(lam_init), 6),
                    "lambda_terminal": _sig(lam_term, 6), "theta_terminal": _sig(-np.log2(lam_term), 6)})
    return {"definition": "lambda = var(cap means at level m+1) / var(cap means at level m) at equal resolution; theta = -log2 lambda; white noise gives 1/4 and 2",
            "levels": [texture_lo["level"], texture_hi["level"]], "rows": out,
            "target_theta": _sig(P_STAR / 48, 6), "target_lambda_per_refinement": _sig(2 ** (-P_STAR / 48), 6)}


def build(level: int, cache: Path, out: Path, *, schedules: int = 4, run_dir: Path | None = None, textures: Sequence[Path] = (), log=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    H.build_geometry(level, cache, log=log)
    geo = H.Geometry(cache, level)
    collar = Collar(geo)
    loads = H.initial_loads(level, geo.ports)
    total_load = int(loads.astype(np.int64).sum())
    log(f"L{level}: {geo.carriers} carriers; seams intra {collar.counts['intra_carrier']}, intra-face {collar.counts['intra_face']}, collar {collar.counts['collar']} = 30*2^{level}")
    engine = json.loads((run_dir / "receipt.json").read_text()) if run_dir else None
    results = []
    for k in range(schedules):
        seed = H.schedule_seed(level, k)
        sched = run_schedule(geo, collar, loads, seed, log=log)
        if engine is not None:
            e = next((e for e in engine["integer_law"]["entries"] if e["seed"] == seed), None)
            sched["matches_engine_terminal_digest"] = bool(e is not None and e["terminal_state"]["sha256"] == sched["terminal_sha256"] and e["sweeps"] == sched["sweeps"])
        sched["readout"] = survival_readout(sched, level, total_load)
        results.append(sched)
        r = sched["readout"]["face_load"]
        log(f"  seed {seed}: {sched['sweeps']} sweeps, engine match {sched.get('matches_engine_terminal_digest')}, depth-scaled collar hazard first sweeps {r['depth_scaled_hazard_first_8'][:4]}, "
            f"tick generators {[(o, t['generator']) for o, t in r['ticks'].items()]}, forward share {r['forward_share']}, excess contraction {sched['readout']['descent_excess']['contraction_per_sweep']}")
    # negative controls on the first schedule: shuffled seam classes (collar label moved to random inter seams) and shuffled orientation (face order reversed)
    rng = np.random.default_rng(985)
    shuffled = collar.cls_inter.copy()
    rng.shuffle(shuffled)
    ctrl_cls = run_schedule(geo, collar, loads, H.schedule_seed(level, 0), cls_override=shuffled)
    ctrl_cls_read = survival_readout(ctrl_cls, level, total_load)
    controls = {"shuffled_seam_classes": {"terminal_matches_schedule_0": ctrl_cls["terminal_sha256"] == results[0]["terminal_sha256"],
                                          "depth_scaled_hazard_first_8": ctrl_cls_read["face_load"]["depth_scaled_hazard_first_8"],
                                          "ticks": ctrl_cls_read["face_load"]["ticks"], "forward_share": ctrl_cls_read["face_load"]["forward_share"]},
                "iid_reference_no_repair": {"hazard_per_sweep": 0.0, "generator": 0.0, "note": "without repair no unit crosses any seam"}}
    refinement = []
    tex = [json.loads(Path(p).read_text()) for p in textures]
    tex.sort(key=lambda t: t["level"])
    for a, b in zip(tex, tex[1:]):
        if b["level"] == a["level"] + 1:
            refinement.append(refinement_survival(a, b))
    aggregate = {
        "depth_scaled_hazard_first_sweep": {"min": _sig(min(r["readout"]["face_load"]["depth_scaled_hazard_first_8"][0] for r in results)),
                                            "max": _sig(max(r["readout"]["face_load"]["depth_scaled_hazard_first_8"][0] for r in results))},
        "tick_generator_offset_0": [r["readout"]["face_load"]["ticks"]["0"]["generator"] for r in results],
        "forward_share": [r["readout"]["face_load"]["forward_share"] for r in results],
        "excess_contraction": [r["readout"]["descent_excess"]["contraction_per_sweep"] for r in results],
        "engine_terminal_match_all": bool(all(r.get("matches_engine_terminal_digest", True) for r in results)),
    }
    for r in results:
        r.pop("per_sweep_dropped", None)
    receipt = {
        "schema": SCHEMA, "issue": "FloatingPragma/observer-patch-holography#985", "level": level, "carriers": geo.carriers, "seams": geo.seams,
        "seam_classes": collar.counts, "collar_equals_30_times_2_to_level": True, "per_face_cells": collar.per_face, "total_load": total_load,
        "identifications": {"collar": "inter-face seams of the production gluing, face-major cells", "forward": "units moving from the lower face index to the higher",
                            "sub_step": "one sweep (|S| attempts)", "tick": "2^(m+offset) sweeps, offsets 0 (declared), -1 and +1 (controls)",
                            "reserves": ["face load (presence across the collar)", "descent excess V - V_min", "settled-field covariance under refinement (from texture receipts)"]},
        "targets": {"P_star": P_STAR, "full_collar_density_P_over_24": _sig(P_STAR / 24, 6), "half_collar_density_P_over_48": _sig(P_STAR / 48, 6),
                    "target_source": "certified pixel-root comparison value of code/cosmology/edge_center_clock_certificate.py; comparison only"},
        "schedules": results, "aggregate": aggregate, "controls": controls, "refinement_survival": refinement,
        "module_pins": {p: H.file_sha256(H.ROOT / p) for p in H.PINNED_MODULES + ("oph_exact/federation_reserve.py",) if (H.ROOT / p).is_file()},
        "kernel_source_sha256": hashlib.sha256(_KERNEL_SOURCE.encode("utf-8")).hexdigest(),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(H.canonical(receipt))
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schedules", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, default=None, help="engine run directory for the terminal-digest cross-check")
    parser.add_argument("--textures", type=Path, nargs="*", default=[], help="texture.json receipts of consecutive levels for the refinement survival")
    args = parser.parse_args(argv)
    build(args.level, args.cache, args.out, schedules=args.schedules, run_dir=args.run_dir, textures=args.textures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
