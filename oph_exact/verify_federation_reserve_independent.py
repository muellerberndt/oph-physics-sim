"""Independent verifier for reserve-generator receipts (``oph_exact.federation_reserve``).

Recomputes every readout from the per-sweep accounting tables of the receipt and checks the
receipt against the engine receipts and texture receipts it cites.  It imports no simulator
module.

Checks: the tower counts (carriers ``20 * 4^L``, collar seams ``30 * 2^L``, the three seam
classes summing to the seam count); every schedule's totals equal the sums of its per-sweep
tables and, when an engine receipt is given, its descents, swaps, waits, unit transfers, sweeps,
``V`` ledger and terminal digest; the per-sweep hazards, depth-scaled hazards, tick survivals
and generators, the forward share and the excess contraction recomputed from the tables; the
refinement-survival rows recomputed from the cited texture receipts; the shuffled-class control
(same terminal digest, different tables); the aggregate summaries; and the comparison targets
``P*/24`` and ``P*/48`` against the certified pixel-root value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

P_STAR = 1.6309682094


class VerificationError(AssertionError):
    pass


def require(c: bool, m: str) -> None:
    if not c:
        raise VerificationError(m)


def _sig(x: float, digits: int = 6) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


def close(a: float | None, b: float | None, rel: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300) + 1e-12


def readout_from_tables(sched: dict[str, Any], level: int, total_load: int) -> dict[str, Any]:
    ps = np.asarray(sched["per_sweep"], dtype=np.float64)
    sweeps = ps.shape[0]
    crossing = ps[:, 2, :, 3].sum(axis=1) + ps[:, 2, :, 1].sum(axis=1)
    hazard = crossing / total_load
    scaled = hazard * 2**level
    ticks = {}
    for off in (0, -1, 1):
        n = 2 ** max(level + off, 0)
        if n <= sweeps:
            u1 = float(np.prod(1.0 - hazard[:n]))
            u2 = float(np.prod(1.0 - hazard[:2 * n])) if 2 * n <= sweeps else None
            ticks[str(off)] = {"u_1": u1, "generator": -np.log(u1), "u_2": u2, "defect": (u2 / u1**2) if u2 is not None else None}
        else:
            ticks[str(off)] = None
    forward = float((ps[:, 2, 0, 3] + ps[:, 2, 0, 1]).sum())
    backward = float((ps[:, 2, 1, 3] + ps[:, 2, 1, 1]).sum())
    led = np.asarray(sched["V_ledger"], dtype=np.float64) - sched["V_minimum"]
    positive = (led[:-1] > 0) & (led[1:] > 0)
    ratios = np.where(positive, led[1:] / np.where(led[:-1] > 0, led[:-1], 1.0), np.nan)
    start = 8 if sweeps > 40 else 2
    window = ratios[start:start + 32]
    window = window[np.isfinite(window)]
    contraction = float(np.exp(np.mean(np.log(window)))) if window.size else None
    return {"hazard": hazard, "scaled": scaled, "ticks": ticks, "forward": forward, "backward": backward, "contraction": contraction, "sweeps": sweeps}


def check_schedule(sched: dict[str, Any], level: int, total_load: int, engine_entry: dict[str, Any] | None) -> dict[str, Any]:
    seed = sched["seed"]
    ps = np.asarray(sched["per_sweep"], dtype=np.int64)
    require(ps.shape == (sched["sweeps"], 3, 2, 4), f"seed {seed}: per-sweep table shape")
    require(np.array_equal(ps.sum(axis=0), np.asarray(sched["totals"], dtype=np.int64)), f"seed {seed}: totals differ from the per-sweep sums")
    require(len(sched["V_ledger"]) == sched["sweeps"] + 1 and sched["V_ledger"][-1] == sched["V_minimum"] and sched["terminated"], f"seed {seed}: ledger end")
    require(all(a >= b for a, b in zip(sched["V_ledger"], sched["V_ledger"][1:])), f"seed {seed}: ledger not monotone")
    tot = ps.sum(axis=0)
    if engine_entry is not None:
        require(engine_entry["terminal_state"]["sha256"] == sched["terminal_sha256"], f"seed {seed}: terminal digest differs from the engine receipt")
        require(engine_entry["sweeps"] == sched["sweeps"] and engine_entry["V_ledger"] == sched["V_ledger"], f"seed {seed}: sweeps or ledger differ from the engine receipt")
        require(int(tot[:, :, 0].sum()) == engine_entry["descents"] and int(tot[:, :, 1].sum()) == engine_entry["swaps"], f"seed {seed}: descents or swaps differ from the engine receipt")
        require(int(tot[:, :, 2].sum()) == engine_entry["waits"] and int(tot[:, :, 3].sum()) == engine_entry["unit_transfers"], f"seed {seed}: waits or unit transfers differ from the engine receipt")
        require(sched.get("matches_engine_terminal_digest") is True, f"seed {seed}: engine-match flag")
    r = readout_from_tables(sched, level, total_load)
    fl = sched["readout"]["face_load"]
    require(all(close(a, _sig(b, 6)) for a, b in zip(fl["hazard_per_sweep_first_8"], r["hazard"][:8].tolist())), f"seed {seed}: hazards")
    require(all(close(a, _sig(b, 6)) for a, b in zip(fl["depth_scaled_hazard_first_8"], r["scaled"][:8].tolist())), f"seed {seed}: depth-scaled hazards")
    require(close(fl["depth_scaled_hazard_sum_over_settlement"], _sig(float(r["scaled"].sum()), 6)), f"seed {seed}: hazard sum")
    for off, t in r["ticks"].items():
        declared = fl["ticks"][off]
        if t is None:
            require(declared["u_1"] is None, f"seed {seed}: tick {off} should be unavailable")
        else:
            require(close(declared["u_1"], _sig(t["u_1"], 8)) and close(declared["generator"], _sig(t["generator"], 6)), f"seed {seed}: tick {off} survival or generator")
            if t["u_2"] is not None:
                require(close(declared["semigroup_defect_u2_over_u1_squared"], _sig(t["defect"], 6)), f"seed {seed}: tick {off} defect")
    require(fl["forward_units"] == r["forward"] and fl["backward_units"] == r["backward"], f"seed {seed}: orientation totals")
    share = r["forward"] / (r["forward"] + r["backward"]) if r["forward"] + r["backward"] > 0 else None
    require(close(fl["forward_share"], _sig(share, 6) if share is not None else None), f"seed {seed}: forward share")
    de = sched["readout"]["descent_excess"]
    require(close(de["contraction_per_sweep"], _sig(r["contraction"], 6) if r["contraction"] else None), f"seed {seed}: excess contraction")
    return {"seed": seed, "sweeps": r["sweeps"], "forward_share": share, "contraction": r["contraction"],
            "tick_generator": r["ticks"]["0"]["generator"] if r["ticks"]["0"] else None, "first_scaled_hazard": float(r["scaled"][0])}


def check_refinement(rows: list[dict[str, Any]], textures: dict[int, dict[str, Any]]) -> None:
    for block in rows:
        lo, hi = block["levels"]
        require(lo in textures and hi in textures and hi == lo + 1, f"refinement block {lo}->{hi}: textures missing or not consecutive")
        rl = {r["resolution"]: r for r in textures[lo]["retention"]["rows"]}
        rh = {r["resolution"]: r for r in textures[hi]["retention"]["rows"]}
        for row in block["rows"]:
            a, b = rl[row["resolution"]], rh[row["resolution"]]
            li = (b["initial_excess_sd"] / a["initial_excess_sd"]) ** 2
            lt = (b["terminal_excess_sd_median"] / a["terminal_excess_sd_median"]) ** 2
            require(close(row["lambda_initial"], _sig(li, 6)) and close(row["theta_initial"], _sig(-np.log2(li), 6)), f"refinement {lo}->{hi} res {row['resolution']}: initial")
            require(close(row["lambda_terminal"], _sig(lt, 6)) and close(row["theta_terminal"], _sig(-np.log2(lt), 6)), f"refinement {lo}->{hi} res {row['resolution']}: terminal")
        require(close(block["target_theta"], _sig(P_STAR / 48, 6)), "refinement target")


def verify(path: Path, *, engine: Path | None = None, textures: list[Path] = (), log=print) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text())
    require(receipt["schema"] == "oph.exact.federation-reserve.v1", "schema")
    level = int(receipt["level"])
    require(receipt["carriers"] == 20 * 4**level, "carrier count")
    sc = receipt["seam_classes"]
    require(sc["intra_carrier"] == 30 * receipt["carriers"] and sc["collar"] == 30 * 2**level, "seam classes")
    require(sc["intra_carrier"] + sc["intra_face"] + sc["collar"] == receipt["seams"] == 30 * receipt["carriers"] + 3 * receipt["carriers"] // 2, "seam class sum")
    require(receipt["collar_equals_30_times_2_to_level"] is True and receipt["per_face_cells"] == 4**level, "collar flags")
    loads = np.random.default_rng(20260909 + level).integers(0, 6, size=12 * receipt["carriers"], dtype=np.int64)
    require(int(loads.sum()) == receipt["total_load"], "total load from the loads rule")
    require(close(receipt["targets"]["full_collar_density_P_over_24"], _sig(P_STAR / 24, 6)) and close(receipt["targets"]["half_collar_density_P_over_48"], _sig(P_STAR / 48, 6)), "targets")
    engine_receipt = json.loads(Path(engine).read_text()) if engine else None
    entries = {e["seed"]: e for e in engine_receipt["integer_law"]["entries"]} if engine_receipt else {}
    if engine_receipt:
        require(engine_receipt["level"] == level, "engine receipt level")
    results = []
    for sched in receipt["schedules"]:
        results.append(check_schedule(sched, level, receipt["total_load"], entries.get(sched["seed"])))
        log(f"seed {sched['seed']}: ok, {results[-1]['sweeps']} sweeps, forward share {results[-1]['forward_share']:.4f}, contraction {results[-1]['contraction']:.4f}, tick generator {results[-1]['tick_generator']}")
    agg = receipt["aggregate"]
    require(agg["tick_generator_offset_0"] == [s["readout"]["face_load"]["ticks"]["0"]["generator"] for s in receipt["schedules"]], "aggregate tick generators")
    require(agg["forward_share"] == [s["readout"]["face_load"]["forward_share"] for s in receipt["schedules"]], "aggregate forward shares")
    require(agg["engine_terminal_match_all"] == all(s.get("matches_engine_terminal_digest", True) for s in receipt["schedules"]), "aggregate engine match")
    ctrl = receipt["controls"]["shuffled_seam_classes"]
    require(ctrl["terminal_matches_schedule_0"] is True, "shuffled-class control must reproduce the trajectory")
    require(ctrl["depth_scaled_hazard_first_8"] != receipt["schedules"][0]["readout"]["face_load"]["depth_scaled_hazard_first_8"], "shuffled-class control must change the collar hazard")
    tex = {}
    for p in textures:
        t = json.loads(Path(p).read_text())
        tex[int(t["level"])] = t
    if receipt["refinement_survival"]:
        require(tex, "refinement rows present but no texture receipts given")
        check_refinement(receipt["refinement_survival"], tex)
        log(f"refinement survival ok: {[(b['levels'], [r['theta_terminal'] for r in b['rows']][:3]) for b in receipt['refinement_survival']]}")
    return {"verdict": "PASS", "level": level, "schedules": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--textures", type=Path, nargs="*", default=[])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = verify(args.receipt, engine=args.engine, textures=args.textures)
    except VerificationError as error:
        print(f"FAIL: {error}")
        return 1
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
