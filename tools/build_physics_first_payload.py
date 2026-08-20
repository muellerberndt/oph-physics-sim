"""Extract the physics-first visualizer payload from an OPH universe run.

Usage:
    python tools/build_physics_first_payload.py <run_dir> <out.json> <qm_dir> [control_run_dir]

The qm_dir is the output of ``python -m oph_fpe.qm_observer --output-dir <dir>``.

Every block carries `earned`: "measured" (computed in this run), "exact"
(closed-form/algebraic, reproducible here), or "declared" (a named bridge the
corpus has not closed, shown so the picture completes).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def r(x, n=6):
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return 0.0 if abs(v) < 1e-15 else round(v, n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------------------------------------------------------------- carrier ---
def icosahedron():
    phi = (1 + 5 ** 0.5) / 2
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s1 * phi, 0, s2)]
    V = np.array(sorted(set(verts)), dtype=float)
    V /= np.linalg.norm(V[0])
    D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    near = np.sort(D[0][D[0] > 1e-9])[0]
    A = (np.abs(D - near) < 1e-9).astype(int)
    np.fill_diagonal(A, 0)
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if A[i, j]]
    faces = [
        (i, j, k)
        for i in range(12) for j in range(i + 1, 12) for k in range(j + 1, 12)
        if A[i, j] and A[j, k] and A[i, k]
    ]
    return V, A, edges, faces


def block_carrier():
    V, A, edges, faces = icosahedron()
    return {
        "earned": "exact",
        "vertices": r(V.tolist(), 5),
        "edges": edges,
        "faces": faces,
        "ports": 12,
        "seams": len(edges),
        "cells": len(faces),
        "independent_cycles": len(edges) - 12 + 1,
        "rotation_group": {"name": "A5", "order": 60,
                           "element_orders": {"1": 1, "2": 15, "3": 20, "5": 24}},
    }


# ------------------------------------------------- why three dimensions ---
def block_dimension():
    """The exact rank-three result: repair damps A5 irreps at different rates
    and the slowest surviving band is three-dimensional."""
    V, A, edges, _ = icosahedron()
    L = np.diag(A.sum(1)) - A
    evals, evecs = np.linalg.eigh(L)
    T = np.eye(12) - L / 60.0

    bands = []
    for lam, mult, name, role in [
        (0.0, 1, "constant", "global offset — carries no position"),
        (5 - 5 ** 0.5, 3, "slow", "survives: becomes the three spatial directions"),
        (6.0, 5, "middle", "damped away"),
        (5 + 5 ** 0.5, 3, "fast", "damped away"),
    ]:
        bands.append({
            "laplacian_eigenvalue": r(lam),
            "multiplicity": mult,
            "damping_per_step": r(1 - lam / 60),
            "name": name,
            "role": role,
        })

    ns = list(range(0, 121, 2))
    P0 = np.ones((12, 12)) / 12.0
    Q = np.eye(12) - P0
    curves = {"n": ns, "share": {"slow": [], "middle": [], "fast": []}}
    ranks = []
    for n in ns:
        Tn = np.linalg.matrix_power(T, 2 * n)
        C = Q @ Tn @ Q
        tr = float(np.trace(C))
        s = 3 * (1 - (5 - 5 ** 0.5) / 60) ** (2 * n)
        m = 5 * 0.9 ** (2 * n)
        f = 3 * (1 - (5 + 5 ** 0.5) / 60) ** (2 * n)
        tot = s + m + f
        curves["share"]["slow"].append(r(s / tot))
        curves["share"]["middle"].append(r(m / tot))
        curves["share"]["fast"].append(r(f / tot))
        if tr > 1e-300:
            w = np.linalg.eigvalsh(12 * C / tr)
            ranks.append({"n": n, "eigs": r(sorted(w.tolist(), reverse=True)[:5], 4)})

    Clim = 4 * (evecs[:, 1:4] @ evecs[:, 1:4].T)
    w = np.linalg.eigvalsh(Clim)
    embed = evecs[:, 1:4] * math.sqrt(4.0)

    return {
        "earned": "exact",
        "operator": "T = I - L_icosahedron/60,  C_n = Q T^(2n) Q,  Q = I - P_0",
        "bands": bands,
        "decay_curves": curves,
        "normalised_limit_eigenvalues": r(sorted(w.tolist(), reverse=True), 4),
        "limit_rank": int((w > 1e-9).sum()),
        "port_embedding_3d": r(embed.tolist(), 5),
        "irrep_decomposition": "12 ports = 1 + 3 + 5 + 3'",
        "receipt": "data/repair_closure/port_gram_completion_bridge_receipt.json",
    }


# ------------------------------------------------------- thermodynamics ---
def block_thermo(run: Path, label: str):
    p = run / "mismatch_trace.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    g = lambda k: np.array([float(x[k]) for x in rows])
    phi, ent, beta = g("phi"), g("record_packet_entropy"), g("beta")
    committed = g("committed_fraction")
    depth = g("modular_depth_mean")
    acts = g("total_repair_actions")
    d = np.diff(ent)
    npatch = int(float(rows[-1]["committed_records"])) or 1
    zero_at = int(np.argmax(phi == 0)) if (phi == 0).any() else None
    return {
        "earned": "measured",
        "label": label,
        "patches": npatch,
        "cycles": [int(x["cycle"]) for x in rows],
        "disagreement": [int(v) for v in phi],
        "record_entropy": r(ent.tolist(), 5),
        "beta": r(beta.tolist(), 4),
        "committed_fraction": r(committed.tolist(), 4),
        "modular_depth_mean": r(depth.tolist(), 5),
        "repair_actions": [int(v) for v in acts],
        "disagreement_monotone": bool((np.diff(phi) <= 0).all()),
        "consensus_cycle": zero_at,
        "entropy_steps": {"up": int((d > 0).sum()), "flat": int((d == 0).sum()),
                          "down": int((d < 0).sum())},
        "negative_step_cycles": [int(i) + 1 for i in np.nonzero(d < 0)[0]],
        "negative_mass_fraction": r(abs(d[d < 0].sum()) / d[d > 0].sum() if (d > 0).any() else 0, 6),
        "entropy_final": r(float(ent[-1]), 6),
        "entropy_max_lnN": r(math.log(npatch), 6),
        "saturation_percent": r(100 * float(ent[-1]) / math.log(npatch), 4),
    }


# --------------------------------------------------------- observer sky ---
def block_screen(run: Path, n_points=1800, n_frames=24):
    p = run / "screen_evolution_frames.npz"
    if not p.exists():
        return None
    z = np.load(p)
    cycles = z["cycles"]
    total = z["field__record_port_entropy"].shape[1]
    rng = np.random.default_rng(20260820)
    idx = np.sort(rng.choice(total, size=min(n_points, total), replace=False))
    fsel = np.linspace(0, len(cycles) - 1, min(n_frames, len(cycles))).astype(int)
    # fibonacci-sphere positions matching the sim's patch family
    i = np.arange(total, dtype=float) + 0.5
    ga = math.pi * (3 - 5 ** 0.5)
    zc = 1 - 2 * i / total
    rad = np.sqrt(np.clip(1 - zc * zc, 0, 1))
    pts = np.stack([np.cos(ga * i) * rad, np.sin(ga * i) * rad, zc], 1)[idx]
    out = {"earned": "measured", "cycles": [int(c) for c in cycles[fsel]],
           "points": r(pts.tolist(), 4), "fields": {}}
    for key, name in [("field__record_port_entropy", "record_entropy"),
                      ("field__local_mismatch_density", "mismatch"),
                      ("field__modular_depth", "modular_depth"),
                      ("field__cumulative_repair_load", "repair_load")]:
        a = z[key][fsel][:, idx]
        q = np.clip(np.round((a + 4) * 24), 0, 255).astype(np.uint8)
        out["fields"][name] = [row.tolist() for row in q]
    out["quantisation"] = "value = code/24 - 4 (per-frame standardised field)"
    return out


def block_harmonic(run: Path):
    p = run / "harmonic_time_trace.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = {"earned": "measured", "cycles": [int(c) for c in z["cycles"]],
           "ell": r(z["ell"].tolist(), 2), "power": {}, "control": {}}
    for f in ["record_port_entropy", "local_mismatch_density", "modular_depth",
              "cumulative_repair_load"]:
        out["power"][f] = r(z[f].sum(1).tolist(), 6)
        ck = f"control__{f}__shuffled_field"
        if ck in z.files:
            out["control"][f] = r(z[ck].sum(1).tolist(), 6)
    return out


# ------------------------------------------------------- standard model ---
def block_standard_model():
    from oph_fpe.defects.z6_matter_grammar_verifier import (
        COMMITTED_BIDEGREE, COMMITTED_CHARGE)
    names = {Fraction(1, 6): ("Q_L", "left quark doublet"),
             Fraction(2, 3): ("u_R", "right up-type quark"),
             Fraction(-1, 3): ("d_R", "right down-type quark"),
             Fraction(-1, 2): ("L", "left lepton doublet"),
             Fraction(-1, 1): ("e_R", "right electron")}
    rows = []
    for i, (q, (c, w)) in enumerate(zip(COMMITTED_CHARGE, COMMITTED_BIDEGREE)):
        Y = Fraction(q, 6)
        conj = Y not in names
        key = -Y if conj else Y
        sym, desc = names.get(key, ("?", "?"))
        rows.append({
            "row": i, "colour_modes": c, "weak_modes": w, "q6Y": q,
            "hypercharge": f"{Y.numerator}/{Y.denominator}" if Y.denominator != 1 else str(Y.numerator),
            "triality": c % 3, "duality": w % 2,
            "descent": (2 * (c % 3) + 3 * (w % 2) + q) % 6,
            "symbol": (sym + "̄") if conj else sym,
            "name": ("anti-" if conj else "") + desc,
            "conjugate": conj,
            "colour_dim": [1, 3, 3, 1][c] if c < 4 else 1,
            "weak_dim": [1, 2, 1][w] if w < 3 else 1,
        })
    return {
        "earned": "exact",
        "rows": rows,
        "charge_binding": "q = -2c + 3w",
        "descent_condition": "2t + 3d + q = 0 (mod 6)",
        "all_rows_descend": all(x["descent"] == 0 for x in rows),
        "gauge_group": "(SU(3) x SU(2) x U(1)) / Z6",
        "note": "colour triality t = c mod 3, weak duality d = w mod 2, q = 6Y",
    }


def block_census():
    from oph_fpe.defects.z6_defect_census import run_census
    c = run_census()
    return {
        "earned": "exact",
        "carrier": {k: c["carrier"][k] for k in ("ports", "seams", "faces", "chord_count")},
        "defect_class_count": c["defect_class_count"],
        "members_total": c["members_total"],
        "sector_changed_by_repair": c["sector_changed_by_repair"],
        "vacuum_multiplicity": c["vacuum"]["multiplicity"],
        "rotation_group": c["rotation_group_receipt"]["order"],
        "gauge_conservation": c["conservation_receipt"]["gauge_conservation"],
        "checked_pairs": c["conservation_receipt"]["checked_pairs"],
    }


# ------------------------------------------------------------ born rule ---
def block_born(qm: Path):
    d = json.load(open(qm / "QM_OBSERVER_VIZ.v1.json"))
    keep = []
    for s in d["scenarios"]:
        e = {"id": s["scenario_id"], "kind": s["kind"]}
        if s["kind"] == "base_context":
            node = s["tree"]["children"][0]
            e.update({"context": node["context"], "counts": node["counts"],
                      "mass": node["mass"], "weights": node["weights"],
                      "base_classes": s["tree"]["class_counts"]})
        elif s["kind"] == "collapse_chain":
            e.update({"context": s["context_sequence"][0] if s["context_sequence"] else None,
                      "events": s["collapse_events"][:4]})
        else:
            e.update({"sequence": s["context_sequence"], "tree_mass": s["tree"]["mass"]})
        keep.append(e)
    return {"earned": "exact-counting", "scenarios": keep,
            "base_population": {"rec0": 111, "rec1": 68, "mass": 179},
            "note": "probabilities are integer count ratios over a finite record ensemble"}



# ------------------------------------------------ observers and H3 chart ---
def block_h3(run: Path):
    p = run / "h3_objects.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    objs = []
    for r in rows:
        try:
            objs.append({
                "id": int(r["object_id"]),
                "xyz": [round(float(r["h3_x"]), 4), round(float(r["h3_y"]), 4), round(float(r["h3_z"]), 4)],
                "observers": int(r["observer_count"]),
                "support": int(r["support_size"]),
                "compactness": round(float(r["h3_compactness_normalized"]), 4),
            })
        except (KeyError, ValueError):
            continue
    rep = run / "observer_chart_object_h3_report.json"
    j = json.load(open(rep)) if rep.exists() else {}
    return {
        "earned": "measured",
        "objects": objs,
        "object_count": j.get("object_count"),
        "shuffled_control_count": j.get("shuffled_localized_object_count"),
        "shuffled_control_p90": j.get("shuffled_localized_object_p90"),
        "localized_fraction": j.get("localized_object_fraction"),
        "median_receipt": j.get("observer_chart_object_h3_median_receipt"),
        "strict_receipt": j.get("observer_chart_object_h3_receipt"),
    }


def block_observers(run: Path):
    p = run / "observer_perspective_rows.csv"
    rep = run / "observer_modular_experience_report.json"
    j = json.load(open(rep)) if rep.exists() else {}
    rows = list(csv.DictReader(open(p))) if p.exists() else []
    per = []
    for r in rows:
        try:
            per.append({
                "id": int(r["observer_id"]),
                "support": int(r["support_patch_count"]),
                "depth": round(float(r["modular_depth_mean"]), 5),
                "load": round(float(r["repair_load_mean"]), 5),
                "mismatch": round(float(r["mismatch_density_mean"]), 6),
            })
        except (KeyError, ValueError):
            continue
    return {
        "earned": "measured",
        "count": j.get("observer_count", len(per)),
        "rows": per[:140],
        "modular_depth_mean": r6(j.get("modular_depth_mean")),
        "modular_depth_std": r6(j.get("modular_depth_std")),
        "gates": j.get("component_gates", {}),
        "blockers": j.get("blockers", []),
        "chart_receipt": j.get("H3_FRAME_FIBER_CHART_RECEIPT"),
    }


def r6(x):
    return None if x is None else round(float(x), 6)


# ------------------------------------------------ bulk contraction (gravity) ---
def block_curvature(run: Path, max_sources=900):
    """The gravity picture: matter sources and the conformal factor they impose."""
    p = run / "universe_timeline/emergent_curved_spacetime_curvature_proxy.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    num = lambda v, d=None: (float(v) if v not in (None, "") else d)
    src = []
    for row in rows:
        cf = num(row.get("local_metric_conformal_factor"))
        if cf is None:
            continue
        src.append({
            "xyz": [round(num(row["x"], 0.0), 4), round(num(row["y"], 0.0), 4),
                    round(num(row["z"], 0.0), 4)],
            "kind": row.get("source_kind", ""),
            "mass": round(num(row.get("mass_proxy"), 0.0), 3),
            "stress": round(num(row.get("stress_energy_proxy"), 0.0), 4),
            "conformal": round(cf, 5),
            "scale": round(num(row.get("emergent_spatial_scale_factor"), 1.0), 5),
            "potential": round(num(row.get("curvature_potential"), 0.0), 5),
            "cycle": num(row.get("cycle")),
        })
    src.sort(key=lambda d: -d["stress"])
    src = src[:max_sources]
    view = run / "universe_timeline/emergent_curved_spacetime.json"
    slices, noncl, extent = [], [], None
    if view.exists():
        v = json.load(open(view))
        slices = [{"i": t.get("sliceIndex"), "cycle": t.get("cycle"),
                   "relative_time": t.get("relativeTime"),
                   "curvature": r(t.get("totalCurvaturePotential"), 4),
                   "source_density": r(t.get("totalSourceDensity"), 4),
                   "events": t.get("eventCount"), "sources": t.get("sourceCount")}
                  for t in v.get("timeSlices", [])]
        noncl = v.get("nonClaims", [])
        extent = v.get("spatialExtent")
    cfs = [d["conformal"] for d in src]
    return {
        "earned": "measured",
        "sources": src,
        "source_count_total": len(rows),
        "time_slices": slices,
        "non_claims": noncl,
        "spatial_extent": extent,
        "conformal_min": r(min(cfs), 5) if cfs else None,
        "conformal_max": r(max(cfs), 5) if cfs else None,
        "note": ("local_metric_conformal_factor below 1 means the bulk contracts there: "
                 "rulers get shorter near mass. That contraction is the curvature."),
    }


# ----------------------------------------------------------- assemble ----
def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2])
    qm = Path(sys.argv[3])
    extra = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    payload = {
        "schema": "oph.physics_first_visualizer.v1",
        "run": {"id": run.name, "path": str(run)},
        "carrier": block_carrier(),
        "dimension": block_dimension(),
        "thermo": block_thermo(run, run.name),
        "screen": block_screen(run),
        "harmonic": block_harmonic(run),
        "standard_model": block_standard_model(),
        "census": block_census(),
        "born": block_born(qm),
        "h3": block_h3(run),
        "curvature": block_curvature(run),
        "observers": block_observers(run),
    }
    if extra and extra.exists():
        payload["thermo_control"] = block_thermo(extra, extra.name)
    for k in [k for k, v in payload.items() if v is None]:
        payload.pop(k)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  {out.stat().st_size/1e6:.2f} MB")
    for k, v in payload.items():
        if isinstance(v, dict) and "earned" in v:
            print(f"  {k:16s} {v['earned']}")


if __name__ == "__main__":
    main()
