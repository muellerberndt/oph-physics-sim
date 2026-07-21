"""Einstein-branch convergence ladder: 16k, 64k, 256k carriers.

Runs the fixed v1 capture (cross-observer reads, spanning snapshots, held-out
geometry transport) at three federation sizes with constant observer density
(one observer per 128 carriers, support width 96), measures the held-out
event-form inertia and cone margin, the coupling spread, and the
cross-observer link count at each rung, and stores compressed proof
artifacts: per rung, a gzipped npz with the event chart, pair-class samples,
fitted form, and readout vectors, plus a json summary, all hash-bound in a
manifest.  Deterministic throughout; pair classes larger than the declared
cap are subsampled by a seeded stride recorded in the artifact.

Usage:
    .venv/bin/python scripts/einstein_convergence_ladder.py [output_dir]

The verdict language is measured, never presumed: the table reports whatever
inertia, margin, and spread the dynamics produces at each rung.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.event_manifold_producer import (
    _event_table,
    _fit_quadratic_form,
    _spectral_embedding,
    _event_chart,
)
from oph_fpe.bulk.modular_normalization_producer import (
    _axis_frame,
    _depth_generator,
    cap_interior_state,
    geometric_flow_rate,
)
from oph_fpe.bulk.stress_coupling_producer import cap_stress_flux

PAIR_CAP = 300_000
# Chain depth (observer_samples) is held constant across rungs so the
# timelike direction of each observer chain has fixed depth; the observer
# count grows with the federation instead.
RUNGS = (
    {"carrier_count": 16_384, "observer_count": 128, "observer_samples": 6,
     "observer_support_size": 96},
    {"carrier_count": 65_536, "observer_count": 256, "observer_samples": 6,
     "observer_support_size": 96},
    # Support width scales with anchor spacing at the top rung so the
    # cross-observer coupling density does not dilute with federation size.
    {"carrier_count": 262_144, "observer_count": 512, "observer_samples": 6,
     "observer_support_size": 384},
)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _pair_classes_capped(table: dict) -> dict:
    causal: list[tuple[int, int]] = []
    spacelike: list[tuple[int, int]] = []
    count = table["count"]
    reachable = table["reachable"]
    for i in range(count):
        reach_i = reachable[i]
        for j in range(i + 1, count):
            if i in reachable[j] or j in reach_i:
                causal.append((i, j))
            else:
                spacelike.append((i, j))
    subsample = {}
    for name, pairs in (("causal", causal), ("spacelike", spacelike)):
        stride = max(1, len(pairs) // PAIR_CAP)
        subsample[name] = pairs[::stride][:PAIR_CAP]
        subsample[f"{name}_total"] = len(pairs)
        subsample[f"{name}_stride"] = stride
    return subsample


def run_rung(config: dict) -> tuple[dict, dict]:
    carriers = config["carrier_count"]
    cfg = {
        "carrier_count": carriers,
        "cycles": 16,
        "seed": 20260751,
        "observer_count": config["observer_count"],
        "observer_support_size": config["observer_support_size"],
        "observer_samples": config["observer_samples"],
        "observer_cross_reads": True,
        "snapshot_coverage": "spanning",
        "geometry_transport": "held_out_flow",
    }
    t0 = time.time()
    capture = capture_physical_source(cfg)
    capture_seconds = time.time() - t0

    events = capture["postrun_capture"]["semantic_events"]
    key_to_event = {e["event_key"]: e for e in events}
    cross = sum(
        1
        for edge in capture["postrun_capture"]["raw_ancestry_relations"]
        if (key_to_event.get(edge["child_event_id"]) or {}).get("observer_token")
        != (key_to_event.get(edge["parent_event_id"]) or {}).get("observer_token")
    )

    table = _event_table(capture)
    pairs = _pair_classes_capped(table)
    touched = sorted({c for f in table["footprints"] for c in f})
    index_of = {c: i for i, c in enumerate(touched)}
    sub = np.zeros((len(touched), len(touched)))
    for row in capture["postrun_capture"]["raw_overlap_relations"]:
        li = index_of.get(int(row["left_carrier_id"].rsplit("-", 1)[1]))
        ri = index_of.get(int(row["right_carrier_id"].rsplit("-", 1)[1]))
        if li is not None and ri is not None:
            sub[li, ri] = sub[ri, li] = 1.0
    embedding_small = _spectral_embedding(sub)
    embedding = np.zeros((carriers, 3))
    for c, i in index_of.items():
        embedding[c] = embedding_small[i]
    chart = _event_chart(table, embedding)
    fit = _fit_quadratic_form(
        chart, {"causal": pairs["causal"], "spacelike": pairs["spacelike"]}
    )

    samples = np.stack(
        [
            np.asarray(r["full_port_state"])
            for s in capture["source_artifacts"]["dynamics"][
                "record_state_snapshots"
            ]
            for r in s["carrier_rows"]
        ]
    )
    geometry = geometric_flow_rate(capture)
    coefficients = []
    for axis in range(6):
        state = cap_interior_state(samples, _axis_frame(axis))
        if state["faithful"]:
            depth = _depth_generator(axis)
            coefficients.append(
                float(np.tensordot(state["modular_hamiltonian"], depth))
                / float(np.tensordot(depth, depth))
                / geometry["rate"]
            )
    flux = np.asarray([cap_stress_flux(capture, axis) for axis in range(6)])
    spread = float((flux.max() - flux.min()) / abs(float(np.median(flux))))

    summary = {
        "config": cfg,
        "capture_sha256": capture["capture_sha256"],
        "capture_seconds": round(capture_seconds, 1),
        "event_count": table["count"],
        "cross_observer_edges": cross,
        "causal_pairs_total": pairs["causal_total"],
        "spacelike_pairs_total": pairs["spacelike_total"],
        "pair_cap": PAIR_CAP,
        "causal_stride": pairs["causal_stride"],
        "spacelike_stride": pairs["spacelike_stride"],
        "held_out_inertia": fit.get("inertia"),
        "cone_margin": fit.get("cone_margin"),
        "eigenvalues": fit.get("eigenvalues"),
        "normalization_coefficients": [round(c, 6) for c in coefficients],
        "coupling_spread": round(spread, 6),
        "raw_moment_min_eig": float(
            np.linalg.eigvalsh(samples.T @ samples / len(samples)).min()
        ),
    }
    arrays = {
        "chart": chart,
        "causal_pairs": np.asarray(pairs["causal"], dtype=np.int32),
        "spacelike_pairs": np.asarray(pairs["spacelike"], dtype=np.int32),
        "flux": flux,
        "coefficients": np.asarray(coefficients),
        "form_eigenvalues": np.asarray(fit.get("eigenvalues", [])),
    }
    return summary, arrays


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/einstein_convergence")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "oph.einstein-convergence-ladder.v1", "rungs": []}
    for config in RUNGS:
        carriers = config["carrier_count"]
        print(f"=== rung {carriers} ===", flush=True)
        summary, arrays = run_rung(config)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        compressed = gzip.compress(buffer.getvalue(), mtime=0)
        artifact = out / f"rung_{carriers}.npz.gz"
        artifact.write_bytes(compressed)
        summary_path = out / f"rung_{carriers}.json"
        summary_bytes = (
            json.dumps(summary, sort_keys=True, indent=1) + "\n"
        ).encode("utf-8")
        summary_path.write_text(summary_bytes.decode("utf-8"), encoding="utf-8")
        manifest["rungs"].append(
            {
                "carrier_count": carriers,
                "summary": summary_path.name,
                "summary_sha256": _sha(summary_bytes),
                "artifact": artifact.name,
                "artifact_sha256": _sha(compressed),
                "held_out_inertia": summary["held_out_inertia"],
                "cone_margin": summary["cone_margin"],
                "coupling_spread": summary["coupling_spread"],
                "cross_observer_edges": summary["cross_observer_edges"],
            }
        )
        print(
            f"rung {carriers}: events={summary['event_count']} "
            f"cross={summary['cross_observer_edges']} "
            f"inertia={summary['held_out_inertia']} "
            f"margin={round(summary['cone_margin'], 4) if summary['cone_margin'] is not None else None} "
            f"spread={summary['coupling_spread']}",
            flush=True,
        )
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    ).encode("utf-8")
    (out / "manifest.json").write_text(
        manifest_bytes.decode("utf-8"), encoding="utf-8"
    )
    print("LADDER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
