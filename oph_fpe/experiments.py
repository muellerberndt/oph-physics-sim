from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from oph_fpe.bulk import (
    dimension_report,
    final_modular_embedding,
    graph_distance_matrix,
    laplacian_embedding,
    modular_lift_dimension_report,
)
from oph_fpe.core import PatchNet, build_patch_graph, pixel_scale_from_config, screen_microphysics_from_config
from oph_fpe.core.records import RecordEvent, update_records
from oph_fpe.defects import DefectTracker, scan_holonomy_defects
from oph_fpe.dynamics import (
    RepairEvent,
    RepairKernel,
    apply_modular_flow,
    beta_at,
    collect_modular_sample,
    dispatch_configured_kernels,
    kernel_dispatch_manifest_summary,
)
from oph_fpe.evidence import RunBundle, verify_local_law
from oph_fpe.groups import get_group


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 1))
    pixel_scale = pixel_scale_from_config(config)
    group = get_group(str(config.get("group", {}).get("name", "Z2")))
    graph = build_patch_graph(config.get("graph", {}), seed=seed)
    screen_microphysics = screen_microphysics_from_config(config, graph.number_of_nodes(), graph.number_of_edges())
    initial = config.get("initial_condition", "maxent_hot")
    if initial == "synchronized":
        net = PatchNet.synchronized(graph, group)
    else:
        net = PatchNet.random(graph, group, seed=seed)

    _inject_defects(net, config.get("defects", {}), seed=seed + 17)

    run_id = config.get("run_id") or _run_id(config.get("name", "oph_fpe"))
    bundle = RunBundle(out_dir, run_id)
    bundle.write_config(config)
    bundle.write_json("pixel_scale.json", pixel_scale.as_jsonable())
    bundle.write_json("pixel_report.json", pixel_scale.as_jsonable())
    bundle.write_json("screen_microphysics.json", screen_microphysics.as_jsonable())
    bundle.write_json("graph_initial.json", _graph_json(graph))
    output = config.get("output", {})
    write_full_state = bool(output.get("write_full_state", graph.number_of_nodes() <= 4096))
    if write_full_state:
        bundle.write_json("state_initial.json", net.as_jsonable())
    else:
        bundle.write_json("state_initial_summary.json", _state_summary(net))

    dynamics = config.get("dynamics", {})
    cycles = int(dynamics.get("cycles", 64))
    repairs_per_cycle = int(dynamics.get("repairs_per_cycle", max(1, graph.number_of_nodes() // 2)))
    commit_cycles = int(dynamics.get("record_commit_cycles", 4))
    kernel = RepairKernel(
        mode=str(dynamics.get("repair", "local_best_plus_metropolis_hot_phase")),
        hot_metropolis=bool(dynamics.get("hot_metropolis", True)),
        seed=seed + 23,
    )

    repair_events: list[RepairEvent] = []
    record_events: list[RecordEvent] = []
    mismatch_trace: list[dict[str, Any]] = []
    tracker = DefectTracker()
    defect_scan_interval = int(config.get("observables", {}).get("defect_scan_interval", 4))
    modular_config = config.get("modular_flow", {})
    modular_trace_interval = int(modular_config.get("trace_interval", 4))
    modular_depth_samples: list[dict[int, float]] = []

    for cycle in range(cycles):
        beta = beta_at(dynamics.get("beta_schedule", {}), cycle, cycles)
        for _ in range(repairs_per_cycle):
            repair_events.append(kernel.step(net, cycle, beta))
        apply_modular_flow(net, modular_config, cycle)
        if modular_config.get("enabled", False) and (
            cycle % modular_trace_interval == 0 or cycle == cycles - 1
        ):
            modular_depth_samples.append(collect_modular_sample(net))
        record_events.extend(update_records(net, cycle, commit_cycles))
        defects = []
        if cycle % defect_scan_interval == 0 or cycle == cycles - 1:
            defects = scan_holonomy_defects(net)
            tracker.update(cycle, defects)
        mismatch_trace.append(
            {
                "cycle": cycle,
                "beta": beta,
                "phi": net.total_phi(),
                "mismatch_edges": len(net.mismatch_edges()),
                "record_entropy": _record_entropy(net),
                "committed_records": sum(1 for state in net.states.values() if state.record is not None),
                "defect_count": len(defects),
            }
        )

    if config.get("observables", {}).get("distance_source") == "modular_lift":
        dimensions = modular_lift_dimension_report(
            net,
            modular_depth_samples,
            config.get("observables", {}).get("modular_lift", {}),
            seed=seed + 313,
        )
        embedding = final_modular_embedding(net)
    else:
        nodes, distances = graph_distance_matrix(graph)
        dimensions = dimension_report(graph, distances)
        embedding = laplacian_embedding(graph, dimensions=3)
    worldlines = tracker.worldlines()
    receipts = verify_local_law(repair_events, record_events, commit_cycles)
    controls = _run_controls(config, graph, seed)

    bundle.write_json("graph_final.json", _graph_json(graph))
    if write_full_state:
        bundle.write_json("state_final.json", net.as_jsonable())
    else:
        bundle.write_json("state_final_summary.json", _state_summary(net))
    bundle.write_mismatch_trace(mismatch_trace)
    bundle.write_repair_events(repair_events)
    bundle.write_json(
        "record_events.json",
        [
            {"cycle": event.cycle, "node": event.node, "stable_count": event.stable_count}
            for event in record_events
        ],
    )
    bundle.write_json("defect_worldlines.json", worldlines)
    bundle.write_json("bulk_embedding.json", embedding)
    bundle.write_json("dimension_report.json", dimensions)
    if modular_depth_samples and bool(output.get("write_modular_depth_trace", graph.number_of_nodes() <= 4096)):
        bundle.write_json("modular_depth_trace.json", modular_depth_samples)
    bundle.write_json(
        "cosmology_observables.json",
        _cosmology_proxy(
            mismatch_trace,
            dimensions,
            worldlines,
            pixel_scale.as_jsonable(),
            screen_microphysics.as_jsonable(),
        ),
    )
    bundle.write_json("controls/control_report.json", controls)
    bundle.write_jsonl("verifier_receipts.jsonl", receipts)
    kernel_dispatch = dispatch_configured_kernels(config, bundle.path, engine="patchnet")
    manifest = {
        "run_id": run_id,
        "name": config.get("name"),
        "claim_boundary": config.get(
            "claim_boundary",
            "CPU MVP receipt: patch repair, records, holonomy defects, and dimension estimators only.",
        ),
        "patch_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "group": group.name,
        "pixel_scale": pixel_scale.as_jsonable(),
        "oph_constants": pixel_scale.constants.as_jsonable(),
        "screen_microphysics": screen_microphysics.as_jsonable(),
        "screen_units": screen_microphysics.as_jsonable()["screen_units"],
        "cycles": cycles,
        "final_phi": net.total_phi(),
        "bundle_files": sorted(path.name for path in bundle.path.iterdir()),
    }
    result = {"run_id": run_id, "path": str(bundle.path), "final_phi": net.total_phi(), "dimensions": dimensions}
    if kernel_dispatch:
        summary = kernel_dispatch_manifest_summary(kernel_dispatch)
        manifest["kernel_dispatch"] = summary
        result["kernel_dispatch"] = summary
    bundle.write_manifest(manifest)
    return result


def _inject_defects(net: PatchNet, config: dict[str, Any], seed: int) -> None:
    count = int(config.get("count", 0))
    if count <= 0:
        return
    rng = np.random.default_rng(seed)
    cycles = nx.cycle_basis(net.graph)
    if not cycles:
        return
    nonidentity = [element for element in net.group.elements if element != net.group.identity]
    for _ in range(count):
        cycle = cycles[int(rng.integers(0, len(cycles)))]
        left, right = cycle[0], cycle[1]
        twist = nonidentity[int(rng.integers(0, len(nonidentity)))]
        current = net.states[left].gauges[right]
        net.set_directed_gauge(left, right, net.group.multiply(current, twist))


def _record_entropy(net: PatchNet) -> float:
    counts: dict[str, int] = {}
    for state in net.states.values():
        if state.record is None:
            continue
        key = repr(state.record)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def _cosmology_proxy(
    trace: list[dict[str, Any]],
    dimensions: dict[str, Any],
    worldlines: list[dict[str, Any]],
    pixel_scale: dict[str, Any],
    screen_microphysics: dict[str, Any],
) -> dict[str, Any]:
    if not trace:
        return {}
    phi0 = max(float(trace[0]["phi"]), 1e-9)
    final_phi = float(trace[-1]["phi"])
    return {
        "Phi_initial": phi0,
        "Phi_final": final_phi,
        "Phi_drop_fraction": (phi0 - final_phi) / phi0,
        "defect_density_final": len(worldlines) / max(1, len(trace)),
        "dimension_estimates": dimensions,
        "pixel_scale": pixel_scale,
        "screen_microphysics": screen_microphysics,
        "modular_flow_status": "regulator_side_surrogate" if dimensions.get("distance_source") == "modular_lift_record_history" else "not_used",
        "boltzmann_adapter_status": "not_implemented_mvp",
    }


def _run_controls(config: dict[str, Any], graph: nx.Graph, seed: int) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    requested = config.get("controls", [])
    if "random_graph" in requested:
        degree = max(2, int(round(sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes()))))
        if degree * graph.number_of_nodes() % 2:
            degree += 1
        degree = min(degree, graph.number_of_nodes() - 1)
        random_graph = nx.random_regular_graph(degree, graph.number_of_nodes(), seed=seed + 101)
        _, distances = graph_distance_matrix(random_graph)
        controls["random_graph"] = dimension_report(random_graph, distances)
    if "no_repair" in requested:
        controls["no_repair"] = {"expected": "Phi should not be driven down", "implemented": "pending_control_runner"}
    if "shuffled_interfaces" in requested:
        controls["shuffled_interfaces"] = {
            "expected": "record-derived geometry should destabilize",
            "implemented": "pending_record_distance_runner",
        }
    if "wrong_orientation" in requested:
        controls["wrong_orientation"] = {
            "expected": "S3 nonabelian holonomy signatures should change under reversed cycle order",
            "implemented": "covered_by_unit_test",
        }
    return controls


def _graph_json(graph: nx.Graph) -> dict[str, Any]:
    return {
        "nodes": list(graph.nodes),
        "node_attrs": {
            str(node): {"screen_xyz": graph.nodes[node].get("screen_xyz")}
            for node in graph.nodes
            if "screen_xyz" in graph.nodes[node]
        },
        "edges": [
            {"left": left, "right": right, "weight": float(graph.edges[left, right].get("weight", 1.0))}
            for left, right in sorted(graph.edges)
        ],
    }


def _state_summary(net: PatchNet) -> dict[str, Any]:
    depths = np.array([state.modular_depth for state in net.states.values()], dtype=float)
    loads = np.array([state.repair_load for state in net.states.values()], dtype=float)
    return {
        "group": net.group.name,
        "nodes": net.graph.number_of_nodes(),
        "edges": net.graph.number_of_edges(),
        "phi": net.total_phi(),
        "committed_records": sum(1 for state in net.states.values() if state.record is not None),
        "modular_depth": _array_summary(depths),
        "repair_load": _array_summary(loads),
    }


def _array_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _run_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).lower()).strip("_")
    return f"{slug}_{int(time.time())}"
