#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oph_fpe.small_universe.model import (  # noqa: E402
    SmallUniverse,
    State,
    all_root_pinned_states,
    build_icosa12_universe,
    cycle_holonomies,
    edge_key,
    globally_consistent,
    manifest,
    phi,
    stable_hash,
    state_id,
)


DEFAULT_CONFIG = ROOT / "configs/sou_v1_icosa12.yml"
REPAIR_BACKEND = "parent_copy_calibration_v1"
CLAIM_BOUNDARY = (
    "Exact fixed-cutoff finite consensus and holonomy-obstruction calibration. "
    "This receipt does not establish endogenous modular flow, the BW 2pi coefficient, "
    "Lorentz symmetry, H3 bulk, particles, or cosmology."
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_from_config(
    config_path: Path = DEFAULT_CONFIG,
    *,
    out_dir: Path,
    seed: int | None = None,
    schedule_replays: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_seed = int(seed if seed is not None else config.get("seed", 20260620))
    replays = int(schedule_replays if schedule_replays is not None else config.get("schedule_replays", 16))
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_universe = build_icosa12_universe(run_seed, frustrate=False)
    control_universe = build_icosa12_universe(run_seed, frustrate=True)
    exact = audit_universe(exact_universe, seed=run_seed, schedule_replays=replays, branch="exact_consensus")
    control = audit_universe(control_universe, seed=run_seed, schedule_replays=replays, branch="frustrated_control")
    bundle = {
        "schema": "small_oph_universe_evidence_bundle_v2",
        "config": str(config_path),
        "seed": run_seed,
        "schedule_replays": replays,
        "exact_consensus": _receipt_summary(exact),
        "frustrated_control": _receipt_summary(control),
        "bundle_receipt": bool(
            exact["FINITE_CONSENSUS_THEOREM_RECEIPT"] and control["HOLONOMY_OBSTRUCTION_RECEIPT"]
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    bundle["bundle_sha256"] = stable_hash(bundle)

    files: dict[str, Any] = {
        "MANIFEST.json": {
            "schema": "small_oph_universe_run_manifest_v1",
            "exact_consensus": manifest(exact_universe, repair_backend=REPAIR_BACKEND, seed=run_seed),
            "frustrated_control": manifest(control_universe, repair_backend=REPAIR_BACKEND, seed=run_seed),
            "bundle_sha256": bundle["bundle_sha256"],
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "source_hashes.json": source_hashes(config_path),
        "cycle_holonomy.json": {
            "exact_consensus": exact["cycle_holonomies"],
            "frustrated_control": control["cycle_holonomies"],
        },
        "exact_consensus_receipt.json": exact,
        "frustrated_control_receipt.json": control,
        "small_oph_universe_evidence.json": bundle,
    }
    for filename, payload in files.items():
        write_json(out_dir / filename, payload)

    shutil.copyfile(config_path, out_dir / "config.yml")
    write_jsonl(out_dir / "all_states.jsonl", exact["state_rows"] + control["state_rows"])
    write_jsonl(out_dir / "repair_transition_table.jsonl", exact["transition_rows"] + control["transition_rows"])
    write_jsonl(out_dir / "schedule_replays.jsonl", exact["schedule_rows"] + control["schedule_rows"])
    write_non_claims(out_dir / "NON_CLAIMS.md")
    write_checksums(out_dir)

    # Drop heavy rows from the returned in-memory object; they are present on disk.
    for report in (exact, control):
        report.pop("state_rows", None)
        report.pop("transition_rows", None)
        report.pop("schedule_rows", None)
    return bundle


def audit_universe(
    universe: SmallUniverse,
    *,
    seed: int,
    schedule_replays: int,
    branch: str,
) -> dict[str, Any]:
    schedules = fair_schedules(universe, seed, schedule_replays)
    canonical_schedule = tuple(sorted(universe.parent, key=lambda node: (universe.depth[node], node)))
    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    unique_terminal_states: set[State] = set()
    strict_descent_violations = 0
    phi_increase_violations = 0
    disjoint_commutation_violations = 0
    local_diamond_violations = 0
    schedule_confluence_violations = 0
    repair_completeness_violations = 0
    terminal_state_count = 0
    globally_consistent_terminal_count = 0
    total_enabled = 0
    max_steps = 0

    states = list(all_root_pinned_states(universe))
    for state_index, state in enumerate(states):
        enabled = independent_enabled_repairs(universe, state)
        total_enabled += len(enabled)
        terminal = not enabled
        consistent = globally_consistent(universe, state)
        if terminal:
            terminal_state_count += 1
            globally_consistent_terminal_count += int(consistent)
        state_rows.append(
            {
                "branch": branch,
                "state_index": state_index,
                "state_id": state_id(state),
                "state": list(state),
                "phi": int(phi(universe, state)),
                "enabled_repair_count": len(enabled),
                "terminal": terminal,
                "globally_consistent": consistent,
            }
        )

        before_phi = phi(universe, state)
        for node in enabled:
            next_state = independent_apply_repair(universe, state, node)
            after_phi = phi(universe, next_state)
            delta = after_phi - before_phi
            if delta >= 0:
                strict_descent_violations += 1
            if delta > 0:
                phi_increase_violations += 1
            transition_rows.append(
                {
                    "branch": branch,
                    "state_index": state_index,
                    "state_id": state_id(state),
                    "node": int(node),
                    "parent": int(universe.parent[node]),
                    "next_state_id": state_id(next_state),
                    "next_state": list(next_state),
                    "phi_before": int(before_phi),
                    "phi_after": int(after_phi),
                    "delta_phi": int(delta),
                    "strict_descent": delta < 0,
                }
            )

        for i, node_a in enumerate(enabled):
            for node_b in enabled[i + 1 :]:
                a_then_b = independent_apply_repair(
                    universe,
                    independent_apply_repair(universe, state, node_a),
                    node_b,
                )
                b_then_a = independent_apply_repair(
                    universe,
                    independent_apply_repair(universe, state, node_b),
                    node_a,
                )
                if _repairs_disjoint(universe, node_a, node_b) and a_then_b != b_then_a:
                    disjoint_commutation_violations += 1
                normal_a, _ = normalize(universe, independent_apply_repair(universe, state, node_a), canonical_schedule)
                normal_b, _ = normalize(universe, independent_apply_repair(universe, state, node_b), canonical_schedule)
                if normal_a != normal_b:
                    local_diamond_violations += 1

        terminals_for_state: set[State] = set()
        for schedule_index, schedule in enumerate(schedules):
            normal, steps = normalize(universe, state, schedule)
            terminals_for_state.add(normal)
            unique_terminal_states.add(normal)
            max_steps = max(max_steps, steps)
            schedule_rows.append(
                {
                    "branch": branch,
                    "state_index": state_index,
                    "state_id": state_id(state),
                    "schedule_index": schedule_index,
                    "schedule": list(schedule),
                    "terminal_state_id": state_id(normal),
                    "terminal_state": list(normal),
                    "steps": int(steps),
                    "terminal_phi": int(phi(universe, normal)),
                }
            )
        if len(terminals_for_state) != 1:
            schedule_confluence_violations += 1

    holonomies = cycle_holonomies(universe)
    zero_holonomy = all(row["holonomy_z2"] == 0 for row in holonomies)
    consistent_states = [state for state in states if globally_consistent(universe, state)]
    terminal_normal_form = next(iter(unique_terminal_states)) if len(unique_terminal_states) == 1 else None
    for state in states:
        terminal = not independent_enabled_repairs(universe, state)
        if zero_holonomy and terminal != globally_consistent(universe, state):
            repair_completeness_violations += 1

    exact_receipt = bool(
        zero_holonomy
        and strict_descent_violations == 0
        and phi_increase_violations == 0
        and disjoint_commutation_violations == 0
        and local_diamond_violations == 0
        and schedule_confluence_violations == 0
        and repair_completeness_violations == 0
        and len(unique_terminal_states) == 1
        and len(consistent_states) == 1
        and terminal_normal_form is not None
        and phi(universe, terminal_normal_form) == 0
    )
    obstruction_receipt = bool(
        (not zero_holonomy)
        and len(consistent_states) == 0
        and terminal_normal_form is not None
        and phi(universe, terminal_normal_form) > 0
        and not globally_consistent(universe, terminal_normal_form)
    )

    report = {
        "schema": "small_oph_universe_exact_consensus_receipt_v2",
        "branch": branch,
        "seed": seed,
        "repair_backend": REPAIR_BACKEND,
        "state_count_exhaustively_enumerated": len(states),
        "enabled_repair_events_checked": total_enabled,
        "schedule_replays_per_state": schedule_replays,
        "strict_descent_violation_count": strict_descent_violations,
        "accepted_phi_increase_violation_count": phi_increase_violations,
        "disjoint_commutation_violation_count": disjoint_commutation_violations,
        "local_diamond_violation_count": local_diamond_violations,
        "schedule_confluence_violation_count": schedule_confluence_violations,
        "local_diamond_or_schedule_confluence_violation_count": (
            local_diamond_violations + schedule_confluence_violations
        ),
        "repair_completeness_violation_count": repair_completeness_violations,
        "terminal_state_count_in_state_space": terminal_state_count,
        "globally_consistent_terminal_count": globally_consistent_terminal_count,
        "global_consistent_state_count": len(consistent_states),
        "unique_terminal_normal_form_count": len(unique_terminal_states),
        "terminal_normal_form": list(terminal_normal_form) if terminal_normal_form is not None else None,
        "terminal_phi": int(phi(universe, terminal_normal_form)) if terminal_normal_form is not None else None,
        "maximum_accepted_repairs_to_normal_form": max_steps,
        "cycle_basis_count": len(holonomies),
        "nonzero_holonomy_cycle_count": sum(int(row["holonomy_z2"] != 0) for row in holonomies),
        "cycle_holonomies": holonomies,
        "FINITE_CONSENSUS_THEOREM_RECEIPT": exact_receipt,
        "HOLONOMY_OBSTRUCTION_RECEIPT": obstruction_receipt,
        "claim_boundary": CLAIM_BOUNDARY,
        "state_rows": state_rows,
        "transition_rows": transition_rows,
        "schedule_rows": schedule_rows,
    }
    report["receipt_sha256"] = stable_hash(
        {k: v for k, v in report.items() if k not in {"state_rows", "transition_rows", "schedule_rows"}}
    )
    return report


def independent_enabled_repairs(universe: SmallUniverse, state: State) -> list[int]:
    out: list[int] = []
    for node, parent in sorted(universe.parent.items()):
        expected = state[parent] ^ universe.offsets[edge_key(parent, node)]
        if state[node] != expected:
            out.append(node)
    return out


def independent_apply_repair(universe: SmallUniverse, state: State, node: int) -> State:
    parent = universe.parent[node]
    expected = state[parent] ^ universe.offsets[edge_key(parent, node)]
    if state[node] == expected:
        return state
    next_state = list(state)
    next_state[node] = expected
    return tuple(next_state)


def normalize(universe: SmallUniverse, state: State, schedule: Iterable[int], *, max_sweeps: int = 32) -> tuple[State, int]:
    current = state
    steps = 0
    order = tuple(schedule)
    for _ in range(max_sweeps):
        changed = False
        for node in order:
            if node == universe.root:
                continue
            next_state = independent_apply_repair(universe, current, node)
            if next_state != current:
                current = next_state
                steps += 1
                changed = True
        if not changed:
            return current, steps
    raise RuntimeError("small-universe normalization did not terminate")


def fair_schedules(universe: SmallUniverse, seed: int, count: int) -> list[tuple[int, ...]]:
    nodes = sorted(universe.parent)
    schedules: list[tuple[int, ...]] = [
        tuple(nodes),
        tuple(reversed(nodes)),
        tuple(sorted(nodes, key=lambda node: (universe.depth[node], node))),
        tuple(sorted(nodes, key=lambda node: (-universe.depth[node], node))),
    ]
    rng = random.Random(seed + 991)
    while len(schedules) < count:
        row = list(nodes)
        rng.shuffle(row)
        schedules.append(tuple(row))
    return schedules[:count]


def source_hashes(config_path: Path) -> dict[str, Any]:
    paths = {
        "config": config_path,
        "verifier": Path(__file__).resolve(),
        "model": ROOT / "oph_fpe/small_universe/model.py",
        "calibration_repair_generator": ROOT / "oph_fpe/small_universe/oph_repair.py",
    }
    return {
        "schema": "small_oph_universe_source_hashes_v1",
        "hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
            if path.exists()
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=list) + "\n")


def write_non_claims(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Non-Claims",
                "",
                "This small-universe evidence bundle is an exact fixed-cutoff calibration harness.",
                "",
                "It does not claim:",
                "",
                "- endogenous modular flow;",
                "- inferred BW/KMS 2pi normalization;",
                "- finite Lorentz theorem instantiation;",
                "- H3 or 3+1D observer experience;",
                "- strict neutral bulk;",
                "- particles or scattering;",
                "- physical CMB, matter power, or Planck likelihood predictions.",
                "",
                "Its positive claim is limited to exact Z2 finite consensus on the unfrustrated 12-patch fixture",
                "and exact holonomy obstruction detection on the one-edge frustrated control.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_checksums(out_dir: Path) -> None:
    rows: list[str] = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256.txt"}):
        rows.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    payload = "\n".join(rows) + "\n"
    (out_dir / "SHA256SUMS").write_text(payload, encoding="utf-8")
    (out_dir / "SHA256.txt").write_text(payload, encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repairs_disjoint(universe: SmallUniverse, node_a: int, node_b: int) -> bool:
    return (
        universe.parent.get(node_a) != node_b
        and universe.parent.get(node_b) != node_a
        and not universe.graph.has_edge(node_a, node_b)
    )


def _receipt_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "branch",
        "state_count_exhaustively_enumerated",
        "enabled_repair_events_checked",
        "schedule_replays_per_state",
        "strict_descent_violation_count",
        "accepted_phi_increase_violation_count",
        "disjoint_commutation_violation_count",
        "local_diamond_or_schedule_confluence_violation_count",
        "repair_completeness_violation_count",
        "global_consistent_state_count",
        "unique_terminal_normal_form_count",
        "terminal_phi",
        "nonzero_holonomy_cycle_count",
        "FINITE_CONSENSUS_THEOREM_RECEIPT",
        "HOLONOMY_OBSTRUCTION_RECEIPT",
        "receipt_sha256",
    ]
    return {key: report.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify exact Small OPH Universe v1 consensus receipts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--schedule-replays", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bundle = run_from_config(
        args.config,
        out_dir=args.out,
        seed=args.seed,
        schedule_replays=args.schedule_replays,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0 if bundle["bundle_receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
