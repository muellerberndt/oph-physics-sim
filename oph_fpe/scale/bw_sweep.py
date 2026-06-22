from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import time
import uuid
from typing import Any

from oph_fpe.experiments import load_config
from oph_fpe.scale.bw_array import run_bw_array_config
from oph_fpe.scale.parallel import sweep_parallel_plan


def run_bw_sweep(
    config_paths: list[Path],
    out_dir: Path,
    *,
    seeds: list[int] | None = None,
    workers: int | None = None,
    inner_jobs: int | None = None,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        seed_values = seeds or [int(config.get("seed", 1))]
        for seed in seed_values:
            job_config = deepcopy(config)
            job_config["seed"] = int(seed)
            name = str(job_config.get("name", config_path.stem))
            job_config["run_id"] = f"{_slug(name)}_seed{int(seed)}_{uuid.uuid4().hex[:8]}"
            jobs.append({"config_path": str(config_path), "seed": int(seed), "config": job_config})

    if not jobs:
        return {"jobs": [], "summary_path": None}

    max_workers, planned_inner_jobs = sweep_parallel_plan(
        job_count=len(jobs),
        workers=workers,
        inner_jobs=inner_jobs,
    )
    for job in jobs:
        bw_cfg = dict(job["config"].get("bw", {}) or {})
        bw_cfg["n_jobs"] = int(planned_inner_jobs)
        job["config"]["bw"] = bw_cfg
        cosmology_cfg = dict(job["config"].get("cosmology", {}) or {})
        angular_cfg = dict(cosmology_cfg.get("angular_power", {}) or {})
        if "n_jobs" not in angular_cfg:
            angular_cfg["n_jobs"] = int(planned_inner_jobs)
        cosmology_cfg["angular_power"] = angular_cfg
        job["config"]["cosmology"] = cosmology_cfg
    started = time.time()
    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for job in jobs:
            results.append(_run_one(job["config"], str(out_dir), job["config_path"], job["seed"]))
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=_process_context()) as pool:
            futures = {
                pool.submit(_run_one, job["config"], str(out_dir), job["config_path"], job["seed"]): job
                for job in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: (str(row.get("config_path")), int(row.get("seed", 0)), str(row.get("run_id", ""))))
    summary = {
        "engine": "bw_sweep",
        "workers": max_workers,
        "inner_jobs": planned_inner_jobs,
        "parallelism_policy": "auto_fill_available_cpus" if workers is None and inner_jobs is None else "explicit_or_partially_explicit",
        "elapsed_seconds": time.time() - started,
        "job_count": len(jobs),
        "results": results,
    }
    summary["large_run_readiness_summary"] = _large_run_readiness_summary(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"bw_sweep_{int(started)}.json"
    import json

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _run_one(config: dict[str, Any], out_dir: str, config_path: str, seed: int) -> dict[str, Any]:
    try:
        result = run_bw_array_config(config, Path(out_dir))
        return {
            "ok": True,
            "config_path": config_path,
            "seed": seed,
            **result,
        }
    except Exception as exc:  # pragma: no cover - exercised only on worker failure
        return {
            "ok": False,
            "config_path": config_path,
            "seed": seed,
            "error": repr(exc),
        }


def _process_context() -> mp.context.BaseContext | None:
    methods = mp.get_all_start_methods()
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    if "spawn" in methods:
        return mp.get_context("spawn")
    return None


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.lower()).strip("_")


def _large_run_readiness_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("ok", False)]
    lane_counts: dict[str, Counter[str]] = {}
    blocker_counts: Counter[str] = Counter()
    recommended_counts: Counter[str] = Counter()
    state_bw_worthwhile = 0
    any_scale_candidate = 0
    for row in ok_results:
        readiness = row.get("large_run_readiness") or {}
        recommended_counts[str(readiness.get("recommended_large_run_lane", "unknown"))] += 1
        if readiness.get("state_bw_expensive_run_worthwhile", False):
            state_bw_worthwhile += 1
        if readiness.get("any_scale_candidate", False):
            any_scale_candidate += 1
        for blocker in readiness.get("blockers", []):
            blocker_counts[str(blocker)] += 1
        for lane_name, lane in (readiness.get("lanes") or {}).items():
            lane_counts.setdefault(str(lane_name), Counter())[str(lane.get("status", "unknown"))] += 1
    return {
        "mode": "large_run_preflight_readiness_summary",
        "job_count": len(results),
        "ok_job_count": len(ok_results),
        "failed_job_count": len(results) - len(ok_results),
        "recommended_large_run_lanes": dict(sorted(recommended_counts.items())),
        "any_scale_candidate_count": int(any_scale_candidate),
        "state_bw_expensive_run_worthwhile_count": int(state_bw_worthwhile),
        "all_ok_jobs_state_bw_worthwhile": bool(ok_results and state_bw_worthwhile == len(ok_results)),
        "lane_status_counts": {
            lane_name: dict(sorted(counts.items()))
            for lane_name, counts in sorted(lane_counts.items())
        },
        "top_blockers": [
            {"blocker": blocker, "count": int(count)}
            for blocker, count in blocker_counts.most_common(12)
        ],
        "claim_boundary": (
            "Aggregate of per-run preflight readiness reports. This is routing metadata for run design, "
            "not a physics receipt."
        ),
    }
