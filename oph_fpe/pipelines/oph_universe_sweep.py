from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
import time
import uuid
from typing import Any

from oph_fpe.experiments import load_config
from oph_fpe.pipelines.oph_universe import run_oph_universe_pipeline
from oph_fpe.scale.parallel import sweep_parallel_plan


def run_oph_universe_sweep(
    config_paths: list[Path],
    out_dir: Path,
    *,
    seeds: list[int] | None = None,
    workers: int | None = None,
    inner_jobs: int | None = None,
    max_screen_points: int = 5000,
    max_observers: int = 128,
    max_h3_objects: int = 512,
    emit_visualizations: bool = True,
) -> dict[str, Any]:
    """Run theorem-following OPH universe jobs across configs/seeds.

    This is the cloud-facing companion to ``run_bw_sweep``. Each outer worker
    runs the full OPH-universe pipeline: base simulation, H3 refit/object sweep,
    proof/readout reports, and visualizer payload export.
    """

    jobs: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        seed_values = seeds or [int(config.get("seed", 1))]
        name = str(config.get("name", config_path.stem))
        for seed in seed_values:
            jobs.append(
                {
                    "config_path": str(config_path),
                    "config_name": name,
                    "seed": int(seed),
                    "run_id": f"{_slug(name)}_seed{int(seed)}_{uuid.uuid4().hex[:8]}",
                }
            )

    if not jobs:
        return {"engine": "oph_universe_sweep", "jobs": [], "summary_path": None}

    max_workers, planned_inner_jobs = sweep_parallel_plan(
        job_count=len(jobs),
        workers=workers,
        inner_jobs=inner_jobs,
    )
    started = time.time()
    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for job in jobs:
            results.append(
                _run_one(
                    job,
                    str(out_dir),
                    planned_inner_jobs,
                    max_screen_points,
                    max_observers,
                    max_h3_objects,
                    emit_visualizations,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=_process_context()) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    job,
                    str(out_dir),
                    planned_inner_jobs,
                    max_screen_points,
                    max_observers,
                    max_h3_objects,
                    emit_visualizations,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda row: (str(row.get("config_path")), int(row.get("seed", 0)), str(row.get("run_id", ""))))
    summary = {
        "engine": "oph_universe_sweep",
        "workers": max_workers,
        "inner_jobs": planned_inner_jobs,
        "parallelism_policy": "auto_fill_available_cpus" if workers is None and inner_jobs is None else "explicit_or_partially_explicit",
        "elapsed_seconds": time.time() - started,
        "job_count": len(jobs),
        "results": results,
        "receipt_summary": _receipt_summary(results),
        "claim_boundary": (
            "Parallel theorem-following OPH universe sweep. Each job keeps failed theorem, neutral-bulk, "
            "particle, and physical-CMB gates explicit; larger scale does not promote a claim unless the "
            "corresponding receipt is true."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"oph_universe_sweep_{int(started)}.json"
    import json

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _run_one(
    job: dict[str, Any],
    out_dir: str,
    inner_jobs: int,
    max_screen_points: int,
    max_observers: int,
    max_h3_objects: int,
    emit_visualizations: bool,
) -> dict[str, Any]:
    try:
        result = run_oph_universe_pipeline(
            config_path=Path(job["config_path"]),
            out_dir=Path(out_dir),
            run_id=str(job["run_id"]),
            seed=int(job["seed"]),
            inner_jobs=int(inner_jobs),
            max_screen_points=int(max_screen_points),
            max_observers=int(max_observers),
            max_h3_objects=int(max_h3_objects),
            emit_visualizations=bool(emit_visualizations),
        )
        return {
            "ok": True,
            "config_path": job["config_path"],
            "config_name": job["config_name"],
            "seed": int(job["seed"]),
            "run_id": job["run_id"],
            "run_dir": result.get("run_dir"),
            "final_receipts": result.get("final_receipts", {}),
            "viewer_outputs": result.get("viewer_outputs", {}),
            "proof_summary": result.get("proof_summary", {}),
            "frontier_artifacts": result.get("frontier_artifacts", {}),
        }
    except Exception as exc:  # pragma: no cover - retained in long-run summaries
        return {
            "ok": False,
            "config_path": job.get("config_path"),
            "config_name": job.get("config_name"),
            "seed": job.get("seed"),
            "run_id": job.get("run_id"),
            "error": repr(exc),
        }


def _receipt_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("ok", False)]
    receipt_counts: Counter[str] = Counter()
    for row in ok_results:
        for key, value in (row.get("final_receipts") or {}).items():
            if bool(value):
                receipt_counts[str(key)] += 1
    return {
        "ok_job_count": len(ok_results),
        "failed_job_count": len(results) - len(ok_results),
        "passed_receipt_counts": dict(sorted(receipt_counts.items())),
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
