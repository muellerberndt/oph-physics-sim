from __future__ import annotations

import os
from typing import Any


def available_cpu_count(*, reserve: int = 1) -> int:
    override = os.environ.get("OPH_FPE_CPUS")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    count = os.cpu_count() or 1
    return max(1, count - max(0, int(reserve)))


def jobs_from_config(value: Any, *, default: int = 1, reserve: int = 1) -> int:
    if value is None:
        return max(1, int(default))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"auto", "all", "-1", "0"}:
            return available_cpu_count(reserve=reserve)
        return max(1, int(text))
    numeric = int(value)
    if numeric <= 0:
        return available_cpu_count(reserve=reserve)
    return numeric


def sweep_parallel_plan(
    *,
    job_count: int,
    workers: int | None,
    inner_jobs: int | None,
    reserve: int = 1,
) -> tuple[int, int]:
    if job_count <= 0:
        return 0, 0
    cpus = available_cpu_count(reserve=reserve)
    if workers is not None:
        worker_count = max(1, min(int(workers), int(job_count)))
    else:
        worker_count = max(1, min(int(job_count), cpus))
    if inner_jobs is not None:
        inner_count = jobs_from_config(inner_jobs, default=1, reserve=reserve)
    else:
        inner_count = max(1, cpus // worker_count)
    return worker_count, inner_count
