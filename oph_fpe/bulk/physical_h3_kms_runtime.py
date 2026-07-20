"""Fail-closed numerical-runtime binding for exact H3/KMS replay.

The physical campaign persists floating-point leaves and hashes derived from
them.  Exact replay therefore depends on the interpreter, NumPy/SciPy build,
machine ABI, and the process-level thread policy in addition to source-code
bytes.  This module records that execution environment before source capture
and rejects replay under a different environment instead of misreporting
ordinary last-bit numerical drift as source-code nondeterminism.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy


NUMERICAL_RUNTIME_SCHEMA = "oph.physical-h3-kms.numerical-runtime.v1"
NUMERICAL_RUNTIME_CONTRACT_ID = "exact-float-replay-runtime-v1"
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
REQUIRED_THREAD_ENVIRONMENT = {key: "1" for key in THREAD_ENVIRONMENT_KEYS}
_RUNTIME_FIELDS = frozenset(
    {
        "schema",
        "contract_id",
        "required_thread_environment",
        "observed_thread_environment",
        "thread_environment_receipt",
        "python",
        "numpy",
        "scipy",
        "platform",
        "floating_point",
        "runtime_sha256",
        "claim_boundary",
    }
)


class NumericalRuntimeError(RuntimeError):
    """Raised before numerical work when the frozen runtime is not active."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _selected_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    row = _mapping(value)
    return {field: row.get(field) for field in fields}


def _numpy_build_observation() -> dict[str, Any]:
    """Return a path-free description of NumPy's numerical backend."""

    config = _mapping(getattr(np.__config__, "CONFIG", {}))
    dependencies = _mapping(config.get("Build Dependencies"))
    compilers = _mapping(config.get("Compilers"))
    machine = _mapping(config.get("Machine Information"))
    simd = _mapping(config.get("SIMD Extensions"))
    dependency_fields = ("name", "version", "detection method", "found")
    compiler_fields = ("name", "version", "linker")
    machine_fields = ("cpu", "endian", "family", "system")
    return {
        "blas": _selected_fields(dependencies.get("blas"), dependency_fields),
        "lapack": _selected_fields(
            dependencies.get("lapack"), dependency_fields
        ),
        "c_compiler": _selected_fields(compilers.get("c"), compiler_fields),
        "cxx_compiler": _selected_fields(
            compilers.get("c++"), compiler_fields
        ),
        "build_machine": _selected_fields(
            machine.get("build"), machine_fields
        ),
        "host_machine": _selected_fields(machine.get("host"), machine_fields),
        "simd": {
            "baseline": list(simd.get("baseline", [])),
            "found": list(simd.get("found", [])),
            "not_found": list(simd.get("not found", [])),
        },
    }


def observed_numerical_runtime() -> dict[str, Any]:
    """Capture the exact, path-free runtime used by a campaign process."""

    observed_threads = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS}
    version = sys.version_info
    material = {
        "schema": NUMERICAL_RUNTIME_SCHEMA,
        "contract_id": NUMERICAL_RUNTIME_CONTRACT_ID,
        "required_thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "observed_thread_environment": observed_threads,
        "thread_environment_receipt": observed_threads
        == REQUIRED_THREAD_ENVIRONMENT,
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": [
                version.major,
                version.minor,
                version.micro,
                version.releaselevel,
                version.serial,
            ],
            "compiler": platform.python_compiler(),
        },
        "numpy": {
            "version": np.__version__,
            "build": _numpy_build_observation(),
        },
        "scipy": {"version": scipy.__version__},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "floating_point": {
            "python_float_radix": sys.float_info.radix,
            "python_float_mant_dig": sys.float_info.mant_dig,
            "python_float_max_exp": sys.float_info.max_exp,
            "numpy_float64_itemsize": np.dtype(np.float64).itemsize,
            "numpy_complex128_itemsize": np.dtype(np.complex128).itemsize,
        },
        "claim_boundary": (
            "This receipt binds a software execution environment for exact byte "
            "replay. It is an instrument receipt, not evidence for a physical "
            "clock, geometry, event manifold, or emergence claim."
        ),
    }
    return {**material, "runtime_sha256": _sha256(material)}


def numerical_runtime_blockers(
    frozen: Mapping[str, Any], *, compare_current: bool = True
) -> list[str]:
    """Validate a frozen runtime and optionally compare it to this process."""

    blockers: list[str] = []
    value = dict(frozen) if isinstance(frozen, Mapping) else {}
    if set(value) != _RUNTIME_FIELDS:
        blockers.append("numerical_runtime_field_set_mismatch")
        return blockers
    if value.get("schema") != NUMERICAL_RUNTIME_SCHEMA:
        blockers.append("numerical_runtime_schema_mismatch")
    if value.get("contract_id") != NUMERICAL_RUNTIME_CONTRACT_ID:
        blockers.append("numerical_runtime_contract_id_mismatch")
    if value.get("required_thread_environment") != REQUIRED_THREAD_ENVIRONMENT:
        blockers.append("required_thread_environment_mismatch")
    observed_threads = value.get("observed_thread_environment")
    if observed_threads != REQUIRED_THREAD_ENVIRONMENT:
        blockers.append("observed_thread_environment_not_canonical")
    if value.get("thread_environment_receipt") is not True:
        blockers.append("thread_environment_receipt_false")
    material = {key: item for key, item in value.items() if key != "runtime_sha256"}
    if value.get("runtime_sha256") != _sha256(material):
        blockers.append("numerical_runtime_sha256_mismatch")
    if compare_current:
        current = observed_numerical_runtime()
        if _canonical_bytes(value) != _canonical_bytes(current):
            for field in (
                "observed_thread_environment",
                "python",
                "numpy",
                "scipy",
                "platform",
                "floating_point",
            ):
                if _canonical_bytes(value.get(field)) != _canonical_bytes(
                    current.get(field)
                ):
                    blockers.append(f"current_{field}_mismatch")
            if not any(item.startswith("current_") for item in blockers):
                blockers.append("current_numerical_runtime_mismatch")
    return list(dict.fromkeys(blockers))


def require_canonical_numerical_runtime() -> dict[str, Any]:
    """Return this runtime or reject before output/RNG materialization."""

    observed = observed_numerical_runtime()
    blockers = numerical_runtime_blockers(observed, compare_current=False)
    if blockers:
        raise NumericalRuntimeError(
            "canonical numerical runtime is not active: " + ",".join(blockers)
        )
    return observed


def require_frozen_numerical_runtime(
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact agreement between a frozen artifact and this process."""

    blockers = numerical_runtime_blockers(frozen, compare_current=True)
    if blockers:
        raise NumericalRuntimeError(
            "frozen numerical runtime differs from current process: "
            + ",".join(blockers)
        )
    return dict(frozen)


__all__ = [
    "NUMERICAL_RUNTIME_CONTRACT_ID",
    "NUMERICAL_RUNTIME_SCHEMA",
    "NumericalRuntimeError",
    "REQUIRED_THREAD_ENVIRONMENT",
    "THREAD_ENVIRONMENT_KEYS",
    "numerical_runtime_blockers",
    "observed_numerical_runtime",
    "require_canonical_numerical_runtime",
    "require_frozen_numerical_runtime",
]
