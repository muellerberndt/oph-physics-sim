"""Frozen, non-demo runner for the physical H3/KMS campaign.

The runner owns the chronology that the campaign needs but the individual
scientific producers cannot prove on their own:

1. construct and validate the complete multi-seed, four-rung plan;
2. write the plan, real independent calibration artifacts, and registered
   code-byte commitments before any source RNG is invoked;
3. run exactly one selected target-blind source cell; and
4. delegate canonical post-run persistence to the disk replay bundle writer,
   then immediately ask the independent disk replayer to verify it.

This is deliberately a physical-only entry point.  It has no demo, force,
nudge, target-value, threshold-override, or receipt-override argument.  Demo
completion belongs to the visualization layer and cannot enter this module's
source path or evidence bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oph_fpe.bulk.physical_h3_kms_aggregate import (
    aggregate_physical_h3_kms_family,
)
from oph_fpe.bulk.physical_h3_kms_postrun import (
    PREREGISTRATION_ENVELOPE_SCHEMA,
)
from oph_fpe.bulk.physical_h3_kms_prerun import (
    CELL_CONFIG_SCHEMA,
    PLAN_SCHEMA,
    REQUIRED_CLOCK_CANDIDATES,
    REQUIRED_GEOMETRY_MODELS,
    REQUIRED_RUNGS,
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
    SCHEMA_VERSION,
    canonical_sha256,
    frozen_campaign_family_sha256,
    physical_h3_kms_prerun_report,
)
from oph_fpe.bulk.physical_h3_kms_replay import (
    registered_instrument_code_bundle,
    registered_role_code_bindings,
    registered_role_code_registries,
    replay_physical_h3_kms_bundle,
    write_physical_h3_kms_replay_bundle,
)
from oph_fpe.bulk.physical_h3_kms_runtime import (
    NumericalRuntimeError,
    observed_numerical_runtime,
    require_canonical_numerical_runtime,
    require_frozen_numerical_runtime,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source


CAMPAIGN_PACKAGE_SCHEMA = "oph.physical-h3-kms.frozen-campaign-package.v2"
FREEZE_RECEIPT_SCHEMA = "oph.physical-h3-kms.pre-source-freeze-receipt.v2"
RUN_RECEIPT_SCHEMA = "oph.physical-h3-kms.campaign-cell-run-receipt.v2"
DEFAULT_CAMPAIGN_ID = "physical-h3-kms-frozen-family-003"
DEFAULT_INSTRUMENT_VERSION = "physical-h3-kms-v2"
DEFAULT_SEEDS = (20_260_751, 20_260_761, 20_260_771)
DEFAULT_REPLICATE_IDS = ("primary",)
HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256 = (
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
)
HISTORICAL_CAMPAIGN_SHA256 = REGISTERED_HISTORICAL_CAMPAIGN_SHA256
DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT = (
    Path(__file__).resolve().parents[3]
    / "survival-proof-4"
    / "outputs"
    / "campaign_receipt.json"
)

_SUPPORT_LEVEL_BY_RUNG = {
    4_096: 4,
    16_384: 5,
    65_536: 6,
    262_144: 7,
}
_CALIBRATION_SEEDS = {
    "clock_calibration": (910_001, 910_002, 910_003),
    "geometry_calibration": (920_001, 920_002, 920_003),
    "curvature_calibration": (930_001, 930_002, 930_003),
}
_CALIBRATION_FILENAMES = {
    "clock_calibration": "clock_calibration.json",
    "geometry_calibration": "geometry_calibration.json",
    "curvature_calibration": "curvature_calibration.json",
}
_FORBIDDEN_PHYSICAL_CONTROL_TOKENS = frozenset(
    {
        "demo",
        "force",
        "forced",
        "nudge",
        "nudged",
        "override",
        "assumption",
        "synthetic-pass",
        "target-value",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PhysicalCampaignError(RuntimeError):
    """Raised when a physical campaign cell cannot produce a valid bundle."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise PhysicalCampaignError("artifact is not finite canonical JSON") from exc


def _byte_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _write_exclusive_json(path: Path, value: Any) -> dict[str, Any]:
    data = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PhysicalCampaignError(f"refusing to overwrite artifact: {path}") from exc
    return {
        "path": path.as_posix(),
        "byte_sha256": _byte_sha256(data),
        "byte_count": len(data),
    }


def _write_exclusive_bytes(path: Path, data: bytes) -> dict[str, Any]:
    """Persist already-committed historical bytes without reserialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PhysicalCampaignError(f"refusing to overwrite artifact: {path}") from exc
    return {
        "path": path.as_posix(),
        "byte_sha256": _byte_sha256(data),
        "byte_count": len(data),
    }


def _read_exact_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhysicalCampaignError(f"invalid frozen JSON artifact: {path}") from exc
    if raw != _canonical_bytes(value):
        raise PhysicalCampaignError(f"artifact is not exact canonical JSON: {path}")
    return value


def _load_historical_campaign_receipt(
    path: str | Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Verify the archived 16k negative cell before any new source capture.

    The historical bytes are an outcome quarantine, not calibration input.  A
    new physical run must carry them forward exactly so the no-retuning claim
    is content-addressed rather than a caller-supplied Boolean.
    """

    receipt_path = Path(path) if path is not None else DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PhysicalCampaignError(
            f"historical campaign receipt is missing or not a real file: {receipt_path}"
        )
    raw = receipt_path.read_bytes()
    observed_hash = _byte_sha256(raw)
    if observed_hash != HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256:
        raise PhysicalCampaignError(
            "historical campaign receipt bytes changed; refusing an unbound no-retune claim"
        )
    try:
        receipt = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhysicalCampaignError("historical campaign receipt is invalid JSON") from exc
    if not isinstance(receipt, Mapping):
        raise PhysicalCampaignError("historical campaign receipt must be an object")
    cells = receipt.get("cells")
    if not isinstance(cells, list):
        raise PhysicalCampaignError("historical campaign receipt has no cell census")
    selected = [
        row
        for row in cells
        if isinstance(row, Mapping)
        and row.get("source_seed") == REGISTERED_HISTORICAL_16K_SOURCE_SEED
        and row.get("patch_count") == 16_384
    ]
    stable_failure = receipt.get("stable_failure")
    stable_established = (
        stable_failure.get("established")
        if isinstance(stable_failure, Mapping)
        else None
    )
    valid = bool(
        receipt.get("schema") == "oph_physical_h3_kms_campaign_receipt_v2"
        and receipt.get("campaign_sha256") == HISTORICAL_CAMPAIGN_SHA256
        and receipt.get("PHYSICAL_H3_KMS_EMERGENCE_RECEIPT") is False
        and receipt.get("BRANCH_RETIREMENT_RECEIPT") is False
        and receipt.get("retuning_permitted") is False
        and stable_established is False
        and len(selected) == 1
        and selected[0].get("artifact_present") is True
        and selected[0].get("artifact_complete") is True
        and selected[0].get("joint_independent_receipt") is False
        and "independently_derived_2pi_clock_failed"
        in selected[0].get("blockers", [])
        and "h3_receipt_failed" in selected[0].get("blockers", [])
    )
    if not valid:
        raise PhysicalCampaignError(
            "historical receipt no longer contains the registered 16k negative cell"
        )
    return raw, {
        "historical_receipt_byte_sha256": observed_hash,
        "historical_campaign_sha256": HISTORICAL_CAMPAIGN_SHA256,
        "historical_16k_source_seed": REGISTERED_HISTORICAL_16K_SOURCE_SEED,
        "historical_16k_rung": 16_384,
        "historical_16k_joint_independent_receipt": False,
        "historical_stable_branch_failure_established": False,
    }


def _counter_uniform53(domain: str, seed: int, index: int) -> float:
    digest = hashlib.sha256(f"{domain}\0{seed}\0{index}".encode("ascii")).digest()
    numerator = int.from_bytes(digest[:8], "big") >> 11
    return numerator / float(1 << 53)


def _ceil_grid(value: float, step: float) -> float:
    return round(math.ceil(value / step - 1.0e-12) * step, 12)


def _floor_grid(value: float, step: float) -> float:
    return round(math.floor(value / step + 1.0e-12) * step, 12)


def _calibration_rows(
    *, domain: str, seeds: Sequence[int], lower: float, width: float
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "sample_index": index,
            "synthetic_measurement": round(
                lower + width * _counter_uniform53(domain, seed, index), 15
            ),
        }
        for seed in seeds
        for index in range(64)
    ]


def independent_calibration_artifacts(
    campaign_seeds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Build deterministic pre-campaign threshold-fixture bytes.

    The samples use a SHA-256 counter construction, not the campaign source
    RNG.  The three calibration seed sets are pairwise disjoint and must also
    be disjoint from all source seeds.  Thresholds are mechanically derived
    from the sample extrema on a frozen 0.05 grid; no archived campaign outcome
    participates.  These rows do not simulate an instrument or noise model and
    therefore carry an explicit false physical-calibration receipt.  They may
    freeze diagnostic thresholds but cannot authorize a physical gate.
    """

    source_seed_set = set(campaign_seeds)
    calibration_seed_sets = [set(values) for values in _CALIBRATION_SEEDS.values()]
    if any(source_seed_set.intersection(values) for values in calibration_seed_sets):
        raise PhysicalCampaignError(
            "campaign source seeds must be disjoint from calibration seeds"
        )
    if any(
        left.intersection(right)
        for index, left in enumerate(calibration_seed_sets)
        for right in calibration_seed_sets[index + 1 :]
    ):
        raise PhysicalCampaignError("calibration seed sets must be pairwise disjoint")

    clock_residual_rows = _calibration_rows(
        domain="oph-clock-absolute-residual-v1",
        seeds=_CALIBRATION_SEEDS["clock_calibration"],
        lower=0.02,
        width=0.16,
    )
    clock_margin_rows = _calibration_rows(
        domain="oph-clock-winning-margin-v1",
        seeds=_CALIBRATION_SEEDS["clock_calibration"],
        lower=0.10,
        width=0.20,
    )
    clock_thresholds = {
        "clock_absolute_residual_max": _ceil_grid(
            max(row["synthetic_measurement"] for row in clock_residual_rows), 0.05
        ),
        "clock_win_margin_min": _floor_grid(
            min(row["synthetic_measurement"] for row in clock_margin_rows), 0.05
        ),
    }
    geometry_rows = _calibration_rows(
        domain="oph-geometry-winning-margin-v1",
        seeds=_CALIBRATION_SEEDS["geometry_calibration"],
        lower=0.05,
        width=0.15,
    )
    geometry_thresholds = {
        "geometry_win_margin_min": _floor_grid(
            min(row["synthetic_measurement"] for row in geometry_rows), 0.05
        )
    }
    curvature_rows = _calibration_rows(
        domain="oph-curvature-detection-power-v1",
        seeds=_CALIBRATION_SEEDS["curvature_calibration"],
        lower=0.90,
        width=0.09,
    )
    curvature_thresholds = {
        "curvature_minimum_power": _floor_grid(
            min(row["synthetic_measurement"] for row in curvature_rows), 0.05
        )
    }
    expected = {
        "clock_absolute_residual_max": 0.2,
        "clock_win_margin_min": 0.1,
        "geometry_win_margin_min": 0.05,
        "curvature_minimum_power": 0.9,
    }
    observed = {**clock_thresholds, **geometry_thresholds, **curvature_thresholds}
    if observed != expected:
        raise PhysicalCampaignError(
            "independent calibration protocol no longer derives registered thresholds"
        )

    common = {
        "independent_of_campaign_source_seeds": True,
        "frozen_before_source_capture": True,
    }
    return {
        "clock_calibration": {
            "schema": "oph.physical-h3-kms.clock-calibration.v1",
            "calibration_id": "clock-independent-synthetic-grid-v1",
            "calibration_seeds": list(_CALIBRATION_SEEDS["clock_calibration"]),
            **common,
            "protocol": {
                "generator": "sha256_counter_uniform53_v1",
                "scope": "deterministic_threshold_fixture_only",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
                "sample_count_per_seed_per_statistic": 64,
                "campaign_source_rng_used": False,
                "archived_campaign_outcomes_used": False,
                "absolute_residual_samples": clock_residual_rows,
                "winning_margin_samples": clock_margin_rows,
                "derivation": {
                    "clock_absolute_residual_max": "ceil(max(samples)/0.05)*0.05",
                    "clock_win_margin_min": "floor(min(samples)/0.05)*0.05",
                },
            },
            "thresholds": clock_thresholds,
        },
        "geometry_calibration": {
            "schema": "oph.physical-h3-kms.geometry-calibration.v1",
            "calibration_id": "geometry-independent-synthetic-grid-v1",
            "calibration_seeds": list(_CALIBRATION_SEEDS["geometry_calibration"]),
            **common,
            "protocol": {
                "generator": "sha256_counter_uniform53_v1",
                "scope": "deterministic_threshold_fixture_only",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
                "sample_count_per_seed": 64,
                "campaign_source_rng_used": False,
                "archived_campaign_outcomes_used": False,
                "winning_margin_samples": geometry_rows,
                "derivation": {
                    "geometry_win_margin_min": "floor(min(samples)/0.05)*0.05"
                },
            },
            "thresholds": geometry_thresholds,
        },
        "curvature_calibration": {
            "schema": "oph.physical-h3-kms.curvature-calibration.v1",
            "calibration_id": "curvature-independent-synthetic-grid-v1",
            "calibration_seeds": list(_CALIBRATION_SEEDS["curvature_calibration"]),
            **common,
            "protocol": {
                "generator": "sha256_counter_uniform53_v1",
                "scope": "deterministic_threshold_fixture_only",
                "physical_threshold_calibration_receipt": False,
                "physical_gate_eligible": False,
                "sample_count_per_seed": 64,
                "campaign_source_rng_used": False,
                "archived_campaign_outcomes_used": False,
                "detection_power_samples": curvature_rows,
                "derivation": {
                    "curvature_minimum_power": "floor(min(samples)/0.05)*0.05"
                },
            },
            "thresholds": curvature_thresholds,
        },
    }


def _strict_seed_list(values: Sequence[int]) -> list[int]:
    seeds = list(values)
    if (
        len(seeds) < 3
        or len(seeds) != len(set(seeds))
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
    ):
        raise PhysicalCampaignError(
            "campaign requires at least three unique exact nonnegative int64 seeds"
        )
    return seeds


def _strict_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PhysicalCampaignError(f"{label} must be a bounded identifier")
    tokens = {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}
    if tokens.intersection(_FORBIDDEN_PHYSICAL_CONTROL_TOKENS):
        raise PhysicalCampaignError(f"{label} contains a forbidden demo/control token")
    return value


def _support_count(rung: int) -> int:
    return 20 * (4 ** _SUPPORT_LEVEL_BY_RUNG[rung])


def _observer_count(rung: int) -> int:
    return min(4_096, max(32, math.ceil(0.5 * math.sqrt(rung))))


def _cell_config(
    *,
    seed: int,
    rung: int,
    replicate_id: str,
    producer_registry: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    source_dynamics = dict(producer_registry["source_dynamics"])
    return {
        "schema": CELL_CONFIG_SCHEMA,
        "cell": {"seed": seed, "rung": rung, "replicate_id": replicate_id},
        "source_federation": {
            "family": "federated_echosahedral_carriers",
            "carrier_count": rung,
            "ports_per_carrier": 12,
            "local_template": "regular_icosahedron_12_30_20_antipode_a5_v1",
            **dict(producer_registry["source_federation"]),
        },
        "support_regulator": {
            "family": "nested_geodesic_icosahedral",
            "patch_basis": "cells",
            "refinement_level": _SUPPORT_LEVEL_BY_RUNG[rung],
            "patch_count": _support_count(rung),
            "drives_source_seams": False,
            "drives_source_repairs": False,
            **dict(producer_registry["support_regulator"]),
        },
        "source_generator": {
            **source_dynamics,
            "state_space": "normalized_complex_amplitude_in_C12",
            "rng_family": "numpy_generator_pcg64_v1",
            "initialization_distribution": "normalized_complex_gaussian_v1",
            "intrinsic_phase_distribution": "uniform_unit_interval_v1",
            "propagation_steps": 4,
            "intrinsic_step": 0.137,
            "coupling_strength": 1.0,
            "geometry_sample_count": 32,
        },
        "repair_dynamics": {
            **source_dynamics,
            "cycles": 160,
            "repair_fraction_per_cycle": 0.0625,
            "record_commit_cycles": 12,
            "seam_update_rule": (
                "disjoint_single_port_endpoint_arithmetic_mean_v1"
            ),
        },
        "observer_capture": {
            **dict(producer_registry["observer_capture"]),
            "observer_count": _observer_count(rung),
            "support_size": 2,
            "samples_per_observer": 6,
            "prediction_control": "semantic_hash_shuffle_v1",
            "feedback_enabled": True,
            "checkpoint_interval": 8,
        },
    }


def build_frozen_campaign(
    *,
    current_seed: int = DEFAULT_SEEDS[0],
    current_rung: int = REQUIRED_RUNGS[0],
    current_replicate_id: str = DEFAULT_REPLICATE_IDS[0],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    replicate_ids: Sequence[str] = DEFAULT_REPLICATE_IDS,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
) -> dict[str, Any]:
    """Construct the complete immutable campaign and static admission report."""

    frozen_seeds = _strict_seed_list(seeds)
    frozen_replicates = [
        _strict_identifier(value, "replicate_id") for value in replicate_ids
    ]
    if not frozen_replicates or len(frozen_replicates) != len(set(frozen_replicates)):
        raise PhysicalCampaignError("replicate_ids must be nonempty and unique")
    _strict_identifier(campaign_id, "campaign_id")
    if type(current_seed) is not int or current_seed not in frozen_seeds:
        raise PhysicalCampaignError("current_seed must be one of the frozen seeds")
    if type(current_rung) is not int or current_rung not in REQUIRED_RUNGS:
        raise PhysicalCampaignError("current_rung must be one of 4k/16k/64k/256k")
    current_replicate_id = _strict_identifier(
        current_replicate_id, "current_replicate_id"
    )
    if current_replicate_id not in frozen_replicates:
        raise PhysicalCampaignError(
            "current_replicate_id must be one of the frozen replicate IDs"
        )

    calibrations = independent_calibration_artifacts(frozen_seeds)
    thresholds = {
        **calibrations["clock_calibration"]["thresholds"],
        **calibrations["geometry_calibration"]["thresholds"],
        **calibrations["curvature_calibration"]["thresholds"],
    }
    registries = registered_role_code_registries()
    role_bindings = registered_role_code_bindings()
    instrument_bundle = registered_instrument_code_bundle()
    current_cell = {
        "seed": current_seed,
        "rung": current_rung,
        "replicate_id": current_replicate_id,
    }
    support_counts = {str(rung): _support_count(rung) for rung in REQUIRED_RUNGS}
    observer_counts = {str(rung): _observer_count(rung) for rung in REQUIRED_RUNGS}
    run_matrix: list[dict[str, Any]] = []
    current_config: dict[str, Any] | None = None
    for seed in frozen_seeds:
        for rung in REQUIRED_RUNGS:
            for replicate_id in frozen_replicates:
                config = _cell_config(
                    seed=seed,
                    rung=rung,
                    replicate_id=replicate_id,
                    producer_registry=registries["producer_registry"],
                )
                row = {
                    "cell": copy.deepcopy(config["cell"]),
                    "cell_config": config,
                    "config_sha256": canonical_sha256(config),
                    "status": "NOT_EVALUATED",
                }
                run_matrix.append(row)
                if config["cell"] == current_cell:
                    current_config = copy.deepcopy(config)
    if current_config is None:
        raise PhysicalCampaignError("current cell was not constructed exactly once")

    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "instrument_version": DEFAULT_INSTRUMENT_VERSION,
        "instrument_commit_sha256": instrument_bundle[
            "instrument_commit_sha256"
        ],
        "seeds": frozen_seeds,
        "rungs": list(REQUIRED_RUNGS),
        "replicate_ids": frozen_replicates,
        "clock_candidates": list(REQUIRED_CLOCK_CANDIDATES),
        "geometry_models": list(REQUIRED_GEOMETRY_MODELS),
        "thresholds": thresholds,
        "calibrations": {
            "clock_calibration_sha256": canonical_sha256(
                calibrations["clock_calibration"]
            ),
            "geometry_calibration_sha256": canonical_sha256(
                calibrations["geometry_calibration"]
            ),
            "curvature_calibration_sha256": canonical_sha256(
                calibrations["curvature_calibration"]
            ),
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "physical_threshold_calibration_receipt": False,
        },
        "split_contract": {
            "algorithm_id": "semantic_hash_split_v1",
            "assignment_salt_sha256": canonical_sha256(
                {
                    "domain": "physical-h3-kms-semantic-holdout-v1",
                    "campaign_id": campaign_id,
                    "seeds": frozen_seeds,
                }
            ),
            "holdout_fraction": 0.25,
            "derivation": "semantic_event_id_hash_threshold_v1",
            "heldout_ids_materialized_before_capture": False,
        },
        "scaling_contract": {
            "carrier_count_law": "exact_rung_cardinality_v1",
            "support_regulator_law": (
                "first_icosahedral_cell_count_at_or_above_rung_v1"
            ),
            "support_counts_by_rung": support_counts,
            "observer_scaling": {
                "law_id": "power_law_ceil_v1",
                "coefficient": 0.5,
                "exponent": 0.5,
                "minimum": 32,
                "maximum": 4_096,
                "counts_by_rung": observer_counts,
            },
            "cycles": 160,
            "repair_fraction_per_cycle": 0.0625,
            "record_commit_cycles": 12,
        },
        "archive_boundary": {
            "frozen_before_source_capture": True,
            "retune_after_freeze": False,
            "archived_16k_failure_preserved": True,
            "archived_outcomes_used_for_threshold_selection": False,
            "historical_receipt_byte_sha256": (
                HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256
            ),
            "historical_campaign_sha256": HISTORICAL_CAMPAIGN_SHA256,
            "historical_16k_source_seed": REGISTERED_HISTORICAL_16K_SOURCE_SEED,
            "historical_16k_rung": 16_384,
            "historical_16k_joint_independent_receipt": False,
            "historical_stable_branch_failure_established": False,
        },
        "producer_registry": registries["producer_registry"],
        "checker_registry": registries["checker_registry"],
        "run_matrix": run_matrix,
        "current_cell": current_cell,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    preregistration = {
        "schema": SCHEMA_VERSION,
        "config": current_config,
        "plan": plan,
    }
    prerun_report = physical_h3_kms_prerun_report(preregistration)
    if (
        prerun_report.get("admission_status") != "VALID_PASS"
        or prerun_report.get("SOURCE_CAPTURE_ALLOWED") is not True
        or prerun_report.get("scientific_status") != "NOT_EVALUATED"
    ):
        raise PhysicalCampaignError(
            "constructed campaign failed static admission: "
            + ",".join(
                str(item)
                for item in [
                    *prerun_report.get("blockers", []),
                    *prerun_report.get("invalidators", []),
                ]
            )
        )
    postrun_envelope = {
        "schema": PREREGISTRATION_ENVELOPE_SCHEMA,
        "preregistration": preregistration,
        "preregistration_report": prerun_report,
        "preregistration_sha256": canonical_sha256(preregistration),
    }
    return {
        "schema": CAMPAIGN_PACKAGE_SCHEMA,
        "preregistration": preregistration,
        "preregistration_report": prerun_report,
        "postrun_preregistration": postrun_envelope,
        "calibration_artifacts": calibrations,
        "role_code_bindings": role_bindings,
        "instrument_code_bundle": instrument_bundle,
        "numerical_runtime": observed_numerical_runtime(),
    }


def _write_and_verify_freeze(
    output_root: Path,
    package: Mapping[str, Any],
    historical_receipt_bytes: bytes,
    historical_witness: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    freeze_root = output_root / "freeze"
    freeze_root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    values = {
        "preregistration": package["preregistration"],
        "preregistration_report": package["preregistration_report"],
        "postrun_preregistration": package["postrun_preregistration"],
        "role_code_bindings": package["role_code_bindings"],
        "instrument_code_bundle": package["instrument_code_bundle"],
        "numerical_runtime": package["numerical_runtime"],
        **package["calibration_artifacts"],
    }
    filenames = {
        "preregistration": "preregistration.json",
        "preregistration_report": "preregistration_report.json",
        "postrun_preregistration": "postrun_preregistration.json",
        "role_code_bindings": "role_code_bindings.json",
        "instrument_code_bundle": "instrument_code_bundle.json",
        "numerical_runtime": "numerical_runtime.json",
        **_CALIBRATION_FILENAMES,
    }
    for name, value in values.items():
        path = freeze_root / filenames[name]
        descriptor = _write_exclusive_json(path, value)
        descriptor["path"] = path.relative_to(output_root).as_posix()
        files[name] = descriptor

    archive_path = freeze_root / "historical_campaign_receipt.json"
    archive_descriptor = _write_exclusive_bytes(
        archive_path, historical_receipt_bytes
    )
    archive_descriptor["path"] = archive_path.relative_to(output_root).as_posix()
    files["historical_campaign_receipt"] = archive_descriptor

    # Re-read the actual disk bytes and rerun every static/code binding before
    # recording that the freeze boundary has completed.
    loaded = {
        name: _read_exact_json(output_root / row["path"])
        for name, row in files.items()
        if name != "historical_campaign_receipt"
    }
    if _canonical_bytes(loaded["preregistration"]) != _canonical_bytes(
        package["preregistration"]
    ):
        raise PhysicalCampaignError("disk preregistration differs from memory freeze")
    replayed_prerun = physical_h3_kms_prerun_report(loaded["preregistration"])
    if _canonical_bytes(replayed_prerun) != _canonical_bytes(
        loaded["preregistration_report"]
    ):
        raise PhysicalCampaignError("disk preregistration report is not exact replay")
    if _canonical_bytes(registered_role_code_bindings()) != _canonical_bytes(
        loaded["role_code_bindings"]
    ):
        raise PhysicalCampaignError("registered role code changed during freeze")
    if _canonical_bytes(registered_instrument_code_bundle()) != _canonical_bytes(
        loaded["instrument_code_bundle"]
    ):
        raise PhysicalCampaignError("instrument code bundle changed during freeze")
    try:
        require_frozen_numerical_runtime(loaded["numerical_runtime"])
    except NumericalRuntimeError as exc:
        raise PhysicalCampaignError(str(exc)) from exc
    plan = loaded["preregistration"]["plan"]
    if dict(plan["archive_boundary"]) != {
        "frozen_before_source_capture": True,
        "retune_after_freeze": False,
        "archived_16k_failure_preserved": True,
        "archived_outcomes_used_for_threshold_selection": False,
        **dict(historical_witness),
    }:
        raise PhysicalCampaignError(
            "historical archive witness differs from the frozen plan"
        )
    if archive_descriptor["byte_sha256"] != plan["archive_boundary"][
        "historical_receipt_byte_sha256"
    ]:
        raise PhysicalCampaignError("historical receipt bytes do not match frozen plan")
    for name in _CALIBRATION_FILENAMES:
        plan_field = f"{name}_sha256"
        if plan["calibrations"][plan_field] != canonical_sha256(loaded[name]):
            raise PhysicalCampaignError(f"{name} disk bytes do not match frozen plan")

    frozen_at = _utc_now()
    freeze_receipt = {
        "schema": FREEZE_RECEIPT_SCHEMA,
        "frozen_at_utc": frozen_at,
        "plan_sha256": plan["plan_sha256"],
        "preregistration_sha256": canonical_sha256(loaded["preregistration"]),
        "instrument_commit_sha256": plan["instrument_commit_sha256"],
        # Snapshot the completed pre-source artifact set.  ``files`` is
        # extended below with the receipt's own descriptor after the receipt
        # has been serialized; retaining the mutable mapping here would make
        # the returned in-memory receipt disagree with the on-disk evidence.
        "artifact_descriptors": copy.deepcopy(files),
        "source_capture_allowed": True,
        "scientific_status": "NOT_EVALUATED",
        "retune_after_freeze": False,
        "archived_16k_failure_preserved": True,
        "archived_outcomes_used_for_threshold_selection": False,
        "demo_or_nudge_controls_accepted": False,
        "claim_boundary": (
            "This operational receipt records a pre-source disk freeze. Its wall-clock "
            "time is ordering metadata, not a scientific result or a trusted timestamp."
        ),
    }
    receipt_path = freeze_root / "freeze_receipt.json"
    descriptor = _write_exclusive_json(receipt_path, freeze_receipt)
    descriptor["path"] = receipt_path.relative_to(output_root).as_posix()
    files["freeze_receipt"] = descriptor
    return files, frozen_at, freeze_receipt


def _current_postrun_row(postrun: Mapping[str, Any]) -> dict[str, Any]:
    campaign = postrun.get("campaign")
    if not isinstance(campaign, Mapping):
        raise PhysicalCampaignError("canonical postrun campaign is missing")
    rows = campaign.get("run_matrix")
    if not isinstance(rows, list):
        raise PhysicalCampaignError("canonical postrun run_matrix is missing")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("preflight") == "PASS"]
    if len(selected) != 1:
        raise PhysicalCampaignError("canonical postrun did not select exactly one cell")
    return dict(selected[0])


def _require_prerequisite_rung_readiness(
    package: Mapping[str, Any],
    *,
    current_rung: int,
    prerequisite_run_directories: Sequence[str | Path],
) -> dict[str, Any] | None:
    """Fresh-replay all lower-rung cells before admitting a larger rung.

    The 4k rung has no prerequisites.  Every later rung must be authorized by
    the independent family reducer under the same frozen family, instrument,
    and numerical runtime.  A caller-authored aggregate JSON is deliberately
    not accepted: the reducer reopens and replays the supplied cell bundles.
    """

    if current_rung == REQUIRED_RUNGS[0]:
        return None
    directories = list(prerequisite_run_directories)
    if not directories:
        raise PhysicalCampaignError(
            f"rung {current_rung} requires replay-verified lower-rung cell bundles"
        )
    aggregate = aggregate_physical_h3_kms_family(directories)
    if aggregate.get("aggregation_instrument_status") != "VALID_PASS":
        raise PhysicalCampaignError(
            "prerequisite family aggregation is instrument invalid: "
            + ",".join(str(value) for value in aggregate.get("blockers", []))
        )

    preregistration = package.get("preregistration")
    plan = (
        preregistration.get("plan")
        if isinstance(preregistration, Mapping)
        else None
    )
    if not isinstance(plan, Mapping):
        raise PhysicalCampaignError("constructed campaign plan is missing")
    expected_family = {
        "campaign_id": plan.get("campaign_id"),
        "instrument_version": plan.get("instrument_version"),
        "instrument_commit": plan.get("instrument_commit_sha256"),
        "frozen_campaign_family_sha256": frozen_campaign_family_sha256(plan),
        "seeds": list(plan.get("seeds", [])),
        "rungs": list(plan.get("rungs", [])),
        "replicate_ids": list(plan.get("replicate_ids", [])),
    }
    mismatches = [
        field
        for field, expected in expected_family.items()
        if aggregate.get(field) != expected
    ]
    if mismatches:
        raise PhysicalCampaignError(
            "prerequisite cells belong to a different frozen family: "
            + ",".join(mismatches)
        )
    ready_by_rung = aggregate.get("ready_for_rung")
    if (
        not isinstance(ready_by_rung, Mapping)
        or ready_by_rung.get(str(current_rung)) is not True
    ):
        raise PhysicalCampaignError(
            f"prerequisite family is not ready for rung {current_rung}"
        )
    return dict(aggregate)


def run_frozen_campaign_cell(
    output_dir: str | Path,
    *,
    current_seed: int = DEFAULT_SEEDS[0],
    current_rung: int = REQUIRED_RUNGS[0],
    current_replicate_id: str = DEFAULT_REPLICATE_IDS[0],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    replicate_ids: Sequence[str] = DEFAULT_REPLICATE_IDS,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    historical_campaign_receipt: str | Path | None = None,
    prerequisite_run_directories: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Run and disk-replay one physical cell under a complete frozen plan."""

    try:
        require_canonical_numerical_runtime()
    except NumericalRuntimeError as exc:
        raise PhysicalCampaignError(str(exc)) from exc
    historical_receipt_bytes, historical_witness = _load_historical_campaign_receipt(
        historical_campaign_receipt
    )
    package = build_frozen_campaign(
        current_seed=current_seed,
        current_rung=current_rung,
        current_replicate_id=current_replicate_id,
        seeds=seeds,
        replicate_ids=replicate_ids,
        campaign_id=campaign_id,
    )
    _require_prerequisite_rung_readiness(
        package,
        current_rung=current_rung,
        prerequisite_run_directories=prerequisite_run_directories,
    )
    output_root = Path(output_dir)
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PhysicalCampaignError(
            "output directory must not already exist; campaign evidence is append-never"
        ) from exc
    if output_root.is_symlink() or not output_root.is_dir():
        raise PhysicalCampaignError("output directory must be a real directory")

    freeze_files, freeze_completed_at, freeze_receipt = _write_and_verify_freeze(
        output_root,
        package,
        historical_receipt_bytes,
        historical_witness,
    )
    source_started_at = _utc_now()
    try:
        source_capture = capture_physical_source(
            package["preregistration_report"]["source_inputs"]
        )
        source_completed_at = _utc_now()
        replay_manifest = write_physical_h3_kms_replay_bundle(
            output_root / "replay_bundle",
            package["preregistration"],
            source_capture,
            package["calibration_artifacts"],
            freeze_receipt=freeze_receipt,
            historical_campaign_receipt_bytes=historical_receipt_bytes,
            numerical_runtime=package["numerical_runtime"],
        )
        replay_completed_at = _utc_now()
        replay_report = replay_physical_h3_kms_bundle(replay_manifest)
        replay_verified_at = _utc_now()
        if replay_report.get("PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT") is not True:
            raise PhysicalCampaignError(
                "canonical disk replay failed: "
                + ",".join(str(item) for item in replay_report.get("blockers", []))
            )

        postrun_path = (
            output_root
            / "replay_bundle"
            / "artifacts"
            / "postrun_report.json"
        )
        postrun = _read_exact_json(postrun_path)
        if not isinstance(postrun, Mapping):
            raise PhysicalCampaignError("canonical postrun artifact is not an object")
        current_row = _current_postrun_row(postrun)
        all_rows = postrun["campaign"]["run_matrix"]
        campaign_complete = bool(
            all_rows
            and all(
                row.get("powered_and_complete") is True
                and row.get("status") in {"VALID_PASS", "VALID_FAIL"}
                for row in all_rows
            )
        )
        promotion_allowed = bool(
            replay_report["PHYSICAL_H3_KMS_DISK_REPLAY_INSTRUMENT_RECEIPT"]
            and campaign_complete
            and all(row.get("status") == "VALID_PASS" for row in all_rows)
        )
        reports = source_capture.get("reports")
        if not isinstance(reports, Mapping):
            raise PhysicalCampaignError("source capture reports are missing")
        export_values = {
            "config.json": source_capture.get("config"),
            "physical_source_observer_contract_report.json": reports.get(
                "source_observer"
            ),
            "physical_h3_kms_refinement_report.json": reports.get("refinement"),
            "prime_geometric_cap_state_report.json": reports.get(
                "prime_geometric_state"
            ),
            "physical_h3_kms_independent_geometry_report.json": reports.get(
                "independent_geometry"
            ),
            "physical_h3_kms_native_bw_payload.json": postrun.get("native_bw"),
            "physical_h3_kms_candidate_interventions_report.json": postrun.get(
                "candidate_interventions"
            ),
            "physical_h3_kms_geometry_controls_report.json": postrun.get(
                "geometry_controls"
            ),
            "semantic_event_reconstruction_report.json": postrun.get(
                "semantic_event"
            ),
            "physical_h3_kms_campaign_manifest.json": postrun.get("campaign"),
            "physical_h3_kms_replay_verification.json": replay_report,
        }
        if any(not isinstance(value, Mapping) for value in export_values.values()):
            raise PhysicalCampaignError(
                "canonical source/postrun bundle is missing a preflight report"
            )
        preflight_exports: dict[str, dict[str, Any]] = {}
        for filename, value in export_values.items():
            descriptor = _write_exclusive_json(output_root / filename, value)
            descriptor["path"] = filename
            preflight_exports[filename] = descriptor
        run_receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "instrument_status": "VALID_PASS",
            "cell_scientific_status": current_row.get("status"),
            "postrun_scientific_failures": list(
                postrun.get("postrun_scientific_failures", [])
            ),
            "postrun_not_evaluated_reasons": list(
                postrun.get("postrun_not_evaluated_reasons", [])
            ),
            "campaign_complete": campaign_complete,
            "physical_promotion_allowed": promotion_allowed,
            "plan_sha256": package["preregistration"]["plan"]["plan_sha256"],
            "source_capture_sha256": source_capture.get("capture_sha256"),
            "replay_manifest_path": replay_manifest.relative_to(output_root).as_posix(),
            "replay_manifest_byte_sha256": _byte_sha256(replay_manifest.read_bytes()),
            "freeze_artifacts": freeze_files,
            "preflight_exports": preflight_exports,
            "freeze_completed_at_utc": freeze_completed_at,
            "source_capture_started_at_utc": source_started_at,
            "source_capture_completed_at_utc": source_completed_at,
            "replay_bundle_completed_at_utc": replay_completed_at,
            "replay_verified_at_utc": replay_verified_at,
            "retune_after_freeze": False,
            "demo_or_nudge_controls_accepted": False,
            "claim_boundary": (
                "Instrument VALID_PASS means exact disk replay succeeded. The cell's "
                "scientific status is reported separately; one cell cannot complete or "
                "promote the frozen multi-seed four-rung campaign."
            ),
        }
        _write_exclusive_json(output_root / "campaign_run_receipt.json", run_receipt)
        return run_receipt
    except Exception as exc:
        failure = {
            "schema": RUN_RECEIPT_SCHEMA,
            "instrument_status": "INSTRUMENT_INVALID",
            "cell_scientific_status": "NOT_EVALUATED",
            "physical_promotion_allowed": False,
            "plan_sha256": package["preregistration"]["plan"]["plan_sha256"],
            "freeze_artifacts": freeze_files,
            "freeze_completed_at_utc": freeze_completed_at,
            "source_capture_started_at_utc": source_started_at,
            "failed_at_utc": _utc_now(),
            "failure": f"{type(exc).__name__}:{exc}",
            "retune_after_freeze": False,
            "demo_or_nudge_controls_accepted": False,
            "claim_boundary": (
                "This failure receipt is operational evidence only. It does not convert "
                "missing or invalid instrumentation into a scientific failure."
            ),
        }
        failure_path = output_root / "campaign_run_receipt.json"
        if not failure_path.exists():
            _write_exclusive_json(failure_path, failure)
        if isinstance(exc, PhysicalCampaignError):
            raise
        raise PhysicalCampaignError(str(exc)) from exc


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one exact, frozen, non-demo physical H3/KMS cell."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--rung", type=int, choices=REQUIRED_RUNGS, default=REQUIRED_RUNGS[0])
    parser.add_argument(
        "--execute-physical-cell",
        action="store_true",
        help="required acknowledgement; there is no demo or forced-receipt mode",
    )
    parser.add_argument(
        "--historical-campaign-receipt",
        type=Path,
        default=DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT,
        help="exact archived receipt that binds the existing 16k negative cell",
    )
    parser.add_argument(
        "--prerequisite-run-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "replay-verified lower-rung cell directory; repeat for every frozen "
            "seed/replicate prerequisite when running above 4k"
        ),
    )
    args = parser.parse_args(argv)
    if not args.execute_physical_cell:
        parser.error("--execute-physical-cell is required")
    receipt = run_frozen_campaign_cell(
        args.output_dir,
        current_seed=args.seed,
        current_rung=args.rung,
        historical_campaign_receipt=args.historical_campaign_receipt,
        prerequisite_run_directories=args.prerequisite_run_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public API
    raise SystemExit(_main())


__all__ = [
    "CAMPAIGN_PACKAGE_SCHEMA",
    "DEFAULT_CAMPAIGN_ID",
    "DEFAULT_SEEDS",
    "DEFAULT_HISTORICAL_CAMPAIGN_RECEIPT",
    "FREEZE_RECEIPT_SCHEMA",
    "HISTORICAL_CAMPAIGN_RECEIPT_BYTE_SHA256",
    "HISTORICAL_CAMPAIGN_SHA256",
    "PhysicalCampaignError",
    "RUN_RECEIPT_SCHEMA",
    "build_frozen_campaign",
    "independent_calibration_artifacts",
    "run_frozen_campaign_cell",
]
