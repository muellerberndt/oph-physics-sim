"""OL-A1 Tier A signature replication driver (instrument INS-01).

Preregistered replication instrument for observation-ledger row OL-A1 in
`reverse-engineering-reality` (lane #737, register row INS-01). The driver
consumes a frozen campaign spec JSON (three arms, one fresh seed, five
declared replicate ids, the preregistration document sha256), runs the
pinned physical source capture once per (arm, replicate), computes the
declared observables O1 through O6 and the two declared controls, and
writes per-run receipts plus a campaign summary containing the frozen
decision-rule evaluation and the P0 conformance block.

The pinned `oph_fpe/` tree and `scripts/einstein_convergence_ladder.py`
are imported read-only. The estimator path matches the archived ladder
rows byte for byte: the same event table, the same capped pair classes,
the same restricted spectral chart, and the same quadratic-form fit with
its 1e-12 inertia threshold. The ancestry-permutation control reuses the
captured data; it adds zero source runs.

The decision rule lives in
`docs/OL_A1_PREREGISTERED_SIGNATURE_REPLICATION_2026-08-12.md` and is
frozen before the seed is drawn. FAILED and INCONCLUSIVE verdicts are
written with the same structure and prominence as REPLICATED. A P0
nonconformance voids the campaign as an execution error; it authorizes
no seed re-draw.

Usage:
    .venv/bin/python scripts/ol_a1_signature_replication.py \
        --spec docs/OL_A1_SEED_TABLE_2026-08-12.json \
        --out data/ol_a1_replication
    .venv/bin/python scripts/ol_a1_signature_replication.py \
        --spec docs/OL_A1_SEED_TABLE_2026-08-12.json \
        --out data/ol_a1_replication --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import einstein_convergence_ladder as ladder  # noqa: E402
from oph_fpe.bulk.event_manifold_producer import (  # noqa: E402
    _event_chart,
    _event_table,
    _fit_quadratic_form,
    _spectral_embedding,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import (  # noqa: E402
    capture_physical_source,
)

SPEC_SCHEMA = "oph.ol-a1-signature-replication.spec.v1"
RECEIPT_SCHEMA = "oph.ol-a1-signature-replication.receipt.v1"
SUMMARY_SCHEMA = "oph.ol-a1-signature-replication.summary.v1"
MANIFEST_SCHEMA = "oph.ol-a1-signature-replication.manifest.v1"
CAMPAIGN_ID = "ol_a1_tier_a_2026-08-12"
PREREG_DOC = "docs/OL_A1_PREREGISTERED_SIGNATURE_REPLICATION_2026-08-12.md"

ROBUST_TAU = 1.0e-3
THRESHOLD = 1.0e-12
MARGIN_RATIO_BAND = (0.35, 0.80)
MARGIN_RATIO_REFERENCE = 0.5725
DEGENERACY_REFERENCE_BAND = (2.0e-5, 4.0e-5)
ANCESTRY_NULL_DOMAIN = "ol_a1_ancestry_null"
REPLICATE_IDS = ("ola1.r1", "ola1.r2", "ola1.r3", "ola1.r4", "ola1.r5")
SCALE_ARMS = ("A1", "A2")
CONTROL_ARM = "C1"
SCALE_ROBUST_REFERENCE = (1, 2, 1)
ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "A1",
        "role": "scale_row",
        "carrier_count": 16_384,
        "observer_count": 128,
        "observer_support_size": 96,
        "reference_robust_inertia": [1, 2, 1],
    },
    {
        "arm_id": "A2",
        "role": "scale_row",
        "carrier_count": 65_536,
        "observer_count": 256,
        "observer_support_size": 96,
        "reference_robust_inertia": [1, 2, 1],
    },
    {
        "arm_id": "C1",
        "role": "support_density_control",
        "carrier_count": 16_384,
        "observer_count": 128,
        "observer_support_size": 6,
        "reference_robust_inertia": [2, 1, 1],
    },
)
PINNED_FILES = (
    "oph_fpe/bulk/event_manifold_producer.py",
    "oph_fpe/bulk/physical_h3_kms_source_capture.py",
    "scripts/einstein_convergence_ladder.py",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class CampaignSpecError(ValueError):
    """Raised when the frozen campaign spec fails validation."""


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "zlib": zlib.ZLIB_VERSION,
    }


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"head": head, "clean": porcelain.strip() == ""}


def load_spec(spec_path: Path) -> dict[str, Any]:
    """Load and validate the frozen campaign spec against driver constants."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != SPEC_SCHEMA:
        raise CampaignSpecError("spec_schema_mismatch")
    if spec.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignSpecError("spec_campaign_id_mismatch")
    seed = spec.get("seed")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise CampaignSpecError("spec_seed_invalid")
    if tuple(spec.get("replicate_ids", ())) != REPLICATE_IDS:
        raise CampaignSpecError("spec_replicate_ids_mismatch")
    declared_arms = spec.get("arms")
    expected_arms = [dict(arm) for arm in ARM_SPECS]
    if declared_arms != expected_arms:
        raise CampaignSpecError("spec_arms_mismatch")
    prereg = spec.get("preregistration")
    if (
        not isinstance(prereg, Mapping)
        or prereg.get("path") != PREREG_DOC
        or not isinstance(prereg.get("sha256"), str)
    ):
        raise CampaignSpecError("spec_preregistration_block_invalid")
    pinned = spec.get("pinned_code_sha256")
    if not isinstance(pinned, Mapping) or set(pinned) != set(PINNED_FILES):
        raise CampaignSpecError("spec_pinned_code_block_invalid")
    return spec


def _source_config(
    arm: Mapping[str, Any],
    seed: int,
    replicate_id: str,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "carrier_count": int(arm["carrier_count"]),
        "cycles": 16,
        "seed": seed,
        "observer_count": int(arm["observer_count"]),
        "observer_support_size": int(arm["observer_support_size"]),
        "observer_samples": 6,
        "observer_cross_reads": True,
        "snapshot_coverage": "spanning",
        "geometry_transport": "held_out_flow",
        "replicate_id": replicate_id,
        "preregistered_plan_sha256": plan_sha256,
    }


def _robust_inertia(eigenvalues: Sequence[float]) -> list[int]:
    """Return [positive, negative, degenerate] at the declared relative tau."""

    values = np.asarray(eigenvalues, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return [0, 0, int(values.size)]
    cut = ROBUST_TAU * scale
    positive = int(np.count_nonzero(values > cut))
    negative = int(np.count_nonzero(values < -cut))
    return [positive, negative, int(values.size) - positive - negative]


def _degeneracy_ratio(eigenvalues: Sequence[float]) -> float | None:
    values = np.abs(np.asarray(eigenvalues, dtype=float))
    top = float(values.max())
    if top == 0.0:
        return None
    return float(values.min()) / top


def _pair_parity_counts(pairs: Sequence[tuple[int, int]]) -> dict[str, int]:
    even = sum(1 for i, j in pairs if (i + j) % 2 == 0)
    return {"kept": len(pairs), "parity_even": even, "parity_odd": len(pairs) - even}


def _ancestry_null_seed(arm_id: str, replicate_id: str) -> int:
    digest = hashlib.sha256(
        f"{ANCESTRY_NULL_DOMAIN}:{arm_id}:{replicate_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _fit_block(fit: Mapping[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "fitted": bool(fit.get("fitted")),
        "blocker": fit.get("blocker"),
    }
    if fit.get("fitted"):
        eigenvalues = [float(value) for value in fit["eigenvalues"]]
        block.update(
            {
                "eigenvalues": eigenvalues,
                "threshold_inertia": [int(v) for v in fit["inertia"]],
                "robust_inertia": _robust_inertia(eigenvalues),
                "degeneracy_ratio": _degeneracy_ratio(eigenvalues),
                "cone_margin": float(fit["cone_margin"]),
                "held_out_pair_count": int(fit["held_out_pair_count"]),
            }
        )
    return block


def run_one(
    arm: Mapping[str, Any],
    seed: int,
    replicate_id: str,
    plan_sha256: str,
) -> dict[str, Any]:
    """Run one (arm, replicate) cell and return its receipt."""

    config = _source_config(arm, seed, replicate_id, plan_sha256)
    capture = capture_physical_source(config)

    events = capture["postrun_capture"]["semantic_events"]
    key_to_event = {event["event_key"]: event for event in events}
    cross_observer_edges = sum(
        1
        for edge in capture["postrun_capture"]["raw_ancestry_relations"]
        if (key_to_event.get(edge["child_event_id"]) or {}).get("observer_token")
        != (key_to_event.get(edge["parent_event_id"]) or {}).get("observer_token")
    )

    table = _event_table(capture)
    pairs = ladder._pair_classes_capped(table)
    touched = sorted(
        {carrier for footprint in table["footprints"] for carrier in footprint}
    )
    index_of = {carrier: index for index, carrier in enumerate(touched)}
    sub = np.zeros((len(touched), len(touched)))
    for row in capture["postrun_capture"]["raw_overlap_relations"]:
        left = index_of.get(int(row["left_carrier_id"].rsplit("-", 1)[1]))
        right = index_of.get(int(row["right_carrier_id"].rsplit("-", 1)[1]))
        if left is not None and right is not None:
            sub[left, right] = sub[right, left] = 1.0
    embedding_small = _spectral_embedding(sub)
    embedding = np.zeros((int(config["carrier_count"]), 3))
    for carrier, index in index_of.items():
        embedding[carrier] = embedding_small[index]
    chart = _event_chart(table, embedding)

    fit_pairs = {"causal": pairs["causal"], "spacelike": pairs["spacelike"]}
    fit_parity0 = _fit_quadratic_form(chart, fit_pairs)
    fit_parity1 = _fit_quadratic_form(chart, fit_pairs, train_parity=1)

    null_seed = _ancestry_null_seed(str(arm["arm_id"]), replicate_id)
    permutation = np.random.Generator(np.random.PCG64(null_seed)).permutation(
        table["count"]
    )
    shuffled_reachable = {
        int(permutation[i]): {int(permutation[j]) for j in table["reachable"][i]}
        for i in range(table["count"])
    }
    null_pairs = ladder._pair_classes_capped(
        {**table, "reachable": shuffled_reachable}
    )
    null_fit = _fit_quadratic_form(
        chart,
        {"causal": null_pairs["causal"], "spacelike": null_pairs["spacelike"]},
    )
    pair_classes_changed = null_pairs["causal"] != pairs["causal"]

    parity0_block = _fit_block(fit_parity0)
    parity1_block = _fit_block(fit_parity1)
    null_block = _fit_block(null_fit)
    ancestry_destroyed = bool(
        parity0_block["fitted"]
        and null_block["fitted"]
        and null_block["threshold_inertia"] != parity0_block["threshold_inertia"]
    )
    concordance = {
        "threshold_concordant": bool(
            parity0_block["fitted"]
            and parity1_block["fitted"]
            and parity0_block["threshold_inertia"]
            == parity1_block["threshold_inertia"]
        ),
        "robust_concordant": bool(
            parity0_block["fitted"]
            and parity1_block["fitted"]
            and parity0_block["robust_inertia"] == parity1_block["robust_inertia"]
        ),
    }

    return {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": str(arm["arm_id"]),
        "arm_role": str(arm["role"]),
        "replicate_id": replicate_id,
        "config": config,
        "capture_sha256": capture["capture_sha256"],
        "runtime_versions": _runtime_versions(),
        "observables": {
            "o1_threshold_inertia": parity0_block.get("threshold_inertia"),
            "o2_robust_inertia": parity0_block.get("robust_inertia"),
            "o3_degeneracy_ratio": parity0_block.get("degeneracy_ratio"),
            "o3_in_reference_band": (
                parity0_block.get("degeneracy_ratio") is not None
                and DEGENERACY_REFERENCE_BAND[0]
                <= parity0_block["degeneracy_ratio"]
                <= DEGENERACY_REFERENCE_BAND[1]
            ),
            "o4_cone_margin": parity0_block.get("cone_margin"),
            "o5_structural": {
                "event_count": int(table["count"]),
                "cross_observer_edges": int(cross_observer_edges),
                "causal_pairs_total": int(pairs["causal_total"]),
                "spacelike_pairs_total": int(pairs["spacelike_total"]),
                "causal_stride": int(pairs["causal_stride"]),
                "spacelike_stride": int(pairs["spacelike_stride"]),
                "causal_pair_parity": _pair_parity_counts(pairs["causal"]),
                "spacelike_pair_parity": _pair_parity_counts(pairs["spacelike"]),
                "held_out_pair_count": parity0_block.get("held_out_pair_count"),
            },
            "o6_split_half": {
                "parity1_fit": parity1_block,
                **concordance,
            },
        },
        "fit_parity0": parity0_block,
        "controls": {
            "c_ancestry": {
                "null_seed": null_seed,
                "null_seed_rule": (
                    "first eight bytes, big endian, of sha256 of "
                    f"'{ANCESTRY_NULL_DOMAIN}:<arm_id>:<replicate_id>'"
                ),
                "pair_classes_changed": bool(pair_classes_changed),
                "null_fit": null_block,
                "destroyed_measured_inertia": ancestry_destroyed,
                "preserved_1_3": bool(
                    null_block.get("threshold_inertia") == [1, 3]
                ),
            },
        },
    }


def _match_rate(flags: Sequence[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags)


def evaluate_decision(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen decision rule over the fifteen receipts.

    The rule is declared in the preregistration document. Gating readouts:
    O2 robust inertia on the scale arms, the O4 margin-magnitude ratio
    A2/A1 per replicate against the declared band, the C-ANCESTRY null,
    and the C-SUPPORT density control. O1, O3, O5, and O6 are recorded
    diagnostics; the O1 threshold rate is reported in every verdict
    sentence together with the near-null eigenvalue caution.
    """

    robust_match: dict[str, list[bool]] = {}
    destroyed: dict[str, list[bool]] = {}
    for arm_id in SCALE_ARMS:
        robust_match[arm_id] = [
            receipts[f"{arm_id}:{rid}"]["observables"]["o2_robust_inertia"]
            == list(SCALE_ROBUST_REFERENCE)
            for rid in REPLICATE_IDS
        ]
        destroyed[arm_id] = [
            bool(
                receipts[f"{arm_id}:{rid}"]["controls"]["c_ancestry"][
                    "destroyed_measured_inertia"
                ]
            )
            for rid in REPLICATE_IDS
        ]

    ratios: list[float | None] = []
    ratio_in_band: list[bool] = []
    for rid in REPLICATE_IDS:
        margin_a1 = receipts[f"A1:{rid}"]["observables"]["o4_cone_margin"]
        margin_a2 = receipts[f"A2:{rid}"]["observables"]["o4_cone_margin"]
        if margin_a1 is None or margin_a2 is None or margin_a1 == 0.0:
            ratios.append(None)
            ratio_in_band.append(False)
            continue
        ratio = abs(float(margin_a2)) / abs(float(margin_a1))
        ratios.append(ratio)
        ratio_in_band.append(
            MARGIN_RATIO_BAND[0] <= ratio <= MARGIN_RATIO_BAND[1]
        )

    c1_scale_match = [
        receipts[f"{CONTROL_ARM}:{rid}"]["observables"]["o2_robust_inertia"]
        == list(SCALE_ROBUST_REFERENCE)
        for rid in REPLICATE_IDS
    ]
    c1_reference_match = [
        receipts[f"{CONTROL_ARM}:{rid}"]["observables"]["o2_robust_inertia"]
        == [2, 1, 1]
        for rid in REPLICATE_IDS
    ]
    cross_a1 = [
        receipts[f"A1:{rid}"]["observables"]["o5_structural"][
            "cross_observer_edges"
        ]
        for rid in REPLICATE_IDS
    ]
    cross_c1 = [
        receipts[f"{CONTROL_ARM}:{rid}"]["observables"]["o5_structural"][
            "cross_observer_edges"
        ]
        for rid in REPLICATE_IDS
    ]
    mean_cross_a1 = sum(cross_a1) / len(cross_a1)
    mean_cross_c1 = sum(cross_c1) / len(cross_c1)
    rate_a1 = _match_rate(robust_match["A1"])
    rate_c1 = _match_rate(c1_scale_match)
    support_degradation = rate_c1 < rate_a1
    support_edges_halved = mean_cross_c1 <= 0.5 * mean_cross_a1
    support_fired = support_degradation and support_edges_halved
    support_inverted = rate_c1 >= rate_a1 and rate_a1 >= 0.8

    failed_clauses = {
        "f_a_robust_breakdown": any(
            sum(robust_match[arm_id]) <= 2 for arm_id in SCALE_ARMS
        ),
        "f_b_ratio_band_breakdown": sum(ratio_in_band) <= 2,
        "f_c_ancestry_null_preserved": any(
            sum(destroyed[arm_id]) <= 3 for arm_id in SCALE_ARMS
        ),
        "f_d_support_control_inverted": support_inverted,
    }
    replicated_clauses = {
        "r_a_robust_signature": all(
            sum(robust_match[arm_id]) >= 4 for arm_id in SCALE_ARMS
        ),
        "r_b_ratio_in_band": sum(ratio_in_band) >= 4,
        "r_c_ancestry_null_fired": all(
            sum(destroyed[arm_id]) >= 4 for arm_id in SCALE_ARMS
        ),
        "r_d_support_control_fired": support_fired,
    }
    if any(failed_clauses.values()):
        verdict = "FAILED"
    elif all(replicated_clauses.values()):
        verdict = "REPLICATED"
    else:
        verdict = "INCONCLUSIVE"

    threshold_13_rate = {
        arm_id: sum(
            receipts[f"{arm_id}:{rid}"]["observables"]["o1_threshold_inertia"]
            == [1, 3]
            for rid in REPLICATE_IDS
        )
        for arm_id in SCALE_ARMS
    }

    return {
        "verdict": verdict,
        "failed_clauses": failed_clauses,
        "replicated_clauses": replicated_clauses,
        "per_replicate": {
            "robust_match": {k: list(v) for k, v in robust_match.items()},
            "ancestry_destroyed": {k: list(v) for k, v in destroyed.items()},
            "margin_ratios_a2_over_a1": ratios,
            "margin_ratio_in_band": ratio_in_band,
            "c1_scale_signature_match": c1_scale_match,
            "c1_reference_robust_match": c1_reference_match,
        },
        "c_support": {
            "scale_signature_rate_a1": rate_a1,
            "scale_signature_rate_c1": rate_c1,
            "mean_cross_edges_a1": mean_cross_a1,
            "mean_cross_edges_c1": mean_cross_c1,
            "degradation": support_degradation,
            "edges_halved": support_edges_halved,
            "fired": support_fired,
            "inverted": support_inverted,
        },
        "reported_threshold_1_3_count_of_5": threshold_13_rate,
        "caution": (
            "The archived (1,3) threshold verdict rides on one eigenvalue "
            "at relative magnitude about 3e-5 against a 1e-12 threshold; "
            "at relative tau 1e-3 the robust inertia of the retained rungs "
            "is (1,2) with one degenerate direction, and a fresh seed can "
            "flip (1,3) to (2,2) with no change in physical content. N = 5 "
            "resolves seed fragility at the 1-in-5 level only."
        ),
    }


def build_p0_block(
    spec: Mapping[str, Any],
    spec_path: Path,
    git_state: Mapping[str, Any],
    runtime_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    prereg_path = REPO_ROOT / str(spec["preregistration"]["path"])
    prereg_ok = (
        prereg_path.is_file()
        and _sha256_file(prereg_path) == spec["preregistration"]["sha256"]
    )
    pinned_checks = {
        name: {
            "declared": spec["pinned_code_sha256"][name],
            "measured": _sha256_file(REPO_ROOT / name),
        }
        for name in PINNED_FILES
    }
    pinned_ok = all(
        row["declared"] == row["measured"] for row in pinned_checks.values()
    )
    env_rows = {name: os.environ.get(name) for name in THREAD_ENV}
    env_ok = all(env_rows[name] == THREAD_ENV[name] for name in THREAD_ENV)
    uniform_runtime = len({json.dumps(row, sort_keys=True) for row in runtime_rows})
    checks = {
        "clean_tree_at_run_start": bool(git_state["clean"]),
        "preregistration_sha256_match": bool(prereg_ok),
        "pinned_code_sha256_match": bool(pinned_ok),
        "thread_pinning_env": bool(env_ok),
        "single_runtime_environment": uniform_runtime == 1,
        "declared_cell_count_executed": len(runtime_rows)
        == len(ARM_SPECS) * len(REPLICATE_IDS),
    }
    conformant = all(checks.values())
    return {
        "checks": checks,
        "git_head_at_run_start": git_state["head"],
        "spec_sha256": _sha256_file(spec_path),
        "pinned_code": pinned_checks,
        "thread_env": env_rows,
        "conformant": conformant,
        "run_status": "CONFORMANT" if conformant else "VOIDED_EXECUTION_ERROR",
        "nonconformance_policy": (
            "A P0 nonconformance voids the campaign as an execution error. "
            "It is reported as such and authorizes no seed re-draw."
        ),
    }


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def run_campaign(spec_path: Path, out_dir: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    for name, expected in THREAD_ENV.items():
        if os.environ.get(name) != expected:
            raise CampaignSpecError(f"thread_env_not_pinned:{name}")
    git_state = _git_state()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_sha256 = str(spec["preregistration"]["sha256"])
    receipts: dict[str, dict[str, Any]] = {}
    receipt_files: dict[str, str] = {}
    runtime_rows: list[dict[str, str]] = []
    for arm in ARM_SPECS:
        for replicate_id in REPLICATE_IDS:
            cell = f"{arm['arm_id']}:{replicate_id}"
            print(f"=== {cell} ===", flush=True)
            receipt = run_one(arm, int(spec["seed"]), replicate_id, plan_sha256)
            receipts[cell] = receipt
            runtime_rows.append(receipt["runtime_versions"])
            name = f"run_{arm['arm_id']}_{replicate_id}.json"
            _write_json(out_dir / name, receipt)
            receipt_files[cell] = name
            print(
                f"{cell}: events={receipt['observables']['o5_structural']['event_count']} "
                f"o1={receipt['observables']['o1_threshold_inertia']} "
                f"o2={receipt['observables']['o2_robust_inertia']} "
                f"margin={receipt['observables']['o4_cone_margin']}",
                flush=True,
            )

    decision = evaluate_decision(receipts)
    p0 = build_p0_block(spec, spec_path, git_state, runtime_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "instrument": "INS-01",
        "ledger_row": "OL-A1",
        "spec": {
            "path": _repo_relative(spec_path),
            "seed": int(spec["seed"]),
            "replicate_ids": list(REPLICATE_IDS),
            "arms": [dict(arm) for arm in ARM_SPECS],
            "preregistration": dict(spec["preregistration"]),
        },
        "declared_constants": {
            "robust_tau_relative": ROBUST_TAU,
            "threshold": THRESHOLD,
            "margin_ratio_band": list(MARGIN_RATIO_BAND),
            "margin_ratio_reference": MARGIN_RATIO_REFERENCE,
            "degeneracy_reference_band": list(DEGENERACY_REFERENCE_BAND),
        },
        "p0_conformance": p0,
        "decision": decision,
        "receipt_files": receipt_files,
        "equal_prominence": (
            "FAILED and INCONCLUSIVE are reported with the same structure "
            "and prominence as REPLICATED."
        ),
    }
    _write_json(out_dir / "campaign_summary.json", summary)

    manifest_rows = {}
    for name in sorted([*receipt_files.values(), "campaign_summary.json"]):
        manifest_rows[name] = _sha256_file(out_dir / name)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "files": manifest_rows,
        "spec_sha256": _sha256_file(spec_path),
        "preregistration_sha256": plan_sha256,
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(
        f"VERDICT={decision['verdict']} run_status={p0['run_status']}",
        flush=True,
    )
    return summary


def validate_campaign(spec_path: Path, out_dir: Path) -> None:
    """Re-derive the decision rule and manifest hashes from stored receipts."""

    spec = load_spec(spec_path)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CampaignSpecError("manifest_schema_mismatch")
    for name, digest in manifest["files"].items():
        if _sha256_file(out_dir / name) != digest:
            raise CampaignSpecError(f"manifest_hash_mismatch:{name}")
    receipts: dict[str, dict[str, Any]] = {}
    for arm in ARM_SPECS:
        for replicate_id in REPLICATE_IDS:
            name = f"run_{arm['arm_id']}_{replicate_id}.json"
            receipt = json.loads((out_dir / name).read_text(encoding="utf-8"))
            if receipt.get("schema") != RECEIPT_SCHEMA:
                raise CampaignSpecError(f"receipt_schema_mismatch:{name}")
            if receipt["config"]["seed"] != spec["seed"]:
                raise CampaignSpecError(f"receipt_seed_mismatch:{name}")
            receipts[f"{arm['arm_id']}:{replicate_id}"] = receipt
    summary = json.loads(
        (out_dir / "campaign_summary.json").read_text(encoding="utf-8")
    )
    decision = evaluate_decision(receipts)
    if decision != summary.get("decision"):
        raise CampaignSpecError("decision_rule_readback_mismatch")
    print(f"CAMPAIGN_VALID verdict={decision['verdict']}", flush=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("data/ol_a1_replication")
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate_only:
        validate_campaign(args.spec, args.out)
    else:
        run_campaign(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
