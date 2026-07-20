"""Renderer-only completion overlay for the physical H3/KMS P0--P8 ladder.

The physical campaign and its replay bundle are deliberately outside this
module.  This module accepts an already-produced preflight snapshot, copies it,
and derives a separate ``DEMO_ASSUMPTION`` presentation layer.  It never edits
the snapshot, never emits a physical receipt, and never authorizes promotion,
branch retirement, or a larger scale run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


DEMO_STATUS = "DEMO_ASSUMPTION"
DEMO_WATERMARK = "DEMO ASSUMPTION — VISUALIZATION ONLY — NOT PHYSICAL EVIDENCE"
PHYSICAL_H3_KMS_DEMO_OVERLAY_SCHEMA = "oph.physical-h3-kms.demo-overlay/1.0.0"
PHYSICAL_H3_KMS_PREFLIGHT_SCHEMA = "oph_physical_h3_kms_preflight_v3"
PHYSICAL_H3_KMS_PREFLIGHT_MODE = "physical_h3_kms_campaign_preflight"
RENDERER_SEED = "oph-physical-h3-kms-demo-overlay-v1"


_STAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "P0_source_dynamics_repair_record_observer",
        "Federated source, repair, records, and observer readback",
        "Illustrate light/readback settling and a repair normal form without claiming a physical repair theorem.",
    ),
    (
        "P1_nested_refinement_and_expectations",
        "Nested refinement and conditional expectations",
        "Illustrate a coherent refinement tower; the displayed convergence curve is not a physical tower receipt.",
    ),
    (
        "P2_prime_geometric_cap_state",
        "Prime geometric cap state",
        "Illustrate a positive finite cap state and modular spectrum without upgrading a constructed state to measured physics.",
    ),
    (
        "P3_independent_geometric_parameter",
        "Independent geometric-flow parameter",
        "Illustrate cross-ratio flow while preserving the requirement that geometric s and modular t be independently produced.",
    ),
    (
        "P4_native_bw01_bw08",
        "Finite Bisognano--Wichmann conditions",
        "Illustrate the eight BW handoffs; visual satisfaction is not support covariance or KMS evidence.",
    ),
    (
        "P5_frozen_candidate_interventions",
        "Frozen clock-candidate intervention",
        "Display 2pi as a post-exposure reference winner over 1x, pi, and 4pi; this is not an independent clock selection.",
    ),
    (
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage",
        "H3 against S2/E3/E4 controls",
        "Illustrate a same-holdout comparison whose H3 curve is visually separated; the scores are synthetic renderer data.",
    ),
    (
        "P7_semantic_event_e1_e4_and_frame_fiber_separation",
        "Semantic events and frame/fiber separation",
        "Illustrate event ancestry, causal links, and local frames without claiming a reconstructed 3+1D event manifold.",
    ),
    (
        "P8_frozen_multiseed_four_rung_campaign",
        "Frozen multi-seed 4k/16k/64k/256k ladder",
        "Illustrate stable presentation across the four registered rungs; this cannot authorize or retire a physical branch.",
    ),
)

PHYSICAL_H3_KMS_STAGE_IDS = tuple(row[0] for row in _STAGE_SPECS)


def build_physical_h3_kms_demo_overlay(
    physical_snapshot: Mapping[str, Any] | None,
    *,
    demo_enabled: bool,
    force_all_missing: bool,
) -> dict[str, Any]:
    """Return an immutable-snapshot P0--P8 display overlay.

    A stage is physically complete only when its copied preflight row reports
    both an instrument ``PASS`` and a ``VALID_PASS`` scientific status.  In
    force-all demo mode every other stage gets deterministic illustrative data
    under ``displayData``.  The raw stage row remains available only through the
    untouched physical snapshot and its digest.
    """

    snapshot, snapshot_blockers, snapshot_trusted = _copy_snapshot(physical_snapshot)
    before_digest = _canonical_digest(snapshot)
    raw_stages = snapshot.get("stages") if isinstance(snapshot.get("stages"), Mapping) else {}
    force_active = bool(demo_enabled and force_all_missing)

    nodes: list[dict[str, Any]] = []
    forced_stage_ids: list[str] = []
    for stage_id, label, boundary in _STAGE_SPECS:
        raw_stage = raw_stages.get(stage_id) if isinstance(raw_stages, Mapping) else None
        raw_stage = raw_stage if isinstance(raw_stage, Mapping) else {}
        gate_status = _status(raw_stage.get("gate_status"), fallback="NOT_EVALUATED")
        evidence = raw_stage.get("evidence") if isinstance(raw_stage.get("evidence"), Mapping) else {}
        scientific_status = _status(
            raw_stage.get("scientific_status", evidence.get("scientific_outcome")),
            fallback="NOT_EVALUATED",
        )
        instrument_passed = bool(
            snapshot_trusted
            and raw_stage.get("passed") is True
            and gate_status == "PASS"
        )
        physical_passed = bool(instrument_passed and scientific_status == "VALID_PASS")
        nudged = bool(force_active and not physical_passed)
        display_data, field_provenance = (
            _stage_display_data(stage_id) if nudged else ({}, [])
        )
        if nudged:
            forced_stage_ids.append(stage_id)
        nodes.append(
            {
                "stageId": stage_id,
                "label": label,
                "rawStagePresent": bool(raw_stage),
                "rawStageDigest": _canonical_digest(raw_stage),
                "physicalGateStatus": gate_status,
                "physicalScientificStatus": scientific_status,
                "physicalInstrumentPassed": instrument_passed,
                "physicalPassed": physical_passed,
                "demoNudgeApplied": nudged,
                "displayStatus": (
                    DEMO_STATUS
                    if nudged
                    else "COMPUTED"
                    if physical_passed
                    else gate_status
                ),
                "displayComplete": bool(physical_passed or nudged),
                "displayOutcome": (
                    "DISPLAY_COMPLETE" if physical_passed or nudged else "DISPLAY_OPEN"
                ),
                "displayData": display_data,
                "fieldProvenance": field_provenance,
                "epistemicStatus": DEMO_STATUS if nudged else "PHYSICAL_SNAPSHOT",
                "visualizationOnly": nudged,
                "watermark": DEMO_WATERMARK if nudged else None,
                "physicalStageStatusUnchanged": True,
                "promotion_allowed": False,
                "retirement_counting_allowed": False,
                "SCALE_CAMPAIGN_ALLOWED": False,
                "claimBoundary": boundary,
            }
        )

    after_digest = _canonical_digest(snapshot)
    return {
        "schema": PHYSICAL_H3_KMS_DEMO_OVERLAY_SCHEMA,
        "epistemicStatus": DEMO_STATUS if force_active else "VISUALIZATION_CONTRACT",
        "enabled": force_active,
        "activationRule": (
            "demo mode plus Force all missing display stages; never callable from the physical campaign"
        ),
        "rendererSeed": RENDERER_SEED,
        "stageOrder": list(PHYSICAL_H3_KMS_STAGE_IDS),
        "stageNodes": nodes,
        "forcedStageIds": forced_stage_ids,
        "displayComplete": bool(nodes and all(row["displayComplete"] for row in nodes)),
        "watermark": DEMO_WATERMARK if force_active else None,
        "physicalSnapshot": snapshot,
        "physicalSnapshotTrusted": snapshot_trusted,
        "physicalSnapshotBlockers": snapshot_blockers,
        "physicalSnapshotDigestBefore": before_digest,
        "physicalSnapshotDigestAfter": after_digest,
        "physicalSnapshotDigestPreserved": before_digest == after_digest,
        "physicalCampaignStatus": snapshot.get("campaign_status", "INCOMPLETE"),
        "physicalPromotionSnapshotValue": snapshot.get(
            "physical_promotion_allowed", False
        ),
        "physicalRetirementSnapshotValue": snapshot.get(
            "retirement_counting_allowed", False
        ),
        "displayGuards": {
            "scientific_receipts_unchanged": True,
            "physical_stage_statuses_unchanged": True,
            "physical_snapshot_mutated": False,
            "promotion_allowed": False,
            "retirement_counting_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
            "target_ancestry_eligible": False,
            "may_write_campaign_artifacts": False,
        },
        "receipts": {
            "DEMO_OVERLAY_RENDER_COMPLETE": bool(
                force_active and nodes and all(row["displayComplete"] for row in nodes)
            ),
            "PHYSICAL_SNAPSHOT_DIGEST_PRESERVED": before_digest == after_digest,
            "PHYSICAL_STAGE_STATUS_MUTATED": False,
            "PHYSICAL_SCIENTIFIC_RECEIPT_MUTATED": False,
            "PHYSICAL_PROMOTION_RECEIPT": False,
            "BRANCH_RETIREMENT_AUTHORIZED": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
            "promotion_allowed": False,
            "retirement_counting_allowed": False,
        },
        "claimBoundary": (
            "The overlay completes only a renderer narrative. Its residuals, spectra, "
            "clock winner, model scores, events, and rung stability are deterministic "
            "DEMO_ASSUMPTION data and cannot pass P0--P8, select 2pi physically, retire "
            "the H3 branch, or authorize 16k/64k/256k work."
        ),
    }


def _stage_display_data(
    stage_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields = _display_field_specs(stage_id)
    data = {field: spec[0] for field, spec in fields.items()}
    provenance = [
        {
            "fieldPath": f"stageNodes[{stage_id}].displayData.{field}",
            "provenanceClass": spec[1],
            "method": spec[2],
            "sourceRefs": list(spec[3]),
            "epistemicStatus": DEMO_STATUS,
            "physicalReceiptEligible": False,
            "targetAncestryEligible": False,
        }
        for field, spec in fields.items()
    ]
    return data, provenance


def _display_field_specs(
    stage_id: str,
) -> dict[str, tuple[Any, str, str, tuple[str, ...]]]:
    synthetic = ("synthetic", "deterministic_renderer_fixture_v1", ("rendererSeed",))
    frozen = (
        "frozen_reference",
        "post_exposure_display_reference_v1",
        ("demoTargets.clock_normalization",),
    )
    fields: dict[str, dict[str, tuple[Any, str, str, tuple[str, ...]]]] = {
        "P0_source_dynamics_repair_record_observer": {
            "repairResidual": ([1.0, 0.56, 0.27, 0.11, 0.035, 0.006, 0.0], *synthetic),
            "carrierPulsePhase": ([0.0, 0.17, 0.36, 0.58, 0.79, 1.0], *synthetic),
            "terminalNormalFormId": ("demo-observable-normal-form-000", *synthetic),
        },
        "P1_nested_refinement_and_expectations": {
            "refinementLevels": ([1, 2, 3, 4], *synthetic),
            "expectationResidual": ([0.18, 0.071, 0.022, 0.006], *synthetic),
        },
        "P2_prime_geometric_cap_state": {
            "displayDensityEigenvalues": ([0.46, 0.28, 0.17, 0.09], *synthetic),
            "displayModularSpectrum": ([0.776529, 1.272966, 1.771957, 2.407946], *synthetic),
        },
        "P3_independent_geometric_parameter": {
            "geometricFlowParameter": ([0.0, 0.4, 0.8, 1.2], *synthetic),
            "displayCrossRatio": ([1.0, 1.491825, 2.225541, 3.320117], *synthetic),
            "independenceClaimed": (False, *synthetic),
        },
        "P4_native_bw01_bw08": {
            "bwDisplayConditions": (
                [
                    {"conditionId": f"BW{index:02d}", "displayState": "ILLUSTRATED"}
                    for index in range(1, 9)
                ],
                *synthetic,
            ),
        },
        "P5_frozen_candidate_interventions": {
            "candidateDisplayLoss": (
                {"1x": 0.73, "pi": 0.34, "2pi": 0.01, "4pi": 0.61},
                *synthetic,
            ),
            "displaySelection": ("2pi", *frozen),
            "independentPhysicalSelection": (False, *frozen),
        },
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage": {
            "sameHoldoutDisplayScore": (
                {"H3": 0.96, "S2": 0.62, "E3": 0.48, "E4": 0.55},
                *synthetic,
            ),
            "displayWinner": ("H3", *synthetic),
        },
        "P7_semantic_event_e1_e4_and_frame_fiber_separation": {
            "eventDisplayPoints": (
                [
                    {"eventId": "demo-event-0", "position": [0.0, 0.0, 0.0], "time": 0.0},
                    {"eventId": "demo-event-1", "position": [0.3, 0.1, 0.2], "time": 0.5},
                    {"eventId": "demo-event-2", "position": [0.55, 0.18, 0.38], "time": 1.0},
                ],
                *synthetic,
            ),
            "causalDisplayLinks": (
                [["demo-event-0", "demo-event-1"], ["demo-event-1", "demo-event-2"]],
                *synthetic,
            ),
        },
        "P8_frozen_multiseed_four_rung_campaign": {
            "rungDisplayRows": (
                [
                    {"carrierCount": 4096, "displayStability": 0.91},
                    {"carrierCount": 16384, "displayStability": 0.94},
                    {"carrierCount": 65536, "displayStability": 0.96},
                    {"carrierCount": 262144, "displayStability": 0.97},
                ],
                *synthetic,
            ),
            "physicalScaleAllowed": (False, *synthetic),
            "physicalBranchRetirement": (False, *synthetic),
        },
    }
    return fields[stage_id]


def _copy_snapshot(
    snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], bool]:
    if snapshot is None:
        return (
            {
                "schema": PHYSICAL_H3_KMS_PREFLIGHT_SCHEMA,
                "mode": PHYSICAL_H3_KMS_PREFLIGHT_MODE,
                "campaign_status": "INCOMPLETE",
                "physical_promotion_allowed": False,
                "retirement_counting_allowed": False,
                "stages": {},
            },
            ["physical_h3_kms_snapshot_not_supplied"],
            False,
        )
    if not isinstance(snapshot, Mapping):
        return {}, ["physical_h3_kms_snapshot_must_be_mapping"], False
    try:
        copied = _finite_json_copy(dict(snapshot))
    except (TypeError, ValueError) as exc:
        return {}, [f"physical_h3_kms_snapshot_not_finite_json:{exc}"], False
    trusted = bool(
        copied.get("schema") == PHYSICAL_H3_KMS_PREFLIGHT_SCHEMA
        and copied.get("mode") == PHYSICAL_H3_KMS_PREFLIGHT_MODE
    )
    return (
        copied,
        [] if trusted else ["untrusted_physical_h3_kms_snapshot_schema_or_mode"],
        trusted,
    )


def _status(value: Any, *, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _finite_json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEMO_STATUS",
    "DEMO_WATERMARK",
    "PHYSICAL_H3_KMS_DEMO_OVERLAY_SCHEMA",
    "PHYSICAL_H3_KMS_STAGE_IDS",
    "build_physical_h3_kms_demo_overlay",
]
