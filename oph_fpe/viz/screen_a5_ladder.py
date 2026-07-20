"""Exact icosahedral screen and A5-to-SM visualization contract.

This module is intentionally renderer-facing.  It exports exact finite
icosahedral combinatorics and the canonical physical A5/SM stage DAG, while
keeping forced/frozen demonstrations in an isolated ``DEMO_ASSUMPTION``
namespace.  Demo state never mutates the physical receipt snapshot and never
authorizes scientific promotion or scale campaigns.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from oph_fpe.gauge.physical_a5_sm_requirements import (
    BASE_GLOBAL_PASS_STAGES,
    FULL_INTERACTING_PASS_STAGES,
    REQUIREMENTS_REPORT_ARTIFACT_TYPE,
    REQUIREMENTS_REPORT_SCHEMA,
    STAGE_DAG_EDGES,
    STAGE_IDS,
    STAGE_SPECS,
    TerminalStatus,
)
from oph_fpe.viz.physical_h3_kms_demo_overlay import (
    build_physical_h3_kms_demo_overlay,
)


SCREEN_A5_LADDER_SCHEMA = "oph.screen-a5-visualization-ladder/1.0.0"
DEMO_UNIVERSE_SCHEMA = "oph.screen-a5-demo-universe/1.0.0"
DEMO_STATUS = "DEMO_ASSUMPTION"
DEMO_WATERMARK = "DEMO ASSUMPTION — VISUALIZATION ONLY — NOT PHYSICAL EVIDENCE"
MAX_RENDERED_CARRIER_SAMPLE = 12
DEMO_UNIVERSE_SEED = "oph-screen-a5-demo-universe-v1"
DEMO_RECORD_LIMIT = 32
DEMO_MAX_ATOM_CENSUS = 1_000_000
DEMO_PARTICLE_ACTOR_COUNT = 48
DEMO_WORLDLINE_FRAME_COUNT = 24
DEMO_OBSERVER_FRAME_COUNT = 24
DEMO_SOFTENED_GRAVITY_MODEL_ID = "demo-softened-gravity-v1"
DEMO_PROVENANCE_CLASSES = frozenset(
    {
        "run_anchored",
        "computed_exact",
        "interpolated",
        "synthetic",
        "frozen_reference",
    }
)

_TARGET_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/-]{0,127}$")
_ALLOWED_CONFIG_FIELDS = frozenset(
    {"enabled", "forceAllStages", "stages", "frozenTargets"}
)

DEFAULT_FROZEN_TARGETS: Mapping[str, Any] = {
    "clock_normalization": {"label": "2pi", "numericValue": 2.0 * math.pi},
    "gauge_lie_algebra": "su(3)+su(2)+u(1)",
    "global_gauge_form": "(SU(3)xSU(2)xU(1))/Z6",
    "generation_count": 3,
    "scalar_doublet_count": 1,
}


def build_screen_a5_ladder_payload(
    *,
    physical_receipt_snapshot: Mapping[str, Any] | None = None,
    physical_h3_kms_snapshot: Mapping[str, Any] | None = None,
    demo_config: Mapping[str, Any] | None = None,
    federation_carrier_count: int = 12,
) -> dict[str, Any]:
    """Build one fail-closed screen/A5 visualization payload.

    Invalid demo configuration is reported in-band and disables every force
    toggle.  The input physical snapshot is deep-copied before any display
    state is derived and is returned unchanged inside the payload.
    """

    snapshot, snapshot_blockers, snapshot_trusted = _copy_physical_snapshot(
        physical_receipt_snapshot
    )
    config = _parse_demo_config(demo_config)
    geometry = _exact_local_icosahedral_carrier()
    trusted_snapshot = snapshot if snapshot_trusted else {}
    stage_rows, forced_stage_ids = _stage_rows(trusted_snapshot, config)
    target_rows = _frozen_target_rows(config)
    federation = _federation_payload(
        federation_carrier_count,
        geometry=geometry,
    )
    physical_receipts = (
        trusted_snapshot.get("receipts", {})
        if isinstance(trusted_snapshot.get("receipts"), Mapping)
        else {}
    )
    physical_global_pass = physical_receipts.get("PHYSICAL_A5_SM_GLOBAL_PASS") is True
    demo_active = bool(config["enabled"] and (forced_stage_ids or target_rows))
    clock_demo_requested = any(
        row.get("targetId") == "clock_normalization" for row in target_rows
    )
    demo_universe = _demo_universe_payload(
        config=config,
        geometry=geometry,
        federation=federation,
        stage_rows=stage_rows,
        physical_snapshot_digest=_canonical_digest(snapshot),
    )
    physical_h3_kms_demo_overlay = build_physical_h3_kms_demo_overlay(
        physical_h3_kms_snapshot,
        demo_enabled=bool(config["valid"] and config["enabled"]),
        force_all_missing=bool(config["forceAllStages"]),
    )
    return {
        "schema": SCREEN_A5_LADDER_SCHEMA,
        "epistemicStatus": "VISUALIZATION_CONTRACT",
        "localCarrier": geometry,
        "federation": federation,
        "observerRepairBridge": _observer_repair_bridge_payload(),
        "a5ToSm": {
            "stageOrder": list(STAGE_IDS),
            "stageNodes": stage_rows,
            "stageEdges": _stage_edges(),
            "tierGroups": _tier_groups(),
            "forcedStageIds": forced_stage_ids,
            "physicalReceiptSnapshotDigest": _canonical_digest(snapshot),
            "physicalReceiptSnapshot": snapshot,
            "physicalSnapshotBlockers": snapshot_blockers,
            "physicalSnapshotTrusted": snapshot_trusted,
            "displayComplete": bool(
                stage_rows and all(row["displayComplete"] is True for row in stage_rows)
            ),
            "promotionAllowed": False,
            "promotion_allowed": False,
            "claimBoundary": (
                "Stage nodes reproduce the simulator contract. A forced display status is "
                "not a physical stage status and cannot enter a scientific conjunction."
            ),
        },
        "clockSeparation": _clock_separation_payload(
            snapshot=trusted_snapshot,
            demo_clock_requested=clock_demo_requested,
            target_rows=target_rows,
        ),
        "physicalH3KmsDemoOverlay": physical_h3_kms_demo_overlay,
        "demoUniverse": demo_universe,
        "demoControls": {
            "configurationStatus": config["status"],
            "configurationValid": config["valid"],
            "enabled": config["enabled"],
            "forceAllStages": config["forceAllStages"],
            "stageToggles": config["stages"],
            "frozenTargetRows": target_rows,
            "blockers": config["blockers"],
            "watermark": DEMO_WATERMARK if demo_active else None,
            "toggleCatalog": _toggle_catalog(),
            "ingestionBoundary": {
                "acceptedByPhysicalA5Verifier": False,
                "acceptedByProductionEnvelope": False,
                "visualizationOnly": True,
                "promotionAllowed": False,
                "scaleCampaignAllowed": False,
            },
        },
        "receipts": {
            "EXACT_ICOSAHEDRAL_12_30_20_RENDER_RECEIPT": True,
            "A5_ORDER60_ACTION_RENDER_RECEIPT": True,
            "A5_1_3_3PRIME_5_SECTOR_RENDER_RECEIPT": True,
            "FEDERATION_SEAM_RENDER_RECEIPT": True,
            "A5_SM_STAGE_DAG_RENDER_RECEIPT": True,
            "DEMO_ASSUMPTION_ACTIVE": demo_active,
            "PHYSICAL_A5_SM_GLOBAL_PASS": physical_global_pass,
            "PHYSICAL_A5_SM_SNAPSHOT_TRUSTED": snapshot_trusted,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
            "physical_receipt_snapshot_mutated": False,
        },
        "claimBoundary": (
            "Exact finite icosahedral/A5 mathematics and renderer completeness are not "
            "Standard-Model emergence evidence. Forced/frozen values are post-exposure "
            "display assumptions, are watermarked, and keep promotion and scale false."
        ),
    }


def screen_geometry_view_contract(ladder: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical ``screenGeometry`` visualization view."""

    receipts = ladder.get("receipts", {}) if isinstance(ladder, Mapping) else {}
    return {
        "viewId": "screenGeometry",
        "sectionKind": "local_icosahedral_carrier_federation",
        "label": "Local icosahedral carriers and global screen federation",
        "visualMetaphor": "federated_12_port_self_reading_cells",
        "description": (
            "Render one exact 12-port, 30-edge, 20-face oriented icosahedral carrier, "
            "then instance it across the federation with explicit seam channels. Keep "
            "each local carrier visually distinct from the global S2 support."
        ),
        "dataSources": [
            "screenA5Ladder.localCarrier",
            "screenA5Ladder.federation",
            "screenA5Ladder.observerRepairBridge",
        ],
        "primaryFields": [
            "screenA5Ladder.localCarrier.ports",
            "screenA5Ladder.localCarrier.edges",
            "screenA5Ladder.localCarrier.faces",
            "screenA5Ladder.federation.seams",
        ],
        "renderLayers": [
            {
                "layer": "local_icosahedral_ports",
                "source": "screenA5Ladder.localCarrier.ports",
            },
            {
                "layer": "local_icosahedral_edges",
                "source": "screenA5Ladder.localCarrier.edges",
            },
            {"layer": "oriented_faces", "source": "screenA5Ladder.localCarrier.faces"},
            {
                "layer": "antipodal_port_links",
                "source": "screenA5Ladder.localCarrier.antipodes",
            },
            {
                "layer": "federation_instances",
                "source": "screenA5Ladder.federation.carrierInstances",
            },
            {"layer": "repair_seams", "source": "screenA5Ladder.federation.seams"},
        ],
        "visualEncodings": [
            {
                "field": "local_vs_global",
                "source": "screenA5Ladder.federation.geometryDistinction",
                "encoding": "solid local cells over a separate translucent global S2 support",
                "palette": "local_global_geometry",
            },
            {
                "field": "seam_readback",
                "source": "screenA5Ladder.federation.seams",
                "encoding": "bidirectional port-to-port repair channels",
                "palette": "repair_seam",
            },
        ],
        "animationChannels": [
            {
                "channel": "observer_repair_flow",
                "source": "screenA5Ladder.observerRepairBridge.steps",
                "timeSource": None,
                "encoding": "pulse from port readback through seam comparison and repair commit",
            }
        ],
        "receipts": {
            "exact_icosahedral_render": receipts.get(
                "EXACT_ICOSAHEDRAL_12_30_20_RENDER_RECEIPT", False
            ),
            "physical_geometry_emergence": False,
            "promotion_allowed": False,
        },
        "exportSufficiency": "exact_finite_carrier_and_federation_render_only",
        "promotionReceiptsRequired": [
            "GEOMETRY_565_PHYSICAL_RECEIPT",
            "ROOT_IMMUTABLE_PACKET_REPLAY_RECEIPT",
        ],
        "nonClaims": [
            "global S2 derived from one local carrier",
            "physical geometry emergence",
            "continuum spacetime",
        ],
        "claimBoundary": ladder.get("claimBoundary")
        if isinstance(ladder, Mapping)
        else None,
    }


def a5_to_standard_model_view_contract(ladder: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical ``a5ToStandardModel`` visualization view."""

    controls = ladder.get("demoControls", {}) if isinstance(ladder, Mapping) else {}
    receipts = ladder.get("receipts", {}) if isinstance(ladder, Mapping) else {}
    h3_overlay = (
        ladder.get("physicalH3KmsDemoOverlay", {})
        if isinstance(ladder, Mapping)
        and isinstance(ladder.get("physicalH3KmsDemoOverlay"), Mapping)
        else {}
    )
    return {
        "viewId": "a5ToStandardModel",
        "sectionKind": "a5_to_standard_model_stage_ladder",
        "label": "A5-to-Standard-Model emergence ladder",
        "visualMetaphor": "receipt_gated_stage_dag",
        "description": (
            "Render the exact A5 action and 1+3+3-prime+5 sectors beside the canonical "
            "physical stage DAG. Demo-forced nodes use a separate watermarked status."
        ),
        "dataSources": [
            "screenA5Ladder.localCarrier.a5",
            "screenA5Ladder.localCarrier.a5.sectors",
            "screenA5Ladder.a5ToSm.stageNodes",
            "screenA5Ladder.a5ToSm.stageEdges",
            "screenA5Ladder.clockSeparation",
            "screenA5Ladder.demoControls",
            "screenA5Ladder.physicalH3KmsDemoOverlay",
        ],
        "primaryFields": [
            "screenA5Ladder.localCarrier.a5.actions",
            "screenA5Ladder.a5ToSm.stageNodes[*].displayStatus",
            "screenA5Ladder.a5ToSm.tierGroups",
            "screenA5Ladder.physicalH3KmsDemoOverlay.stageNodes",
        ],
        "renderLayers": [
            {
                "layer": "a5_port_action",
                "source": "screenA5Ladder.localCarrier.a5.actions",
            },
            {
                "layer": "a5_module_sectors",
                "source": "screenA5Ladder.localCarrier.a5.sectors",
            },
            {"layer": "physical_stage_dag", "source": "screenA5Ladder.a5ToSm"},
            {
                "layer": "clock_candidate_panel",
                "source": "screenA5Ladder.clockSeparation",
            },
            {
                "layer": "demo_watermark",
                "source": "screenA5Ladder.demoControls.watermark",
            },
            {
                "layer": "physical_h3_kms_demo_overlay",
                "source": "screenA5Ladder.physicalH3KmsDemoOverlay.stageNodes",
            },
        ],
        "visualEncodings": [
            {
                "field": "stage_status",
                "source": "screenA5Ladder.a5ToSm.stageNodes",
                "encoding": "physical status badge plus separate demo outline",
                "palette": "physical_vs_demo_status",
            },
            {
                "field": "claim_tier",
                "source": "screenA5Ladder.a5ToSm.tierGroups",
                "encoding": "nested structural, interacting, and continuum bands",
                "palette": "claim_tiers",
            },
        ],
        "animationChannels": [
            {
                "channel": "stage_ladder_walk",
                "source": "screenA5Ladder.a5ToSm.stageEdges",
                "timeSource": None,
                "encoding": "walk dependencies without changing physical receipt colors",
            }
        ],
        "receipts": {
            "a5_stage_dag_render": True,
            "demo_assumption_active": bool(controls.get("enabled", False)),
            "physical_a5_sm_global_pass": bool(
                receipts.get("PHYSICAL_A5_SM_GLOBAL_PASS") is True
            ),
            "physical_h3_kms_demo_overlay_complete": bool(
                h3_overlay.get("displayComplete") is True
            ),
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
        "exportSufficiency": "complete_stage_dag_visualization_not_emergence_evidence",
        "promotionReceiptsRequired": [
            "PHYSICAL_A5_SM_GLOBAL_PASS",
            "SCALE_CAMPAIGN_ALLOWED",
        ],
        "nonClaims": [
            "physical Standard Model emergence",
            "numerical Yukawa prediction",
            "continuum Wightman theory",
            "prospective prediction from frozen display targets",
        ],
        "claimBoundary": ladder.get("claimBoundary")
        if isinstance(ladder, Mapping)
        else None,
    }


def demo_universe_view_contract(ladder: Mapping[str, Any]) -> dict[str, Any]:
    """Return the isolated ``demoUniverse`` end-to-end view contract."""

    demo_universe = (
        ladder.get("demoUniverse", {}) if isinstance(ladder, Mapping) else {}
    )
    enabled = bool(demo_universe.get("enabled") is True)
    return {
        "viewId": "demoUniverse",
        "sectionKind": "forced_frozen_end_to_end_demo_universe",
        "label": "Forced/frozen demo universe",
        "visualMetaphor": "watermarked_observer_to_cosmology_storyboard",
        "description": (
            "Render the explicit visualization-only chain from carrier readback through "
            "repair and an observable normal form, exact A5 sectors, a conventional SM "
            "and composite-matter catalogue, H3/event and gravity display models, then "
            "finish inside a modular-clock observer camera."
        ),
        "dataSources": [
            "screenA5Ladder.demoUniverse",
            "screenA5Ladder.localCarrier",
            "screenA5Ladder.federation",
            "screenA5Ladder.a5ToSm",
            "screenA5Ladder.demoUniverse.publicCinematicSequence",
            "screenA5Ladder.demoUniverse.observerSpacetimeFinale",
        ],
        "primaryFields": [
            "screenA5Ladder.demoUniverse.segments",
            "screenA5Ladder.demoUniverse.animationTimeline",
            "screenA5Ladder.demoUniverse.addressSpaces",
            "screenA5Ladder.demoUniverse.publicCinematicSequence.scenes",
            "screenA5Ladder.demoUniverse.observerSpacetimeFinale",
        ],
        "renderLayers": [
            {
                "layer": segment.get("segmentId"),
                "source": (f"screenA5Ladder.demoUniverse.segments[{index}]"),
            }
            for index, segment in enumerate(demo_universe.get("segments", []))
            if isinstance(segment, Mapping)
        ],
        "visualEncodings": [
            {
                "field": "epistemic_status",
                "source": "screenA5Ladder.demoUniverse.segments[*].epistemicStatus",
                "encoding": "persistent DEMO_ASSUMPTION watermark and dashed display shell",
                "palette": "demo_assumption_only",
            },
            {
                "field": "finite_address",
                "source": "screenA5Ladder.demoUniverse.addressSpaces",
                "encoding": "deterministic chunk/index address, never expanded implicitly",
                "palette": "finite_census",
            },
        ],
        "animationChannels": [
            {
                "channel": "end_to_end_demo_ladder",
                "source": (
                    "screenA5Ladder.demoUniverse.publicCinematicSequence.scenes"
                ),
                "timeSource": (
                    "screenA5Ladder.demoUniverse.publicCinematicSequence.scenes"
                    "[*].durationFrames"
                ),
                "encoding": (
                    "continuous cinematic passage from the federation into one carrier, "
                    "normal form, matter, spacetime, and the observer finale"
                ),
            }
        ],
        "receipts": {
            "demo_assumption_active": enabled,
            "deterministic_finite_address_space": bool(
                demo_universe.get("addressSpaces")
            ),
            "physical_universe_emergence": False,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
        "exportSufficiency": (
            "finite_deterministic_demo_storyboard_not_physical_emergence_evidence"
        ),
        "promotionReceiptsRequired": [
            "PHYSICAL_A5_SM_GLOBAL_PASS",
            "INDEPENDENT_CLOCK_SELECTS_2PI_RECEIPT",
            "SCALE_CAMPAIGN_ALLOWED",
        ],
        "nonClaims": [
            "physical Standard Model particle emergence",
            "Einstein gravity derivation",
            "literal cosmological history",
            "every carrier or atom in the physical universe",
            "drawing every finite procedural carrier or atom simultaneously",
        ],
        "claimBoundary": demo_universe.get("claimBoundary"),
    }


def _demo_universe_payload(
    *,
    config: Mapping[str, Any],
    geometry: Mapping[str, Any],
    federation: Mapping[str, Any],
    stage_rows: Sequence[Mapping[str, Any]],
    physical_snapshot_digest: str,
) -> dict[str, Any]:
    """Build a bounded, deterministic, display-only universe storyboard."""

    requested = bool(config["valid"] and config["enabled"])
    enabled = bool(requested and config["forceAllStages"])
    carrier_count = int(federation.get("declaredCarrierCount", 1))
    atom_count = _demo_atom_count(config, carrier_count=carrier_count)
    address_spaces = {
        "carriers": _procedural_address_space(
            namespace_id="demo-carriers-v1",
            finite_count=carrier_count,
            record_id_template="carrier-{index:06d}",
            chunk_size=4096,
            generator_algorithm="fibonacci_s2_v1_then_local_a5_frame_index_mod_60",
        ),
        "carrierPulses": _procedural_address_space(
            namespace_id="demo-carrier-pulses-v1",
            finite_count=carrier_count * 12,
            record_id_template="carrier-pulse-{index:012d}",
            chunk_size=4096,
            generator_algorithm="carrier_major_then_port_major_single_tick_pulse_v1",
            index_mapping={
                "carrierIndex": "floor(index/12)",
                "portIndex": "index mod 12",
                "tick": "index",
                "path": (
                    "carrier-{carrierIndex:06d}/port-{portIndex:02d}/pulse-{tick:012d}"
                ),
            },
        ),
        "atoms": _procedural_address_space(
            namespace_id="demo-atoms-v1",
            finite_count=atom_count,
            record_id_template="atom-{index:08d}",
            chunk_size=256,
            generator_algorithm="splitmix64_indexed_atom_census_v1",
        ),
        "particleActors": _procedural_address_space(
            namespace_id="demo-particle-actors-v1",
            finite_count=DEMO_PARTICLE_ACTOR_COUNT,
            record_id_template="scene-actor-{index:03d}",
            chunk_size=48,
            generator_algorithm="fixed_post_exposure_species_family_v1",
        ),
        "particleWorldlineSamples": _procedural_address_space(
            namespace_id="demo-particle-worldlines-v1",
            finite_count=(DEMO_PARTICLE_ACTOR_COUNT * DEMO_WORLDLINE_FRAME_COUNT),
            record_id_template="scene-worldline-{index:05d}",
            chunk_size=DEMO_WORLDLINE_FRAME_COUNT,
            generator_algorithm="deterministic_softened_gravity_symplectic_euler_v1",
            index_mapping={
                "actorIndex": (f"floor(index/{DEMO_WORLDLINE_FRAME_COUNT})"),
                "frameIndex": (f"index mod {DEMO_WORLDLINE_FRAME_COUNT}"),
            },
        ),
        "observerFrames": _procedural_address_space(
            namespace_id="demo-observer-frames-v1",
            finite_count=DEMO_OBSERVER_FRAME_COUNT,
            record_id_template="camera-frame-{index:02d}",
            chunk_size=DEMO_OBSERVER_FRAME_COUNT,
            generator_algorithm="modular_clock_observer_orbit_v1",
        ),
        "modularClockSamples": _procedural_address_space(
            namespace_id="demo-modular-clock-samples-v1",
            finite_count=DEMO_OBSERVER_FRAME_COUNT,
            record_id_template="modular-clock-sample-{index:02d}",
            chunk_size=DEMO_OBSERVER_FRAME_COUNT,
            generator_algorithm="frozen_2pi_display_candidate_phase_v1",
        ),
    }
    records_by_segment = _demo_segment_records(
        geometry=geometry,
        federation=federation,
        atom_count=atom_count,
    )
    segment_specs = [
        (
            "carrier_light_readback_settling",
            "Carrier light/readback settling",
            "Illuminate each local 12-port carrier and display deterministic readback "
            "settling without treating the light field as a physical derivation.",
        ),
        (
            "repair_fixed_point",
            "Observer repair fixed point",
            "Animate seam comparison and a finite residual sequence ending at a "
            "displayed repair fixed point.",
        ),
        (
            "a5_sectors",
            "A5 representation sectors",
            "Expose the exact 12-dimensional permutation-sector decomposition "
            "1+3+3-prime+5 without assigning particles to sectors.",
        ),
        (
            "forced_sm_catalogue_and_interactions",
            "Forced SM catalogue and interactions",
            "Show a conventional finite Standard-Model species catalogue and interaction "
            "families only after every A5-to-SM display stage is explicitly forced.",
        ),
        (
            "gravity_response",
            "Assumed gravity response",
            "Display a deterministic curvature-response proxy sourced by the assumed "
            "matter catalogue; this is not an Einstein-equation receipt.",
        ),
        (
            "virtual_observer_camera",
            "Virtual observer camera",
            "Orbit a finite observer-local camera through the assumed scene with no "
            "hidden claim of a privileged physical observer.",
        ),
        (
            "finite_atom_census",
            "Finite atom census",
            "Render a bounded atom sample while retaining an exact index/chunk address "
            "for every atom in the finite procedural census.",
        ),
        (
            "cosmology",
            "Assumed cosmology",
            "Animate a short finite expansion storyboard whose scale-factor samples are "
            "frozen display assumptions, not inferred cosmological measurements.",
        ),
    ]
    segments: list[dict[str, Any]] = []
    for index, (segment_id, label, summary) in enumerate(segment_specs):
        records = records_by_segment[segment_id] if enabled else []
        segments.append(
            {
                "segmentId": segment_id,
                "index": index,
                "label": label,
                "summary": summary,
                "enabled": enabled,
                "forcedForDisplay": enabled,
                "displayStatus": DEMO_STATUS if enabled else "DISABLED",
                "epistemicStatus": DEMO_STATUS if enabled else "DISABLED",
                "visualizationOnly": True,
                "watermark": DEMO_WATERMARK if enabled else None,
                "records": records,
                "explicitRecordCount": len(records),
                "recordDigest": _canonical_digest(records),
                "promotion_allowed": False,
                "SCALE_CAMPAIGN_ALLOWED": False,
            }
        )
    blockers: list[str] = []
    if not requested:
        blockers.append("demo_universe_disabled_by_default")
    elif not config["forceAllStages"]:
        blockers.append("force_all_stages_required_for_end_to_end_demo")
    cinematic_sequence = _public_cinematic_sequence(enabled=enabled)
    seeded_disorder_opening = _seeded_disorder_opening(
        enabled=enabled,
        carrier_count=carrier_count,
        carrier_records=records_by_segment["carrier_light_readback_settling"],
    )
    observer_finale = _observer_spacetime_finale(
        enabled=enabled,
        camera_records=records_by_segment["virtual_observer_camera"],
    )
    return {
        "schema": DEMO_UNIVERSE_SCHEMA,
        "enabled": enabled,
        "requested": requested,
        "activationRule": (
            "valid demoControls.enabled=true and forceAllStages=true; per-stage toggles "
            "never silently activate the complete universe composition"
        ),
        "status": DEMO_STATUS if enabled else "DISABLED",
        "epistemicStatus": DEMO_STATUS if enabled else "DISABLED",
        "watermark": DEMO_WATERMARK if enabled else None,
        "segmentOrder": [item[0] for item in segment_specs],
        "segments": segments,
        "animationTimeline": [
            {
                "frameIndex": index,
                "segmentId": segment_id,
                "segmentRef": f"screenA5Ladder.demoUniverse.segments[{index}]",
                "durationFrames": 120,
                "enabled": enabled,
                "status": DEMO_STATUS if enabled else "DISABLED",
                "watermark": DEMO_WATERMARK if enabled else None,
            }
            for index, (segment_id, _, _) in enumerate(segment_specs)
        ],
        "addressSpaces": address_spaces,
        "finiteCensus": {
            "carrierCount": carrier_count,
            "carrierPulseCount": carrier_count * 12,
            "atomCount": atom_count,
            "particleActorCount": DEMO_PARTICLE_ACTOR_COUNT,
            "particleWorldlineSampleCount": (
                DEMO_PARTICLE_ACTOR_COUNT * DEMO_WORLDLINE_FRAME_COUNT
            ),
            "particleWorldlineFramesPerActor": DEMO_WORLDLINE_FRAME_COUNT,
            "observerFrameCount": DEMO_OBSERVER_FRAME_COUNT,
            "modularClockSampleCount": DEMO_OBSERVER_FRAME_COUNT,
            "carrierAddressSpaceRef": "screenA5Ladder.demoUniverse.addressSpaces.carriers",
            "carrierPulseAddressSpaceRef": (
                "screenA5Ladder.demoUniverse.addressSpaces.carrierPulses"
            ),
            "atomAddressSpaceRef": "screenA5Ladder.demoUniverse.addressSpaces.atoms",
            "particleActorAddressSpaceRef": (
                "screenA5Ladder.demoUniverse.addressSpaces.particleActors"
            ),
            "particleWorldlineAddressSpaceRef": (
                "screenA5Ladder.demoUniverse.addressSpaces.particleWorldlineSamples"
            ),
            "observerFrameAddressSpaceRef": (
                "screenA5Ladder.demoUniverse.addressSpaces.observerFrames"
            ),
            "modularClockSampleAddressSpaceRef": (
                "screenA5Ladder.demoUniverse.addressSpaces.modularClockSamples"
            ),
            "literalRowsEmitted": sum(len(rows) for rows in records_by_segment.values())
            if enabled
            else 0,
            "implicitBillionRowExpansionAllowed": False,
        },
        "publicCinematicSequence": cinematic_sequence,
        "seededDisorderOpening": seeded_disorder_opening,
        "observerSpacetimeFinale": observer_finale,
        "provenanceContract": {
            "allowedClasses": sorted(DEMO_PROVENANCE_CLASSES),
            "definitions": {
                "run_anchored": (
                    "copied from or aggregated over a named immutable run artifact"
                ),
                "computed_exact": (
                    "exact finite combinatorics or direct evaluation of an explicitly "
                    "declared exact formula; never a numerical physics inference"
                ),
                "interpolated": (
                    "display samples inserted between or across declared reference points"
                ),
                "synthetic": (
                    "seeded visualization content generated to complete the public story"
                ),
                "frozen_reference": (
                    "post-exposure conventional or target reference held fixed for display"
                ),
            },
            "everyExplicitRecordHasClassAndSourceRefs": enabled,
            "countsByClass": _provenance_counts(records_by_segment)
            if enabled
            else {name: 0 for name in sorted(DEMO_PROVENANCE_CLASSES)},
            "runAnchoredMeans": (
                "copied or summarized from a named run artifact; none is inferred merely "
                "because a demo record looks physical"
            ),
        },
        "publicDemoSynthesisPolicy": {
            "preferExistingRunData": True,
            "missingDisplayDataMayBeApproximated": True,
            "approximationOrder": [
                "interpolate_between_run_anchored_samples",
                "apply_declared_deterministic_finite_model",
                "use_post_exposure_conventional_reference",
                "generate_seeded_synthetic_display_content",
            ],
            "provenanceLabelRequired": True,
            "mayEnterPhysicalReceipt": False,
            "mayTunePhysicalProducer": False,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
        "forcedSmStageIds": [
            str(row.get("stageId"))
            for row in stage_rows
            if enabled and row.get("forcedForDisplay") is True
        ],
        "traceability": _demo_traceability(enabled=enabled),
        "displayComplete": bool(
            enabled
            and len(segments) == 8
            and all(row["displayStatus"] == DEMO_STATUS for row in segments)
            and all(row.get("forcedForDisplay") is True for row in stage_rows)
        ),
        "blockers": blockers,
        "physicalReceiptSnapshotDigestBefore": physical_snapshot_digest,
        "physicalReceiptSnapshotDigestAfter": physical_snapshot_digest,
        "receipts": {
            "DETERMINISTIC_FINITE_DEMO_RECORDS_RECEIPT": enabled,
            "EVERY_DEMO_CARRIER_ADDRESSABLE_RECEIPT": enabled,
            "EVERY_DEMO_CARRIER_PULSE_ADDRESSABLE_RECEIPT": enabled,
            "EVERY_DEMO_ATOM_ADDRESSABLE_RECEIPT": enabled,
            "FINITE_48_ACTOR_24_FRAME_SCENE_RENDER_RECEIPT": enabled,
            "OBSERVER_MODULAR_CLOCK_FINALE_RENDER_RECEIPT": enabled,
            "PUBLIC_CINEMATIC_SEQUENCE_RENDER_RECEIPT": enabled,
            "PHYSICAL_RECEIPT_SNAPSHOT_MUTATED": False,
            "PHYSICAL_UNIVERSE_EMERGENCE_RECEIPT": False,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "claimBoundary": (
            "This is a deterministic finite display composition. All generated rows are "
            "DEMO_ASSUMPTION, the physical snapshot is immutable, and no displayed SM, "
            "gravity, atom, camera, or cosmology content is production evidence."
        ),
    }


def _procedural_address_space(
    *,
    namespace_id: str,
    finite_count: int,
    record_id_template: str,
    chunk_size: int,
    generator_algorithm: str,
    index_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    chunk_count = max(1, math.ceil(finite_count / chunk_size))
    candidate_indices = sorted({0, chunk_count - 1})
    return {
        "namespaceId": namespace_id,
        "proceduralSeed": DEMO_UNIVERSE_SEED,
        "generatorAlgorithm": generator_algorithm,
        "exactFiniteCount": finite_count,
        "globalIndexRangeInclusive": [0, finite_count - 1],
        "recordIdTemplate": record_id_template,
        "recordAddress": f"{namespace_id}/record/{{index}}",
        "chunkSize": chunk_size,
        "chunkCount": chunk_count,
        "chunkIndexRangeInclusive": [0, chunk_count - 1],
        "chunkRefTemplate": f"{namespace_id}/chunk-{{chunkIndex:06d}}",
        "boundaryChunkRefs": [
            {
                "chunkIndex": index,
                "chunkRef": f"{namespace_id}/chunk-{index:06d}",
                "firstRecordIndex": index * chunk_size,
                "lastRecordIndex": min(finite_count, (index + 1) * chunk_size) - 1,
            }
            for index in candidate_indices
        ],
        "resolutionRule": (
            "chunkIndex=floor(index/chunkSize); offset=index mod chunkSize; reject "
            "indices outside globalIndexRangeInclusive"
        ),
        "indexMapping": dict(index_mapping or {"recordIndex": "index"}),
        "allRecordsAddressableWithoutMaterialization": True,
        "literalBulkExpansionAllowed": False,
    }


def _demo_atom_count(config: Mapping[str, Any], *, carrier_count: int) -> int:
    configured = config["frozenTargets"].get("demo_atom_census_count")
    if type(configured) is int and 1 <= configured <= DEMO_MAX_ATOM_CENSUS:
        return configured
    return min(4096, max(60, carrier_count * 5))


def _seeded_disorder_value(*, port_index: int, carrier_index: int = 0) -> float:
    seed_material = (
        f"{DEMO_UNIVERSE_SEED}:carrier:{carrier_index}:port:{port_index}"
    ).encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    unit_value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return _rounded(0.08 + 0.84 * unit_value)


def _demo_traceability(*, enabled: bool) -> dict[str, Any]:
    chain = {
        "recordId": "demo-observer-trace-000",
        "traceId": "demo-observer-trace-000",
        "observerFrameId": "camera-frame-00",
        "modularClockSampleId": "modular-clock-sample-00",
        "clockCandidateLabel": "2pi",
        "visibleParticleActorId": "particle-actor-up-000",
        "visibleAtomId": "atom-00000000",
        "gravityResponseId": "gravity-response-00",
        "interactionEventId": "sm-event-strong-00",
        "forcedSmStageId": "COMPLETE_COUPLED_DYNAMICS",
        "carrierPulseAddress": "demo-carrier-pulses-v1/record/0",
        "carrierId": "carrier-000000",
        "portId": "port-00",
        "pulseTick": 0,
        "orderedRefs": [
            "modular-clock-sample-00",
            "camera-frame-00",
            "gravity-response-00",
            "particle-actor-up-000",
            "sm-event-strong-00",
            "COMPLETE_COUPLED_DYNAMICS",
            "demo-carrier-pulses-v1/record/0",
        ],
        "provenanceClass": "synthetic",
        "sourceRefs": [
            "camera-frame-00",
            "modular-clock-sample-00",
            "gravity-response-00",
            "sm-event-strong-00",
            "demo-carrier-pulses-v1/record/0",
        ],
    }
    return {
        "enabled": enabled,
        "traceRows": [_demo_record(chain)] if enabled else [],
        "requiredLinkKinds": [
            "observer_frame_to_visible_actor_and_atom",
            "observer_frame_to_gravity_response",
            "gravity_response_to_particle_and_interaction",
            "interaction_to_forced_sm_stage",
            "particle_and_atom_to_carrier_pulse_address",
        ],
        "claimBoundary": (
            "Trace links make the demo provenance inspectable; they do not convert an "
            "assumed trajectory, coupling, gravity response, or atom into physical evidence."
        ),
    }


def _demo_segment_records(
    *,
    geometry: Mapping[str, Any],
    federation: Mapping[str, Any],
    atom_count: int,
) -> dict[str, list[dict[str, Any]]]:
    ports = [row for row in geometry.get("ports", []) if isinstance(row, Mapping)]
    sectors = [
        row
        for row in (geometry.get("a5", {}) or {}).get("sectors", [])
        if isinstance(row, Mapping)
    ]
    carrier_initial_records: list[dict[str, Any]] = []
    carrier_iteration_records: list[dict[str, Any]] = []
    carrier_settled_records: list[dict[str, Any]] = []
    for index, port in enumerate(ports):
        port_id = str(port.get("portId"))
        initial_state = _seeded_disorder_value(port_index=index)
        normal_form_target = _rounded(0.42 + ((index * 5) % 17) / 50.0)
        initial_id = f"carrier-initial-state-{index:02d}"
        initial_residual = abs(initial_state - normal_form_target)
        carrier_initial_records.append(
            _demo_record(
                {
                    "recordId": initial_id,
                    "recordKind": "seeded_disordered_carrier_port_state",
                    "portId": port_id,
                    "portIndex": index,
                    "seed": DEMO_UNIVERSE_SEED,
                    "seedAlgorithm": "sha256_seed_carrier_index_port_index_v1",
                    "initialReadbackState": initial_state,
                    "normalFormTarget": normal_form_target,
                    "initialResidualNorm": _rounded(initial_residual),
                    "initiallySettled": False,
                    "prepopulatedSolvedState": False,
                    "normalFormId": "demo-observable-normal-form-000",
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        f"screenA5Ladder.localCarrier.ports[{index}]",
                        f"seed:{DEMO_UNIVERSE_SEED}",
                    ],
                }
            )
        )
        iteration_ids: list[str] = []
        for iteration in range(8):
            iteration_id = f"readback-iteration-{index:02d}-{iteration:02d}"
            contraction = 0.45**iteration
            state = (
                normal_form_target + (initial_state - normal_form_target) * contraction
            )
            if iteration == 7:
                state = normal_form_target
            residual = abs(state - normal_form_target)
            iteration_ids.append(iteration_id)
            carrier_iteration_records.append(
                _demo_record(
                    {
                        "recordId": iteration_id,
                        "recordKind": "carrier_light_readback_iteration",
                        "portId": port_id,
                        "portIndex": index,
                        "iteration": iteration,
                        "readbackState": _rounded(state),
                        "normalFormTarget": normal_form_target,
                        "residualNorm": _rounded(residual),
                        "incidentLight": _rounded(0.25 + (index + 1) / 24.0),
                        "settled": iteration == 7,
                        "normalFormId": "demo-observable-normal-form-000",
                        "initialStateRef": initial_id,
                        "previousIterationRef": (
                            iteration_ids[-2] if iteration > 0 else None
                        ),
                        "carrierPulseAddress": (
                            f"demo-carrier-pulses-v1/record/{index}"
                        ),
                        "provenanceClass": "synthetic",
                        "sourceRefs": [
                            initial_id,
                            f"demo-carrier-pulses-v1/record/{index}",
                            *([iteration_ids[-2]] if iteration > 0 else []),
                        ],
                    }
                )
            )
        carrier_settled_records.append(
            _demo_record(
                {
                    "recordId": f"readback-{index:02d}",
                    "recordKind": "carrier_light_readback_settled_summary",
                    "portId": port_id,
                    "incidentLight": _rounded(0.25 + (index + 1) / 24.0),
                    "initialReadbackState": initial_state,
                    "settledReadback": normal_form_target,
                    "initiallySettled": False,
                    "settled": True,
                    "settlingIterationCount": 8,
                    "normalFormId": "demo-observable-normal-form-000",
                    "initialStateRef": initial_id,
                    "iterationRecordRefs": iteration_ids,
                    "settlingFrame": index,
                    "carrierIndex": 0,
                    "carrierPulseIndex": index,
                    "carrierPulseAddress": (f"demo-carrier-pulses-v1/record/{index}"),
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        initial_id,
                        *iteration_ids,
                        f"demo-carrier-pulses-v1/record/{index}",
                    ],
                }
            )
        )
    carrier_records = [
        *carrier_initial_records,
        *carrier_iteration_records,
        *carrier_settled_records,
    ]
    repair_records = [
        _demo_record(
            {
                "recordId": f"repair-step-{index:02d}",
                "recordKind": "observer_repair_normal_form_sample",
                "iteration": index,
                "residualNorm": _rounded(2.0 ** (-index)),
                "fixedPointDisplayed": index == 7,
                "normalFormId": "demo-observable-normal-form-000",
                "normalFormReached": index == 7,
                "sourceCarrierPulseAddress": (f"demo-carrier-pulses-v1/record/{index}"),
                "provenanceClass": "synthetic",
                "sourceRefs": [
                    f"readback-{index % 12:02d}",
                    f"demo-carrier-pulses-v1/record/{index}",
                    *([f"repair-step-{index - 1:02d}"] if index > 0 else []),
                ],
            }
        )
        for index in range(8)
    ]
    sector_records = [
        _demo_record(
            {
                "recordId": f"a5-sector-{row.get('sectorId')}",
                "recordKind": "exact_a5_representation_sector",
                "sectorId": row.get("sectorId"),
                "dimension": row.get("dimension"),
                "exactCharacters": row.get("exactCharacters"),
                "particleAssignment": None,
                "provenanceClass": "computed_exact",
                "sourceRefs": [
                    "screenA5Ladder.localCarrier.a5.actions",
                    f"screenA5Ladder.localCarrier.a5.sectors[{index}]",
                ],
            }
        )
        for index, row in enumerate(sectors)
    ]
    carrier_count = max(1, int(federation.get("declaredCarrierCount", 1)))
    carrier_pulse_count = carrier_count * 12
    raw_sm_records = _sm_catalogue_records(carrier_pulse_count=carrier_pulse_count)
    sm_records = [_demo_record(row) for row in raw_sm_records]
    actor_rows = [
        row for row in raw_sm_records if row.get("recordKind") == "particle_actor"
    ]
    actor_ids = [str(row["actorId"]) for row in actor_rows]
    worldline_ids_by_frame: dict[int, list[str]] = {
        frame_index: [] for frame_index in range(DEMO_WORLDLINE_FRAME_COUNT)
    }
    for row in raw_sm_records:
        if row.get("recordKind") == "particle_worldline_sample":
            worldline_ids_by_frame[int(row["frameIndex"])].append(str(row["recordId"]))
    gravity_records = [
        _demo_record(
            {
                "recordId": DEMO_SOFTENED_GRAVITY_MODEL_ID,
                "recordKind": "softened_gravity_display_model",
                "algorithm": "deterministic_symplectic_euler",
                "centralAttractorPosition": [0.0, 0.0, 0.0],
                "gravitationalParameterProxy": 0.55,
                "softeningLength": 0.35,
                "timeStep": 0.08,
                "einsteinEquationSolved": False,
                "physicalUnitsClaimed": False,
                "provenanceClass": "frozen_reference",
                "sourceRefs": [
                    "post-exposure:softened-central-gravity-display-convention"
                ],
            }
        )
    ]
    for index in range(DEMO_WORLDLINE_FRAME_COUNT):
        rotating = [actor_ids[(index + offset) % len(actor_ids)] for offset in range(8)]
        if index == 0:
            rotating = [
                "particle-actor-up-000",
                "particle-actor-electron-000",
                *rotating,
            ]
        source_actor_ids = list(dict.fromkeys(rotating))
        interaction_event_id = (
            "sm-event-strong-00" if index % 2 == 0 else "sm-event-electromagnetic-00"
        )
        pulse_index = index % carrier_pulse_count
        gravity_records.append(
            _demo_record(
                {
                    "recordId": f"gravity-response-{index:02d}",
                    "recordKind": "softened_gravity_response_frame",
                    "sampleIndex": index,
                    "sourceProxy": _rounded((index + 1) / DEMO_WORLDLINE_FRAME_COUNT),
                    "curvatureResponseProxy": _rounded(-0.00125 * (index + 1) ** 2),
                    "einsteinEquationSolved": False,
                    "gravityModelRef": DEMO_SOFTENED_GRAVITY_MODEL_ID,
                    "sourceActorIds": source_actor_ids,
                    "sourceWorldlineSampleIds": worldline_ids_by_frame[index],
                    "sourceInteractionEventIds": [interaction_event_id],
                    "sourceCarrierPulseAddress": (
                        f"demo-carrier-pulses-v1/record/{pulse_index}"
                    ),
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        DEMO_SOFTENED_GRAVITY_MODEL_ID,
                        interaction_event_id,
                        f"demo-carrier-pulses-v1/record/{pulse_index}",
                        *worldline_ids_by_frame[index],
                    ],
                }
            )
        )
        gravity_records.append(
            _demo_record(
                {
                    "recordId": f"h3-event-frame-{index:02d}",
                    "recordKind": "h3_event_display_frame",
                    "frameIndex": index,
                    "poincareBallPosition": [
                        _rounded(0.72 * math.cos(2.0 * math.pi * index / 24.0)),
                        _rounded(0.18 * math.sin(4.0 * math.pi * index / 24.0)),
                        _rounded(0.72 * math.sin(2.0 * math.pi * index / 24.0)),
                    ],
                    "eventConeRadiusProxy": _rounded(0.08 + 0.3 * index / 23.0),
                    "h3PhysicsDerived": False,
                    "gravityResponseRef": f"gravity-response-{index:02d}",
                    "provenanceClass": "interpolated",
                    "sourceRefs": [
                        f"gravity-response-{index:02d}",
                        "post-exposure:h3-visual-coordinate-convention",
                    ],
                }
            )
        )
    atom_elements = [
        ("H", 1, "hydrogen-atom", "particle-actor-hydrogen-atom-000"),
        ("He", 2, "helium-atom", "particle-actor-helium-atom-000"),
        ("C", 6, "carbon-atom", "particle-actor-carbon-atom-000"),
        ("O", 8, "oxygen-atom", "particle-actor-oxygen-atom-000"),
    ]
    atom_records = []
    for index in range(min(atom_count, DEMO_RECORD_LIMIT)):
        element, atomic_number, species_id, scene_actor_ref = atom_elements[
            index % len(atom_elements)
        ]
        pulse_index = ((index % carrier_count) * 12) % carrier_pulse_count
        atom_records.append(
            _demo_record(
                {
                    "recordId": f"atom-{index:08d}",
                    "recordKind": "finite_atom_census_record",
                    "atomIndex": index,
                    "element": element,
                    "atomicNumber": atomic_number,
                    "carrierIndex": index % carrier_count,
                    "proceduralAddress": f"demo-atoms-v1/record/{index}",
                    "sourceCarrierPulseAddress": (
                        f"demo-carrier-pulses-v1/record/{pulse_index}"
                    ),
                    "sceneActorRef": scene_actor_ref,
                    "constituentActorRefs": [
                        "particle-actor-up-000",
                        "particle-actor-down-000",
                        "particle-actor-electron-000",
                    ],
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        f"matter-species-{species_id}",
                        scene_actor_ref,
                        f"demo-carrier-pulses-v1/record/{pulse_index}",
                    ],
                }
            )
        )
    composite_actor_ids = [
        str(row["actorId"])
        for row in actor_rows
        if row.get("actorClass") in {"composite_baryon", "composite_atom"}
    ]
    camera_records: list[dict[str, Any]] = []
    for index in range(DEMO_OBSERVER_FRAME_COUNT):
        phase = 2.0 * math.pi * index / DEMO_OBSERVER_FRAME_COUNT
        clock_id = f"modular-clock-sample-{index:02d}"
        camera_records.append(
            _demo_record(
                {
                    "recordId": clock_id,
                    "recordKind": "modular_clock_display_sample",
                    "sampleIndex": index,
                    "clockCandidateLabel": "2pi",
                    "clockCandidateExact": "2*pi",
                    "modularPhaseRadians": _rounded(phase),
                    "normalizedPhase": _rounded(index / DEMO_OBSERVER_FRAME_COUNT),
                    "physicalClockSelected": False,
                    "postExposureDisplayCandidate": True,
                    "provenanceClass": "interpolated",
                    "sourceRefs": [
                        "screenA5Ladder.clockSeparation.candidates[2]",
                        "post-exposure:frozen-clock-candidate-2pi",
                    ],
                }
            )
        )
        rotating_visible = [
            actor_ids[(index * 3 + offset) % len(actor_ids)] for offset in range(10)
        ]
        if index == 0:
            rotating_visible = [
                "particle-actor-electron-000",
                "particle-actor-up-000",
                *rotating_visible,
            ]
        visible_actor_ids = list(dict.fromkeys(rotating_visible))
        visible_worldlines = [
            f"{actor_id}-worldline-{index:02d}" for actor_id in visible_actor_ids
        ]
        visible_atom_id = f"atom-{index % min(atom_count, DEMO_RECORD_LIMIT):08d}"
        pulse_index = index % carrier_pulse_count
        camera_records.append(
            _demo_record(
                {
                    "recordId": f"camera-frame-{index:02d}",
                    "recordKind": "observer_spacetime_camera_frame",
                    "frameIndex": index,
                    "position": [
                        _rounded(3.4 * math.cos(phase)),
                        _rounded(0.35 * math.sin(2.0 * phase)),
                        _rounded(3.4 * math.sin(phase)),
                    ],
                    "lookAt": [0.0, 0.0, 0.0],
                    "fieldOfViewDegrees": 55.0,
                    "modularClockSampleId": clock_id,
                    "clockCandidateLabel": "2pi",
                    "visibleParticleActorIds": visible_actor_ids,
                    "visibleCompositeActorIds": [
                        actor_id
                        for actor_id in visible_actor_ids
                        if actor_id in composite_actor_ids
                    ],
                    "visibleWorldlineSampleIds": visible_worldlines,
                    "visibleAtomIds": [visible_atom_id],
                    "gravityResponseIds": [f"gravity-response-{index:02d}"],
                    "h3EventFrameId": f"h3-event-frame-{index:02d}",
                    "cosmologyFrameId": f"cosmology-frame-{index:02d}",
                    "sourceCarrierPulseAddress": (
                        f"demo-carrier-pulses-v1/record/{pulse_index}"
                    ),
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        clock_id,
                        f"gravity-response-{index:02d}",
                        f"h3-event-frame-{index:02d}",
                        visible_atom_id,
                        f"demo-carrier-pulses-v1/record/{pulse_index}",
                        *visible_worldlines,
                    ],
                }
            )
        )
    cosmology_records = [
        _demo_record(
            {
                "recordId": f"cosmology-frame-{index:02d}",
                "recordKind": "frozen_reference_cosmology_frame",
                "frameIndex": index,
                "displayTime": _rounded(index / (DEMO_OBSERVER_FRAME_COUNT - 1)),
                "assumedScaleFactor": _rounded(
                    0.08
                    + 0.92 * (index / (DEMO_OBSERVER_FRAME_COUNT - 1)) ** (2.0 / 3.0)
                ),
                "measurementDerived": False,
                "gravityResponseRef": f"gravity-response-{index:02d}",
                "h3EventFrameRef": f"h3-event-frame-{index:02d}",
                "provenanceClass": "frozen_reference",
                "sourceRefs": [
                    "post-exposure:matter-dominated-scale-factor-display-curve",
                    f"gravity-response-{index:02d}",
                    f"h3-event-frame-{index:02d}",
                ],
            }
        )
        for index in range(DEMO_OBSERVER_FRAME_COUNT)
    ]
    return {
        "carrier_light_readback_settling": carrier_records,
        "repair_fixed_point": repair_records,
        "a5_sectors": sector_records,
        "forced_sm_catalogue_and_interactions": sm_records,
        "gravity_response": gravity_records,
        "virtual_observer_camera": camera_records,
        "finite_atom_census": atom_records,
        "cosmology": cosmology_records,
    }


def _sm_catalogue_records(*, carrier_pulse_count: int) -> list[dict[str, Any]]:
    fermions = [
        ("up", "quark", 1, "+2/3"),
        ("down", "quark", 1, "-1/3"),
        ("charm", "quark", 2, "+2/3"),
        ("strange", "quark", 2, "-1/3"),
        ("top", "quark", 3, "+2/3"),
        ("bottom", "quark", 3, "-1/3"),
        ("electron", "lepton", 1, "-1"),
        ("electron-neutrino", "lepton", 1, "0"),
        ("muon", "lepton", 2, "-1"),
        ("muon-neutrino", "lepton", 2, "0"),
        ("tau", "lepton", 3, "-1"),
        ("tau-neutrino", "lepton", 3, "0"),
    ]
    catalogue = [
        {
            "recordId": f"sm-species-{species}",
            "recordKind": "particle_species",
            "speciesId": species,
            "family": family,
            "generation": generation,
            "canonicalElectricCharge": charge,
            "derivedByDemo": False,
            "postExposureReference": True,
            "provenanceClass": "frozen_reference",
            "sourceRefs": [
                "post-exposure:conventional-standard-model-species-catalogue"
            ],
        }
        for species, family, generation, charge in fermions
    ]
    catalogue.extend(
        {
            "recordId": f"sm-species-{species}",
            "recordKind": "particle_species",
            "speciesId": species,
            "family": family,
            "generation": None,
            "canonicalElectricCharge": charge,
            "derivedByDemo": False,
            "postExposureReference": True,
            "provenanceClass": "frozen_reference",
            "sourceRefs": [
                "post-exposure:conventional-standard-model-species-catalogue"
            ],
        }
        for species, family, charge in [
            ("gluon", "gauge_boson", "0"),
            ("photon", "gauge_boson", "0"),
            ("W-plus-minus", "gauge_boson", "+/-1"),
            ("Z-zero", "gauge_boson", "0"),
            ("higgs", "scalar", "0"),
        ]
    )
    catalogue.extend(
        {
            "recordId": f"matter-species-{species}",
            "recordKind": "composite_matter_species",
            "speciesId": species,
            "family": family,
            "canonicalElectricCharge": charge,
            "conventionalConstituentSpecies": constituents,
            "derivedByDemo": False,
            "postExposureReference": True,
            "provenanceClass": "frozen_reference",
            "sourceRefs": ["post-exposure:conventional-composite-matter-reference"],
        }
        for species, family, charge, constituents in [
            ("proton", "composite_baryon", "+1", ["up", "up", "down"]),
            ("neutron", "composite_baryon", "0", ["up", "down", "down"]),
            ("hydrogen-atom", "composite_atom", "0", ["proton", "electron"]),
            (
                "helium-atom",
                "composite_atom",
                "0",
                ["proton", "proton", "neutron", "neutron", "electron", "electron"],
            ),
            (
                "carbon-atom",
                "composite_atom",
                "0",
                ["carbon-nucleus", "electron-cloud"],
            ),
            (
                "oxygen-atom",
                "composite_atom",
                "0",
                ["oxygen-nucleus", "electron-cloud"],
            ),
        ]
    )
    catalogue.extend(
        {
            "recordId": f"sm-interaction-{interaction}",
            "recordKind": "interaction_family",
            "interactionId": interaction,
            "displayMediator": mediator,
            "couplingPredicted": False,
            "postExposureReference": True,
            "provenanceClass": "frozen_reference",
            "sourceRefs": [
                "post-exposure:conventional-standard-model-interaction-catalogue"
            ],
        }
        for interaction, mediator in [
            ("strong", "gluon"),
            ("electromagnetic", "photon"),
            ("charged-weak", "W-plus-minus"),
            ("neutral-weak", "Z-zero"),
            ("yukawa", "higgs"),
        ]
    )
    actor_rows = _scene_actor_specs(carrier_pulse_count=carrier_pulse_count)
    catalogue.extend(actor_rows)
    catalogue.extend(_softened_gravity_worldline_records(actor_rows))
    event_specs = [
        (
            "strong",
            ["particle-actor-up-000", "particle-actor-gluon-000"],
        ),
        (
            "charged-weak",
            [
                "particle-actor-electron-000",
                "particle-actor-electron-neutrino-000",
                "particle-actor-W-plus-minus-000",
            ],
        ),
        (
            "neutral-weak",
            [
                "particle-actor-electron-neutrino-000",
                "particle-actor-Z-zero-000",
            ],
        ),
        (
            "electromagnetic",
            ["particle-actor-electron-000", "particle-actor-photon-000"],
        ),
        (
            "yukawa",
            ["particle-actor-electron-000", "particle-actor-higgs-000"],
        ),
    ]
    catalogue.extend(
        {
            "recordId": f"sm-event-{interaction}-00",
            "recordKind": "interaction_event",
            "interactionId": interaction,
            "participantActorIds": actor_ids,
            "frameIndex": index + 1,
            "position": [_rounded(0.1 * index), 0.0, _rounded(-0.1 * index)],
            "forcedStageRef": "COMPLETE_COUPLED_DYNAMICS",
            "couplingPredicted": False,
            "trajectoryAssumed": True,
            "sourceCarrierPulseAddress": (
                f"demo-carrier-pulses-v1/record/{index % carrier_pulse_count}"
            ),
            "provenanceClass": "synthetic",
            "sourceRefs": [
                f"sm-interaction-{interaction}",
                "COMPLETE_COUPLED_DYNAMICS",
                f"demo-carrier-pulses-v1/record/{index % carrier_pulse_count}",
                *actor_ids,
            ],
        }
        for index, (interaction, actor_ids) in enumerate(event_specs)
    )
    return catalogue


def _scene_actor_specs(*, carrier_pulse_count: int) -> list[dict[str, Any]]:
    properties: dict[str, tuple[str, int | None, str, float]] = {
        "up": ("+2/3", 1, "elementary_reference", 0.45),
        "down": ("-1/3", 1, "elementary_reference", 0.48),
        "charm": ("+2/3", 2, "elementary_reference", 0.62),
        "strange": ("-1/3", 2, "elementary_reference", 0.58),
        "top": ("+2/3", 3, "elementary_reference", 0.88),
        "bottom": ("-1/3", 3, "elementary_reference", 0.74),
        "electron": ("-1", 1, "elementary_reference", 0.22),
        "electron-neutrino": ("0", 1, "elementary_reference", 0.12),
        "muon": ("-1", 2, "elementary_reference", 0.34),
        "muon-neutrino": ("0", 2, "elementary_reference", 0.12),
        "tau": ("-1", 3, "elementary_reference", 0.52),
        "tau-neutrino": ("0", 3, "elementary_reference", 0.12),
        "gluon": ("0", None, "elementary_reference", 0.18),
        "photon": ("0", None, "elementary_reference", 0.16),
        "W-plus-minus": ("+/-1", None, "elementary_reference", 0.66),
        "Z-zero": ("0", None, "elementary_reference", 0.68),
        "higgs": ("0", None, "elementary_reference", 0.72),
        "proton": ("+1", None, "composite_baryon", 2.2),
        "neutron": ("0", None, "composite_baryon", 2.25),
        "hydrogen-atom": ("0", None, "composite_atom", 3.1),
        "helium-atom": ("0", None, "composite_atom", 4.2),
        "carbon-atom": ("0", None, "composite_atom", 5.4),
        "oxygen-atom": ("0", None, "composite_atom", 6.0),
    }
    elementary_species = [
        "up",
        "down",
        "charm",
        "strange",
        "top",
        "bottom",
        "electron",
        "electron-neutrino",
        "muon",
        "muon-neutrino",
        "tau",
        "tau-neutrino",
        "gluon",
        "photon",
        "W-plus-minus",
        "Z-zero",
        "higgs",
        "up",
        "down",
        "electron",
        "photon",
        "gluon",
        "muon",
        "tau",
    ]
    actor_species = [
        *elementary_species,
        *(["proton"] * 8),
        *(["neutron"] * 8),
        "hydrogen-atom",
        "helium-atom",
        "carbon-atom",
        "oxygen-atom",
        "hydrogen-atom",
        "helium-atom",
        "carbon-atom",
        "oxygen-atom",
    ]
    if len(actor_species) != DEMO_PARTICLE_ACTOR_COUNT:
        raise RuntimeError("demo_particle_actor_count_contract_broken")
    counters: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for actor_index, species in enumerate(actor_species):
        instance_index = counters.get(species, 0)
        counters[species] = instance_index + 1
        charge, generation, actor_class, mass_proxy = properties[species]
        actor_id = f"particle-actor-{species}-{instance_index:03d}"
        catalogue_prefix = (
            "sm-species" if actor_class == "elementary_reference" else "matter-species"
        )
        event_id = _interaction_event_for_species(species, actor_class=actor_class)
        pulse_index = actor_index % carrier_pulse_count
        rows.append(
            {
                "recordId": actor_id,
                "recordKind": "particle_actor",
                "actorId": actor_id,
                "actorIndex": actor_index,
                "speciesId": species,
                "speciesCatalogRef": f"{catalogue_prefix}-{species}",
                "actorClass": actor_class,
                "generation": generation,
                "canonicalElectricCharge": charge,
                "displayMassProxy": mass_proxy,
                "interactionEventRefs": [event_id],
                "trajectoryAssumed": True,
                "conventionalPostExposureReference": True,
                "sourceCarrierPulseAddress": (
                    f"demo-carrier-pulses-v1/record/{pulse_index}"
                ),
                "provenanceClass": "synthetic",
                "sourceRefs": [
                    f"{catalogue_prefix}-{species}",
                    event_id,
                    f"demo-carrier-pulses-v1/record/{pulse_index}",
                ],
            }
        )
    return rows


def _interaction_event_for_species(species: str, *, actor_class: str) -> str:
    if species in {
        "up",
        "down",
        "charm",
        "strange",
        "top",
        "bottom",
        "gluon",
        "proton",
        "neutron",
    }:
        return "sm-event-strong-00"
    if "neutrino" in species or species == "Z-zero":
        return "sm-event-neutral-weak-00"
    if species == "W-plus-minus":
        return "sm-event-charged-weak-00"
    if species == "higgs":
        return "sm-event-yukawa-00"
    if actor_class == "composite_atom" or species in {
        "electron",
        "muon",
        "tau",
        "photon",
    }:
        return "sm-event-electromagnetic-00"
    return "sm-event-neutral-weak-00"


def _softened_gravity_worldline_records(
    actor_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mu = 0.55
    softening = 0.35
    time_step = 0.08
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    records: list[dict[str, Any]] = []
    for actor_index, actor in enumerate(actor_rows):
        actor_id = str(actor["actorId"])
        angle = actor_index * golden_angle
        radial = 1.25 + 0.07 * (actor_index % 8)
        position = np.array(
            [
                radial * math.cos(angle),
                -0.7 + 0.28 * (actor_index % 6),
                radial * math.sin(angle),
            ],
            dtype=float,
        )
        radius_squared = float(position @ position)
        orbital_speed = math.sqrt(
            mu * radial * radial / (radius_squared + softening * softening) ** 1.5
        )
        velocity = np.array(
            [
                -orbital_speed * math.sin(angle),
                0.015 * ((actor_index % 3) - 1),
                orbital_speed * math.cos(angle),
            ],
            dtype=float,
        )
        for frame_index in range(DEMO_WORLDLINE_FRAME_COUNT):
            softened_radius = math.sqrt(
                float(position @ position) + softening * softening
            )
            acceleration = -mu * position / (softened_radius**3)
            record_id = f"{actor_id}-worldline-{frame_index:02d}"
            records.append(
                {
                    "recordId": record_id,
                    "recordKind": "particle_worldline_sample",
                    "actorId": actor_id,
                    "actorIndex": actor_index,
                    "speciesCatalogRef": actor["speciesCatalogRef"],
                    "frameIndex": frame_index,
                    "displayTime": _rounded(frame_index * time_step),
                    "position": [_rounded(value) for value in position],
                    "velocity": [_rounded(value) for value in velocity],
                    "softenedGravityAcceleration": [
                        _rounded(value) for value in acceleration
                    ],
                    "gravityPotentialProxy": _rounded(-mu / softened_radius),
                    "gravityModelRef": DEMO_SOFTENED_GRAVITY_MODEL_ID,
                    "sourceCarrierPulseAddress": actor["sourceCarrierPulseAddress"],
                    "trajectoryAssumed": True,
                    "couplingPredicted": False,
                    "provenanceClass": "synthetic",
                    "sourceRefs": [
                        actor_id,
                        str(actor["speciesCatalogRef"]),
                        DEMO_SOFTENED_GRAVITY_MODEL_ID,
                        str(actor["sourceCarrierPulseAddress"]),
                    ],
                }
            )
            velocity = velocity + acceleration * time_step
            position = position + velocity * time_step
    return records


def _demo_record(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    provenance_class = copied.get("provenanceClass", "synthetic")
    if provenance_class not in DEMO_PROVENANCE_CLASSES:
        raise ValueError(f"invalid_demo_provenance_class:{provenance_class}")
    source_refs = copied.get("sourceRefs")
    if (
        not isinstance(source_refs, list)
        or not source_refs
        or not all(isinstance(item, str) and item for item in source_refs)
    ):
        source_refs = ["screenA5Ladder.demoUniverse.seededDisplayModel"]
    return {
        **copied,
        "provenanceClass": provenance_class,
        "sourceRefs": source_refs,
        "status": DEMO_STATUS,
        "epistemicStatus": DEMO_STATUS,
        "visualizationOnly": True,
        "physicalEvidence": False,
        "watermark": DEMO_WATERMARK,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
    }


def _provenance_counts(
    records_by_segment: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    counts = {name: 0 for name in sorted(DEMO_PROVENANCE_CLASSES)}
    for rows in records_by_segment.values():
        for row in rows:
            provenance_class = row.get("provenanceClass")
            if isinstance(provenance_class, str) and provenance_class in counts:
                counts[provenance_class] += 1
    return counts


def _seeded_disorder_opening(
    *,
    enabled: bool,
    carrier_count: int,
    carrier_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    initial_rows = [
        row
        for row in carrier_records
        if row.get("recordKind") == "seeded_disordered_carrier_port_state"
    ]
    iteration_rows = [
        row
        for row in carrier_records
        if row.get("recordKind") == "carrier_light_readback_iteration"
    ]
    settled_rows = [
        row
        for row in carrier_records
        if row.get("recordKind") == "carrier_light_readback_settled_summary"
    ]
    return {
        "openingId": "seeded-disorder-to-normal-form-v1",
        "enabled": enabled,
        "seed": DEMO_UNIVERSE_SEED,
        "seedAlgorithm": "sha256_seed_carrier_index_port_index_v1",
        "federationSeedAlgorithm": "sha256_seed_carrier_index_port_index_v1",
        "prepopulatedSolvedLattice": False,
        "exactFederatedInitialPortStateCount": carrier_count * 12 if enabled else 0,
        "federationInitialStateAddressRule": (
            "seeded-disorder-v1/carrier/{carrierIndex}/port/{portIndex}"
        ),
        "initialPortCount": len(initial_rows) if enabled else 0,
        "iterationsPerPort": 8 if enabled else 0,
        "iterationRecordCount": len(iteration_rows) if enabled else 0,
        "settledPortCount": len(settled_rows) if enabled else 0,
        "initialStateRefs": [str(row["recordId"]) for row in initial_rows]
        if enabled
        else [],
        "iterationRefs": [str(row["recordId"]) for row in iteration_rows]
        if enabled
        else [],
        "settledSummaryRefs": [str(row["recordId"]) for row in settled_rows]
        if enabled
        else [],
        "normalFormId": "demo-observable-normal-form-000",
        "provenanceClass": "synthetic",
        "sourceRefs": [
            f"seed:{DEMO_UNIVERSE_SEED}",
            "screenA5Ladder.localCarrier.ports",
            "screenA5Ladder.demoUniverse.segments[0].records",
        ],
        "physicalEvidence": False,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
    }


def _public_cinematic_sequence(*, enabled: bool) -> dict[str, Any]:
    scene_specs = [
        {
            "sceneId": "federated_screen_overview",
            "label": "The screen awakens in seeded disorder",
            "summary": (
                "Begin before any solution exists: reveal the finite federation with "
                "deterministic, nontrivial 12-port initial states across its luminous "
                "compute elements."
            ),
            "cameraMove": "slow_orbital_reveal_then_dolly_toward_selected_carrier",
            "visualDirection": (
                "deep indigo field, warm gold carrier edges, visibly irregular port "
                "amplitudes, sparse volumetric bloom"
            ),
            "dataRefs": [
                "screenA5Ladder.federation",
                "screenA5Ladder.demoUniverse.addressSpaces.carriers",
                "screenA5Ladder.demoUniverse.addressSpaces.carrierPulses",
                "screenA5Ladder.demoUniverse.seededDisorderOpening",
                "screenA5Ladder.demoUniverse.segments[0].records",
            ],
            "provenanceClass": "synthetic",
        },
        {
            "sceneId": "single_12_port_carrier_zoom",
            "label": "Inside one echosahedral compute element",
            "summary": (
                "Resolve one exact icosahedral carrier: twelve ports, thirty edges, "
                "twenty faces, six antipodal pairs, and twelve explicitly unsettled "
                "seeded port states before light processing begins."
            ),
            "cameraMove": "seamless_macro_zoom_through_one_face_then_centered_orbit",
            "visualDirection": (
                "crystal-like faces with precise wire geometry and softly breathing ports"
            ),
            "dataRefs": [
                "screenA5Ladder.localCarrier",
                "screenA5Ladder.federation.carrierInstances[0]",
                "screenA5Ladder.demoUniverse.seededDisorderOpening.initialStateRefs",
            ],
            "provenanceClass": "synthetic",
        },
        {
            "sceneId": "light_readback_settling",
            "label": "Light computes and reads itself",
            "summary": (
                "Follow every light/readback iteration through all twelve ports as the "
                "residual contracts from seeded disorder and reaches the displayed normal form."
            ),
            "cameraMove": "port_follow_cam_with_brief_internal_cutaways",
            "visualDirection": (
                "directional photon trails, interference filaments, and a calm settled glow"
            ),
            "dataRefs": [
                "screenA5Ladder.demoUniverse.segments[0].records",
                "screenA5Ladder.demoUniverse.addressSpaces.carrierPulses",
            ],
            "provenanceClass": "synthetic",
        },
        {
            "sceneId": "observer_repair_normal_form",
            "label": "Repair settles to an observable normal form",
            "summary": (
                "Make disagreement visible across seams, then let the repair residual "
                "contract into one displayed finite normal form."
            ),
            "cameraMove": "pull_back_to_neighboring_cells_then_lock_on_fixed_point",
            "visualDirection": (
                "contrasting seam colors merge into a single coherent spectral signature"
            ),
            "dataRefs": [
                "screenA5Ladder.observerRepairBridge",
                "screenA5Ladder.demoUniverse.segments[1].records",
            ],
            "provenanceClass": "synthetic",
        },
        {
            "sceneId": "a5_representation_sectors",
            "label": "Icosahedral symmetry resolves into A5 sectors",
            "summary": (
                "Transform the twelve-port action into the exact 1 + 3 + 3-prime + 5 "
                "representation-sector composition."
            ),
            "cameraMove": "radial_explode_into_four_harmonic_sector_orbits",
            "visualDirection": (
                "four distinct harmonic color families synchronized to exact character data"
            ),
            "dataRefs": [
                "screenA5Ladder.localCarrier.a5.actions",
                "screenA5Ladder.demoUniverse.segments[2].records",
            ],
            "provenanceClass": "computed_exact",
        },
        {
            "sceneId": "standard_model_and_composite_matter",
            "label": "From symmetry language to visible matter",
            "summary": (
                "Introduce the post-exposure Standard Model reference catalogue, all "
                "three fermion generations, then conventional protons, neutrons, and atoms."
            ),
            "cameraMove": "flow_through_species_constellation_into_composite_matter_clusters",
            "visualDirection": (
                "generation-coded particle families with restrained interaction ribbons"
            ),
            "dataRefs": [
                "screenA5Ladder.a5ToSm.stageNodes",
                "screenA5Ladder.demoUniverse.segments[3].records",
                "screenA5Ladder.demoUniverse.addressSpaces.particleActors",
            ],
            "provenanceClass": "frozen_reference",
        },
        {
            "sceneId": "events_h3_gravity_and_cosmology",
            "label": "Events curve into a finite cosmos",
            "summary": (
                "Animate interaction events, the H3 display chart, softened gravitational "
                "response, and the short finite cosmology as one continuous transition."
            ),
            "cameraMove": "event_closeups_expand_into_h3_ball_then_cosmic_wide_shot",
            "visualDirection": (
                "interaction sparks bend into luminous trajectories over a quiet curvature grid"
            ),
            "dataRefs": [
                "screenA5Ladder.demoUniverse.segments[3].records",
                "screenA5Ladder.demoUniverse.segments[4].records",
                "screenA5Ladder.demoUniverse.segments[7].records",
            ],
            "provenanceClass": "interpolated",
        },
        {
            "sceneId": "observer_modular_spacetime_finale",
            "label": "Spacetime as seen by one modular observer",
            "summary": (
                "Finish inside the observer camera: the 2pi modular clock turns while "
                "particles, protons, neutrons, and atoms move visibly under the gravity proxy."
            ),
            "cameraMove": "crossfade_into_first_person_observer_frame_then_slow_forward_drift",
            "visualDirection": (
                "cinematic star-dark spacetime, legible modular clock halo, tangible matter depth"
            ),
            "dataRefs": [
                "screenA5Ladder.demoUniverse.observerSpacetimeFinale",
                "screenA5Ladder.demoUniverse.segments[5].records",
                "screenA5Ladder.demoUniverse.addressSpaces.particleWorldlineSamples",
            ],
            "provenanceClass": "synthetic",
        },
    ]
    scenes = [
        {
            "recordId": f"cinematic-scene-{index:02d}",
            "sceneIndex": index,
            "durationFrames": 240 if index in {0, 7} else 180,
            "enabled": enabled,
            "sourceRefs": list(spec["dataRefs"]),
            "visualizationOnly": True,
            "physicalEvidence": False,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
            **spec,
        }
        for index, spec in enumerate(scene_specs)
    ]
    return {
        "sequenceId": "oph-public-cinematic-emergence-ladder-v1",
        "enabled": enabled,
        "publicFacingNarrativeOnly": True,
        "showTechnicalStatusPanels": False,
        "showPassFailControls": False,
        "sceneOrder": [row["sceneId"] for row in scenes],
        "scenes": scenes if enabled else [],
        "transitionRule": "continuous_camera_and_color_motif_across_every_scene",
        "renderIntent": (
            "beautiful cinematic explanation first; provenance remains available in data "
            "but technical receipt state is outside the public presentation"
        ),
        "physicalEvidence": False,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
    }


def _observer_spacetime_finale(
    *,
    enabled: bool,
    camera_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frames = [
        row
        for row in camera_records
        if row.get("recordKind") == "observer_spacetime_camera_frame"
    ]
    clock_samples = [
        row
        for row in camera_records
        if row.get("recordKind") == "modular_clock_display_sample"
    ]
    final_frame = frames[-1] if enabled and frames else {}
    final_clock = clock_samples[-1] if enabled and clock_samples else {}
    return {
        "finaleId": "observer-modular-spacetime-finale-v1",
        "enabled": enabled,
        "provenanceClass": "synthetic",
        "sourceRefs": [
            "screenA5Ladder.clockSeparation.candidates[2]",
            "screenA5Ladder.demoUniverse.segments[3].records",
            "screenA5Ladder.demoUniverse.segments[4].records",
            "screenA5Ladder.demoUniverse.segments[5].records",
            "screenA5Ladder.demoUniverse.segments[6].records",
        ],
        "modularClock": {
            "candidateLabel": "2pi",
            "candidateExact": "2*pi",
            "selectionKind": "frozen_post_exposure_display_reference",
            "physicalSelection": False,
            "sampleCount": len(clock_samples) if enabled else 0,
            "sampleRefs": [str(row["recordId"]) for row in clock_samples]
            if enabled
            else [],
        },
        "camera": {
            "model": "finite_observer_local_orbit_then_first_person_finale",
            "frameCount": len(frames) if enabled else 0,
            "frameRefs": [str(row["recordId"]) for row in frames] if enabled else [],
            "finalFrameRef": final_frame.get("recordId"),
            "finalClockSampleRef": final_clock.get("recordId"),
            "visibleActorIdsAtFinalFrame": final_frame.get(
                "visibleParticleActorIds", []
            ),
            "visibleCompositeActorIdsAtFinalFrame": final_frame.get(
                "visibleCompositeActorIds", []
            ),
            "visibleAtomIdsAtFinalFrame": final_frame.get("visibleAtomIds", []),
            "gravityResponseRefsAtFinalFrame": final_frame.get(
                "gravityResponseIds", []
            ),
        },
        "stableJoinContract": {
            "actorToSpeciesCatalogField": "speciesCatalogRef",
            "actorToInteractionEventField": "interactionEventRefs",
            "actorToCarrierPulseField": "sourceCarrierPulseAddress",
            "cameraToWorldlineSamplesField": "visibleWorldlineSampleIds",
            "cameraToAtomField": "visibleAtomIds",
            "cameraToGravityField": "gravityResponseIds",
            "cameraToModularClockField": "modularClockSampleId",
        },
        "physicalSpacetimeDerived": False,
        "physicalEvidence": False,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
    }


def _parse_demo_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    stage_toggles = {stage_id: False for stage_id in STAGE_IDS}
    if config is None:
        return {
            "status": TerminalStatus.PASS.value,
            "valid": True,
            "enabled": False,
            "forceAllStages": False,
            "stages": stage_toggles,
            "frozenTargets": {},
            "blockers": [],
        }
    blockers: list[str] = []
    if not isinstance(config, Mapping):
        blockers.append("demo_config_must_be_mapping")
        config = {}
    unexpected = sorted(set(config) - _ALLOWED_CONFIG_FIELDS)
    blockers.extend(f"unknown_demo_config_field:{key}" for key in unexpected)

    enabled_raw = config.get("enabled", False)
    force_raw = config.get("forceAllStages", False)
    if type(enabled_raw) is not bool:
        blockers.append("enabled_must_be_literal_boolean")
    if type(force_raw) is not bool:
        blockers.append("forceAllStages_must_be_literal_boolean")
    enabled = enabled_raw if type(enabled_raw) is bool else False
    force_all = force_raw if type(force_raw) is bool else False

    raw_stages = config.get("stages", {})
    if not isinstance(raw_stages, Mapping):
        blockers.append("stages_must_be_mapping")
        raw_stages = {}
    for stage_id, value in raw_stages.items():
        if stage_id not in stage_toggles:
            blockers.append(f"unknown_stage_toggle:{stage_id}")
            continue
        if type(value) is not bool:
            blockers.append(f"stage_toggle_must_be_literal_boolean:{stage_id}")
            continue
        stage_toggles[stage_id] = value

    raw_targets = config.get("frozenTargets", {})
    frozen_targets: dict[str, Any] = {}
    if not isinstance(raw_targets, Mapping):
        blockers.append("frozenTargets_must_be_mapping")
    else:
        for target_id, value in raw_targets.items():
            if (
                not isinstance(target_id, str)
                or _TARGET_ID_RE.fullmatch(target_id) is None
            ):
                blockers.append(f"invalid_frozen_target_id:{target_id}")
                continue
            if target_id == "demo_atom_census_count" and (
                type(value) is not int or not 1 <= value <= DEMO_MAX_ATOM_CENSUS
            ):
                blockers.append(
                    "demo_atom_census_count_must_be_integer_between_1_and_1000000"
                )
                continue
            try:
                frozen_targets[target_id] = _finite_json_copy(value)
            except (TypeError, ValueError) as exc:
                blockers.append(f"invalid_frozen_target_value:{target_id}:{exc}")

    if blockers:
        enabled = False
        force_all = False
        stage_toggles = {stage_id: False for stage_id in STAGE_IDS}
        frozen_targets = {}
    elif "enabled" not in config:
        enabled = bool(force_all or any(stage_toggles.values()) or frozen_targets)
    return {
        "status": TerminalStatus.FAIL.value if blockers else TerminalStatus.PASS.value,
        "valid": not blockers,
        "enabled": enabled,
        "forceAllStages": force_all,
        "stages": stage_toggles,
        "frozenTargets": frozen_targets,
        "blockers": blockers,
    }


def _stage_rows(
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    snapshot_stages = (
        snapshot.get("stages", {})
        if isinstance(snapshot.get("stages"), Mapping)
        else {}
    )
    spec_by_id = {spec.stage_id: spec for spec in STAGE_SPECS}
    forced_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for stage_id in STAGE_IDS:
        physical = (
            snapshot_stages.get(stage_id, {})
            if isinstance(snapshot_stages.get(stage_id), Mapping)
            else {}
        )
        physical_status = physical.get("status")
        if physical_status not in {status.value for status in TerminalStatus}:
            physical_status = TerminalStatus.OPEN.value
        physical_passed = bool(
            physical_status == TerminalStatus.PASS.value
            and physical.get("passed") is True
        )
        forced = bool(
            config["valid"]
            and config["enabled"]
            and (config["forceAllStages"] or config["stages"].get(stage_id) is True)
        )
        if forced:
            forced_ids.append(stage_id)
        spec = spec_by_id[stage_id]
        rows.append(
            {
                "stageId": stage_id,
                "allDependencies": list(spec.all_dependencies),
                "anyDependencyGroups": [
                    list(group) for group in spec.any_dependency_groups
                ],
                "routeIds": [route.route_id for route in spec.routes],
                "physicalStatus": physical_status,
                "physicalPassed": physical_passed,
                "displayStatus": DEMO_STATUS if forced else physical_status,
                "displayComplete": bool(forced or physical_passed),
                "forcedForDisplay": forced,
                "epistemicStatus": DEMO_STATUS if forced else "PHYSICAL_SNAPSHOT",
                "visualizationOnly": forced,
                "watermark": DEMO_WATERMARK if forced else None,
                "promotion_allowed": False,
                "SCALE_CAMPAIGN_ALLOWED": False,
                "targetExposure": "post_exposure_display_only" if forced else None,
                "claimBoundary": spec.claim_boundary,
            }
        )
    return rows, forced_ids


def _stage_edges() -> list[dict[str, Any]]:
    alternative_lookup: dict[tuple[str, str], str] = {}
    for spec in STAGE_SPECS:
        for index, group in enumerate(spec.any_dependency_groups):
            for dependency in group:
                alternative_lookup[(dependency, spec.stage_id)] = (
                    f"{spec.stage_id}:alternative-{index}"
                )
    return [
        {
            "sourceStageId": source,
            "targetStageId": target,
            "dependencyKind": (
                "alternative_group"
                if (source, target) in alternative_lookup
                else "required"
            ),
            "alternativeGroupId": alternative_lookup.get((source, target)),
        }
        for source, target in STAGE_DAG_EDGES
    ]


def _tier_groups() -> list[dict[str, Any]]:
    return [
        {
            "tierId": "structural",
            "stageIds": list(BASE_GLOBAL_PASS_STAGES),
            "q2Alternatives": [
                ["Q2_H"],
                ["Q2_E", "POSITIVITY_OR_POSITIVE_TRANSFER"],
            ],
            "promotionReceipt": "PHYSICAL_A5_SM_STRUCTURAL_GLOBAL_PASS",
        },
        {
            "tierId": "full_interacting",
            "stageIds": list(FULL_INTERACTING_PASS_STAGES),
            "requiresTier": "structural",
            "promotionReceipt": "PHYSICAL_A5_SM_FULL_INTERACTING_PASS",
        },
        {
            "tierId": "continuum",
            "stageIds": ["Q4_OS"],
            "requiresTier": "full_interacting",
            "promotionReceipt": "NONPERTURBATIVE_CONTINUUM_WIGHTMAN_PHYSICAL_PASS",
        },
    ]


@lru_cache(maxsize=1)
def _cached_local_carrier() -> dict[str, Any]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw_rows: list[tuple[tuple[float, float, float], tuple[str, str, str]]] = []
    for first, second in itertools.product((-1, 1), repeat=2):
        raw_rows.append(
            (
                (0.0, float(first), float(second) * phi),
                ("0", _sign(first), _phi(second)),
            )
        )
    for first, second in itertools.product((-1, 1), repeat=2):
        raw_rows.append(
            (
                (float(first), float(second) * phi, 0.0),
                (_sign(first), _phi(second), "0"),
            )
        )
    for first, second in itertools.product((-1, 1), repeat=2):
        raw_rows.append(
            (
                (float(second) * phi, 0.0, float(first)),
                (_phi(second), "0", _sign(first)),
            )
        )
    raw = np.asarray([row[0] for row in raw_rows], dtype=np.float64)
    radius = math.sqrt(phi + 2.0)
    positions = raw / radius
    squared = np.sum((raw[:, None, :] - raw[None, :, :]) ** 2, axis=2)
    edge_distance_squared = min(
        value for value in squared.ravel().tolist() if value > 1e-12
    )
    edge_indices = [
        (left, right)
        for left in range(12)
        for right in range(left + 1, 12)
        if math.isclose(squared[left, right], edge_distance_squared, abs_tol=1e-10)
    ]
    edge_set = set(edge_indices)
    face_indices = [
        triple
        for triple in itertools.combinations(range(12), 3)
        if all(
            tuple(sorted(pair)) in edge_set
            for pair in itertools.combinations(triple, 2)
        )
    ]
    oriented_faces: list[tuple[int, int, int]] = []
    for face in face_indices:
        left, middle, right = face
        normal = np.cross(raw[middle] - raw[left], raw[right] - raw[left])
        if float(np.dot(normal, raw[list(face)].mean(axis=0))) < 0.0:
            middle, right = right, middle
        oriented_faces.append((left, middle, right))
    antipode_indices = [
        int(np.argmin(np.linalg.norm(raw + vertex, axis=1))) for vertex in raw
    ]
    actions, generator_ids = _rotation_actions(raw, edge_set, set(face_indices))
    if not (
        len(raw) == 12
        and len(edge_indices) == 30
        and len(oriented_faces) == 20
        and len(actions) == 60
        and all(
            antipode_indices[antipode_indices[index]] == index for index in range(12)
        )
    ):
        raise RuntimeError("internal_icosahedral_geometry_invariant_failed")
    return {
        "schema": "oph.local-icosahedral-carrier-render/1.0.0",
        "carrierId": "icosahedral-12-port-prototype",
        "carrierKind": "local_observer_cell_patch",
        "isLocalCarrier": True,
        "isGlobalS2Support": False,
        "geometryDistinction": (
            "One local icosahedral carrier has an S2-like closed boundary, but it is "
            "not the global screen S2 support. The global support is the federation."
        ),
        "exactCoordinateConvention": {
            "phi": "(1+sqrt(5))/2",
            "normalization": "1/sqrt(phi+2)",
            "vertexOrbit": "(0,±1,±phi) plus cyclic coordinate permutations",
        },
        "counts": {
            "portCount": 12,
            "edgeCount": 30,
            "faceCount": 20,
            "antipodalPairCount": 6,
            "a5ActionCount": 60,
        },
        "ports": [
            {
                "portId": f"port-{index:02d}",
                "index": index,
                "exactCoordinate": list(raw_rows[index][1]),
                "position": [_rounded(value) for value in positions[index]],
                "antipodePortId": f"port-{antipode_indices[index]:02d}",
                "edgeDegree": 5,
            }
            for index in range(12)
        ],
        "edges": [
            {
                "edgeId": f"edge-{index:02d}",
                "portIds": [f"port-{left:02d}", f"port-{right:02d}"],
            }
            for index, (left, right) in enumerate(edge_indices)
        ],
        "faces": [
            {
                "faceId": f"face-{index:02d}",
                "orientation": "outward_right_handed",
                "portIds": [f"port-{vertex:02d}" for vertex in face],
            }
            for index, face in enumerate(oriented_faces)
        ],
        "antipodes": [
            {
                "antipodePairId": f"antipode-{index:02d}",
                "portIds": [f"port-{left:02d}", f"port-{right:02d}"],
                "fixedPointFree": True,
            }
            for index, (left, right) in enumerate(
                (pair for pair in edge_free_antipodes(antipode_indices))
            )
        ],
        "a5": {
            "group": "A5",
            "interpretation": "orientation_preserving_icosahedral_rotation_group",
            "order": 60,
            "actions": actions,
            "generatorActionIds": generator_ids,
            "sectors": _a5_sector_rows(),
            "permutationRepresentationDecomposition": "12 = 1 + 3 + 3-prime + 5",
        },
        "receipts": {
            "exact_12_30_20_combinatorics": True,
            "fixed_point_free_antipode": True,
            "orientation_preserving_a5_order60": True,
            "a5_action_preserves_edges_and_faces": True,
            "physical_geometry_emergence": False,
        },
    }


def _exact_local_icosahedral_carrier() -> dict[str, Any]:
    return deepcopy(_cached_local_carrier())


def edge_free_antipodes(antipodes: Sequence[int]) -> list[tuple[int, int]]:
    return [(left, right) for left, right in enumerate(antipodes) if left < right]


def _rotation_actions(
    vertices: np.ndarray,
    edges: set[tuple[int, int]],
    faces: set[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], list[str]]:
    base_ids = (0, 1, 4)
    base = vertices[list(base_ids)].T
    base_gram = base.T @ base
    inverse = np.linalg.inv(base)
    rotations: dict[tuple[int, ...], np.ndarray] = {}
    for target_ids in itertools.permutations(range(12), 3):
        target = vertices[list(target_ids)].T
        if not np.allclose(target.T @ target, base_gram, atol=1e-10):
            continue
        rotation = target @ inverse
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            continue
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
            continue
        permutation: list[int] = []
        for vertex in vertices:
            distances = np.linalg.norm(vertices - rotation @ vertex, axis=1)
            mapped = int(np.argmin(distances))
            if distances[mapped] > 1e-8:
                permutation = []
                break
            permutation.append(mapped)
        if len(permutation) == 12 and len(set(permutation)) == 12:
            rotations[tuple(permutation)] = rotation
    class_order = {"1A": 0, "2A": 1, "3A": 2, "5A": 3, "5B": 4}
    ordered = sorted(
        rotations.items(),
        key=lambda item: (
            class_order[_rotation_class(item[0], item[1])],
            item[0],
        ),
    )
    action_ids = {
        permutation: f"a5-action-{index:02d}"
        for index, (permutation, _) in enumerate(ordered)
    }
    actions: list[dict[str, Any]] = []
    for permutation, rotation in ordered:
        mapped_edges = {
            tuple(sorted((permutation[left], permutation[right])))
            for left, right in edges
        }
        mapped_faces = {
            tuple(sorted(permutation[index] for index in face)) for face in faces
        }
        if mapped_edges != edges or mapped_faces != faces:
            raise RuntimeError("a5_action_failed_geometry_preservation")
        actions.append(
            {
                "actionId": action_ids[permutation],
                "conjugacyClass": _rotation_class(permutation, rotation),
                "order": _permutation_order(permutation),
                "determinant": 1,
                "portPermutation": list(permutation),
                "portIdPermutation": [f"port-{index:02d}" for index in permutation],
                "rotationMatrix": [
                    [_rounded(value) for value in row] for row in rotation.tolist()
                ],
            }
        )
    generator_pair = _find_a5_generators([item[0] for item in ordered])
    return actions, [action_ids[item] for item in generator_pair]


def _rotation_class(permutation: tuple[int, ...], rotation: np.ndarray) -> str:
    order = _permutation_order(permutation)
    if order == 1:
        return "1A"
    if order == 2:
        return "2A"
    if order == 3:
        return "3A"
    if order == 5:
        return "5A" if float(np.trace(rotation)) > 0.0 else "5B"
    raise RuntimeError(f"unexpected_icosahedral_rotation_order:{order}")


def _permutation_order(permutation: Sequence[int]) -> int:
    visited: set[int] = set()
    order = 1
    for start in range(len(permutation)):
        if start in visited:
            continue
        cursor = start
        length = 0
        while cursor not in visited:
            visited.add(cursor)
            cursor = permutation[cursor]
            length += 1
        if length:
            order = math.lcm(order, length)
    return order


def _find_a5_generators(
    permutations: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order_five = [item for item in permutations if _permutation_order(item) == 5]
    order_three = [item for item in permutations if _permutation_order(item) == 3]
    for first in order_five:
        for second in order_three:
            if len(_permutation_closure((first, second))) == 60:
                return first, second
    raise RuntimeError("a5_generator_pair_not_found")


def _permutation_closure(generators: Sequence[tuple[int, ...]]) -> set[tuple[int, ...]]:
    identity = tuple(range(len(generators[0])))
    closure = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            composed = tuple(generator[current[index]] for index in range(len(current)))
            if composed not in closure:
                closure.add(composed)
                frontier.append(composed)
    return closure


def _a5_sector_rows() -> list[dict[str, Any]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_conjugate = (1.0 - math.sqrt(5.0)) / 2.0
    return [
        _sector("1", 1, [1, 1, 1, 1, 1], ["1", "1", "1", "1", "1"]),
        _sector(
            "3", 3, [3, -1, 0, phi, phi_conjugate], ["3", "-1", "0", "phi", "phi-prime"]
        ),
        _sector(
            "3-prime",
            3,
            [3, -1, 0, phi_conjugate, phi],
            ["3", "-1", "0", "phi-prime", "phi"],
        ),
        _sector("5", 5, [5, 1, -1, 0, 0], ["5", "1", "-1", "0", "0"]),
    ]


def _sector(
    sector_id: str,
    dimension: int,
    characters: Sequence[float],
    exact_characters: Sequence[str],
) -> dict[str, Any]:
    return {
        "sectorId": sector_id,
        "dimension": dimension,
        "conjugacyClassOrder": ["1A", "2A", "3A", "5A", "5B"],
        "characters": [_rounded(value) for value in characters],
        "exactCharacters": list(exact_characters),
        "physicalInterpretation": None,
        "interpretationBoundary": "representation sector, not a particle assignment",
    }


def _federation_payload(
    carrier_count: int,
    *,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(carrier_count) is not int
        or carrier_count < 1
        or carrier_count > 1_000_000_000
    ):
        carrier_count = 1
        count_blockers = ["federation_carrier_count_invalid_defaulted_to_one"]
    else:
        count_blockers = []
    sample_count = min(carrier_count, MAX_RENDERED_CARRIER_SAMPLE)
    instances: list[dict[str, Any]] = []
    for index in range(sample_count):
        y = 1.0 - (2.0 * (index + 0.5) / sample_count)
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * math.pi * (3.0 - math.sqrt(5.0))
        instances.append(
            {
                "carrierId": f"carrier-{index:06d}",
                "prototypeRef": "screenA5Ladder.localCarrier",
                "globalSupportPoint": [
                    _rounded(radial * math.cos(angle)),
                    _rounded(y),
                    _rounded(radial * math.sin(angle)),
                ],
                "a5FrameActionId": f"a5-action-{index % 60:02d}",
                "sampledInstance": True,
            }
        )
    seams = [
        {
            "seamId": f"seam-{index:06d}",
            "leftCarrierId": instances[index]["carrierId"],
            "rightCarrierId": instances[(index + 1) % sample_count]["carrierId"],
            "leftPortId": f"port-{index % 12:02d}",
            "rightPortId": f"port-{(index + 6) % 12:02d}",
            "channelKind": "bidirectional_overlap_readback_and_repair",
            "physicalReceipt": False,
        }
        for index in range(sample_count)
        if sample_count > 1
    ]
    return {
        "federationKind": "large_federation_of_local_icosahedral_carriers",
        "declaredCarrierCount": carrier_count,
        "renderedCarrierSampleCount": sample_count,
        "sampledOnly": sample_count < carrier_count,
        "localCarrierPrototypeId": geometry.get("carrierId"),
        "carrierInstances": instances,
        "seams": seams,
        "seamTemplate": {
            "connects": "one local port to a neighboring carrier port",
            "transports": ["readback", "records", "repair_proposals"],
            "doesNotIdentify": "local carrier with global S2 support",
        },
        "globalS2Support": {
            "kind": "federated_screen_support",
            "isLocalCarrier": False,
            "isGlobalS2Support": True,
            "renderAs": "separate translucent support carrying local cell instances",
        },
        "geometryDistinction": {
            "localCarrier": "one closed icosahedral cell boundary with twelve ports",
            "globalSupport": "the network-level S2 screen assembled/read through the federation",
            "mustRenderSeparately": True,
        },
        "blockers": count_blockers,
    }


def _observer_repair_bridge_payload() -> dict[str, Any]:
    step_ids = [
        "local_state",
        "twelve_port_readback",
        "seam_overlap_comparison",
        "repair_transaction",
        "immutable_record_commit",
        "source_current_response",
        "a5_module_decomposition",
        "physical_stage_dag",
    ]
    return {
        "bridgeId": "simulator_architecture_to_observer_repair_to_a5_sm",
        "steps": [
            {
                "stepId": step_id,
                "index": index,
                "nextStepId": step_ids[index + 1]
                if index + 1 < len(step_ids)
                else None,
                "physicalPromotion": False,
            }
            for index, step_id in enumerate(step_ids)
        ],
        "selfReadingObserverStructure": {
            "boundedLocalState": True,
            "portsOrBoundaries": True,
            "readback": True,
            "records": True,
            "feedbackOrRepairMoves": True,
            "publicEvidenceBoundary": True,
        },
        "claimBoundary": (
            "This is the architectural ladder to the scientific stage contract, not a "
            "receipt that the later physics has emerged."
        ),
    }


def _frozen_target_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not config["valid"] or not config["enabled"]:
        return []
    targets: dict[str, Any] = {}
    if config["forceAllStages"]:
        targets.update(deepcopy(dict(DEFAULT_FROZEN_TARGETS)))
    targets.update(config["frozenTargets"])
    return [
        {
            "targetId": target_id,
            "displayValue": _finite_json_copy(value),
            "status": DEMO_STATUS,
            "epistemicStatus": DEMO_STATUS,
            "exposureStatus": "post_exposure_display_only",
            "usedToSelectPhysicalCandidate": False,
            "usedToTunePhysicalProducer": False,
            "visualizationOnly": True,
            "watermark": DEMO_WATERMARK,
            "promotion_allowed": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        }
        for target_id, value in sorted(targets.items())
    ]


def _clock_separation_payload(
    *,
    snapshot: Mapping[str, Any],
    demo_clock_requested: bool,
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipts = (
        snapshot.get("receipts", {})
        if isinstance(snapshot.get("receipts"), Mapping)
        else {}
    )
    physical_clock = receipts.get("INDEPENDENT_CLOCK_SELECTS_2PI_RECEIPT") is True
    frozen_clock = next(
        (
            row.get("displayValue")
            for row in target_rows
            if row.get("targetId") == "clock_normalization"
        ),
        None,
    )
    return {
        "candidateOrder": ["1x", "pi", "2pi", "4pi"],
        "candidates": [
            {"candidateId": "1x", "exactValue": "1", "numericValue": 1.0},
            {"candidateId": "pi", "exactValue": "pi", "numericValue": math.pi},
            {"candidateId": "2pi", "exactValue": "2*pi", "numericValue": 2.0 * math.pi},
            {"candidateId": "4pi", "exactValue": "4*pi", "numericValue": 4.0 * math.pi},
        ],
        "physicalSelection": "2pi" if physical_clock else None,
        "physicalReceipt": physical_clock,
        "demoSelection": "2pi" if demo_clock_requested else None,
        "frozenDisplayTarget": frozen_clock,
        "demoStatus": DEMO_STATUS if demo_clock_requested else None,
        "watermark": DEMO_WATERMARK if demo_clock_requested else None,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "claimBoundary": (
            "A demo-selected 2pi clock is post-exposure display content. Only an "
            "independently replayed physical clock receipt may populate physicalSelection."
        ),
    }


def _toggle_catalog() -> dict[str, Any]:
    return {
        "forceAllStages": {
            "type": "boolean",
            "default": False,
            "label": "Force complete ladder display",
        },
        "stageToggles": [
            {
                "stageId": stage_id,
                "type": "boolean",
                "default": False,
                "label": f"Force {stage_id} display",
            }
            for stage_id in STAGE_IDS
        ],
        "frozenTargetFields": [
            {
                "targetId": target_id,
                "defaultDisplayValue": _finite_json_copy(value),
                "exposureStatus": "post_exposure_display_only",
            }
            for target_id, value in DEFAULT_FROZEN_TARGETS.items()
        ],
        "safety": {
            "watermarkRequired": True,
            "physicalSnapshotMutable": False,
            "promotionAllowed": False,
            "scaleCampaignAllowed": False,
        },
    }


def _copy_physical_snapshot(
    snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], bool]:
    if snapshot is None:
        return (
            {
                "schema": "oph.empty-physical-a5-sm-snapshot/1.0.0",
                "status": TerminalStatus.OPEN.value,
                "passed": False,
                "stages": {},
                "receipts": {
                    "PHYSICAL_A5_SM_GLOBAL_PASS": False,
                    "SCALE_CAMPAIGN_ALLOWED": False,
                },
            },
            [],
            False,
        )
    if not isinstance(snapshot, Mapping):
        return {}, ["physical_receipt_snapshot_must_be_mapping"], False
    try:
        copied = _finite_json_copy(dict(snapshot))
    except (TypeError, ValueError) as exc:
        return {}, [f"physical_receipt_snapshot_not_finite_json:{exc}"], False
    if not isinstance(copied, dict):
        return {}, ["physical_receipt_snapshot_copy_not_object"], False
    trusted = bool(
        copied.get("schema") == REQUIREMENTS_REPORT_SCHEMA
        and copied.get("artifact_type") == REQUIREMENTS_REPORT_ARTIFACT_TYPE
    )
    blockers = (
        [] if trusted else ["untrusted_physical_snapshot_schema_or_artifact_type"]
    )
    return copied, blockers, trusted


def _finite_json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sign(value: int) -> str:
    return "+1" if value > 0 else "-1"


def _phi(value: int) -> str:
    return "+phi" if value > 0 else "-phi"


def _rounded(value: float) -> float:
    return round(float(value), 15)


__all__ = [
    "DEFAULT_FROZEN_TARGETS",
    "DEMO_UNIVERSE_SCHEMA",
    "DEMO_STATUS",
    "DEMO_WATERMARK",
    "SCREEN_A5_LADDER_SCHEMA",
    "a5_to_standard_model_view_contract",
    "build_screen_a5_ladder_payload",
    "demo_universe_view_contract",
    "screen_geometry_view_contract",
]
