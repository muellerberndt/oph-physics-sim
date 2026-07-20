from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from oph_fpe.evidence.production_envelope import verify_production_bundle_manifest
from oph_fpe.gauge.physical_a5_sm_requirements import (
    STAGE_DAG_EDGES,
    STAGE_IDS,
    STAGE_SPECS,
    verify_physical_a5_sm_requirements,
)
from oph_fpe.viz.screen_a5_ladder import (
    DEMO_STATUS,
    DEMO_WATERMARK,
    a5_to_standard_model_view_contract,
    build_screen_a5_ladder_payload,
)
from oph_fpe.viz.universe_timeline_viewer import build_universe_timeline_payload
from oph_fpe.viz.visualization_schema import validate_visualization_payload


def test_exact_local_icosahedral_geometry_and_a5_action_catalog() -> None:
    ladder = build_screen_a5_ladder_payload()
    carrier = ladder["localCarrier"]

    assert carrier["counts"] == {
        "portCount": 12,
        "edgeCount": 30,
        "faceCount": 20,
        "antipodalPairCount": 6,
        "a5ActionCount": 60,
    }
    assert len(carrier["ports"]) == 12
    assert len(carrier["edges"]) == 30
    assert len(carrier["faces"]) == 20
    assert len(carrier["antipodes"]) == 6
    assert all(row["edgeDegree"] == 5 for row in carrier["ports"])
    assert all(row["fixedPointFree"] is True for row in carrier["antipodes"])

    actions = carrier["a5"]["actions"]
    assert len(actions) == 60
    assert sorted(row["order"] for row in actions).count(1) == 1
    assert sorted(row["order"] for row in actions).count(2) == 15
    assert sorted(row["order"] for row in actions).count(3) == 20
    assert sorted(row["order"] for row in actions).count(5) == 24
    assert {row["conjugacyClass"] for row in actions} == {"1A", "2A", "3A", "5A", "5B"}
    assert all(sorted(row["portPermutation"]) == list(range(12)) for row in actions)
    sectors = carrier["a5"]["sectors"]
    assert [(row["sectorId"], row["dimension"]) for row in sectors] == [
        ("1", 1),
        ("3", 3),
        ("3-prime", 3),
        ("5", 5),
    ]
    assert sum(row["dimension"] for row in sectors) == 12
    assert carrier["isLocalCarrier"] is True
    assert carrier["isGlobalS2Support"] is False
    assert ladder["federation"]["globalS2Support"]["isGlobalS2Support"] is True
    assert ladder["federation"]["geometryDistinction"]["mustRenderSeparately"] is True


def test_a5_stage_catalog_and_edges_are_exactly_the_physical_contract() -> None:
    ladder = build_screen_a5_ladder_payload()
    a5_to_sm = ladder["a5ToSm"]

    assert a5_to_sm["stageOrder"] == list(STAGE_IDS)
    assert [row["stageId"] for row in a5_to_sm["stageNodes"]] == list(STAGE_IDS)
    spec_by_id = {spec.stage_id: spec for spec in STAGE_SPECS}
    for row in a5_to_sm["stageNodes"]:
        spec = spec_by_id[row["stageId"]]
        assert row["allDependencies"] == list(spec.all_dependencies)
        assert row["anyDependencyGroups"] == [
            list(group) for group in spec.any_dependency_groups
        ]
        assert row["routeIds"] == [route.route_id for route in spec.routes]
    assert {
        (row["sourceStageId"], row["targetStageId"]) for row in a5_to_sm["stageEdges"]
    } == set(STAGE_DAG_EDGES)
    assert [row["tierId"] for row in a5_to_sm["tierGroups"]] == [
        "structural",
        "full_interacting",
        "continuum",
    ]


def test_default_payload_is_disabled_physical_false_and_has_toggle_catalog() -> None:
    ladder = build_screen_a5_ladder_payload()

    assert ladder["demoControls"]["enabled"] is False
    assert ladder["demoControls"]["configurationValid"] is True
    assert ladder["demoControls"]["toggleCatalog"]["stageToggles"]
    assert len(ladder["demoControls"]["toggleCatalog"]["stageToggles"]) == len(
        STAGE_IDS
    )
    assert ladder["a5ToSm"]["forcedStageIds"] == []
    assert ladder["a5ToSm"]["displayComplete"] is False
    assert all(row["physicalPassed"] is False for row in ladder["a5ToSm"]["stageNodes"])
    assert all(
        row["forcedForDisplay"] is False for row in ladder["a5ToSm"]["stageNodes"]
    )
    assert ladder["receipts"]["PHYSICAL_A5_SM_GLOBAL_PASS"] is False
    assert ladder["receipts"]["promotion_allowed"] is False
    assert ladder["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False
    assert ladder["clockSeparation"]["demoSelection"] is None
    assert ladder["demoUniverse"]["enabled"] is False
    assert ladder["demoUniverse"]["displayComplete"] is False


def test_force_all_is_display_complete_watermarked_and_physically_false() -> None:
    snapshot = {
        "schema": "test-physical-snapshot",
        "status": "OPEN",
        "passed": False,
        "stages": {},
        "receipts": {
            "PHYSICAL_A5_SM_GLOBAL_PASS": False,
            "SCALE_CAMPAIGN_ALLOWED": False,
        },
    }
    before = deepcopy(snapshot)
    ladder = build_screen_a5_ladder_payload(
        physical_receipt_snapshot=snapshot,
        demo_config={"enabled": True, "forceAllStages": True},
    )

    assert snapshot == before
    assert ladder["a5ToSm"]["physicalReceiptSnapshot"] == before
    assert ladder["a5ToSm"]["forcedStageIds"] == list(STAGE_IDS)
    assert ladder["a5ToSm"]["displayComplete"] is True
    for row in ladder["a5ToSm"]["stageNodes"]:
        assert row["displayStatus"] == DEMO_STATUS
        assert row["displayComplete"] is True
        assert row["physicalPassed"] is False
        assert row["watermark"] == DEMO_WATERMARK
        assert row["promotion_allowed"] is False
        assert row["SCALE_CAMPAIGN_ALLOWED"] is False
        assert row["targetExposure"] == "post_exposure_display_only"
    assert ladder["clockSeparation"]["demoSelection"] == "2pi"
    assert all(
        row["exposureStatus"] == "post_exposure_display_only"
        for row in ladder["demoControls"]["frozenTargetRows"]
    )
    assert ladder["receipts"]["promotion_allowed"] is False
    assert ladder["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False
    demo_universe = ladder["demoUniverse"]
    assert demo_universe["enabled"] is True
    assert demo_universe["displayComplete"] is True
    assert len(demo_universe["segments"]) == 8
    assert all(row["displayStatus"] == DEMO_STATUS for row in demo_universe["segments"])
    assert all(row["watermark"] == DEMO_WATERMARK for row in demo_universe["segments"])
    assert (
        demo_universe["physicalReceiptSnapshotDigestBefore"]
        == (demo_universe["physicalReceiptSnapshotDigestAfter"])
    )
    assert demo_universe["receipts"]["promotion_allowed"] is False
    assert demo_universe["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_per_stage_a5_force_never_selects_the_independent_clock() -> None:
    geometry_only = build_screen_a5_ladder_payload(
        demo_config={"stages": {"GEOMETRY_565": True}}
    )
    current_only = build_screen_a5_ladder_payload(
        demo_config={"stages": {"CURRENT_566": True}}
    )

    geometry_rows = {
        row["stageId"]: row for row in geometry_only["a5ToSm"]["stageNodes"]
    }
    assert geometry_only["a5ToSm"]["forcedStageIds"] == ["GEOMETRY_565"]
    assert geometry_rows["GEOMETRY_565"]["displayStatus"] == DEMO_STATUS
    assert geometry_rows["CURRENT_566"]["displayStatus"] == "OPEN"
    assert geometry_only["a5ToSm"]["displayComplete"] is False
    assert geometry_only["clockSeparation"]["demoSelection"] is None
    assert geometry_only["demoControls"]["frozenTargetRows"] == []

    assert current_only["a5ToSm"]["forcedStageIds"] == ["CURRENT_566"]
    assert current_only["clockSeparation"]["demoSelection"] is None
    assert current_only["demoControls"]["frozenTargetRows"] == []
    assert current_only["demoUniverse"]["enabled"] is False


def test_clock_demo_requires_explicit_frozen_target_or_force_all() -> None:
    explicit = build_screen_a5_ladder_payload(
        demo_config={
            "frozenTargets": {
                "clock_normalization": {
                    "label": "2pi",
                    "numericValue": 6.283185307179586,
                }
            }
        }
    )

    assert explicit["clockSeparation"]["demoSelection"] == "2pi"
    assert explicit["demoUniverse"]["enabled"] is False


def test_demo_universe_has_exact_finite_procedural_addresses_and_atom_override() -> (
    None
):
    ladder = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={
            "forceAllStages": True,
            "frozenTargets": {"demo_atom_census_count": 8192},
        },
    )
    universe = ladder["demoUniverse"]
    carriers = universe["addressSpaces"]["carriers"]
    carrier_pulses = universe["addressSpaces"]["carrierPulses"]
    atoms = universe["addressSpaces"]["atoms"]

    assert universe["finiteCensus"]["carrierCount"] == 4096
    assert universe["finiteCensus"]["atomCount"] == 8192
    assert carriers["exactFiniteCount"] == 4096
    assert carriers["globalIndexRangeInclusive"] == [0, 4095]
    assert carriers["recordIdTemplate"] == "carrier-{index:06d}"
    assert universe["finiteCensus"]["carrierPulseCount"] == 4096 * 12
    assert carrier_pulses["exactFiniteCount"] == 4096 * 12
    assert carrier_pulses["globalIndexRangeInclusive"] == [0, 4096 * 12 - 1]
    assert carrier_pulses["indexMapping"] == {
        "carrierIndex": "floor(index/12)",
        "portIndex": "index mod 12",
        "tick": "index",
        "path": ("carrier-{carrierIndex:06d}/port-{portIndex:02d}/pulse-{tick:012d}"),
    }
    assert atoms["exactFiniteCount"] == 8192
    assert atoms["globalIndexRangeInclusive"] == [0, 8191]
    assert atoms["chunkCount"] == 32
    assert atoms["allRecordsAddressableWithoutMaterialization"] is True
    assert atoms["literalBulkExpansionAllowed"] is False
    atom_segment = next(
        row for row in universe["segments"] if row["segmentId"] == "finite_atom_census"
    )
    assert len(atom_segment["records"]) == 32
    assert all(row["status"] == DEMO_STATUS for row in atom_segment["records"])
    assert all(row["watermark"] == DEMO_WATERMARK for row in atom_segment["records"])
    assert universe["receipts"]["EVERY_DEMO_CARRIER_PULSE_ADDRESSABLE_RECEIPT"] is True


def test_demo_trace_links_observer_particle_atom_gravity_sm_and_carrier_pulse() -> None:
    ladder = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    universe = ladder["demoUniverse"]
    segments = {row["segmentId"]: row["records"] for row in universe["segments"]}
    sm_by_id = {
        row["recordId"]: row for row in segments["forced_sm_catalogue_and_interactions"]
    }
    gravity_by_id = {row["recordId"]: row for row in segments["gravity_response"]}
    camera_by_id = {row["recordId"]: row for row in segments["virtual_observer_camera"]}
    atom_by_id = {row["recordId"]: row for row in segments["finite_atom_census"]}
    trace = universe["traceability"]["traceRows"][0]

    camera = camera_by_id[trace["observerFrameId"]]
    gravity = gravity_by_id[trace["gravityResponseId"]]
    actor = sm_by_id[trace["visibleParticleActorId"]]
    event = sm_by_id[trace["interactionEventId"]]
    atom = atom_by_id[trace["visibleAtomId"]]

    assert trace["visibleParticleActorId"] in camera["visibleParticleActorIds"]
    assert trace["visibleAtomId"] in camera["visibleAtomIds"]
    assert trace["gravityResponseId"] in camera["gravityResponseIds"]
    assert trace["visibleParticleActorId"] in gravity["sourceActorIds"]
    assert trace["interactionEventId"] in gravity["sourceInteractionEventIds"]
    assert trace["visibleParticleActorId"] in event["participantActorIds"]
    assert event["forcedStageRef"] == trace["forcedSmStageId"]
    assert trace["forcedSmStageId"] in universe["forcedSmStageIds"]
    assert actor["sourceCarrierPulseAddress"] == trace["carrierPulseAddress"]
    assert atom["sourceCarrierPulseAddress"] == trace["carrierPulseAddress"]
    pulse_index = int(trace["carrierPulseAddress"].rsplit("/", 1)[1])
    assert pulse_index // 12 == 0
    assert pulse_index % 12 == 0
    assert trace["carrierId"] == "carrier-000000"
    assert trace["portId"] == "port-00"
    for row in [trace, camera, gravity, actor, event, atom]:
        assert row["status"] == DEMO_STATUS
        assert row["watermark"] == DEMO_WATERMARK
        assert row["promotion_allowed"] is False
        assert row["SCALE_CAMPAIGN_ALLOWED"] is False


def test_public_demo_has_48_actors_24_frames_composites_and_softened_gravity() -> None:
    ladder = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    universe = ladder["demoUniverse"]
    segments = {row["segmentId"]: row["records"] for row in universe["segments"]}
    matter_records = segments["forced_sm_catalogue_and_interactions"]
    actor_rows = [
        row for row in matter_records if row["recordKind"] == "particle_actor"
    ]
    worldline_rows = [
        row
        for row in matter_records
        if row["recordKind"] == "particle_worldline_sample"
    ]

    assert len(actor_rows) == 48
    assert len(worldline_rows) == 48 * 24
    assert universe["finiteCensus"]["particleActorCount"] == 48
    assert universe["finiteCensus"]["particleWorldlineSampleCount"] == 48 * 24
    assert universe["addressSpaces"]["particleActors"]["exactFiniteCount"] == 48
    assert (
        universe["addressSpaces"]["particleWorldlineSamples"]["exactFiniteCount"]
        == 48 * 24
    )
    assert {row["generation"] for row in actor_rows if row["generation"]} == {
        1,
        2,
        3,
    }
    assert {row["actorClass"] for row in actor_rows} == {
        "elementary_reference",
        "composite_baryon",
        "composite_atom",
    }
    matter_catalog_ids = {
        row["recordId"]
        for row in matter_records
        if row["recordKind"] == "composite_matter_species"
    }
    assert {
        "matter-species-proton",
        "matter-species-neutron",
        "matter-species-hydrogen-atom",
        "matter-species-helium-atom",
        "matter-species-carbon-atom",
        "matter-species-oxygen-atom",
    } <= matter_catalog_ids

    actor_ids = {row["actorId"] for row in actor_rows}
    samples_by_actor = {
        actor_id: [row for row in worldline_rows if row["actorId"] == actor_id]
        for actor_id in actor_ids
    }
    assert all(len(rows) == 24 for rows in samples_by_actor.values())
    up_samples = samples_by_actor["particle-actor-up-000"]
    assert up_samples[0]["position"] != up_samples[-1]["position"]
    assert up_samples[0]["softenedGravityAcceleration"] != [0.0, 0.0, 0.0]
    assert all(
        row["gravityModelRef"] == "demo-softened-gravity-v1" for row in worldline_rows
    )

    repeat = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    repeat_matter = next(
        row["records"]
        for row in repeat["demoUniverse"]["segments"]
        if row["segmentId"] == "forced_sm_catalogue_and_interactions"
    )
    assert matter_records == repeat_matter


def test_public_cinematic_sequence_is_complete_narrative_without_status_panels() -> (
    None
):
    ladder = build_screen_a5_ladder_payload(demo_config={"forceAllStages": True})
    universe = ladder["demoUniverse"]
    sequence = universe["publicCinematicSequence"]

    assert sequence["sceneOrder"] == [
        "federated_screen_overview",
        "single_12_port_carrier_zoom",
        "light_readback_settling",
        "observer_repair_normal_form",
        "a5_representation_sectors",
        "standard_model_and_composite_matter",
        "events_h3_gravity_and_cosmology",
        "observer_modular_spacetime_finale",
    ]
    assert sequence["publicFacingNarrativeOnly"] is True
    assert sequence["showTechnicalStatusPanels"] is False
    assert sequence["showPassFailControls"] is False
    assert sequence["promotion_allowed"] is False
    assert sequence["SCALE_CAMPAIGN_ALLOWED"] is False
    assert len(sequence["scenes"]) == 8
    assert sequence["scenes"][0]["dataRefs"][0] == "screenA5Ladder.federation"
    assert sequence["scenes"][1]["dataRefs"][0] == "screenA5Ladder.localCarrier"
    assert all(row["sourceRefs"] for row in sequence["scenes"])

    repair_records = universe["segments"][1]["records"]
    assert repair_records[-1]["normalFormReached"] is True
    assert repair_records[-1]["normalFormId"] == "demo-observable-normal-form-000"


def test_public_opening_starts_disordered_then_settles_deterministically() -> None:
    ladder = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    universe = ladder["demoUniverse"]
    carrier_records = universe["segments"][0]["records"]
    initial_rows = [
        row
        for row in carrier_records
        if row["recordKind"] == "seeded_disordered_carrier_port_state"
    ]
    iteration_rows = [
        row
        for row in carrier_records
        if row["recordKind"] == "carrier_light_readback_iteration"
    ]
    settled_rows = [
        row
        for row in carrier_records
        if row["recordKind"] == "carrier_light_readback_settled_summary"
    ]

    assert len(initial_rows) == 12
    assert len(iteration_rows) == 12 * 8
    assert len(settled_rows) == 12
    assert len({row["initialReadbackState"] for row in initial_rows}) > 1
    assert all(row["initiallySettled"] is False for row in initial_rows)
    assert all(row["prepopulatedSolvedState"] is False for row in initial_rows)
    assert all(row["initialResidualNorm"] > 0.0 for row in initial_rows)
    for port_index in range(12):
        port_rows = sorted(
            (row for row in iteration_rows if row["portIndex"] == port_index),
            key=lambda row: row["iteration"],
        )
        residuals = [row["residualNorm"] for row in port_rows]
        assert len(port_rows) == 8
        assert all(left > right for left, right in zip(residuals, residuals[1:]))
        assert residuals[-1] == 0.0
        assert all(row["settled"] is False for row in port_rows[:-1])
        assert port_rows[-1]["settled"] is True
        assert all(row["provenanceClass"] == "synthetic" for row in port_rows)

    opening = universe["seededDisorderOpening"]
    assert opening["prepopulatedSolvedLattice"] is False
    assert opening["exactFederatedInitialPortStateCount"] == 4096 * 12
    assert opening["initialPortCount"] == 12
    assert opening["iterationRecordCount"] == 96
    assert opening["settledPortCount"] == 12
    assert opening["provenanceClass"] == "synthetic"
    assert opening["physicalEvidence"] is False
    assert opening["promotion_allowed"] is False
    assert opening["SCALE_CAMPAIGN_ALLOWED"] is False
    assert (
        "seeded disorder"
        in universe["publicCinematicSequence"]["scenes"][0]["label"].lower()
    )
    assert universe["receipts"]["PHYSICAL_UNIVERSE_EMERGENCE_RECEIPT"] is False
    assert universe["promotion_allowed"] is False

    repeat = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    assert repeat["demoUniverse"]["segments"][0]["records"] == carrier_records


def test_observer_spacetime_finale_and_every_demo_record_have_safe_provenance() -> None:
    ladder = build_screen_a5_ladder_payload(
        federation_carrier_count=4096,
        demo_config={"forceAllStages": True},
    )
    universe = ladder["demoUniverse"]
    segments = {row["segmentId"]: row["records"] for row in universe["segments"]}
    finale = universe["observerSpacetimeFinale"]
    camera_records = segments["virtual_observer_camera"]
    frame_rows = [
        row
        for row in camera_records
        if row["recordKind"] == "observer_spacetime_camera_frame"
    ]
    clock_rows = [
        row
        for row in camera_records
        if row["recordKind"] == "modular_clock_display_sample"
    ]

    assert finale["modularClock"]["candidateLabel"] == "2pi"
    assert finale["modularClock"]["physicalSelection"] is False
    assert finale["modularClock"]["sampleCount"] == 24
    assert finale["camera"]["frameCount"] == 24
    assert finale["camera"]["finalFrameRef"] == "camera-frame-23"
    assert finale["camera"]["finalClockSampleRef"] == "modular-clock-sample-23"
    assert finale["camera"]["visibleCompositeActorIdsAtFinalFrame"]
    assert finale["camera"]["visibleAtomIdsAtFinalFrame"]
    assert finale["physicalSpacetimeDerived"] is False
    assert len(frame_rows) == 24
    assert len(clock_rows) == 24

    record_ids = {
        row["recordId"]
        for records in segments.values()
        for row in records
        if "recordId" in row
    }
    final_frame = frame_rows[-1]
    assert final_frame["modularClockSampleId"] in record_ids
    assert all(item in record_ids for item in final_frame["visibleWorldlineSampleIds"])
    assert all(item in record_ids for item in final_frame["visibleAtomIds"])
    assert all(item in record_ids for item in final_frame["gravityResponseIds"])

    allowed_provenance = {
        "run_anchored",
        "computed_exact",
        "interpolated",
        "synthetic",
        "frozen_reference",
    }
    explicit_records = [
        row for records in segments.values() for row in records
    ] + universe["traceability"]["traceRows"]
    assert explicit_records
    assert all(row["provenanceClass"] in allowed_provenance for row in explicit_records)
    assert all(row["sourceRefs"] for row in explicit_records)
    assert all(row["promotion_allowed"] is False for row in explicit_records)
    assert all(row["SCALE_CAMPAIGN_ALLOWED"] is False for row in explicit_records)
    assert set(universe["provenanceContract"]["definitions"]) == allowed_provenance
    assert (
        "exact finite combinatorics"
        in universe["provenanceContract"]["definitions"]["computed_exact"]
    )

    for row in segments["forced_sm_catalogue_and_interactions"]:
        if row["recordKind"] == "particle_actor":
            assert row["speciesCatalogRef"] in record_ids
            assert all(item in record_ids for item in row["interactionEventRefs"])
        elif row["recordKind"] == "particle_worldline_sample":
            assert row["actorId"] in record_ids
            assert row["speciesCatalogRef"] in record_ids
            assert row["gravityModelRef"] in record_ids
        elif row["recordKind"] == "interaction_event":
            assert all(item in record_ids for item in row["participantActorIds"])
    for row in segments["gravity_response"]:
        if row["recordKind"] == "softened_gravity_response_frame":
            assert row["gravityModelRef"] in record_ids
            assert all(item in record_ids for item in row["sourceActorIds"])
            assert all(item in record_ids for item in row["sourceWorldlineSampleIds"])
            assert all(item in record_ids for item in row["sourceInteractionEventIds"])
        elif row["recordKind"] == "h3_event_display_frame":
            assert row["gravityResponseRef"] in record_ids
    for row in segments["finite_atom_census"]:
        assert row["sceneActorRef"] in record_ids
        assert all(item in record_ids for item in row["constituentActorRefs"])
    for row in frame_rows:
        assert row["modularClockSampleId"] in record_ids
        assert row["h3EventFrameId"] in record_ids
        assert row["cosmologyFrameId"] in record_ids
        assert all(item in record_ids for item in row["visibleParticleActorIds"])
        assert all(item in record_ids for item in row["visibleCompositeActorIds"])
        assert all(item in record_ids for item in row["visibleWorldlineSampleIds"])
        assert all(item in record_ids for item in row["visibleAtomIds"])
        assert all(item in record_ids for item in row["gravityResponseIds"])
    for row in segments["cosmology"]:
        assert row["gravityResponseRef"] in record_ids
        assert row["h3EventFrameRef"] in record_ids

    assert universe["publicDemoSynthesisPolicy"]["mayEnterPhysicalReceipt"] is False
    assert universe["promotion_allowed"] is False
    assert universe["SCALE_CAMPAIGN_ALLOWED"] is False


def test_truthy_nonbooleans_and_unknown_stages_fail_safe() -> None:
    cases = [
        {"forceAllStages": 1},
        {"enabled": "true", "stages": {"ROOT": True}},
        {"stages": {"ROOT": 1}},
        {"stages": {"NOT_A_STAGE": True}},
        {"frozenTargets": {"demo_atom_census_count": True}},
        {"frozenTargets": {"demo_atom_census_count": 1_000_001}},
    ]
    for config in cases:
        ladder = build_screen_a5_ladder_payload(demo_config=config)
        assert ladder["demoControls"]["configurationStatus"] == "FAIL"
        assert ladder["demoControls"]["configurationValid"] is False
        assert ladder["demoControls"]["enabled"] is False
        assert ladder["a5ToSm"]["forcedStageIds"] == []
        assert all(
            row["physicalPassed"] is False for row in ladder["a5ToSm"]["stageNodes"]
        )
        assert ladder["receipts"]["promotion_allowed"] is False
        assert ladder["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_passed_looking_untyped_snapshot_is_preserved_but_never_trusted() -> None:
    snapshot = {
        "schema": "test-passed-looking-physical-snapshot",
        "status": "PASS",
        "passed": True,
        "stages": {
            stage_id: {"status": "PASS", "passed": True} for stage_id in STAGE_IDS
        },
        "receipts": {
            "PHYSICAL_A5_SM_GLOBAL_PASS": True,
            "SCALE_CAMPAIGN_ALLOWED": True,
            "INDEPENDENT_CLOCK_SELECTS_2PI_RECEIPT": True,
        },
    }
    ladder = build_screen_a5_ladder_payload(physical_receipt_snapshot=snapshot)
    view = a5_to_standard_model_view_contract(ladder)

    assert ladder["a5ToSm"]["physicalReceiptSnapshot"] == snapshot
    assert ladder["a5ToSm"]["physicalSnapshotTrusted"] is False
    assert ladder["a5ToSm"]["physicalSnapshotBlockers"] == [
        "untrusted_physical_snapshot_schema_or_artifact_type"
    ]
    assert ladder["a5ToSm"]["displayComplete"] is False
    assert all(row["physicalPassed"] is False for row in ladder["a5ToSm"]["stageNodes"])
    assert view["receipts"]["physical_a5_sm_global_pass"] is False
    assert ladder["clockSeparation"]["physicalSelection"] is None
    assert ladder["receipts"]["promotion_allowed"] is False
    assert ladder["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False
    assert view["receipts"]["promotion_allowed"] is False
    assert view["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_actual_requirements_verifier_report_is_accepted_as_typed_snapshot(
    tmp_path: Path,
) -> None:
    report = verify_physical_a5_sm_requirements(tmp_path / "missing-root.json")
    ladder = build_screen_a5_ladder_payload(physical_receipt_snapshot=report)

    assert ladder["a5ToSm"]["physicalReceiptSnapshot"] == json.loads(
        json.dumps(report, allow_nan=False)
    )
    assert ladder["a5ToSm"]["physicalSnapshotTrusted"] is True
    assert ladder["a5ToSm"]["physicalSnapshotBlockers"] == []
    assert ladder["receipts"]["PHYSICAL_A5_SM_SNAPSHOT_TRUSTED"] is True
    assert ladder["receipts"]["PHYSICAL_A5_SM_GLOBAL_PASS"] is False
    assert all(row["physicalPassed"] is False for row in ladder["a5ToSm"]["stageNodes"])


def test_demo_object_is_rejected_by_production_and_physical_a5_ingestion() -> None:
    demo = build_screen_a5_ladder_payload(demo_config={"forceAllStages": True})

    production = verify_production_bundle_manifest(demo)  # type: ignore[arg-type]
    physical = verify_physical_a5_sm_requirements(demo)  # type: ignore[arg-type]

    assert production["status"] == "FAIL"
    assert production["promotion_allowed"] is False
    assert physical["status"] == "FAIL"
    assert physical["passed"] is False
    assert physical["receipts"]["PHYSICAL_A5_SM_GLOBAL_PASS"] is False


def test_universe_payload_integration_and_schema_validation(tmp_path: Path) -> None:
    small = tmp_path / "small"
    observer = tmp_path / "observer"
    small.mkdir()
    observer.mkdir()
    payload = build_universe_timeline_payload(
        small_universe_dir=small,
        observer_run_dir=observer,
        consensus_pack_dir=None,
        consensus_readout_dir=None,
        max_screen_points=8,
        max_observers=2,
        max_objective_observer_views=None,
        max_h3_objects=2,
        screen_a5_demo_config={"stages": {"GEOMETRY_565": True}},
    )

    result = validate_visualization_payload(payload)

    assert result["valid"] is True
    assert payload["screenA5Ladder"]["a5ToSm"]["forcedStageIds"] == ["GEOMETRY_565"]
    assert payload["visualizationViews"]["screenGeometry"]["viewId"] == (
        "screenGeometry"
    )
    assert payload["visualizationViews"]["a5ToStandardModel"]["viewId"] == (
        "a5ToStandardModel"
    )
    assert payload["visualizationViews"]["demoUniverse"]["viewId"] == ("demoUniverse")
    assert "screenGeometry" in payload["visualizationRenderData"]["viewContracts"]
    assert "a5ToStandardModel" in payload["visualizationRenderData"]["viewContracts"]
    assert "demoUniverse" in payload["visualizationRenderData"]["viewContracts"]
    assert payload["screenA5Ladder"]["receipts"]["promotion_allowed"] is False
    assert payload["screenA5Ladder"]["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False
