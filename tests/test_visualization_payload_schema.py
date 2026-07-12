import json
from pathlib import Path


SCHEMA_PATH = Path("docs/oph_universe_timeline_visualization_payload_v1.schema.json")
MANUAL_PATH = Path("docs/VISUALIZATION_APP_AGENT_MANUAL.md")


def test_universe_timeline_visualization_schema_is_standalone_json():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schemaVersion"]["const"] == "oph_universe_timeline_visualization_payload_v1"
    assert schema["properties"]["schema"]["const"] == "oph_universe_timeline_visualization_payload_v1"
    assert schema["properties"]["visualizationRenderData"]["$ref"] == "#/$defs/visualizationRenderData"
    assert schema["properties"]["simulationAssumptions"]["properties"]["schema"]["const"] == (
        "oph_visualization_assumption_payload_v1"
    )
    assert set(schema["required"]) == {
        "schemaVersion",
        "schema",
        "title",
        "claimBoundary",
        "ophDifferentiator",
        "sourcePaths",
        "coordinateSystems",
        "simulationAssumptions",
        "smallUniverse",
        "screen",
        "subjectiveObserverCameras",
        "observerModularTime",
        "consensusBulk",
        "pnSilenceToObservation",
        "cmbComparison",
        "comparableObservations",
        "geometriesAndSymmetries",
        "visualizationViews",
        "visualizationRenderData",
        "effectiveStringTheory",
        "emergentCurvedSpacetime",
        "assumedDs4Spacetime",
        "observerCinema",
        "hilbertSpaceObserverAlgebra",
        "observerAnatomy",
        "paperAccuracy",
    }


def test_visualization_agent_manual_specifies_cinematic_accessible_story() -> None:
    manual = MANUAL_PATH.read_text(encoding="utf-8")

    for marker in (
        "Canonical Pedagogical Storyboard",
        "Bounded self-reading patches",
        "Overlap repair",
        "Shared records and consensus",
        "Enter one observer's 3+1D view",
        "Populate the observer-facing H3 bulk",
        "Defect worldlines styled as matter",
        "CMB-shaped sky and comparison",
        "explanatory overview (not observer-visible)",
        "ASSUMED VISUAL LAYER",
        "prefers-reduced-motion",
        "Performance And Streaming Budget",
        "256,000,000-byte ceiling",
    ):
        assert marker in manual


def test_universe_timeline_visualization_schema_preserves_required_viewer_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    defs = schema["$defs"]

    observer_frame_required = set(defs["objectiveObserverFrame"]["required"])
    assert {
        "visibleObjectPackets",
        "visibleRecordPackets",
        "polarFieldReadout",
        "framePacketSource",
    } <= observer_frame_required

    worldline_required = set(defs["h3ProtoParticleWorldline"]["required"])
    assert {
        "h3PathLength",
        "meanH3Step",
        "events",
        "particleLike",
        "bulkLocalizationPass",
    } <= worldline_required

    overlap_link = defs["observerOverlapLink"]
    # repairTrajectory is optional: only the strongest links carry a measured
    # trajectory; every link must declare how to derive one via trajectorySource.
    assert "trajectorySource" in overlap_link["required"]
    assert "repairTrajectory" in overlap_link["properties"]
    trajectory_frame = defs["overlapTrajectoryFrame"]
    assert "overlapMismatchDensity" in trajectory_frame["properties"]

    screen_clusters_required = set(defs["screenClusters"]["required"])
    assert {"clusters", "snapshots", "rawSnapshotCount", "snapshotSource"} <= screen_clusters_required

    camera_required = set(defs["subjectiveObserverCamera"]["required"])
    assert {
        "cameraId",
        "eye",
        "lookAt",
        "up",
        "right",
        "forward",
        "h3TangentFrame",
        "coordinateContract",
        "timeFrames",
        "claimBoundary",
    } <= camera_required

    ds4_required = set(defs["assumedDs4Spacetime"]["required"])
    assert {"provenance", "geometry", "scaleFactorSamples", "observerReferenceFrames", "receipts"} <= ds4_required

    subjective_sighting_required = set(defs["observerProtoWorldlineSighting"]["required"])
    assert "observerLocalReadout" in subjective_sighting_required
    assert "h3SpatialPoint" not in subjective_sighting_required
    local_readout_required = set(defs["observerLocalProtoWorldlineReadout"]["required"])
    assert {"u", "v", "range", "coordinateSystem", "hiddenGlobalH3Suppressed"} <= local_readout_required

    comparable_required = set(defs["comparableObservations"]["required"])
    assert {"measurementLanes", "datasets", "receipts", "claimBoundary"} <= comparable_required

    lane_required = set(defs["measurementLane"]["required"])
    assert {"id", "lane", "runCount", "metrics"} <= lane_required

    dataset_required = set(defs["comparableDataset"]["required"])
    assert {"id", "kind", "rowCount"} <= dataset_required

    view_set_required = set(defs["visualizationViews"]["required"])
    assert {
        "fluctuatingQuantumVacuum",
        "observerCamera",
        "emergentCurvedSpacetime",
        "effectiveStringTheory",
    } <= view_set_required
    assert "fractionalQuotientSectors" in defs["visualizationViews"]["properties"]
    top_level_required = set(schema["required"])
    assert {
        "effectiveStringTheory",
        "emergentCurvedSpacetime",
        "observerCinema",
        "hilbertSpaceObserverAlgebra",
        "observerAnatomy",
        "paperAccuracy",
    } <= top_level_required

    small_universe_required = set(defs["smallUniverse"]["required"])
    assert {
        "contentAvailable",
        "dataMode",
        "receiptSource",
        "bundleReceiptKind",
        "renderableExactMiniUniverseReceipt",
        "contentBlockers",
    } <= small_universe_required
    small_universe_properties = defs["smallUniverse"]["properties"]
    assert set(small_universe_properties["dataMode"]["enum"]) == {
        "exact_mini_universe",
        "theorem_receipt_summary_only",
    }
    assert set(small_universe_properties["receiptSource"]["type"]) == {"string", "null"}

    effective_string_required = set(defs["effectiveStringTheory"]["required"])
    assert {
        "finiteEdgeStringVibrationReceipt",
        "finiteEdgeStringVibrationBlockers",
        "layerAvailability",
        "hiddenLayers",
        "emptyLayerPolicy",
    } <= effective_string_required
    assert "finite_edge_string_vibration_pulses" in defs["effectiveStringTheory"]["properties"][
        "layerAvailability"
    ]["required"]

    render_data_required = set(defs["visualizationRenderData"]["required"])
    assert {
        "schema",
        "availability",
        "cameraPresets",
        "sceneGraph",
        "animationTimeline",
        "plotSeries",
        "legend",
        "claimBadges",
        "viewContracts",
        "claimBoundary",
    } <= render_data_required

    scene_graph_required = set(defs["visualizationRenderData"]["properties"]["sceneGraph"]["required"])
    assert {"screen", "observerGraph", "bulk", "finiteRepairGraph"} <= scene_graph_required

    view_required = set(defs["visualizationView"]["required"])
    assert {
        "viewId",
        "sectionKind",
        "label",
        "visualMetaphor",
        "dataSources",
        "primaryFields",
        "renderLayers",
        "visualEncodings",
        "animationChannels",
        "receipts",
        "exportSufficiency",
        "promotionReceiptsRequired",
        "nonClaims",
        "claimBoundary",
    } <= view_required

    encoding_required = set(defs["visualEncoding"]["required"])
    assert {"field", "source", "encoding"} <= encoding_required

    channel_required = set(defs["animationChannel"]["required"])
    assert {"channel", "source", "encoding"} <= channel_required
