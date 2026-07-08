import json
from pathlib import Path


SCHEMA_PATH = Path("docs/oph_universe_timeline_visualization_payload_v1.schema.json")


def test_universe_timeline_visualization_schema_is_standalone_json():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schemaVersion"]["const"] == "oph_universe_timeline_visualization_payload_v1"
    assert schema["properties"]["schema"]["const"] == "oph_universe_timeline_visualization_payload_v1"
    assert schema["properties"]["visualizationRenderData"]["$ref"] == "#/$defs/visualizationRenderData"
    assert set(schema["required"]) == {
        "schemaVersion",
        "schema",
        "title",
        "claimBoundary",
        "ophDifferentiator",
        "sourcePaths",
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
        "effectiveStringTheory",
        "emergentCurvedSpacetime",
        "observerCinema",
        "hilbertSpaceObserverAlgebra",
        "observerAnatomy",
        "paperAccuracy",
    }


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
    assert "repairTrajectory" in overlap_link["required"]
    assert "trajectorySource" in overlap_link["properties"]

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
        "timeFrames",
        "claimBoundary",
    } <= camera_required

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
