import json
from pathlib import Path


SCHEMA_PATH = Path("docs/oph_universe_timeline_visualization_payload_v1.schema.json")


def test_universe_timeline_visualization_schema_is_standalone_json():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema"]["const"] == "oph_universe_timeline_visualization_payload_v1"
    assert set(schema["required"]) == {
        "schema",
        "title",
        "claimBoundary",
        "ophDifferentiator",
        "sourcePaths",
        "smallUniverse",
        "screen",
        "observerModularTime",
        "consensusBulk",
        "pnSilenceToObservation",
        "cmbComparison",
        "geometriesAndSymmetries",
    }


def test_universe_timeline_visualization_schema_preserves_required_viewer_fields():
    defs = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["$defs"]

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
