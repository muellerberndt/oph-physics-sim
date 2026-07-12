from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import yaml

from oph_fpe.simulation_assumptions import simulation_assumption_manifest
from oph_fpe.viz.universe_timeline_viewer import (
    _assumed_cmb_visualization_payload,
    _canonical_h3_spatial_components,
    _h3_distance,
    _h3_exp_map,
    _h3_geodesic_interpolate,
    _h3_log_map,
    _assumed_ds4_visualization_payload,
    _project_h3_point_to_observer_camera,
    _project_h3_point_to_observer_directional_readout,
    build_universe_timeline_payload,
)
from oph_fpe.viz.visualization_schema import validate_visualization_payload
from oph_fpe.viz.visualizer_pack import (
    _safe_pack_path,
    build_visualizer_pack,
    read_visualizer_pack_payload,
)


def test_h3_spatial_components_use_intrinsic_hyperboloid_geometry() -> None:
    distance = 1.7
    origin = [0.0, 0.0, 0.0]
    target = [math.sinh(distance), 0.0, 0.0]

    assert _h3_distance(origin, target) == pytest.approx(distance, abs=1.0e-12)
    midpoint = _h3_geodesic_interpolate(origin, target, 0.5)
    assert midpoint == pytest.approx([math.sinh(distance / 2.0), 0.0, 0.0], abs=1.0e-12)
    assert _h3_distance(origin, midpoint) == pytest.approx(distance / 2.0, abs=1.0e-12)
    tangent = _h3_log_map(origin, target)
    assert _h3_exp_map(origin, tangent) == pytest.approx(target, abs=1.0e-12)


def test_h3_vec4_ingress_accepts_only_future_unit_hyperboloid_ordering() -> None:
    radial = 1.3
    ambient = [math.cosh(radial), math.sinh(radial), 0.0, 0.0]

    assert _canonical_h3_spatial_components(ambient) == pytest.approx(
        [math.sinh(radial), 0.0, 0.0]
    )
    assert _canonical_h3_spatial_components([-ambient[0], *ambient[1:]]) is None
    assert _canonical_h3_spatial_components([ambient[1], 0.0, 0.0, ambient[0]]) is None


def test_configured_assumed_cmb_reference_is_byte_pinned_and_fails_closed_on_hash_drift() -> None:
    config_path = Path("configs/e4_shared_observer_bulk_64k_object_chart.yml")
    manifest = simulation_assumption_manifest(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    payload = _assumed_cmb_visualization_payload(
        manifest,
        manifest_source=str(config_path),
    )
    assert payload["dataAvailable"] is True
    assert payload["provenance"]["sha256Matches"] is True
    assert len(payload["referenceRows"]) == len(payload["assumedModelRows"]) > 0
    assert payload["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False

    manifest["cmb_visualization_parameters"]["reference_sha256"] = "sha256:" + "0" * 64
    drifted = _assumed_cmb_visualization_payload(manifest, manifest_source=str(config_path))
    assert drifted["dataAvailable"] is False
    assert drifted["provenance"]["sha256Matches"] is False
    assert "pinned_cmb_reference_sha256_mismatch" in drifted["blockers"]


def test_observer_projection_keeps_nominal_fov_and_peripheral_diagnostic_separate() -> None:
    eye = [math.sinh(1.0), 0.0, 0.0]
    forward = [-1.0, 0.0, 0.0]
    right = [0.0, 1.0, 0.0]
    up = [0.0, 0.0, 1.0]
    nominal = _project_h3_point_to_observer_camera(
        [0.0, 0.0, 0.0],
        eye=eye,
        forward=forward,
        right=right,
        up=up,
        fov_degrees=72.0,
    )
    behind = [math.sinh(2.0), 0.0, 0.0]
    assert nominal is not None
    assert nominal["distance"] == pytest.approx(1.0, abs=1.0e-12)
    assert _project_h3_point_to_observer_camera(
        behind,
        eye=eye,
        forward=forward,
        right=right,
        up=up,
        fov_degrees=72.0,
    ) is None
    peripheral = _project_h3_point_to_observer_directional_readout(
        behind,
        eye=eye,
        forward=forward,
        right=right,
        up=up,
        fov_degrees=72.0,
    )
    assert peripheral is not None
    assert peripheral["outsideNominalFov"] is True


def test_assumed_ds4_normalizes_hubble_to_inverse_curvature_radius(tmp_path: Path) -> None:
    (tmp_path / "simulation_assumption_manifest.json").write_text(
        json.dumps(
            {
                "schema": "oph_simulation_assumption_manifest_v1",
                "profile": "test",
                "policy_id": "test-policy",
                "SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT": True,
                "assumptions": {
                    "ds4_open_slicing_background": {"assumed": True},
                    "observer_tetrad_visualization": {"assumed": True},
                    "topological_defects_render_as_matter": {"assumed": True},
                },
                "ds4_visualization_parameters": {
                    "curvature_radius": 2.0,
                    "hubble_parameter": 9.0,
                    "time_sample_count": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _assumed_ds4_visualization_payload(
        observer_payload={"source": str(tmp_path)},
        subjective_cameras=[],
        bulk_payload={},
    )
    geometry = result["geometry"]
    assert geometry["curvatureRadius"] == 2.0
    assert geometry["hubbleParameter"] == 0.5
    assert geometry["curvatureRadius"] * geometry["hubbleParameter"] == 1.0
    assert geometry["parameterNormalization"]["declaredConsistent"] is False
    assert result["receipts"]["derived_physical_ds4_receipt"] is False


def test_visualizer_pack_is_deterministic_chunked_validated_and_hard_gated(tmp_path: Path) -> None:
    for name in ("small", "observer", "pack", "bundle"):
        (tmp_path / name).mkdir()
    payload = build_universe_timeline_payload(
        small_universe_dir=tmp_path / "small",
        observer_run_dir=tmp_path / "observer",
        consensus_pack_dir=tmp_path / "pack",
        consensus_readout_dir=None,
        max_screen_points=8,
        max_observers=2,
        max_objective_observer_views=2,
        max_h3_objects=2,
    )
    payload["screen"]["repairTrace"] = [
        {"cycle": index, "phi": 10_000 - index, "committed_fraction": index / 10_000.0}
        for index in range(10_000)
    ]
    assert validate_visualization_payload(payload)["valid"] is True

    first = build_visualizer_pack(
        bundle_dir=tmp_path / "bundle",
        out_path=tmp_path / "first.tar.zst",
        payload=payload,
        max_bytes=2_000_000,
        target_bytes=1_000_000,
        chunk_bytes=64_000,
    )
    second = build_visualizer_pack(
        bundle_dir=tmp_path / "bundle",
        out_path=tmp_path / "second.tar.zst",
        payload=payload,
        max_bytes=2_000_000,
        target_bytes=1_000_000,
        chunk_bytes=64_000,
    )
    assert first["sha256"] == second["sha256"]
    assert first["byte_count"] < 2_000_000
    assert first["chunk_count"] > 0
    assert first["payload_validation"]["valid"] is True
    assert "schemaPath" not in first["payload_validation"]
    reconstructed, reconstruction = read_visualizer_pack_payload(tmp_path / "first.tar.zst")
    assert reconstructed == json.loads(json.dumps(payload, default=str))
    assert reconstruction["allManifestedHashesVerified"] is True
    assert reconstruction["exactPayloadReconstructionReceipt"] is True

    rejected = tmp_path / "too_small.tar.zst"
    rejected.write_bytes(b"stale-pack")
    with pytest.raises(ValueError, match="hard gate requires"):
        build_visualizer_pack(
            bundle_dir=tmp_path / "bundle",
            out_path=rejected,
            payload=payload,
            max_bytes=100,
            target_bytes=50,
            chunk_bytes=64_000,
        )
    assert not rejected.exists()


def test_visualizer_pack_member_references_cannot_escape_unpack_root(tmp_path: Path) -> None:
    assert _safe_pack_path(tmp_path, "sections/payload.json") == (
        tmp_path / "sections" / "payload.json"
    ).resolve()
    with pytest.raises(ValueError, match="unsafe visualizer pack member reference"):
        _safe_pack_path(tmp_path, "../secret")
    with pytest.raises(ValueError, match="unsafe visualizer pack member reference"):
        _safe_pack_path(tmp_path, "/absolute/secret")
