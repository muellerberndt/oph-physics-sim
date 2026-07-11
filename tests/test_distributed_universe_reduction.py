from __future__ import annotations

import json
from pathlib import Path

import pytest

from oph_fpe.pipelines.distributed_universe import (
    prepare_distributed_oph_universe,
    reduce_distributed_oph_universe,
)
from oph_fpe.viz.visualization_schema import validate_visualization_payload


def test_distributed_reducer_writes_fail_closed_global_cmb_report(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, shard_count=2)
    shard_root = tmp_path / "shards"
    _write_shard(shard_root / "u_shard0000")
    _write_shard(shard_root / "u_shard0001")

    summary = reduce_distributed_oph_universe(
        manifest_path=manifest,
        shard_root=shard_root,
        out_dir=tmp_path / "reduced",
    )

    cmb = summary["physical_cmb_global_reduction"]
    assert cmb["finite_source_global_reduction_receipt"] is False
    assert cmb["physical_cmb_input_contract_receipt"] is False
    assert cmb["physical_cmb_prediction_receipt"] is False
    assert "A_zeta_not_finite_derived" in cmb["blockers"]
    assert (tmp_path / "reduced" / "physical_cmb_global" / "physical_cmb_input_report.json").exists()
    assert (tmp_path / "reduced" / "halo_exchange_global" / "global_halo_exchange_report.json").exists()
    assert (tmp_path / "reduced" / "observer_modular_time_global" / "observer_modular_time_global_payload.json").exists()
    assert (tmp_path / "reduced" / "proto_particles_global" / "global_proto_particle_worldlines_report.json").exists()
    assert (tmp_path / "reduced" / "pn_resonance_global" / "global_pn_resonance_report.json").exists()
    assert (tmp_path / "reduced" / "DISTRIBUTED_RUN_PACK_CONTRACT.json").exists()
    assert (tmp_path / "reduced" / "global_carrier_contract" / "GLOBAL_CARRIER_CONTRACT.json").exists()
    assert summary["global_carrier_contract_receipt"] is True
    assert summary["one_global_carrier_before_partition_receipt"] is True
    assert summary["stable_global_identity_initial_state_receipt"] is True
    assert summary["distributed_realization_event_certificate_receipt"] is False
    assert summary["reducer_halo_exchange_replay_receipt"] is True
    assert summary["seam_metadata_replay_receipt"] is True
    assert summary["cross_shard_overlap_repair_receipt"] is False
    assert summary["online_cross_shard_overlap_repair_receipt"] is False
    assert summary["per_cycle_cross_shard_halo_exchange_receipt"] is False
    assert summary["global_observer_modular_time_export_receipt"] is True
    observer_time = summary["global_observer_modular_time_export"]
    assert observer_time["execution_clock_fields_separated_receipt"] is True
    assert observer_time["observer_clock_naturality_receipt"] is False
    assert summary["global_proto_particle_worldline_export_receipt"] is True
    assert summary["all_shards_local_scale_compressed_pn_witness_receipt"] is True
    assert summary["global_pn_resonance_receipt"] is False
    assert summary["strict_single_global_neutral_bulk_receipt"] is False

    payload = json.loads((tmp_path / "reduced" / "distributed_visualization_payload.json").read_text())
    assert validate_visualization_payload(payload)["variant"] == "distributed"
    assert payload["schemaVersion"] == "oph_universe_timeline_visualization_payload_v1"
    assert payload["distributedSchema"] == "oph_distributed_universe_visualization_payload_v1"
    assert payload["coordinateSystems"]["h3_hyperboloid_spatial_components_v1"]["model"] == (
        "future_unit_hyperboloid_spatial_components"
    )
    assert payload["assumedDs4Spacetime"]["enabled"] is False
    assert payload["physicalCMB"]["globalReduction"]["physical_cmb_prediction_receipt"] is False
    assert payload["cmbComparison"]["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert payload["observerModularTime"]["globalExport"]["objectiveObserverViewCount"] == 2
    assert payload["observerModularTime"]["observers"]
    assert payload["observerModularTime"]["timeFrames"]
    assert payload["subjectiveObserverCameras"]
    assert payload["hilbertSpaceObserverAlgebra"]["finiteSupportAlgebraPopulated"] is True
    assert payload["consensusBulk"]["objects"]
    assert payload["visualizationViews"]["effectiveStringTheory"]["viewId"] == "effectiveStringTheory"
    assert payload["visualizationRenderData"]["schema"] == "oph_visualization_render_data_v1"
    assert payload["visualizationRenderData"]["availability"]["subjectiveCameraCount"] >= 1
    assert payload["visualizationRenderData"]["availability"]["h3ObjectCount"] >= 1
    assert payload["visualizationRenderData"]["availability"]["protoWorldlineCount"] >= 1
    assert payload["visualizationRenderData"]["cameraPresets"]
    assert payload["visualizationRenderData"]["sceneGraph"]["bulk"]["protoWorldlines"][0]["polyline"]
    assert payload["visualizationRenderData"]["sceneGraph"]["assumedDs4Spacetime"]["sourcePath"] == (
        "assumedDs4Spacetime"
    )
    assert summary["visualizer_pack"]["under_hard_limit"] is True
    assert summary["visualizer_pack"]["byte_count"] < 256_000_000
    assert Path(summary["visualizer_pack"]["path"]).exists()
    observer_sidecar = json.loads(
        (tmp_path / "reduced" / "observer_modular_time_global" / "observer_modular_time_global_payload.json").read_text()
    )
    exported_view = observer_sidecar["objectiveObserverViews"][0]
    assert exported_view["execution_clock_fields_separated_receipt"] is True
    exported_frame = exported_view["timeFrames"][0]
    assert "execution_epoch" in exported_frame
    assert "scheduler_event_index" in exported_frame
    assert "observer_record_order" in exported_frame
    assert "observer_modular_parameter" in exported_frame
    assert "observer_clock_uncertainty" in exported_frame
    assert payload["consensusBulk"]["protoParticleCandidates"]["globalStitchReport"]["movingWorldlineCount"] == 2
    contract = json.loads((tmp_path / "reduced" / "DISTRIBUTED_RUN_PACK_CONTRACT.json").read_text())
    assert contract["distributed_artifact_packaging_smoke_receipt"] is True
    assert contract["observer_visualization_payload_ready_receipt"] is False
    assert contract["distributed_kernel_scaling_readiness_receipt"] is False
    assert "distributed_realization_event_certificate_receipt" in contract["profile_blockers"]["distributed_kernel_scaling"]
    assert "online_cross_shard_overlap_repair_receipt" in contract["profile_blockers"]["distributed_kernel_scaling"]


def test_distributed_reducer_carries_explicit_visual_universe_assumptions(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, shard_count=2)
    manifest = json.loads(manifest_path.read_text())
    config_path = Path(manifest["config_path"])
    config_path.write_text(
        config_path.read_text()
        + """
simulation_assumptions:
  enabled: true
  scope: visualization_only
  profile: known_observer_universe_v1
  assumed:
    screen_s2: true
    bw_2pi_geometric_branch: true
    h3_observer_chart: true
    record_population_on_h3: true
    ds4_open_slicing_background: true
    positive_cosmological_constant: true
    observer_tetrad_visualization: true
    topological_defects_render_as_matter: true
  ds4:
    curvature_radius: 2.0
    hubble_parameter: 0.5
    time_sample_count: 8
""",
        encoding="utf-8",
    )
    shard_root = tmp_path / "shards"
    _write_shard(shard_root / "u_shard0000")
    _write_shard(shard_root / "u_shard0001")

    summary = reduce_distributed_oph_universe(
        manifest_path=manifest_path,
        shard_root=shard_root,
        out_dir=tmp_path / "reduced",
    )

    payload = json.loads((tmp_path / "reduced" / "distributed_visualization_payload.json").read_text())
    ds4 = payload["assumedDs4Spacetime"]
    assert summary["simulation_assumptions"]["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is True
    assert ds4["receipts"]["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is True
    assert ds4["receipts"]["assumed_topological_defects_render_as_matter_receipt"] is True
    assert ds4["receipts"]["derived_physical_ds4_receipt"] is False
    assert ds4["receipts"]["physical_particle_matter_receipt"] is False
    assert all(
        row["assumed"] is True
        for row in ds4["assumptionLedger"]["assumptions"].values()
    )
    assert ds4["assumptionLedger"]["computedTheoremReceiptsUnchanged"] is True
    assert ds4["geometry"]["curvatureRadius"] == 2.0
    assert ds4["geometry"]["hubbleParameter"] == 0.5
    assert ds4["observerReferenceFrames"]
    assert ds4["defectMatterRendering"]["worldlineRefs"]
    assert all(
        row["renderAs"] == "matter_worldline_visual"
        for row in ds4["defectMatterRendering"]["worldlineRefs"]
    )
    assert validate_visualization_payload(payload)["variant"] == "distributed"


def test_distributed_reducer_reduces_finite_cmb_inputs_across_shards(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, shard_count=2)
    shard_root = tmp_path / "shards"
    _write_shard(shard_root / "u_shard0000")
    _write_shard(shard_root / "u_shard0001")
    _write_finite_cmb_sources(shard_root / "u_shard0000", a_zeta=2.0e-9, eta=0.030, gamma=0.10)
    _write_finite_cmb_sources(shard_root / "u_shard0001", a_zeta=2.2e-9, eta=0.034, gamma=0.12)

    summary = reduce_distributed_oph_universe(
        manifest_path=manifest,
        shard_root=shard_root,
        out_dir=tmp_path / "reduced",
    )

    cmb = summary["physical_cmb_global_reduction"]
    assert cmb["finite_source_global_reduction_receipt"] is False
    assert cmb["physical_cmb_input_contract_receipt"] is False
    assert cmb["physical_cmb_promotion_ready"] is False
    assert cmb["physical_cmb_prediction_receipt"] is False
    assert "global_pooled_sufficient_statistics_reducer_missing" in cmb["blockers"]
    assert "A_zeta_not_finite_derived" in cmb["blockers"]

    reduced_cert = json.loads(
        (tmp_path / "reduced" / "physical_cmb_global" / "finite_certificate_report.json").read_text()
    )
    assert reduced_cert["theorem_grade_finite_inputs"] is False
    assert reduced_cert["derived_outputs"]["A_zeta"] is None
    assert reduced_cert["derived_outputs"]["diagnostic_A_zeta_shard_mean"] == pytest.approx(2.1e-9)
    assert len(reduced_cert["derived_outputs"]["diagnostic_rho_A_a_rows"]) == 2

    input_report = json.loads(
        (tmp_path / "reduced" / "physical_cmb_global" / "physical_cmb_input_report.json").read_text()
    )
    assert input_report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "A_zeta_not_finite_derived" in input_report["blockers"]

    capacity = json.loads(
        (tmp_path / "reduced" / "physical_cmb_global" / "screen_capacity_closure_report.json").read_text()
    )
    assert capacity["observed_branch_normalization"]["N_CRC"] == pytest.approx(1000.0)
    assert capacity["observed_branch_normalization"]["N_CRC_additive_sum_diagnostic"] == pytest.approx(2000.0)
    assert capacity["readiness_gates"]["additive_capacity_schema_declared"] is False


def test_distributed_reducer_rejects_conflicting_no_data_firewall_flags(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, shard_count=2)
    shard_root = tmp_path / "shards"
    _write_shard(shard_root / "u_shard0000")
    _write_shard(shard_root / "u_shard0001")
    _write_json(
        shard_root / "u_shard0000" / "finite_certificate_report.json",
        {
            "no_cmb_data_used": True,
            "fit_to_planck": True,
            "theorem_grade_finite_inputs": True,
            "derived_outputs": {"A_zeta": 2.0e-9, "screen_to_primordial_lift_receipt": True},
        },
    )

    summary = reduce_distributed_oph_universe(
        manifest_path=manifest,
        shard_root=shard_root,
        out_dir=tmp_path / "reduced",
    )

    cmb = summary["physical_cmb_global_reduction"]
    assert cmb["no_data_use_receipt"] is False
    no_data = json.loads((tmp_path / "reduced" / "physical_cmb_global" / "no_data_use_receipt.json").read_text())
    status = no_data["source_status"]["finite_certificate_report.json"]
    assert status["measurement_data_used"] is True


def test_prepare_distributed_universe_emits_global_carrier_artifacts(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, shard_count=4)
    manifest = json.loads(manifest_path.read_text())

    carrier = manifest["global_carrier"]
    artifact_paths = {
        name: manifest_path.parent / row["path"]
        for name, row in carrier["artifacts"].items()
    }
    assert set(artifact_paths) == {
        "global_graph",
        "global_initial_state",
        "partition_map",
        "cut_interfaces",
        "global_observer_registry",
    }
    assert all(path.exists() for path in artifact_paths.values())

    partition = json.loads(artifact_paths["partition_map"].read_text())
    cut = json.loads(artifact_paths["cut_interfaces"].read_text())
    registry = json.loads(artifact_paths["global_observer_registry"].read_text())
    assert partition["node_count"] == 64
    assert [row["shard_id"] for row in partition["shards"]] == [
        f"u_shard{index:04d}" for index in range(4)
    ]
    assert cut["cut_edge_count"] > 0
    assert registry["observer_count"] == 32
    assert registry["schema"] == "oph_global_observer_registry_v2"
    assert registry["observer_kinds"] == ["patch", "cap", "future"]
    assert registry["registered_identity_count"] == 96
    assert registry["global_observer_registry_namespace_receipt"] is True
    sample = registry["sample_observers"][0]
    assert sample["distributed_observer_uid"].startswith("u:patch:")
    assert sample["observer_kind"] == "patch"
    assert str(sample["local_anchor_patch_id"]).startswith("patch:")
    assert sample["local_anchor_patch_id"] != str(sample["local_observer_index"])


def test_distributed_reducer_fails_closed_without_global_carrier_contract(tmp_path: Path) -> None:
    manifest = _write_manifest_without_carrier(tmp_path, shard_count=2)
    shard_root = tmp_path / "shards"
    _write_shard(shard_root / "u_shard0000")
    _write_shard(shard_root / "u_shard0001")

    summary = reduce_distributed_oph_universe(
        manifest_path=manifest,
        shard_root=shard_root,
        out_dir=tmp_path / "reduced",
    )

    contract = summary["distributed_run_pack_contract"]
    carrier = summary["global_carrier_contract"]
    assert summary["global_carrier_contract_receipt"] is False
    assert contract["distributed_artifact_packaging_smoke_receipt"] is False
    assert "global_carrier_contract_receipt" in contract["profile_blockers"]["distributed_artifact_packaging_smoke"]
    assert "global_graph_manifest_artifact_entry_missing" in carrier["blockers"]


def _write_manifest(tmp_path: Path, *, shard_count: int) -> Path:
    config = tmp_path / "base.yml"
    config.write_text(
        "\n".join(
            [
                "name: test_distributed",
                "seed: 7",
                "graph:",
                "  patch_count: 16",
                "dynamics:",
                "  repairs_per_cycle: 4",
                "observers:",
                "  sample_count: 8",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "pack"
    prepare_distributed_oph_universe(
        config_path=config,
        out_dir=out_dir,
        run_id="u",
        shard_count=shard_count,
        patch_count_per_shard=16,
        observers_per_shard=8,
        worker_count=max(1, shard_count),
    )
    return out_dir / "distributed_universe_manifest.json"


def _write_manifest_without_carrier(tmp_path: Path, *, shard_count: int) -> Path:
    shards = [
        {
            "shard_index": index,
            "shard_id": f"u_shard{index:04d}",
            "run_id": f"u_shard{index:04d}",
            "worker_index": index,
            "patch_count": 16,
            "observer_count": 8,
        }
        for index in range(shard_count)
    ]
    manifest = {
        "kernel": "distributed_observer_patch_kernel_v1",
        "run_id": "u",
        "claim_boundary": "test distributed universe",
        "shard_count": shard_count,
        "total_observer_capacity": shard_count * 8,
        "unified_universe_atlas": {
            "shards": [],
            "seam_links": [
                {
                    "link_id": "seam_0000_0001",
                    "source_shard_index": 0,
                    "target_shard_index": 1,
                    "source_shard_id": "u_shard0000",
                    "target_shard_id": "u_shard0001",
                    "claim_boundary": "test seam",
                }
            ],
        },
        "shards": shards,
    }
    path = tmp_path / "manifest.json"
    _write_json(path, manifest)
    return path


def _write_shard(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    receipts = {
        "observer_like_self_reading_system_receipt": True,
        "observer_modular_time_receipt": True,
        "observer_facing_3p1d_h3_experience_receipt": True,
        "theorem_assisted_consensus_3d_bulk_readout_receipt": True,
        "scale_compressed_pn_silence_to_observation_receipt": True,
        "physical_cmb_prediction_receipt": False,
    }
    _write_json(run_dir / "AUTO_THEOREM_UNIVERSE_SUMMARY.json", {"final_receipts": receipts})
    _write_json(run_dir / "manifest.json", {"patch_count": 16, "edge_count": 32})
    _write_json(run_dir / "emergence_status_report.json", {"bulk_3d_established": True})
    _write_json(run_dir / "observer_modular_experience_report.json", {"observer_count": 8})
    _write_json(run_dir / "observer_chart_object_h3_report.json", {"object_count": 2, "localized_object_count": 1})
    _write_json(
        run_dir / "defect_h3_worldlines_report.json",
        {
            "worldline_count": 1,
            "persistent_h3_worldline_count": 1,
            "worldlines": [
                {
                    "worldline_id": "w0",
                    "observation_count": 2,
                    "class_mode": "s3_defect",
                    "bulk_localization_pass": True,
                    "events": [
                        {"cycle": 0, "h3_spatial_point": [0.0, 0.0, 0.0], "support_node_count": 3},
                        {"cycle": 1, "h3_spatial_point": [0.1, 0.0, 0.0], "support_node_count": 3},
                    ],
                }
            ],
        },
    )
    (run_dir / "mismatch_trace.csv").write_text(
        "cycle,phi,mismatch_edges,committed_records,committed_fraction,observer_readback_drive_edges\n"
        "0,1.0,4,0,0.25,2\n"
        "63,0.0,0,16,1.0,0\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "silence_to_observation_report.json",
        {"scale_compressed_pn_silence_to_observation_receipt": True},
    )
    _write_json(
        run_dir / "universe_timeline" / "visualization_payload.json",
        {
            "observerModularTime": {
                "objectiveObserverViews": [
                    {
                        "observerId": "o0",
                        "axis": [1.0, 0.0, 0.0],
                        "supportPatchCount": 4,
                        "timeFrames": [
                            {
                                "cycle": cycle,
                                "visibleObjectPackets": [{"objectId": "obj0", "salience": 1.0}],
                                "visibleRecordPackets": [{"recordId": "rec0", "weight": 1.0}],
                            }
                            for cycle in range(32)
                        ],
                    }
                ],
                "overlapLinks": [
                    {
                        "linkId": "l0",
                        "sourceObserverId": "o0",
                        "targetObserverId": "o0",
                        "jaccard": 1.0,
                    }
                ],
            },
            "screen": {"clusters": {"snapshots": []}},
            "consensusBulk": {
                "objects": [
                    {
                        "objectId": "obj0",
                        "x": 0.1,
                        "y": 0.2,
                        "z": 0.3,
                        "observerCount": 2,
                        "supportSize": 4,
                    }
                ],
                "receipts": {
                    "observer_modular_time_receipt": True,
                    "observer_h3_object_population_receipt": True,
                    "theorem_assisted_consensus_3d_bulk_readout_receipt": True,
                },
                "protoParticleCandidates": {"worldlines": []},
            },
        },
    )


def _write_finite_cmb_sources(run_dir: Path, *, a_zeta: float, eta: float, gamma: float) -> None:
    _write_json(
        run_dir / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_finite_lattice_derived": True,
            "primary": {"eta_R_estimate": eta, "gamma_continuous": gamma},
        },
    )
    _write_json(
        run_dir / "finite_certificate_report.json",
        {
            "theorem_grade_finite_inputs": True,
            "derived_outputs": {
                "A_zeta": a_zeta,
                "rho_A_a": [[0.5, a_zeta * 1.0e8]],
                "screen_to_primordial_lift_receipt": True,
            },
        },
    )
    _write_json(run_dir / "B_A_kernel_report.json", {"B_A_KERNEL_RECEIPT": True, "B_A_k_a": [[0.1, 0.5, 1.0]]})
    _write_json(
        run_dir / "b_a_parent_report.json",
        {
            "readiness": {"checks": {"no_cmb_data_used": True}},
            "rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "base_epsilon_cmi": 0.2}],
        },
    )
    _write_json(
        run_dir / "scale_compressed_repair_report.json",
        {
            "scale_compressed_operator_receipt": True,
            "logical_repair_rounds": 24,
            "cmb_parameter_readouts": {"q_IR": 0.25, "ell_IR": 32.0},
        },
    )
    _write_json(
        run_dir / "screen_capacity_closure_report.json",
        {
            "observed_branch_normalization": {"N_CRC": 1000.0},
            "readiness_gates": {"observed_branch_N_scr_readout_available": True},
        },
    )
    _write_json(run_dir / "strict_neutral_bulk_report.json", {"strict_neutral_bulk": False})
    _write_json(
        run_dir / "oph_compressed_likelihood_report.json",
        {"official_likelihood_ready": True, "cdm_limit_regression_passed": True},
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
