from copy import deepcopy

from oph_fpe.simulation_assumptions import (
    ASSUMPTION_DEFINITIONS,
    revalidate_simulation_assumption_manifest,
    simulation_assumption_manifest,
)


def _complete_config() -> dict:
    return {
        "simulation_assumptions": {
            "enabled": True,
            "scope": "visualization_only",
            "profile": "known_observer_universe_v1",
            "assumed": {name: True for name in ASSUMPTION_DEFINITIONS},
            "ds4": {
                "curvature_radius": 1.0,
                "hubble_parameter": 1.0,
                "proper_time_min_over_h": 0.05,
                "proper_time_max_over_h": 3.0,
                "time_sample_count": 96,
                "units": "simulation_units",
            },
            "observer_camera": {
                "coordinate_system": "h3_hyperboloid_spatial_components_v1",
                "h3_radial_coordinate": 1.18,
                "look_at": [0.0, 0.0, 0.0],
                "orientation": "inward_radial",
                "fov_degrees": 72.0,
            },
            "cmb_visualization": {
                "reference_label": "pinned-test-reference",
                "reference_path": "data/test-reference.txt",
                "reference_source_url": "https://example.invalid/test-reference.txt",
                "reference_sha256": "sha256:" + "a" * 64,
                "transfer_model": "pinned_tt_reference_best_fit_visualization",
                "sky_realization_seed": 17,
            },
        }
    }


def test_visual_assumptions_complete_scene_without_promoting_proof_receipts():
    report = simulation_assumption_manifest(_complete_config())

    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is True
    assert report["computed_theorem_receipts_unchanged"] is True
    assert all(row["proof_receipt"] is False for row in report["assumptions"].values())


def test_visual_assumptions_fail_closed_when_not_explicitly_enabled():
    report = simulation_assumption_manifest({})

    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is False
    assert report["missing_assumptions"]


def test_visual_assumptions_require_literal_booleans():
    report = simulation_assumption_manifest(
        {
            "simulation_assumptions": {
                "enabled": "true",
                "assumed": {"screen_s2": "true"},
            }
        }
    )

    assert report["enabled"] is False
    assert report["assumptions"]["screen_s2"]["assumed"] is False
    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is False


def test_visual_assumptions_cannot_claim_a_proof_scope():
    report = simulation_assumption_manifest(
        {
            "simulation_assumptions": {
                "enabled": True,
                "scope": "computed_theorem_receipts",
                "assumed": {
                    "screen_s2": True,
                    "bw_2pi_geometric_branch": True,
                    "h3_observer_chart": True,
                    "record_population_on_h3": True,
                    "ds4_open_slicing_background": True,
                    "positive_cosmological_constant": True,
                    "observer_tetrad_visualization": True,
                    "topological_defects_render_as_matter": True,
                },
            }
        }
    )

    assert report["scope_valid"] is False
    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is False
    assert not any(row["assumed"] for row in report["assumptions"].values())


def test_ds4_assumption_parameters_are_finite_and_bounded():
    report = simulation_assumption_manifest(
        {
            "simulation_assumptions": {
                "enabled": True,
                "ds4": {
                    "curvature_radius": float("inf"),
                    "hubble_parameter": float("nan"),
                    "time_sample_count": 10**9,
                },
            }
        }
    )

    ds4 = report["ds4_visualization_parameters"]
    assert ds4["curvature_radius"] == 1.0
    assert ds4["hubble_parameter"] == 1.0
    assert ds4["time_sample_count"] == 4096
    assert ds4["parameter_inputs_valid"] is False
    assert ds4["de_sitter_radius_relation_valid"] is False


def test_ds4_radius_and_hubble_must_obey_de_sitter_relation():
    config = _complete_config()
    config["simulation_assumptions"]["ds4"].update(
        {"curvature_radius": 2.0, "hubble_parameter": 2.0}
    )
    report = simulation_assumption_manifest(config)

    ds4 = report["ds4_visualization_parameters"]
    assert ds4["curvature_radius"] == 2.0
    assert ds4["hubble_parameter"] == 0.5
    assert ds4["de_sitter_radius_relation_valid"] is False
    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is False


def test_numeric_parameters_reject_boolean_and_nonintegral_values():
    config = _complete_config()
    section = config["simulation_assumptions"]
    section["ds4"]["curvature_radius"] = True
    section["ds4"]["time_sample_count"] = 96.0
    section["observer_camera"]["h3_radial_coordinate"] = False
    section["observer_camera"]["look_at"] = [True, 0.0, 0.0]
    section["cmb_visualization"]["sky_realization_seed"] = 17.0

    report = simulation_assumption_manifest(config)

    assert report["parameter_sets_valid"] is False
    assert report["SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT"] is False
    assert report["ds4_visualization_parameters"]["parameter_inputs_valid"] is False
    assert report["ds4_visualization_parameters"]["time_sample_count_valid"] is False
    assert report["observer_camera_visualization_parameters"]["parameter_inputs_valid"] is False
    assert report["cmb_visualization_parameters"]["parameter_inputs_valid"] is False


def test_serialized_manifest_revalidation_ignores_copied_receipts_and_rejects_tampering():
    manifest = simulation_assumption_manifest(_complete_config())
    tampered = deepcopy(manifest)
    tampered["assumptions"]["screen_s2"]["assumed"] = "true"
    tampered["SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT"] = True
    tampered["proof_receipt"] = True

    validated = revalidate_simulation_assumption_manifest(tampered)

    assert validated["manifest_integrity_valid"] is False
    assert validated["SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT"] is False
    assert validated["assumptions"]["screen_s2"]["assumed"] is False
