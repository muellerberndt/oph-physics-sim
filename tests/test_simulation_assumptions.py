from oph_fpe.simulation_assumptions import simulation_assumption_manifest


def test_visual_assumptions_complete_scene_without_promoting_proof_receipts():
    report = simulation_assumption_manifest(
        {
            "simulation_assumptions": {
                "enabled": True,
                "profile": "known_observer_universe_v1",
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
    assumed = {
        "screen_s2": True,
        "bw_2pi_geometric_branch": True,
        "h3_observer_chart": True,
        "record_population_on_h3": True,
        "ds4_open_slicing_background": True,
        "positive_cosmological_constant": True,
        "observer_tetrad_visualization": True,
        "topological_defects_render_as_matter": True,
    }
    report = simulation_assumption_manifest(
        {
            "simulation_assumptions": {
                "enabled": True,
                "scope": "visualization_only",
                "assumed": assumed,
                "ds4": {"curvature_radius": 2.0, "hubble_parameter": 2.0},
            }
        }
    )

    ds4 = report["ds4_visualization_parameters"]
    assert ds4["curvature_radius"] == 2.0
    assert ds4["hubble_parameter"] == 0.5
    assert ds4["de_sitter_radius_relation_valid"] is False
    assert report["SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"] is False
