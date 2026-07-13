from math import isclose, sqrt
from pathlib import Path

import json
import pytest

from oph_fpe.constants.oph_pixel import (
    ALPHA_INV_SOURCE_CANDIDATE,
    PIXEL_MODE_SOURCE_CANDIDATE,
    P_STAR,
    P_SOURCE_CANDIDATE,
    OPHPixelConstants,
    PixelParameterProfile,
    cap_area_planck,
    cap_entropy_capacity,
    equal_cell_entropy,
    pixel_constants_for_profile,
    pixel_parameter_profile,
    scale_factor_from_patch_count,
    screen_radius_planck,
)
from oph_fpe.core.pixel_scale import pixel_scale_from_config
from oph_fpe.core.screen_microphysics import screen_microphysics_from_config
from oph_fpe.experiments import load_config
from oph_fpe.scale import run_array_screen_config


def test_pixel_scale_derives_local_units():
    scale = pixel_scale_from_config({"oph_constants": {"P": P_STAR, "P_source": "endpoint_public"}})

    assert isclose(scale.a_cell, P_STAR)
    assert isclose(scale.cell_area_planck, P_STAR)
    assert isclose(scale.cell_entropy_capacity, P_STAR / 4.0)
    assert isclose(scale.ellbar_shared, P_STAR / 4.0)
    assert isclose(scale.g_natural, 1.0)
    assert isclose(scale.length_unit, sqrt(P_STAR))
    assert scale.source == "endpoint_public"
    assert scale.pixel_mode == "endpoint_calibrated"


def test_oph_pixel_constants_and_cap_capacity_helpers():
    constants = OPHPixelConstants()
    weights = [1.0, 0.5, 0.0]

    assert isclose(constants.P, P_STAR)
    assert isclose(constants.alpha_from_P, (P_STAR - constants.phi) / constants.sqrt_pi)
    assert isclose(constants.cell_entropy_capacity, P_STAR / 4.0)
    assert isclose(cap_area_planck(weights, P_STAR), 1.5 * P_STAR)
    assert isclose(cap_entropy_capacity(weights, P_STAR / 4.0), 1.5 * P_STAR / 4.0)
    assert isclose(equal_cell_entropy(3, P_STAR)[0], P_STAR / 4.0)
    assert isclose(scale_factor_from_patch_count(400, 100), 2.0)
    assert isclose(screen_radius_planck(100, P_STAR), sqrt((100 * P_STAR) / (4.0 * 3.141592653589793)))


def test_source_candidate_pixel_mode_is_distinct_from_endpoint():
    scale = pixel_scale_from_config({"oph_constants": {"pixel_mode": PIXEL_MODE_SOURCE_CANDIDATE}})
    as_json = scale.as_jsonable()

    assert scale.source == "source_candidate"
    assert scale.pixel_mode == PIXEL_MODE_SOURCE_CANDIDATE
    assert scale.epistemic_profile == PixelParameterProfile.SOURCE_CANDIDATE.value
    assert isclose(scale.ratio_p, P_SOURCE_CANDIDATE)
    assert isclose(as_json["alpha_inverse_from_P"], ALPHA_INV_SOURCE_CANDIDATE, rel_tol=1e-12)


def test_pixel_parameter_profiles_keep_empirical_and_measured_branches_out_of_core():
    empirical = pixel_parameter_profile(PixelParameterProfile.EMPIRICAL_HADRON_CLOSURE)
    measured = pixel_parameter_profile(PixelParameterProfile.MEASURED_COMPARISON)
    source = pixel_constants_for_profile(PixelParameterProfile.SOURCE_CANDIDATE)

    assert empirical.interval is not None
    assert empirical.recovered_core_allowed is False
    assert measured.simulation_role.startswith("comparison only")
    assert source.epistemic_profile == "source_candidate"
    with pytest.raises(ValueError, match="interval-valued"):
        pixel_constants_for_profile(PixelParameterProfile.EMPIRICAL_HADRON_CLOSURE)


def test_pixel_scale_rejects_implicit_interval_profile():
    with pytest.raises(ValueError, match="interval-valued"):
        pixel_scale_from_config(
            {"oph_constants": {"epistemic_profile": "empirical_hadron_closure"}}
        )


def test_array_screen_receipts_include_pixel_scale(tmp_path: Path):
    config = load_config(Path("configs/e1_s3_modular_screen_4k.yml"))
    config = dict(config)
    config["run_id"] = "pixel_scale_array_smoke"
    config["graph"] = dict(config["graph"], patch_count=128, neighbors=6)
    config["dynamics"] = dict(config["dynamics"], cycles=4, repairs_per_cycle=256)
    config["observables"] = dict(config["observables"])
    config["observables"]["modular_lift"] = {"max_points": 1024, "center_samples": 64}

    result = run_array_screen_config(config, tmp_path)
    pixel_scale = json.loads((Path(result["path"]) / "pixel_scale.json").read_text(encoding="utf-8"))
    screen_microphysics = json.loads(
        (Path(result["path"]) / "screen_microphysics.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((Path(result["path"]) / "manifest.json").read_text(encoding="utf-8"))

    assert pixel_scale["P"] == P_STAR
    assert pixel_scale["ratio_P"] == P_STAR
    assert pixel_scale["P_source"] == "endpoint_public"
    assert "not a BW normalization factor" in pixel_scale["role"]
    assert screen_microphysics["carrier"] == "federated_echosahedral_patch"
    assert screen_microphysics["ports_per_patch"] == 12
    assert isclose(screen_microphysics["cell_area"], P_STAR)
    assert screen_microphysics["screen_units"]["mode"] == "numerical_regulator"
    assert screen_microphysics["screen_units"]["total_area_planck"] is None
    assert isclose(screen_microphysics["screen_units"]["regulator_entropy_weight_sum"], 128 * P_STAR / 4.0)
    assert manifest["pixel_scale"]["ratio_P"] == P_STAR
    assert manifest["oph_constants"]["P"] == P_STAR
    assert manifest["screen_microphysics"]["pixel_scale"]["ratio_P"] == P_STAR
    assert (Path(result["path"]) / "pixel_report.json").exists()


def test_screen_microphysics_uses_p_as_cell_area():
    micro = screen_microphysics_from_config(
        {
            "oph_constants": {"P": P_STAR, "P_source": "endpoint_public"},
            "screen": {"ports_per_patch": 12},
        },
        patch_count=100,
        edge_count=600,
    )

    assert isclose(micro.cell_area, P_STAR)
    assert isclose(micro.screen_area, 100 * P_STAR)
    assert micro.port_budget == 1200
    assert isclose(micro.routed_port_fraction, 1.0)


def test_physical_cell_toy_mode_reports_radius():
    micro = screen_microphysics_from_config(
        {
            "oph_constants": {"P": P_STAR, "P_source": "endpoint_public"},
            "screen_units": {"mode": "physical_cell_toy_universe"},
            "screen": {"ports_per_patch": 12},
        },
        patch_count=100,
        edge_count=600,
    ).as_jsonable()

    assert micro["screen_units"]["mode"] == "physical_cell_toy_universe"
    assert isclose(micro["screen_units"]["total_area_planck"], 100 * P_STAR)
    assert isclose(micro["screen_units"]["total_entropy_capacity"], 100 * P_STAR / 4.0)
    assert isclose(micro["screen_units"]["physical_radius_planck"], screen_radius_planck(100, P_STAR))
