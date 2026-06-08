from pathlib import Path

import yaml

from oph_universe.arrow.scenarios import run_scenario


def _config(name: str) -> dict:
    path = Path("experiments/arrow_configs") / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_faithful_record_chain_mvp_passes():
    result = run_scenario(_config("faithful_record_chain"), seed=1)
    assert result.summary["selected_is_faithful"] is True
    assert result.summary["bound_satisfied"] is True
    assert result.summary["branch_orientation"] == "forward"
    assert result.summary["fake_deficit_bits"] == 0.0


def test_hidden_export_sweep_relaxes_entropy_bound():
    result = run_scenario(_config("hidden_export_sweep"), seed=1)
    rows = result.summary["sweep"]
    assert rows[0]["s_of_bound_bits"] < rows[-1]["s_of_bound_bits"]
    assert result.summary["linear_relaxation_pass"] is True


def test_fake_sweep_matches_2_power_minus_f():
    cfg = _config("fake_past_sweep")
    cfg["trials"] = 20000
    cfg["fake_deficit_bits"] = 10
    result = run_scenario(cfg, seed=2)
    assert result.summary["within_binomial_tolerance"] is True


def test_janus_neck_two_arrows():
    result = run_scenario(_config("janus_neck"), seed=3)
    assert result.summary["plus_orientation"] == "forward"
    assert result.summary["minus_orientation_away_from_neck"] is True


def test_record_reversal_cost_scenario():
    result = run_scenario(_config("record_reversal"), seed=4)
    assert result.summary["erasure_cost_bits"] >= 56
    assert result.summary["entropy_export_required_bits"] >= 56


def test_coarse_grain_refinement_stability():
    result = run_scenario(_config("coarse_grain_refinement"), seed=5)
    assert result.summary["coarse_fine_stable"] is True
