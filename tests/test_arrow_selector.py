import inspect

from oph_universe.arrow.scenarios import _dummy_candidate
from oph_universe.arrow.selector import ArrowSelectorWeights, j_arrow, lane8_falsifier, select_normal_form_ancestry


def test_selector_has_no_entropy_term():
    source = inspect.getsource(j_arrow)
    assert "s_of" not in source
    assert "n_of" not in source


def test_selector_prefers_faithful_when_fake_penalty_large():
    faithful = _dummy_candidate("faithful", i_rec=64, n_res=64)
    faithful.is_faithful = True
    fake = _dummy_candidate("fake", i_rec=64, n_res=0)
    fake.is_faithful = False
    selected = select_normal_form_ancestry([fake, faithful], ArrowSelectorWeights(lambda_fake=100))
    assert selected.candidate_id == "faithful"


def test_high_entropy_zero_fake_deficit_flags_falsifier():
    fake = _dummy_candidate("fake_zero_deficit", i_rec=64, n_res=64)
    fake.is_faithful = False
    selected = select_normal_form_ancestry([fake], ArrowSelectorWeights())
    assert lane8_falsifier(fake, selected) is True

