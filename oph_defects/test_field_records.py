"""Viewer boundaries must preserve the independently reviewed scientific records."""
from copy import deepcopy
from fractions import Fraction as F
import json
import pytest

from . import field_records as viewer


@pytest.mark.parametrize('value', ['1/3', '1/10', '999999999999/1000000000000',
    '232756804634/10000000000000000000000000000000000', '18446744073709551616'])
def test_display_bounds_are_outward_exact_rationals(value):
    shown = viewer.scalar(value)
    assert F(shown['lowerText']) <= F(value) <= F(shown['upperText'])


def test_export_retains_source_units_and_complete_outcomes():
    value = viewer.build()
    assert value['scope']['physicalParticleIdentities'] == []
    assert value['scope']['sourceDerivedMassesGeV'] == []
    assert not value['scope']['sameAsClassicalLoadRun']
    assert [r['records'] for r in value['quark']] == [88, 88, 160]
    assert value['quark'][0]['energy'] == value['quark'][1]['energy']
    assert F(value['quark'][2]['energy']['exact'][0]) > F(value['quark'][0]['energy']['exact'][1])
    assert all(0 < F(r['probability']['exact'][0]) < F(r['probability']['exact'][1]) < F(1,10**20) for r in value['quark'])
    assert F(value['interaction']['completeLawSeparation']['exact']) > 2*F(value['interaction']['cutoffError']['exact'])
    assert F(value['preparation'][1]['groundError']['exact']) < F(value['preparation'][0]['groundError']['exact'])


def test_public_science_cannot_be_changed_behind_valid_source_pins(monkeypatch):
    original = viewer.receipt
    def altered(root, path):
        value = deepcopy(original(root, path))
        value['families']['uc']['coarse_original_attempt_records'] = 1
        return value
    monkeypatch.setattr(viewer, 'receipt', altered)
    with pytest.raises(ValueError, match='scientific payload changed'):
        viewer.load(viewer.theory_repository(), 'native_quark_spatial_spectrum')


def test_generated_browser_data_matches_the_reviewed_sources():
    assert json.loads(viewer.OUTPUT.read_text()) == viewer.build()
