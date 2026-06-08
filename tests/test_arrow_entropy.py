from oph_universe.arrow.entropy import (
    fake_deficit_bits,
    landauer_record_bound_ok,
    selected_ancestry_entropy_bound,
)


def test_landauer_record_bound():
    assert landauer_record_bound_ok(10.0, 5.0, 5.0)
    assert not landauer_record_bound_ok(10.0, 4.0, 5.0)


def test_fake_deficit():
    assert fake_deficit_bits(100, 60, 30, 10) == 0
    assert fake_deficit_bits(100, 40, 30, 10) == 20


def test_selected_ancestry_entropy_bound_relaxes_with_hidden_budget():
    assert selected_ancestry_entropy_bound(256, 64, 0, 0, 0) == 192
    assert selected_ancestry_entropy_bound(256, 64, 8, 0, 0) == 200

