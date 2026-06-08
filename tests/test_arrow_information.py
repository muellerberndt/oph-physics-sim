import math

from oph_universe.arrow.information import (
    chain_rule_payload,
    fano_payload_lower_bound,
    mutual_information_discrete,
)


def test_fano_payload_lower_bound():
    assert math.isclose(fano_payload_lower_bound(8.0, 256, 0.0), 8.0)
    assert fano_payload_lower_bound(8.0, 256, 0.1) < 8.0


def test_mutual_information_copy():
    x = [0, 1, 0, 1] * 16
    assert mutual_information_discrete(x, x) == 1.0
    y = [0, 0, 1, 1] * 16
    assert mutual_information_discrete(x, y) == 0.0


def test_redundant_records_do_not_double_count_payload():
    source = [0, 1, 0, 1] * 16
    increments = chain_rule_payload(source, [source, list(source), list(source)])
    assert math.isclose(increments[0], 1.0)
    assert increments[1] == 0.0
    assert increments[2] == 0.0

