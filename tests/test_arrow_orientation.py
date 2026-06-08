from oph_universe.arrow.orientation import (
    RecordTowerPoint,
    infer_record_arrow,
    record_reversal_erasure_cost_bits,
)


def test_record_tower_orientation_forward():
    tower = [
        RecordTowerPoint(t=0, record_capacity_bits=1, atom_count=2, payload_bits=1),
        RecordTowerPoint(t=1, record_capacity_bits=4, atom_count=16, payload_bits=4),
    ]
    assert infer_record_arrow(tower) == "forward"


def test_record_reversal_erasure_cost():
    assert record_reversal_erasure_cost_bits(4, 1) == 3

