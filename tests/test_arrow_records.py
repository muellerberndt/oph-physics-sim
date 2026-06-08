from oph_universe.arrow.records import RecordAlgebra, RecordEvent, record_tower_capacity


def _event(record_id: str, source: str, value: str, bits: float) -> RecordEvent:
    return RecordEvent(record_id, 0, (0,), source, bits, value, "id", 0.0, (), ("p",), "s")


def test_record_algebra_payload_counts_nonredundant_sources():
    alg = RecordAlgebra("a", 0, {})
    alg.append(_event("r1", "x", "same", 8.0))
    alg.append(_event("r2", "x", "same", 8.0))
    alg.append(_event("r3", "x", "new", 4.0))
    assert alg.payload_bits() == 12.0
    assert alg.capacity_bits() == 3.0


def test_record_tower_capacity():
    a = RecordAlgebra("a", 0, {"r": _event("r", "x", "v", 1)})
    b = RecordAlgebra("b", 1, {"r": _event("r", "x", "v", 1), "s": _event("s", "y", "w", 1)})
    assert record_tower_capacity([a, b]) == [1.0, 2.0]

