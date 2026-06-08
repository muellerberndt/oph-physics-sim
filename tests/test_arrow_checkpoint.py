from oph_universe.arrow.checkpoints import Checkpoint, checkpoint_equivalent, future_law_signature
from oph_universe.arrow.records import RecordAlgebra


def _checkpoint(name: str) -> Checkpoint:
    return Checkpoint(name, 0, RecordAlgebra("alg", 0, {}), "acc", "iface", "sched", "prov", "macro", 10, 20, 0)


def test_checkpoint_equivalence_same_future_law():
    a = _checkpoint("a")
    b = _checkpoint("b")
    assert checkpoint_equivalent(a, b)
    assert future_law_signature(a, "law") == future_law_signature(b, "law")

