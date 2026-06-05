from oph_fpe.core.graph import build_patch_graph
from oph_fpe.core.patchnet import PatchNet
from oph_fpe.core.records import update_records
from oph_fpe.dynamics.repair import RepairKernel
from oph_fpe.groups import get_group


def test_single_collar_flip_repairs():
    group = get_group("Z2")
    graph = build_patch_graph({"family": "path", "patch_count": 2}, seed=1)
    net = PatchNet.synchronized(graph, group)
    net.states[0].ports[1] = 1

    assert net.total_phi() == 1.0

    event = RepairKernel("local_best", hot_metropolis=False, seed=3).step(net, cycle=0, beta=1.0)

    assert event.accepted
    assert event.delta_phi < 0
    assert net.total_phi() == 0.0


def test_records_commit_only_after_stability_window():
    group = get_group("Z2")
    graph = build_patch_graph({"family": "path", "patch_count": 2}, seed=1)
    net = PatchNet.synchronized(graph, group)

    assert update_records(net, cycle=0, commit_cycles=3) == []
    assert update_records(net, cycle=1, commit_cycles=3) == []
    events = update_records(net, cycle=2, commit_cycles=3)

    assert len(events) == 2
    assert all(event.stable_count == 3 for event in events)
