from oph_fpe.core.graph import build_patch_graph
from oph_fpe.core.patchnet import PatchNet
from oph_fpe.defects import cycle_holonomy, scan_holonomy_defects
from oph_fpe.groups import get_group


def test_z2_clean_sync_has_zero_phi():
    group = get_group("Z2")
    graph = build_patch_graph({"family": "cycle", "patch_count": 8}, seed=1)
    net = PatchNet.synchronized(graph, group)

    assert net.total_phi() == 0.0
    assert scan_holonomy_defects(net) == []


def test_z2_frustrated_cycle_is_exposed():
    group = get_group("Z2")
    graph = build_patch_graph({"family": "cycle", "patch_count": 5}, seed=1)
    net = PatchNet.synchronized(graph, group)
    net.set_directed_gauge(0, 1, 1)

    defects = scan_holonomy_defects(net)

    assert len(defects) == 1
    assert defects[0].holonomy == 1


def test_patchnet_mismatch_uses_directed_gauge_transport():
    group = get_group("Z2")
    graph = build_patch_graph({"family": "cycle", "patch_count": 4}, seed=1)
    net = PatchNet.synchronized(graph, group)

    net.set_directed_gauge(0, 1, 1)

    assert net.edge_mismatch(0, 1).distance == 1.0
    net.states[0].ports[1] = 1
    assert net.edge_mismatch(0, 1).distance == 0.0
    assert net.edge_mismatch(1, 0).distance == 0.0


def test_s3_holonomy_depends_on_orientation():
    group = get_group("S3")
    graph = build_patch_graph({"family": "cycle", "patch_count": 3}, seed=1)
    net = PatchNet.synchronized(graph, group)
    a = group.parse("(01)")
    b = group.parse("(12)")
    net.set_directed_gauge(0, 1, a)
    net.set_directed_gauge(1, 2, b)

    forward = cycle_holonomy(net, [0, 1, 2])
    reverse = cycle_holonomy(net, [0, 2, 1])

    assert forward != group.identity
    assert reverse != group.identity
    assert forward != reverse
