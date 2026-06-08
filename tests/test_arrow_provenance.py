from oph_universe.arrow.provenance import (
    ProvenanceDAG,
    ProvenanceEdge,
    ProvenanceNode,
    provenance_bottleneck_ok,
)


def test_provenance_bottleneck_blocks_unpaid_payload():
    dag = ProvenanceDAG(
        nodes={
            "s": ProvenanceNode("s", 0, "source", 20, 20, True),
            "r": ProvenanceNode("r", 1, "record", 20, 20, True),
        },
        edges=[ProvenanceEdge("s", "r", 10)],
    )
    assert not provenance_bottleneck_ok(dag, {"s"}, {"r"}, 20)
    assert provenance_bottleneck_ok(dag, {"s"}, {"r"}, 10)


def test_hidden_sector_does_not_help_without_accessible_edge():
    dag = ProvenanceDAG(
        nodes={
            "s": ProvenanceNode("s", 0, "source", 10, 10, True),
            "h": ProvenanceNode("h", 0, "hidden", 100, 100, False),
            "r": ProvenanceNode("r", 1, "record", 50, 50, True),
        },
        edges=[ProvenanceEdge("s", "r", 10)],
    )
    assert not provenance_bottleneck_ok(dag, {"s", "h"}, {"r"}, 50)
    dag.edges.append(ProvenanceEdge("h", "r", 40))
    assert provenance_bottleneck_ok(dag, {"s", "h"}, {"r"}, 50)

