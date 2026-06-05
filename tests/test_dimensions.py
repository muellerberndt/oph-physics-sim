from oph_fpe.bulk import dimension_report, graph_distance_matrix
from oph_fpe.core.graph import build_patch_graph


def test_planted_2d_grid_dimension_is_close_to_two():
    graph = build_patch_graph({"family": "grid_2d", "patch_count": 100}, seed=1)
    _, distances = graph_distance_matrix(graph)
    report = dimension_report(graph, distances)

    assert 1.4 <= report["volume_growth_dimension"]["estimate"] <= 2.5
    assert 1.4 <= report["correlation_dimension"]["estimate"] <= 2.5


def test_planted_3d_lattice_dimension_is_above_two():
    graph = build_patch_graph({"family": "lattice_3d", "patch_count": 343}, seed=1)
    _, distances = graph_distance_matrix(graph)
    report = dimension_report(graph, distances)

    assert report["volume_growth_dimension"]["estimate"] > 2.0
    assert report["correlation_dimension"]["estimate"] > 2.0
