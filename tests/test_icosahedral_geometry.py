import math

import numpy as np
import pytest

from oph_fpe.core.graph import build_patch_graph
from oph_fpe.core.icosahedral import (
    UnsupportedIcosahedralPatchCount,
    build_geodesic_icosahedral_tower,
    geodesic_icosahedral_graph,
    geodesic_icosahedral_patch_arrays,
    icosahedral_a5_equivariance_report,
    icosahedral_defect_port_report,
    nominal_campaign_rung_mapping,
    supported_icosahedral_count,
)
from oph_fpe.experiments import _graph_json


@pytest.mark.parametrize("level", range(4))
def test_geodesic_levels_have_exact_closed_sphere_topology(level):
    mesh = build_geodesic_icosahedral_tower(level).levels[level]

    assert mesh.vertex_count == 10 * 4**level + 2
    assert mesh.edge_count == 30 * 4**level
    assert mesh.face_count == 20 * 4**level
    assert mesh.euler_characteristic == 2
    assert np.max(np.abs(np.linalg.norm(mesh.vertices, axis=1) - 1.0)) < 1.0e-14
    assert abs(np.sum(mesh.spherical_face_areas) - 4.0 * math.pi) < 1.0e-12
    assert mesh.receipt()["GEODESIC_ICOSAHEDRAL_GEOMETRY_RECEIPT"] is True


def test_level_zero_is_a_regular_icosahedron_not_only_a_sphere_triangulation():
    mesh = build_geodesic_icosahedral_tower(0).levels[0]
    edge_lengths = np.linalg.norm(
        mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]],
        axis=1,
    )

    assert mesh.vertex_count == 12
    assert mesh.edge_count == 30
    assert mesh.face_count == 20
    assert np.max(edge_lengths) - np.min(edge_lengths) < 1.0e-14
    assert np.max(mesh.spherical_face_areas) - np.min(mesh.spherical_face_areas) < 1.0e-14
    assert np.allclose(mesh.spherical_face_areas, math.pi / 5.0, atol=1.0e-14)
    for vertex in mesh.vertices:
        assert np.min(np.linalg.norm(mesh.vertices + vertex, axis=1)) < 1.0e-14


def test_refinement_is_nested_and_deduplicates_shared_edge_midpoints():
    tower = build_geodesic_icosahedral_tower(3)

    for level in range(1, 4):
        coarse = tower.levels[level - 1]
        fine = tower.levels[level]
        assert np.array_equal(fine.vertices[: coarse.vertex_count], coarse.vertices)
        assert fine.vertex_count - coarse.vertex_count == coarse.edge_count

        coarse_edges = {tuple(int(value) for value in edge) for edge in coarse.edges}
        for vertex in range(coarse.vertex_count):
            assert fine.vertex_parent_support[vertex] == ((vertex, 1.0),)
        new_supports = fine.vertex_parent_support[coarse.vertex_count :]
        assert len(new_supports) == coarse.edge_count
        assert {
            tuple(sorted(parent for parent, _ in support)) for support in new_supports
        } == coarse_edges
        assert all(tuple(weight for _, weight in support) == (0.5, 0.5) for support in new_supports)


def test_cell_lineage_partitions_parent_area_and_has_four_children():
    tower = build_geodesic_icosahedral_tower(3)

    for level, mapping in enumerate(tower.cell_refinements, start=1):
        coarse = tower.levels[level - 1]
        fine = tower.levels[level]
        assert mapping.child_to_parent.shape == (fine.face_count,)
        assert len(mapping.children_by_parent) == coarse.face_count
        assert sorted(child for children in mapping.children_by_parent for child in children) == list(
            range(fine.face_count)
        )
        for parent, children in enumerate(mapping.children_by_parent):
            child_ids = np.asarray(children, dtype=np.int64)
            assert tuple(mapping.child_to_parent[child_ids]) == (parent,) * 4
            child_area = float(np.sum(fine.spherical_face_areas[child_ids]))
            assert abs(child_area - coarse.spherical_face_areas[parent]) < 1.0e-14


def test_cell_embedding_and_conditional_expectation_are_state_preserving():
    tower = build_geodesic_icosahedral_tower(2)
    rng = np.random.default_rng(73)

    for mapping in tower.cell_refinements:
        coarse_values = rng.normal(size=(len(mapping.children_by_parent), 3))
        fine_values = rng.normal(size=(mapping.child_to_parent.size, 3))

        embedded = mapping.embed(coarse_values)
        assert np.allclose(mapping.conditional_expectation(embedded), coarse_values, atol=1.0e-14)
        assert np.allclose(mapping.embed(np.ones(len(mapping.children_by_parent))), 1.0)

        coarse_state = np.tensordot(mapping.coarse_reference_weights, coarse_values, axes=(0, 0))
        fine_state = np.tensordot(mapping.fine_reference_weights, embedded, axes=(0, 0))
        assert np.allclose(coarse_state, fine_state, atol=1.0e-14)

        expected_fine = mapping.embed(mapping.conditional_expectation(fine_values))
        assert np.allclose(
            mapping.embed(mapping.conditional_expectation(expected_fine)),
            expected_fine,
            atol=1.0e-14,
        )
        assert mapping.normalization_residual < 1.0e-14
        assert mapping.state_preservation_residual < 1.0e-14
        assert mapping.receipt()["PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE"] is False


def test_multilevel_maps_compose_and_expose_cell_ancestry():
    tower = build_geodesic_icosahedral_tower(3)
    coarse = np.arange(tower.levels[0].face_count, dtype=float)

    staged_embedding = tower.cell_refinements[2].embed(
        tower.cell_refinements[1].embed(tower.cell_refinements[0].embed(coarse))
    )
    tower_embedding = tower.embed_cells(coarse, coarse_level=0, fine_level=3)
    assert np.array_equal(tower_embedding, staged_embedding)

    ancestors = tower.cell_ancestor_ids(fine_level=3, coarse_level=0)
    assert np.array_equal(tower_embedding, coarse[ancestors])
    assert np.allclose(
        tower.conditional_expectation_cells(
            tower_embedding,
            fine_level=3,
            coarse_level=0,
        ),
        coarse,
        atol=1.0e-14,
    )


def test_geometry_and_lineage_hashes_are_deterministic():
    tower = build_geodesic_icosahedral_tower(2)

    assert tower.levels[0].geometry_hash == "e333556ce101cb224d94882270c17e0d7e36469908871ca133e5bbbee2c6eafd"
    assert tower.levels[2].geometry_hash == "5d8764f9c4c92b07f804de4c9e6677800f6be3653a14e509eab247668f4542c7"
    assert tower.cell_refinements[1].map_hash == "4f9ed21cb104e403629ccc72bd2f2bf2f6238c56d80ae16591866f9b7b6a9727"
    assert build_geodesic_icosahedral_tower(2).levels[2].geometry_hash == tower.levels[2].geometry_hash


def test_cell_and_vertex_graphs_use_exact_mesh_entities():
    cell_graph = geodesic_icosahedral_graph(2, patch_basis="cells", nominal_patch_count=256)
    vertex_graph = geodesic_icosahedral_graph(2, patch_basis="vertices")

    assert cell_graph.number_of_nodes() == 320
    assert cell_graph.number_of_edges() == 480
    assert set(dict(cell_graph.degree()).values()) == {3}
    assert cell_graph.graph["nominal_count_is_exact"] is False
    assert cell_graph.graph["nominal_count_mapping"]["lower"]["count"] == 80
    assert cell_graph.graph["nominal_count_mapping"]["upper"]["count"] == 320
    assert vertex_graph.number_of_nodes() == 162
    assert vertex_graph.number_of_edges() == 480
    assert set(dict(vertex_graph.degree()).values()) == {5, 6}
    assert all("screen_xyz" in data for _, data in cell_graph.nodes(data=True))
    assert all("screen_xyz" in data for _, data in vertex_graph.nodes(data=True))


@pytest.mark.parametrize("patch_basis", ["cells", "vertices"])
def test_array_adapter_matches_networkx_topology(patch_basis):
    graph = geodesic_icosahedral_graph(2, patch_basis=patch_basis)
    points, left, right = geodesic_icosahedral_patch_arrays(2, patch_basis=patch_basis)

    assert points.shape == (graph.number_of_nodes(), 3)
    assert left.shape == right.shape == (graph.number_of_edges(),)
    assert np.max(np.abs(np.linalg.norm(points, axis=1) - 1.0)) < 1.0e-14
    assert {tuple(sorted((int(a), int(b)))) for a, b in zip(left, right, strict=True)} == {
        tuple(sorted((int(a), int(b)))) for a, b in graph.edges
    }


def test_graph_builder_rejects_arbitrary_count_unless_policy_is_explicit():
    with pytest.raises(UnsupportedIcosahedralPatchCount, match="not an exact cells count"):
        build_patch_graph(
            {"family": "subdivided_icosahedral_screen", "patch_count": 4096},
            seed=11,
        )

    with pytest.raises(ValueError, match="exactly 320 patches"):
        build_patch_graph(
            {
                "family": "nested_geodesic_icosahedral",
                "patch_basis": "cells",
                "refinement_level": 2,
                "patch_count": 256,
            },
            seed=11,
        )

    graph = build_patch_graph(
        {
            "family": "nested_geodesic_icosahedral",
            "patch_basis": "cells",
            "refinement_level": 2,
            "nominal_patch_count": 256,
        },
        seed=11,
    )
    assert graph.number_of_nodes() == 320
    assert graph.graph["nominal_patch_count"] == 256

    policy_graph = build_patch_graph(
        {
            "family": "nested_geodesic_icosahedral",
            "patch_basis": "cells",
            "patch_count": 256,
            "patch_count_policy": "nearest",
        },
        seed=11,
    )
    assert policy_graph.number_of_nodes() == 320
    assert policy_graph.graph["nominal_patch_count"] == 256
    assert policy_graph.graph["nominal_count_is_exact"] is False

    serialized = _graph_json(graph)
    assert serialized["graph_metadata"]["geometry_family"] == "nested_geodesic_icosahedral"
    assert serialized["graph_metadata"]["geometry_hash"] == graph.graph["geometry_hash"]


def test_nominal_campaign_rungs_are_labels_with_honest_mesh_brackets():
    report = nominal_campaign_rung_mapping()
    expected = {
        4096: (1280, 5120, 5120),
        16384: (5120, 20480, 20480),
        65536: (20480, 81920, 81920),
        262144: (81920, 327680, 327680),
    }

    assert report["NO_ARBITRARY_COUNT_TRUNCATION_RECEIPT"] is True
    for rung in report["rungs"]:
        requested = rung["nominal_patch_count"]
        lower, upper, recommended = expected[requested]
        assert rung["cell_basis"]["exact_supported_count"] is False
        assert rung["cell_basis"]["lower"]["count"] == lower
        assert rung["cell_basis"]["upper"]["count"] == upper
        assert rung["recommended_geometry"]["actual_patch_count"] == recommended
        assert rung["recommended_geometry"]["patch_basis"] == "cells"


def test_supported_count_formula_matches_tower():
    for level in range(5):
        mesh = build_geodesic_icosahedral_tower(level).levels[level]
        assert supported_icosahedral_count(level, "cells") == mesh.face_count
        assert supported_icosahedral_count(level, "vertices") == mesh.vertex_count


def test_exact_a5_action_intertwines_nested_geometry_and_expectations():
    report = icosahedral_a5_equivariance_report(3)

    assert report["base_rotation_count"] == 60
    assert report["integer_permutation_group_closed"] is True
    assert report["integer_permutation_inverses_present"] is True
    assert report["A5_ROTATION_GROUP_ORDER_60_RECEIPT"] is True
    assert report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is True
    assert {row["order"] for row in report["generator_rows"]} == {3, 5}
    assert all(row["passed"] for row in report["level_rows"])
    assert report["PHYSICAL_A5_PORT_EMERGENCE_RECEIPT"] is False


def test_twelve_screen_sieve_channels_are_primal_vertices_not_cell_patches():
    report = icosahedral_defect_port_report(4)

    assert report["charge_domain"] == "primal_vertex_triangulation"
    assert report["large_array_patch_domain"] == "dual_spherical_triangular_cells"
    assert report["domains_are_distinct"] is True
    assert report["persistent_port_vertex_ids"] == list(range(12))
    assert report["TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT"] is True
    assert report["COMBINATORIAL_DEFECT_CHARGE_IS_PHYSICAL_CURVATURE_RECEIPT"] is False
    assert report["PHYSICAL_ATOMIC_DEFECT_PROJECTION_RECEIPT"] is False
    assert all(row["total_combinatorial_charge"] == 12 for row in report["levels"])
    assert all(row["unit_defect_vertex_ids"] == list(range(12)) for row in report["levels"])
