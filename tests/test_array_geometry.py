from __future__ import annotations

import pytest

from oph_fpe.core.array_geometry import array_screen_geometry_from_config
from oph_fpe.scale.array_screen import _knn_edges


def test_array_geometry_instantiates_nested_icosahedral_cell_federation():
    geometry = array_screen_geometry_from_config(
        {
            "family": "nested_geodesic_icosahedral",
            "patch_basis": "cells",
            "refinement_level": 2,
            "nominal_patch_count": 256,
        },
        knn_builder=_knn_edges,
    )
    report = geometry.report

    assert geometry.patch_count == 320
    assert geometry.edge_count == 480
    assert report["nominal_patch_count"] == 256
    assert report["actual_patch_count"] == 320
    assert report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is True
    assert report["NESTED_ICOSAHEDRAL_LINEAGE_RECEIPT"] is True
    assert report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is True
    assert report["TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT"] is True
    assert report["PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE"] is False


def test_array_geometry_does_not_fabricate_nominal_rung_as_exact_mesh():
    with pytest.raises(ValueError, match="use nominal_patch_count"):
        array_screen_geometry_from_config(
            {
                "family": "nested_geodesic_icosahedral",
                "patch_basis": "cells",
                "refinement_level": 4,
                "patch_count": 4096,
            },
            knn_builder=_knn_edges,
        )


def test_legacy_fibonacci_cellulation_is_explicitly_only_a_control():
    geometry = array_screen_geometry_from_config(
        {"family": "fibonacci_sphere", "patch_count": 64, "neighbors": 4},
        knn_builder=_knn_edges,
    )

    assert geometry.patch_count == 64
    assert geometry.report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is False
    assert geometry.report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is False
