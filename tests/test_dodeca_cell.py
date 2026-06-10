from oph_fpe.microphysics.dodeca_cell import (
    dodeca_cell_report,
    dodecahedron_face_normals,
    dodecahedral_cell,
    pentagon_loop_modes,
)


def test_dodeca_cell_report_passes():
    report = dodeca_cell_report()

    assert report["passed"] is True
    assert report["node_count"] == 20
    assert report["edge_count"] == 30
    assert report["face_count"] == 12
    assert report["neutral_oph_bulk_claim"] is False


def test_pentagon_loop_modes_close():
    cell = dodecahedral_cell()
    rows = pentagon_loop_modes(cell, max_mode=1)

    assert len(rows) == 12
    assert all(row["loop_length"] == 5 for row in rows)
    assert all(row["closure_residual"] < 1.0e-12 for row in rows)


def test_face_normals_are_screen_points():
    points = dodecahedron_face_normals()

    assert points.shape == (12, 3)
    assert abs((points**2).sum(axis=1).mean() - 1.0) < 1.0e-12
