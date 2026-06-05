import numpy as np

from oph_fpe.bulk.cap_geometry import sample_caps
from oph_fpe.bulk.markov_collar import collar_markov_report
from oph_fpe.core.graph import fibonacci_sphere_points


def test_collar_markov_report_schema():
    points = fibonacci_sphere_points(512)
    caps = sample_caps(points, count=3, theta_values=[0.55, 0.75], seed=5, collar_width=0.08)
    fields = {
        "record_signature": np.arange(points.shape[0]) % 17,
        "committed_mask": np.ones(points.shape[0]),
        "stable_count": np.arange(points.shape[0]) % 5,
        "repair_load": np.linspace(0.0, 1.0, points.shape[0]),
        "s3_class_density": np.linspace(1.0, 0.0, points.shape[0]),
    }

    report = collar_markov_report(points, caps, fields, max_triplets=128, seed=9)

    assert report["mode"] == "diagonal_empirical_collar_state"
    assert report["cap_count"] == 3
    assert report["rows"]
    assert "epsilon_cmi" in report["rows"][0]
    assert report["rows"][0]["claim_boundary"].startswith("classical diagonal")
