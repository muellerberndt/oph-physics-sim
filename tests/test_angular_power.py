import numpy as np

from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.cosmology.angular_power import angular_power_report


def test_dipole_field_has_l1_dominance():
    points = fibonacci_sphere_points(512)
    report = angular_power_report(
        points,
        {"z_dipole": points[:, 2]},
        ell_max=8,
        pair_samples=0,
        seed=7,
        exact_pair_limit=512 * 512 + 1,
    )

    spectrum = report["fields"]["z_dipole"]["spectrum"]
    c_by_l = {row["ell"]: row["C_ell"] for row in spectrum}

    assert c_by_l[1] > c_by_l[2]
    assert c_by_l[1] > c_by_l[3]
    assert c_by_l[1] > abs(c_by_l[4])
    comparison = report["fields"]["z_dipole"]["control_comparison"]
    assert comparison["control_count"] == 2
    assert comparison["min_relative_l2_delta"] >= 0.0
    assert "shuffled_field" in comparison["by_control"]


def test_parallel_angular_power_matches_single_job():
    points = fibonacci_sphere_points(768)
    fields = {"z_dipole": points[:, 2], "x_dipole": points[:, 0]}
    single = angular_power_report(
        points,
        fields,
        ell_max=6,
        pair_samples=0,
        seed=11,
        controls=["shuffled_field", "random_gaussian"],
        n_jobs=1,
    )
    parallel = angular_power_report(
        points,
        fields,
        ell_max=6,
        pair_samples=0,
        seed=11,
        controls=["shuffled_field", "random_gaussian"],
        n_jobs=2,
    )

    assert parallel["n_jobs"] == 2
    for field_name in fields:
        single_spectrum = single["fields"][field_name]["spectrum"]
        parallel_spectrum = parallel["fields"][field_name]["spectrum"]
        assert [row["C_ell"] for row in parallel_spectrum] == [
            row["C_ell"] for row in single_spectrum
        ]
        for control_name in ["shuffled_field", "random_gaussian"]:
            single_control = single["controls"][field_name][control_name]["spectrum"]
            parallel_control = parallel["controls"][field_name][control_name]["spectrum"]
            assert [row["C_ell"] for row in parallel_control] == [
                row["C_ell"] for row in single_control
            ]
