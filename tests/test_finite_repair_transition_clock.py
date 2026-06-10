from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.finite_repair_transition_clock import write_finite_repair_transition_clock_report
from oph_fpe.cosmology.repair_clock import repair_clock_report


def test_finite_repair_transition_clock_writes_matrix_and_scalar_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
    )

    report = write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    assert report["finite_transition_matrix_ready"] is True
    assert report["state_count"] == 2
    assert report["transition_count"] == 9
    assert report["primary"]["lambda_2"] is not None
    assert (out / "finite_repair_transition_matrix.npz").exists()
    assert (out / "finite_repair_transition_matrix_report.json").exists()
    assert (out / "scalar_repair_semigroup_report.json").exists()


def test_finite_transition_clock_is_visible_to_repair_clock_but_not_auto_certified(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_observer_views(
        run / "observer_views.jsonl",
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
    )
    write_finite_repair_transition_clock_report(
        run,
        out,
        packet_fields=("checkpoint_class",),
        primary_matrix="raw_empirical",
    )

    aggregate = repair_clock_report([out])

    assert aggregate["summary"]["estimator_count"] == 1
    assert aggregate["rows"][0]["estimator"] == "scalar_repair_semigroup_gap"
    assert aggregate["rows"][0]["source"] == "finite_state_transition_matrix"
    assert aggregate["rows"][0]["finite_lattice_derived"] is True
    assert aggregate["finite_repair_clock_certificate"] is False


def _write_observer_views(path: Path, paths: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for observer_id, values in enumerate(paths):
            steps = [
                {
                    "checkpoint_class": value,
                    "stable_flag": 1,
                    "s3_sector_class": 1,
                    "repair_load_bucket": 0,
                }
                for value in values
            ]
            handle.write(
                json.dumps(
                    {
                        "observer_id": observer_id,
                        "transition_history_mean_modal_mass": 1.0,
                        "transition_history_descriptor": {
                            "fields": [
                                "checkpoint_class",
                                "stable_flag",
                                "s3_sector_class",
                                "repair_load_bucket",
                            ],
                            "steps": steps,
                        },
                    }
                )
                + "\n"
            )
