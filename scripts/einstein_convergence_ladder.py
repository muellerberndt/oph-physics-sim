"""Produce and verify the four-row Einstein-branch scaling instrument.

The frozen instrument contains three support-adjusted scaling rows and one
same-size support-width control.  In particular, the two 262,144-carrier
captures are distinct canonical rows.  Each row has its own stable identifier
and filename, so a producer replay cannot overwrite the sparse control with
the support-384 capture.

The canonical bundle contains no wall-clock measurements.  It consists of a
JSON summary and a deterministic ``npz.gz`` payload for each row, plus an
exact four-row manifest.  The validator checks the row/configuration bijection,
hashes, payload schema, and summary-to-array readbacks.  Byte identity means a
replay with the same source revision and recorded runtime versions; it is not
a cross-runtime numerical portability claim.

Usage:
    .venv/bin/python scripts/einstein_convergence_ladder.py [output_dir]
    .venv/bin/python scripts/einstein_convergence_ladder.py \
        --normalize-existing [output_dir]
    .venv/bin/python scripts/einstein_convergence_ladder.py \
        --validate-only [output_dir]

``--normalize-existing`` is a migration tool.  It canonicalizes the four
frozen payloads already present in ``output_dir``, removes legacy wall-clock
fields from their summaries, and rebuilds the manifest.  It does not claim to
rerun the physical source producer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import sys
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oph_fpe.bulk.event_manifold_producer import (
    _event_chart,
    _event_table,
    _fit_quadratic_form,
    _spectral_embedding,
)
from oph_fpe.bulk.modular_normalization_producer import (
    _axis_frame,
    _depth_generator,
    cap_interior_state,
    geometric_flow_rate,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.stress_coupling_producer import cap_stress_flux

PAIR_CAP = 300_000
SCHEMA = "oph.einstein-convergence-ladder.v2"
FIT_INERTIA_EIGENVALUE_THRESHOLD = 1.0e-12
FIT_ABSOLUTE_TOLERANCE = 0.0
ARRAY_NAMES = (
    "causal_pairs",
    "chart",
    "coefficients",
    "flux",
    "form_eigenvalues",
    "spacelike_pairs",
)

# Chain depth is held constant.  The final row changes only support width at
# fixed carrier and observer counts, so it is a same-size sensitivity control,
# not a fourth scale point.
RUNG_SPECS: tuple[dict[str, Any], ...] = (
    {
        "rung_id": "scale_16384_support_96",
        "stem": "rung_16384",
        "variant": "support_96_scale_row",
        "carrier_count": 16_384,
        "observer_count": 128,
        "observer_samples": 6,
        "observer_support_size": 96,
    },
    {
        "rung_id": "scale_65536_support_96",
        "stem": "rung_65536",
        "variant": "support_96_scale_row",
        "carrier_count": 65_536,
        "observer_count": 256,
        "observer_samples": 6,
        "observer_support_size": 96,
    },
    {
        "rung_id": "control_262144_support_96",
        "stem": "rung_262144",
        "variant": "same_size_sparse_support_control",
        "carrier_count": 262_144,
        "observer_count": 512,
        "observer_samples": 6,
        "observer_support_size": 96,
    },
    {
        "rung_id": "scale_262144_support_384",
        "stem": "rung_262144_dense",
        "variant": "support_adjusted_scale_row",
        "carrier_count": 262_144,
        "observer_count": 512,
        "observer_samples": 6,
        "observer_support_size": 384,
    },
)


class LadderValidationError(ValueError):
    """Raised when a ladder bundle fails its frozen contract."""


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _source_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "carrier_count": int(spec["carrier_count"]),
        "cycles": 16,
        "seed": 20260751,
        "observer_count": int(spec["observer_count"]),
        "observer_support_size": int(spec["observer_support_size"]),
        "observer_samples": int(spec["observer_samples"]),
        "observer_cross_reads": True,
        "snapshot_coverage": "spanning",
        "geometry_transport": "held_out_flow",
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "zlib": zlib.ZLIB_VERSION,
    }


def _validate_specs(specs: Sequence[Mapping[str, Any]]) -> None:
    if len(specs) != 4:
        raise LadderValidationError("ladder_requires_exactly_four_rows")
    for key in ("rung_id", "stem"):
        values = [str(spec.get(key, "")) for spec in specs]
        if any(not value for value in values) or len(set(values)) != 4:
            raise LadderValidationError(f"ladder_{key}_bijection_failed")
    top_rows = [spec for spec in specs if int(spec.get("carrier_count", -1)) == 262_144]
    if sorted(int(spec["observer_support_size"]) for spec in top_rows) != [
        96,
        384,
    ]:
        raise LadderValidationError(
            "ladder_requires_distinct_262144_support_96_and_384_rows"
        )


def _pair_classes_capped(table: dict[str, Any]) -> dict[str, Any]:
    causal: list[tuple[int, int]] = []
    spacelike: list[tuple[int, int]] = []
    count = table["count"]
    reachable = table["reachable"]
    for i in range(count):
        reach_i = reachable[i]
        for j in range(i + 1, count):
            if i in reachable[j] or j in reach_i:
                causal.append((i, j))
            else:
                spacelike.append((i, j))
    subsample: dict[str, Any] = {}
    for name, pairs in (("causal", causal), ("spacelike", spacelike)):
        stride = max(1, len(pairs) // PAIR_CAP)
        subsample[name] = pairs[::stride][:PAIR_CAP]
        subsample[f"{name}_total"] = len(pairs)
        subsample[f"{name}_stride"] = stride
    return subsample


def run_rung(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one declared ladder row without recording wall-clock diagnostics."""

    cfg = _source_config(spec)
    carriers = cfg["carrier_count"]
    capture = capture_physical_source(cfg)

    events = capture["postrun_capture"]["semantic_events"]
    key_to_event = {event["event_key"]: event for event in events}
    cross = sum(
        1
        for edge in capture["postrun_capture"]["raw_ancestry_relations"]
        if (key_to_event.get(edge["child_event_id"]) or {}).get("observer_token")
        != (key_to_event.get(edge["parent_event_id"]) or {}).get("observer_token")
    )

    table = _event_table(capture)
    pairs = _pair_classes_capped(table)
    touched = sorted(
        {carrier for footprint in table["footprints"] for carrier in footprint}
    )
    index_of = {carrier: index for index, carrier in enumerate(touched)}
    sub = np.zeros((len(touched), len(touched)))
    for row in capture["postrun_capture"]["raw_overlap_relations"]:
        left = index_of.get(int(row["left_carrier_id"].rsplit("-", 1)[1]))
        right = index_of.get(int(row["right_carrier_id"].rsplit("-", 1)[1]))
        if left is not None and right is not None:
            sub[left, right] = sub[right, left] = 1.0
    embedding_small = _spectral_embedding(sub)
    embedding = np.zeros((carriers, 3))
    for carrier, index in index_of.items():
        embedding[carrier] = embedding_small[index]
    chart = _event_chart(table, embedding)
    fit = _fit_quadratic_form(
        chart,
        {"causal": pairs["causal"], "spacelike": pairs["spacelike"]},
    )

    samples = np.stack(
        [
            np.asarray(row["full_port_state"])
            for snapshot in capture["source_artifacts"]["dynamics"][
                "record_state_snapshots"
            ]
            for row in snapshot["carrier_rows"]
        ]
    )
    geometry = geometric_flow_rate(capture)
    coefficients = []
    for axis in range(6):
        state = cap_interior_state(samples, _axis_frame(axis))
        if state["faithful"]:
            depth = _depth_generator(axis)
            coefficients.append(
                float(np.tensordot(state["modular_hamiltonian"], depth))
                / float(np.tensordot(depth, depth))
                / geometry["rate"]
            )
    flux = np.asarray([cap_stress_flux(capture, axis) for axis in range(6)])
    spread = float((flux.max() - flux.min()) / abs(float(np.median(flux))))

    summary = {
        "config": cfg,
        "capture_sha256": capture["capture_sha256"],
        "event_count": table["count"],
        "cross_observer_edges": cross,
        "causal_pairs_total": pairs["causal_total"],
        "spacelike_pairs_total": pairs["spacelike_total"],
        "pair_cap": PAIR_CAP,
        "causal_stride": pairs["causal_stride"],
        "spacelike_stride": pairs["spacelike_stride"],
        "held_out_inertia": fit.get("inertia"),
        "cone_margin": fit.get("cone_margin"),
        "eigenvalues": fit.get("eigenvalues"),
        "normalization_coefficients": [round(value, 6) for value in coefficients],
        "coupling_spread": round(spread, 6),
        "raw_moment_min_eig": float(
            np.linalg.eigvalsh(samples.T @ samples / len(samples)).min()
        ),
    }
    arrays = {
        "chart": chart,
        "causal_pairs": np.asarray(pairs["causal"], dtype=np.int32),
        "spacelike_pairs": np.asarray(pairs["spacelike"], dtype=np.int32),
        "flux": flux,
        "coefficients": np.asarray(coefficients),
        "form_eigenvalues": np.asarray(fit.get("eigenvalues", [])),
    }
    return summary, arrays


def _canonical_artifact_bytes(arrays: Mapping[str, Any]) -> bytes:
    """Serialize numeric arrays with fixed member order and timestamps."""

    if set(arrays) != set(ARRAY_NAMES):
        raise LadderValidationError("artifact_array_schema_mismatch")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in ARRAY_NAMES:
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise LadderValidationError(f"artifact_object_dtype_forbidden:{name}")
            npy_buffer = io.BytesIO()
            np.lib.format.write_array(
                npy_buffer,
                array,
                version=(2, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_buffer.getvalue())
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=compressed_buffer,
        mtime=0,
    ) as stream:
        stream.write(archive_buffer.getvalue())
    return compressed_buffer.getvalue()


def _load_artifact_bytes(
    payload: bytes,
    *,
    require_canonical_order: bool = True,
) -> dict[str, np.ndarray]:
    try:
        archive_bytes = gzip.decompress(payload)
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            names = archive.namelist()
            expected = [f"{name}.npy" for name in ARRAY_NAMES]
            if (
                set(names) != set(expected)
                or len(names) != len(set(names))
                or require_canonical_order
                and names != expected
            ):
                raise LadderValidationError("artifact_member_schema_mismatch")
            arrays = {
                name: np.load(
                    io.BytesIO(archive.read(f"{name}.npy")),
                    allow_pickle=False,
                )
                for name in ARRAY_NAMES
            }
    except LadderValidationError:
        raise
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise LadderValidationError(
            f"artifact_decode_failed:{type(exc).__name__}"
        ) from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise LadderValidationError("artifact_object_dtype_forbidden")
    return arrays


def _array_specs(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
        }
        for name in ARRAY_NAMES
    }


def _manifest_row(
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    summary_bytes: bytes,
    artifact_bytes: bytes,
) -> dict[str, Any]:
    stem = str(spec["stem"])
    return {
        "rung_id": str(spec["rung_id"]),
        "variant": str(spec["variant"]),
        "config": _source_config(spec),
        "summary": f"{stem}.json",
        "summary_sha256": _sha(summary_bytes),
        "artifact": f"{stem}.npz.gz",
        "artifact_sha256": _sha(artifact_bytes),
        "artifact_arrays": _array_specs(arrays),
        "held_out_inertia": summary["held_out_inertia"],
        "cone_margin": summary["cone_margin"],
        "coupling_spread": summary["coupling_spread"],
        "cross_observer_edges": summary["cross_observer_edges"],
    }


def _write_row(
    output_dir: Path,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    if "capture_seconds" in summary:
        raise LadderValidationError("canonical_summary_contains_wall_clock")
    summary_bytes = _canonical_json_bytes(summary)
    artifact_bytes = _canonical_artifact_bytes(arrays)
    stem = str(spec["stem"])
    (output_dir / f"{stem}.json").write_bytes(summary_bytes)
    (output_dir / f"{stem}.npz.gz").write_bytes(artifact_bytes)
    numeric_arrays = {name: np.asarray(arrays[name]) for name in ARRAY_NAMES}
    return _manifest_row(
        spec,
        summary,
        numeric_arrays,
        summary_bytes,
        artifact_bytes,
    )


def write_ladder(
    output_dir: Path,
    *,
    runner: Callable[
        [Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = run_rung,
    specs: Sequence[Mapping[str, Any]] = RUNG_SPECS,
) -> dict[str, Any]:
    """Run all four declared rows and write their canonical bundle."""

    _validate_specs(specs)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        print(f"=== {spec['rung_id']} ===", flush=True)
        summary, arrays = runner(spec)
        rows.append(_write_row(output_dir, spec, summary, arrays))
        print(
            f"{spec['rung_id']}: events={summary['event_count']} "
            f"cross={summary['cross_observer_edges']} "
            f"inertia={summary['held_out_inertia']} "
            f"margin={summary['cone_margin']} "
            f"spread={summary['coupling_spread']}",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA,
        "bundle_origin": {
            "mode": "full_physical_source_run",
            "source_replay_claimed": True,
            "runtime_versions": _runtime_versions(),
        },
        "fit_readback_contract": {
            "producer": ("oph_fpe.bulk.event_manifold_producer._fit_quadratic_form"),
            "inertia_eigenvalue_threshold": (FIT_INERTIA_EIGENVALUE_THRESHOLD),
            "eigenvalue_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "cone_margin_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "runtime_scope": (
                "exact equality; a byte-replay claim requires the recorded "
                "runtime versions"
            ),
        },
        "rungs": rows,
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    validate_ladder(output_dir, specs=specs)
    return manifest


def normalize_existing_ladder(
    output_dir: Path,
    *,
    specs: Sequence[Mapping[str, Any]] = RUNG_SPECS,
) -> dict[str, Any]:
    """Canonicalize the four frozen rows without claiming a source replay."""

    _validate_specs(specs)
    rows = []
    for spec in specs:
        stem = str(spec["stem"])
        summary_path = output_dir / f"{stem}.json"
        artifact_path = output_dir / f"{stem}.npz.gz"
        if not summary_path.is_file() or not artifact_path.is_file():
            raise LadderValidationError(f"missing_frozen_row:{stem}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("capture_seconds", None)
        arrays = _load_artifact_bytes(
            artifact_path.read_bytes(),
            require_canonical_order=False,
        )
        rows.append(_write_row(output_dir, spec, summary, arrays))
    manifest = {
        "schema": SCHEMA,
        "bundle_origin": {
            "mode": "deterministic_reserialization_of_frozen_arrays",
            "source_replay_claimed": False,
            "runtime_versions": _runtime_versions(),
        },
        "fit_readback_contract": {
            "producer": ("oph_fpe.bulk.event_manifold_producer._fit_quadratic_form"),
            "inertia_eigenvalue_threshold": (FIT_INERTIA_EIGENVALUE_THRESHOLD),
            "eigenvalue_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "cone_margin_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "runtime_scope": (
                "exact equality; a byte-replay claim requires the recorded "
                "runtime versions"
            ),
        },
        "rungs": rows,
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    validate_ladder(output_dir, specs=specs)
    return manifest


def _require(condition: bool, blocker: str) -> None:
    if not condition:
        raise LadderValidationError(blocker)


def _validate_summary_arrays(
    summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    _require("capture_seconds" not in summary, "summary_contains_wall_clock")
    event_count = int(summary["event_count"])
    _require(
        arrays["chart"].shape == (event_count, 4),
        "chart_shape_mismatch",
    )
    for label in ("causal", "spacelike"):
        pairs = arrays[f"{label}_pairs"]
        total = int(summary[f"{label}_pairs_total"])
        stride = int(summary[f"{label}_stride"])
        expected_stride = max(1, total // int(summary["pair_cap"]))
        expected_rows = min(
            int(summary["pair_cap"]),
            (total + expected_stride - 1) // expected_stride,
        )
        _require(pairs.dtype == np.dtype("int32"), f"{label}_pair_dtype_mismatch")
        _require(
            pairs.shape == (expected_rows, 2),
            f"{label}_pair_shape_mismatch",
        )
        _require(stride == expected_stride, f"{label}_stride_mismatch")
        if len(pairs):
            _require(
                bool(
                    np.all(pairs[:, 0] >= 0)
                    and np.all(pairs[:, 0] < pairs[:, 1])
                    and np.all(pairs[:, 1] < event_count)
                ),
                f"{label}_pair_index_invalid",
            )
    eigenvalues = arrays["form_eigenvalues"]
    _require(eigenvalues.shape == (4,), "form_eigenvalue_shape_mismatch")
    _require(
        np.array_equal(eigenvalues, np.asarray(summary["eigenvalues"])),
        "form_eigenvalue_summary_mismatch",
    )
    pairs = {
        label: [(int(left), int(right)) for left, right in arrays[f"{label}_pairs"]]
        for label in ("causal", "spacelike")
    }
    refit = _fit_quadratic_form(arrays["chart"], pairs)
    _require(bool(refit.get("fitted")), "frozen_pair_refit_failed")
    refit_eigenvalues = np.asarray(refit["eigenvalues"])
    _require(
        np.allclose(
            refit_eigenvalues,
            eigenvalues,
            rtol=0.0,
            atol=FIT_ABSOLUTE_TOLERANCE,
        ),
        "frozen_pair_refit_eigenvalue_mismatch",
    )
    _require(
        abs(float(refit["cone_margin"]) - float(summary["cone_margin"]))
        <= FIT_ABSOLUTE_TOLERANCE,
        "frozen_pair_refit_cone_margin_mismatch",
    )
    inertia = [
        int(np.count_nonzero(refit_eigenvalues > FIT_INERTIA_EIGENVALUE_THRESHOLD)),
        int(np.count_nonzero(refit_eigenvalues < -FIT_INERTIA_EIGENVALUE_THRESHOLD)),
    ]
    _require(
        inertia == list(refit["inertia"]),
        "producer_inertia_threshold_contract_mismatch",
    )
    _require(inertia == summary["held_out_inertia"], "inertia_readback_mismatch")
    _require(arrays["flux"].shape == (6,), "flux_shape_mismatch")
    _require(arrays["coefficients"].shape == (6,), "coefficient_shape_mismatch")
    coefficients = [round(float(value), 6) for value in arrays["coefficients"]]
    _require(
        coefficients == summary["normalization_coefficients"],
        "coefficient_summary_mismatch",
    )
    median_flux = float(np.median(arrays["flux"]))
    _require(median_flux != 0.0, "zero_median_flux")
    spread = round(
        float((arrays["flux"].max() - arrays["flux"].min()) / abs(median_flux)),
        6,
    )
    _require(spread == summary["coupling_spread"], "spread_readback_mismatch")


def validate_ladder(
    output_dir: Path,
    *,
    specs: Sequence[Mapping[str, Any]] = RUNG_SPECS,
) -> dict[str, Any]:
    """Fail closed unless ``output_dir`` is the exact declared four-row bundle."""

    _validate_specs(specs)
    manifest_path = output_dir / "manifest.json"
    _require(manifest_path.is_file(), "manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == SCHEMA, "manifest_schema_mismatch")
    origin = manifest.get("bundle_origin")
    _require(isinstance(origin, Mapping), "manifest_bundle_origin_invalid")
    _require(
        (
            origin.get("mode") == "full_physical_source_run"
            and origin.get("source_replay_claimed") is True
            or origin.get("mode") == "deterministic_reserialization_of_frozen_arrays"
            and origin.get("source_replay_claimed") is False
        ),
        "manifest_bundle_origin_invalid",
    )
    runtime_versions = origin.get("runtime_versions")
    _require(
        isinstance(runtime_versions, Mapping)
        and set(runtime_versions) == {"python", "numpy", "scipy", "zlib"}
        and all(
            isinstance(value, str) and value for value in runtime_versions.values()
        ),
        "manifest_runtime_versions_invalid",
    )
    _require(
        manifest.get("fit_readback_contract")
        == {
            "producer": ("oph_fpe.bulk.event_manifold_producer._fit_quadratic_form"),
            "inertia_eigenvalue_threshold": (FIT_INERTIA_EIGENVALUE_THRESHOLD),
            "eigenvalue_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "cone_margin_absolute_tolerance": FIT_ABSOLUTE_TOLERANCE,
            "runtime_scope": (
                "exact equality; a byte-replay claim requires the recorded "
                "runtime versions"
            ),
        },
        "manifest_fit_readback_contract_mismatch",
    )
    rows = manifest.get("rungs")
    _require(isinstance(rows, list) and len(rows) == 4, "manifest_row_count_mismatch")
    row_by_id = {row.get("rung_id"): row for row in rows if isinstance(row, Mapping)}
    expected_ids = {str(spec["rung_id"]) for spec in specs}
    _require(
        len(row_by_id) == 4 and set(row_by_id) == expected_ids,
        "manifest_rung_bijection_failed",
    )
    expected_files = {"manifest.json"}
    for spec in specs:
        rung_id = str(spec["rung_id"])
        stem = str(spec["stem"])
        row = row_by_id[rung_id]
        summary_name = f"{stem}.json"
        artifact_name = f"{stem}.npz.gz"
        expected_files.update((summary_name, artifact_name))
        _require(row.get("variant") == spec["variant"], f"{rung_id}:variant_mismatch")
        _require(
            row.get("config") == _source_config(spec), f"{rung_id}:config_mismatch"
        )
        _require(row.get("summary") == summary_name, f"{rung_id}:summary_path_mismatch")
        _require(
            row.get("artifact") == artifact_name, f"{rung_id}:artifact_path_mismatch"
        )
        summary_bytes = (output_dir / summary_name).read_bytes()
        artifact_bytes = (output_dir / artifact_name).read_bytes()
        _require(
            row.get("summary_sha256") == _sha(summary_bytes),
            f"{rung_id}:summary_hash_mismatch",
        )
        _require(
            row.get("artifact_sha256") == _sha(artifact_bytes),
            f"{rung_id}:artifact_hash_mismatch",
        )
        summary = json.loads(summary_bytes)
        arrays = _load_artifact_bytes(artifact_bytes)
        _require(
            summary.get("config") == _source_config(spec),
            f"{rung_id}:summary_config_mismatch",
        )
        _require(
            row.get("artifact_arrays") == _array_specs(arrays),
            f"{rung_id}:artifact_spec_mismatch",
        )
        for key in (
            "held_out_inertia",
            "cone_margin",
            "coupling_spread",
            "cross_observer_edges",
        ):
            _require(
                row.get(key) == summary.get(key),
                f"{rung_id}:{key}_summary_mismatch",
            )
        _validate_summary_arrays(summary, arrays)
    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and (
            path.name == "manifest.json"
            or path.name.startswith("rung_")
            and path.name.endswith((".json", ".npz.gz"))
        )
    }
    _require(actual_files == expected_files, "canonical_file_set_mismatch")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data/einstein_convergence",
        type=Path,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--normalize-existing",
        action="store_true",
        help="canonicalize four existing rows without rerunning the source",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen four-row bundle without rewriting it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate_only:
        validate_ladder(args.output_dir)
        print("LADDER_VALID", flush=True)
    elif args.normalize_existing:
        normalize_existing_ladder(args.output_dir)
        print("LADDER_NORMALIZED", flush=True)
    else:
        write_ladder(args.output_dir)
        print("LADDER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
