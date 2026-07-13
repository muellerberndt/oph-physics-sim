from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oph_fpe.consensus.transactional_repair import canonical_state_hash
from oph_fpe.evidence.hashes import canonical_json_bytes, stable_json_hash


def test_stable_json_hash_is_mapping_order_independent_and_type_sensitive() -> None:
    left = {"b": np.int64(2), "a": np.float32(1.5)}
    right = {"a": 1.5, "b": 2}

    assert stable_json_hash(left) == stable_json_hash(right)
    assert stable_json_hash({"value": 1}) != stable_json_hash({"value": 1.0})


def test_stable_json_hash_normalizes_float_edge_cases() -> None:
    assert stable_json_hash(-0.0) == stable_json_hash(0.0)
    assert stable_json_hash(float("nan")) == stable_json_hash(np.float64("nan"))
    assert stable_json_hash(float("inf")) != stable_json_hash(float("-inf"))


def test_stable_json_hash_retains_numpy_shape_and_dtype() -> None:
    row = np.asarray([1, 2, 3], dtype=np.int16)
    column = row.reshape(3, 1)

    assert stable_json_hash(row) != stable_json_hash(column)
    assert stable_json_hash(row) != stable_json_hash(row.astype(np.int64))


def test_stable_json_hash_supports_complex_arrays_without_repr_fallback() -> None:
    matrix = np.asarray([[1.0 + 2.0j, -0.0 - 3.0j]], dtype=np.complex128)

    assert stable_json_hash(matrix) == stable_json_hash(matrix.copy())
    assert stable_json_hash(matrix) != stable_json_hash(np.conjugate(matrix))


def test_canonical_encoder_supports_paths_bytes_sets_and_tuples() -> None:
    payload = {
        "path": Path("runs/example"),
        "bytes": b"\x00\xff",
        "set": {3, 1, 2},
        "tuple": ("x", 1),
    }

    assert canonical_json_bytes(payload) == canonical_json_bytes(payload)


def test_canonical_encoder_rejects_display_repr_fallbacks() -> None:
    class DisplayOnly:
        def __str__(self) -> str:
            return "unstable"

    with pytest.raises(TypeError, match="unsupported value"):
        stable_json_hash(DisplayOnly())


def test_transaction_state_hash_uses_canonical_typed_serialization() -> None:
    assert canonical_state_hash({"x": np.float64(0.25), "y": np.int32(2)}) == canonical_state_hash(
        {"y": 2, "x": 0.25}
    )
