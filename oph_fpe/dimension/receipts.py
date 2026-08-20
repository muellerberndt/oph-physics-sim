"""Canonical receipt serialization: sorted keys, rounded floats, SHA-256.

Exploratory, non-evidential.  The serialization convention is DESIGN.md
section 7: every float is rounded to 10 significant digits before
serialization, ``NaN`` becomes ``null``, keys are sorted, separators are
compact, and the document ends with one newline.  The SHA-256 of the exact
bytes is the receipt identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

FLOAT_SIGNIFICANT_DIGITS = 10


def canonical_value(value):
    """Recursively normalize a value for canonical serialization."""

    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [canonical_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return float(f"{number:.{FLOAT_SIGNIFICANT_DIGITS}g}")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical_bytes(document: dict) -> bytes:
    """The canonical byte serialization of a receipt document."""

    normalized = canonical_value(document)
    text = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def receipt_sha256(document: dict) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def write_receipt(document: dict, out_path: str | Path) -> tuple[Path, str]:
    """Write the canonical receipt and its SHA-256 sidecar; return both."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    path.write_bytes(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return path, digest


def write_timings(timings: dict, receipt_path: str | Path) -> Path:
    """Write the unhashed timing sidecar next to the receipt.

    Wall-clock values vary between runs; they live outside the hashed
    receipt so the receipt bytes stay reproducible (DESIGN.md section 7,
    declared convention C9).
    """

    path = Path(receipt_path).with_name("dimension_probe_timings.json")
    text = json.dumps(canonical_value(timings), sort_keys=True, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path
