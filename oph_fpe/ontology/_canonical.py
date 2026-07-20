"""Canonical immutable values and content-addressed serialization.

The ontology deliberately accepts a much smaller value language than Python.
That keeps hashes independent of object identity, insertion order, locale, and
mutable containers.  Floats and bytes receive explicit tagged encodings so
that their canonical representation cannot collide with ordinary JSON data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import base64
import hashlib
import json
import math
import re
from typing import Any


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESERVED_KEY = "$oph_type"


class CanonicalValueError(ValueError):
    """A value cannot be represented by the canonical ontology language."""


@dataclass(frozen=True, slots=True)
class FrozenMap(Mapping[str, Any]):
    """A deeply immutable, lexicographically ordered string-key mapping."""

    items_tuple: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        keys: list[str] = []
        normalized: list[tuple[str, Any]] = []
        for key, value in self.items_tuple:
            if not isinstance(key, str) or not key:
                raise CanonicalValueError(
                    "canonical mapping keys must be nonempty strings"
                )
            if key == _RESERVED_KEY:
                raise CanonicalValueError(
                    f"{_RESERVED_KEY!r} is a reserved canonical key"
                )
            keys.append(key)
            normalized.append((key, freeze_value(value)))
        if len(set(keys)) != len(keys):
            raise CanonicalValueError("canonical mapping keys must be unique")
        normalized.sort(key=lambda item: item[0])
        object.__setattr__(self, "items_tuple", tuple(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | FrozenMap) -> FrozenMap:
        if isinstance(value, FrozenMap):
            return value
        if not isinstance(value, Mapping):
            raise CanonicalValueError("expected a mapping")
        return cls(tuple(value.items()))

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self.items_tuple:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.items_tuple)

    def __len__(self) -> int:
        return len(self.items_tuple)

    def to_jsonable(self) -> dict[str, Any]:
        return {key: thaw_json(value) for key, value in self.items_tuple}


def freeze_value(value: Any) -> Any:
    """Deep-freeze one canonical scalar, sequence, or mapping.

    Sets, arbitrary objects, non-finite floats, and non-string mapping keys are
    rejected instead of being serialized through unstable ``repr`` output.
    """

    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalValueError("canonical floats must be finite")
        return value
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap.from_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_value(item) for item in value)
    raise CanonicalValueError(
        f"unsupported canonical value type: {type(value).__module__}.{type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """Convert a frozen value into its unambiguous JSON representation."""

    if isinstance(value, FrozenMap):
        return value.to_jsonable()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, bytes):
        return {
            _RESERVED_KEY: "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, float):
        return {_RESERVED_KEY: "float_hex", "value": value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise CanonicalValueError(f"value was not frozen: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    frozen = freeze_value(value)
    return json.dumps(
        thaw_json(frozen),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("ascii")


def canonical_hash(value: Any, *, domain: str) -> str:
    if not domain:
        raise ValueError("hash domain must be nonempty")
    material = FrozenMap.from_mapping({"domain": domain, "value": value})
    return "sha256:" + hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a sha256:<64 lowercase hex> digest")
    return value


def require_nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value
