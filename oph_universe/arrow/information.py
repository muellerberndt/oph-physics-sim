from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any


def binary_entropy(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def fano_payload_lower_bound(h_x_bits: float, alphabet_size: int, epsilon: float) -> float:
    eps = min(max(float(epsilon), 0.0), 1.0)
    alphabet = max(int(alphabet_size), 1)
    if alphabet <= 1:
        return max(0.0, float(h_x_bits))
    bound = float(h_x_bits) - binary_entropy(eps) - eps * math.log2(alphabet - 1)
    return _clamp_nonnegative(bound)


def entropy_discrete(x: Sequence[Any]) -> float:
    if not x:
        return 0.0
    counts = Counter(_key(item) for item in x)
    n = float(len(x))
    return -sum((count / n) * math.log2(count / n) for count in counts.values() if count)


def mutual_information_discrete(x: Sequence[Any], y: Sequence[Any]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not x:
        return 0.0
    hx = entropy_discrete(x)
    hy = entropy_discrete(y)
    hxy = entropy_discrete(list(zip(map(_key, x), map(_key, y))))
    return _clamp_nonnegative(hx + hy - hxy)


def conditional_mutual_information_discrete(x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> float:
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y, and z must have the same length")
    if not x:
        return 0.0
    hxz = entropy_discrete(list(zip(map(_key, x), map(_key, z))))
    hyz = entropy_discrete(list(zip(map(_key, y), map(_key, z))))
    hz = entropy_discrete(z)
    hxyz = entropy_discrete(list(zip(map(_key, x), map(_key, y), map(_key, z))))
    return _clamp_nonnegative(hxz + hyz - hz - hxyz)


def chain_rule_payload(source: Sequence[Any], record_fragments: Sequence[Sequence[Any]]) -> list[float]:
    if any(len(fragment) != len(source) for fragment in record_fragments):
        raise ValueError("all record fragments must match source length")
    increments: list[float] = []
    previous: list[tuple[Any, ...]] = [tuple() for _ in source]
    for fragment in record_fragments:
        inc = conditional_mutual_information_discrete(source, fragment, previous)
        increments.append(inc)
        previous = [(*prev, _key(value)) for prev, value in zip(previous, fragment)]
    return increments


def _key(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _key(item)) for key, item in value.items()))
    return value


def _clamp_nonnegative(value: float, tol: float = 1e-12) -> float:
    if value < 0.0 and abs(value) <= tol:
        return 0.0
    return max(0.0, float(value))

