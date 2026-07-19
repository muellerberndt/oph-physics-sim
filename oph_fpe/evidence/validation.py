"""Small fail-closed validation helpers shared by evidence contracts."""

from __future__ import annotations


def utf8_byte_length(value: str) -> int | None:
    """Return strict UTF-8 byte length, or ``None`` for invalid Unicode text."""

    if not isinstance(value, str):
        return None
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None
