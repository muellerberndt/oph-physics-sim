"""Deterministic companion paths for standalone evidence artifacts."""

from __future__ import annotations

from pathlib import Path


def companion_input_packet_path(
    destination: Path,
    *,
    canonical_certificate_filename: str,
    canonical_input_filename: str,
) -> Path:
    """Return a collision-free replay-packet path for ``destination``.

    Directory-mode writers retain their documented canonical pair.  An
    explicitly named certificate gets a companion derived from its own stem,
    so two certificates in one directory cannot silently overwrite each
    other's replay input.
    """

    destination = Path(destination)
    if destination.name == canonical_certificate_filename:
        return destination.parent / canonical_input_filename
    return destination.with_name(f"{destination.stem}_input_packet.json")
