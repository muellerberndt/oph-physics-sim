from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .line_fan import LineFanPeak
from .receipts import fail, pass_report


def _identifier(peak: LineFanPeak) -> tuple:
    return (
        round(float(peak.energy), 12),
        round(float(peak.intensity), 12),
        round(float(peak.gate_slope), 12),
        peak.polarization,
        peak.tau,
        round(float(peak.total_charge), 12),
        peak.eta,
    )


def identify_optical_sector(peaks: Iterable[LineFanPeak]) -> dict:
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for peak in peaks:
        buckets[_identifier(peak)].append(peak.label)
    ambiguous = {str(key): labels for key, labels in buckets.items() if len(labels) > 1}
    if ambiguous:
        return fail("OPTICAL_SECTOR_AMBIGUOUS", details={"ambiguous_identifiers": ambiguous})
    return pass_report(
        receipts={"OPTICAL_LINE_FAN_INJECTIVE": True},
        details={"identifiers": {label: key for key, labels in buckets.items() for label in labels}},
    )
