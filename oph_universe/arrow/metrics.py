from __future__ import annotations

import math
from dataclasses import dataclass

from oph_universe.arrow.entropy import EntropyState, fake_deficit_bits


@dataclass(frozen=True)
class ArrowMetricRow:
    t: int
    phi: float
    s_of_bits: float
    s_max_bits: float
    n_of_bits: float
    record_count: int
    record_capacity_bits: float
    payload_bits: float
    blank_negentropy_bits: float
    hidden_export_bits: float
    preexisting_provenance_bits: float
    fake_deficit_bits: float
    j_arrow: float
    branch_orientation: str


def metric_row(
    *,
    t: int,
    phi: float,
    entropy: EntropyState,
    record_count: int,
    payload_bits: float,
    j_arrow_value: float,
    branch_orientation: str,
) -> ArrowMetricRow:
    return ArrowMetricRow(
        t=int(t),
        phi=float(phi),
        s_of_bits=float(entropy.s_of_bits),
        s_max_bits=float(entropy.s_max_bits),
        n_of_bits=float(entropy.n_of_bits),
        record_count=int(record_count),
        record_capacity_bits=float(entropy.record_capacity_bits),
        payload_bits=float(payload_bits),
        blank_negentropy_bits=float(entropy.blank_negentropy_bits),
        hidden_export_bits=float(entropy.hidden_entropy_export_bits),
        preexisting_provenance_bits=float(entropy.preexisting_provenance_bits),
        fake_deficit_bits=fake_deficit_bits(
            payload_bits,
            entropy.n_of_bits,
            entropy.hidden_entropy_export_bits,
            entropy.preexisting_provenance_bits,
            entropy.approximation_error_bits,
        ),
        j_arrow=float(j_arrow_value),
        branch_orientation=str(branch_orientation),
    )


def fake_probability_bound(fake_deficit: float) -> float:
    return 2.0 ** (-max(0.0, float(fake_deficit)))

