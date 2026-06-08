from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntropyState:
    t: int
    s_max_bits: float
    s_of_bits: float
    record_capacity_bits: float
    blank_negentropy_bits: float
    hidden_entropy_export_bits: float
    preexisting_provenance_bits: float
    approximation_error_bits: float = 0.0

    @property
    def n_of_bits(self) -> float:
        return float(self.s_max_bits) - float(self.s_of_bits)


def landauer_record_bound_ok(
    delta_record_payload_bits: float,
    blank_negentropy_bits: float,
    entropy_export_bits: float,
    preexisting_provenance_bits: float = 0.0,
    tol: float = 1e-9,
) -> bool:
    return (
        float(blank_negentropy_bits)
        + float(entropy_export_bits)
        + float(preexisting_provenance_bits)
        + float(tol)
        >= float(delta_record_payload_bits)
    )


def fake_deficit_bits(
    i_rec_bits: float,
    n_res_bits: float,
    b_hid_max_bits: float,
    i_pre_bits: float,
    approx_bits: float = 0.0,
) -> float:
    return max(
        0.0,
        float(i_rec_bits)
        - float(n_res_bits)
        - float(b_hid_max_bits)
        - float(i_pre_bits)
        - float(approx_bits),
    )


def selected_ancestry_entropy_bound(
    s_max_bits: float,
    i_rec_bits: float,
    b_hid_max_bits: float,
    i_pre_bits: float,
    approx_bits: float,
) -> float:
    return (
        float(s_max_bits)
        - float(i_rec_bits)
        + float(b_hid_max_bits)
        + float(i_pre_bits)
        + float(approx_bits)
    )

