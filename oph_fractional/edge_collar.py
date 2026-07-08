from __future__ import annotations

from dataclasses import dataclass

from .receipts import pass_report


@dataclass(frozen=True)
class EdgeLedger:
    edge_spectrum_certified: bool
    bulk_edge_matching: bool
    edge_modes: tuple[str, ...] = ()


def bulk_edge_consistency(edge: EdgeLedger) -> dict:
    return pass_report(
        receipts={
            "EDGE_SPECTRUM": edge.edge_spectrum_certified,
            "BULK_EDGE_CONSISTENCY": edge.bulk_edge_matching,
        },
        details={"edge_modes": list(edge.edge_modes)},
    )
