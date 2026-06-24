from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.collar_state import (
    cap_collar_partition,
    classical_diagonal_cmi_nats,
    collar_triplet_packets,
    empirical_packet_distribution,
    fawzi_renner_bound,
    sector_conditioned_cmi,
    visible_packet_encoding_report,
    visible_packets,
)


def collar_markov_report(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    *,
    packet_bins: dict[str, int] | None = None,
    max_triplets: int = 4096,
    seed: int = 1,
) -> dict[str, Any]:
    packets = visible_packets(raw_fields, packet_bins)
    packet_encoding = visible_packet_encoding_report(raw_fields, packet_bins)
    sector_packets = _sector_packets(raw_fields, packets)
    rows = []
    for cap_id, cap in enumerate(caps):
        partition = cap_collar_partition(points, cap, cap.collar_width)
        a, b, d, collar_nodes = collar_triplet_packets(
            points,
            packets,
            partition,
            max_triplets=max_triplets,
            seed=seed + cap_id,
        )
        if a.size:
            epsilon_cmi = classical_diagonal_cmi_nats(a, b, d)
            sector_cmi = sector_conditioned_cmi(a, b, d, sector_packets[collar_nodes])
        else:
            epsilon_cmi = 0.0
            sector_cmi = {}
        alphabet = empirical_packet_distribution(packets, partition.cap_weights > 0.5)
        rows.append(
            {
                "cap_id": cap_id,
                "theta0": float(cap.theta0),
                "collar_width": float(partition.collar_width),
                "inside_count": int(np.sum(partition.inside_mask)),
                "collar_count": int(np.sum(partition.collar_mask)),
                "outside_count": int(np.sum(partition.outside_mask)),
                "epsilon_cmi": float(epsilon_cmi),
                "classical_diagonal_cmi_nats": float(epsilon_cmi),
                "state_semantics": "classical_commuting",
                "log_unit": "nat",
                "sector_conditioned_cmi": sector_cmi,
                "r_fr_bound": fawzi_renner_bound(epsilon_cmi),
                "sample_count": int(points.shape[0]),
                "packet_alphabet_size": int(len(alphabet)),
                "triplet_count": int(a.size),
                "claim_boundary": (
                    "classical diagonal collar-Markov diagnostic in nats; not a noncommutative "
                    "collar CMI, not a modular source charge, and not a finite Einstein-source proof"
                ),
            }
        )
    eps = [row["epsilon_cmi"] for row in rows]
    return {
        "mode": "diagonal_empirical_collar_state",
        "state_semantics": "classical_commuting",
        "log_unit": "nat",
        "packet_encoding": packet_encoding,
        "SOURCE_LOCALIZATION_SATURATION_RECEIPT": False,
        "MODULAR_SOURCE_CHARGE_RECEIPT": False,
        "claim_boundary": (
            "classical diagonal collar-Markov diagnostic in nats; not final noncommutative BW proof, "
            "not a modular source charge, and not a physical anomaly-density source"
        ),
        "cap_count": len(rows),
        "median_epsilon_cmi": float(np.median(eps)) if eps else 0.0,
        "mean_epsilon_cmi": float(np.mean(eps)) if eps else 0.0,
        "p90_epsilon_cmi": float(np.percentile(eps, 90)) if eps else 0.0,
        "rows": rows,
    }


def _sector_packets(raw_fields: dict[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    sectors = raw_fields.get("s3_sector_class")
    if sectors is not None:
        return np.asarray(sectors, dtype=np.int64)
    density = raw_fields.get("s3_class_density")
    if density is None:
        return np.zeros_like(fallback, dtype=np.int64)
    values = np.asarray(density, dtype=float)
    return np.clip(np.rint(values * 2.0), 0, 2).astype(np.int64)
