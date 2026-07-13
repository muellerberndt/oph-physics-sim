from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.defects.array_s3_holonomy import (
    S3_CLASS,
    S3_INV,
    S3_MUL,
    defect_interaction_report,
    particle_likeness_report,
)


def controlled_s3_particle_assay_report(
    *,
    patch_count: int = 65_536,
    observation_count: int = 5,
    support_node_count: int = 8,
    holonomy: int = 1,
    cycle_stride: int = 8,
    max_support_fraction: float = 0.05,
) -> dict[str, Any]:
    """Validate the finite S3 particle gates on a planted inverse-defect pair.

    The ordinary array runs should not claim matter particles unless the
    dynamics actually produces transport, inverse fusion candidates, scattering
    repeatability, and bulk localization. This assay is deliberately a planted
    detector check: it proves the gate can turn positive on the required S3
    structure, not that a production run generated particles spontaneously.
    """

    patch_count = max(1, int(patch_count))
    observation_count = max(3, int(observation_count))
    support_node_count = max(1, int(support_node_count))
    holonomy = int(holonomy) % int(len(S3_INV))
    if holonomy == 0:
        holonomy = 1
    inverse = int(S3_INV[holonomy])
    identity_product = bool(int(S3_MUL[holonomy, inverse]) == 0 or int(S3_MUL[inverse, holonomy]) == 0)
    timeline = _controlled_timeline(
        patch_count=patch_count,
        observation_count=observation_count,
        support_node_count=support_node_count,
        holonomy=holonomy,
        inverse=inverse,
        cycle_stride=int(cycle_stride),
    )
    interaction = defect_interaction_report(
        timeline,
        min_observations=3,
        min_class_stability=0.8,
        min_transport_distance=1.0e-9,
        min_scattering_transitions=2,
    )
    particle = particle_likeness_report(
        timeline,
        interaction,
        bulk_localization_pass=True,
        max_support_fraction=float(max_support_fraction),
        min_observations=3,
        min_class_stability=0.8,
    )
    return {
        "mode": "controlled_s3_inverse_defect_particle_gate_assay",
        "controlled_planted_assay": True,
        "patch_count": patch_count,
        "observation_count": observation_count,
        "support_node_count": support_node_count,
        "holonomy": holonomy,
        "inverse_holonomy": inverse,
        "s3_inverse_identity_pass": identity_product,
        "interaction_proxy_receipt": bool(interaction.get("interaction_proxy_receipt", False)),
        "fusion_conservation_proxy_pass": bool(interaction.get("fusion_conservation_proxy_pass", False)),
        "particle_detector_positive_receipt": bool(
            particle.get("particle_detector_positive_receipt", False)
        ),
        "particle_like_count": int(particle.get("particle_like_count", 0)),
        "physical_particle_emergence": False,
        "timeline_report": timeline,
        "interaction_report": interaction,
        "particle_likeness_report": particle,
        "claim_boundary": (
            "Controlled/planted S3 inverse-defect assay. A positive receipt validates the detector "
            "and gate logic for transport, inverse fusion, scattering repeatability, and declared "
            "bulk localization. It is not evidence that a production OPH-FPE run spontaneously "
            "generated matter particles."
        ),
    }


def write_controlled_s3_particle_assay_report(
    out_path: Path,
    *,
    patch_count: int = 65_536,
    observation_count: int = 5,
    support_node_count: int = 8,
    holonomy: int = 1,
    cycle_stride: int = 8,
    max_support_fraction: float = 0.05,
) -> dict[str, Any]:
    report = controlled_s3_particle_assay_report(
        patch_count=patch_count,
        observation_count=observation_count,
        support_node_count=support_node_count,
        holonomy=holonomy,
        cycle_stride=cycle_stride,
        max_support_fraction=max_support_fraction,
    )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _controlled_timeline(
    *,
    patch_count: int,
    observation_count: int,
    support_node_count: int,
    holonomy: int,
    inverse: int,
    cycle_stride: int,
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    worldline_events: dict[str, list[dict[str, Any]]] = {"worldline_000000": [], "worldline_000001": []}
    for index in range(observation_count):
        cycle = int(index * max(1, cycle_stride))
        left_centroid = [float(-0.35 + 0.09 * index), float(0.12), float(0.93)]
        right_centroid = [float(0.35 - 0.09 * index), float(-0.12), float(0.93)]
        left_cluster = _cluster(
            cluster_id=f"controlled_left_{index:03d}",
            worldline_id="worldline_000000",
            holonomy=holonomy,
            inverse_holonomy=inverse,
            support_node_count=support_node_count,
            centroid=left_centroid,
            support_start=10_000 + 10 * index,
            patch_count=patch_count,
        )
        right_cluster = _cluster(
            cluster_id=f"controlled_right_{index:03d}",
            worldline_id="worldline_000001",
            holonomy=inverse,
            inverse_holonomy=holonomy,
            support_node_count=support_node_count,
            centroid=right_centroid,
            support_start=20_000 + 10 * index,
            patch_count=patch_count,
        )
        snapshots.append(
            {
                "cycle": cycle,
                "triangle_count": 2,
                "defect_triangle_count": 2,
                "cluster_count": 2,
                "clusters": [left_cluster, right_cluster],
            }
        )
        for cluster in (left_cluster, right_cluster):
            worldline_events[str(cluster["worldline_id"])].append(
                {
                    "cycle": cycle,
                    "cluster_id": cluster["cluster_id"],
                    "event": "birth" if index == 0 else "continue",
                    "class": cluster["class"],
                    "holonomy_mode": cluster["holonomy_mode"],
                    "inverse_holonomy_mode": cluster["inverse_holonomy_mode"],
                    "support_node_count": support_node_count,
                    "support_nodes": cluster["support_nodes"],
                    "support_size": support_node_count,
                    "centroid": cluster["centroid"],
                    "transport_distance": None if index == 0 else 0.18,
                }
            )
    worldlines = []
    for worldline_id, events in worldline_events.items():
        cycles = [int(event["cycle"]) for event in events]
        worldlines.append(
            {
                "worldline_id": worldline_id,
                "birth_cycle": min(cycles),
                "death_cycle": max(cycles),
                "lifetime_cycles": max(cycles) - min(cycles),
                "observation_count": len(events),
                "mean_transport_distance": 0.18,
                "max_support_node_count": support_node_count,
                "persistent": True,
                "events": events,
            }
        )
    return {
        "schema": "oph_controlled_s3_defect_timeline_v1",
        "mode": "controlled_s3_defect_timeline",
        "patch_count": patch_count,
        "snapshot_count": len(snapshots),
        "max_triangles": 2,
        "snapshots": snapshots,
        "worldlines": worldlines,
        "worldline_count": len(worldlines),
        "persistent_worldline_count": len(worldlines),
        "max_observation_count": observation_count,
        "max_lifetime_cycles": (observation_count - 1) * max(1, cycle_stride),
        "persistent_worldline_precursor_receipt": True,
        "particle_promotion_inputs_complete": True,
        "truncation_reasons": [],
        "particle_matter_receipt": False,
        "claim_boundary": "controlled planted inverse S3 defect-pair timeline for detector validation",
    }


def _cluster(
    *,
    cluster_id: str,
    worldline_id: str,
    holonomy: int,
    inverse_holonomy: int,
    support_node_count: int,
    centroid: list[float],
    support_start: int,
    patch_count: int,
) -> dict[str, Any]:
    class_id = int(S3_CLASS[int(holonomy)])
    return {
        "cluster_id": cluster_id,
        "worldline_id": worldline_id,
        "class": {1: "transposition", 2: "threecycle"}.get(class_id, "identity"),
        "holonomy_mode": int(holonomy),
        "inverse_holonomy_mode": int(inverse_holonomy),
        "support_node_count": int(support_node_count),
        "support_nodes": [int((support_start + offset) % max(1, patch_count)) for offset in range(support_node_count)],
        "centroid": centroid,
    }
