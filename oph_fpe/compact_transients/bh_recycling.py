from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_GENERATION_PRIOR_INPUTS = (
    "ringdown_residual",
    "ringdown_residuals",
    "postfit_repair_tail_amplitude",
    "post_fit_repair_tail_amplitude",
    "echo_score",
    "echo_scores",
    "repair_tail_score",
)


@dataclass
class GenealogyDAG:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add_seed(self, node_id: str, *, generation: int = 1, **attrs: Any) -> None:
        self.nodes[node_id] = {"generation": int(generation), **attrs}

    def recycle(self, parent_a: str, parent_b: str, remnant: str, gw_packet_id: str, **attrs: Any) -> dict[str, Any]:
        gen_a = int(self.nodes.get(parent_a, {}).get("generation", 1))
        gen_b = int(self.nodes.get(parent_b, {}).get("generation", 1))
        generation = 1 + max(gen_a, gen_b)
        self.nodes[remnant] = {
            "generation": generation,
            "parents": [parent_a, parent_b],
            "gw_packet_id": gw_packet_id,
            **attrs,
        }
        self.edges.append((parent_a, remnant, gw_packet_id))
        self.edges.append((parent_b, remnant, gw_packet_id))
        return {
            "BH_GENEALOGY_DAG_RECEIPT": True,
            "remnant": remnant,
            "generation": generation,
            "edge_count": len(self.edges),
        }

    def visible_dag(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {"parent": parent, "child": child, "packet": packet}
                for parent, child, packet in self.edges
            ],
        }


def generation_prior_leakage_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    haystack = " ".join(str(key).lower() for key in inputs)
    hits = sorted(token for token in FORBIDDEN_GENERATION_PRIOR_INPUTS if token in haystack)
    return {
        "NO_GENERATION_LEAKAGE_RECEIPT": not hits,
        "forbidden_hits": hits,
        "allowed_inputs": [
            "M1",
            "M2",
            "chi1",
            "chi2",
            "mass_ratio_q",
            "host_environment",
            "formation_channel_prior",
        ],
        "status": "pass" if not hits else "generation_prior_reads_ringdown_target",
    }


def generation_prior_score(inputs: dict[str, Any]) -> dict[str, Any]:
    audit = generation_prior_leakage_audit(inputs)
    if not audit["NO_GENERATION_LEAKAGE_RECEIPT"]:
        return {**audit, "p_generation_ge_2": None}
    m1 = float(inputs.get("M1", inputs.get("m1", 0.0)))
    m2 = float(inputs.get("M2", inputs.get("m2", 0.0)))
    chi1 = abs(float(inputs.get("chi1", 0.0)))
    chi2 = abs(float(inputs.get("chi2", 0.0)))
    q = float(inputs.get("mass_ratio_q", inputs.get("q", 1.0)))
    env = float(inputs.get("host_environment", inputs.get("h_env", 0.0)))
    raw = 0.03 * max(0.0, m1 + m2 - 80.0) + 0.8 * max(0.0, chi1 + chi2 - 1.0) + 0.2 * abs(q - 1.0) + env
    p = 1.0 / (1.0 + math.exp(-raw))
    return {**audit, "p_generation_ge_2": p}


def linear_repair_tail_template(
    times: list[float],
    *,
    generation_probability: float,
    recycled_mismatch: float,
    gamma_rep: float,
    omega_rep: float,
    phase: float = 0.0,
    t0: float = 0.0,
) -> dict[str, Any]:
    amplitude = float(generation_probability) * float(recycled_mismatch)
    values = [
        amplitude
        * math.exp(-float(gamma_rep) * (float(t) - float(t0)))
        * math.cos(float(omega_rep) * (float(t) - float(t0)) + float(phase))
        for t in times
    ]
    return {
        "LINEAR_REPAIR_TAIL_RECEIPT": True,
        "generation_probability": float(generation_probability),
        "recycled_mismatch": float(recycled_mismatch),
        "amplitude": amplitude,
        "gamma_rep": float(gamma_rep),
        "omega_rep": float(omega_rep),
        "times": [float(t) for t in times],
        "template": values,
        "claim_boundary": "Frozen damped-sinusoid repair-tail template, not arbitrary echo hunting.",
    }


def bh_recycling_control_family() -> dict[str, Any]:
    return {
        "M0": "kerr_only",
        "M1": "kerr_plus_unconstrained_damped_residual",
        "M2": "kerr_plus_generation_correlated_oph_repair_tail",
        "forbidden_path": "ringdown_residual -> generation_label -> claim_success",
    }
