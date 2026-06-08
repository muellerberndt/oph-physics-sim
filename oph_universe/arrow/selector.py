from __future__ import annotations

from dataclasses import dataclass

from oph_universe.arrow.checkpoints import Checkpoint
from oph_universe.arrow.continuation import ContinuationStep
from oph_universe.arrow.entropy import fake_deficit_bits
from oph_universe.arrow.provenance import ProvenanceDAG


@dataclass
class AncestryCandidate:
    candidate_id: str
    checkpoint: Checkpoint
    continuation_steps: list[ContinuationStep]
    provenance_dag: ProvenanceDAG
    i_rec_bits: float
    n_res_bits: float
    b_hid_max_bits: float
    i_pre_bits: float
    approx_bits: float
    phi_cont: float
    k_hidden: float
    k_impl: float
    r_unsupported: float
    is_faithful: bool = False

    @property
    def f_fake(self) -> float:
        return fake_deficit_bits(
            self.i_rec_bits,
            self.n_res_bits,
            self.b_hid_max_bits,
            self.i_pre_bits,
            self.approx_bits,
        )


@dataclass(frozen=True)
class ArrowSelectorWeights:
    lambda_fake: float = 100.0
    lambda_hidden: float = 1.0
    lambda_impl: float = 1.0
    lambda_unsupported: float = 10.0


def j_arrow(candidate: AncestryCandidate, w: ArrowSelectorWeights) -> float:
    return (
        candidate.phi_cont
        + w.lambda_fake * candidate.f_fake
        + w.lambda_hidden * candidate.k_hidden
        + w.lambda_impl * candidate.k_impl
        + w.lambda_unsupported * candidate.r_unsupported
    )


def select_normal_form_ancestry(
    candidates: list[AncestryCandidate],
    weights: ArrowSelectorWeights,
) -> AncestryCandidate:
    if not candidates:
        raise ValueError("at least one ancestry candidate is required")
    return min(candidates, key=lambda candidate: (j_arrow(candidate, weights), candidate.candidate_id))


def lane8_falsifier(candidate: AncestryCandidate, selected: AncestryCandidate) -> bool:
    return (
        selected.candidate_id == candidate.candidate_id
        and not candidate.is_faithful
        and candidate.f_fake <= 1e-12
        and candidate.k_hidden <= 1e-12
        and candidate.phi_cont <= 1e-12
    )

