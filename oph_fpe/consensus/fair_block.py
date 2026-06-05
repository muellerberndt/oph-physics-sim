from __future__ import annotations

import math
from typing import Any

from oph_fpe.claims import CONTINUATION, with_claim_metadata


def fair_block_consensus_certificate(
    *,
    lambda_contraction: float,
    epsilon_noise: float,
    beta: float,
    lipschitz_L: float,
    block_count: int,
    active_fraction: float,
) -> dict[str, Any]:
    lam = float(lambda_contraction)
    eps = float(epsilon_noise)
    beta_value = float(beta)
    lipschitz = float(lipschitz_L)
    blocks = int(block_count)
    active = float(active_fraction)
    constants_finite = all(math.isfinite(value) for value in (lam, eps, beta_value, lipschitz, active))
    contraction_margin = 1.0 - lam
    receipt = constants_finite and blocks > 0 and 0.0 <= active <= 1.0 and 0.0 <= lam < 1.0 and eps >= 0.0
    report = {
        "mode": "conditional_noisy_fair_block_consensus_certificate",
        "FAIR_BLOCK_CONSENSUS_CERTIFICATE": receipt,
        "receipt": receipt,
        "lambda_contraction": lam,
        "epsilon_noise": eps,
        "beta": beta_value,
        "lipschitz_L": lipschitz,
        "block_count": blocks,
        "active_fraction": active,
        "contraction_margin": float(contraction_margin),
        "constants_finite": constants_finite,
        "claim_boundary": (
            "conditional finite fair-block certificate constants for a simulated update schedule; "
            "not a generic distributed-consensus speedup theorem"
        ),
    }
    return with_claim_metadata(report, claim_level=CONTINUATION, receipt="FAIR_BLOCK_CONSENSUS_CERTIFICATE")
