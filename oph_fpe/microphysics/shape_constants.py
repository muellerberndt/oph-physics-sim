from __future__ import annotations

import math


ALPHA_EM = 0.0072973525643
PHI = (1.0 + math.sqrt(5.0)) / 2.0
DELTA_P = ALPHA_EM * math.sqrt(math.pi)
P_SHAPE = PHI + DELTA_P


def loop_detuning_phase(alpha: float = ALPHA_EM) -> float:
    delta = float(alpha) * math.sqrt(math.pi)
    return 2.0 * math.pi * delta / PHI


def pentagon_kL_detuned(alpha: float = ALPHA_EM) -> float:
    return (2.0 * math.pi + loop_detuning_phase(alpha)) / 5.0
