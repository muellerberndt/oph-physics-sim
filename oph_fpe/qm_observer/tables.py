"""Declared integer branch tables of the observer-frame probe.

Exploratory, non-evidential. This module carries declared literal data
only: class labels, integer branch tables, the committed context list, the
conditioned-class map, and the base population. Every entry is a
transcription of an exact conditional weight Tr(p F) of the committed
effect matrices, cited in ``DESIGN.md`` section 3.3; the transcription is
verified against the trace formula in ``receipt.py``, never here. This
module imports nothing from the counting or amplitude modules and nothing
from ``oph_fpe.quantum``.
"""

from __future__ import annotations

# The committed run population (reference receipt inputs run_counts,
# run_mass): 111 micro-configurations in class rec0, 68 in class rec1.
BASE_POPULATION: tuple[tuple[str, int], ...] = (
    ("rec0", 111),
    ("rec1", 68),
)

# The eight committed contexts in reference-receipt order, each with the
# key of its exact effect matrix.
CONTEXTS: tuple[tuple[str, str], ...] = (
    ("web_diagonal", "P_rec"),
    ("web_conjugated_0", "P_rec"),
    ("web_conjugated_1", "P_rec"),
    ("web_conjugated_2", "E_A"),
    ("web_conjugated_3", "E_A"),
    ("web_conjugated_4", "E_B"),
    ("web_conjugated_5", "E_B"),
    ("phase", "Y_plus"),
)

CONTEXT_EFFECT: dict[str, str] = dict(CONTEXTS)

# The conditioned record class of (effect key, outcome): the class whose
# projector is the outcome effect (the effect for outcome 0, its
# complement for outcome 1).
CONDITIONED_CLASS: dict[tuple[str, int], str] = {
    ("P_rec", 0): "rec0",
    ("P_rec", 1): "rec1",
    ("E_A", 0): "A0",
    ("E_A", 1): "A1",
    ("E_B", 0): "B0",
    ("E_B", 1): "B1",
    ("Y_plus", 0): "Y0",
    ("Y_plus", 1): "Y1",
}

# Branch tables: for each record class, the exact outcome-0 fraction of
# each effect, as an integer pair (numerator, denominator). Each entry is
# the transcription of Tr(p F) for the class projector p and the effect F
# (DESIGN.md section 3.3).
BRANCH_TABLES: dict[str, dict[str, tuple[int, int]]] = {
    "rec0": {"P_rec": (1, 1), "E_A": (1, 4), "E_B": (1, 4), "Y_plus": (1, 2)},
    "rec1": {"P_rec": (0, 1), "E_A": (3, 4), "E_B": (3, 4), "Y_plus": (1, 2)},
    "A0": {"P_rec": (1, 4), "E_A": (1, 1), "E_B": (1, 4), "Y_plus": (1, 2)},
    "A1": {"P_rec": (3, 4), "E_A": (0, 1), "E_B": (3, 4), "Y_plus": (1, 2)},
    "B0": {"P_rec": (1, 4), "E_A": (1, 4), "E_B": (1, 1), "Y_plus": (1, 2)},
    "B1": {"P_rec": (3, 4), "E_A": (3, 4), "E_B": (0, 1), "Y_plus": (1, 2)},
    "Y0": {"P_rec": (1, 2), "E_A": (1, 2), "E_B": (1, 2), "Y_plus": (1, 1)},
    "Y1": {"P_rec": (1, 2), "E_A": (1, 2), "E_B": (1, 2), "Y_plus": (0, 1)},
}
