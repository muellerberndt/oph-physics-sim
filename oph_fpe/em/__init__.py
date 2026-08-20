"""Electromagnetism handoff package (#733) on the committed base carrier.

Design-only, non-evidential.  Exact rational base-carrier incidence
(``base_carrier``), the exact Green solve with the three handoff
observables (``green``), the replay harness against the committed RER
receipt (``replay``), and the exact leapfrog lane for the declared PR-66
temporal update with fail-closed receipts (``temporal``).  Nothing in this
package arms an instrument, freezes a decision rule, or states a physical
claim; the step index of the temporal lane is a declared evolution
parameter, not physical time; the conserved staggered form of the temporal
lane is indefinite and is not an energy.
"""

from oph_fpe.em.base_carrier import (
    BaseCarrierError,
    CHORD_SEAMS,
    FACES,
    ORIENTED_FACES,
    PORTS,
    SEAM_LEFT,
    SEAM_RIGHT,
    SEAMS,
    TREE_SEAMS,
    base_carrier_receipt,
    boundary,
    coboundary,
    face_curvature,
    face_codifferential,
    face_incidence_matrix,
    fundamental_cycles,
    laplacian_matrix,
    local_maxwell_operator,
    tree_solution,
)
from oph_fpe.em.green import (
    GreenSolveError,
    chord_field_strength_components,
    coulomb_field,
    field_strength,
    green_matrix,
    green_potential,
    port_potential_difference,
    seam_flux,
    thomson_decomposition,
)
from oph_fpe.em.replay import default_receipt_path, replay_committed_receipt
from oph_fpe.em.temporal import (
    LeapfrogLane,
    TemporalReceiptError,
    constant_history,
    demo_bundle,
)

__all__ = [
    "BaseCarrierError",
    "CHORD_SEAMS",
    "FACES",
    "GreenSolveError",
    "LeapfrogLane",
    "ORIENTED_FACES",
    "PORTS",
    "SEAM_LEFT",
    "SEAM_RIGHT",
    "SEAMS",
    "TREE_SEAMS",
    "TemporalReceiptError",
    "base_carrier_receipt",
    "boundary",
    "chord_field_strength_components",
    "coboundary",
    "constant_history",
    "coulomb_field",
    "default_receipt_path",
    "demo_bundle",
    "face_codifferential",
    "face_curvature",
    "face_incidence_matrix",
    "field_strength",
    "fundamental_cycles",
    "green_matrix",
    "green_potential",
    "laplacian_matrix",
    "local_maxwell_operator",
    "port_potential_difference",
    "replay_committed_receipt",
    "seam_flux",
    "thomson_decomposition",
    "tree_solution",
]
