"""Exploratory spatial-dimensionality probe over the committed tower.

Non-evidential package.  It assembles symmetric weighted Laplacians on
multi-level unions of the committed geodesic icosahedral tower, with
inter-level coupling mirroring the committed join/refinement transport
(`oph_fpe/core/icosahedral.py` cell refinement maps; Lean anchors
`Lean/QFT/CarrierJoinTransport.lean` and `Lean/QFT/JoinNetMorphism.lean`),
and measures heat-kernel spectral dimension and Weyl-law dimension per
configuration.  The design, pins, and declared conventions are in
``DESIGN.md`` next to this file.  Every artifact is labeled
``exploratory_non_evidential``; the deliverable is a table of numbers with
no verdict attached.
"""

from oph_fpe.dimension import estimators, geometry, operators  # noqa: F401

EVIDENTIAL_STATUS = "exploratory_non_evidential"
