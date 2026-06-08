SPARC measurement tables used by the OPH-FPE static galaxy lane.

Downloaded from the official SPARC site:

- https://astroweb.case.edu/SPARC/RAR.mrt
- https://astroweb.case.edu/SPARC/BTFR_Lelli2019.mrt
- https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt

The host certificate was expired during the 2026-06-06 download, so the local
fetch used `curl -k`.  The files are plain MRT tables published by the SPARC
team.  `oph_fpe.cosmology.galaxy_static` parses these tables into the external
static RAR/BTFR measurement report.

Claim boundary: this dataset supports a relaxed-galaxy measurement comparison.
It is not a CMB prediction, not a populated 3D bulk proof, and not a dynamic
cluster/cosmology claim.
