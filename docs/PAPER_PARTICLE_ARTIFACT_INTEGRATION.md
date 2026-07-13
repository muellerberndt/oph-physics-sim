# Paper and Particle Artifact Integration

`oph-physics-sim` consumes selected outputs from the sibling
`reverse-engineering-reality` repository through a hash-pinned diagnostic bridge:

```bash
python3 tools/import_oph_artifacts.py RUN_OR_STAGING_DIR
```

The importer records the paper release, research commit, dirty-worktree digest,
tracked/untracked state, artifact schema/identity, byte count, and SHA-256. It
writes:

- `oph_cross_repo_artifact_manifest.json`;
- `particle_frontier_report.json`;
- `paper_geometry_regression_report.json`;
- the pinned source artifacts under `imported_oph_artifacts/`;
- the issue-503 summary as `realized_branch_receipt_report.json` for compatibility.

## Claim boundary

An imported theorem, finite witness, empirical closure, conditional envelope,
or no-go result is not a simulator run receipt. Every manifest row has
`simulation_receipt_eligible=false`, and the verifier rejects attempts to alter
that field. Run promotion requires an independent simulator verifier operating
on primitive events, matrices, actions, spectra, hashes, and refinement ancestry.

The current particle-facing interpretations are:

- the weighted-cycle neutrino candidate is a rejected historical benchmark, not
  an OPH mass default;
- the electroweak mass envelope is conditional/display-only;
- the Ward-projected hadronic spectral measure is an empirical external-data
  closure, not an OPH source theorem;
- charged-lepton and quark obstruction artifacts suppress unsupported absolute
  mass/Yukawa packets;
- the local lattice engine is a real but toy-scale diagnostic and cannot close
  the production `N_f=2+1` hadron gate;
- paper-side geometry fixtures are golden interface regressions and never flip
  the run-specific Einstein branch-entry receipt.

## Pixel parameter profiles

Code selecting a particle-facing value of `P` must declare an epistemic profile:

- `source_candidate`;
- `hierarchy_public`;
- `empirical_hadron_closure` (interval-valued; cannot be selected implicitly);
- `measured_comparison`.

Empirical and measured profiles are forbidden in recovered-core or generative
receipts. `particle_input_non_circularity_report` enforces this boundary for
source laws, repair kernels, initial states, carrier actions, species assignments,
and promotion gates.

## Dirty sources

Dirty research artifacts may be imported for diagnostics because their exact
bytes and worktree digest are pinned. They remain non-promoting. Release-grade
comparison packs should be regenerated after the research repository is clean
and committed.
