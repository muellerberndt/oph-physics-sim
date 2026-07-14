# Earned OPH universe runs

This directory holds repository-visible copies of selected earned run bundles.
The generated working directories remain under `runs/`, which is intentionally
ignored by Git.

These are OPH self-reading simulator bundles, not generic cosmology output. The
runs instantiate bounded software patches with local observer state, explicit
boundaries and interfaces, readback records, settling or repair feedback, and
public evidence receipts. Every claim remains limited by the receipts and
closed promotion gates recorded in its bundle.

## Included bundles

- `oph_universe_64k_3p1d_reearned/`: byte-for-byte copy of the populated local
  run bundle.
- `oph_universe_128k_3p1d_earned/`: byte-for-byte copy of the populated local
  run bundle.
- `oph_universe_1m_earned/`: the named local source directory was empty when
  this committed-data surface was created on 2026-07-14. Its placeholder makes
  that absence explicit; it is not a run receipt.

Treat these directories as immutable evidence snapshots. Regenerate or extend
a run under `runs/`, then replace its snapshot here as a deliberate update.

Four artifacts exceed GitHub's ordinary per-file blob limit and are assigned
to Git LFS in the repository `.gitattributes` file. Install Git LFS before
staging or committing updates to this directory.
