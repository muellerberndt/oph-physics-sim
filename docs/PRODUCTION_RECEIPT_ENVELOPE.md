# Production receipt envelope

Status: implemented P0 inventory resolver; scientific checkers remain open.

The shared resolver in `oph_fpe.evidence.production_envelope` closes the
byte/provenance boundary required by the A5-to-SM and W/Z source-to-pole
programs. It does not close a physics theorem. A producer-authored `PASS`, an
all-true output, or a syntactically valid receipt remains untrusted.

## Public API

```python
from oph_fpe.evidence.production_envelope import (
    PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE,
    PRODUCTION_BUNDLE_REPORT_SCHEMA,
    verify_production_bundle_manifest,
)

report = verify_production_bundle_manifest("/immutable/bundle/production_bundle_manifest.json")
```

The argument must be an on-disk regular, nonsymlink file. A Python mapping is
rejected. Consumers must call the verifier again from the manifest; they must
not accept a stored report or caller-supplied receipt dictionary.

Stable report boundary:

- `schema == oph.production-evidence.bundle-report/1.0.0`;
- `artifact_type == OPH_PRODUCTION_BUNDLE_INVENTORY_REPLAY`;
- `envelopes` is a dictionary keyed by `artifact_id`;
- `inventory_replay_passed` means only the P0 byte and declaration checks
  passed;
- `scientific_replay_passed`, `promotion_allowed`, and `passed` are always
  false until independently implemented scientific checkers are registered;
- an inventory-valid report has status `OPEN`; malformed evidence has status
  `FAIL`.

Each envelope report row is flat and exposes `artifact_id`, `stage_id`,
`receipt_type`, `profile`, `claim_lane`, `claim_scope`, `source_root_hash`,
`branch_id`, `freeze_id`, `producer_status`, `subject_digest`, `output_digest`,
`parent_artifact_ids`, `shared_contract_hashes`, `receipts`, and the three
separate inventory/science/promotion booleans. Its `evidence_class` is always
`IMMUTABLE_INVENTORY_ONLY`.

## Closed bundle manifest

The manifest schema is
`oph.production-evidence.bundle-manifest/1.0.0` and has exactly:

```json
{
  "schema": "oph.production-evidence.bundle-manifest/1.0.0",
  "bundle_id": "bounded.identifier",
  "manifest_payload_sha256": "sha256:...",
  "files": [
    {
      "path": "relative/path",
      "sha256": "sha256:...",
      "byte_count": 123,
      "media_type": "application/json"
    }
  ],
  "envelope_paths": ["envelopes/stage.json"]
}
```

`manifest_payload_sha256` is the OPH canonical-JSON SHA-256 of the manifest
after removing that field. The manifest does not list itself. Every other file
below the bundle root must appear exactly once. Unlisted files, missing files,
absolute paths, parent traversal, non-normalized paths, symlinks, non-regular
files, hash drift, byte-count drift, duplicate JSON keys, and non-finite JSON
numbers fail closed.

Reports should be written outside the immutable bundle. Adding a report beside
the manifest after the freeze correctly creates an unlisted-extra-file failure.

## Envelope

The envelope schema is `oph.production-evidence.envelope/1.0.0`. Its required
identity fields are:

```text
artifact_id, stage_id, receipt_type
profile, claim_lane, claim_scope
source_root_hash, branch_id, freeze_id
schema_id, schema_version, schema_sha256, schema_ref
subject_type, subject_canonicalization, subject_ref, subject_digest
output_ref, output_digest
producer, checker, freeze_anchor_ref, parent_receipts
shared_contract_hashes, numeric_precision_and_rounding
target_firewall, status, blockers, generated_utc
```

Every `*_ref` repeats the exact manifest row (`path`, `sha256`, `byte_count`,
`media_type`). This intentional redundancy detects substitution between a
producer envelope and the closed bundle inventory. The resolver loads the
schema, exact runtime subject, output, producer source/executable/environment,
checker source/executable, pre-outcome anchor, and every parent envelope from
those references.

The canonical subject/output digest policy is `OPH_CANONICAL_JSON_V1`: UTF-8,
sorted object keys, no insignificant whitespace, no ASCII escaping, and no
non-finite numbers. It is a frozen equivalent, not a claim that Python's JSON
encoder implements every RFC 8785 number-formatting detail.

Producer/checker IDs must differ. Their source and executable hashes may not
overlap, and the envelope cannot be its own checker. Parent references include
the parent envelope bytes plus its subject and output digests. Missing hashes,
unknown parents, self-parenting, ancestry cycles, and mixed
source-root/branch/freeze families fail closed.

## Profiles

Only two production profiles exist:

- `COMMON_STAGE` uses the common envelope without imposing W/Z-specific
  scientific coordinates.
- `WZ_SOURCE_TO_POLE` additionally requires one consistent hash family for
  `action_ast_hash`, `field_census_hash`, `scheme_hash`,
  `fj_convention_hash`, `term_mask_hash`, `analytic_sheet_hash`, and
  `units_basis_hash`.

Every value in `shared_contract_hashes` must equal the verified SHA-256 of at
least one listed bundle file. A hash-shaped string with no resolved bytes is
rejected.

`DEMO_ASSUMPTION`, visualization overlays, WZH0 synthetic controls, and any
unknown profile are not production evidence and cannot enter this manifest.

## Freeze and target boundary

The freeze anchor is a separately hashed JSON object with schema
`oph.production-evidence.freeze-anchor/1.0.0`, the same source root, branch and
freeze IDs, `commitment_phase=pre_outcome`, and a valid second-precision UTC
freeze time.  Every envelope must have `frozen_utc < generated_utc`, and every
envelope in one source-root/branch/freeze family must resolve to the identical
anchor bytes.  These checks close internal chronology and anchor-substitution
gaps; an on-disk declaration still does not independently prove that the
commitment existed before an externally observed outcome.

The target-firewall object declares that targets are absent from ancestry,
selection, producer tuning, checker tuning, and producer/checker mounts, and
that the comparison process cannot mutate the bundle. The comparison boundary
must be `read_only_separate_process`. W/Z envelopes must be labeled
`post_exposure_validation`; they cannot claim prospective blindness for
already exposed W/Z targets.

## Epistemic boundary

Allowed producer statuses are exactly `PASS`, `OPEN`, `UNRESOLVED`, `FAIL`, and
`NOT_APPLICABLE`. They are preserved under `producer_status` and
`ignored_producer_statuses`, never treated as checker results. The registered
scientific-checker table is intentionally empty. Consequently:

```text
P0_IMMUTABLE_INVENTORY_REPLAY_RECEIPT may be true
SCIENTIFIC_REPLAY_RECEIPT = false
PROMOTION_RECEIPT = false
```

Future scientific verifiers must consume primitive records, recompute their
domain residuals independently, and then be explicitly registered. They must
not weaken or bypass this immutable P0 replay layer.
