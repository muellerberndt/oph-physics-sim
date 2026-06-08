Below is the build plan I would hand to the coding agent. It separates **measurement data**, **3D bulk emergence**, and **proto-particles** into distinct receipts, because the latest handoff says the sim has finite repair, direct BW/KMS sanity, chart-level H3, screen (C_\ell) proxies, and S3 screen/collar defects, but still fails populated bulk, physical CMB, and particle receipts. 

I also unpacked the latest zip in the sandbox and tried to run the bundled tests. The bundle itself is not reproducible as sent: tests fail immediately because modules such as `oph_fpe.bulk.cap_normals`, `oph_fpe.claims`, and `oph_fpe.experiments` are missing. So the very first engineering step is to make the handoff bundle self-contained again.

---

# 0. The core diagnosis

The simulator is trying to make three different claims with one tangled run:

```text
A. OPH recovered-core chart:
   S2 caps + BW/KMS -> Conf+(S2) ~ SO+(3,1) -> H3 chart

B. Populated 3D bulk:
   observer records / objects / defects populate that H3 chart under controls

C. Measurement prediction:
   OPH-derived observables compare to real CMB / galaxies / BAO / growth
```

These must be split.

The latest handoff already has the right separation: chart-level conformal Lorentz/H3 is present, while `bulk_3d_established`, `physical_cmb_prediction`, and `particle_matter_receipt` remain false. It also says wrong-scale H3 controls still win, and blind observer features are not S2-leaky but still do not form a 3D neutral bulk. 

The OPH papers support this separation. The screen/microphysics paper says the spherical screen is a geometry chart for support-visible cuts, while the underlying implementation is a finite patch federation; caps on (S^2) supply the Lorentz bridge because the conformal group of (S^2) is (SO^+(3,1)).  The synthesis paper says the recovered core proceeds from overlap repair, to modular-flow Lorentz kinematics, to compact-gauge reconstruction and MAR selection; it also says (P) and (N_{\rm scr}) belong to declared quantitative branches, not the recovered-core bulk proof. 

So the build target should be:

```text
1. Keep chart-level H3/Lorentz as a recovered-core receipt.
2. Build a separate populated-H3 receipt.
3. Build measurement lanes that do not wait for full populated bulk.
4. Build proto-particles only after screen defects first pass dynamics receipts.
```

---

# 1. Fix reproducibility before physics work

## Problem

The current zip claims `150 passed`, but in the sandbox the zip cannot run its tests because several imported modules are missing. The latest handoff lists the modules and tests that should be present, including `h3_chart.py`, `observer_reconstruction.py`, `modular_response_kernel.py`, `oph_cmb_adapter.py`, and tests for these modules.  The supplied zip is not sufficient to reproduce that state.

## Required files to include in every Pro/Codex handoff

```text
oph_fpe/__init__.py
oph_fpe/claims.py
oph_fpe/experiments.py
oph_fpe/bulk/cap_normals.py
oph_fpe/bulk/cap_geometry.py
oph_fpe/bulk/h3_chart.py
oph_fpe/bulk/h3_response_fit.py
oph_fpe/bulk/modular_response_kernel.py
oph_fpe/bulk/observer_reconstruction.py
oph_fpe/bulk/record_to_h3.py
oph_fpe/core/*
oph_fpe/groups/*
oph_fpe/evidence/*
oph_fpe/cosmology/*
configs/*
tests/*
pyproject.toml
REPRODUCTION.md
```

## Add this CI check

```python
# tools/check_handoff_bundle.py

REQUIRED_IMPORTS = [
    "oph_fpe.claims",
    "oph_fpe.experiments",
    "oph_fpe.bulk.cap_normals",
    "oph_fpe.bulk.cap_geometry",
    "oph_fpe.bulk.h3_chart",
    "oph_fpe.bulk.observer_reconstruction",
    "oph_fpe.bulk.modular_response_kernel",
    "oph_fpe.bulk.h3_response_fit",
    "oph_fpe.bulk.record_to_h3",
    "oph_fpe.cosmology.oph_cmb_adapter",
    "oph_fpe.cosmology.anomaly_fluid",
]

def main():
    import importlib
    failed = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append((name, repr(exc)))

    if failed:
        for name, exc in failed:
            print(f"IMPORT_FAIL {name}: {exc}")
        raise SystemExit(1)

    print("HANDOFF_IMPORTS_OK")

if __name__ == "__main__":
    main()
```

Run before sending any bundle:

```bash
python tools/check_handoff_bundle.py
python -m pytest -q
```

No physics result should be accepted from a non-reproducible bundle.

---

# 2. Freeze the receipt ladder

The current handoff says a previous overclaim bug could mark `bulk_3d_established = true` from object/H3 candidates even when the real bulk reconstruction was false, and that this was blocked.  Keep that fix, but simplify the ladder further.

## Canonical receipts

```python
# oph_fpe/claims.py

REPAIR_CORE_RECEIPT = "REPAIR_CORE_RECEIPT"
RECORD_COMMIT_RECEIPT = "RECORD_COMMIT_RECEIPT"

CHART_LORENTZ_H3_RECEIPT = "CHART_LORENTZ_H3_RECEIPT"
BW_KMS_DIRECT_2PI_RECEIPT = "BW_KMS_DIRECT_2PI_RECEIPT"
ENDOGENOUS_MODULAR_GENERATOR_RECEIPT = "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"

H3_RESPONSE_CANDIDATE_RECEIPT = "H3_RESPONSE_CANDIDATE_RECEIPT"
OBJECT_CHART_RECEIPT = "OBJECT_CHART_RECEIPT"
OBJECT_BULK_POPULATION_RECEIPT = "OBJECT_BULK_POPULATION_RECEIPT"

SCREEN_DEFECT_DYNAMICS_RECEIPT = "SCREEN_DEFECT_DYNAMICS_RECEIPT"
H3_DEFECT_WORLDLINE_RECEIPT = "H3_DEFECT_WORLDLINE_RECEIPT"
PROTO_PARTICLE_RECEIPT = "PROTO_PARTICLE_RECEIPT"

SCREEN_PROXY_CMB_RECEIPT = "SCREEN_PROXY_CMB_RECEIPT"
STATIC_GALAXY_RAR_BTFR_RECEIPT = "STATIC_GALAXY_RAR_BTFR_RECEIPT"
DYNAMIC_DARK_TRANSPORT_RECEIPT = "DYNAMIC_DARK_TRANSPORT_RECEIPT"
OPH_BOLTZMANN_KERNEL_RECEIPT = "OPH_BOLTZMANN_KERNEL_RECEIPT"
PHYSICAL_CMB_RECEIPT = "PHYSICAL_CMB_RECEIPT"
```

## Mandatory metadata

```python
def receipt_meta(
    *,
    receipt: str,
    claim_level: str,
    physical_claim: bool,
    observable_id: str,
    fit_objective: str,
) -> dict:
    return {
        "receipt_schema_version": "2026-06-06b",
        "receipt": receipt,
        "claim_level": claim_level,
        "physical_claim": bool(physical_claim),
        "observable_id": observable_id,
        "fit_objective": fit_objective,
    }
```

## Gate hierarchy

```python
def compute_emergence_status(reports):
    status = {}

    status["repair_core"] = reports["repair"]["final_phi"] == 0
    status["records_commit"] = reports["records"]["global_committed_fraction"] >= 0.95

    # Core recovered-chart receipt.
    # This is NOT the populated bulk receipt.
    status["chart_lorentz_h3"] = (
        reports["h3_chart"].get("conformal_h3_spatial_chart_receipt", False)
        and reports["bw_direct"].get("selected_label") == "2pi"
        and reports["bw_direct"].get("correct_beats_controls", False)
    )

    # Later strengthening only.
    status["endogenous_modular_generator"] = (
        reports["state_modular"].get("endogenous_modular_generator", False)
        and reports["state_modular"].get("selected_label") == "2pi"
        and reports["state_modular"].get("correct_beats_controls", False)
    )

    # Populated H3 bulk: do not use raw neutral dimension as a primary gate.
    status["object_bulk_population"] = (
        reports["h3_response"].get(H3_RESPONSE_CANDIDATE_RECEIPT, False)
        and reports["object_chart"].get(OBJECT_CHART_RECEIPT, False)
        and reports["object_chart"].get(OBJECT_BULK_POPULATION_RECEIPT, False)
        and reports["blind_audit"].get("s2_leakage_audit_pass", False)
    )

    status["bulk_3d_established"] = status["object_bulk_population"]

    status["proto_particles"] = (
        status["bulk_3d_established"]
        and reports["defects"].get(H3_DEFECT_WORLDLINE_RECEIPT, False)
        and reports["defects"].get(PROTO_PARTICLE_RECEIPT, False)
    )

    status["physical_cmb_prediction"] = (
        reports["boltzmann"].get(OPH_BOLTZMANN_KERNEL_RECEIPT, False)
        and reports["cmb"].get(PHYSICAL_CMB_RECEIPT, False)
    )

    return status
```

This prevents the neutral dimension diagnostic from blocking or creating a bulk claim. The handoff itself asks whether the right interpretation should be chart-level H3 first, observer records populating that chart second, and dimension estimation as secondary sanity check. That is the correct interpretation. 

---

# 3. Build lane A: core Lorentz/H3 chart

## Goal

Make this lane boring and stable:

```text
finite repair -> records -> direct 2pi cap/BW sanity -> conformal H3 chart
```

Do not require:

* neutral dimension = 3,
* object population,
* endogenous modular generator,
* CMB,
* particles.

## Required report

```json
{
  "receipt": "CHART_LORENTZ_H3_RECEIPT",
  "claim_level": "recovered_core_chart",
  "physical_claim": false,
  "chart": "H3 = SO+(3,1)/SO(3)",
  "source": "S2 cap conformal/BW chart",
  "bulk_populated": false
}
```

## Pseudocode

```python
def chart_lorentz_h3_report(caps, bw_report, h3_chart_report):
    h3_chart_ok = h3_chart_report["conformal_h3_spatial_chart_receipt"]

    direct_bw_ok = (
        bw_report["primary_source"] == "kms_collar_transport_response"
        and bw_report["selected_label"] == "2pi"
        and bw_report["correct_beats_controls"]
        and not bw_report.get("response_degenerate", False)
    )

    return {
        **receipt_meta(
            receipt=CHART_LORENTZ_H3_RECEIPT,
            claim_level="recovered_core_chart",
            physical_claim=False,
            observable_id="s2_caps_bw_conformal_h3",
            fit_objective="direct_2pi_bw_sanity",
        ),
        "conformal_group": "Conf+(S2) ~ SO+(3,1)",
        "spatial_chart": "H3 ~ SO+(3,1)/SO(3)",
        "chart_receipt": bool(h3_chart_ok and direct_bw_ok),
        "populated_bulk_claim": False,
        "endogenous_generator_required": False,
    }
```

The papers and handoff support this separation: the direct finite BW/KMS cap-flow sanity currently selects `2pi`, while the endogenous history-kernel surrogate still fails; these are different surfaces. 

---

# 4. Build lane B: populated H3 bulk

The main blocker is still here. The latest handoff says object families exist, but object incidence is close to shuffled controls, wrong-scale controls win, and blind reconstruction does not produce a 3D window. 

## 4.1 Replace the finite endogenous modular surrogate

### Problem

The current endogenous `history_transition_kernel` selects `1x` and fails controls.  The concern in the handoff is that the finite (\rho_C) / transition-kernel construction may be too diagonal/classical and may not encode the noncommutative/collar operator structure. 

### Replacement object

Use a **finite collar operator system**, not only a diagonal token Markov kernel.

For each cap (C), define an operator basis:

```text
record projectors
checkpoint projectors
sector projectors
repair-load bucket projectors
collar edge transition operators
perturb-resettle response operators
```

Let this basis be:

[
\mathcal B_C = {O_1,\dots,O_m}.
]

Construct a finite Gram/covariance state:

[
G_{ab}=\langle O_a^\dagger O_b\rangle_C.
]

Regularize:

[
\rho_C = \frac{G+\epsilon I}{\operatorname{Tr}(G+\epsilon I)}.
]

Then:

[
K_C^{(a)}=-\log(\rho_C+aI).
]

Compare the induced modular automorphism to direct cap transport in the same operator basis.

### Pseudocode

```python
def build_collar_operator_basis(cap, observer_rows, graph_state):
    basis = []

    basis += record_projector_basis(observer_rows, cap)
    basis += checkpoint_projector_basis(observer_rows, cap)
    basis += sector_projector_basis(observer_rows, cap)
    basis += repair_load_projector_basis(observer_rows, cap)
    basis += collar_edge_transition_basis(graph_state, cap)
    basis += perturb_response_operator_basis(observer_rows, graph_state, cap)

    return orthonormalize_operator_basis(basis)


def finite_collar_density_matrix(basis, samples, eps=1e-9):
    m = len(basis)
    G = np.zeros((m, m), dtype=np.complex128)

    for sample in samples:
        vals = np.array([op.eval(sample) for op in basis], dtype=np.complex128)
        G += np.outer(np.conjugate(vals), vals)

    G /= max(len(samples), 1)
    G = 0.5 * (G + G.conj().T)
    G += eps * np.eye(m)
    G /= np.trace(G).real
    return G


def modular_generator_from_density(rho, regularizer=1e-12):
    evals, evecs = np.linalg.eigh(rho)
    evals = np.maximum(evals, regularizer)
    K = evecs @ np.diag(-np.log(evals)) @ evecs.conj().T
    K = 0.5 * (K + K.conj().T)
    return K


def direct_cap_transport_matrix(cap, basis, scale, time, samples):
    # Matrix representation of O -> O ∘ lambda_C(scale*time)
    m = len(basis)
    T = np.zeros((m, m), dtype=np.complex128)

    transported_samples = [lambda_cap_sample(s, cap, scale * time) for s in samples]

    # Least-squares projection back into operator basis.
    X = np.array([[op.eval(s) for op in basis] for s in samples])
    Y = np.array([[op.eval(s2) for op in basis] for s2 in transported_samples])

    # Solve X @ T ~= Y
    T, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return T


def bw_operator_receipt(cap, basis, rho, samples, times, scales):
    K = modular_generator_from_density(rho)

    results = {}
    for scale in scales:
        residuals = []
        for t in times:
            U = scipy.linalg.expm(1j * (scale / (2*np.pi)) * K * t)
            modular_action = U.conj().T @ observable_matrix(basis) @ U

            direct = direct_cap_transport_matrix(cap, basis, scale, t, samples)

            residuals.append(matrix_residual(modular_action, direct))
        results[scale_label(scale)] = np.median(residuals)

    return {
        "selected_label": min(results, key=results.get),
        "residuals": results,
        "correct_beats_controls": results["2pi"] < min(
            results[label] for label in results if label != "2pi"
        ),
    }
```

Acceptance:

```text
ENDOGENOUS_MODULAR_GENERATOR_RECEIPT
fires only if:
  selected_label == 2pi
  2pi beats 1x, pi, 4pi, no-flow, shuffled-observable
  result holds across at least 3/4 seeds at 64k
```

---

## 4.2 Fix wrong-scale controls

### Current problem

The current configs still use:

```yaml
perturb_budget_mode: fixed_collar_fraction
fixed_perturb_fraction: 0.25
```

at 64k and 256k. That makes wrong-scale controls too similar, because every scale perturbs the same fraction of collar edges. Meanwhile, the H3 report fits wrong-scale controls with per-channel affine nuisance parameters, which can make wrong-scale controls artificially easy to match.

The handoff says wrong-scale controls win strongly, with `wrong_scale_win_fraction ~ 0.90-0.95`. 

### Required config change

```yaml
h3_modular_response:
  perturb_budget_mode: modular_amount
  fixed_perturb_fraction: null
```

### Required scoring change

Do **not** compare:

```text
H3 fit vs affine fit of wrong-scale control to the target
```

Compare:

```text
2pi model vs wrong-scale model under the same H3 geometric model class
```

### Pseudocode

```python
def build_response_for_scale(scale, config):
    return modular_response_kernel(
        transport_scale=scale,
        perturb_budget_mode="modular_amount",
        fixed_perturb_fraction=None,
        transform="signed_zscore",
        **config,
    )


def fit_scale_family(scales, config, train_mask):
    fits = {}

    for scale in scales:
        R_s = build_response_for_scale(scale, config)

        fit_s = joint_h3_response_fit(
            kernel=R_s,
            fit_mode="joint_global",
            nuisance="same_as_2pi",
            train_mask=train_mask,
        )

        fits[scale_label(scale)] = fit_s

    return fits


def wrong_scale_receipt(fits):
    scores = {
        label: fit["heldout_normalized_rmse"]
        for label, fit in fits.items()
    }

    selected = min(scores, key=scores.get)

    return {
        "selected_label": selected,
        "scores": scores,
        "two_pi_wins": selected == "2pi",
        "wrong_scale_win_fraction": float(selected != "2pi"),
    }
```

Pass:

```text
H3_RESPONSE_CANDIDATE_RECEIPT requires:
  2pi H3 fit beats:
    S2 boundary
    shuffled response
    no perturbation
    wrong 1x
    wrong pi
    wrong 4pi
```

This should be tested first at 4k, not 256k.

---

## 4.3 Replace object extraction with persistent multi-view record objects

### Problem

Current object extraction is still close to shuffled controls. The latest 256k run had 349 observer-chart objects and 172 localized-not-boundary objects, but the shuffled p90 was 171, so it cannot be accepted. 

### Correct object definition

An object should not be a support overlap or final histogram. It should be a **persistent multi-observer record normal form**.

Each observer (i) has a history:

[
Y_i(t) =
(\text{record family}, \text{checkpoint class}, \text{stable flag}, \text{sector class}, \text{repair bucket}).
]

An object is a cluster of observers whose histories show the same persistent transition pattern and whose cap perturbation responses agree.

### Pseudocode

```python
def observer_state_token(row, cycle):
    return (
        row.record_family_id[cycle],
        row.checkpoint_class[cycle],
        row.stable_flag[cycle],
        row.s3_sector_class[cycle],
        row.repair_load_bucket[cycle],
    )


def observer_history_signature(row, t0, window=4):
    hist = [observer_state_token(row, t) for t in range(t0-window+1, t0+1)]

    return {
        "transition_counts": count_transitions(hist),
        "state_counts": count_states(hist),
        "persistence": longest_constant_run(hist),
        "sector_changes": count_sector_changes(hist),
        "checkpoint_changes": count_checkpoint_changes(hist),
    }


def object_feature(row, t0):
    history = observer_history_signature(row, t0)
    response = row.perturb_resettle_signature
    spectrum = row.repair_response_spectrum

    return concat_features(
        normalize_counts(history["transition_counts"]),
        normalize_counts(history["state_counts"]),
        [history["persistence"], history["sector_changes"], history["checkpoint_changes"]],
        response,
        spectrum,
    )


def build_record_objects(observer_rows, t0, *, min_observers=6):
    X = np.vstack([object_feature(row, t0) for row in observer_rows])

    # Use density clustering; avoid forcing every observer into an object.
    labels = hdbscan_or_dbscan(X, min_cluster_size=min_observers)

    objects = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = np.flatnonzero(labels == label)
        if len(members) >= min_observers:
            objects.append({
                "object_id": int(label),
                "observer_indices": members.tolist(),
                "feature_centroid": np.mean(X[members], axis=0).tolist(),
                "persistence": median_persistence(observer_rows, members, t0),
            })
    return objects
```

### H3 placement

```python
def place_objects_in_h3(objects, observer_h3_points):
    placed = []
    for obj in objects:
        pts = observer_h3_points[obj["observer_indices"]]
        center = hyperbolic_frechet_mean(pts)
        compactness = median_h3_distance(pts, center)

        placed.append({
            **obj,
            "h3_center": center.tolist(),
            "h3_compactness": float(compactness),
            "observer_count": len(obj["observer_indices"]),
        })
    return placed
```

### Controls

```python
def object_population_controls(objects, observer_h3_points, observer_s2_points, n=64):
    actual = object_compactness_report(objects, observer_h3_points, observer_s2_points)

    shuffled = []
    for seed in range(n):
        shuffled_objects = shuffle_object_memberships(objects, seed=seed)
        shuffled.append(object_compactness_report(
            shuffled_objects,
            observer_h3_points,
            observer_s2_points,
        ))

    return {
        "actual": actual,
        "shuffled_p95": percentile(shuffled, 95),
        "shuffled_p99": percentile(shuffled, 99),
        "actual_beats_shuffle": actual["localized_not_boundary_count"] > percentile(
            [x["localized_not_boundary_count"] for x in shuffled], 95
        ),
    }
```

### Receipt

```python
def object_bulk_population_receipt(report):
    return (
        report["object_count"] >= 12
        and report["localized_not_boundary_count"] >= 3
        and report["localized_not_boundary_count"] > report["shuffle_localized_p95"]
        and report["median_h3_compactness"] < report["median_s2_boundary_compactness"]
        and report["blind_s2_leakage_audit_pass"]
    )
```

---

## 4.4 Neutral reconstruction should become a diagnostic only

Do not require the neutral blind dimension estimate to return 3. It is already useful because it now reports low S2 leakage, but the handoff says its dimension is still around 4.65–5.29 at 256k. 

Use it this way:

```text
NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC:
  verifies no forbidden S2/radial leakage
  reports blind feature dimension
  does not decide bulk_3d_established
```

### Pseudocode

```python
def neutral_debug_report(features):
    D = pairwise_distance(features)
    dim = dimension_diagnostics(D)

    return {
        "receipt": "NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        "physical_claim": False,
        "s2_leakage": corr(D, s2_distance_matrix),
        "s2_leakage_pass": abs(corr(D, s2_distance_matrix)) < 0.05,
        "dimension_debug": dim,
        "bulk_gate_participation": "leakage_audit_only",
    }
```

---

# 5. Build lane C: proto-particles

The first particles should be **defect excitations**, not Standard Model particles. The particle paper says actual particle content appears only after overlap consistency, compact gauge reconstruction, and MAR selection; structural carriers such as photon, gluon, and graviton are symmetry-protected zeros on the realized branch, while quarks/leptons and hadrons are downstream. 

For the simulator, the first visual proto-particles should be:

```text
Z2 parity defects
S3 holonomy defects
H3-localized defect worldlines
```

The consensus paper says obstructions are holonomic: abelian cycle sums first, crossed-module/higher-gauge defects later. 

## 5.1 Screen defect dynamics receipt

Before H3:

```text
SCREEN_DEFECT_DYNAMICS_RECEIPT
```

### Pseudocode

```python
def triangle_holonomy(edge_labels, triangle):
    a, b, c = triangle
    return compose(compose(edge_labels[a, b], edge_labels[b, c]), edge_labels[c, a])


def detect_s3_defects(mesh, edge_labels):
    defects = []
    for tri in mesh.triangles:
        h = triangle_holonomy(edge_labels, tri)
        cls = conjugacy_class_s3(h)
        if cls != "identity":
            defects.append({
                "triangle": tri,
                "holonomy": h,
                "class": cls,
                "centroid_s2": triangle_centroid(tri),
            })
    return defects


def cluster_defects(defects, adjacency):
    return connected_components_by_class_and_adjacency(defects, adjacency)


def track_defect_worldlines(defect_clusters_by_cycle):
    tracks = []
    active = []

    for cycle, clusters in enumerate(defect_clusters_by_cycle):
        matches = match_clusters(active, clusters, cost="s2_distance+class_mismatch")
        active, closed = update_tracks(active, clusters, matches, cycle)
        tracks += closed

    return tracks + active
```

### Receipt

```python
def screen_defect_dynamics_receipt(tracks):
    persistent = [tr for tr in tracks if tr.lifetime >= MIN_LIFETIME]
    conserved = [tr for tr in persistent if tr.class_preserved]
    return {
        "receipt": SCREEN_DEFECT_DYNAMICS_RECEIPT,
        "persistent_count": len(persistent),
        "class_conserved_count": len(conserved),
        "passed": len(conserved) >= 3,
        "physical_claim": False,
    }
```

## 5.2 Fusion and annihilation

```python
def detect_fusion_events(tracks):
    events = []
    for t in time_slices(tracks):
        incoming = tracks_ending_near_same_place(t)
        outgoing = tracks_starting_near_same_place(t+1)

        if len(incoming) >= 2 and len(outgoing) >= 1:
            events.append({
                "time": t,
                "incoming_classes": [tr.cls for tr in incoming],
                "outgoing_classes": [tr.cls for tr in outgoing],
                "group_law_ok": s3_fusion_allowed(incoming, outgoing),
            })

        if len(incoming) >= 2 and len(outgoing) == 0:
            events.append({
                "time": t,
                "incoming_classes": [tr.cls for tr in incoming],
                "outgoing_classes": [],
                "annihilation": True,
            })

    return events
```

## 5.3 H3 defect worldlines

Only after `OBJECT_BULK_POPULATION_RECEIPT`.

```python
def map_defects_to_h3(defects, observer_rows, observer_h3_points):
    mapped = []

    for defect in defects:
        seeing_observers = observers_that_see_defect(defect, observer_rows)
        if len(seeing_observers) < MIN_OBSERVERS:
            continue

        pts = observer_h3_points[seeing_observers]
        center = hyperbolic_frechet_mean(pts)
        compactness = median_h3_distance(pts, center)

        mapped.append({
            **defect,
            "h3_center": center,
            "h3_compactness": compactness,
        })

    return mapped
```

## Proto-particle receipt

```python
def proto_particle_receipt(h3_tracks, fusion_events):
    particle_like = []

    for tr in h3_tracks:
        if (
            tr.lifetime >= MIN_PARTICLE_LIFETIME
            and tr.h3_compactness_median <= MAX_PARTICLE_COMPACTNESS
            and tr.class_preserved
            and tr.transportable
            and tr.energy_excess_stable
        ):
            particle_like.append(tr)

    return {
        "receipt": PROTO_PARTICLE_RECEIPT,
        "particle_like_count": len(particle_like),
        "fusion_events_checked": len(fusion_events),
        "fusion_law_pass_fraction": mean(e.group_law_ok for e in fusion_events),
        "passed": len(particle_like) >= 1,
        "claim": "proto-particle topological excitation, not SM particle",
    }
```

---

# 6. Build lane D: usable measurement data

You need **any data** comparable to real observations. There are four levels.

## Level D0: Screen proxy, keep but label honestly

The latest handoff says the screen (C_\ell) proxy has normalized-axis Planck-lite correlation around 0.455 and cannot compare in real (\ell)-space because the sim (\ell_{\max}) is 48 while Planck binned (\ell_{\max}) is about 2499; it is not a physical CMB prediction.  It also says the screen-basis transfer reaches test correlation around 0.748 against controls, but fits weights and is not a prediction. 

Keep it as:

```text
SCREEN_PROXY_CMB_RECEIPT
```

### Pseudocode

```python
def screen_proxy_cmb_report(run):
    fields = [
        "record_signature",
        "record_signature_smooth_k16",
        "record_signature_smooth_k32",
        "record_repair_mix",
        "record_sector_mix",
        "defect_density",
        "repair_load",
    ]

    rows = []
    for field in fields:
        cl = angular_power(run.screen_points, run.field[field], ell_max=64)
        score = positive_amplitude_shape_score(cl, planck_tt_binned_low_l)
        controls = score_controls(cl, planck_tt_binned_low_l)

        rows.append({
            "field": field,
            "shape_correlation": score.corr,
            "normalized_rmse": score.rmse,
            "max_control_corr": max(c.corr for c in controls),
            "beats_controls": score.corr > max(c.corr for c in controls),
        })

    return {
        "receipt": SCREEN_PROXY_CMB_RECEIPT,
        "physical_cmb_prediction": False,
        "rows": rows,
        "claim_boundary": "screen-level angular proxy only",
    }
```

## Level D1: Static galaxy law — first real measurement lane

This should be the first serious data lane.

The OPH dark matter paper separates three claim layers: static galaxy law, dynamic dark component, and cosmological dark matter. The first layer contains RAR, BTFR, effective dark density, no-slip lensing, and the Poisson/(\mathbb Z_6) coefficient. 

Use SPARC first. SPARC contains 175 galaxies with Spitzer 3.6 μm photometry and high-quality HI/Hα rotation curves, and is explicitly designed as a public test bed for mass models. ([arXiv][1])

### Static OPH law

[
\nu_{\rm OPH}(x)=\frac{1}{1-\exp[-\lambda_{\rm collar}\sqrt{x}]},
\qquad
x=\frac{g_b}{a_{0,\rm OPH}}.
]

The dark paper says this static galaxy formula is an equilibrium projection of a transported stress component, not the master law for clusters or early cosmology. 

### Pseudocode

```python
# oph_fpe/cosmology/galaxy_static.py

def nu_oph(x, lambda_collar):
    x = np.maximum(np.asarray(x, dtype=float), 1e-30)
    return 1.0 / (1.0 - np.exp(-lambda_collar * np.sqrt(x)))


def g_model(g_b, a0, lambda_collar):
    x = g_b / a0
    return nu_oph(x, lambda_collar) * g_b


def v_model_from_baryons(r, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge, a0, lambda_collar):
    # SPARC convention: baryonic velocity contributions combine as acceleration.
    v_bar2 = (
        np.abs(v_gas) * v_gas
        + upsilon_disk * np.abs(v_disk) * v_disk
        + upsilon_bulge * np.abs(v_bulge) * v_bulge
    )

    g_b = np.maximum(v_bar2 / np.maximum(r, 1e-12), 1e-30)
    g_obs = g_model(g_b, a0, lambda_collar)

    return np.sqrt(np.maximum(g_obs * r, 0.0))


def fit_galaxy(galaxy):
    def loss(theta):
        log_a0, log_lambda, log_ud, log_ub = theta
        a0 = np.exp(log_a0)
        lam = np.exp(log_lambda)
        ud = np.exp(log_ud)
        ub = np.exp(log_ub)

        pred = v_model_from_baryons(
            galaxy.r,
            galaxy.v_gas,
            galaxy.v_disk,
            galaxy.v_bulge,
            ud,
            ub,
            a0,
            lam,
        )

        return np.sum(((galaxy.v_obs - pred) / galaxy.v_err) ** 2)

    result = scipy.optimize.minimize(loss, initial_guess())
    return result


def sparc_population_fit(galaxies):
    # Shared lambda and a0; galaxy-specific mass-to-light nuisance.
    def global_loss(theta):
        log_a0 = theta[0]
        log_lambda = theta[1]
        nuisance = theta[2:]

        total = 0.0
        for idx, galaxy in enumerate(galaxies):
            log_ud, log_ub = nuisance[2*idx], nuisance[2*idx+1]
            total += galaxy_loss(galaxy, log_a0, log_lambda, log_ud, log_ub)
            total += stellar_prior(log_ud, log_ub)
        return total

    result = scipy.optimize.minimize(global_loss, initial_guess_global())
    return result
```

### Receipts

```python
def static_galaxy_receipt(fit):
    return {
        "receipt": STATIC_GALAXY_RAR_BTFR_RECEIPT,
        "dataset": "SPARC",
        "galaxy_count": fit.galaxy_count,
        "shared_a0": fit.a0,
        "shared_lambda_collar": fit.lambda_collar,
        "rar_scatter_dex": fit.rar_scatter_dex,
        "btfr_slope": fit.btfr_slope,
        "chi2_per_dof": fit.chi2_per_dof,
        "passed": (
            fit.galaxy_count >= 100
            and fit.rar_scatter_dex < 0.15
            and abs(fit.btfr_slope - 4.0) < 0.3
        ),
        "physical_claim": True,
        "claim_boundary": "static relaxed-galaxy law only; not CMB or cluster prediction",
    }
```

This is the fastest route to usable measurement data.

## Level D2: Dynamic dark transport

Clusters, offsets, and time-dependent systems need layer 2. The dark paper says the anomaly is a transported stress that relaxes toward static equilibrium over (\tau_{\rm rec}). 

### Pseudocode

```python
# oph_fpe/cosmology/dark_transport.py

def anomaly_equilibrium_density(g_b, baryon_density, a0, lambda_collar):
    response = nu_oph(g_b / a0, lambda_collar) - 1.0
    return response * effective_poisson_source(baryon_density)


def evolve_anomaly_density(rho_A, rho_A_eq, velocity_field, tau_rec, dt, diffusion=0.0):
    adv = advect(rho_A, velocity_field, dt)
    relax = adv + dt * (rho_A_eq - adv) / max(tau_rec, 1e-12)
    if diffusion > 0:
        relax = relax + dt * diffusion * laplacian(relax)
    return np.maximum(relax, 0.0)


def cluster_merger_proxy(initial_baryons, steps, params):
    rho_A = initialize_anomaly(initial_baryons)
    outputs = []

    for t in range(steps):
        baryons = update_baryons(initial_baryons, t)
        rho_A_eq = anomaly_equilibrium_density(
            baryons.g_b,
            baryons.rho,
            params.a0,
            params.lambda_collar,
        )
        rho_A = evolve_anomaly_density(
            rho_A,
            rho_A_eq,
            baryons.velocity,
            params.tau_rec,
            params.dt,
            params.diffusion,
        )
        outputs.append(measure_lensing_offset(baryons.rho, rho_A))

    return outputs
```

Receipt:

```python
DYNAMIC_DARK_TRANSPORT_RECEIPT fires if:
  relaxation solver stable
  stress remains nonnegative
  lensing/source offset can be produced
  static limit recovers RAR/BTFR
```

## Level D3: OPH Boltzmann kernel

The latest handoff says the physical CMB adapter is gated closed until it has theorem-grade (\rho_A(a)), (\rho_{A,\rm eq}(a)), (\Gamma_{\rm rec}(k,a)), non-fit (B_A(k,a)), and a CAMB/CLASS anomaly module. 

Do this before any physical CMB claim.

### Pseudocode

```python
# oph_fpe/cosmology/boltzmann_inputs.py

@dataclass
class OPHAnomalyKernels:
    a: np.ndarray
    k: np.ndarray
    rho_A: np.ndarray
    rho_A_eq: np.ndarray
    w_A: np.ndarray
    cs2_A: np.ndarray
    gamma_rec: np.ndarray      # Gamma_rec(k,a)
    B_A: np.ndarray            # baryon-response kernel B_A(k,a)


def derive_B_A_from_finite_collar_perturbations(runs):
    # Non-fit kernel: controlled finite differences.
    samples = []

    for run in runs:
        for k_bin in run.k_bins:
            for a_bin in run.scale_factor_bins:
                base = measure_anomaly_response(run, k_bin, a_bin, perturb=False)
                pert = measure_anomaly_response(run, k_bin, a_bin, perturb=True)
                delta_b = measure_baryon_perturbation(run, k_bin, a_bin)

                B = (pert - base) / max(delta_b, 1e-12)
                samples.append((k_bin, a_bin, B))

    return aggregate_kernel(samples)


def anomaly_fluid_ode(a, y, kernels):
    delta_A, theta_A = y

    H = H_of_a(a)
    w = interp(kernels.w_A, a)
    cs2 = interp(kernels.cs2_A, a)
    gamma = interp2(kernels.gamma_rec, k, a)
    B = interp2(kernels.B_A, k, a)

    delta_b = baryon_delta(k, a)

    d_delta_A = -(1 + w) * (theta_A - 3 * phi_dot(k, a)) \
                - 3 * H * (cs2 - w) * delta_A \
                - gamma * (delta_A - B * delta_b)

    d_theta_A = -H * (1 - 3*w) * theta_A \
                + k*k * (cs2 * delta_A / (1 + w) + psi(k, a))

    return [d_delta_A, d_theta_A]
```

Receipt:

```python
def boltzmann_kernel_receipt(kernels):
    return {
        "receipt": OPH_BOLTZMANN_KERNEL_RECEIPT,
        "has_rho_A": finite(kernels.rho_A),
        "has_gamma_rec": finite(kernels.gamma_rec),
        "has_B_A": finite(kernels.B_A),
        "nonfit_B_A": kernels.B_A_source == "finite_collar_perturbation",
        "passed": (
            finite(kernels.rho_A)
            and finite(kernels.gamma_rec)
            and finite(kernels.B_A)
            and kernels.B_A_source == "finite_collar_perturbation"
        ),
    }
```

Only after this should CAMB/CLASS be used for OPH CMB. DESI provides BAO expansion-history targets; DESI’s first-year BAO result was described as the largest BAO dataset by galaxy count and volume with 0.52% precision, and later DR2 BAO results used observations of 14 million galaxies and quasars from the first three years of DESI operations. ([DESI][2])

---

# 7. The readable 3D viewer

The viewer should not wait for perfect physical CMB. It should wait for:

```text
CHART_LORENTZ_H3_RECEIPT
OBJECT_CHART_RECEIPT
at least one of:
  OBJECT_BULK_POPULATION_RECEIPT
  or H3_DEFECT_WORLDLINE_RECEIPT as proto/demo
```

## Viewer data model

```json
{
  "chart": {
    "type": "H3_hyperboloid",
    "claim": "chart-level conformal H3, not by itself physical bulk"
  },
  "observers": [
    {
      "observer_id": 12,
      "h3_point": [x0, x1, x2, x3],
      "record_state": "...",
      "sector_class": "S3:transposition"
    }
  ],
  "objects": [
    {
      "object_id": 7,
      "h3_center": [x0, x1, x2, x3],
      "member_observers": [12, 43, 91],
      "persistence": 6,
      "bulk_population_receipt": true
    }
  ],
  "defects": [
    {
      "defect_id": 3,
      "class": "S3:threecycle",
      "worldline": [[t, x0, x1, x2, x3], "..."],
      "proto_particle": true
    }
  ],
  "fields": {
    "repair_load": "...",
    "record_signature": "...",
    "defect_density": "..."
  }
}
```

## Viewer pseudocode

```python
def write_h3_viewer(run, out_html):
    chart = load_h3_chart(run)
    observers = load_observer_h3_points(run)
    objects = load_object_h3_population(run)
    defects = load_h3_defect_worldlines(run)

    payload = {
        "chart": chart,
        "observers": observers,
        "objects": objects,
        "defects": defects,
        "receipts": load_receipts(run),
    }

    html = render_template("h3_viewer.html", payload=json.dumps(payload))
    Path(out_html).write_text(html)
```

Visual convention:

```text
gray dots: observer records
blue clusters: object normal forms
orange/purple curves: S3 defect worldlines
red flash: fusion/annihilation
transparent shell: S2 support chart
solid interior: H3 chart projection
```

---

# 8. Run plan

## Stage 0: reproducibility

```bash
python tools/check_handoff_bundle.py
python -m pytest -q
```

Required:

```text
all imports OK
all tests pass
```

## Stage 1: 4k conceptual runs

```bash
python -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_4k_modular_amount.yml \
  --seeds 20261001,20261002,20261003,20261004 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs/stage1_4k_h3_modular_amount
```

Pass:

```text
CHART_LORENTZ_H3_RECEIPT: 4/4
H3_RESPONSE_CANDIDATE_RECEIPT: >=3/4
wrong-scale 2pi wins: >=3/4
OBJECT_CHART_RECEIPT: >=3/4
```

Do not care yet about physical CMB.

## Stage 2: 64k confirmation

```bash
python -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_64k_modular_amount.yml \
  --seeds 20261011,20261012,20261013,20261014 \
  --workers 2 \
  --inner-jobs 1 \
  --out-dir runs/stage2_64k_h3_modular_amount
```

Pass:

```text
H3_RESPONSE_CANDIDATE_RECEIPT: >=3/4
OBJECT_CHART_RECEIPT: >=3/4
OBJECT_BULK_POPULATION_RECEIPT: >=1/4 or clear positive trend
```

## Stage 3: 256k confirmation

```bash
python -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_256k_modular_amount.yml \
  --seeds 20261021,20261022,20261023,20261024 \
  --workers 2 \
  --inner-jobs 1 \
  --out-dir runs/stage3_256k_h3_modular_amount
```

Pass:

```text
OBJECT_BULK_POPULATION_RECEIPT: >=2/4
localized_not_boundary_object_count > shuffled p95
H3 defects can be projected
```

## Stage 4: proto-particle run

```bash
python -m oph_fpe.cli run-defect-dynamics \
  --config configs/p1_s3_screen_defect_dynamics_64k.yml \
  --out-dir runs/stage4_s3_defect_dynamics
```

Pass:

```text
SCREEN_DEFECT_DYNAMICS_RECEIPT
fusion table reported
persistent worldlines reported
```

Then:

```bash
python -m oph_fpe.cli run-defect-h3-worldlines \
  --bulk-run runs/stage3_256k_h3_modular_amount/... \
  --defect-run runs/stage4_s3_defect_dynamics/... \
  --out-dir runs/stage5_h3_proto_particles
```

Pass:

```text
H3_DEFECT_WORLDLINE_RECEIPT
PROTO_PARTICLE_RECEIPT
```

## Stage 5: static galaxy data

```bash
python -m oph_fpe.cli run-galaxy-static \
  --dataset data/measurements/sparc \
  --config configs/d1_static_galaxy_rar_btfr.yml \
  --out-dir runs/stage5_static_galaxy_sparc
```

Pass:

```text
STATIC_GALAXY_RAR_BTFR_RECEIPT
```

## Stage 6: OPH Boltzmann kernels

```bash
python -m oph_fpe.cli derive-oph-boltzmann-kernels \
  --run-dirs runs/stage2_64k_h3_modular_amount runs/stage3_256k_h3_modular_amount \
  --out-dir runs/stage6_oph_kernels
```

Pass:

```text
OPH_BOLTZMANN_KERNEL_RECEIPT
```

Only then run physical CMB.

---

# 9. New config template

Create:

```text
configs/e4_shared_observer_bulk_4k_modular_amount.yml
configs/e4_shared_observer_bulk_64k_modular_amount.yml
configs/e4_shared_observer_bulk_256k_modular_amount.yml
```

Key changes:

```yaml
h3_modular_response:
  enabled: true
  observable_mode: perturb_resettle_transition
  times: [0.125, 0.25, 0.5]

  transition_observables:
    - checkpoint_class
    - stable_flag
    - record_family
    - s3_sector_class
    - repair_load_bucket

  transform: signed_zscore

  # Critical fix.
  perturb_budget_mode: modular_amount
  fixed_perturb_fraction: null

  perturb_selection_mode: lambda_displacement
  perturb_strength: 1.0

  wrong_scales:
    - 1.0
    - 3.141592653589793
    - 12.566370614359172

  transport_scale: 6.283185307179586

  fit_mode: joint_global
  control_fit_mode: same_h3_model_not_affine_target_fit
  candidate_count: 4096
  candidate_radius: 2.0
  softness: 0.25
  heldout_fraction: 0.25
  anchor_weight: 0.0
  max_iterations: 4
  pass_ratio: 1.0
  min_observers: 64
  min_features: 16
  feature_selection: change_probability_only
  min_feature_std: 0.0001

observer_objects:
  enabled: true
  family_mode: transition_affinity
  history_window: 4
  transition_history_fields:
    - record_family
    - checkpoint_class
    - stable_flag
    - s3_sector_class
    - repair_load_bucket
  transition_affinity_fields:
    - checkpoint_class
    - record_family
    - s3_sector_class
    - repair_load_bucket
    - cumulative_repair_load_bucket
  transition_bins: 8
  record_family_modulus: 16
  persistence_horizon: 8
  counterfactual_perturbations: 16

observer_chart_population:
  enabled: true
  incidence_mode: persistent_transition_response_cluster
  history_window: 4
  min_persistence: 3
  min_objects: 12
  min_observers_per_object: 6
  max_observer_fraction_per_object: 0.25
  max_h3_compactness: 0.35
  min_localized_objects: 3
  shuffle_control_count: 64
  pass_ratio: 1.0

neutral_reconstruction:
  enabled: true
  used_for_bulk_gate: leakage_audit_only
  require_candidate_3d_window: false
  forbid_radial_depth_evidence: true

cosmology:
  freezeout:
    enabled: true
    allow_screen_proxy_without_bulk: true
    require_neutral_reconstruction: false
  angular_power:
    ell_max: 64
    estimator: spherical_harmonic
```

---

# 10. What counts as “usable data”

## Usable now

```text
SCREEN_PROXY_CMB_RECEIPT
STATIC_GALAXY_RAR_BTFR_RECEIPT
```

The screen proxy is not physical CMB; it is a diagnostic. The static galaxy lane is the first real external-data lane.

## Usable next

```text
DYNAMIC_DARK_TRANSPORT_RECEIPT
OPH_BOLTZMANN_KERNEL_RECEIPT
```

## Usable physical CMB only after

```text
rho_A(a)
rho_A_eq(a)
Gamma_rec(k,a)
B_A(k,a)
CAMB/CLASS anomaly module
```

The latest handoff explicitly says the current OPH-CMB adapter is scaffolded but gated closed until those quantities exist. 

---

# 11. What counts as “readable 3D bulk with proto-particles”

Minimum viewer-ready status:

```text
CHART_LORENTZ_H3_RECEIPT = true
H3_RESPONSE_CANDIDATE_RECEIPT = true
OBJECT_CHART_RECEIPT = true
OBJECT_BULK_POPULATION_RECEIPT = true or demo-labeled partial
SCREEN_DEFECT_DYNAMICS_RECEIPT = true
H3_DEFECT_WORLDLINE_RECEIPT = true
```

Show this as:

```text
S2 screen shell
H3 interior chart
observer record points
object clusters
S3/Z2 defect worldlines
fusion/annihilation markers
```

Caption it carefully:

> “Proto-particle topological defects localized in the emergent H3 chart.”

Not:

> “Electrons/quarks/photons emerged.”

Standard Model particles come later; the particle paper is explicit that particle content is read after overlap consistency, compact gauge reconstruction, and MAR selection, and that massless structural carriers belong to the realized gauge/gravity branches, not the first defect toy lane. 

---

# 12. The coding-agent task list

Give this to the agent in order.

```text
TASK 0 — Reproducible bundle
- Add missing modules to handoff zip.
- Add tools/check_handoff_bundle.py.
- Verify pytest passes from clean unzip.

TASK 1 — Receipt freeze
- Add canonical receipts to claims.py.
- Update bw_array.py gate hierarchy.
- Stop using neutral dimension as bulk gate.
- Mark neutral reconstruction as leakage/debug only.

TASK 2 — Wrong-scale control fix
- Change configs from fixed_collar_fraction to modular_amount.
- Remove fixed_perturb_fraction.
- Change H3 wrong-scale scoring from affine target-fit to same-H3-model scale competition.
- Emit per cap/time/observable wrong-scale audit.

TASK 3 — Endogenous modular operator v2
- Build finite collar operator basis.
- Construct rho_C from non-diagonal Gram/covariance.
- Compute K_C = -log(rho_C+aI).
- Compare exp(iK_C t) with direct cap transport in same basis.
- Gate ENDOGENOUS_MODULAR_GENERATOR_RECEIPT separately from direct BW sanity.

TASK 4 — Object extraction v2
- Replace support/final histogram objects with persistent transition-response objects.
- Cluster observer histories and perturb-response spectra.
- Add object shuffle controls with 64 shuffles.
- Require localized_not_boundary_count > shuffled p95.

TASK 5 — Populated H3 viewer
- Use object centers from hyperbolic Fréchet means.
- Write viewer payload only after object chart passes.
- Add visual distinction between chart-level and populated-bulk claims.

TASK 6 — Defect dynamics
- Track Z2/S3 screen holonomy defects over cycles.
- Add persistence, transport, fusion, annihilation.
- Then map defects into H3 only after object bulk passes.

TASK 7 — Static galaxy data
- Implement nu_OPH(x).
- Add SPARC loader.
- Fit RAR/BTFR with shared a0/lambda and nuisance M/L.
- Emit STATIC_GALAXY_RAR_BTFR_RECEIPT.

TASK 8 — OPH Boltzmann kernel
- Derive B_A(k,a) from finite collar perturbation response.
- Implement anomaly fluid ODE.
- Only then wire CAMB/CLASS physical CMB.
```

---

# 13. The shortest path to success

The fastest route to something meaningful is:

```text
1. Fix wrong-scale controls.
2. Make object extraction persistent-transition based.
3. Get 4k H3 response to beat wrong-scale controls.
4. Get 64k object bulk population above shuffled p95.
5. Render H3 objects and S3 defect worldlines.
6. In parallel, fit SPARC RAR/BTFR.
```

That gives you:

```text
readable 3D bulk with proto-particles
+
real measurement-facing galaxy data
```

before attempting physical CMB.

[1]: https://arxiv.org/abs/1606.09251?utm_source=chatgpt.com "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves"
[2]: https://www.desi.lbl.gov/2024/04/04/desi-y1-results-april-4-guide/?utm_source=chatgpt.com "DESI 2024 Results: April 4 Guide"
