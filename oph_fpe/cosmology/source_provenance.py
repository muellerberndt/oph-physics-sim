from __future__ import annotations

from typing import Any


PROMOTED_CMB_SOURCE_QUANTITIES = (
    "eta_R",
    "Gamma_rec",
    "A_zeta",
    "q_IR",
    "ell_IR",
    "B_A_k_a",
    "rho_A_a",
    "N_CRC",
)

FINITE_CMB_PROVENANCE_SOURCES = {
    "finite_lattice",
    "finite_repair_transition_clock",
    "parent_collar_finite_difference",
    "neutral_bulk_freezeout",
    "scale_compressed_24_round_finite_ladder",
}

THEOREM_SIDE_PROVENANCE_SOURCES = {
    "OPH_pixel_branch_predeclared",
    "OPH_direct_public_record_capacity",
    "OPH_independent_scale_bridge_supplied",
}

FORBIDDEN_SOURCE_KINDS = {
    "diagnostic_proxy",
    "measurement",
    "measurement_fit",
    "measurement_calibrated",
    "residual_fit",
    "likelihood",
    "cmb_likelihood_fit",
    "posterior",
    "posterior_selected",
    "fitted_parameter",
    "shard_local_average",
    "observed_horizon_comparison",
    "OPH_screen_capacity_observed_branch_readout",
    "OPH_screen_capacity_branch_predeclared",
    "electroweak_bridge_comparison",
    "operational_resolution_comparison",
}

POOLED_REDUCER_CHECKS = (
    "units_validated",
    "coordinate_grid_validated",
    "coverage_validated",
    "duplicates_checked",
    "interpolation_policy_frozen",
    "covariance_validated",
)

TRANSITIVE_SOURCE_ANCESTRY_RECEIPT = "TRANSITIVE_SOURCE_ANCESTRY_RECEIPT"
HERMETIC_READ_SET_RECEIPT = "HERMETIC_READ_SET_RECEIPT"
SOURCE_MODEL_FREEZE_RECEIPT = "SOURCE_MODEL_FREEZE_RECEIPT"


def certify_cmb_source_provenance(
    nodes: list[dict[str, Any]],
    reducers: dict[str, dict[str, Any]],
    *,
    global_checks: dict[str, Any] | None = None,
    required_quantities: tuple[str, ...] = PROMOTED_CMB_SOURCE_QUANTITIES,
) -> dict[str, Any]:
    """Certify source-only provenance for promoted CMB input quantities.

    This receipt is intentionally upstream of likelihood evaluation. It says
    whether OPH source artifacts and reducer metadata are clean enough to enter
    the physical-CMB input contract; it does not claim that the later Boltzmann
    or likelihood gates have passed.
    """

    global_checks = global_checks or {}
    blockers: list[str] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    quantity_nodes: dict[str, list[dict[str, Any]]] = {quantity: [] for quantity in required_quantities}

    for index, node in enumerate(nodes):
        node_id = _node_id(node, index)
        if node_id in node_by_id:
            blockers.append(f"duplicate_provenance_node:{node_id}")
            continue
        node_by_id[node_id] = node
        quantity = str(node.get("quantity", ""))
        if quantity in quantity_nodes:
            quantity_nodes[quantity].append(node)

    blockers.extend(_graph_blockers(node_by_id))

    contradiction_free = True
    transitive_ancestry = True
    source_only_status: dict[str, Any] = {}
    for quantity, rows in quantity_nodes.items():
        if not rows:
            blockers.append(f"{quantity}_provenance_node_missing")
            source_only_status[quantity] = {"present": False, "source_only": False}
            continue
        if len(rows) > 1:
            blockers.append(f"{quantity}_ambiguous_provenance_nodes")
        row = rows[0]
        node_id = _node_id(row, 0)
        ancestry = _transitive_ancestry_status(node_id, node_by_id)
        if ancestry["blockers"]:
            contradiction_free = False
            transitive_ancestry = False
            blockers.extend(ancestry["blockers"])
        source_ok = _source_allowed_for_quantity(quantity, str(row.get("source", row.get("source_kind", ""))))
        source_kind = str(row.get("source_kind", row.get("source", "")))
        if source_kind in FORBIDDEN_SOURCE_KINDS or str(row.get("source", "")) in FORBIDDEN_SOURCE_KINDS:
            source_ok = False
            blockers.append(f"{quantity}_forbidden_source_kind:{source_kind}")
        if not bool(row.get("source_only", False)):
            source_ok = False
            blockers.append(f"{quantity}_not_source_only")
        if not ancestry["source_only"]:
            source_ok = False
        source_only_status[quantity] = {
            "present": True,
            "source": row.get("source"),
            "source_report": row.get("source_report"),
            "source_only": source_ok,
            "node_id": node_id,
            "ancestry": ancestry,
        }
        if not source_ok:
            blockers.append(f"{quantity}_not_source_derived")

    reducer_status, reducer_blockers = _reducer_status(required_quantities, reducers)
    blockers.extend(reducer_blockers)

    n_crc_status = reducer_status.get("N_CRC") or _n_crc_status({})
    if not n_crc_status["receipt"]:
        blockers.append("N_CRC_direct_public_record_capacity_receipt_missing")

    global_status, global_blockers = _global_likelihood_status(global_checks)
    blockers.extend(global_blockers)
    hermetic_read_set = bool(global_checks.get(HERMETIC_READ_SET_RECEIPT, global_checks.get("hermetic_read_set_receipt", False)))
    source_model_freeze = bool(global_checks.get(SOURCE_MODEL_FREEZE_RECEIPT, global_checks.get("source_model_freeze_receipt", False)))
    if not hermetic_read_set:
        blockers.append("hermetic_read_set_receipt_missing")
    if not source_model_freeze:
        blockers.append("source_model_freeze_receipt_missing")

    blockers = _unique_strings(blockers)
    reducer_receipt = all(bool(row.get("receipt", False)) for row in reducer_status.values())
    receipt = len(blockers) == 0
    return {
        "mode": "cmb_source_provenance_certificate_v0",
        "CMB_SOURCE_PROVENANCE_RECEIPT": receipt,
        "source_provenance_receipt": receipt,
        TRANSITIVE_SOURCE_ANCESTRY_RECEIPT: bool(transitive_ancestry),
        HERMETIC_READ_SET_RECEIPT: hermetic_read_set,
        SOURCE_MODEL_FREEZE_RECEIPT: source_model_freeze,
        "pooled_source_reducer_receipt": reducer_receipt,
        "contradiction_free_provenance_receipt": contradiction_free,
        "N_CRC_direct_public_record_capacity_receipt": n_crc_status["receipt"],
        # Compatibility field consumed by the older physical-CMB dataclass.
        # Its semantics are now the strict direct-public-record receipt above.
        "N_CRC_consensus_invariant_receipt": n_crc_status["receipt"],
        "global_likelihood_reduction_receipt": global_status["receipt"],
        "blockers": blockers,
        "required_quantities": list(required_quantities),
        "quantity_status": source_only_status,
        "reducer_status": reducer_status,
        "N_CRC_status": n_crc_status,
        "global_likelihood_status": global_status,
        "claim_boundary": (
            "Source-provenance receipt for promoted CMB inputs. It certifies a source-only "
            "dependency DAG and global pooled reducers for eta_R, Gamma_rec, A_zeta, q_IR, "
            "ell_IR, B_A(k,a), and rho_A(a). N_CRC additionally requires a target-free, "
            "complete public-record fiber with common M_0=alpha(G_q) and robust closure. "
            "Observed-horizon, electroweak, and operational-resolution comparisons cannot be producers."
        ),
    }


def _graph_blockers(node_by_id: dict[str, dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    for node_id, node in node_by_id.items():
        for parent in _parents(node):
            if parent not in node_by_id:
                blockers.append(f"missing_provenance_parent:{node_id}->{parent}")

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            blockers.append(f"cyclic_provenance_dependency:{node_id}")
            return
        visiting.add(node_id)
        for parent in _parents(node_by_id[node_id]):
            if parent in node_by_id:
                visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_by_id:
        visit(node_id)
    return blockers


def _transitive_ancestry_status(node_id: str, node_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    visited: set[str] = set()
    source_only = True

    def visit(current_id: str, path: list[str]) -> None:
        nonlocal source_only
        if current_id in visited:
            return
        visited.add(current_id)
        node = node_by_id[current_id]
        local = _local_source_blockers(current_id, node)
        if local:
            source_only = False
            path_text = "->".join(path + [current_id])
            blockers.extend(local)
            blockers.extend(f"{blocker}|path:{path_text}" for blocker in local)
        source_kind = str(node.get("source_kind", node.get("source", "")))
        if source_kind in FORBIDDEN_SOURCE_KINDS or str(node.get("source", "")) in FORBIDDEN_SOURCE_KINDS:
            source_only = False
            blockers.append(f"forbidden_source_ancestor:{current_id}:{source_kind}|path:{'->'.join(path + [current_id])}")
        if not bool(node.get("source_only", False)):
            source_only = False
            blockers.append(f"ancestor_not_source_only:{current_id}|path:{'->'.join(path + [current_id])}")
        for parent in _parents(node):
            if parent in node_by_id:
                visit(parent, path + [current_id])

    if node_id not in node_by_id:
        return {"source_only": False, "ancestor_ids": [], "blockers": [f"missing_ancestry_root:{node_id}"]}
    visit(node_id, [])
    return {"source_only": source_only, "ancestor_ids": sorted(visited), "blockers": _unique_strings(blockers)}


def _local_source_blockers(node_id: str, node: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    no_cmb_data_used = bool(node.get("no_cmb_data_used", False))
    measurement_flags = (
        "fit_to_planck",
        "fit_to_measurement",
        "measurement_data_used",
        "cmb_data_used",
        "cmb_data_used_for_input",
        "planck_data_used_for_input",
        "uses_measurements_to_set_inputs",
    )
    measurement_used = any(bool(node.get(flag, False)) for flag in measurement_flags)
    if no_cmb_data_used and measurement_used:
        blockers.append(f"contradictory_no_data_use_provenance:{node_id}")
    if not no_cmb_data_used:
        blockers.append(f"cmb_data_use_not_ruled_out:{node_id}")
    return blockers


def _source_allowed_for_quantity(quantity: str, source: str) -> bool:
    if quantity == "N_CRC":
        return source == "OPH_direct_public_record_capacity"
    return source in FINITE_CMB_PROVENANCE_SOURCES


def _reducer_status(
    required_quantities: tuple[str, ...],
    reducers: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    status: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for quantity in required_quantities:
        reducer = reducers.get(quantity) or {}
        if not reducer:
            blockers.append(f"{quantity}_source_reducer_missing")
            status[quantity] = {"receipt": False, "mode": "missing"}
            continue
        if quantity == "N_CRC":
            n_crc = _n_crc_status(reducer)
            status[quantity] = {
                **n_crc,
                "mode": reducer.get("mode", "unknown"),
            }
            continue
        if bool(reducer.get("shard_local_nonlinear_average", False)):
            blockers.append(f"{quantity}_shard_local_nonlinear_average_forbidden")
        mode = str(reducer.get("mode", ""))
        if mode == "single_global_source" and bool(reducer.get("single_global_source", False)):
            receipt = not bool(reducer.get("shard_local_nonlinear_average", False))
        elif mode == "pooled_sufficient_statistics" and bool(reducer.get("pooled_sufficient_statistics", False)):
            missing = [key for key in POOLED_REDUCER_CHECKS if not bool(reducer.get(key, False))]
            for key in missing:
                blockers.append(f"{quantity}_reducer_{key}_missing")
            receipt = not missing and not bool(reducer.get("shard_local_nonlinear_average", False))
        else:
            receipt = False
            blockers.append(f"{quantity}_source_reducer_not_pooled_or_global")
        status[quantity] = {
            "receipt": receipt,
            "mode": mode or "unknown",
            "single_global_source": bool(reducer.get("single_global_source", False)),
            "pooled_sufficient_statistics": bool(reducer.get("pooled_sufficient_statistics", False)),
        }
    return status, blockers


def _n_crc_status(reducer: dict[str, Any]) -> dict[str, Any]:
    exact_evaluator = bool(reducer.get("exact_public_record_capacity_evaluator", False))
    complete_fiber = bool(reducer.get("complete_terminal_fiber_receipt", False))
    common_readback = bool(reducer.get("whole_fiber_scalarization_receipt", False))
    target_free = bool(reducer.get("target_free_capacity_producer_receipt", False))
    robust = bool(reducer.get("robust_closure_receipt", False))
    unique_slack = bool(reducer.get("unique_regulator_stable_slack_zero_receipt", False))
    horizon_saturation = bool(reducer.get("horizon_record_saturation_receipt", False))
    physical_n = bool(reducer.get("physical_N_closure_receipt", False))
    return {
        "receipt": bool(
            exact_evaluator
            and complete_fiber
            and common_readback
            and target_free
            and robust
            and unique_slack
            and horizon_saturation
            and physical_n
        ),
        "exact_public_record_capacity_evaluator": exact_evaluator,
        "complete_terminal_fiber_receipt": complete_fiber,
        "whole_fiber_scalarization_receipt": common_readback,
        "target_free_capacity_producer_receipt": target_free,
        "robust_closure_receipt": robust,
        "unique_regulator_stable_slack_zero_receipt": unique_slack,
        "horizon_record_saturation_receipt": horizon_saturation,
        "physical_N_closure_receipt": physical_n,
        "claim_boundary": (
            "N_CRC is eligible only from a target-free complete public-record terminal fiber "
            "with common M_0=alpha(G_q), robust F_set(D)={D}, a unique regulator-stable slack zero, "
            "and the independent horizon-record saturation receipt. Consensus or additive counts alone fail."
        ),
    }


def _global_likelihood_status(global_checks: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    official_rollup = str(global_checks.get("official_likelihood_rollup", "missing"))
    cdm_rollup = str(global_checks.get("cdm_limit_rollup", "missing"))
    if official_rollup in {"shard_any", "any"}:
        blockers.append("official_likelihood_shard_any_rollup_forbidden")
    if cdm_rollup in {"shard_any", "any"}:
        blockers.append("cdm_limit_shard_any_rollup_forbidden")
    if official_rollup != "global":
        blockers.append("official_likelihood_global_rollup_missing")
    if cdm_rollup != "global":
        blockers.append("cdm_limit_global_rollup_missing")
    receipt = not blockers
    return (
        {
            "receipt": receipt,
            "official_likelihood_rollup": official_rollup,
            "cdm_limit_rollup": cdm_rollup,
            "claim_boundary": (
                "Official likelihood and CDM-limit readiness may be false, but their "
                "reduction status must be global rather than a shard-local any() rollup."
            ),
        },
        blockers,
    )


def _node_id(node: dict[str, Any], index: int) -> str:
    return str(node.get("node_id") or node.get("id") or node.get("quantity") or f"node_{index}")


def _parents(node: dict[str, Any]) -> list[str]:
    parents = node.get("parents") or []
    if isinstance(parents, str):
        return [parents]
    if isinstance(parents, list):
        return [str(parent) for parent in parents]
    return []


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out
