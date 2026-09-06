"""Queries on the byte-pinned Whitney charged observer execution.

The action and dressed scalar evaluator are imported explicitly from RER at
RER_COMMIT. Only named pure function definitions are loaded from authenticated
source bytes; importing this module executes no research source. Local point
geometry and density quadrature are implemented here and checked against the
recorded action integrals. Loading replays exact software events, not the ODE.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

RER_COMMIT = "c711eef134ea4290519759bc1524bad9f9004f75"
RECEIPT_PINS = {
    "charged_instrument": "d4a40859a5c50cbfc473f005b41c60548b3f873a7587b3340d71c52fbaf19d2a",
    "ephemeris_clock": "be97cacfdd4445b679301631695e37a8309a616e9a470957ebf928314ad991c4",
    "real_continuum": "9277a3053cde7ec08cb1efa0184cf94260dcdce4bded94c3525aaf631362da13",
    "quantum_history": "1a2b9a1afe2ba32cbb551352c1ca3bc4d45251921268feab6fdcee7749587bbe",
}
SYSTEM_ID = "whitney_charged_observer_execution_v1"
ACTION_SOURCE = "code/electromagnetism/verify_whitney_charged_dynamics.py"
EVENT_SOURCE = "code/electromagnetism/verify_whitney_charged_instrument.py"
PAIRS = tuple(combinations(range(4), 2))
ENERGY_ATOL = 2e-10
ENERGY_RTOL = 2e-10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def _exact(actual: Any, expected: Any, label: str) -> None:
    _require(_canonical(actual) == _canonical(expected), label)


def _strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def floating(value):
        result = float(value)
        _require(math.isfinite(result), "nonfinite JSON number")
        return result
    def constant(_):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
        parse_float=floating, parse_constant=constant)
    _require(type(value) is dict, "receipt must be a JSON object")
    return value


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.PIPE, text=True, encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("a local pinned RER checkout is required") from exc


def _source_functions(raw: bytes, path: str, names: Sequence[str], namespace: dict[str, Any]) -> SimpleNamespace:
    """Compile only named function bodies from already verified source bytes.

    The reviewed functions are pure array/event calculations; module imports,
    file operations, module-level statements and CLI bodies are not executed.
    This is a pinned adapter, not an arbitrary-source plugin interface.
    """
    tree = ast.parse(raw.decode("utf-8"), filename=path)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    _require({node.name for node in selected} == set(names), "pinned action function inventory")
    context = {"__builtins__": __builtins__, **namespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), context)
    return SimpleNamespace(**{name: context[name] for name in names})


def _index(value: int, count: int, label: str) -> int:
    _require(type(value) is int and 0 <= value < count, f"{label} must be an integer in [0,{count})")
    return value


def _barycentric(values: Sequence[float]) -> np.ndarray:
    _require(isinstance(values, (list, tuple, np.ndarray)), "four barycentric coordinates required")
    _require(len(values) == 4, "four barycentric coordinates required")
    _require(all(not isinstance(x, (bool, np.bool_)) and isinstance(x, (int, float, Fraction, np.integer, np.floating))
                 for x in values), "barycentric coordinates must be real numbers")
    result = np.asarray(values, dtype=float)
    _require(result.shape == (4,) and np.isfinite(result).all(), "finite barycentric coordinates required")
    _require(bool(np.all(result >= 0) and np.all(result <= 1)), "point lies outside tetrahedron")
    _require(abs(math.fsum(result)-1) <= 8*np.finfo(float).eps, "barycentric coordinates must sum to one")
    return result


def _complex(value: Any) -> list[Any]:
    array = np.asarray(value)
    return np.stack((array.real, array.imag), axis=-1).tolist()


class WhitneySystem:
    """An authenticated imported execution, queried in its supplied model units.

    Construct with ``load``. Results are independent copies; no method mutates
    the source execution or extends its recorded time interval.
    """

    @classmethod
    def load(cls, rer_root: str | Path | None = None) -> "WhitneySystem":
        root = Path(rer_root) if rer_root is not None else Path(__file__).resolve().parents[3]/"reverse-engineering-reality"
        root = root.resolve()
        _revision(root)  # Require a real checkout; all consumed bytes are pinned below.
        packets, authenticated = {}, {}
        for stem, digest in RECEIPT_PINS.items():
            path = f"code/electromagnetism/runtime/whitney_{stem}_receipt.json"
            raw = _read_bytes(root/path)
            _require(hashlib.sha256(raw).hexdigest() == digest, f"receipt SHA256 mismatch: {stem}")
            packets[stem] = _strict_json(raw)
            authenticated[path] = raw
        for packet in packets.values():
            for relative, digest in packet["source_pins"].items():
                path = PurePosixPath(relative)
                _require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative, "unsafe source pin path")
                raw = _read_bytes(root/relative)
                _require(hashlib.sha256(raw).hexdigest() == digest, f"source pin mismatch: {relative}")
                authenticated[relative] = raw
        event = _source_functions(authenticated[EVENT_SOURCE], EVENT_SOURCE,
            ("require", "canonical", "exact", "keys", "fraction", "expected_contract", "replay_events"),
            {"np": np, "Q": Fraction, "hashlib": hashlib, "json": json})
        packet = packets["charged_instrument"]
        _exact(packet["contract"], event.expected_contract(), "charged action contract mismatch")
        decoded, advances, event_root = event.replay_events(packet["events"])
        _exact(packet["event_root"], event_root, "event root mismatch")
        _require(len(packet["frames"]) == len(decoded) == 81 and len(advances) == 80, "execution census mismatch")
        for frame, header in zip(packet["frames"], decoded, strict=True):
            _exact({key: frame[key] for key in header}, header, "decoded frame provenance mismatch")
            def expand(values):
                x = [float(Fraction(v)) for v in values]
                return [x[0]]*12+[0.0]*30+[x[1]]+[x[3]]*12+[x[2]]+[x[4]]*12
            _exact(frame["configuration"], expand(frame["q_exact"]), "full field configuration does not match decoded coordinates")
            _exact(frame["velocity"], expand(frame["velocity_exact"]), "full field velocity does not match decoded coordinates")
        clock = packets["ephemeris_clock"]
        _exact(clock["source"]["consumed_fields"], ["frames[].q_exact"], "clock input boundary")
        _require(clock["source"]["sha256"] == RECEIPT_PINS["charged_instrument"], "clock belongs to another execution")
        _require(len(clock["cumulative_clock"]) == len(decoded), "clock frame census")
        self = cls.__new__(cls)
        self._packets = packets
        self._action = _source_functions(authenticated[ACTION_SOURCE], ACTION_SOURCE, ("element_fields",), {"np": np})
        self._source_hashes = {path: hashlib.sha256(raw).hexdigest() for path, raw in authenticated.items()}
        self._elements = self._make_elements(packet["mesh"])
        self._quadrature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._event_ids = {(e["op"], tuple(e["args"])): e["id"] for e in packet["events"]}
        return self

    def _provenance(self) -> dict[str, Any]:
        return {"rer_commit": RER_COMMIT, "receipt_sha256": dict(RECEIPT_PINS),
            "equation_source": ACTION_SOURCE, "equation_source_sha256": self._source_hashes[ACTION_SOURCE],
            "equation_function": "element_fields", "observer_replay_source": EVENT_SOURCE,
            "observer_replay_source_sha256": self._source_hashes[EVENT_SOURCE],
            "imported_action": True, "simulator_trajectory_executed": False,
            "exact_observer_events_replayed": True, "ode_reintegrated_by_query_loader": False,
            "physical_measurement_or_prediction": False}

    def describe(self) -> dict[str, Any]:
        packet = self._packets["charged_instrument"]
        return {"schema": "oph.physical_system_descriptor.v1", "system_id": SYSTEM_ID,
            "scope": "QUERYABLE_IMPORTED_FINITE_CHARGED_ACTION__MODEL_UNITS",
            "patches": 5, "frames": 81, "events": 1782, "cells": 20,
            "contract": deepcopy(packet["contract"]), "provenance": self._provenance(),
            "available_queries": ["mesh", "observer", "event", "field", "energy", "clock", "real_continuum", "quantum"],
            "same_quantum_classical_history": False, "physical_completion": False}

    def mesh(self) -> dict[str, Any]:
        return {**deepcopy(self._packets["charged_instrument"]["mesh"]), "provenance": self._provenance()}

    def event(self, event_id: int) -> dict[str, Any]:
        events = self._packets["charged_instrument"]["events"]
        return {"schema": "oph.physical_system_event_query.v1", "system_id": SYSTEM_ID,
            "event": deepcopy(events[_index(event_id, len(events), "event id")]), "provenance": self._provenance()}

    def _frame(self, frame_index: int) -> dict[str, Any]:
        frames = self._packets["charged_instrument"]["frames"]
        return frames[_index(frame_index, len(frames), "frame index")]

    def observer(self, patch_id: int, frame_index: int) -> dict[str, Any]:
        patch_id = _index(patch_id, 5, "patch id")
        frame = self._frame(frame_index)
        previous, following = (patch_id-1) % 5, (patch_id+1) % 5
        records = {}
        addresses = {"baseline": ("baseline", (frame_index, patch_id)),
            "outgoing_response": ("response", (frame_index, patch_id, following)),
            "outgoing_feedback": ("feedback", (frame_index, patch_id, following)),
            "incoming_baseline": ("baseline", (frame_index, previous)),
            "incoming_response": ("response", (frame_index, previous, patch_id)),
            "decode": ("decode", (frame_index,))}
        for name, address in addresses.items():
            records[name] = deepcopy(self._packets["charged_instrument"]["events"][self._event_ids[address]])
        return {"schema": "oph.physical_system_observer_query.v1", "system_id": SYSTEM_ID,
            "patch_id": patch_id, "frame_index": frame_index,
            "coordinate": self._packets["charged_instrument"]["contract"]["patch_coordinates"][patch_id],
            "local_state": {"q_exact": frame["q_exact"][patch_id], "v_exact": frame["velocity_exact"][patch_id]},
            "ports": {"incoming": [previous, patch_id], "outgoing": [patch_id, following]},
            "records": records, "decode_event_id": frame["decode_event_id"],
            "scope": "computational patch; whole-field queries use all five decoded patches",
            "units": self._packets["charged_instrument"]["contract"]["units"], "provenance": self._provenance()}

    @staticmethod
    def _make_elements(mesh: dict[str, Any]) -> list[dict[str, Any]]:
        vertices = np.asarray(mesh["vertices"], dtype=float)
        edges = {tuple(pair): i for i, pair in enumerate(mesh["edges"])}
        result = []
        for cell in mesh["tetrahedra"]:
            xyz = vertices[cell]
            gradients = np.linalg.inv(np.column_stack((np.ones(4), xyz)))[1:].T
            determinant = float(abs(np.linalg.det(xyz[1:]-xyz[0])))
            ids, signs = [], []
            for i, j in PAIRS:
                pair = (cell[i], cell[j]); sign = 1 if pair in edges else -1
                ids.append(edges[pair if sign == 1 else pair[::-1]]); signs.append(sign)
            path_grad = np.zeros((4, 6, 3))
            for edge, (i, j) in enumerate(PAIRS):
                path_grad[i, edge], path_grad[j, edge] = gradients[j], -gradients[i]
            result.append({"tet": np.asarray(cell), "ids": np.asarray(ids), "signs": np.asarray(signs),
                "grad": gradients, "grad_theta_jac": path_grad,
                "curls": np.asarray([2*np.cross(gradients[i], gradients[j]) for i, j in PAIRS]),
                "points": xyz, "determinant": determinant})
        return result

    @staticmethod
    def _at(element: dict[str, Any], barycentric: np.ndarray) -> dict[str, Any]:
        lam = np.atleast_2d(barycentric)
        gradients = element["grad"]
        basis = np.stack([lam[:, i, None]*gradients[j]-lam[:, j, None]*gradients[i] for i, j in PAIRS], axis=1)
        paths = np.zeros((len(lam), 4, 6))
        for edge, (i, j) in enumerate(PAIRS):
            paths[:, i, edge], paths[:, j, edge] = lam[:, j], -lam[:, i]
        return {**element, "lam": lam, "basis": basis, "theta_jac": paths}

    def _densities(self, frame: dict[str, Any], element: dict[str, Any]) -> dict[str, Any]:
        q, velocity = np.asarray(frame["configuration"]), np.asarray(frame["velocity"])
        charge, mass, coupling = .25, .5, .25
        field = self._action.element_fields(q, velocity, element, charge=charge)
        local_a = q[element["ids"]]*element["signs"]
        local_v = velocity[element["ids"]]*element["signs"]
        electric = -np.einsum("e,qec->qc", local_v, element["basis"])
        magnetic = np.broadcast_to(local_a@element["curls"], electric.shape)
        potential = np.einsum("e,qec->qc", local_a, element["basis"])
        squared = abs(field["scalar"])**2
        density = {"electric": np.sum(electric**2, axis=1)/2, "magnetic": np.sum(magnetic**2, axis=1)/2,
            "matter_time": abs(field["scalar_dot"])**2, "matter_space": np.sum(abs(field["spatial"])**2, axis=1),
            "mass_term": mass*squared, "quartic_term": coupling*squared**2/2}
        density["kinetic"] = density["electric"]+density["matter_time"]
        density["potential"] = density["magnetic"]+density["matter_space"]+density["mass_term"]+density["quartic_term"]
        density["total"] = density["kinetic"]+density["potential"]
        density["lagrangian"] = density["kinetic"]-density["potential"]
        _require(all(np.isfinite(x).all() for x in density.values()), "nonfinite action density")
        return {"field": field, "electric": electric, "magnetic": magnetic, "vector_potential": potential,
            "charge_density": 2*charge*np.imag(field["scalar"].conj()*field["scalar_dot"]), "density": density}

    def field(self, frame_index: int, cell_index: int, barycentric: Sequence[float]) -> dict[str, Any]:
        frame = self._frame(frame_index)
        element = self._elements[_index(cell_index, len(self._elements), "cell index")]
        lam = _barycentric(barycentric)
        sample = self._densities(frame, self._at(element, lam))
        field = sample["field"]
        return {"schema": "oph.physical_system_field_query.v1", "system_id": SYSTEM_ID,
            "frame_index": frame_index, "cell_index": cell_index, "barycentric": lam.tolist(),
            "position": (lam@element["points"]).tolist(), "decode_event_id": frame["decode_event_id"],
            "action_time_exact": frame["action_time_exact"],
            "scalar": _complex(field["scalar"][0]),
            "covariant_time_derivative": _complex(field["scalar_dot"][0]),
            "covariant_spatial_derivative": _complex(field["spatial"][0]),
            "electric": sample["electric"][0].tolist(), "magnetic": sample["magnetic"][0].tolist(),
            "vector_potential": sample["vector_potential"][0].tolist(), "scalar_potential": 0.0,
            "charge_load_density": float(sample["charge_density"][0]),
            "charge_convention": "2e Im(conj(Psi) DtPsi), the source action's nodal rho_load convention",
            "energy_density": {key: float(value[0]) for key, value in sample["density"].items()},
            "units": "supplied model fields; action-energy per model volume; no SI calibration",
            "mass_interpretation": "m_squared=1/2 is an imported action parameter; no particle mass or mass-density identification",
            "gauge": "temporal phi=0; potential-dressed nodal scalar and oriented Whitney one-forms",
            "spatial_evaluation": "finite-element evaluation at the requested point; no temporal interpolation",
            "provenance": self._provenance()}

    def _rule(self, order: int) -> tuple[np.ndarray, np.ndarray]:
        _require(type(order) is int and 4 <= order <= 8, "quadrature order must be an integer from 4 through 8")
        if order not in self._quadrature_cache:
            points, weights = np.polynomial.legendre.leggauss(order)
            points, weights = (points+1)/2, weights/2
            lam, volume = [], []
            for i, j, k in product(range(order), repeat=3):
                x, y, z = points[i], points[j], points[k]
                lam.append([(1-x)*(1-y)*(1-z), x, (1-x)*y, (1-x)*(1-y)*z])
                volume.append(weights[i]*weights[j]*weights[k]*(1-x)**2*(1-y))
            self._quadrature_cache[order] = np.asarray(lam), np.asarray(volume)
        return self._quadrature_cache[order]

    def energy(self, frame_index: int, quadrature_order: int = 4) -> dict[str, Any]:
        frame = self._frame(frame_index)
        lam, weights = self._rule(quadrature_order)
        cells = []
        for cell_index, element in enumerate(self._elements):
            densities = self._densities(frame, self._at(element, lam))["density"]
            integrals = {key: float(element["determinant"]*(weights@value)) for key, value in densities.items()}
            cells.append({"cell_index": cell_index, "volume": element["determinant"]/6, "integrals": integrals})
        totals = {key: math.fsum(row["integrals"][key] for row in cells) for key in cells[0]["integrals"]}
        errors = {}
        for key, source in (("total", "energy"), ("kinetic", "kinetic_energy"), ("potential", "potential_energy"), ("lagrangian", "lagrangian")):
            errors[key] = abs(totals[key]-frame[source])
            _require(math.isfinite(totals[key]) and errors[key] <= ENERGY_ATOL+ENERGY_RTOL*abs(frame[source]),
                f"density-integrated {key} disagrees with recorded action")
        return {"schema": "oph.physical_system_energy_query.v1", "system_id": SYSTEM_ID,
            "frame_index": frame_index, "decode_event_id": frame["decode_event_id"],
            "action_time_exact": frame["action_time_exact"], "cells": cells, "totals": totals,
            "quadrature_order": quadrature_order, "quadrature": "Duffy product Gauss; degree-four density after symmetry phase cancellation",
            "recorded_action_comparison": {"passed": True, "absolute_errors": errors, "atol": ENERGY_ATOL, "rtol": ENERGY_RTOL},
            "units": "supplied dimensionless action energy; no SI calibration",
            "numerical_error_enclosure": False, "provenance": self._provenance()}

    def clock(self, frame_index: int) -> dict[str, Any]:
        frame = self._frame(frame_index)
        clock = self._packets["ephemeris_clock"]
        return {"schema": "oph.physical_system_clock_query.v1", "system_id": SYSTEM_ID,
            "frame_index": frame_index, "action_time_exact": frame["action_time_exact"],
            "ephemeris_model_time": clock["cumulative_clock"][frame_index],
            "completed_repair_cycles": frame["completed_repair_cycles"],
            "contract": deepcopy(clock["contract"]), "clock_scope": clock["scope"],
            "readout_recomputed_by_query": False, "physical_clock_calibrated": False,
            "units": "distinct model action/ephemeris times and software event counter", "provenance": self._provenance()}

    def real_continuum(self) -> dict[str, Any]:
        packet = self._packets["real_continuum"]
        return {"schema": "oph.physical_system_real_continuum_descriptor.v1",
            **{key: deepcopy(packet[key]) for key in ("scope", "analytic_scope", "mesh_checks", "uniform_bounds", "trajectories")},
            "intermediate_history_provided": False, "same_charged_trajectory": False,
            "units": "separate supplied real-sector model coordinates and time", "provenance": self._provenance()}

    def quantum(self) -> dict[str, Any]:
        packet = self._packets["quantum_history"]
        return {"schema": "oph.physical_system_quantum_descriptor.v1",
            **{key: deepcopy(packet[key]) for key in ("scope", "state", "parameters", "phase_configurations", "history", "error_certificate", "integration")},
            "same_classical_history": False, "continuum_QFT": False,
            "units": packet["error_certificate"]["time_units"], "provenance": self._provenance()}
