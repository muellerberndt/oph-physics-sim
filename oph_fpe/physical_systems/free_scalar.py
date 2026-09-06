"""All-mode coherent states of a declared free real scalar on a finite mesh.

This changes the charged action to e=g=0 and takes u=sqrt(2) Re(Psi),
an independent tensor factor of the decoupled complex field. It is not a
quantum restriction to the interacting classical symmetry submanifold.
Canonical quantization, hbar, the mesh and model time are supplied.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.linalg import eigh

from .whitney import _barycentric, _index, _require


class FreeScalar:
    """L2(R^N,dQ), equivalently bosonic Fock space over all N FE modes.

    No occupation cutoff is used. Coherent states are represented by N complex
    amplitudes; that representation is not a finite-dimensional Hilbert space.
    Numerical diagonalization is checked, not an interval error certificate.
    """

    def __init__(self, mesh: dict, *, mass_squared: float = .5, hbar: float = 1.):
        self.mass_squared = self._positive(mass_squared, "mass squared")
        self.hbar = self._positive(hbar, "hbar")
        self.vertices = np.asarray(mesh["vertices"], dtype=float)
        raw_cells = np.asarray(mesh["tetrahedra"])
        _require(raw_cells.ndim == 2 and raw_cells.shape[1] == 4 and raw_cells.dtype.kind in "iu",
                 "integer tetrahedron indices required")
        self.cells = raw_cells.astype(int, copy=True)
        n = len(self.vertices)
        _require(self.vertices.shape == (n, 3) and 4 <= n <= 512 and np.isfinite(self.vertices).all(),
                 "finite three-dimensional mesh with 4 through 512 nodes required")
        _require(len(self.cells) > 0 and self.cells.min() >= 0 and self.cells.max() < n,
                 "tetrahedron index outside mesh")
        _require(len({tuple(sorted(c)) for c in self.cells}) == len(self.cells), "duplicate tetrahedron")
        self.mass = np.zeros((n, n)); self.stiffness = np.zeros((n, n))
        gradients, volumes = [], []
        for cell in self.cells:
            xyz = self.vertices[cell]
            volume = abs(np.linalg.det(xyz[1:]-xyz[0]))/6
            _require(math.isfinite(volume) and volume > 0, "degenerate tetrahedron")
            grad = np.linalg.inv(np.column_stack((np.ones(4), xyz)))[1:].T
            self.mass[np.ix_(cell, cell)] += volume*(np.ones((4, 4))+np.eye(4))/20
            self.stiffness[np.ix_(cell, cell)] += volume*(grad@grad.T)
            gradients.append(grad); volumes.append(volume)
        self.gradients, self.volumes = np.array(gradients), np.array(volumes)
        self.operator = self.stiffness+self.mass_squared*self.mass
        eigenvalues, self.modes = eigh(self.operator, self.mass)
        _require(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0, "positive massive spectrum required")
        self.omega = np.sqrt(eigenvalues)
        self.defects = {
            "mass_orthonormality": float(np.max(abs(self.modes.T@self.mass@self.modes-np.eye(n)))),
            "relative_eigen_residual": float(np.linalg.norm(self.operator@self.modes-
                (self.mass@self.modes)*eigenvalues)/max(1., np.linalg.norm(self.operator@self.modes))),
        }
        _require(max(self.defects.values()) < 1e-9, "mode decomposition failed")

    @staticmethod
    def _positive(value, name):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)), name)
        value = float(value)
        _require(math.isfinite(value) and value > 0, f"positive finite {name} required")
        return value

    def _vector(self, value, name):
        result = np.asarray(value, dtype=float)
        _require(result.shape == (len(self.vertices),) and np.isfinite(result).all(), f"finite nodal {name} required")
        return result

    def prepare(self, field: Sequence[float], velocity: Sequence[float]) -> "CoherentState":
        q = self.modes.T@self.mass@self._vector(field, "field")
        p = self.modes.T@self.mass@self._vector(velocity, "velocity")
        return CoherentState(self, (np.sqrt(self.omega)*q+1j*p/np.sqrt(self.omega))/np.sqrt(2*self.hbar))

    def describe(self):
        return {"schema": "oph.free_scalar_hilbert.v1", "mode_count": len(self.omega),
            "hilbert_space": "L2(R^N,dQ), equivalently symmetric Fock(C^N); infinite occupation space",
            "field_normalization": "u=sqrt(2) Re(Psi) in the independent e=g=0 complex-field tensor factor",
            "hamiltonian": "sum_j hbar*omega_j*(a_j^dagger*a_j+1/2)",
            "parameters": {"e": 0, "g": 0, "m_squared": self.mass_squared, "hbar": self.hbar},
            "frequencies": self.omega.tolist(), "numerical_defects": dict(self.defects),
            "boundary": "natural Neumann", "units": "supplied model space, time and action units",
            "quantization_supplied": True, "same_interacting_quantum_history": False,
            "occupation_truncation": False, "continuum_error_certified": False,
            "physical_measurement_or_prediction": False,
            "vacuum_scope": "pointwise variances and zero-point energy are finite-cutoff quantities",
            "reference": "https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S2.html"}


class CoherentState:
    def __init__(self, model: FreeScalar, alpha: np.ndarray):
        self.model, self.alpha = model, np.array(alpha, complex, copy=True)

    def _at(self, time):
        _require(not isinstance(time, (bool, np.bool_)) and isinstance(time, (int, float, np.number))
                 and math.isfinite(float(time)), "finite model time required")
        m = self.model
        angles = m.omega*float(time)
        _require(np.isfinite(angles).all(), "time-frequency product overflow")
        alpha = self.alpha*np.exp(-1j*angles)
        q = np.sqrt(2*m.hbar/m.omega)*alpha.real
        p = np.sqrt(2*m.hbar*m.omega)*alpha.imag
        return q, p

    def state(self, time: float):
        m = self.model; q, p = self._at(time)
        return {"schema": "oph.free_scalar_coherent_state.v1", "model_time": float(time),
            "field_mean": (m.modes@q).tolist(), "velocity_mean": (m.modes@p).tolist(),
            "canonical_nodal_momentum_mean": (m.mass@m.modes@p).tolist(),
            "mean_occupations": (abs(self.alpha)**2).tolist(),
            "normal_ordered_energy": float(np.sum(m.hbar*m.omega*abs(self.alpha)**2)),
            "cutoff_vacuum_energy": float(m.hbar*np.sum(m.omega)/2),
            "model": m.describe()}

    def field(self, time: float, cell_index: int, barycentric: Sequence[float]):
        m = self.model; q, p = self._at(time)
        k = _index(cell_index, len(m.cells), "cell index"); lam = _barycentric(barycentric)
        cell = m.cells[k]; basis = lam@m.modes[cell]; grad_basis = m.gradients[k].T@m.modes[cell]
        mean, velocity, gradient = float(basis@q), float(basis@p), grad_basis@q
        variance = float(np.sum(basis**2*m.hbar/(2*m.omega)))
        return {"model_time": float(time), "position": (lam@m.vertices[cell]).tolist(),
            "field_mean": mean, "velocity_mean": velocity, "gradient_mean": gradient.tolist(),
            "cutoff_field_variance": variance,
            "normal_ordered_energy_density": float((velocity**2+gradient@gradient+m.mass_squared*mean**2)/2),
            "cutoff_vacuum_energy_density": float(m.hbar/4*(np.sum(basis**2*m.omega)+
                np.sum((np.sum(grad_basis**2, axis=0)+m.mass_squared*basis**2)/m.omega))),
            "units": "supplied model units; pointwise covariance depends on spatial cutoff"}

    def smeared(self, time: float, nodal_test_function: Sequence[float]):
        m = self.model; q, p = self._at(time)
        f = m._vector(nodal_test_function, "test function")
        weights = m.modes.T@m.mass@f
        return {"model_time": float(time), "observable": "integral f_h(x) u_h(x) dx",
            "mean": float(weights@q), "time_derivative_mean": float(weights@p),
            "variance": float(m.hbar*np.sum(weights**2/m.omega)/2),
            "time_derivative_variance": float(m.hbar*np.sum(weights**2*m.omega)/2),
            "commutator_with_smeared_velocity_over_i": float(m.hbar*(f@m.mass@f)),
            "continuum_error_certified": False}
