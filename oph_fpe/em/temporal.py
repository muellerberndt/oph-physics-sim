"""Exact leapfrog lane for the declared PR-66 temporal update on the
committed base carrier.

Design-only, non-evidential module.  The arithmetic mirrors the theorem
statements of ``Lean/Screen/TemporalMaxwellEvolution.lean`` over exact
``fractions.Fraction`` values, with the staggered convention of the
committed module:

* seam potential ``A(n)`` and port potential ``phi(n)`` at integer steps;
* electric seam field on the half step:
  ``E(n) = -(A(n+1) - A(n)) - d(phi(n))``;
* magnetic face field at integer steps: ``B(n) = C A(n)``;
* the declared PR-66 leapfrog Ampere update:
  ``E(n+1) - E(n) = C^T B(n+1) - J(n)``;
* Faraday as an identity of the definitions:
  ``B(n+1) - B(n) = -C E(n)``;
* Gauss propagation ``boundary(E(n)) = rho(n)`` exactly when the
  continuity equation ``rho(n+1) - rho(n) + boundary(J(n)) = 0`` holds
  (both directions);
* the exact staggered quadratic form
  ``(1/2)|E(n)|^2 + (1/2)<B(n), B(n+1)>`` with per-step balance
  ``form(n+1) = form(n) - (1/2)<E(n) + E(n+1), J(n)>``; the form is
  indefinite on the committed spectrum and is not an energy;
* time-dependent gauge invariance
  ``A -> A + d(chi(n))``, ``phi -> phi - (chi(n+1) - chi(n))``;
* the static Coulomb join and the zero-current wave recursion with neutral
  port load
  ``A(n+2) - 2 A(n+1) + A(n) + C^T C A(n+1) + d(phi(n+1) - phi(n)) = 0``.

Every identity is checked per step and fails closed
(``TemporalReceiptError``).

Boundaries.  The update is declared under PR-66, not derived from a repair
or variational law.  The step index ``n`` is a declared evolution parameter,
not physical time (PR-15 open).  The sources ``rho`` and ``J`` are declared
inputs, not source-produced currents (PR-54 open).  No photon, propagation
speed, physical time, continuum limit, or laboratory readout is claimed
(PR-53 open).  The staggered quadratic form is not proved positive here.
This module computes no spectrum and certifies no stability property.
Finding F3 of ``plan/audits/RER_POST_R2020_DEEP_AUDIT_2026-08-20.md`` and
the boundary section of ``Lean/Screen/TemporalMaxwellEvolution.lean`` record
eigenvalues of the committed ``C^T C`` above the unit-step leapfrog
stability threshold 4, reported there as 5 and 3 + sqrt(5), so growing
modes exist on the committed carrier and no stable propagation follows from
this lane.  That audit records the scaled continuation condition
``h^2 lambda_max <= 4``, hence ``h <= 2/sqrt(3 + sqrt(5))`` on the committed
carrier, as required work rather than as a result of this module.
Everything is finite exact mathematics on the committed carrier; nothing
here is a frozen instrument or a physical claim.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from oph_fpe.em import base_carrier, green
from oph_fpe.em.base_carrier import (
    FACES,
    PORTS,
    SEAMS,
    Matrix,
    Vector,
    ZERO,
    ONE,
)

HALF = Fraction(1, 2)

History = list[Vector]


class TemporalReceiptError(ValueError):
    """A temporal evolution receipt failed closed."""


def _vec(values: Sequence[Fraction], size: int, label: str) -> Vector:
    if len(values) != size:
        raise TemporalReceiptError(f"{label} arity drift")
    return [Fraction(x) for x in values]


def zero_ports() -> Vector:
    return [ZERO] * PORTS


def zero_seams() -> Vector:
    return [ZERO] * SEAMS


def constant_history(vector: Sequence[Fraction], count: int) -> History:
    return [[Fraction(x) for x in vector] for _ in range(count)]


class LeapfrogLane:
    """Exact leapfrog evolution with fail-closed per-step receipts.

    ``face_matrix`` defaults to the committed oriented face incidence; a
    mutated matrix exists for mutation-guard tests and is caught by the
    Faraday, quadratic-form, and codifferential-boundary receipts.
    """

    def __init__(self, face_matrix: Matrix | None = None) -> None:
        self.face_matrix = (
            base_carrier.face_incidence_matrix()
            if face_matrix is None
            else [list(row) for row in face_matrix]
        )

    # -- staggered fields ---------------------------------------------------

    def electric_field(self, a: History, phi: History, n: int) -> Vector:
        """``E(n) = -(A(n+1) - A(n)) - d(phi(n))`` on the half step."""
        diff = [x - y for x, y in zip(a[n + 1], a[n], strict=True)]
        gradient = base_carrier.coboundary(phi[n])
        return [-dx - gx for dx, gx in zip(diff, gradient, strict=True)]

    def magnetic_field(self, a: History, n: int) -> Vector:
        """``B(n) = C A(n)`` at the integer step."""
        return base_carrier.face_curvature(a[n], self.face_matrix)

    def field_energy(self, a: History, phi: History, n: int) -> Fraction:
        """Staggered quadratic form
        ``(1/2)|E(n)|^2 + (1/2)<B(n), B(n+1)>``.  The form is indefinite
        on the committed spectrum and is not an energy."""
        electric = self.electric_field(a, phi, n)
        return HALF * base_carrier.seam_energy(electric) + HALF * (
            base_carrier.face_inner(
                self.magnetic_field(a, n), self.magnetic_field(a, n + 1)
            )
        )

    # -- evolution ----------------------------------------------------------

    def evolve(
        self,
        a0: Sequence[Fraction],
        a1: Sequence[Fraction],
        phi: History,
        current: History,
        steps: int,
        rho: History | None = None,
        verify: bool = True,
    ) -> dict:
        """Evolve the declared leapfrog Ampere law for ``steps`` steps.

        Produces ``A(0..steps)`` from ``A(0)``, ``A(1)`` via
        ``A(n+2) = 2 A(n+1) - A(n) - C^T C A(n+1) + J(n)
        + d(phi(n) - phi(n+1))``, the unique history solving the Ampere
        update.  ``phi`` supplies indices ``0..steps-1`` and ``current``
        supplies ``0..steps-2``.  With ``verify`` the full per-step receipt
        set runs fail-closed.
        """
        if steps < 1:
            raise TemporalReceiptError("steps must be at least 1")
        if len(phi) < steps:
            raise TemporalReceiptError("phi history too short")
        if len(current) < max(steps - 1, 0):
            raise TemporalReceiptError("current history too short")
        a: History = [
            _vec(a0, SEAMS, "A(0)"),
            _vec(a1, SEAMS, "A(1)"),
        ]
        for n in range(steps - 1):
            kinetic = base_carrier.local_maxwell_operator(
                a[n + 1], self.face_matrix
            )
            gradient = base_carrier.coboundary(
                [x - y for x, y in zip(phi[n], phi[n + 1], strict=True)]
            )
            a.append(
                [
                    2 * a[n + 1][e] - a[n][e] - kinetic[e]
                    + current[n][e] + gradient[e]
                    for e in range(SEAMS)
                ]
            )
        result = {"A": a, "phi": phi, "J": current}
        if verify:
            result["receipts"] = self.verify_history(
                a, phi, current, rho=rho, fail_closed=True
            )
        return result

    # -- receipts -----------------------------------------------------------

    def verify_history(
        self,
        a: History,
        phi: History,
        current: History,
        rho: History | None = None,
        fail_closed: bool = True,
    ) -> dict:
        """Recompute every committed temporal identity on the history.

        Receipts: the Ampere residual, the Faraday identity, the vanishing
        boundary of the codifferential, Gauss propagation with the two-way
        continuity equivalence, the exact staggered quadratic-form balance
        with source work term, and zero form drift for zero-current
        histories with neutral port load.  With
        ``fail_closed`` the first collection of failures raises
        ``TemporalReceiptError``.
        """
        last = len(a) - 1
        if last < 1:
            raise TemporalReceiptError("history needs at least two steps")
        if len(phi) < last:
            raise TemporalReceiptError("phi history too short")
        if len(current) < last - 1:
            raise TemporalReceiptError("current history too short")

        failures: list[str] = []
        electric = [self.electric_field(a, phi, n) for n in range(last)]
        magnetic = [self.magnetic_field(a, n) for n in range(last + 1)]

        # Ampere residual: E(n+1) - E(n) - C^T B(n+1) + J(n) == 0.
        ampere_ok = True
        for n in range(last - 1):
            codif = base_carrier.face_codifferential(
                magnetic[n + 1], self.face_matrix
            )
            residual = [
                electric[n + 1][e] - electric[n][e] - codif[e] + current[n][e]
                for e in range(SEAMS)
            ]
            if any(x != 0 for x in residual):
                ampere_ok = False
                failures.append(f"ampere residual nonzero at step {n}")
                break

        # Faraday identity: B(n+1) - B(n) == -C E(n).
        faraday_ok = True
        for n in range(last):
            curl = base_carrier.face_curvature(electric[n], self.face_matrix)
            if any(
                magnetic[n + 1][f] - magnetic[n][f] != -curl[f]
                for f in range(FACES)
            ):
                faraday_ok = False
                failures.append(f"faraday identity fails at step {n}")
                break

        # Boundary of the codifferential vanishes (transpose of C.d = 0).
        codifferential_ok = True
        for n in range(last + 1):
            codif = base_carrier.face_codifferential(
                magnetic[n], self.face_matrix
            )
            if any(x != 0 for x in base_carrier.boundary(codif)):
                codifferential_ok = False
                failures.append(
                    f"boundary of codifferential nonzero at step {n}"
                )
                break

        # Gauss and continuity, both directions.
        gauss_report: dict | None = None
        if rho is not None:
            if len(rho) < last:
                raise TemporalReceiptError("rho history too short")
            gauss_flags = [
                base_carrier.boundary(electric[n]) == list(rho[n])
                for n in range(last)
            ]
            continuity_flags = [
                all(
                    rho[n + 1][p] - rho[n][p]
                    + base_carrier.boundary(current[n])[p] == 0
                    for p in range(PORTS)
                )
                for n in range(last - 1)
            ]
            # Step equivalence: given Gauss at n, Gauss at n+1 holds exactly
            # when continuity holds at n.
            iff_consistent = all(
                (not gauss_flags[n])
                or (gauss_flags[n + 1] == continuity_flags[n])
                for n in range(last - 1)
            )
            first_gauss_failure = next(
                (n for n, ok in enumerate(gauss_flags) if not ok), None
            )
            gauss_report = {
                "initial": gauss_flags[0],
                "propagated_all_steps": all(gauss_flags),
                "continuity_all_steps": all(continuity_flags),
                "step_iff_consistent": iff_consistent,
                "first_gauss_failure_step": first_gauss_failure,
            }
            if not iff_consistent:
                failures.append("gauss/continuity equivalence violated")
            if not gauss_flags[0]:
                failures.append("initial gauss constraint fails")
            if all(continuity_flags) and not all(gauss_flags):
                failures.append(
                    "gauss propagation fails under a conserved source"
                )
            if not all(continuity_flags):
                failures.append(
                    "continuity violated: declared source inconsistent, "
                    f"gauss breaks at step {first_gauss_failure}"
                )

        # Exact staggered quadratic-form balance with source work term.
        forms = [
            HALF * base_carrier.seam_energy(electric[n])
            + HALF * base_carrier.face_inner(magnetic[n], magnetic[n + 1])
            for n in range(last)
        ]
        balance_ok = True
        for n in range(last - 1):
            work = HALF * base_carrier.seam_inner(
                [
                    electric[n][e] + electric[n + 1][e]
                    for e in range(SEAMS)
                ],
                current[n],
            )
            if forms[n + 1] != forms[n] - work:
                balance_ok = False
                failures.append(f"quadratic-form balance fails at step {n}")
                break
        zero_current = all(
            all(x == 0 for x in current[n]) for n in range(last - 1)
        )
        drift_zero: bool | None = None
        if zero_current:
            drift_zero = all(value == forms[0] for value in forms)
            if not drift_zero:
                failures.append(
                    "quadratic-form drift under zero-current evolution"
                )

        receipts = {
            "schema": "oph.sim.em_temporal_receipts.v1",
            "design_only": True,
            "steps": last,
            "ampere_exact": ampere_ok,
            "faraday_identity_exact": faraday_ok,
            "codifferential_boundary_zero": codifferential_ok,
            "gauss": gauss_report,
            "energy": {
                "balance_exact": balance_ok,
                "source_free": zero_current,
                "drift_zero": drift_zero,
                "initial": str(forms[0]),
                "final": str(forms[-1]),
            },
            "failures": failures,
            "statement": (
                "the unit-step Ampere update is declared under PR-66; the "
                "conserved staggered form is an indefinite quadratic form, "
                "not an energy; this module computes no spectrum; RER "
                "audit finding F3 and the boundary section of "
                "Lean/Screen/TemporalMaxwellEvolution.lean record "
                "eigenvalues 5 and 3 + sqrt(5) of the committed C^T C, "
                "above the unit-step leapfrog stability threshold 4, so "
                "growing modes exist on the committed carrier and no "
                "stable propagation follows from this lane; that audit "
                "records the scaled continuation condition "
                "h^2 lambda_max <= 4, hence h <= 2/sqrt(3 + sqrt(5)), as "
                "required work; step index is a declared evolution "
                "parameter, not physical time; sources are declared "
                "inputs; PR-15, PR-53, PR-54 open"
            ),
        }
        if fail_closed and failures:
            raise TemporalReceiptError("; ".join(failures))
        return receipts

    # -- gauge invariance ---------------------------------------------------

    def gauge_transform(
        self, a: History, phi: History, chi: History
    ) -> tuple[History, History]:
        """``A(n) -> A(n) + d(chi(n))``,
        ``phi(n) -> phi(n) - (chi(n+1) - chi(n))``."""
        if len(chi) < len(a):
            raise TemporalReceiptError("chi history too short")
        a_new = [
            [
                x + g
                for x, g in zip(
                    a[n], base_carrier.coboundary(chi[n]), strict=True
                )
            ]
            for n in range(len(a))
        ]
        phi_new = [
            [
                phi[n][p] - (chi[n + 1][p] - chi[n][p])
                for p in range(PORTS)
            ]
            for n in range(len(phi))
        ]
        return a_new, phi_new

    def gauge_invariance_receipt(
        self, a: History, phi: History, current: History, chi: History
    ) -> dict:
        """Exact invariance of ``E``, ``B``, the Ampere residual, and the
        staggered quadratic form under the time-dependent gauge
        transformation."""
        a_new, phi_new = self.gauge_transform(a, phi, chi)
        last = len(a) - 1
        failures: list[str] = []
        for n in range(last):
            if self.electric_field(a_new, phi_new, n) != self.electric_field(
                a, phi, n
            ):
                failures.append(f"electric field moves under gauge at {n}")
        for n in range(last + 1):
            if self.magnetic_field(a_new, n) != self.magnetic_field(a, n):
                failures.append(f"magnetic field moves under gauge at {n}")
        for n in range(last - 1):
            if self.field_energy(a_new, phi_new, n) != self.field_energy(
                a, phi, n
            ):
                failures.append(f"quadratic form moves under gauge at {n}")
        original = self.verify_history(
            a, phi, current, fail_closed=False
        )
        transformed = self.verify_history(
            a_new, phi_new, current, fail_closed=False
        )
        if original["ampere_exact"] != transformed["ampere_exact"]:
            failures.append("ampere law not gauge invariant")
        if failures:
            raise TemporalReceiptError("; ".join(failures))
        return {
            "gauge_invariant": True,
            "steps": last,
            "electric_invariant": True,
            "magnetic_invariant": True,
            "energy_invariant": True,
            "ampere_preserved": True,
        }

    # -- static Coulomb join ------------------------------------------------

    def static_join_receipt(
        self, a: History, phi: History, current: History
    ) -> dict:
        """Uniqueness direction of the committed static join: a constant
        zero-current solution has a forced-neutral port load, zero magnetic
        field, and electric field exactly the committed Coulomb field."""
        last = len(a) - 1
        if any(a[n] != a[0] for n in range(last + 1)):
            raise TemporalReceiptError("history is not constant in A")
        if any(phi[n] != phi[0] for n in range(last)):
            raise TemporalReceiptError("history is not constant in phi")
        if any(
            any(x != 0 for x in current[n]) for n in range(max(last - 1, 0))
        ):
            raise TemporalReceiptError("static join requires J = 0")
        self.verify_history(a, phi, current, fail_closed=True)
        electric0 = self.electric_field(a, phi, 0)
        load = base_carrier.boundary(electric0)
        if sum(load, start=ZERO) != 0:
            raise TemporalReceiptError("static load fails forced neutrality")
        coulomb = green.coulomb_field(load)
        failures = []
        for n in range(last):
            if self.electric_field(a, phi, n) != coulomb:
                failures.append(f"static E differs from Green field at {n}")
        for n in range(last + 1):
            if any(x != 0 for x in self.magnetic_field(a, n)):
                failures.append(f"static magnetic field nonzero at {n}")
        if failures:
            raise TemporalReceiptError("; ".join(failures))
        return {
            "static_join": True,
            "neutral": True,
            "magnetic_zero": True,
            "electric_equals_green_field": True,
            "load": [str(x) for x in load],
        }

    def coulomb_static_solution(
        self, rho: Sequence[Fraction], steps: int
    ) -> dict:
        """Existence direction: for a neutral load the canonical Green
        potential supplies a static solution whose electric field is exactly
        the committed Coulomb field and whose Gauss load is exactly
        ``rho``."""
        load = _vec(rho, PORTS, "rho")
        potential = green.green_potential(load)
        phi = constant_history([-x for x in potential], steps)
        current = constant_history(zero_seams(), max(steps - 1, 1))
        rho_history = constant_history(load, steps)
        run = self.evolve(
            zero_seams(), zero_seams(), phi, current, steps,
            rho=rho_history, verify=True,
        )
        receipt = self.static_join_receipt(run["A"], phi, current)
        coulomb = green.coulomb_field(load)
        for n in range(steps):
            if self.electric_field(run["A"], phi, n) != coulomb:
                raise TemporalReceiptError(
                    "existence direction: E differs from the Coulomb field"
                )
        receipt["existence_direction"] = True
        receipt["steps"] = steps
        return receipt

    # -- wave recursion -----------------------------------------------------

    def wave_recursion_receipt(self, a: History, phi: History) -> dict:
        """Zero-current second-order recursion with neutral port load:
        ``A(n+2) - 2 A(n+1) + A(n) + C^T C A(n+1)
        + d(phi(n+1) - phi(n)) = 0`` at every interior step."""
        last = len(a) - 1
        for n in range(last - 1):
            kinetic = base_carrier.local_maxwell_operator(
                a[n + 1], self.face_matrix
            )
            gradient = base_carrier.coboundary(
                [
                    phi[n + 1][p] - phi[n][p]
                    for p in range(PORTS)
                ]
            )
            residual = [
                a[n + 2][e] - 2 * a[n + 1][e] + a[n][e]
                + kinetic[e] + gradient[e]
                for e in range(SEAMS)
            ]
            if any(x != 0 for x in residual):
                raise TemporalReceiptError(
                    f"wave recursion fails at step {n}"
                )
        return {"wave_recursion_exact": True, "interior_steps": last - 1}


def demo_bundle(steps: int = 64) -> dict:
    """The committed nonstatic inhabitant: zero initial data kicked by the
    face-zero boundary cycle, evolved at zero current with neutral port
    load.  Mirrors the committed anchors ``A(1)(0) = 1``, ``B(0) = 0``,
    ``B(1)(0) = 3`` and the exact conservation of the staggered quadratic
    form.  Conservation of an indefinite form is not a stability
    statement."""
    lane = LeapfrogLane()
    kick = base_carrier.face_codifferential(
        [ONE if f == 0 else ZERO for f in range(FACES)]
    )
    phi = constant_history(zero_ports(), steps)
    current = constant_history(zero_seams(), max(steps - 1, 1))
    rho = constant_history(zero_ports(), steps)
    run = lane.evolve(
        zero_seams(), kick, phi, current, steps, rho=rho, verify=True
    )
    a = run["A"]
    receipts = run["receipts"]
    anchors = {
        "A1_seam0": a[1][0],
        "B0_zero": all(x == 0 for x in lane.magnetic_field(a, 0)),
        "B1_face0": lane.magnetic_field(a, 1)[0],
        "nonstatic": a[1] != a[0],
    }
    if anchors["A1_seam0"] != 1:
        raise TemporalReceiptError("demo anchor A(1)(0) != 1")
    if not anchors["B0_zero"]:
        raise TemporalReceiptError("demo anchor B(0) != 0")
    if anchors["B1_face0"] != 3:
        raise TemporalReceiptError("demo anchor B(1)(0) != 3")
    if not anchors["nonstatic"]:
        raise TemporalReceiptError("demo inhabitant is static")
    wave = lane.wave_recursion_receipt(a, phi)
    return {
        "schema": "oph.sim.em_temporal_demo.v1",
        "design_only": True,
        "steps": steps,
        "anchors": {
            "A1_seam0": str(anchors["A1_seam0"]),
            "B0_zero": anchors["B0_zero"],
            "B1_face0": str(anchors["B1_face0"]),
            "nonstatic": anchors["nonstatic"],
        },
        "receipts": receipts,
        "wave": wave,
    }
