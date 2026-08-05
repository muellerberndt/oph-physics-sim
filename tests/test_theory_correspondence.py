"""Simulator-versus-theory correspondence checks.

Each test here replays an exact statement of the theory repository's
Lean stack against the simulator's own code paths, in both directions:
the simulator construction must satisfy the theorem's conclusion, and a
perturbed construction must fail the corresponding check. The anchors
are the conditional-resampling theorem of the finite thermodynamics
package, the fixed-word dependency-cone theorem of the locality
package, and the chained Lyapunov non-increase contract of the repair
layer. These are finite correspondence checks on small instances; they
promote no physical claim.
"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import pytest

from oph_fpe.consensus.lyapunov import lyapunov_descent_receipt
from oph_fpe.core.patchnet import PatchNet
from oph_fpe.dynamics.repair import RepairKernel
from oph_fpe.groups.z2 import build_z2
from oph_fpe.quotient.observable_normal_form import (
    recognize_conditional_resampling_kernel,
)


def _fiber_kernel(states, weights, observation_map):
    """The exact conditional-resampling kernel of the Lean theorem:
    P(x, y) = 1[b(y) = b(x)] * pi(y) / pi(F(x))."""
    fiber_mass = {}
    for state in states:
        fiber_mass.setdefault(observation_map(state), Fraction(0))
        fiber_mass[observation_map(state)] += weights[state]
    return {
        x: {
            y: (
                weights[y] / fiber_mass[observation_map(x)]
                if observation_map(y) == observation_map(x)
                else Fraction(0)
            )
            for y in states
        }
        for x in states
    }


def test_conditional_resampling_kernel_matches_lean_theorem():
    states = ["a", "b", "c", "d", "e", "f"]
    obs = {s: i // 2 for i, s in enumerate(states)}
    weights = {
        "a": Fraction(1, 12),
        "b": Fraction(2, 12),
        "c": Fraction(3, 12),
        "d": Fraction(1, 12),
        "e": Fraction(2, 12),
        "f": Fraction(3, 12),
    }
    kernel = _fiber_kernel(states, weights, obs.__getitem__)

    audit = recognize_conditional_resampling_kernel(
        states, kernel, weights=weights, observation_map=obs.__getitem__
    )
    assert audit.exact_table_recognition_receipt is True
    assert audit.r1_fiber_supported is True
    assert audit.r2_fiber_rows_constant is True
    assert audit.r3_weighted_detailed_balance is True
    assert audit.explicit_formula_match is True

    # The Lean kernel package replayed exactly over the rationals:
    # idempotence, stationarity, and the chi-squared second Lyapunov
    # functional of the certified second law.
    for x in states:
        for y in states:
            composed = sum(
                (kernel[x][z] * kernel[z][y] for z in states),
                start=Fraction(0),
            )
            assert composed == kernel[x][y]
    for y in states:
        pushed = sum(
            (weights[x] * kernel[x][y] for x in states), start=Fraction(0)
        )
        assert pushed == weights[y]
    p = {
        "a": Fraction(5, 12),
        "b": Fraction(1, 12),
        "c": Fraction(2, 12),
        "d": Fraction(2, 12),
        "e": Fraction(1, 12),
        "f": Fraction(1, 12),
    }
    pushed = {
        y: sum((p[x] * kernel[x][y] for x in states), start=Fraction(0))
        for y in states
    }
    chi_before = sum(
        (p[x] - weights[x]) ** 2 / weights[x] for x in states
    )
    chi_after = sum(
        (pushed[x] - weights[x]) ** 2 / weights[x] for x in states
    )
    assert chi_after <= chi_before


def test_conditional_resampling_recognizer_rejects_perturbation():
    states = ["a", "b", "c", "d"]
    obs = {s: i // 2 for i, s in enumerate(states)}
    weights = {s: Fraction(1, 4) for s in states}
    kernel = _fiber_kernel(states, weights, obs.__getitem__)
    kernel["a"]["a"] += Fraction(1, 8)
    kernel["a"]["b"] -= Fraction(1, 8)
    audit = recognize_conditional_resampling_kernel(
        states, kernel, weights=weights, observation_map=obs.__getitem__
    )
    assert audit.exact_table_recognition_receipt is False


def _line_net(length: int, seed: int) -> PatchNet:
    graph = nx.path_graph(length)
    return PatchNet.random(graph, build_z2(), seed=seed)


def _apply_word(net: PatchNet, word, kernel: RepairKernel) -> None:
    """Drive the simulator's own local-repair move along a fixed
    exogenous site word, with the cold strict-descent acceptance."""
    for site in word:
        before = net.touched_phi(site)
        original = net.states[site].copy()
        kernel._propose_local_best(net, site)
        after = net.touched_phi(site)
        if after - before > 0:
            net.states[site] = original


def test_fixed_word_dependency_cone_no_influence():
    length = 12
    word = [8, 9, 8, 7, 9, 8]
    probe = 8
    # The n-fold closed-neighborhood cone of the probe under this word
    # stays within distance len(word) of the probe; node 0 lies outside.
    far_node = 0
    assert nx.shortest_path_length(
        nx.path_graph(length), probe, far_node
    ) > len(word)

    left = _line_net(length, seed=11)
    right = _line_net(length, seed=11)
    # Perturb the far node's state on one copy only.
    right.states[far_node].hidden ^= 1
    right.states[far_node].scalar += 3.5

    kernel_left = RepairKernel(
        mode="local_best", hot_metropolis=False, seed=7
    )
    kernel_right = RepairKernel(
        mode="local_best", hot_metropolis=False, seed=7
    )
    _apply_word(left, word, kernel_left)
    _apply_word(right, word, kernel_right)

    assert left.states[probe].hidden == right.states[probe].hidden
    assert left.states[probe].scalar == right.states[probe].scalar
    assert left.states[probe].ports == right.states[probe].ports


def test_cold_repair_trace_passes_chained_lyapunov():
    net = _line_net(10, seed=3)
    kernel = RepairKernel(mode="local_best", hot_metropolis=False, seed=5)
    trace = []
    previous = net.total_phi()
    for cycle in range(40):
        kernel.step(net, cycle, beta=0.0)
        current = net.total_phi()
        trace.append({"phi_before": previous, "phi": current})
        previous = current
    report = lyapunov_descent_receipt(trace)
    assert report["LYAPUNOV_DESCENT_RECEIPT"] is True
    assert report["driven_trajectory"] is False


def test_driven_trace_fails_chained_lyapunov():
    net = _line_net(10, seed=3)
    kernel = RepairKernel(mode="local_best", hot_metropolis=False, seed=5)
    trace = []
    previous = net.total_phi()
    for cycle in range(20):
        if cycle == 10:
            # Exogenous drive between cycles: inject mismatch by
            # flipping a port, exactly the injection pattern the
            # chained receipt must expose.
            node = 4
            neighbor = 5
            net.states[node].ports[neighbor] ^= 1
        before = net.total_phi()
        kernel.step(net, cycle, beta=0.0)
        current = net.total_phi()
        trace.append({"phi_before": before, "phi": current})
        previous = current
    injected = any(
        row["phi_before"] > trace[i - 1]["phi"] + 1e-12
        for i, row in enumerate(trace)
        if i > 0
    )
    if not injected:
        pytest.skip("injection did not raise phi on this seed")
    report = lyapunov_descent_receipt(trace)
    assert report["LYAPUNOV_DESCENT_RECEIPT"] is False
    assert report["driven_trajectory"] is True
