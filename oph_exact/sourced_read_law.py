"""EXPLORATORY LANE: record density acting back on the read law (a proposed law, not the declared architecture).

The declared source-net read law is geometric and fixed: ``(j+1, s)`` reads ``(j, t)`` whenever
``|t - s| <= a``, on a fixed population, so every site produces the same number of records per
layer and the count measure of the FLRW fragment carries a supplied scale factor.  This lane
inverts the fragment's identity ``n = rho sigma^4 Delta v`` locally: each site's own record count
sets its own scale, ``sigma(s) = (n(s)/N0)^(g/4)``, the physical read radius is ``sigma(s) a``,
the physical distance between two sites is the mean scale times the comoving distance, and a
site's record production is the number of reads it performs times ``sigma(s)^beta``.  The
declared law is the point ``g = 0``.  ``g = 1`` with ``beta = 1`` is the fragment's identity read
backwards; larger ``g`` couples the geometry to the records more strongly.  A finite capacity
``cap`` (records per site per layer) is optional.

Read relation (mode ``physical``):  ``(j+1, s)`` reads ``(j, t)`` iff
``((sigma(s) + sigma(t)) / 2) |t - s| <= sigma(s) a``, that is
``|t - s| <= a * 2 sigma(s) / (sigma(s) + sigma(t))``; self read included.  Record production:
``n_{j+1}(s) = min(cap, sigma_j(s)^beta * reads_j(s))``.

Linear theory of this map around the uniform fixed point ``n = N0`` (derived in the docstring
of ``linear_prediction``): with ``delta = n/N0 - 1`` and ``delta_bar`` its ball average, one
round maps ``delta -> (g/4) (beta delta + (3/2)(delta - delta_bar))``.  Sub-radius modes grow by
``lambda_short = (g/4)(beta + 3/2)`` per round, super-radius modes by ``lambda_long = beta g/4``,
and the uniform mode obeys ``n -> N0 (n/N0)^(beta g/4)`` exactly.  The lane measures the growth
of record-density contrast by scale on the real golden population, against these predictions,
and the run's own expansion history ``sigma_bar_j``.

Everything here is a model exploration: no coupling is derived, positions are floats on a torus
of side ``L``, and nothing is asserted about the declared architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from oph_exact.source_net import site_coordinates

SCHEMA = "oph.exploratory.sourced-read-law.v1"
PHI = (1 + 5**0.5) / 2
L = 2.0 / (PHI + 2) ** 0.5


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sig(x: float, digits: int = 6) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


def linear_prediction(g: float, beta: float) -> dict[str, float]:
    """Growth factors per round of the linearised map.

    Around ``n = N0``: ``sigma = 1 + (g/4) delta``.  The reads of ``s`` are the sites within
    ``a * 2 sigma(s) / (sigma(s) + sigma(t))``; to first order the radius is
    ``a (1 + (sigma(s) - sigma_bar) / 2)`` with ``sigma_bar`` the mean scale over the ball, so
    the count is ``N0 (1 + (3/2)(g/4)(delta - delta_bar))``.  Multiplying by ``sigma^beta``
    adds ``beta (g/4) delta``.  Modes shorter than the ball have ``delta_bar = 0``; modes much
    longer have ``delta_bar = delta``.
    """

    return {"lambda_short": (g / 4.0) * (beta + 1.5), "lambda_long": beta * g / 4.0, "uniform_exponent": beta * g / 4.0}


class Population:
    def __init__(self, q: int, periodic: bool = True) -> None:
        self.q = q
        labels = site_coordinates(q, 3)  # integer labels b in [0, q)^3, product order
        xi = (np.arange(q, dtype=np.float64) * PHI) % 1.0  # xi_b = frac(b phi), the golden coordinate of the papers
        self.sites = L * xi[labels]  # s(b) = L (xi_b1, xi_b2, xi_b3) in [0, L)^3
        self.n = self.sites.shape[0]
        self.a = L / np.sqrt(q)
        self.periodic = periodic
        self.last_ceiling_fraction = 0.0
        box = L if periodic else None
        self.tree = cKDTree(self.sites, boxsize=box)
        pairs = self.tree.query_pairs(2.0 * self.a, output_type="ndarray")  # candidates for every ratio <= 2
        self.pa = pairs[:, 0].astype(np.int64)
        self.pb = pairs[:, 1].astype(np.int64)
        d = self.sites[self.pa] - self.sites[self.pb]
        if periodic:
            d -= L * np.round(d / L)
        self.dist = np.sqrt((d * d).sum(axis=1))
        within = self.dist <= self.a
        self.fixed_counts = 1 + np.bincount(self.pa[within], minlength=self.n) + np.bincount(self.pb[within], minlength=self.n)
        self.N0 = float(self.fixed_counts.mean())

    def reads(self, sigma: np.ndarray, mode: str) -> np.ndarray:
        """Number of reads performed by every site (self read included) under the scale field."""

        sa = sigma[self.pa]
        sb = sigma[self.pb]
        if mode == "physical":
            ra = self.a * 2.0 * sa / (sa + sb)
            rb = self.a * 2.0 * sb / (sa + sb)
        elif mode == "comoving_radius":  # radius stays comoving, distances are physical
            ra = rb = self.a * 2.0 / (sa + sb)
        elif mode == "fixed":
            ra = rb = np.full(self.dist.shape, self.a)
        else:
            raise ValueError(mode)
        self.last_ceiling_fraction = float(np.mean(np.maximum(ra, rb) > 2.0 * self.a))  # radii the candidate list cannot resolve
        return 1 + np.bincount(self.pa[self.dist <= ra], minlength=self.n) + np.bincount(self.pb[self.dist <= rb], minlength=self.n)


def coarse_variances(sites: np.ndarray, delta: np.ndarray, scales: Sequence[int]) -> dict[str, float]:
    out = {}
    for m in scales:
        key = np.minimum((sites / L * m).astype(np.int64), m - 1)
        lab = key[:, 0] * m * m + key[:, 1] * m + key[:, 2]
        ids, inv = np.unique(lab, return_inverse=True)
        counts = np.bincount(inv)
        means = np.bincount(inv, weights=delta) / counts
        out[str(m)] = float(means.var())
    return out


def step(pop: Population, n: np.ndarray, *, g: float, beta: float, mode: str, cap: float | None) -> np.ndarray:
    sigma = (n / pop.N0) ** (g / 4.0)
    out = sigma**beta * pop.reads(sigma, mode).astype(np.float64)
    if cap:
        out = np.minimum(out, cap)
    return np.maximum(out, 1.0)


def tangent_response(pop: Population, *, g: float, beta: float, mode: str = "physical", relax: int = 8, measure: int = 6, cap: float | None = None,
                     amplitude: float = 0.05, seed: int = 20260924, scales: Sequence[int] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    """Growth of a small perturbation as the difference of two trajectories from the relaxed native field.

    The native field (the fixed-law neighbour counts) is iterated ``relax`` rounds; the perturbed
    trajectory starts from that state times ``1 + amplitude * noise``; the difference of the two
    trajectories, divided by the mean, is binned at every scale and its amplitude growth per round
    is the geometric mean of the ratios over the measured rounds.
    """

    rng = np.random.default_rng(seed)
    n = pop.fixed_counts.astype(np.float64)
    relaxation = []
    for _ in range(relax):
        n = step(pop, n, g=g, beta=beta, mode=mode, cap=cap)
        relaxation.append(_sig(float((n / n.mean() - 1.0).std())))
    base = n.copy()
    pert = np.maximum(n * (1.0 + amplitude * rng.standard_normal(pop.n)), 1.0)
    variances = []
    for j in range(measure + 1):
        d = (pert - base) / base.mean()
        variances.append(coarse_variances(pop.sites, d, scales))
        if j == measure:
            break
        base = step(pop, base, g=g, beta=beta, mode=mode, cap=cap)
        pert = step(pop, pert, g=g, beta=beta, mode=mode, cap=cap)
    growth = {}
    for m in scales:
        k = str(m)
        ratios = [variances[j + 1][k] / variances[j][k] for j in range(measure) if variances[j][k] > 0 and variances[j + 1][k] > 0]
        growth[k] = _sig(float(np.exp(np.mean(np.log(ratios)))) ** 0.5) if ratios else None
    return {"relaxed_contrast_sd_by_round": relaxation, "relaxed_sigma_bar": _sig(float((base.mean() / pop.N0) ** (g / 4.0))),
            "perturbation_amplitude": amplitude, "amplitude_growth_per_round_by_scale": growth}


def run(pop: Population, *, g: float, beta: float, mode: str = "physical", rounds: int = 24, cap: float | None = None,
        initial: str = "seeded", amplitude: float = 0.01, seed: int = 20260924, scales: Sequence[int] = (1, 2, 4, 8, 16), log=None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if initial == "seeded":
        n = pop.N0 * (1.0 + amplitude * rng.standard_normal(pop.n))
    elif initial == "native":
        n = pop.fixed_counts.astype(np.float64)
    else:
        raise ValueError(initial)
    n = np.maximum(n, 1.0)
    history = []
    started = time.perf_counter()
    for j in range(rounds + 1):
        mean = float(n.mean())
        delta = n / mean - 1.0
        row = {"round": j, "mean_records": _sig(mean), "sigma_bar": _sig((mean / pop.N0) ** (g / 4.0)), "contrast_sd": _sig(float(delta.std())),
               "max_over_mean": _sig(float(n.max() / mean)), "min_over_mean": _sig(float(n.min() / mean)),
               "at_capacity_fraction": _sig(float(np.mean(n >= cap))) if cap else 0.0,
               "radius_ceiling_fraction": _sig(pop.last_ceiling_fraction),
               "coarse_variance": {k: _sig(v, 8) for k, v in coarse_variances(pop.sites, delta, scales).items()}}
        history.append(row)
        if log:
            log(f"  g={g} beta={beta} round {j}: mean {mean:.2f} sd(delta) {delta.std():.4g} max/mean {n.max()/mean:.3g} {time.perf_counter()-started:.0f}s")
        if j == rounds:
            break
        n = step(pop, n, g=g, beta=beta, mode=mode, cap=cap)
    pred = linear_prediction(g, beta)
    response = tangent_response(pop, g=g, beta=beta, mode=mode, cap=cap, seed=seed, scales=scales)
    return {"g": g, "beta": beta, "mode": mode, "rounds": rounds, "cap": cap, "initial": initial, "amplitude": amplitude, "seed": seed,
            "linear_prediction": {k: _sig(v) for k, v in pred.items()},
            "tangent_response": response,
            "measured_amplitude_growth_per_round_by_scale": response["amplitude_growth_per_round_by_scale"],
            "bin_side_over_read_radius": {str(m): _sig(L / m / pop.a) for m in scales},
            "history": history, "seconds": round(time.perf_counter() - started, 2)}


def build(q: int, *, couplings: Sequence[float], betas: Sequence[float], modes: Sequence[str], rounds: int, cap: float | None,
          initial: str, amplitude: float, log=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    pop = Population(q)
    log(f"q={q}: {pop.n} sites, a={pop.a:.5f}, N0={pop.N0:.2f} (fixed-law neighbours min {pop.fixed_counts.min()} max {pop.fixed_counts.max()}), candidate pairs {pop.pa.size}")
    results = []
    for mode in modes:
        for g in couplings:
            for beta in betas:
                r = run(pop, g=g, beta=beta, mode=mode, rounds=rounds, cap=cap, initial=initial, amplitude=amplitude)
                h = r["history"]
                log(f"  mode {mode} g={g} beta={beta}: predicted short {r['linear_prediction']['lambda_short']:.3f} long {r['linear_prediction']['lambda_long']:.3f}; "
                    f"measured growth/round by bin side/a {r['bin_side_over_read_radius']} -> {r['measured_amplitude_growth_per_round_by_scale']}; "
                    f"sigma_bar end {h[-1]['sigma_bar']}, contrast sd {h[0]['contrast_sd']} -> {h[-1]['contrast_sd']}, at cap {h[-1]['at_capacity_fraction']}; "
                    f"relaxed native contrast {r['tangent_response']['relaxed_contrast_sd_by_round'][-1]} sigma_bar {r['tangent_response']['relaxed_sigma_bar']}")
                results.append(r)
    return {"schema": SCHEMA, "status": "EXPLORATORY PROPOSED LAW; not the declared architecture; no coupling derived",
            "population": {"q": q, "sites": pop.n, "L": L, "read_radius": pop.a, "periodic_torus": pop.periodic,
                           "radius_ceiling": "candidate pairs are listed within 2a, so a read radius above 2a is truncated; radius_ceiling_fraction reports how often",
                           "N0_fixed_law_mean_neighbours": _sig(pop.N0), "fixed_law_neighbour_min": int(pop.fixed_counts.min()), "fixed_law_neighbour_max": int(pop.fixed_counts.max()),
                           "sites_sha256": hashlib.sha256(pop.sites.tobytes()).hexdigest()},
            "law": {"scale": "sigma(s) = (n(s)/N0)^(g/4)", "read_relation_physical": "|t - s| <= a * 2 sigma(s) / (sigma(s) + sigma(t)), self read included",
                    "read_relation_comoving_radius": "|t - s| <= a * 2 / (sigma(s) + sigma(t))", "production": "n_{j+1}(s) = min(cap, sigma_j(s)^beta * reads_j(s))",
                    "linear_theory": "delta -> (g/4)(beta delta + 1.5 (delta - delta_bar)); uniform n -> N0 (n/N0)^(beta g/4)"},
            "scan": {"couplings": list(couplings), "betas": list(betas), "modes": list(modes), "rounds": rounds, "cap": cap, "initial": initial, "amplitude": amplitude},
            "results": results,
            "nonclaims": ["a proposed feedback law scanned over its coupling; g = 1, beta = 1 is the FLRW fragment's identity read locally, the rest is exploration",
                          "float positions on a periodic torus; not the exact Q(sqrt5) lane and not the windowed population of the receipts",
                          "no physical scale, no derivation of the coupling, no claim about the declared architecture"],
            "seconds": round(time.perf_counter() - t0, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q", type=int, default=21)
    parser.add_argument("--couplings", type=float, nargs="*", default=[0.0, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--betas", type=float, nargs="*", default=[1.0, 0.0])
    parser.add_argument("--modes", nargs="*", default=["physical"])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--initial", default="seeded")
    parser.add_argument("--amplitude", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.q, couplings=args.couplings, betas=args.betas, modes=args.modes, rounds=args.rounds, cap=args.cap, initial=args.initial, amplitude=args.amplitude)
    if args.out:
        args.out.write_bytes(canonical(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
