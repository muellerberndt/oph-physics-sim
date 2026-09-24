"""Finite readings of the FLRW record-density identities on the exact source-net diamonds.

The spatially flat FLRW family has the causal order of flat space in comoving coordinates,
so the layered read law generates one finite order for every scale-factor profile; the
profile enters only through the count measure ``M_sigma(I) = rho sum_(j,s) sigma_j^4 Delta v_s``
(Lean: ``SourceNetConformalRecordDensity``).  This module reads that measure off the exact
record-metric receipts: every vertical diamond of the golden three-dimensional family carries
its per-layer event counts ``n_j``, so for a supplied profile ``sigma_j`` on the layers

* ``M_sigma(I) = sum_j sigma_j^4 n_j`` (uniform cells ``v_s = 1``, tick ``Delta = 1``),
* the sandwich ``sigma_min^4 |I| <= M_sigma(I) <= sigma_max^4 |I|``,
* the expanding count clock ``(M_sigma(I) / M_sigma(J))^(1/4)`` for the count-clock pair
  (``I`` the outermost diamond, ``J`` the reference diamond of ``floor(K/2)`` layers) with its
  enclosure ``[(s^I_min / s^J_max) (|I|/|J|)^(1/4), (s^I_max / s^J_min) (|I|/|J|)^(1/4)]`` and the
  proper-time ratio ``tau_I / tau_J`` of a comoving worldline (``tau = integral sigma d eta``),
* the redshift reading ``(n_0 / n_e)^(1/4)`` between two congruent copies of the reference
  diamond, one at the emission epoch (layers ``0 .. k_J``) and one at the reception epoch
  (layers ``K - k_J .. K``), with the sandwich enclosure ``[s^0_min / s^e_max, s^0_max / s^e_min]``

are evaluated exactly (integers and floats from the receipt) and compared.  The congruent copy
at the reception epoch has the same comoving counts because every layer carries the same site
population and the same read rule (layer translation invariance of the construction).

Profiles are supplied data, as the paper states; four are read: constant, de Sitter
``sigma = 1 / (1 - h j)``, radiation ``sigma = 1 + g j`` and matter ``sigma = (1 + g j)^2``, each
normalised to ``sigma_0 = 1`` and to a doubling of the scale across the outermost diamond.
Nothing here selects a profile, attaches a physical clock, or treats spatial curvature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/exact/source_net_causal_limit_receipt.json"
OUTPUT = ROOT / "data/exact/flrw_record_density_readout.json"
SCHEMA = "oph.exact.flrw-record-density-readout.v1"
SIG = 12


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sig(x: float, digits: int = SIG) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


def profiles(K: int) -> dict[str, dict[str, Any]]:
    """Scale factor per conformal tick on layers 0..K, sigma_0 = 1, sigma_K = 2."""

    h = 1.0 / (2.0 * K)  # de Sitter: sigma(eta) = -1/(H eta), H Delta = h, eta_0 = -1/H
    g = 1.0 / K  # radiation: sigma proportional to eta
    m = (math.sqrt(2.0) - 1.0) / K  # matter: sigma proportional to eta^2
    return {
        "constant": {"sigma": lambda j: 1.0, "proper_time": lambda k: float(k), "definition": "sigma_j = 1",
                     "parameter": None},
        "de_sitter": {"sigma": lambda j: 1.0 / (1.0 - h * j), "proper_time": lambda k: -math.log(1.0 - h * k) / h,
                      "definition": "sigma_j = 1 / (1 - h j), H Delta = h; sigma(eta) = -1/(H eta) with eta_0 = -1/H",
                      "parameter": {"h": h}},
        "radiation": {"sigma": lambda j: 1.0 + g * j, "proper_time": lambda k: k + g * k * k / 2.0,
                      "definition": "sigma_j = 1 + g j (sigma proportional to conformal time)", "parameter": {"g": g}},
        "matter": {"sigma": lambda j: (1.0 + m * j) ** 2, "proper_time": lambda k: ((1.0 + m * k) ** 3 - 1.0) / (3.0 * m),
                   "definition": "sigma_j = (1 + g j)^2 (sigma proportional to conformal time squared)", "parameter": {"g": m}},
    }


def mass(sigma: Callable[[int], float], counts: list[int], offset: int = 0) -> float:
    return float(sum(sigma(offset + j) ** 4 * n for j, n in enumerate(counts)))


def extremes(sigma: Callable[[int], float], layers: int, offset: int = 0) -> tuple[float, float]:
    values = [sigma(offset + j) for j in range(layers + 1)]
    return min(values), max(values)


def read_level(row: dict[str, Any]) -> dict[str, Any]:
    fam = row["families"][0]
    if fam["dimension"] != 3:
        raise ValueError("the first family is not the three-dimensional golden population")
    K = int(fam["layer_steps"])
    ladder = {int(e["layers"]): [int(c) for c in e["counts_by_layer"]] for e in fam["vertical_intervals"]}
    for k, counts in ladder.items():
        if len(counts) != k + 1:
            raise ValueError("per-layer counts do not match the diamond height")
    clock = fam["count_clock"]
    kI, kJ = int(clock["interval_layers"]), int(clock["reference_layers"])
    nI, nJ = ladder[kI], ladder[kJ]
    if sum(nI) != clock["interval_count"] or sum(nJ) != clock["reference_count"]:
        raise ValueError("count-clock diamonds do not match the ladder")
    flat_clock = (sum(nI) / sum(nJ)) ** 0.25
    out = {"q": int(row["q"]), "K": K, "interval_layers": kI, "reference_layers": kJ, "interval_count": sum(nI),
           "reference_count": sum(nJ), "counts_by_layer_interval": nI, "counts_by_layer_reference": nJ,
           "flat_count_clock": _sig(flat_clock), "comoving_duration_ratio": _sig(kI / kJ), "profiles": {}}
    for name, prof in profiles(K).items():
        sigma = prof["sigma"]
        rows = []
        sandwich_ok = True
        for k in sorted(ladder):
            counts = ladder[k]
            M = mass(sigma, counts)
            lo, hi = extremes(sigma, k)
            total = sum(counts)
            ok = lo**4 * total <= M * (1 + 1e-12) and M <= hi**4 * total * (1 + 1e-12)
            sandwich_ok &= ok
            rows.append({"layers": k, "count": total, "mass": _sig(M), "sigma_min": _sig(lo), "sigma_max": _sig(hi), "sandwich_holds": ok})
        MI, MJ = mass(sigma, nI), mass(sigma, nJ)
        loI, hiI = extremes(sigma, kI)
        loJ, hiJ = extremes(sigma, kJ)
        physical = (MI / MJ) ** 0.25
        enclosure = ((loI / hiJ) * flat_clock, (hiI / loJ) * flat_clock)
        tau_ratio = prof["proper_time"](kI) / prof["proper_time"](kJ)
        quotient = (hiI / loI) * (hiJ / loJ)
        # redshift between congruent reference diamonds at the emission and reception epochs
        n_e = mass(sigma, nJ, offset=0)
        n_0 = mass(sigma, nJ, offset=K - kJ)
        reading = (n_0 / n_e) ** 0.25
        lo_e, hi_e = extremes(sigma, kJ, 0)
        lo_0, hi_0 = extremes(sigma, kJ, K - kJ)
        z_enclosure = (lo_0 / hi_e, hi_0 / lo_e)
        mid = sigma((K - kJ) + kJ / 2.0) / sigma(kJ / 2.0)
        out["profiles"][name] = {
            "definition": prof["definition"], "parameter": prof["parameter"],
            "sigma_0": _sig(sigma(0)), "sigma_K": _sig(sigma(K)),
            "ladder": rows, "sandwich_holds_all": sandwich_ok,
            "expanding_count_clock": {
                "physical_clock": _sig(physical), "flat_clock": _sig(flat_clock),
                "enclosure": [_sig(enclosure[0]), _sig(enclosure[1])],
                "enclosure_holds": bool(enclosure[0] * (1 - 1e-12) <= physical <= enclosure[1] * (1 + 1e-12)),
                "enclosure_quotient": _sig(quotient),
                "proper_time_ratio": _sig(tau_ratio),
                "physical_clock_over_proper_time_ratio": _sig(physical / tau_ratio),
                "comoving_duration_ratio": _sig(kI / kJ),
                "flat_clock_over_comoving_ratio": _sig(flat_clock / (kI / kJ)),
            },
            "redshift": {
                "emission_layers": [0, kJ], "reception_layers": [K - kJ, K],
                "n_e": _sig(n_e), "n_0": _sig(n_0), "one_plus_z_reading": _sig(reading),
                "enclosure": [_sig(z_enclosure[0]), _sig(z_enclosure[1])],
                "enclosure_holds": bool(z_enclosure[0] * (1 - 1e-12) <= reading <= z_enclosure[1] * (1 + 1e-12)),
                "mid_layer_scale_ratio": _sig(mid),
                "reading_over_mid_layer_ratio": _sig(reading / mid),
                "exact_for_constant_profile_per_diamond": True,
            },
        }
    return out


def build(receipt_path: Path = RECEIPT) -> dict[str, Any]:
    data = receipt_path.read_bytes()
    receipt = json.loads(data)
    levels = [read_level(row) for row in receipt["levels"]]
    return {
        "schema": SCHEMA,
        "source_receipt": {"path": str(receipt_path.relative_to(ROOT)) if receipt_path.is_relative_to(ROOT) else str(receipt_path),
                           "sha256": hashlib.sha256(data).hexdigest(), "schema": receipt.get("schema")},
        "construction": {
            "order": "layered read law on the golden population in comoving coordinates; conformally invariant, so the finite order is the flat one for every profile",
            "count_measure": "M_sigma(I) = sum_{(j,s) in I} sigma_j^4 Delta v_s with uniform cells v_s = 1 and tick Delta = 1",
            "diamonds": "vertical intervals of the receipt: the ladder of k-layer diamonds from the centre event, per-layer counts n_j",
            "count_clock_pair": "I = outermost K-layer diamond, J = reference diamond of floor(K/2) layers, both from the centre layer",
            "redshift_pair": "two congruent copies of J at layers [0, k_J] and [K - k_J, K]; equal comoving counts by layer translation invariance",
            "profiles": "supplied data, normalised to sigma_0 = 1 and sigma_K = 2 on the outermost diamond",
            "lean": "Lean/Geometry/SourceNetConformalRecordDensity.lean (order invariance, mass identity and sandwich, clock enclosure, redshift identity)",
        },
        "levels": levels,
        "summary": {
            "sandwich_holds_all": all(p["sandwich_holds_all"] for lv in levels for p in lv["profiles"].values()),
            "clock_enclosure_holds_all": all(p["expanding_count_clock"]["enclosure_holds"] for lv in levels for p in lv["profiles"].values()),
            "redshift_enclosure_holds_all": all(p["redshift"]["enclosure_holds"] for lv in levels for p in lv["profiles"].values()),
            "flat_clock_by_q": {str(lv["q"]): lv["flat_count_clock"] for lv in levels},
            "physical_over_proper_time_by_q": {name: {str(lv["q"]): lv["profiles"][name]["expanding_count_clock"]["physical_clock_over_proper_time_ratio"] for lv in levels}
                                               for name in ("constant", "de_sitter", "radiation", "matter")},
            "redshift_reading_over_mid_by_q": {name: {str(lv["q"]): lv["profiles"][name]["redshift"]["reading_over_mid_layer_ratio"] for lv in levels}
                                               for name in ("de_sitter", "radiation", "matter")},
        },
        "nonclaims": [
            "the scale-factor profile is a supplied datum; no source law selecting it is derived or used",
            "the finite orders are the flat comoving orders of the receipt; expansion enters only through the count measure",
            "cells are uniform and the tick is one; no physical clock, no physical identification of the tick or the cell",
            "spatial curvature is not treated; the identities are those of the spatially flat family",
            "finite K: the proper-time comparison is a finite reading, the limit statements are the Lean theorems",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    data = canonical(build(args.receipt))
    if args.write:
        args.out.write_bytes(data)
    if args.check and args.out.read_bytes() != data:
        print("FLRW_RECORD_DENSITY_READOUT_STALE")
        return 1
    print("FLRW_RECORD_DENSITY_READOUT", len(data), hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
