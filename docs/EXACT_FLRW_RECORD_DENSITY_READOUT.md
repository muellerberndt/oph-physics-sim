# FLRW record-density readout on the exact source-net diamonds

`oph_exact/flrw_record_density_readout.py` reads the count measure of the spatially flat
FLRW family off the exact record-metric receipts. It computes nothing new about the order:
the layered read law in comoving coordinates is conformally invariant, so the finite orders
of `data/exact/source_net_causal_limit_receipt.json` are the orders of every scale-factor
profile. The profile enters only through the count measure

    M_sigma(I) = sum_{(j,s) in I} sigma_j^4 Delta v_s

(uniform cells `v_s = 1`, tick `Delta = 1`), and every vertical diamond of the golden
three-dimensional family carries its per-layer event counts `n_j`, so `M_sigma` is an exact
finite sum. The identities are those of
`Lean/Geometry/SourceNetConformalRecordDensity.lean` in the research repository (order
invariance, mass identity and sandwich, expanding count-clock enclosure, redshift identity).

## What is read

For each level `q` and each of four supplied profiles on the layers `0..K`, normalised to
`sigma_0 = 1` and `sigma_K = 2` across the outermost diamond:

| Profile | `sigma_j` | proper time of `k` ticks |
| --- | --- | --- |
| constant | `1` | `k` |
| de Sitter | `1 / (1 - h j)`, `h = 1/(2K)` | `-ln(1 - h k) / h` |
| radiation | `1 + g j`, `g = 1/K` | `k + g k^2 / 2` |
| matter | `(1 + g j)^2`, `g = (sqrt2 - 1)/K` | `((1 + g k)^3 - 1) / (3 g)` |

- the ladder masses `M_sigma` of the `k`-layer diamonds and the sandwich
  `sigma_min^4 |I| <= M_sigma(I) <= sigma_max^4 |I|`;
- the expanding count clock `(M_sigma(I)/M_sigma(J))^(1/4)` for the receipt's count-clock
  pair (`I` the `K`-layer diamond, `J` the reference diamond of `floor(K/2)` layers, both from
  the centre layer), its enclosure
  `[(s^I_min/s^J_max)(|I|/|J|)^(1/4), (s^I_max/s^J_min)(|I|/|J|)^(1/4)]`, and the proper-time
  ratio `tau_I/tau_J` of a comoving worldline through the two diamonds;
- the redshift reading `(n_0/n_e)^(1/4)` between two congruent copies of `J`, at the
  emission epoch (layers `0..k_J`) and the reception epoch (layers `K-k_J..K`), with the
  enclosure `[s^0_min/s^e_max, s^0_max/s^e_min]`. The copy at the reception epoch has the
  same comoving counts because every layer carries the same population and the same read
  rule.

## Readings through q = 89

Every sandwich and every enclosure holds at every level. The finite readings:

| q | K | flat clock `(|I|/|J|)^(1/4)` (limit 2) | de Sitter physical clock / proper-time ratio | matter | radiation | de Sitter redshift reading / mid-layer ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 3 | 2.515 | 0.824 | 0.830 | 0.832 | 1.030 |
| 8 | 3 | 3.114 | 1.015 | 1.025 | 1.028 | 1.030 |
| 13 | 4 | 1.705 | 0.840 | 0.856 | 0.860 | 1.001 |
| 21 | 5 | 2.010 | 0.795 | 0.811 | 0.814 | 1.000 |
| 34 | 6 | 2.088 | 1.024 | 1.043 | 1.048 | 1.005 |
| 55 | 8 | 1.907 | 0.938 | 0.956 | 0.961 | 1.003 |
| 89 | 10 | 2.051 | 1.008 | 1.027 | 1.032 | 1.004 |

The constant profile reproduces the flat clock exactly and reads a redshift of one. For the
de Sitter profile at q = 89 the physical clock reads 2.429 against a proper-time ratio of
2.409 inside the enclosure `[1.538, 4.102]` (enclosure quotient 8/3); the redshift reading
between the two reference copies is 1.4054 against a mid-layer scale ratio of 1.4, inside
`[1, 2]`. The readings are finite; the limit statements are the Lean theorems.

## Nonclaims

The scale-factor profile is a supplied datum: no source law selecting it is derived or used.
Cells are uniform and the tick is one; no physical clock and no physical identification of
the tick or the cell is made. Spatial curvature is not treated. The finite orders are the
flat comoving orders of the receipt.

## Reproduction

    python3 -m oph_exact.flrw_record_density_readout --check
    python3 -m pytest -q tests/test_exact_flrw_record_density_readout.py

The receipt `data/exact/flrw_record_density_readout.json` pins the source receipt by
SHA-256 and is regenerated with `--write`.
