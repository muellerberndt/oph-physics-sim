"""Density correlations of a canonical series against the law of the native operator.

    python -m oph_defects.ou_test runs/series/L3_canonical_1-3 --modes 24

For a stationary run with density snapshots (``--series-every``), the projection of the load field on an
eigenvector f_j of the port Laplacian L should decorrelate as exp(-lambda_j s / 2) after s sweeps (one attempt
acts on the mean as I - L / (2 M); a sweep is M attempts), with stationary variance chi * (V / (V - 1)) for a
unit, mean-zero f_j under the uniform fixed-number law, chi = rho (1 - rho). In scaled time
tau = s / (2 * 4^level) the rate is 4^level lambda_j.

Reported per mode: predicted rate, measured rate from a weighted fit of the log autocorrelation, their ratio
with a block standard error, the variance ratio and the excess kurtosis (zero for a Gaussian).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spl

from . import fedgeom as FG
from .dataset import Dataset
from .gluing_control import laplacian


def modes(level: int, k: int):
    geo = FG.build(level)
    L = laplacian(geo)
    vals, vecs = spl.eigsh(L, k=k + 1, sigma=-1e-4 / 4 ** level, which='LM')
    order = np.argsort(vals)
    return vals[order][1:], vecs[:, order][:, 1:]


def autocorr(y: np.ndarray, max_lag: int) -> np.ndarray:
    y = y - y.mean()
    n = y.size
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:max_lag + 1] / (n - np.arange(max_lag + 1))
    return acf / acf[0]


def fit_rate(y: np.ndarray, dt: float, predicted: float) -> float:
    """Decay rate per sweep from the regression of the projection on itself at one lag.

    The conditional mean of law C is exactly linear, so sum a(t) a(t + lag) / sum a(t)^2 estimates
    exp(-rate * lag) without the bias of fitting the logarithm of a noisy autocorrelation. The lag is about
    0.3 predicted decay times.
    """
    k = max(1, min(y.size // 4, int(round(0.3 / (predicted * dt)))))
    y = y - y.mean()
    r = float((y[:-k] * y[k:]).sum() / (y[:-k] ** 2).sum())
    return float(-np.log(r) / (k * dt)) if r > 0 else float('nan')


def _simulate(spec):
    """One canonical run with on-the-fly projections on the eigenvectors (no files)."""
    from .universe import COSTS, Engine, prepare_loads, schedule_seed
    level, seed, sweeps, every, k = spec
    geo = FG.build(level)
    lam, vec = modes(level, k)
    loads, _ = prepare_loads(f'canonical:1/3:{seed}', level, geo.ports)
    eng = Engine(geo, loads, schedule_seed(level, 5000 + seed), COSTS['binding'], 5, False, slots=2, register=False, fast_rng=True)
    vec32 = vec.astype(np.float32)
    out = np.zeros((sweeps // every, k), np.float32)
    for s in range(sweeps):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
        if (s + 1) % every == 0:
            out[(s + 1) // every - 1] = (eng.x.astype(np.float32) - np.float32(1 / 3)) @ vec32
    return out


def simulate_main(a):
    from multiprocessing import Pool
    lam, vec = modes(a.level, a.modes)
    with Pool(a.workers) as pool:
        Ys = pool.map(_simulate, [(a.level, seed, a.simulate, a.every, a.modes) for seed in range(a.seed_offset, a.seed_offset + a.seeds)])
    V = 240 * 4 ** a.level
    chi = (1 / 3) * (2 / 3) * V / (V - 1)
    rows = []
    f = 1 / 3
    cross = []
    for Y in Ys:
        c = np.corrcoef(Y.T)
        cross.append(np.abs(c[np.triu_indices(a.modes, 1)]).mean())
    for j in range(a.modes):
        pred = lam[j] / 2
        rates = np.array([fit_rate(Y[:, j], a.every, pred) for Y in Ys])
        rates = rates[np.isfinite(rates)]
        allv = np.concatenate([Y[:, j] - Y[:, j].mean() for Y in Ys])
        kurt_pred = float((vec[:, j] ** 4).sum() * (1 - 6 * f * (1 - f)) / (f * (1 - f)))
        rows.append({'mode': j + 1, 'scaled_eigenvalue': float(4 ** a.level * lam[j]), 'rate_predicted_per_sweep': float(pred),
                     'rate_measured_per_sweep': float(rates.mean()), 'ratio': float(rates.mean() / pred),
                     'ratio_standard_error': float(rates.std(ddof=1) / np.sqrt(rates.size) / pred) if rates.size > 1 else None,
                     'decay_times_per_seed': float(a.simulate * pred), 'variance_ratio': float(allv.var() / chi),
                     'excess_kurtosis': float((allv ** 4).mean() / allv.var() ** 2 - 3), 'excess_kurtosis_predicted': kurt_pred})
    print(f'level {a.level}: {a.seeds} seeds x {a.simulate} sweeps ({a.simulate / (2 * 4 ** a.level):.0f} tau each), projection on port eigenvectors every {a.every} sweeps')
    print(f"{'mode':>4}{'4^l lambda':>12}{'rate pred':>12}{'rate meas':>12}{'ratio':>8}{'+-':>7}{'decay times':>12}{'variance':>9}{'kurtosis':>9}")
    for r in rows:
        print(f"{r['mode']:>4}{r['scaled_eigenvalue']:12.6f}{r['rate_predicted_per_sweep']:12.3e}{r['rate_measured_per_sweep']:12.3e}{r['ratio']:8.3f}"
              f"{r['ratio_standard_error']:7.3f}{r['decay_times_per_seed']:12.1f}{r['variance_ratio']:9.3f}{r['excess_kurtosis']:9.3f}")
    ratios = np.array([r['ratio'] for r in rows]); ses = np.array([r['ratio_standard_error'] for r in rows])
    w = 1 / ses ** 2
    print(f'weighted mean ratio over {a.modes} modes: {(w * ratios).sum() / w.sum():.4f} +- {1 / np.sqrt(w.sum()):.4f}; '
          f'mean variance ratio {np.mean([r["variance_ratio"] for r in rows]):.4f}; mean excess kurtosis {np.mean([r["excess_kurtosis"] for r in rows]):+.4f} '
          f'(predicted {np.mean([r["excess_kurtosis_predicted"] for r in rows]):+.5f}); mean |cross-mode correlation| {np.mean(cross):.4f}')
    exact = {}
    for lag in (0.3, 0.05, 1.0):
        er = exact_test(Ys, lam, a.every, chi, lag)
        dev = np.array([r['relative_rate_deviation'] for r in er]); se = np.array([r['standard_error'] for r in er])
        w = 1 / se ** 2
        var = np.array([r['variance_ratio'] for r in er]); vse = np.array([r['variance_ratio_standard_error'] for r in er])
        wv = 1 / vse ** 2
        exact[f'lag_{lag}'] = {'modes': er, 'weighted_mean_relative_rate_deviation': float((w * dev).sum() / w.sum()),
                               'standard_error': float(1 / np.sqrt(w.sum())), 'chi2_of_zero': float(((dev / se) ** 2).sum()),
                               'weighted_mean_variance_ratio': float((wv * var).sum() / wv.sum()), 'variance_standard_error': float(1 / np.sqrt(wv.sum()))}
        o = exact[f'lag_{lag}']
        print(f"bias-free test, lag {lag} decay times: rate deviation {100 * o['weighted_mean_relative_rate_deviation']:+.3f} +- {100 * o['standard_error']:.3f} percent "
              f"(chi2 against zero {o['chi2_of_zero']:.1f} for {a.modes} modes); variance ratio {o['weighted_mean_variance_ratio']:.4f} +- {o['variance_standard_error']:.4f}")
    if a.json:
        np.savez_compressed(a.json.with_suffix('.npz'), **{f'seed{i}': Y for i, Y in enumerate(Ys)}, eigenvalues=lam)
        a.json.write_text(json.dumps({'level': a.level, 'seeds': a.seeds, 'sweeps_per_seed': a.simulate, 'every': a.every, 'modes': rows, 'bias_free_test': exact}))


def exact_test(Ys, lam, every: int, chi: float, lag_decay_times: float = 0.3):
    """Test of the predicted rates without estimator bias.

    Under law C the conditional mean of a projection is exactly linear: E[a(t + s) | a(t)] = exp(-rate s) a(t)
    (to O(lambda^2 / M)), and its stationary mean is exactly zero because the eigenvector is orthogonal to the
    conserved total. With the predicted rate, the residual e(t) = a(t + s) - exp(-rate_pred s) a(t) is therefore
    uncorrelated with a(t): N = sum a(t) e(t) has mean exactly zero if the prediction is right, with no
    correction of order 1 / (run length). No sample mean is subtracted anywhere. The implied relative rate
    deviation is -N / (D rho s rate_pred) with D = sum a(t)^2, and its error comes from the scatter of N over
    independent seeds. The variance ratio uses mean(a^2) against chi.
    """
    rows = []
    for j in range(lam.size):
        pred = lam[j] / 2
        k = max(1, int(round(lag_decay_times / (pred * every))))
        rho = float(np.exp(-pred * k * every))
        N = np.array([float(((Y[k:, j].astype(np.float64) - rho * Y[:-k, j]) * Y[:-k, j]).sum()) for Y in Ys])
        D = np.array([float((Y[:-k, j].astype(np.float64) ** 2).sum()) for Y in Ys])
        scale = D.sum() * rho * k * every * pred
        dev = -N.sum() / scale
        se = float(np.sqrt(N.size * N.var(ddof=1)) / scale) if N.size > 1 else float('nan')
        v = np.array([float((Y[:, j].astype(np.float64) ** 2).mean()) for Y in Ys]) / chi
        rows.append({'mode': j + 1, 'rate_predicted_per_sweep': float(pred), 'lag_sweeps': k * every,
                     'relative_rate_deviation': float(dev), 'standard_error': se, 'z': float(dev / se) if se > 0 else float('nan'),
                     'variance_ratio': float(v.mean()), 'variance_ratio_standard_error': float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float('nan')})
    return rows


def reanalyse_main(a):
    z = np.load(a.reanalyse)
    lam = z['eigenvalues']
    Ys = [z[k] for k in sorted((k for k in z.files if k.startswith('seed')), key=lambda s: int(s[4:]))]
    V = 240 * 4 ** a.level
    chi = (1 / 3) * (2 / 3) * V / (V - 1)
    out = {}
    for lag in (0.3, 0.05, 1.0):
        rows = exact_test(Ys, lam, a.every, chi, lag)
        dev = np.array([r['relative_rate_deviation'] for r in rows]); se = np.array([r['standard_error'] for r in rows])
        w = 1 / se ** 2
        var = np.array([r['variance_ratio'] for r in rows]); vse = np.array([r['variance_ratio_standard_error'] for r in rows])
        wv = 1 / vse ** 2
        out[f'lag_{lag}'] = {'modes': rows, 'weighted_mean_relative_rate_deviation': float((w * dev).sum() / w.sum()),
                             'standard_error': float(1 / np.sqrt(w.sum())), 'chi2_of_zero': float(((dev / se) ** 2).sum()),
                             'weighted_mean_variance_ratio': float((wv * var).sum() / wv.sum()), 'variance_standard_error': float(1 / np.sqrt(wv.sum()))}
        o = out[f'lag_{lag}']
        print(f"lag {lag} decay times: rate deviation {100 * o['weighted_mean_relative_rate_deviation']:+.3f} +- {100 * o['standard_error']:.3f} percent "
              f"(chi2 against zero {o['chi2_of_zero']:.1f} for {lam.size} modes); variance ratio {o['weighted_mean_variance_ratio']:.4f} +- {o['variance_standard_error']:.4f}")
        if lag == 0.3:
            print(f"{'mode':>4}{'rate pred':>12}{'lag':>6}{'deviation %':>12}{'+-':>7}{'z':>6}{'variance':>9}{'+-':>7}")
            for r in rows:
                print(f"{r['mode']:>4}{r['rate_predicted_per_sweep']:12.3e}{r['lag_sweeps']:>6}{100 * r['relative_rate_deviation']:12.2f}{100 * r['standard_error']:7.2f}"
                      f"{r['z']:6.1f}{r['variance_ratio']:9.3f}{r['variance_ratio_standard_error']:7.3f}")
    if a.json:
        a.json.write_text(json.dumps({'source': str(a.reanalyse), 'level': a.level, 'every': a.every, 'seeds': len(Ys), **out}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, nargs='?')
    p.add_argument('--simulate', type=int, default=0, help='run fresh canonical universes of this many sweeps and project on the fly')
    p.add_argument('--level', type=int, default=3)
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--every', type=int, default=2)
    p.add_argument('--seed-offset', type=int, default=0, help='first preparation and schedule seed of a --simulate run')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--modes', type=int, default=24)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--json', type=Path, default=None)
    p.add_argument('--reanalyse', type=Path, default=None, help='saved projections (.npz of a --simulate run): bias-free test of the predicted rates')
    a = p.parse_args(argv)
    if a.reanalyse:
        return reanalyse_main(a)
    if a.simulate:
        return simulate_main(a)
    ds = Dataset(a.run)
    level, every = ds.level, ds.meta['config']['series_every']
    geo = ds.geometry
    V = geo.ports
    lam, vec = modes(level, a.modes)
    rho = ds.meta['totals']['total_load'] / V
    chi = rho * (1 - rho) * V / (V - 1)
    proj_cells = vec.reshape(geo.carriers, 12, -1).mean(axis=1)            # cell-constant part of each eigenvector
    cell_weight = (12 * (proj_cells ** 2).sum(axis=0))                      # fraction of the mode carried by cell totals
    Y, sweeps, source = [], [], None
    for block in ds.series():
        if 'ports' in block:
            source = 'ports'
            Y.append((block['ports'].astype(np.float64) - rho) @ vec)
        else:
            source = 'cells'
            Y.append((block['cells'].astype(np.float64) - 12 * rho) @ proj_cells)
        sweeps.append(block['sweep'])
    Y = np.concatenate(Y); sweeps = np.concatenate(sweeps)
    n = Y.shape[0]
    rows = []
    for j in range(a.modes):
        pred = lam[j] / 2                                                  # per sweep
        y = Y[:, j]
        rate = fit_rate(y, every, pred)
        parts = [fit_rate(seg, every, pred) for seg in np.array_split(y, a.blocks)]
        parts = [v for v in parts if np.isfinite(v)]
        se = float(np.std(parts) / np.sqrt(len(parts))) if len(parts) > 1 else float('nan')
        var_pred = chi if source == 'ports' else chi * cell_weight[j]
        yc = y - y.mean()
        rows.append({'mode': j + 1, 'scaled_eigenvalue': float(4 ** level * lam[j]), 'rate_predicted_per_sweep': float(pred),
                     'rate_measured_per_sweep': rate, 'ratio': rate / pred, 'ratio_block_standard_error': se / pred,
                     'decay_times_in_series': float(n * every * pred), 'variance_ratio': float(yc.var() / var_pred),
                     'excess_kurtosis': float((yc ** 4).mean() / yc.var() ** 2 - 3), 'cell_weight': float(cell_weight[j])})
    out = {'run': str(a.run), 'level': level, 'snapshots': int(n), 'sweeps_between_snapshots': every, 'source': source,
           'density': rho, 'chi': chi, 'modes': rows}
    print(f'level {level}: {n} snapshots every {every} sweeps ({n * every / (2 * 4 ** level):.1f} tau), projections from {source}, density {rho:.4f}')
    print(f"{'mode':>4}{'4^l lambda':>12}{'rate pred':>12}{'rate meas':>12}{'ratio':>8}{'+-':>7}{'decay times':>12}{'variance':>9}{'kurtosis':>9}")
    for r in rows:
        print(f"{r['mode']:>4}{r['scaled_eigenvalue']:12.6f}{r['rate_predicted_per_sweep']:12.3e}{r['rate_measured_per_sweep']:12.3e}{r['ratio']:8.3f}"
              f"{r['ratio_block_standard_error']:7.3f}{r['decay_times_in_series']:12.1f}{r['variance_ratio']:9.3f}{r['excess_kurtosis']:9.3f}")
    if a.json:
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
