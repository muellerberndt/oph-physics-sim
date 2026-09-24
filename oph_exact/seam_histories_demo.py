"""Edge histories under the repair law: the values on collar seams after every sweep of one settled schedule (level 6)."""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from oph_exact import federation_huge as H
S = sys.argv[2]
geo = H.Geometry(S + "/geo_cache", 6)
loads = H.initial_loads(6, geo.ports)
lib = H.native_kernel()
x = np.ascontiguousarray(loads.astype(np.int8).copy())
tpl_a = np.ascontiguousarray(geo.template[:, 0], dtype=np.int8); tpl_b = np.ascontiguousarray(geo.template[:, 1], dtype=np.int8)
exp = H.expectation(geo, loads); v_min = int(exp["balanced_minimum"])
v = np.array([int(np.dot(x.astype(np.int64), x.astype(np.int64)))], dtype=np.int64); counts = np.zeros(5, dtype=np.int64)
rng = np.random.default_rng(H.schedule_seed(6, 0))
inter = geo.inter64; per = 4**6
collar = np.flatnonzero(inter[:, 0] // per != inter[:, 2] // per)
pick = np.random.default_rng(11).choice(collar, size=200, replace=False)
ia, ib = geo.ia[pick].astype(np.int64), geo.ib[pick].astype(np.int64)
hist = [np.stack([x[ia], x[ib]], axis=1).astype(np.int64)]
first = -1; sweep = 0
while first < 0 and sweep < 1000:
    seq, coin, _ = H.draw_sweep(rng, geo.seams, coins=True)
    first = int(lib.oph_huge_integer(H._ptr(x, ctypes.c_int8), H._ptr(tpl_a, ctypes.c_int8), H._ptr(tpl_b, ctypes.c_int8), H._ptr(geo.ia, ctypes.c_int32), H._ptr(geo.ib, ctypes.c_int32), geo.intra_count, H._ptr(seq, ctypes.c_int32), H._ptr(coin, ctypes.c_int8), geo.seams, H._ptr(counts, ctypes.c_int64), H._ptr(v, ctypes.c_int64), v_min))
    sweep += 1
    hist.append(np.stack([x[ia], x[ib]], axis=1).astype(np.int64))
h = np.stack(hist)  # (sweeps+1, 200, 2)
d = h[:, :, 0] - h[:, :, 1]  # seam difference per sweep
T = h.shape[0] - 1
# oscillation tests on the seam difference and on the endpoint values
sign_changes = np.mean([np.sum(np.diff(np.sign(d[:, k][d[:, k] != 0])) != 0) for k in range(200)])
# autocorrelation of the endpoint value increments at lags 1..4 (an oscillator gives negative lag-1..2 structure with a period; a random walk to agreement gives ~0)
inc = np.diff(h[:, :, 0], axis=0).astype(float)
ac = []
for lag in (1, 2, 3, 4):
    a, b = inc[:-lag].ravel(), inc[lag:].ravel()
    m = (a != 0) | (b != 0)
    ac.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else None)
# power spectrum of the mean seam difference magnitude vs sweep, and of individual endpoints
mag = np.abs(d).mean(axis=1)
spec = np.abs(np.fft.rfft(h[:, :, 0] - h[:, :, 0].mean(axis=0), axis=0)) ** 2
mean_spec = spec.mean(axis=1)
peak_bin = int(np.argmax(mean_spec[1:]) + 1)
out = {"level": 6, "seed": H.schedule_seed(6, 0), "sweeps": T, "collar_seams_sampled": 200,
       "mean_abs_seam_difference_by_sweep": [round(float(m), 4) for m in mag[: min(T + 1, 70)]],
       "mean_sign_changes_of_seam_difference_per_seam": round(float(sign_changes), 3),
       "endpoint_increment_autocorrelation_lags_1_to_4": [None if a is None else round(a, 4) for a in ac],
       "endpoint_value_power_spectrum_peak_bin_of_%d" % (T // 2 + 1): peak_bin,
       "endpoint_value_power_spectrum_first_6_over_total": [round(float(mean_spec[k] / mean_spec[1:].sum()), 4) for k in range(1, 7)],
       "fraction_of_sweeps_with_any_change_on_sampled_seams": round(float(np.mean(np.any(np.diff(h, axis=0) != 0, axis=(1, 2)))), 3)}
json.dump(out, open(S + "/seam_histories_L6.json", "w"), indent=1)
print(json.dumps(out)[:900])
