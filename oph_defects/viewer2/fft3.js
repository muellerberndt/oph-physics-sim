// Radix-2 complex FFT along the three axes of an n^3 grid (n a power of two); separate real and imaginary arrays.
// Index of a grid point: (z * n + y) * n + x, so x runs fastest (the layout of a WebGL 3D texture).
// sign -1: forward, sum_x f(x) e^{-ikx}; sign +1: inverse, (1/n^3) sum_k F(k) e^{ikx}.
export function makeFFT3(n) {
  const bits = Math.round(Math.log2(n)), rev = new Int32Array(n);
  if (1 << bits !== n) throw new Error('n must be a power of two');
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  const cw = new Float64Array(n / 2), sw = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) { cw[k] = Math.cos(2 * Math.PI * k / n); sw[k] = Math.sin(2 * Math.PI * k / n); }
  const lr = new Float64Array(n), li = new Float64Array(n);
  function line(re, im, off, stride, sign) {
    for (let i = 0; i < n; i++) { const j = off + rev[i] * stride; lr[i] = re[j]; li[i] = im[j]; }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) for (let k = 0; k < half; k++) {
        const wr = cw[k * step], wi = sign * sw[k * step], a = s + k, b = a + half;
        const tr = wr * lr[b] - wi * li[b], ti = wr * li[b] + wi * lr[b];
        lr[b] = lr[a] - tr; li[b] = li[a] - ti; lr[a] += tr; li[a] += ti;
      }
    }
    for (let i = 0; i < n; i++) { const j = off + i * stride; re[j] = lr[i]; im[j] = li[i]; }
  }
  return function fft3(re, im, sign) {
    const n2 = n * n;
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) line(re, im, a * n2 + b * n, 1, sign);
    for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) line(re, im, a * n2 + c, n, sign);
    for (let b = 0; b < n; b++) for (let c = 0; c < n; c++) line(re, im, b * n + c, n2, sign);
    if (sign > 0) { const f = 1 / (n * n2); for (let i = 0; i < re.length; i++) { re[i] *= f; im[i] *= f; } }
  };
}
