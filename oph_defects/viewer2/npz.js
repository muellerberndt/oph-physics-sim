// Minimal reader for NumPy .npz archives (zip of .npy, stored or deflated) in the browser.
const DTYPES = {
  '|u1': Uint8Array, '|i1': Int8Array, '|b1': Uint8Array, '<u2': Uint16Array, '<i2': Int16Array,
  '<u4': Uint32Array, '<i4': Int32Array, '<f4': Float32Array, '<f8': Float64Array, '<i8': BigInt64Array, '<u8': BigUint64Array,
};

async function inflateRaw(bytes) {
  const ds = new DecompressionStream('deflate-raw');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function parseNpy(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = (major === 1 ? 10 : 12) + hlen;
  const header = new TextDecoder().decode(bytes.subarray(major === 1 ? 10 : 12, start));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const T = DTYPES[descr];
  if (!T) throw new Error('unsupported dtype ' + descr);
  const count = shape.reduce((a, b) => a * b, 1);
  let data = new T(bytes.buffer.slice(bytes.byteOffset + start, bytes.byteOffset + start + count * T.BYTES_PER_ELEMENT));
  if (T === BigInt64Array || T === BigUint64Array) data = Float64Array.from(data, Number);
  data.shape = shape;
  return data;
}

export async function loadNpz(url, only) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('fetch failed: ' + url);
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  let eocd = buf.byteLength - 22;
  while (eocd >= 0 && dv.getUint32(eocd, true) !== 0x06054b50) eocd--;
  if (eocd < 0) throw new Error('not a zip archive: ' + url);
  let count = dv.getUint16(eocd + 10, true), cdOff = dv.getUint32(eocd + 16, true);
  if (cdOff === 0xffffffff || count === 0xffff) {
    const z64 = Number(dv.getBigUint64(eocd - 20 + 8, true));
    count = Number(dv.getBigUint64(z64 + 32, true));
    cdOff = Number(dv.getBigUint64(z64 + 48, true));
  }
  const out = {};
  let p = cdOff;
  const jobs = [];
  for (let i = 0; i < count; i++) {
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true);
    const nlen = dv.getUint16(p + 28, true), xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
    let local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(buf, p + 46, nlen));
    let x = p + 46 + nlen;
    const xend = x + xlen;
    while (x + 4 <= xend) {
      const id = dv.getUint16(x, true), size = dv.getUint16(x + 2, true);
      if (id === 1) {
        let q = x + 4;
        if (usize === 0xffffffff) { usize = Number(dv.getBigUint64(q, true)); q += 8; }
        if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(q, true)); q += 8; }
        if (local === 0xffffffff) { local = Number(dv.getBigUint64(q, true)); }
      }
      x += 4 + size;
    }
    p = xend + clen;
    const key = name.replace(/\.npy$/, '');
    if (only && !only.includes(key)) continue;
    const ln = dv.getUint16(local + 26, true), lx = dv.getUint16(local + 28, true);
    const raw = new Uint8Array(buf, local + 30 + ln + lx, csize);
    jobs.push((async () => { out[key] = parseNpy(method === 0 ? raw : await inflateRaw(raw)); })());
  }
  await Promise.all(jobs);
  return out;
}
