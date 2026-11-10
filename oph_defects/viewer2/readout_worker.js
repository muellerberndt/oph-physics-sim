// Worker of the Readout tab: runs the continuum field (readout_field.js) off the main thread.
import { BornReadoutField as ContinuumField } from './readout_instrument.js';
import { packetObservables, invalidatePacketObservables } from './readout_observables.js';

let F = null;
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'setup') { F = new ContinuumField(m.n); self.postMessage({ type: 'ready' }); return; }
  if (!F) return;
  if (m.type === 'add') { for (const p of m.parts) F.add(p); }
  else if (m.type === 'remove') { for (const id of m.ids) F.parts.delete(id); }
  else if (m.type === 'clear') { F.parts.clear(); }
  else if (m.type === 'frame') {
    const observables = new Map([...F.parts].map(([id, p]) => [id, packetObservables(F, p)]));
    const r = F.frame(m.t, new Set(m.records || []), m.sigmaD);
    for (const row of r.stats) row.observables = observables.get(row.id);
    for (const row of r.records) invalidatePacketObservables(F.parts.get(row.id));
    self.postMessage({ type: 'frame', seq: m.seq, t: m.t, rgba: r.rgba, stats: r.stats, records: r.records }, [r.rgba.buffer]);
  }
};
