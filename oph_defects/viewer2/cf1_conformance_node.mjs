// node cf1_conformance_node.mjs <dir> : runs the cases written by repair-law/work/SIM/cf1/cf1_conformance.py through the browser engine
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Routing } from './cf1_engine.js';
const dir = process.argv[2], cases = JSON.parse(readFileSync(dir + '/cases.json', 'utf8')), states = [];
for (const c of cases) {
  const w = new Routing(...c.shape); w.re.set(c.re); w.im.set(c.im);
  if (c.links) w.links = c.links.map((l) => Float64Array.from(l));
  if (c.links_im) w.linksIm = c.links_im.map((l) => Float64Array.from(l));
  const theta = typeof c.theta === 'number' ? c.theta : Float64Array.from(c.theta);
  for (let t = 0; t < c.ticks; t++) w.tickAlt(theta, c.kind);
  states.push({ re: Array.from(w.re), im: Array.from(w.im) });
}
const sha = createHash('sha256').update(readFileSync(fileURLToPath(new URL('./cf1_engine.js', import.meta.url)))).digest('hex');
writeFileSync(dir + '/node.json', JSON.stringify({ engine_sha256: sha, states }));
