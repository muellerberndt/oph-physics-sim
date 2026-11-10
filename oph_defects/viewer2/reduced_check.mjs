// node reduced_check.mjs : the reduced walk against the full engine and the published numbers
import { ReducedWalk } from './reduced_walk.js';
import { Walk, EDGES } from './walk_engine.js';
const near = (x, y, name, tol = 1e-12) => { const ok = Math.abs(x - y) < tol; console.log((ok ? 'ok   ' : 'FAIL ') + name + ': ' + x.toPrecision(12) + ' expected ' + y.toPrecision(12)); if (!ok) process.exitCode = 1; };
const sa = Int32Array.from(EDGES.map((e) => e[0])), sb = Int32Array.from(EDGES.map((e) => e[1]));
const r = new ReducedWalk(12, sa, sb, 60); r.embed(0); const pop = new Float64Array(12);
const f = new Walk({ fibre: 1 }); f.embed(0, 0, 0);
r.step(); f.tick(); let rec = r.populations(pop);
near(pop[0], 11 / 12, 'one carrier, return after one step'); near(rec, 1 / 12, 'correction record after one step');
r.step(); f.tick(); rec = r.populations(pop); near(pop[0], 643 / 900, 'return after two steps');
let worst = 0;
for (let t = 0; t < 40; t++) { r.step(); f.tick(); const R = r.populations(pop), F = f.populations().port; for (let p = 0; p < 12; p++) worst = Math.max(worst, Math.abs(pop[p] - F[p])); worst = Math.max(worst, Math.abs(R - f.recordWeight())); }
near(worst, 0, 'largest difference to the full engine over 40 further steps (populations and record weight)', 1e-11);
// with link signs: three flipped links, spinor level
const word = 2097185, zr = Float64Array.from(EDGES.map((_, i) => ((word >>> i) & 1) ? -1 : 1));
const r2 = new ReducedWalk(12, sa, sb, 60, zr); r2.embed(0); const f2 = new Walk({ fibre: 5, words: [word] }); f2.embed(0, 0, 3);
let w2 = 0; for (let t = 0; t < 30; t++) { r2.step(); f2.tick(); r2.populations(pop); const F = f2.populations().port; for (let p = 0; p < 12; p++) w2 = Math.max(w2, Math.abs(pop[p] - F[p])); }
near(w2, 0, 'the same with three flipped links on a spinor level', 1e-11);
