// node carrier_check.mjs : the engine against the published exact numbers of the public programs
import { Walk, EDGES, FACES, edgeIndex, addrIndex, regionOf, reachable, faceFlux, committedStep } from './walk_engine.js';
const near = (x, y, name) => { const ok = Math.abs(x - y) < 1e-12; console.log((ok ? 'ok   ' : 'FAIL ') + name + ': ' + x.toPrecision(12) + ' expected ' + y.toPrecision(12)); if (!ok) process.exitCode = 1; };
// 1. scalar walk: return to the starting port
let w = new Walk({ fibre: 1 }); w.embed(0, 0, 0);
w.tick(); near(w.populations().port[0], 11 / 12, 'return after one step');
w.tick(); near(w.populations().port[0], 643 / 900, 'return after two steps');
let pc = new Float64Array(12); pc[0] = 1; pc = committedStep(committedStep(pc)); near(pc[0], 101 / 120, 'committed rule after two attempts');
// 2. five-level fibre, one flipped link (0,1): three-step return amplitude of the doublet endpoint
const amp = (word, j) => { const x = new Walk({ fibre: 5, words: [word] }); x.embed(0, 0, j); x.tick(); x.tick(); x.tick(); return x.overlap(0, 0, j)[0]; };
near(amp(1, 3) - amp(0, 3), -1 / 6750, 'doublet: change of the three-step return amplitude by one flipped link');
near(amp(1, 0) - amp(0, 0), 0, 'triplet: no change');
const pr = (word) => { const x = new Walk({ fibre: 5, words: [word] }); x.embed(0, 0, 3); x.tick(); x.tick(); x.tick(); const a = x.overlap(0, 0, 3); return a[0] ** 2 + a[1] ** 2; };
near(pr(1) - pr(0), -9353 / 91125000, 'doublet: change of the three-step return probability');
// 3. defect and wave: region around port 5
const R = regionOf(5), words = reachable(1, R.interior);
console.log('reachable link words:', words.length, words.join(' '), ' interior edges:', R.interior.map((e) => EDGES[e].join('-')).join(' '));
const run = (g0) => { const x = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(g0); x.embed(g, 0, 3, Math.SQRT1_2); x.embed(g, 5, 3, Math.SQRT1_2); x.tick(); return x.basisProbability(0, 3, 0, addrIndex(0, 5)); };
near(run(1), 1 / 120, 'link word 1: wave record at port 0, level 3, blank record, address (0,5)');
near(run(3), 0, 'link word 3: the same record');
const defect = (port) => { const x = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(1); x.embed(g, port, 3); x.unstep(); x.tick(); const fp = x.faceProbabilities(R.faces); return R.faces.map((f) => FACES[f].join(',') + ': ' + fp[f].toFixed(6)).join('  '); };
console.log('defect after the tick, wave prepared to arrive at port 5: ', defect(5));
console.log('defect after the tick, wave prepared to arrive at port 6: ', defect(6));
const x = new Walk({ fibre: 5, words, interior: R.interior }); x.embed(words.indexOf(1), 0, 3, 0.6); x.embed(words.indexOf(1), 5, 1, 0.8);
for (let t = 0; t < 7; t++) x.tick();
const P = x.populations(); near(P.type[0], 0.64, 'charge sector weights are conserved: triplet'); near(P.type[1], 0.36, 'doublet');
near(P.port.reduce((a, b) => a + b, 0), 1, 'norm after seven ticks');

// 4. recording changes the result (coherent_defect_wave, GAUSS.md): positive Gauss sector, one tick, undo the moves
{
  const psi = new Walk({ fibre: 5, words: [1] }); psi.embed(0, 5, 3, Math.SQRT1_2); psi.embed(0, 6, 3, Math.SQRT1_2);
  const prep = () => { let x = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(1); x.embed(g, 5, 3, Math.SQRT1_2); x.embed(g, 6, 3, Math.SQRT1_2); x.unstep(); x = x.gaussProject(5, 1); x.normalize(); return x; };
  const a = prep(); a.tick(); a.unmoves();
  near(a.returnProbability(psi, 5), 1, 'reverse the coupling without a record: return probability');
  const b = prep(); b.tick();
  let ret = 0, hist = [];
  for (const br of b.faceBranches(R.faces)) { if (br.p < 1e-14) { hist.push(FACES[br.face].join(',') + ': 0'); continue; } br.state.unmoves(); const r = br.state.returnProbability(psi, 5); ret += r; hist.push(FACES[br.face].join(',') + ': yes ' + r.toFixed(4) + ' no ' + (br.p - r).toFixed(4)); }
  near(ret, 0.5, 'read the face of the flux first, then reverse: return probability');
  console.log('face and return histories:', hist.join(' | '));
  const c = prep(); const neg = (() => { let x = new Walk({ fibre: 5, words, interior: R.interior }); const g = words.indexOf(1); x.embed(g, 5, 3, Math.SQRT1_2); x.embed(g, 6, 3, Math.SQRT1_2); x.unstep(); return x.gaussProject(5, 1).norm2(); })();
  near(neg, 0.5, 'weight of the positive Gauss sector of a definite link word');
}

// 5. energy from one tick (theory session, lead L30): P(record not blank) = <x|L|x>/60
{ const a = new Walk({ fibre: 1 }); a.embed(0, 0, 0); a.tick(); near(a.recordWeight(), 1 / 12, 'correction probability of the first tick from one port'); }
