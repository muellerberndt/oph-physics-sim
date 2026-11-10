// node story_rule_check.mjs : statistics of the one-carrier rule over many seeds, against the run's catalogue
import { Carrier, SEAMS, OFACES } from './story_rule.js';
const bad = SEAMS.filter((s) => s.fp < 0 || s.fm < 0 || s.fp === s.fm).length;
console.log('seams with both faces found:', SEAMS.length - bad, 'of', SEAMS.length, '; faces', OFACES.length);
const tot = { birth: 0, move: 0, annihilation: 0, fusion: 0, split: 0, exchange: 0 }; let lastBirth = 0, endDef = 0, endMobile = 0, sumCheck = 0, births = { '1,5': 0, '2,4': 0, '3,3': 0 }; const N = 4000;
const endComp = {};
for (let seed = 1; seed <= N; seed++) {
  const c = new Carrier(seed);
  for (let sw = 0; sw < 200; sw++) for (const e of c.runSweep()) { tot[e.kind]++; if (e.kind === 'birth') { lastBirth = Math.max(lastBirth, sw); const k = Math.min(e.after[0], e.after[1]); births[k + ',' + (6 - k)]++; } }
  endDef += c.defects(); endMobile += c.mobile(); sumCheck += c.cls.reduce((s, k) => s + k, 0) % 6;
  const comp = Array.from(c.cls).filter((k) => k).sort().join(','); endComp[comp || 'none'] = (endComp[comp || 'none'] || 0) + 1;
}
console.log('events per carrier over 200 sweeps:', Object.fromEntries(Object.entries(tot).map(([k, v]) => [k, +(v / N).toFixed(3)])));
console.log('births by kind:', births, ' latest birth at sweep', lastBirth);
console.log('defect faces per carrier at the end:', (endDef / N).toFixed(4), ' mobile (classes 1, 5):', (endMobile / N).toFixed(4), ' carriers whose classes do not sum to 0 mod 6:', sumCheck);
console.log('end compositions (top 8):', Object.entries(endComp).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, v]) => `{${k}} ${(100 * v / N).toFixed(1)}%`).join('  '));
