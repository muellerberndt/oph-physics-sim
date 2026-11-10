// The findings page: the current measured facts as cards, and the ladder of runs.
import { $, S, fmt } from './app.js';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const TAG = { measured: '#3fe0d0', exact: '#5fe3a1', verified: '#5fe3a1', computed: '#6fb1ff', declared: '#ffb44a', 'forced by data': '#ffe45c', scanned: '#b197ff', audit: '#b197ff', proposal: '#ff5fb0', open: '#93a3c0', partly: '#6fb1ff', running: '#ffb44a', done: '#5fe3a1' };
export async function renderFindings() {
  const host = $('findings'); if (!host) return;
  const F = await (await fetch('/findings.json')).json();
  let h = `<h2 style="margin-top:0">Findings</h2><p>${esc(F.lead)} <span class="small">Updated ${esc(F.updated)}.</span></p>`;
  for (const sec of F.sections) {
    h += `<div class="f-sec" style="--c:${sec.color}"><h3>${esc(sec.title)}</h3><p class="small" style="margin:0 0 10px">${esc(sec.blurb)}</p><div class="f-grid">`;
    for (const c of sec.cards) h += `<div class="f-card"><div class="f-val">${esc(c.value)}<span>${esc(c.unit)}</span></div>
      <div class="f-title">${esc(c.title)} <i style="color:${TAG[c.tag] || '#93a3c0'};border-color:${TAG[c.tag] || '#93a3c0'}55">${esc(c.tag)}</i></div><div class="f-text">${esc(c.text)}</div></div>`;
    h += '</div></div>';
  }
  h += `<div class="f-sec" style="--c:#c5d0e6"><h3>The ladder of exact runs</h3><p class="small" style="margin:0 0 10px">${esc(F.runs_note)}</p>
    <table class="t"><tr><th>run</th><th>level</th><th>observers</th><th>ports</th><th>seams</th><th>sweeps</th><th>slices</th><th>identities</th><th>law</th><th>markers drawn</th></tr>`;
  for (const r of S.runs) h += `<tr><td><a href="?run=${encodeURIComponent(r.name)}"><b>${esc(r.name)}</b></a></td><td class="num">${r.level}</td><td class="num">${fmt(r.carriers)}</td><td class="num">${fmt(r.ports)}</td>
    <td class="num">${fmt(r.seams)}</td><td class="num">${fmt(r.sweeps)}</td><td class="num">${fmt(r.cuts)}</td><td class="num">${fmt(r.identities)}</td>
    <td>${r.twists ? 'closed-loop extension' : 'law C, exact'}</td><td>${r.sample > 1 ? '1 in ' + r.sample : 'all'}</td></tr>`;
  host.innerHTML = h + '</table></div>';
}
