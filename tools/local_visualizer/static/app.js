"use strict";

const VIEW_DEFS = [
  {
    id: "screen",
    label: "Screen geometry",
    kicker: "Primary view · exact finite carrier",
    title: "Icosahedral screen federation",
    summary: "One exact 12-port, 30-edge, 20-face carrier is shown beside a windowed census of the larger federation. Typed seams and the global S2 support remain distinct from each local cell."
  },
  {
    id: "a5sm",
    label: "A5 → SM ladder",
    kicker: "Primary view · gated dependency DAG",
    title: "A5 currents and Standard-Model ladder",
    summary: "The exact A5 action and 1 + 3 + 3-prime + 5 sectors feed a dependency DAG. Physical status and display status remain separate; the Q2-H and Q2-E-plus-positivity routes are alternatives."
  },
  {
    id: "observer",
    label: "Observer repair / camera",
    kicker: "Self-reading patch · record ancestry",
    title: "Repair transaction to observer-visible record",
    summary: "Port readback, overlap mismatch, proof-carrying repair, atomic commit, semantic record, and observer checkpoint are linked without exposing hidden global coordinates in the subjective camera."
  },
  {
    id: "emergent",
    label: "Universe / gravity / matter",
    kicker: "Downstream diagnostic layers",
    title: "Emergent universe, gravity, particles, and atoms",
    summary: "Exported H3 objects, defect worldlines, curvature proxies, particle candidates, and atom census rows remain independently gated. Beautiful geometry is not a gravity, particle, or atomic-emergence receipt."
  },
  {
    id: "cosmology",
    label: "Cosmology / CMB",
    kicker: "Measurement-facing comparison",
    title: "Observer-facing cosmology and CMB diagnostics",
    summary: "Finite screen rows, pinned observational references, residuals, and assumed dS4 layers are shown with distinct provenance. Diagnostic resemblance, usable comparison, and physical prediction are separate gates."
  },
  {
    id: "census",
    label: "Census / provenance",
    kicker: "Progressive local data access",
    title: "Addressable finite census and manifested sidecars",
    summary: "Select any stable census index or request one manifested row/byte page. Only the visible chunk is decoded and drawn; the interface never claims that the entire finite population is literally rendered at once."
  }
];

const COLORS = {
  bg: "#07131f", grid: "#173049", silver: "#b9c6d8", teal: "#42d6c7",
  amber: "#f5c66b", gold: "#f2b84b", coral: "#ff6b6b", violet: "#b39dff",
  blue: "#60a5fa", magenta: "#f472b6", muted: "#72879d", text: "#e8f1fa"
};

const state = {
  mode: "RECEIPT",
  view: "screen",
  summary: null,
  manifest: null,
  forcedStages: new Set(),
  forceAll: false,
  selectedStage: null,
  selectedA5Action: null,
  selectedCensusIndex: 0,
  visibleRows: [],
  visibleRowsSource: "none",
  visibleRowsKind: null,
  visibleBinaryBytes: null,
  hitTargets: [],
  animationTime: 0,
  animationFrame: 0,
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches
};

const el = {};

window.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  cacheElements();
  buildTabs();
  attachEvents();
  try {
    const [summary, manifest] = await Promise.all([
      fetchJson("/api/summary"),
      fetchJson("/api/manifest")
    ]);
    state.summary = summary;
    state.manifest = manifest;
    state.forceAll = ladder()?.demoControls?.forceAllStages === true;
    el.forceAll.checked = state.forceAll;
    document.title = `${summary.title || "OPH"} · local visualizer`;
    el.runTitle.textContent = summary.title || "Local visualization payload";
    el.claimBoundary.textContent = summary.claimBoundary ||
      "Every physical claim remains controlled by its canonical receipt.";
    populateCensusKinds();
    populateSidecars();
    buildStageToggles();
    await loadVisibleWindow();
    setMode(requestedLocalMode());
  } catch (error) {
    el.runTitle.textContent = "Payload could not be loaded";
    el.viewSummary.textContent = String(error);
    drawMessage("Local payload/API error", String(error), COLORS.coral);
  }
}

function cacheElements() {
  const ids = [
    "run-title", "receipt-mode", "demo-mode", "demo-banner", "view-tabs", "view-kicker",
    "view-title", "view-summary", "view-status", "scene", "render-disclosure", "census-kind",
    "census-start", "census-count", "load-window", "census-readout", "demo-controls",
    "force-all", "reset-demo", "stage-toggles", "selection-details", "clock-strip", "stage-table",
    "manifest-status", "sidecar-select", "row-offset", "row-limit", "load-sidecar",
    "sidecar-preview", "claim-boundary"
  ];
  for (const id of ids) el[toCamel(id)] = document.getElementById(id);
}

function toCamel(value) {
  return value.replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
}

function buildTabs() {
  for (const view of VIEW_DEFS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "view-tab";
    button.dataset.view = view.id;
    button.textContent = view.label;
    button.addEventListener("click", () => {
      state.view = view.id;
      renderAll();
    });
    el.viewTabs.append(button);
  }
}

function attachEvents() {
  el.receiptMode.addEventListener("click", () => setMode("RECEIPT"));
  el.demoMode.addEventListener("click", () => setMode("DEMO_ASSUMPTION"));
  el.forceAll.addEventListener("change", () => {
    state.forceAll = el.forceAll.checked;
    renderAll();
  });
  el.resetDemo.addEventListener("click", () => {
    state.forceAll = false;
    state.forcedStages.clear();
    el.forceAll.checked = false;
    for (const input of el.stageToggles.querySelectorAll("input")) input.checked = false;
    renderAll();
  });
  el.loadWindow.addEventListener("click", loadVisibleWindow);
  el.censusStart.addEventListener("change", loadVisibleWindow);
  el.censusCount.addEventListener("change", loadVisibleWindow);
  el.censusKind.addEventListener("change", () => {
    el.censusStart.value = "0";
    state.visibleRows = [];
    loadVisibleWindow();
  });
  el.loadSidecar.addEventListener("click", loadSidecarPage);
  el.scene.addEventListener("click", selectCanvasTarget);
  window.addEventListener("resize", scheduleCanvas);
  document.addEventListener("visibilitychange", updateAnimationLoop);
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  motionQuery.addEventListener("change", event => {
    state.reducedMotion = event.matches;
    updateAnimationLoop();
    renderAll();
  });
}

function setMode(mode) {
  state.mode = mode === "DEMO_ASSUMPTION" ? "DEMO_ASSUMPTION" : "RECEIPT";
  const demo = state.mode === "DEMO_ASSUMPTION";
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("mode", demo ? "demo" : "receipt");
  window.history.replaceState(null, "", url);
  el.receiptMode.classList.toggle("active", !demo);
  el.receiptMode.classList.remove("demo-active");
  el.receiptMode.setAttribute("aria-pressed", String(!demo));
  el.demoMode.classList.toggle("active", demo);
  el.demoMode.classList.toggle("demo-active", demo);
  el.demoMode.setAttribute("aria-pressed", String(demo));
  el.demoBanner.classList.toggle("hidden", !demo);
  el.demoControls.classList.toggle("hidden", !demo);
  renderAll();
}

function requestedLocalMode() {
  const requested = new URLSearchParams(window.location.search).get("mode");
  return requested === "demo" ? "DEMO_ASSUMPTION" : "RECEIPT";
}

function sections() {
  return state.summary?.sections || {};
}

function ladder() {
  return sections().screenA5Ladder || {};
}

function stages() {
  const rows = ladder()?.a5ToSm?.stageNodes;
  return Array.isArray(rows) ? rows.filter(row => row && !row.truncated) : [];
}

function h3KmsStages() {
  const rows = ladder()?.physicalH3KmsDemoOverlay?.stageNodes;
  return Array.isArray(rows)
    ? rows.filter(row => row && !row.truncated).map(row => ({ ...row, stageFamily: "physical_h3_kms" }))
    : [];
}

function auditStages() {
  return [...h3KmsStages(), ...stages()];
}

function displayForced(stageId) {
  return state.mode === "DEMO_ASSUMPTION" && (state.forceAll || state.forcedStages.has(stageId));
}

function physicalPassed(stage) {
  if (stage?.stageFamily === "physical_h3_kms") {
    const trusted = ladder()?.physicalH3KmsDemoOverlay?.physicalSnapshotTrusted === true;
    return trusted && stage?.physicalPassed === true
      && stage?.physicalGateStatus === "PASS"
      && stage?.physicalScientificStatus === "VALID_PASS";
  }
  const trustedSnapshot = ladder()?.receipts?.PHYSICAL_A5_SM_SNAPSHOT_TRUSTED === true;
  return trustedSnapshot && stage?.physicalPassed === true && stage?.physicalStatus === "PASS";
}

function physicalStatusLabel(stage) {
  if (!stage) return "—";
  if (stage.stageFamily === "physical_h3_kms") {
    const trusted = ladder()?.physicalH3KmsDemoOverlay?.physicalSnapshotTrusted === true;
    if (!trusted) return "UNTRUSTED SNAPSHOT";
    const gate = typeof stage.physicalGateStatus === "string" ? stage.physicalGateStatus : "NOT_EVALUATED";
    const science = typeof stage.physicalScientificStatus === "string" ? stage.physicalScientificStatus : "NOT_EVALUATED";
    return `${gate} · ${science}`;
  }
  const trustedSnapshot = ladder()?.receipts?.PHYSICAL_A5_SM_SNAPSHOT_TRUSTED === true;
  const declared = typeof stage.physicalStatus === "string"
    ? stage.physicalStatus.toUpperCase()
    : "OPEN";
  if (!trustedSnapshot) return "UNTRUSTED SNAPSHOT";
  if ((stage.physicalPassed === true) !== (declared === "PASS")) return "INCONSISTENT";
  return stage.physicalPassed === true ? "PASS" : declared;
}

function displayStatus(stage) {
  if (displayForced(stage.stageId) && !physicalPassed(stage)) return "forced_demo_assumption";
  return physicalPassed(stage) ? "computed" : "blocked";
}

function buildStageToggles() {
  el.stageToggles.replaceChildren();
  for (const stage of auditStages()) {
    const label = document.createElement("label");
    label.className = "toggle-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.stageId = stage.stageId;
    input.addEventListener("change", () => {
      if (input.checked) state.forcedStages.add(stage.stageId);
      else state.forcedStages.delete(stage.stageId);
      renderAll();
    });
    const text = document.createElement("span");
    text.textContent = `${stage.stageId} · display only`;
    label.append(input, text);
    el.stageToggles.append(label);
  }
  if (!auditStages().length) {
    const note = document.createElement("p");
    note.className = "fine-print";
    note.textContent = "No A5/SM or physical H3/KMS stage catalog is present in this export.";
    el.stageToggles.append(note);
  }
}

function populateCensusKinds() {
  const census = state.summary?.census || {};
  const options = [
    ["carriers", "Carriers", census.declaredCarrierCount || 0, census.loadedCarrierCount || 0],
    ["carrier-pulses", "Carrier port pulses", census.carrierPulseCount || 0, census.loadedCarrierPulseCount || 0],
    ["screen", "Screen points", census.screenPointCount || 0, census.screenPointCount || 0],
    ["observers", "Observers", census.observerCount || 0, census.observerCount || 0],
    ["objects", "H3 / record objects", census.h3ObjectCount || 0, census.h3ObjectCount || 0],
    ["particles", "Particles / worldlines", census.particleOrWorldlineCount || 0, census.particleOrWorldlineCount || 0],
    ["atoms", "Atoms", census.atomCount || 0, census.loadedAtomCount || 0]
  ];
  el.censusKind.replaceChildren();
  for (const [value, label, count, loaded] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.dataset.count = String(count);
    option.dataset.loaded = String(loaded);
    option.textContent = `${label} · ${formatInteger(count)}`;
    el.censusKind.append(option);
  }
}

function populateSidecars() {
  const files = state.manifest?.files || [];
  el.sidecarSelect.replaceChildren();
  for (const file of files) {
    const option = document.createElement("option");
    option.value = file.fileId;
    option.textContent = `${file.logicalName} · ${formatBytes(file.byteCount)}`;
    el.sidecarSelect.append(option);
  }
  el.manifestStatus.textContent = state.manifest?.manifestStatus || "no manifest";
}

function renderAll() {
  if (!state.summary) return;
  const view = VIEW_DEFS.find(item => item.id === state.view) || VIEW_DEFS[0];
  for (const tab of el.viewTabs.querySelectorAll(".view-tab")) {
    tab.classList.toggle("active", tab.dataset.view === view.id);
  }
  el.viewKicker.textContent = view.kicker;
  el.viewTitle.textContent = view.title;
  el.viewSummary.textContent = view.summary;
  renderViewStatus();
  renderStageTable();
  renderClockStrip();
  renderSelectionDetails();
  renderMotionDisclosure();
  scheduleCanvas();
  updateAnimationLoop();
}

function renderMotionDisclosure() {
  const base = "Windowed renderer: only the selected visible chunk is drawn; every finite census index remains addressable.";
  if (state.mode !== "DEMO_ASSUMPTION") {
    el.renderDisclosure.textContent = base;
    return;
  }
  const recordsAvailable = ladder()?.demoUniverse?.enabled === true;
  const source = recordsAvailable
    ? "Animation source: screenA5Ladder.demoUniverse DEMO_ASSUMPTION records."
    : "Animation source: clearly synthetic renderer fallback; no demoUniverse records were exported.";
  const motion = state.reducedMotion ? " Reduced motion: static settled/fixed-point frame." : " Motion pauses while this tab is hidden.";
  el.renderDisclosure.textContent = `${base} ${source}${motion}`;
}

function updateAnimationLoop() {
  if (state.animationFrame) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = 0;
  }
  const shouldAnimate = state.mode === "DEMO_ASSUMPTION" && !state.reducedMotion && !document.hidden;
  if (!shouldAnimate) {
    if (state.mode === "DEMO_ASSUMPTION" && state.reducedMotion) state.animationTime = 9_999;
    scheduleCanvas();
    return;
  }
  const frame = timestamp => {
    state.animationTime = timestamp;
    drawCurrentView();
    state.animationFrame = requestAnimationFrame(frame);
  };
  state.animationFrame = requestAnimationFrame(frame);
}

function demoPhase(periodMs = 2400) {
  if (state.reducedMotion) return 0.999;
  return (state.animationTime % periodMs) / periodMs;
}

function renderViewStatus() {
  el.viewStatus.replaceChildren();
  addBadge("SCALE_CAMPAIGN_ALLOWED = false", "closed");
  addBadge("promotion_allowed = false", "closed");
  if (state.mode === "DEMO_ASSUMPTION") addBadge("DEMO · RECEIPTS UNCHANGED", "demo");
  else addBadge("CANONICAL RECEIPT STATE", "pass");
}

function addBadge(label, kind) {
  const badge = document.createElement("span");
  badge.className = `status-badge ${kind}`;
  badge.textContent = label;
  el.viewStatus.append(badge);
}

function renderStageTable() {
  el.stageTable.replaceChildren();
  const rows = auditStages();
  if (!rows.length) {
    const tr = document.createElement("tr");
    appendCells(tr, ["No A5/SM or H3/KMS stage payload", "missing", "blocked", "Legacy payload; no status inferred"]);
    el.stageTable.append(tr);
    return;
  }
  for (const stage of rows) {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.addEventListener("click", () => {
      state.selectedStage = stage.stageId;
      renderAll();
    });
    const cells = [
      stage.stageId,
      physicalStatusLabel(stage),
      displayStatus(stage),
      stage.claimBoundary || "Independent receipt required"
    ];
    appendCells(tr, cells);
    tr.children[1].className = physicalPassed(stage) ? "physical-pass" : "physical-open";
    if (displayStatus(stage) === "forced_demo_assumption") tr.children[2].className = "display-demo";
    el.stageTable.append(tr);
  }
}

function appendCells(row, values) {
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = String(value ?? "—");
    row.append(cell);
  }
}

function renderClockStrip() {
  el.clockStrip.replaceChildren();
  const clockData = ladder()?.clockSeparation || {};
  const candidates = Array.isArray(clockData.candidates)
    ? clockData.candidates.map(candidate => {
      const id = candidate?.candidateId || "?";
      const exact = candidate?.exactValue || candidate?.numericValue || "?";
      return `${id} (${exact})`;
    })
    : [];
  const physicalClock = clockData.physicalReceipt === true && clockData.physicalSelection
    ? `physical selection ${clockData.physicalSelection}`
    : "no independent physical selection";
  const displayClock = clockData.demoSelection
    ? `display-only freeze ${clockData.demoSelection}`
    : "no display freeze";
  const definitions = [
    ["BW / geometric", `${candidates.length ? candidates.join(" · ") : "1x · pi · 2pi · 4pi"}; ${physicalClock}; ${displayClock}`],
    ["A5 primitive volume", "Oriented-volume clock for global-form descent; never the BW clock"],
    ["W/Z operational", "Transition clock for physical units; independently receipted"]
  ];
  for (const [name, fallback] of definitions) {
    const item = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = `${name}: `;
    item.append(strong, document.createTextNode(fallback));
    el.clockStrip.append(item);
  }
  if (Object.keys(clockData).length) el.clockStrip.dataset.payloadAvailable = "true";
}

function renderSelectionDetails() {
  el.selectionDetails.replaceChildren();
  const kind = el.censusKind?.value || "carriers";
  const start = numberValue(el.censusStart, 0);
  const selectedStage = auditStages().find(stage => stage.stageId === state.selectedStage);
  const entries = [
    ["Mode", state.mode],
    ["View", state.view],
    ["Census address", stableCensusAddress(kind, state.selectedCensusIndex || start)],
    ["Stage", selectedStage?.stageId || "none"],
    ["Physical", physicalStatusLabel(selectedStage)],
    ["Display", selectedStage ? displayStatus(selectedStage) : "—"],
    ["Scale", "immutable false"]
  ];
  for (const [term, description] of entries) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = String(description);
    el.selectionDetails.append(dt, dd);
  }
}

function scheduleCanvas() {
  requestAnimationFrame(drawCurrentView);
}

function prepareCanvas() {
  const canvas = el.scene;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(300, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d", { alpha: false });
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cssWidth = width / dpr;
  const cssHeight = height / dpr;
  context.fillStyle = COLORS.bg;
  context.fillRect(0, 0, cssWidth, cssHeight);
  drawGrid(context, cssWidth, cssHeight);
  state.hitTargets = [];
  return [context, cssWidth, cssHeight];
}

function drawCurrentView() {
  if (!el.scene || !state.summary) return;
  const [ctx, width, height] = prepareCanvas();
  if (state.view === "screen") drawScreen(ctx, width, height);
  else if (state.view === "a5sm") drawLadder(ctx, width, height);
  else if (state.view === "observer") drawObserver(ctx, width, height);
  else if (state.view === "emergent") drawEmergent(ctx, width, height);
  else if (state.view === "cosmology") drawCosmology(ctx, width, height);
  else drawCensus(ctx, width, height);
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.35;
  for (let x = 20; x < width; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 20; y < height; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawScreen(ctx, width, height) {
  const carrier = ladder()?.localCarrier;
  if (!carrier?.ports || !carrier?.edges) {
    drawMessageOn(ctx, width, height, "screenA5Ladder unavailable", "No decorative replacement is inferred.", COLORS.amber);
    return;
  }
  const split = width * 0.57;
  drawLabel(ctx, 24, 32, "ONE EXACT LOCAL CARRIER", COLORS.teal);
  const center = [split * 0.5, height * 0.52];
  const scale = Math.min(split, height) * 0.32;
  const actions = Array.isArray(carrier.a5?.actions) ? carrier.a5.actions : [];
  const automaticAction = state.mode === "DEMO_ASSUMPTION" && actions.length
    ? Math.min(actions.length - 1, Math.floor(demoPhase(7200) * actions.length))
    : 0;
  const actionIndex = state.selectedA5Action === null
    ? automaticAction
    : Math.min(actions.length - 1, Math.max(0, state.selectedA5Action));
  const action = actions[actionIndex];
  const positions = new Map();
  for (const port of carrier.ports) {
    const raw = port.position || [0, 0, 0];
    const matrix = action?.rotationMatrix;
    const p = Array.isArray(matrix) && matrix.length === 3
      ? matrix.map(row => Number(row[0]) * Number(raw[0]) + Number(row[1]) * Number(raw[1]) + Number(row[2]) * Number(raw[2]))
      : raw;
    const rotatedX = p[0] * 0.84 + p[2] * 0.54;
    const rotatedZ = -p[0] * 0.54 + p[2] * 0.84;
    const perspective = 1 / (1.35 - rotatedZ * 0.22);
    positions.set(port.portId, [center[0] + rotatedX * scale * perspective, center[1] - p[1] * scale * perspective, rotatedZ]);
  }
  const sortedFaces = Array.isArray(carrier.faces) ? [...carrier.faces].sort((a, b) => faceDepth(a, positions) - faceDepth(b, positions)) : [];
  for (const face of sortedFaces) {
    const points = face.portIds.map(id => positions.get(id)).filter(Boolean);
    if (points.length !== 3) continue;
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    ctx.lineTo(points[1][0], points[1][1]); ctx.lineTo(points[2][0], points[2][1]); ctx.closePath();
    ctx.fillStyle = "rgba(96,165,250,.055)"; ctx.fill();
  }
  ctx.strokeStyle = COLORS.silver; ctx.lineWidth = 1.2;
  for (const edge of carrier.edges) {
    const a = positions.get(edge.portIds?.[0]);
    const b = positions.get(edge.portIds?.[1]);
    if (!a || !b) continue;
    ctx.globalAlpha = 0.35 + 0.4 * ((a[2] + b[2] + 2) / 4);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const port of carrier.ports) {
    const p = positions.get(port.portId);
    ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.teal; ctx.fill();
    ctx.strokeStyle = COLORS.bg; ctx.lineWidth = 2; ctx.stroke();
    state.hitTargets.push({ x: p[0], y: p[1], radius: 11, id: port.portId, kind: "port" });
  }
  if (state.mode === "DEMO_ASSUMPTION") {
    drawCarrierReadbackPulses(ctx, carrier, positions);
  }
  drawA5ActionNavigator(ctx, split, height, actions, actionIndex);
  drawLabel(ctx, 24, height - 18, "12 PORTS · 30 EDGES · 20 FACES · 6 ANTIPODAL PAIRS · A5 ORDER 60", COLORS.silver);

  ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(split, 15); ctx.lineTo(split, height - 15); ctx.stroke();
  drawLabel(ctx, split + 24, 32, "VISIBLE FEDERATION WINDOW", COLORS.violet);
  drawVirtualWindow(ctx, split + 18, 52, width - split - 36, height - 90, "carriers");
}

function drawA5ActionNavigator(ctx, paneWidth, height, actions, selectedIndex) {
  if (actions.length !== 60) return;
  const left = 25;
  const usable = Math.max(120, paneWidth - 50);
  const spacing = usable / 29;
  drawLabel(ctx, left, height - 76, `A5 ACTION ${selectedIndex + 1}/60 · ${actions[selectedIndex]?.actionId || "unknown"} · CLICK TO PIN`, COLORS.violet);
  actions.forEach((action, index) => {
    const row = Math.floor(index / 30);
    const column = index % 30;
    const x = left + column * spacing;
    const y = height - 59 + row * 13;
    ctx.fillStyle = index === selectedIndex ? COLORS.gold : "rgba(179,157,255,.36)";
    ctx.beginPath(); ctx.arc(x, y, index === selectedIndex ? 3.5 : 2, 0, Math.PI * 2); ctx.fill();
    state.hitTargets.push({ x, y, radius: 6, id: index, kind: "a5Action", actionId: action.actionId });
  });
}

function faceDepth(face, positions) {
  const values = face.portIds.map(id => positions.get(id)?.[2] || 0);
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function drawCarrierReadbackPulses(ctx, carrier, positions) {
  const records = demoSegmentRecords("carrier_light_readback_settling");
  const edges = carrier.edges || [];
  const pulseCount = Math.min(6, Math.max(1, records.length || 4));
  const phase = demoPhase(1900);
  for (let index = 0; index < pulseCount; index += 1) {
    const edge = edges[(index * 7 + Math.floor(phase * edges.length)) % edges.length];
    if (!edge) continue;
    const a = positions.get(edge.portIds?.[0]);
    const b = positions.get(edge.portIds?.[1]);
    if (!a || !b) continue;
    const localT = (phase + index / pulseCount) % 1;
    const x = a[0] + (b[0] - a[0]) * localT;
    const y = a[1] + (b[1] - a[1]) * localT;
    ctx.shadowColor = COLORS.gold; ctx.shadowBlur = 12;
    ctx.fillStyle = COLORS.gold; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.shadowBlur = 0;
  const source = records.length ? "DEMO RECORD PULSES" : "SYNTHETIC FALLBACK PULSES";
  drawLabel(ctx, 24, 50, `${source} · LIGHT ZAPS / SETTLES`, COLORS.gold);
}

function drawLadder(ctx, width, height) {
  const stageRows = stages();
  if (!stageRows.length) {
    drawMessageOn(ctx, width, height, "A5→SM stage DAG unavailable", "Physical status remains missing; no pass is inferred.", COLORS.amber);
    return;
  }
  const edges = Array.isArray(ladder()?.a5ToSm?.stageEdges) ? ladder().a5ToSm.stageEdges : [];
  const columns = ladderColumns(stageRows);
  const positions = new Map();
  const margin = 34;
  const usableWidth = width - margin * 2;
  const columnWidth = usableWidth / Math.max(1, columns.length);
  columns.forEach((column, columnIndex) => {
    const rowHeight = (height - 100) / Math.max(1, column.length);
    column.forEach((stage, rowIndex) => {
      positions.set(stage.stageId, [
        margin + columnIndex * columnWidth + columnWidth * 0.5,
        70 + rowIndex * rowHeight + rowHeight * 0.5
      ]);
    });
  });
  for (const edge of edges) {
    const a = positions.get(edge.sourceStageId);
    const b = positions.get(edge.targetStageId);
    if (!a || !b) continue;
    ctx.strokeStyle = edge.dependencyKind === "alternative_group" ? COLORS.violet : COLORS.muted;
    ctx.setLineDash(edge.dependencyKind === "alternative_group" ? [5, 5] : []);
    ctx.beginPath(); ctx.moveTo(a[0] + 44, a[1]);
    ctx.bezierCurveTo((a[0] + b[0]) / 2, a[1], (a[0] + b[0]) / 2, b[1], b[0] - 44, b[1]); ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const stage of stageRows) {
    const point = positions.get(stage.stageId);
    const forced = displayStatus(stage) === "forced_demo_assumption";
    const passed = physicalPassed(stage);
    ctx.fillStyle = forced ? COLORS.gold : (passed ? "#123b3a" : "#152436");
    ctx.strokeStyle = passed ? COLORS.teal : (forced ? COLORS.gold : COLORS.amber);
    if (stage.stageId === state.selectedStage) ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = stage.stageId === state.selectedStage ? 3 : 1.4;
    roundedRect(ctx, point[0] - 44, point[1] - 19, 88, 38, 8);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = forced ? "#271a02" : COLORS.text;
    ctx.font = "600 10px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(shortLabel(stage.stageId, 13), point[0], point[1], 80);
    state.hitTargets.push({ x: point[0], y: point[1], radius: 45, id: stage.stageId, kind: "stage" });
  }
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  drawLabel(ctx, 24, 28, "PHYSICAL OUTLINE + INDEPENDENT DISPLAY FILL · Q2 ALTERNATIVE ROUTES", COLORS.teal);
  drawA5SectorStrip(ctx, width, height);
  drawLabel(ctx, 24, height - 22, "EXACT_SMALL_ORACLE: CLOSED · SCALE_CAMPAIGN_ALLOWED: FALSE", COLORS.amber);
}

function drawA5SectorStrip(ctx, width, height) {
  const sectors = ladder()?.localCarrier?.a5?.sectors;
  if (!Array.isArray(sectors) || !sectors.length) return;
  const palette = { "1": COLORS.silver, "3": COLORS.blue, "3-prime": COLORS.violet, "5": COLORS.teal };
  const dimensions = sectors.map(row => Number(row.dimension) || 1);
  const total = dimensions.reduce((sum, value) => sum + value, 0);
  const stripWidth = Math.min(360, width * .43);
  let cursor = width - stripWidth - 24;
  const y = height - 48;
  for (let index = 0; index < sectors.length; index += 1) {
    const sector = sectors[index];
    const cellWidth = stripWidth * (dimensions[index] / total);
    ctx.fillStyle = `${palette[sector.sectorId] || COLORS.muted}33`;
    ctx.strokeStyle = palette[sector.sectorId] || COLORS.muted;
    ctx.fillRect(cursor, y, cellWidth, 18); ctx.strokeRect(cursor, y, cellWidth, 18);
    ctx.fillStyle = COLORS.text; ctx.font = "700 9px system-ui"; ctx.textAlign = "center";
    ctx.fillText(String(sector.sectorId), cursor + cellWidth / 2, y + 12, Math.max(8, cellWidth - 4));
    cursor += cellWidth;
  }
  ctx.textAlign = "left";
}

function ladderColumns(stageRows) {
  const preferred = [
    ["ROOT"], ["GEOMETRY_565", "CURRENT_566"], ["GLOBAL_FORM_567"],
    ["SPIN_EXCHANGE_314"],
    ["SOURCE_REGISTRY_AND_PRIMITIVE_RESIDUES", "SCALAR_CHANNEL", "FAMILY_ATTACHMENT_569"], ["Q1_LOCAL_ACTION"],
    ["Q2_H", "Q2_E", "POSITIVITY_OR_POSITIVE_TRANSFER"],
    ["REFINEMENT_COMPLETENESS", "PHYSICAL_IDENTIFICATION"],
    ["COMPLETE_COUPLED_DYNAMICS", "FAMILY_BREAKING_OR_DESCENT", "VERTEX_1PI"], ["Q4_OS"]
  ];
  const byId = new Map(stageRows.map(stage => [stage.stageId, stage]));
  const used = new Set();
  const columns = preferred.map(ids => ids.map(id => byId.get(id)).filter(Boolean)).filter(column => column.length);
  for (const stage of stageRows) {
    if (columns.some(column => column.includes(stage))) used.add(stage.stageId);
  }
  const extras = stageRows.filter(stage => !used.has(stage.stageId));
  if (extras.length) columns.push(extras);
  return columns;
}

function drawObserver(ctx, width, height) {
  const steps = ladder()?.observerRepairBridge?.steps;
  const rows = Array.isArray(steps) && steps.length ? steps : [
    { stepId: "port_readback" }, { stepId: "overlap_mismatch" }, { stepId: "proof_carrying_repair" },
    { stepId: "atomic_commit_or_rollback" }, { stepId: "semantic_record" }, { stepId: "observer_checkpoint" }
  ];
  const y = height * 0.30;
  const spacing = (width - 90) / Math.max(1, rows.length - 1);
  rows.forEach((step, index) => {
    const x = 45 + index * spacing;
    if (index + 1 < rows.length) {
      ctx.strokeStyle = COLORS.muted; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x + 12, y); ctx.lineTo(x + spacing - 12, y); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2);
    const progressIndex = state.mode === "DEMO_ASSUMPTION"
      ? Math.min(rows.length - 1, Math.floor(demoPhase(3200) * rows.length))
      : -1;
    ctx.fillStyle = index <= progressIndex ? COLORS.gold : (index < rows.length - 2 ? COLORS.teal : COLORS.violet); ctx.fill();
    ctx.fillStyle = COLORS.text; ctx.font = "10px system-ui"; ctx.textAlign = "center";
    wrapText(ctx, String(step.stepId || `step-${index}`), x, y + 30, Math.max(80, spacing - 8), 13);
  });
  ctx.textAlign = "left";
  const panelY = height * 0.53;
  drawCameraPanel(ctx, 30, panelY, width * 0.44, height - panelY - 28, "EXPLANATORY OVERVIEW — NOT OBSERVER VISIBLE", false);
  drawCameraPanel(ctx, width * 0.52, panelY, width * 0.45, height - panelY - 28, "SUBJECTIVE LOCAL READOUT · HIDDEN GLOBAL H3 MASKED", true);
  drawLabel(ctx, 24, 30, "PRIMITIVE REPLAY → RECORD ANCESTRY → OBSERVER-VISIBLE CHECKPOINT", COLORS.teal);
  if (state.mode === "DEMO_ASSUMPTION") drawRepairResidual(ctx, width, height);
}

function drawRepairResidual(ctx, width, height) {
  const records = demoSegmentRecords("repair_fixed_point");
  const residuals = records.length
    ? records.map(row => Number(row.residualNorm)).filter(Number.isFinite)
    : Array.from({ length: 8 }, (_unused, index) => 2 ** (-index));
  const progress = Math.min(residuals.length - 1, Math.floor(demoPhase(3200) * residuals.length));
  const residual = residuals[Math.max(0, progress)] ?? 0;
  const label = records.length ? "DEMO RECORD RESIDUAL" : "SYNTHETIC FALLBACK RESIDUAL";
  drawLabel(ctx, 24, 52, `${label} · step ${progress + 1}/${residuals.length} · ||mismatch|| ${residual.toExponential(2)}`, COLORS.gold);
  const barWidth = width * .42;
  ctx.strokeStyle = COLORS.gold; ctx.strokeRect(width * .29, height * .43, barWidth, 7);
  ctx.fillStyle = COLORS.gold; ctx.fillRect(width * .29, height * .43, barWidth * ((progress + 1) / residuals.length), 7);
  if (progress === residuals.length - 1) drawLabel(ctx, width * .73, height * .445, "DISPLAY FIXED POINT", COLORS.gold);
}

function drawCameraPanel(ctx, x, y, width, height, label, subjective) {
  ctx.fillStyle = subjective ? "rgba(196,181,253,.05)" : "rgba(185,198,216,.04)";
  ctx.strokeStyle = subjective ? COLORS.violet : COLORS.silver;
  ctx.lineWidth = 1.3; roundedRect(ctx, x, y, width, height, 8); ctx.fill(); ctx.stroke();
  drawLabel(ctx, x + 12, y + 22, label, subjective ? COLORS.violet : COLORS.silver);
  ctx.strokeStyle = subjective ? COLORS.violet : COLORS.muted;
  ctx.beginPath();
  ctx.moveTo(x + width * .5, y + height * .72);
  ctx.lineTo(x + width * .25, y + height * .3);
  ctx.lineTo(x + width * .75, y + height * .3);
  ctx.closePath(); ctx.stroke();
  if (subjective) {
    ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.arc(x + width * .5, y + height * .46, Math.min(width, height) * .18, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawEmergent(ctx, width, height) {
  drawLabel(ctx, 24, 30, "H3 CHART · CURVATURE PROXY · DEFECTS · PARTICLES · ATOMS", COLORS.blue);
  const centerX = width * 0.38;
  const centerY = height * 0.53;
  for (let ring = 1; ring <= 5; ring += 1) {
    ctx.strokeStyle = `rgba(96,165,250,${0.28 - ring * .025})`;
    ctx.beginPath(); ctx.ellipse(centerX, centerY, ring * width * .062, ring * height * .055, 0, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.strokeStyle = COLORS.magenta; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
  for (let line = 0; line < 5; line += 1) {
    ctx.beginPath();
    for (let step = 0; step < 35; step += 1) {
      const t = step / 34;
      const x = 30 + t * width * .67;
      const y = centerY + Math.sin(t * 5 + line) * 20 + (line - 2) * 24;
      if (step === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (state.mode === "DEMO_ASSUMPTION") drawDemoWorldlineActor(ctx, width, height);
  ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(width * .72, 45); ctx.lineTo(width * .72, height - 30); ctx.stroke();
  drawVirtualWindow(ctx, width * .74, 60, width * .24, height - 110, el.censusKind.value || "particles");
  drawLabel(ctx, 24, height - 22, "DIAGNOSTIC/ASSUMED LAYERS DO NOT PROMOTE GRAVITY, PARTICLE, OR ATOM RECEIPTS", COLORS.amber);
}

function drawDemoWorldlineActor(ctx, width, height) {
  const smRecords = demoSegmentRecords("forced_sm_catalogue_and_interactions");
  const worldlineRows = smRecords.filter(row => row.recordKind === "particle_worldline_sample");
  const actorRows = smRecords.filter(row => row.recordKind === "particle_actor");
  const eventRows = smRecords.filter(row => row.recordKind === "interaction_event");
  const gravityRows = demoSegmentRecords("gravity_response");
  const cameraRows = demoSegmentRecords("virtual_observer_camera");
  const atomRows = demoSegmentRecords("finite_atom_census");
  const traceRows = ladder()?.demoUniverse?.traceability?.traceRows;
  const trace = Array.isArray(traceRows) ? traceRows.find(row => row && !row.truncated) : null;
  const byId = (rows, keys) => new Map(rows.flatMap(row => {
    const id = keys.map(key => row?.[key]).find(Boolean);
    return id ? [[id, row]] : [];
  }));
  const actors = byId(actorRows, ["actorId", "recordId"]);
  const events = byId(eventRows, ["recordId"]);
  const gravity = byId(gravityRows, ["recordId"]);
  const cameras = byId(cameraRows, ["recordId"]);
  const atoms = byId(atomRows, ["recordId"]);
  const grouped = new Map();
  for (const row of worldlineRows) {
    if (!row?.actorId || !Array.isArray(row.position) || row.position.length < 3) continue;
    if (!grouped.has(row.actorId)) grouped.set(row.actorId, []);
    grouped.get(row.actorId).push(row);
  }
  for (const rows of grouped.values()) {
    rows.sort((left, right) => Number(left.frameIndex || 0) - Number(right.frameIndex || 0));
  }
  const palette = [COLORS.gold, COLORS.teal, COLORS.magenta, COLORS.blue, COLORS.violet, COLORS.coral];
  const projectedGroups = [];
  let colorIndex = 0;
  for (const [actorId, rows] of grouped) {
    const points = rows.map(row => projectDemoPosition(row.position, width, height));
    if (points.length < 2) continue;
    const color = actorId === trace?.visibleParticleActorId
      ? COLORS.gold
      : palette[colorIndex++ % palette.length];
    ctx.strokeStyle = color;
    ctx.globalAlpha = actorId === trace?.visibleParticleActorId ? .95 : .52;
    ctx.lineWidth = actorId === trace?.visibleParticleActorId ? 2.4 : 1.15;
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
    ctx.stroke();
    projectedGroups.push({ actorId, points, color });
  }
  ctx.globalAlpha = 1;

  let fallback = false;
  if (!projectedGroups.length) {
    fallback = true;
    const points = Array.from({ length: 16 }, (_unused, index) => {
      const t = index / 15;
      return [width * (.08 + t * .58), height * (.56 + Math.sin(t * Math.PI * 3) * .12)];
    });
    ctx.strokeStyle = COLORS.gold; ctx.lineWidth = 1.5; ctx.setLineDash([4, 5]); ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
    ctx.stroke(); ctx.setLineDash([]);
    projectedGroups.push({ actorId: "synthetic-fallback-actor", points, color: COLORS.gold });
  }

  const phase = demoPhase(3600);
  for (const group of projectedGroups) {
    const scaled = phase * (group.points.length - 1);
    const left = Math.floor(scaled);
    const right = Math.min(group.points.length - 1, left + 1);
    const local = scaled - left;
    const x = group.points[left][0] + (group.points[right][0] - group.points[left][0]) * local;
    const y = group.points[left][1] + (group.points[right][1] - group.points[left][1]) * local;
    ctx.shadowColor = group.color; ctx.shadowBlur = group.actorId === trace?.visibleParticleActorId ? 16 : 7;
    ctx.fillStyle = group.color; ctx.beginPath(); ctx.arc(x, y, group.actorId === trace?.visibleParticleActorId ? 6 : 3.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.shadowBlur = 0;

  const linkedActor = actors.get(trace?.visibleParticleActorId);
  const linkedEvent = events.get(trace?.interactionEventId);
  const linkedGravity = gravity.get(trace?.gravityResponseId);
  const linkedCamera = cameras.get(trace?.observerFrameId);
  const linkedAtom = atoms.get(trace?.visibleAtomId);
  const tracePath = projectedGroups.find(group => group.actorId === trace?.visibleParticleActorId)?.points;
  const gravityJoinsEvent = linkedGravity?.sourceInteractionEventIds?.includes(trace?.interactionEventId) === true;
  const gravityJoinsActor = linkedGravity?.sourceActorIds?.includes(trace?.visibleParticleActorId) === true;
  const cameraJoinsGravity = linkedCamera?.gravityResponseIds?.includes(trace?.gravityResponseId) === true;
  const cameraSeesActor = linkedCamera?.visibleParticleActorIds?.includes(trace?.visibleParticleActorId) === true;
  const cameraSeesAtom = linkedCamera?.visibleAtomIds?.includes(trace?.visibleAtomId) === true;
  const atomJoinsActor = linkedAtom?.constituentActorRefs?.includes(trace?.visibleParticleActorId) === true;
  const eventPoint = linkedEvent?.position
    ? projectDemoPosition(linkedEvent.position, width, height)
    : tracePath?.at(-1);
  const gravityPoint = [width * .625, height * .27];
  const cameraPoint = [width * .685, height * .48];
  const atomPoint = [width * .625, height * .70];
  if (
    tracePath && linkedActor && linkedEvent && linkedGravity && linkedCamera && linkedAtom && eventPoint
    && gravityJoinsEvent && gravityJoinsActor && cameraJoinsGravity
    && cameraSeesActor && cameraSeesAtom && atomJoinsActor
  ) {
    ctx.strokeStyle = "rgba(242,184,75,.58)"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(eventPoint[0], eventPoint[1]); ctx.lineTo(gravityPoint[0], gravityPoint[1]); ctx.lineTo(cameraPoint[0], cameraPoint[1]); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(eventPoint[0], eventPoint[1]); ctx.lineTo(atomPoint[0], atomPoint[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = COLORS.coral; ctx.beginPath(); ctx.arc(eventPoint[0], eventPoint[1], 8, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = COLORS.blue; ctx.beginPath(); ctx.arc(gravityPoint[0], gravityPoint[1], 14 + Math.abs(Number(linkedGravity.curvatureResponseProxy || 0)) * 180, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = COLORS.teal; ctx.beginPath(); ctx.arc(cameraPoint[0], cameraPoint[1], 10, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cameraPoint[0] - 15, cameraPoint[1]); ctx.quadraticCurveTo(cameraPoint[0], cameraPoint[1] - 13, cameraPoint[0] + 15, cameraPoint[1]); ctx.quadraticCurveTo(cameraPoint[0], cameraPoint[1] + 13, cameraPoint[0] - 15, cameraPoint[1]); ctx.stroke();
    ctx.fillStyle = COLORS.magenta; ctx.beginPath(); ctx.arc(atomPoint[0], atomPoint[1], 7, 0, Math.PI * 2); ctx.fill();
    drawLabel(ctx, width * .575, height * .20, "GRAVITY RESPONSE", COLORS.blue);
    drawLabel(ctx, width * .645, height * .42, "OBSERVER FRAME", COLORS.teal);
    drawLabel(ctx, width * .59, height * .77, `${linkedAtom.element || "ATOM"} RECORD`, COLORS.magenta);
  }
  const sourceLabel = fallback
    ? "SYNTHETIC FALLBACK · NO EXPORTED WORLDLINE SAMPLES"
    : `RECORD-JOINED · ${projectedGroups.length} ACTORS · ${worldlineRows.length} WORLDLINE SAMPLES${trace ? " · TRACE LINKED" : " · TRACE MISSING"}`;
  drawLabel(ctx, 24, 50, sourceLabel, fallback ? COLORS.amber : COLORS.gold);
}

function projectDemoPosition(position, width, height) {
  const x = Number(position?.[0]) || 0;
  const y = Number(position?.[1]) || 0;
  const z = Number(position?.[2]) || 0;
  return [width * (.12 + (x + .8) * .36), height * (.50 - y * .42 - z * .18)];
}

function drawCosmology(ctx, width, height) {
  const cmb = sections().cmbComparison || {};
  const cmbSeries = findNumericSeries(cmb);
  const demoCosmology = demoSegmentRecords("cosmology");
  const demoSeries = demoCosmology.map(row => Number(row.assumedScaleFactor)).filter(Number.isFinite);
  const useDemoSeries = !cmbSeries.length && state.mode === "DEMO_ASSUMPTION" && demoSeries.length;
  const series = useDemoSeries ? demoSeries : cmbSeries;
  drawLabel(ctx, 24, 30, "FINITE READBACK → DIAGNOSTIC SKY → PINNED REFERENCE → RESIDUAL", COLORS.blue);
  const plot = { x: 55, y: 70, width: width - 100, height: height * .58 };
  ctx.strokeStyle = COLORS.silver; ctx.beginPath(); ctx.moveTo(plot.x, plot.y); ctx.lineTo(plot.x, plot.y + plot.height); ctx.lineTo(plot.x + plot.width, plot.y + plot.height); ctx.stroke();
  if (series.length >= 2) {
    const values = series.slice(0, 180);
    const min = Math.min(...values);
    const max = Math.max(...values);
    ctx.strokeStyle = useDemoSeries ? COLORS.gold : COLORS.blue; ctx.lineWidth = 2; ctx.beginPath();
    values.forEach((value, index) => {
      const x = plot.x + (index / Math.max(1, values.length - 1)) * plot.width;
      const y = plot.y + plot.height - ((value - min) / Math.max(1e-12, max - min)) * plot.height;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (useDemoSeries) {
      const position = demoPhase(4200) * (values.length - 1);
      const index = Math.min(values.length - 1, Math.floor(position));
      const x = plot.x + (index / Math.max(1, values.length - 1)) * plot.width;
      const y = plot.y + plot.height - ((values[index] - min) / Math.max(1e-12, max - min)) * plot.height;
      ctx.fillStyle = COLORS.gold; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      drawLabel(ctx, plot.x + 14, plot.y + 22, "DEMO COSMOLOGY RECORDS · ASSUMED SCALE FACTOR", COLORS.gold);
    }
  } else {
    ctx.fillStyle = COLORS.muted; ctx.font = "14px system-ui";
    ctx.fillText("No compact numeric CMB series in this summary; use a manifested row page.", plot.x + 20, plot.y + plot.height * .5);
  }
  const badges = [
    ["DIAGNOSTIC RESEMBLANCE", COLORS.teal],
    ["USABLE COMPARISON", COLORS.amber],
    ["PHYSICAL PREDICTION", COLORS.coral]
  ];
  badges.forEach(([label, color], index) => {
    ctx.strokeStyle = color; ctx.fillStyle = "rgba(7,19,31,.82)";
    roundedRect(ctx, 60 + index * 190, height - 90, 170, 34, 17); ctx.fill(); ctx.stroke();
    ctx.fillStyle = color; ctx.font = "700 10px system-ui"; ctx.textAlign = "center";
    ctx.fillText(label, 145 + index * 190, height - 69);
  });
  ctx.textAlign = "left";
}

function drawCensus(ctx, width, height) {
  drawLabel(ctx, 24, 30, "VIRTUALIZED CENSUS · STABLE ADDRESS · MANIFESTED PAGE", COLORS.teal);
  drawVirtualWindow(ctx, 25, 55, width - 50, height - 110, el.censusKind.value || "carriers");
  drawLabel(ctx, 24, height - 22, "VISIBLE WINDOW ONLY — NO SIMULTANEOUS LITERAL RENDERING CLAIM", COLORS.amber);
}

function drawVirtualWindow(ctx, x, y, width, height, kind) {
  const total = selectedCensusTotal(kind);
  const start = Math.min(numberValue(el.censusStart, 0), Math.max(0, total - 1));
  const requested = Math.min(1000, Math.max(1, numberValue(el.censusCount, 120)));
  const visible = Math.max(0, Math.min(requested, total - start));
  const loaded = selectedCensusLoaded(kind);
  if (!total) {
    ctx.fillStyle = COLORS.muted; ctx.font = "13px system-ui";
    ctx.fillText(`${kind}: unavailable in this export`, x + 10, y + 30);
    return;
  }
  const columns = Math.max(1, Math.floor(width / 24));
  const rows = Math.max(1, Math.floor(height / 24));
  const renderCount = Math.min(visible, columns * rows);
  const layout = Array.from({ length: renderCount }, (_unused, local) => {
    const index = start + local;
    const record = state.visibleRowsKind === kind ? state.visibleRows[local] : null;
    return {
      index,
      record,
      x: x + 12 + (local % columns) * 24,
      y: y + 12 + Math.floor(local / columns) * 24
    };
  });
  if (kind === "carriers") drawVisibleFederationSeams(ctx, layout);
  for (let local = 0; local < renderCount; local += 1) {
    const { index, record, x: px, y: py } = layout[local];
    const selected = index === state.selectedCensusIndex;
    ctx.strokeStyle = selected ? COLORS.text : (kind === "atoms" ? COLORS.magenta : COLORS.violet);
    ctx.fillStyle = selected ? COLORS.teal : "rgba(179,157,255,.08)";
    if (kind === "carriers") {
      const settling = state.mode === "DEMO_ASSUMPTION"
        ? 0.82 + 0.30 * Math.max(0, Math.sin((demoPhase(1800) * Math.PI * 2) - local * .22))
        : 1;
      drawMiniCarrier(ctx, px, py, 7 * settling, record?.a5FrameActionId);
    }
    else {
      ctx.beginPath(); ctx.arc(px, py, selected ? 6 : 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    state.hitTargets.push({ x: px, y: py, radius: 10, id: index, kind: "census" });
  }
  ctx.fillStyle = COLORS.muted; ctx.font = "11px system-ui";
  const sourceLabel = state.visibleRowsKind === kind
    ? state.visibleRowsSource
    : "window not loaded for this census kind";
  ctx.fillText(`total ${formatInteger(total)} · loaded/exported ${formatInteger(loaded)} · visible ${formatInteger(renderCount)} · start ${formatInteger(start)} · ${sourceLabel}`, x + 8, y + height - 6, width - 16);
}

function drawVisibleFederationSeams(ctx, layout) {
  const byId = new Map(layout.map(row => [
    row.record?.recordId || `carrier-${String(row.index).padStart(6, "0")}`,
    row
  ]));
  const seams = Array.isArray(ladder()?.federation?.seams) ? ladder().federation.seams : [];
  let drawn = 0;
  for (const seam of seams) {
    const left = byId.get(seam.leftCarrierId);
    const right = byId.get(seam.rightCarrierId);
    if (!left || !right) continue;
    ctx.strokeStyle = "rgba(66,214,199,.34)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke();
    if (state.mode === "DEMO_ASSUMPTION") {
      const t = (demoPhase(2100) + drawn * .137) % 1;
      const pulseX = left.x + (right.x - left.x) * t;
      const pulseY = left.y + (right.y - left.y) * t;
      ctx.fillStyle = COLORS.gold; ctx.beginPath(); ctx.arc(pulseX, pulseY, 1.8, 0, Math.PI * 2); ctx.fill();
    }
    drawn += 1;
  }
}

function drawMiniCarrier(ctx, x, y, radius, frameActionId = null) {
  const carrier = ladder()?.localCarrier;
  const ports = Array.isArray(carrier?.ports) ? carrier.ports : [];
  const edges = Array.isArray(carrier?.edges) ? carrier.edges : [];
  if (ports.length !== 12 || edges.length !== 30) {
    ctx.beginPath(); ctx.arc(x, y, Math.max(1.5, radius * .35), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    return;
  }
  const projected = new Map();
  const action = frameActionId
    ? carrier.a5?.actions?.find(row => row.actionId === frameActionId)
    : null;
  for (const port of ports) {
    const raw = port.position || [0, 0, 0];
    const matrix = action?.rotationMatrix;
    const p = Array.isArray(matrix) && matrix.length === 3
      ? matrix.map(row => Number(row[0]) * Number(raw[0]) + Number(row[1]) * Number(raw[1]) + Number(row[2]) * Number(raw[2]))
      : raw;
    const rotatedX = Number(p[0]) * .82 + Number(p[2]) * .57;
    const rotatedZ = -Number(p[0]) * .57 + Number(p[2]) * .82;
    const perspective = 1 / (1.35 - rotatedZ * .20);
    projected.set(port.portId, [x + rotatedX * radius * perspective, y - Number(p[1]) * radius * perspective]);
  }
  const originalAlpha = ctx.globalAlpha;
  ctx.globalAlpha = originalAlpha * .78;
  for (const edge of edges) {
    const a = projected.get(edge.portIds?.[0]);
    const b = projected.get(edge.portIds?.[1]);
    if (!a || !b) continue;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.globalAlpha = originalAlpha;
  for (const point of projected.values()) {
    ctx.beginPath(); ctx.arc(point[0], point[1], Math.max(.45, radius * .08), 0, Math.PI * 2); ctx.fill();
  }
}

function censusOption(kind) {
  if (!kind || kind === el.censusKind.value) return el.censusKind.selectedOptions?.[0];
  return Array.from(el.censusKind.options).find(option => option.value === kind);
}

function selectedCensusTotal(kind = el.censusKind.value) {
  const option = censusOption(kind);
  return Number(option?.dataset.count || 0);
}

function selectedCensusLoaded(kind = el.censusKind.value) {
  const option = censusOption(kind);
  return Number(option?.dataset.loaded || 0);
}

async function loadVisibleWindow() {
  const total = selectedCensusTotal();
  const kind = el.censusKind.value;
  let start = Math.max(0, Math.floor(numberValue(el.censusStart, 0)));
  const count = Math.min(1000, Math.max(1, Math.floor(numberValue(el.censusCount, 120))));
  if (total) start = Math.min(start, total - 1);
  el.censusStart.value = String(start);
  el.censusCount.value = String(count);
  state.selectedCensusIndex = start;
  const visible = Math.max(0, Math.min(count, total - start));
  const manifested = matchingCensusSidecar(kind);
  if (!visible) {
    state.visibleRows = [];
    state.visibleRowsSource = "empty census";
    state.visibleRowsKind = kind;
  } else if (manifested) {
    try {
      const page = await fetchJson(`${manifested.rowsEndpoint}?offset=${start}&limit=${visible}`);
      state.visibleRows = Array.isArray(page.rows) ? page.rows : [];
      state.visibleRowsSource = `manifested rows: ${manifested.logicalName}`;
      state.visibleRowsKind = kind;
    } catch (_error) {
      state.visibleRows = proceduralCensusWindow(kind, start, visible);
      state.visibleRowsSource = "procedural contract fallback after manifested page error";
      state.visibleRowsKind = kind;
    }
  } else {
    state.visibleRows = proceduralCensusWindow(kind, start, visible);
    state.visibleRowsSource = state.visibleRows.length
      ? "deterministic payload address-space generator"
      : "address-only (no generator contract for this census)";
    state.visibleRowsKind = kind;
  }
  const first = state.visibleRows[0];
  const provenance = first?.visualizationProvenance?.status || first?.provenanceStatus || "not supplied";
  el.censusReadout.textContent = `${stableCensusAddress(kind, start)} · total ${formatInteger(total)} · loaded/exported ${formatInteger(selectedCensusLoaded())} · visible ${formatInteger(visible)} · source ${state.visibleRowsSource} · provenance ${provenance}`;
  scheduleCanvas();
}

function matchingCensusSidecar(kind) {
  const aliases = {
    carriers: ["carrier", "federation"],
    "carrier-pulses": ["carrier_pulse", "carrier-pulse", "pulse"],
    atoms: ["atom"],
    screen: ["screen"],
    observers: ["observer", "camera"],
    objects: ["h3", "object", "event"],
    particles: ["particle", "worldline"]
  }[kind] || [];
  return (state.manifest?.files || []).find(file => {
    if (!file?.rowsEndpoint || file.fileId === "payload") return false;
    const name = String(file.logicalName || "").toLowerCase();
    return aliases.some(alias => name.includes(alias));
  });
}

function proceduralCensusWindow(kind, start, count) {
  if (!["carriers", "carrier-pulses", "atoms"].includes(kind)) return [];
  return Array.from({ length: count }, (_unused, offset) =>
    resolveProceduralCensusRecord(kind, start + offset)
  ).filter(Boolean);
}

function resolveProceduralCensusRecord(kind, index) {
  const key = { carriers: "carriers", "carrier-pulses": "carrierPulses", atoms: "atoms" }[kind];
  const contract = ladder()?.demoUniverse?.addressSpaces?.[key];
  if (!contract || !Number.isInteger(index) || index < 0 || index >= Number(contract.exactFiniteCount || 0)) return null;
  const address = String(contract.recordAddress || "record/{index}").replace("{index}", String(index));
  const base = {
    recordAddress: address,
    generatorAlgorithm: contract.generatorAlgorithm,
    proceduralSeed: contract.proceduralSeed,
    visualizationProvenance: {
      status: "synthetic",
      sourceRefs: [`screenA5Ladder.demoUniverse.addressSpaces.${key}`],
      deterministic: true
    }
  };
  if (kind === "carriers") {
    const total = Number(contract.exactFiniteCount);
    const y = 1 - 2 * ((index + .5) / total);
    const radial = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = index * Math.PI * (3 - Math.sqrt(5));
    return {
      ...base,
      recordId: `carrier-${String(index).padStart(6, "0")}`,
      carrierIndex: index,
      globalSupportPoint: [radial * Math.cos(angle), y, radial * Math.sin(angle)],
      a5FrameActionId: `a5-action-${String(index % 60).padStart(2, "0")}`,
      prototypeRef: "screenA5Ladder.localCarrier"
    };
  }
  if (kind === "carrier-pulses") {
    const carrierIndex = Math.floor(index / 12);
    const portIndex = index % 12;
    return {
      ...base,
      recordId: `carrier-pulse-${String(index).padStart(12, "0")}`,
      carrierIndex,
      portIndex,
      tick: index,
      path: `carrier-${String(carrierIndex).padStart(6, "0")}/port-${String(portIndex).padStart(2, "0")}/pulse-${String(index).padStart(12, "0")}`
    };
  }
  const sample = deterministicAtomSample(String(contract.proceduralSeed || ""), index);
  return {
    ...base,
    recordId: `atom-${String(index).padStart(8, "0")}`,
    atomIndex: index,
    element: sample.element,
    atomicNumber: sample.atomicNumber,
    carrierIndex: sample.carrierIndex % Math.max(1, Number(ladder()?.federation?.declaredCarrierCount || 1)),
    rendererResolution: "deterministic splitmix64 interpretation of exported generator contract"
  };
}

function deterministicAtomSample(seedText, index) {
  const mask = (1n << 64n) - 1n;
  let seed = 1469598103934665603n;
  for (const character of seedText) {
    seed ^= BigInt(character.codePointAt(0));
    seed = (seed * 1099511628211n) & mask;
  }
  let value = (seed + BigInt(index) + 0x9e3779b97f4a7c15n) & mask;
  value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
  value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask;
  value ^= value >> 31n;
  const elements = [["H", 1], ["He", 2], ["C", 6], ["O", 8]];
  const selected = elements[Number(value & 3n)];
  return { element: selected[0], atomicNumber: selected[1], carrierIndex: Number((value >> 8n) & 0xffffffffn) };
}

async function loadSidecarPage() {
  const fileId = el.sidecarSelect.value;
  const file = state.manifest?.files?.find(item => item.fileId === fileId);
  if (!file) return;
  const offset = Math.max(0, Math.floor(numberValue(el.rowOffset, 0)));
  const limit = Math.min(2000, Math.max(1, Math.floor(numberValue(el.rowLimit, 100))));
  el.sidecarPreview.textContent = "Loading one bounded page…";
  try {
    if (file.rowsEndpoint && file.fileId !== "payload") {
      const data = await fetchJson(`${file.rowsEndpoint}?offset=${offset}&limit=${limit}`);
      state.visibleRows = data.rows || [];
      state.visibleBinaryBytes = null;
      el.sidecarPreview.textContent = JSON.stringify({
        fileId, offset: data.offset, returned: data.returned, hasMore: data.hasMore,
        rows: data.rows?.slice(0, 12)
      }, null, 2);
    } else {
      const pageSize = Math.min(4 * 1024 * 1024, Math.max(1024, limit * 1024));
      const response = await fetch(`${file.pageEndpoint}?page=${offset}&pageSize=${pageSize}`);
      if (!response.ok) throw new Error(`page request ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      state.visibleBinaryBytes = bytes;
      state.visibleRows = [];
      el.sidecarPreview.textContent = `${file.logicalName}\npage ${offset} · ${bytes.length} bytes\nfirst bytes: ${Array.from(bytes.slice(0, 64)).join(" ")}`;
    }
    state.view = "census";
    renderAll();
  } catch (error) {
    el.sidecarPreview.textContent = String(error);
  }
}

function selectCanvasTarget(event) {
  const rect = el.scene.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const target = [...state.hitTargets].reverse().find(item => Math.hypot(item.x - x, item.y - y) <= item.radius);
  if (!target) return;
  if (target.kind === "stage") state.selectedStage = target.id;
  if (target.kind === "a5Action") {
    state.selectedA5Action = state.selectedA5Action === Number(target.id)
      ? null
      : Number(target.id);
  }
  if (target.kind === "census") {
    state.selectedCensusIndex = Number(target.id);
    el.censusReadout.textContent = stableCensusAddress(el.censusKind.value, state.selectedCensusIndex);
  }
  if (target.kind === "port") {
    el.censusReadout.textContent = `${target.id} · exact local carrier port`;
  }
  renderAll();
}

function stableCensusAddress(kind, index) {
  const safeIndex = Math.max(0, Number(index) || 0);
  const key = { carriers: "carriers", "carrier-pulses": "carrierPulses", atoms: "atoms" }[kind];
  const address = key ? ladder()?.demoUniverse?.addressSpaces?.[key]?.recordAddress : null;
  if (typeof address === "string" && address.includes("{index}")) {
    return address.replace("{index}", String(safeIndex));
  }
  return `census:${kind}:${String(safeIndex).padStart(9, "0")}`;
}

function findNumericSeries(value, depth = 0) {
  if (depth > 8) return [];
  if (Array.isArray(value)) {
    const direct = value.map(item => Number(item)).filter(Number.isFinite);
    if (direct.length >= 2) return direct;
    for (const item of value.slice(0, 64)) {
      if (item && typeof item === "object") {
        const numerics = Object.values(item).map(field => Number(field)).filter(Number.isFinite);
        if (numerics.length) return value.slice(0, 256).map(row => Number(Object.values(row).find(field => Number.isFinite(Number(field))))).filter(Number.isFinite);
      }
    }
  }
  if (value && typeof value === "object") {
    for (const nested of Object.values(value)) {
      const result = findNumericSeries(nested, depth + 1);
      if (result.length >= 2) return result;
    }
  }
  return [];
}

function demoSegmentRecords(segmentId) {
  const segments = ladder()?.demoUniverse?.segments;
  if (!Array.isArray(segments)) return [];
  const segment = segments.find(row => row?.segmentId === segmentId);
  return Array.isArray(segment?.records) ? segment.records.filter(row => row && !row.truncated) : [];
}

function drawMessage(title, detail, color) {
  const [ctx, width, height] = prepareCanvas();
  drawMessageOn(ctx, width, height, title, detail, color);
}

function drawMessageOn(ctx, width, height, title, detail, color) {
  ctx.textAlign = "center";
  ctx.fillStyle = color; ctx.font = "700 18px system-ui"; ctx.fillText(title, width / 2, height / 2 - 12);
  ctx.fillStyle = COLORS.muted; ctx.font = "13px system-ui"; ctx.fillText(detail, width / 2, height / 2 + 18, width - 60);
  ctx.textAlign = "left";
}

function drawLabel(ctx, x, y, label, color) {
  ctx.fillStyle = color; ctx.font = "700 11px system-ui"; ctx.textAlign = "left"; ctx.fillText(label, x, y);
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r); ctx.arcTo(x, y, x + width, y, r); ctx.closePath();
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
  const words = text.replaceAll("_", " ").split(" ");
  let line = "";
  let lineIndex = 0;
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width > maxWidth && line) {
      ctx.fillText(line, x, y + lineIndex * lineHeight);
      line = word; lineIndex += 1;
    } else line = candidate;
  }
  if (line) ctx.fillText(line, x, y + lineIndex * lineHeight);
}

function shortLabel(value, limit) {
  const text = String(value).replaceAll("_", " ");
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function numberValue(input, fallback) {
  const value = Number(input?.value);
  return Number.isFinite(value) ? value : fallback;
}

function formatInteger(value) {
  return new Intl.NumberFormat().format(Number(value) || 0);
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
