// Operator expectations of the finite positive-energy Dirac packet used by the viewer.
// These are state diagnostics, not simultaneous position/momentum detector outcomes.
const cache = new WeakMap();
export function invalidatePacketObservables(packet) { cache.delete(packet); }

export function packetObservables(field, packet) {
  const prior = cache.get(packet);
  if (prior && prior.time === packet.tb) return prior.value;
  let weight = 0, energy = 0, energy2 = 0, speed = 0;
  const momentum = [0, 0, 0], second = [0, 0, 0], velocity = [0, 0, 0];
  const marginals = Array.from({length:3}, () => Array(field.n).fill(0));
  const axes = [field.kx, field.ky, field.kz];
  for (let i = 0; i < field.N; i++) {
    let w = 0;
    for (let c = 0; c < 4; c++) w += packet.sr[c][i] ** 2 + packet.si[c][i] ** 2;
    const E = packet.E[i]; weight += w; energy += w * E; energy2 += w * E * E;
    const bins = [i % field.n, Math.floor(i / field.n) % field.n, Math.floor(i / (field.n * field.n))];
    const p = Math.sqrt(field.k2[i]);
    speed += E > 0 ? w * p / E : 0;
    for (let a = 0; a < 3; a++) {
      const k = axes[a][i]; momentum[a] += w * k; second[a] += w * k * k;
      marginals[a][bins[a]] += w;
      if (E > 0) velocity[a] += w * k / E;
    }
  }
  if (!(weight > 0) || !Number.isFinite(weight)) throw new Error('Packet has no finite positive norm');
  const mean = momentum.map(x => x / weight);
  const value = {
    meanMomentum: mean,
    momentumStd: second.map((x, a) => Math.sqrt(Math.max(0, x / weight - mean[a] ** 2))),
    meanEnergy: energy / weight,
    energyStd: Math.sqrt(Math.max(0, energy2 / weight - (energy / weight) ** 2)),
    meanVelocity: velocity.map(x => x / weight), meanSpeed: speed / weight,
    momentumMarginals: marginals.map(row => row.map(x => x / weight)),
    momentumGrid: Array.from({length:field.n}, (_,j) => 2*Math.PI*(j<field.n/2 ? j : j-field.n)/field.n),
    restEnergy: Math.abs(packet.m), norm: weight / field.N,
    momentumUnit: 'hbar per grid spacing', energyUnit: 'hbar per grid time',
    interpretation: 'expectations before this frame’s detector update; not joint measured values',
  };
  cache.set(packet, { time: packet.tb, value });
  return value;
}

export function physicalReadout(observables, units, gridCells) {
  const energyUnitMeV = units.hbar / (gridCells * units.tick);
  const speedUnit = units.cell * 1e-15 / units.tick; // cell in fm, tick in s
  return {
    energyMeV: observables.meanEnergy * energyUnitMeV,
    energyStdMeV: observables.energyStd * energyUnitMeV,
    restEnergyMeV: observables.restEnergy * energyUnitMeV,
    momentumMeVPerC: observables.meanMomentum.map(x => x * energyUnitMeV),
    momentumStdMeVPerC: observables.momentumStd.map(x => x * energyUnitMeV),
    velocityMetresPerSecond: observables.meanVelocity.map(x => x * speedUnit),
    meanSpeedMetresPerSecond: observables.meanSpeed * speedUnit,
    sourceSpeedMetresPerSecond: speedUnit,
    unitAncestry: 'uses the viewer’s supplied cell length and tick duration',
  };
}

export function recordSnapshot({ run, observer, slice, fieldTime, particle, family, units, grid }) {
  return {
    schema: 'oph.viewer.defect_readout.v2',
    source: { run, observer, slice, defectId: particle.id, centreClass: particle.cls },
    interpretation: { familyLabel: family.name, status: 'family candidate in the declared field display',
      measuredStandardModelSpeciesEstablished: false },
    field: { model: 'finite periodic free Dirac packet', gridPointsPerAxis: grid.n,
      cellsPerGridPoint: grid.h, fieldTime, packet: { initialPosition: [...particle.g0],
        initialDirection: [...particle.dir], initializedAt: particle.born },
      preparation: particle.preparation ? JSON.parse(JSON.stringify(particle.preparation)) : null,
      stateExpectations: particle.stats?.observables ?? null },
    units: { tickSeconds: units.tick, cellFemtometres: units.cell, hbarMeVSeconds: units.hbar,
      ancestry: 'supplied viewer conversion; no independent GeV prediction' },
    records: (particle.transcript ?? particle.recs).map(r => ({ fieldTime: r.t, gridPosition: [...r.x],
      band: r.band ?? null, positionProbability: r.positionProbability ?? null,
      bandProbability: r.bandProbability ?? null, jointProbability: r.probability ?? null,
      sourceEnergyChange: r.sourceEnergyChange ?? null, conversionWork: r.conversionWork ?? null,
      detectorWidth: r.detectorWidth ?? null, samplerDraws: r.samplerDraws ? [...r.samplerDraws] : null,
      samplerMode: r.samplerMode ?? null })),
    totalRecords: particle.nrec,
    displayedRecordLimit: 18,
    completePacketTranscript: Array.isArray(particle.transcript) && particle.transcript.length === particle.nrec,
    replay: { command: 'node readout_replay.mjs saved-readout.json',
      samplerAncestry: 'Recorded viewer sampling variates; the native repair tape does not supply these draws.' },
    detector: { model: 'complete Gaussian centre and energy-band instrument; rejected band explicitly converted',
      completeOnDeclaredFiniteDiracCarrier: true, conversionIsAnApparatusOperation: true,
      completeNativeSourceInstrumentEstablished: false },
  };
}
