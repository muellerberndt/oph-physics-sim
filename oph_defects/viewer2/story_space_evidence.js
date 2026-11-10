// Observation rules for the Story page. No particle identity is inferred here.
export function describeRegisterEvent(event) {
  const before = event.before.filter(Boolean).join(' + ') || 'blank faces';
  const after = event.after.filter(Boolean).join(' + ') || 'blank faces';
  return `${event.kind}: face classes ${before} → ${after}`;
}

// A declared diagonal instrument for the translation-phase POVM.
// A uniform M-state controlled-translation clock has amplitude
// exp(i*(M-1)*d/2) sin(M*d/2)/(M*sin(d/2)). Removing that phase
// gives the real Kraus amplitude used by Story. The removal is an
// additional momentum-diagonal unitary, not merely a global phase.
// At a removable singularity we choose the real amplitude +1; an
// outcome-dependent unitary can also supply this sign convention.
export function realClockAmplitude(momentum, outcome, states) {
  if (!Number.isInteger(states) || states < 1 || !Number.isInteger(outcome) || outcome < 0 || outcome >= states)
    throw new Error('Invalid finite clock outcome');
  const d = momentum - 2 * Math.PI * outcome / states, s = Math.sin(d / 2);
  return Math.abs(s) < 1e-12 ? 1 : Math.sin(states * d / 2) / (states * s);
}
