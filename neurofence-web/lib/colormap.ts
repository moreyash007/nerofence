/**
 * Colour ramps for the activation matrix.
 *
 * The default is a **semantic heat** ramp — the one multi-hue sequential form the
 * data-viz method sanctions, and only ever with a scale legend, which the heatmap
 * always renders. Its steps are the project's status palette (critical → serious →
 * warning) arranged so **lightness increases monotonically with magnitude**: dark
 * = quiet, yellow-hot = most anomalous. That monotonicity is what makes it legible
 * under colour-vision deficiency, which a green→red ramp is not.
 *
 * The `console` ramp reproduces the desktop PyQt LUT exactly, for operators moving
 * between the two tools. It is offered, not default: green→red is the classic
 * deuteranopia failure, so the safe ramp wins ties.
 */

export type RampId = "heat" | "console";
export type ThemeMode = "dark" | "light";

type Stop = [number, [number, number, number]];

const HEAT_DARK: Stop[] = [
  [0.0, [26, 26, 25]],
  [0.15, [61, 23, 20]],
  [0.32, [122, 29, 29]],
  [0.5, [208, 59, 59]],
  [0.7, [236, 131, 90]],
  [0.88, [250, 178, 25]],
  [1.0, [255, 233, 168]],
];

const HEAT_LIGHT: Stop[] = [
  [0.0, [252, 252, 251]],
  [0.15, [253, 239, 196]],
  [0.32, [250, 178, 25]],
  [0.5, [236, 131, 90]],
  [0.7, [208, 59, 59]],
  [0.88, [143, 32, 32]],
  [1.0, [74, 17, 17]],
];

/** The exact control points used by HeatmapCanvas in app.py. */
const CONSOLE: Stop[] = [
  [0.0, [9, 22, 18]],
  [0.12, [0, 78, 52]],
  [0.3, [0, 158, 84]],
  [0.48, [118, 200, 60]],
  [0.64, [222, 214, 44]],
  [0.78, [247, 166, 28]],
  [0.9, [250, 96, 26]],
  [1.0, [255, 44, 44]],
];

function stopsFor(ramp: RampId, theme: ThemeMode): Stop[] {
  if (ramp === "console") return CONSOLE;
  return theme === "light" ? HEAT_LIGHT : HEAT_DARK;
}

/** Build a 256-entry RGB lookup table. */
export function buildLut(ramp: RampId, theme: ThemeMode, size = 256): Uint8ClampedArray {
  const stops = stopsFor(ramp, theme);
  const lut = new Uint8ClampedArray(size * 3);
  let segment = 0;
  for (let i = 0; i < size; i += 1) {
    const t = i / (size - 1);
    while (segment < stops.length - 2 && t > stops[segment + 1][0]) segment += 1;
    const [t0, c0] = stops[segment];
    const [t1, c1] = stops[segment + 1];
    const span = t1 - t0 || 1;
    const k = Math.min(1, Math.max(0, (t - t0) / span));
    lut[i * 3] = c0[0] + (c1[0] - c0[0]) * k;
    lut[i * 3 + 1] = c0[1] + (c1[1] - c0[1]) * k;
    lut[i * 3 + 2] = c0[2] + (c1[2] - c0[2]) * k;
  }
  return lut;
}

/** CSS `linear-gradient` stops for the scale legend, low → high. */
export function gradientCss(ramp: RampId, theme: ThemeMode, toRight = true): string {
  const stops = stopsFor(ramp, theme)
    .map(([t, [r, g, b]]) => `rgb(${r} ${g} ${b}) ${(t * 100).toFixed(1)}%`)
    .join(", ");
  return `linear-gradient(${toRight ? "to right" : "to top"}, ${stops})`;
}

/**
 * Log-compress a 0..1 value so low-energy texture stays visible.
 * Mirrors `HeatmapCanvas._compress` in the desktop app.
 */
export function compress(normalised: number, knee = 60): number {
  const v = Math.min(1, Math.max(0, normalised));
  return Math.log1p(v * knee) / Math.log1p(knee);
}

/**
 * Normalise a value against a ceiling **in log space**.
 *
 * Z-scores span six orders of magnitude in a poisoned model: a dormant neuron
 * with sigma = 0 clips at 1e6 while healthy layers peak around 10. Dividing by
 * the ceiling first and compressing afterwards drives every honest layer to
 * 1e-5 and renders the map as one bright speck on black. Taking the ratio of
 * logarithms instead keeps each decade the same visual distance, so a layer at
 * Z=16 is still clearly readable beside one at Z=1e6.
 */
export function logNormalize(value: number, ceiling: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  const top = Math.log10(1 + Math.max(ceiling, 1));
  if (top <= 0) return 0;
  return Math.min(1, Math.max(0, Math.log10(1 + value) / top));
}

/* ------------------------------------------------------------------ status */

export type StatusRole = "good" | "warning" | "serious" | "critical";

/** Fixed status palette — never themed, never reused as a series colour. */
export const STATUS: Record<StatusRole, string> = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
};

/** Map a 0–100 safety score onto a status role. Bands match VERDICT_BANDS. */
export function statusForScore(score: number): StatusRole {
  if (score >= 90) return "good";
  if (score >= 70) return "warning";
  if (score >= 45) return "serious";
  return "critical";
}

/** Map an integrity finding severity onto a status role. */
export function statusForSeverity(severity: string): StatusRole {
  switch (severity.toUpperCase()) {
    case "CRITICAL":
      return "critical";
    case "HIGH":
      return "serious";
    case "MEDIUM":
      return "warning";
    default:
      return "good";
  }
}

/**
 * Glyph paired with every status colour, so state is never carried by hue alone
 * — required because `warning` and `serious` fall below 3:1 on a light surface.
 */
export const STATUS_GLYPH: Record<StatusRole, string> = {
  good: "✓",
  warning: "!",
  serious: "▲",
  critical: "✕",
};
