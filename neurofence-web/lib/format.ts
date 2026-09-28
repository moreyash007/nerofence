import type { NeuronAnomaly } from "./types";

/** Z-scores from zero-variance neurons are clipped at 1e6 by the detector. */
export function formatZ(z: number, clipped?: boolean): string {
  if (!Number.isFinite(z)) return "∞";
  const text =
    Math.abs(z) >= 1000
      ? z.toLocaleString(undefined, { maximumFractionDigits: 0 })
      : z.toFixed(1);
  return clipped ? `>${text}` : text;
}

/** Compact significant-figure formatting for activation magnitudes. */
export function formatSci(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e6 || abs < 1e-4) return value.toExponential(2);
  return Number(value.toPrecision(digits)).toString();
}

export function formatInt(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
}

export function formatPercent(fraction: number, digits = 0): string {
  if (!Number.isFinite(fraction)) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** `model.layers.17.mlp.down_proj` → `L17·down_proj` (matches the desktop axis). */
export function shortLayerName(name: string): string {
  const parts = name.split(".");
  const index = parts.find((p) => /^\d+$/.test(p));
  const tail = parts[parts.length - 1] ?? name;
  return index !== undefined ? `L${index}·${tail}` : tail.slice(-16);
}

/** Single-line, printable rendition of a possibly-hostile trigger prompt. */
export function previewPrompt(prompt: string, width = 120): string {
  if (!prompt) return "";
  let flat = "";
  for (const ch of prompt) {
    const code = ch.codePointAt(0) ?? 0;
    if (ch === "\n") flat += "\\n";
    else if (ch === "\r") flat += "\\r";
    else if (ch === "\t") flat += "\\t";
    else if (code < 0x20 || code === 0x7f) flat += `\\x${code.toString(16).padStart(2, "0")}`;
    else if (isInvisible(code)) flat += `\\u${code.toString(16).padStart(4, "0")}`;
    else flat += ch;
  }
  return flat.length > width ? `${flat.slice(0, width - 1)}…` : flat;
}

/** Zero-width, bidi-override and other smuggling characters worth revealing. */
function isInvisible(code: number): boolean {
  return (
    code === 0x00ad ||
    (code >= 0x200b && code <= 0x200f) ||
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2060 && code <= 0x2064) ||
    code === 0xfeff
  );
}

export function anomalyKey(a: NeuronAnomaly, index: number): string {
  return `${a.layer}#${a.neuron}#${index}`;
}

/** Human-readable score-breakdown labels; unknown keys fall back to the raw key. */
const BREAKDOWN_LABELS: Record<string, string> = {
  peak_severity: "Peak severity",
  neuron_density: "Neuron density",
  layer_spread: "Layer spread",
  dormancy: "Dormancy",
  amplification: "Amplification",
  near_threshold_residual: "Near-threshold residual",
  total_penalty: "Total penalty",
};

export function breakdownLabel(key: string): string {
  if (BREAKDOWN_LABELS[key]) return BREAKDOWN_LABELS[key];
  if (key.startsWith("cap:")) return `Integrity cap — ${key.slice(4)}`;
  return key.replace(/_/g, " ");
}
