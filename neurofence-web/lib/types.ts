/**
 * Shape of a NeuroFence report, mirroring `ScanReport.to_dict()` in detector.py
 * plus the optional `heatmap` block that app.py embeds on export.
 *
 * Everything here is treated as untrusted input: a report is a file a user
 * drops in, so the parser validates rather than assumes.
 */

export interface NeuronAnomaly {
  layer: string;
  layer_index: number;
  neuron: number;
  z_score: number;
  z_clipped: boolean;
  baseline_mean: number;
  baseline_std: number;
  baseline_peak?: number;
  fuzz_peak: number;
  delta: number;
  dormancy: number;
  spatial_z?: number;
  amplification?: number;
  trigger_case: number;
  trigger_label: string;
  trigger_category: string;
  trigger_prompt: string;
}

export interface LayerFinding {
  layer: string;
  layer_index: number;
  channels: number;
  flagged: number;
  flagged_fraction: number;
  max_z: number;
  mean_z: number;
  p99_z: number;
  max_dormancy: number;
}

export interface IntegrityFinding {
  severity: "INFO" | "MEDIUM" | "HIGH" | "CRITICAL" | string;
  code: string;
  message: string;
  evidence?: string[];
}

export interface ReportTotals {
  layers: number;
  neurons: number;
  flagged_neurons: number;
  flagged_layers: number;
  max_z: number;
  baseline_passes: number;
  fuzz_passes: number;
}

export interface HeatmapBlock {
  encoding: string;
  rows: number;
  bins: number;
  layers: string[];
  channels: number[];
  baseline: string | null;
  fuzz: string | null;
  zmap: string | null;
}

export interface ScanReport {
  safety_score: number;
  verdict: string;
  rationale: string;
  z_threshold: number;
  totals: ReportTotals;
  score_breakdown: Record<string, number>;
  anomalies: NeuronAnomaly[];
  layer_findings: LayerFinding[];
  integrity_findings: IntegrityFinding[];
  metadata: Record<string, unknown>;
  heatmap?: HeatmapBlock;
}

/** Decoded matrices, ready for the canvas. */
export interface DecodedHeatmap {
  rows: number;
  bins: number;
  layers: string[];
  channels: number[];
  baseline: Float32Array | null;
  fuzz: Float32Array | null;
  zmap: Float32Array | null;
}

export type MatrixKind = "baseline" | "fuzz" | "zmap";

export class ReportParseError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Validate an arbitrary parsed JSON value as a NeuroFence report.
 *
 * Throws {@link ReportParseError} with an operator-readable message rather than
 * letting a malformed file surface as a render-time crash.
 */
export function parseReport(raw: unknown): ScanReport {
  if (!isRecord(raw)) {
    throw new ReportParseError("This file is not a JSON object.");
  }
  if (typeof raw.safety_score !== "number" || typeof raw.verdict !== "string") {
    throw new ReportParseError(
      "Missing 'safety_score' or 'verdict'. This does not look like a NeuroFence report — " +
        "export one from the desktop app with File → Export report."
    );
  }

  const totalsRaw = isRecord(raw.totals) ? raw.totals : {};
  const totals: ReportTotals = {
    layers: num(totalsRaw.layers),
    neurons: num(totalsRaw.neurons),
    flagged_neurons: num(totalsRaw.flagged_neurons),
    flagged_layers: num(totalsRaw.flagged_layers),
    max_z: num(totalsRaw.max_z),
    baseline_passes: num(totalsRaw.baseline_passes),
    fuzz_passes: num(totalsRaw.fuzz_passes),
  };

  const breakdown: Record<string, number> = {};
  if (isRecord(raw.score_breakdown)) {
    for (const [key, value] of Object.entries(raw.score_breakdown)) {
      if (typeof value === "number" && Number.isFinite(value)) breakdown[key] = value;
    }
  }

  return {
    safety_score: raw.safety_score,
    verdict: raw.verdict,
    rationale: typeof raw.rationale === "string" ? raw.rationale : "",
    z_threshold: num(raw.z_threshold, 4.5),
    totals,
    score_breakdown: breakdown,
    anomalies: Array.isArray(raw.anomalies) ? (raw.anomalies as NeuronAnomaly[]) : [],
    layer_findings: Array.isArray(raw.layer_findings)
      ? (raw.layer_findings as LayerFinding[])
      : [],
    integrity_findings: Array.isArray(raw.integrity_findings)
      ? (raw.integrity_findings as IntegrityFinding[])
      : [],
    metadata: isRecord(raw.metadata) ? raw.metadata : {},
    heatmap: isRecord(raw.heatmap) ? (raw.heatmap as unknown as HeatmapBlock) : undefined,
  };
}
