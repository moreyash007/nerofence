import type { DecodedHeatmap, HeatmapBlock, ScanReport } from "./types";

/**
 * Decode one base64 little-endian float32 matrix written by
 * `worker.build_report_heatmap`.
 *
 * Returns null (rather than throwing) when the payload is absent or the wrong
 * length — a report exported before the heatmap block existed is still a
 * perfectly valid report, and the viewer degrades to the layer profile.
 */
export function decodeMatrix(
  b64: string | null | undefined,
  expected: number
): Float32Array | null {
  if (!b64 || expected <= 0) return null;
  let binary: string;
  try {
    binary = atob(b64);
  } catch {
    return null;
  }
  if (binary.length !== expected * 4) return null;

  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);

  // The Python side writes '<f4' explicitly. Every platform that runs a browser
  // is little-endian, but read through a DataView when it is not, rather than
  // handing back silently byte-swapped numbers.
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(expected);
  if (isLittleEndian()) {
    out.set(new Float32Array(bytes.buffer, 0, expected));
  } else {
    for (let i = 0; i < expected; i += 1) out[i] = view.getFloat32(i * 4, true);
  }
  return out;
}

let cachedEndian: boolean | null = null;
function isLittleEndian(): boolean {
  if (cachedEndian === null) {
    const probe = new Uint8Array(new Uint16Array([1]).buffer);
    cachedEndian = probe[0] === 1;
  }
  return cachedEndian;
}

/** Decode the whole heatmap block, or null when the report has none. */
export function decodeHeatmap(report: ScanReport): DecodedHeatmap | null {
  const block: HeatmapBlock | undefined = report.heatmap;
  if (!block) return null;

  const rows = Number(block.rows) || 0;
  const bins = Number(block.bins) || 0;
  if (rows <= 0 || bins <= 0) return null;
  const expected = rows * bins;

  const baseline = decodeMatrix(block.baseline, expected);
  const fuzz = decodeMatrix(block.fuzz, expected);
  const zmap = decodeMatrix(block.zmap, expected);
  if (!baseline && !fuzz && !zmap) return null;

  return {
    rows,
    bins,
    layers: Array.isArray(block.layers) ? block.layers : [],
    channels: Array.isArray(block.channels) ? block.channels : [],
    baseline,
    fuzz,
    zmap,
  };
}

/** Per-row maximum across whichever matrices are present, for shared scaling. */
export function rowScales(
  matrices: (Float32Array | null)[],
  rows: number,
  bins: number
): Float32Array {
  const scales = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    let best = 0;
    for (const m of matrices) {
      if (!m) continue;
      const base = r * bins;
      for (let c = 0; c < bins; c += 1) {
        const v = m[base + c];
        if (v > best) best = v;
      }
    }
    scales[r] = best > 0 ? best : 1;
  }
  return scales;
}

/** Largest finite value in a matrix. */
export function matrixMax(matrix: Float32Array | null): number {
  if (!matrix) return 0;
  let best = 0;
  for (let i = 0; i < matrix.length; i += 1) {
    const v = matrix[i];
    if (Number.isFinite(v) && v > best) best = v;
  }
  return best;
}
