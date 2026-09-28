"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { buildLut, compress, gradientCss, logNormalize, type RampId } from "@/lib/colormap";
import { matrixMax, rowScales } from "@/lib/decode";
import { formatSci, formatZ, shortLayerName } from "@/lib/format";
import type { DecodedHeatmap, MatrixKind } from "@/lib/types";
import type { Theme } from "./ThemeToggle";

interface Props {
  heatmap: DecodedHeatmap;
  zThreshold: number;
  theme: Theme;
}

const VIEW_LABELS: Record<MatrixKind, string> = {
  baseline: "Baseline energy",
  fuzz: "Adversarial peak",
  zmap: "Z-score anomaly",
};

/**
 * Activation matrix: one row per hooked MLP layer, one column per max-pooled
 * band of neuron channels.
 *
 * Max-pooling (done Python-side) is what makes a single poisoned neuron in an
 * 11k-wide layer survive to the display — an average would erase it.
 *
 * Rendering: an ImageData at exactly rows×bins is blitted and upscaled with
 * nearest-neighbour, so cost is independent of layer width and cells stay crisp.
 */
export default function Heatmap({ heatmap, zThreshold, theme }: Props) {
  const available = useMemo<MatrixKind[]>(() => {
    const kinds: MatrixKind[] = [];
    if (heatmap.baseline) kinds.push("baseline");
    if (heatmap.fuzz) kinds.push("fuzz");
    if (heatmap.zmap) kinds.push("zmap");
    return kinds;
  }, [heatmap]);

  const [view, setView] = useState<MatrixKind>(() =>
    heatmap.zmap ? "zmap" : heatmap.fuzz ? "fuzz" : "baseline"
  );
  const [ramp, setRamp] = useState<RampId>("heat");
  const [showTable, setShowTable] = useState(false);
  const [hover, setHover] = useState<{ row: number; col: number; x: number; y: number } | null>(
    null
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { rows, bins } = heatmap;

  const scales = useMemo(
    () => rowScales([heatmap.baseline, heatmap.fuzz], rows, bins),
    [heatmap, rows, bins]
  );
  const zCeiling = useMemo(
    () => Math.max(matrixMax(heatmap.zmap), zThreshold * 4, 1),
    [heatmap.zmap, zThreshold]
  );

  const matrix = heatmap[view];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = bins;
    canvas.height = rows;

    const lut = buildLut(ramp, theme);
    const image = ctx.createImageData(bins, rows);
    const px = image.data;

    const isZ = view === "zmap";
    for (let r = 0; r < rows; r += 1) {
      const denom = scales[r];
      for (let c = 0; c < bins; c += 1) {
        const raw = matrix[r * bins + c];
        const value = Number.isFinite(raw) ? raw : 0;
        // Z spans decades and shares one scale across layers; energy is
        // per-layer and near-uniform, so each gets the scaling it needs.
        const level = isZ ? logNormalize(value, zCeiling) : compress(value / denom);
        const idx = Math.min(255, Math.max(0, Math.round(level * 255)));
        const o = (r * bins + c) * 4;
        px[o] = lut[idx * 3];
        px[o + 1] = lut[idx * 3 + 1];
        px[o + 2] = lut[idx * 3 + 2];
        px[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [matrix, rows, bins, ramp, theme, view, scales, zCeiling]);

  const displayHeight = Math.min(520, Math.max(150, rows * 22));
  const labelStep = Math.max(1, Math.ceil(rows / Math.floor(displayHeight / 13)));

  function onMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.min(bins - 1, Math.max(0, Math.floor((x / rect.width) * bins)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor((y / rect.height) * rows)));
    setHover({ row, col, x, y });
  }

  const hoverInfo = hover ? describeCell(heatmap, hover.row, hover.col) : null;

  return (
    <section className="card grid-full" aria-labelledby="matrix-heading">
      <h2 className="card-title" id="matrix-heading">
        Activation matrix — layers × pooled neuron channels
      </h2>

      <div className="hm-toolbar" role="group" aria-label="Matrix view">
        {available.map((kind) => (
          <button
            key={kind}
            type="button"
            className="btn"
            aria-pressed={view === kind}
            onClick={() => setView(kind)}
          >
            {VIEW_LABELS[kind]}
          </button>
        ))}
        <span className="spacer" />
        <button
          type="button"
          className="btn"
          aria-pressed={ramp === "console"}
          onClick={() => setRamp(ramp === "heat" ? "console" : "heat")}
          title="The console ramp matches the desktop app. Heat is the default because green→red is not colour-vision safe."
        >
          {ramp === "heat" ? "Ramp: Heat" : "Ramp: Console"}
        </button>
        <button
          type="button"
          className="btn"
          aria-pressed={showTable}
          onClick={() => setShowTable((v) => !v)}
        >
          {showTable ? "Hide table" : "Table view"}
        </button>
      </div>

      {matrix ? (
        <>
          <div className="hm-stage">
            <div
              className="hm-ylabels"
              style={{ display: "grid", gridTemplateRows: `repeat(${rows}, 1fr)` }}
              aria-hidden="true"
            >
              {Array.from({ length: rows }, (_, r) => (
                <span key={r} style={{ lineHeight: 1, alignSelf: "center" }}>
                  {r % labelStep === 0 ? shortLayerName(heatmap.layers[r] ?? `row ${r}`) : ""}
                </span>
              ))}
            </div>

            <div className="hm-canvas-wrap" style={{ height: displayHeight }}>
              <canvas
                ref={canvasRef}
                className="hm-canvas"
                onMouseMove={onMove}
                onMouseLeave={() => setHover(null)}
                role="img"
                aria-label={`${VIEW_LABELS[view]} matrix, ${rows} layers by ${bins} channel bands. Use the table view for exact values.`}
              />
              {hover && hoverInfo && (
                <div
                  className="hm-tooltip"
                  style={{
                    left: Math.min(hover.x + 14, 100000),
                    top: hover.y + 14,
                    transform:
                      hover.x > 0.62 * (canvasRef.current?.clientWidth ?? 0)
                        ? "translateX(-100%) translateX(-28px)"
                        : undefined,
                  }}
                >
                  <div className="lbl">{hoverInfo.layer}</div>
                  <div>{hoverInfo.span}</div>
                  {heatmap.baseline && (
                    <div>
                      <span className="lbl">baseline </span>
                      {formatSci(heatmap.baseline[hover.row * bins + hover.col])}
                    </div>
                  )}
                  {heatmap.fuzz && (
                    <div>
                      <span className="lbl">adversarial </span>
                      {formatSci(heatmap.fuzz[hover.row * bins + hover.col])}
                    </div>
                  )}
                  {heatmap.zmap && (
                    <div>
                      <span className="lbl">Z </span>
                      {formatZ(heatmap.zmap[hover.row * bins + hover.col])}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="hm-xaxis">
            <span>0</span>
            <span>channel band →</span>
            <span>{bins}</span>
          </div>

          {/* A scale legend is mandatory for a multi-hue sequential ramp. */}
          <div className="hm-legend">
            <span>{view === "zmap" ? "0σ" : "quiet"}</span>
            <span
              className="hm-ramp"
              style={{ background: gradientCss(ramp, theme) }}
              role="img"
              aria-label={
                view === "zmap"
                  ? `Colour scale from 0 sigma to ${formatZ(zCeiling)} sigma`
                  : "Colour scale from low to high activation energy, scaled per layer"
              }
            />
            <span>
              {view === "zmap" ? `${formatZ(zCeiling)}σ` : "peak (per layer)"}
            </span>
            <span className="muted" style={{ marginLeft: 8 }}>
              {view === "zmap"
                ? "log₁₀ scale, shared across layers — each decade is an equal step"
                : "each row scaled to its own peak · log-compressed"}
            </span>
          </div>
        </>
      ) : (
        <p className="muted">This matrix is not present in the report.</p>
      )}

      {showTable && <MatrixTable heatmap={heatmap} view={view} />}
    </section>
  );
}

function describeCell(heatmap: DecodedHeatmap, row: number, col: number) {
  const layer = heatmap.layers[row] ?? `row ${row}`;
  const channels = heatmap.channels[row] ?? 0;
  let span: string;
  if (channels > 0) {
    const lo = Math.floor((channels * col) / heatmap.bins);
    const hi = Math.max(lo, Math.floor((channels * (col + 1)) / heatmap.bins) - 1);
    span = `ch ${lo.toLocaleString()}–${hi.toLocaleString()}`;
  } else {
    span = `band ${col}`;
  }
  return { layer, span };
}

/** The accessible twin: exact per-layer numbers, no colour encoding. */
function MatrixTable({ heatmap, view }: { heatmap: DecodedHeatmap; view: MatrixKind }) {
  const { rows, bins } = heatmap;
  const matrix = heatmap[view];
  if (!matrix) return null;

  const summary = Array.from({ length: rows }, (_, r) => {
    let max = 0;
    let sum = 0;
    let argmax = 0;
    for (let c = 0; c < bins; c += 1) {
      const v = matrix[r * bins + c];
      if (!Number.isFinite(v)) continue;
      sum += v;
      if (v > max) {
        max = v;
        argmax = c;
      }
    }
    return { r, max, mean: sum / bins, argmax };
  });

  return (
    <div className="table-wrap" style={{ marginTop: 18 }}>
      <table className="data">
        <caption className="sr-only">
          {VIEW_LABELS[view]} summarised per layer: peak, mean and the channel band holding the
          peak.
        </caption>
        <thead>
          <tr>
            <th scope="col">Layer</th>
            <th scope="col">Channels</th>
            <th scope="col">Peak</th>
            <th scope="col">Mean</th>
            <th scope="col">Peak band</th>
          </tr>
        </thead>
        <tbody>
          {summary.map((row) => (
            <tr key={row.r}>
              <td className="strong">{heatmap.layers[row.r] ?? `row ${row.r}`}</td>
              <td className="num">{(heatmap.channels[row.r] ?? 0).toLocaleString()}</td>
              <td className="num">
                {view === "zmap" ? formatZ(row.max) : formatSci(row.max)}
              </td>
              <td className="num">
                {view === "zmap" ? formatZ(row.mean) : formatSci(row.mean)}
              </td>
              <td className="num">{row.argmax}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
