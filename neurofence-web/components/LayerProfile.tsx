"use client";

import { STATUS } from "@/lib/colormap";
import { formatInt, formatZ, shortLayerName } from "@/lib/format";
import type { LayerFinding } from "@/lib/types";

/**
 * Peak Z-score per layer.
 *
 * One measure, one axis — deliberately not a dual-axis plot against flagged
 * count, which would invent a relationship between two different scales. The
 * flagged count rides along as a direct label on the rows where it is non-zero.
 *
 * The scale is log10, because a clipped Z of 1e6 beside a healthy 16 would
 * otherwise collapse every honest layer to a hairline. Colour is status, not
 * identity: a layer either has flagged neurons or it does not.
 */
export default function LayerProfile({
  findings,
  zThreshold,
}: {
  findings: LayerFinding[];
  zThreshold: number;
}) {
  if (findings.length === 0) {
    return (
      <section className="card" aria-labelledby="layers-heading">
        <h2 className="card-title" id="layers-heading">
          Layer profile
        </h2>
        <p className="muted">No per-layer findings in this report.</p>
      </section>
    );
  }

  const ordered = [...findings].sort((a, b) => a.layer_index - b.layer_index);
  const ceiling = Math.max(...ordered.map((f) => f.max_z), zThreshold * 2, 10);
  const logCeiling = Math.log10(ceiling + 1);
  const thresholdPct = (Math.log10(zThreshold + 1) / logCeiling) * 100;

  return (
    <section className="card grid-full" aria-labelledby="layers-heading">
      <h2 className="card-title" id="layers-heading">
        Peak Z-score by layer
      </h2>

      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--series-1)" }} />
          Within tolerance
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: STATUS.critical }} />
          Has flagged neurons
        </span>
        <span className="legend-item muted">log₁₀ scale · vertical rule = {zThreshold}σ threshold</span>
      </div>

      <div style={{ position: "relative" }}>
        {/* Threshold reference rule. Solid hairline; the legend names it. */}
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `calc(104px + 10px + (100% - 104px - 74px - 20px) * ${thresholdPct / 100})`,
            width: 1,
            background: "var(--axis)",
          }}
        />
        {ordered.map((f) => {
          const pct = Math.max(0.6, (Math.log10(Math.max(f.max_z, 0) + 1) / logCeiling) * 100);
          const flagged = f.flagged > 0;
          return (
            <div className="bar-row" key={f.layer}>
              <span className="bar-label" title={f.layer}>
                {shortLayerName(f.layer)}
              </span>
              <span className="bar-track">
                <span
                  className="bar-fill"
                  style={{
                    width: `${pct}%`,
                    background: flagged ? STATUS.critical : "var(--series-1)",
                  }}
                />
              </span>
              <span className="bar-value">
                {formatZ(f.max_z)}
                {flagged && (
                  <span style={{ color: STATUS.critical }}> ·{formatInt(f.flagged)}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      <details style={{ marginTop: 14 }}>
        <summary className="muted" style={{ cursor: "pointer", fontSize: 12 }}>
          Table view — exact per-layer values
        </summary>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Layer</th>
                <th scope="col">Neurons</th>
                <th scope="col">Flagged</th>
                <th scope="col">Max Z</th>
                <th scope="col">Mean Z</th>
                <th scope="col">p99 Z</th>
                <th scope="col">Max dormancy</th>
              </tr>
            </thead>
            <tbody>
              {ordered.map((f) => (
                <tr key={f.layer}>
                  <td className="strong">{f.layer}</td>
                  <td className="num">{formatInt(f.channels)}</td>
                  <td className="num" style={f.flagged ? { color: STATUS.critical } : undefined}>
                    {formatInt(f.flagged)}
                  </td>
                  <td className="num">{formatZ(f.max_z)}</td>
                  <td className="num">{formatZ(f.mean_z)}</td>
                  <td className="num">{formatZ(f.p99_z)}</td>
                  <td className="num">{(f.max_dormancy * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
