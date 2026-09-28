"use client";

import { useState } from "react";

import { STATUS } from "@/lib/colormap";
import { anomalyKey, formatSci, formatZ, previewPrompt } from "@/lib/format";
import type { NeuronAnomaly } from "@/lib/types";

/**
 * Ranked dormant-neuron findings.
 *
 * A table, not a chart: nine heterogeneous columns per row is tabular data, and
 * the forensic value is in the exact layer name, neuron index and the prompt
 * that woke it — every one of which an analyst copies verbatim.
 *
 * Trigger prompts are rendered through `previewPrompt`, which escapes control
 * characters and reveals zero-width / bidi-override smuggling rather than
 * letting a hostile prompt render as invisible text in the report.
 */
export default function AnomalyTable({ anomalies }: { anomalies: NeuronAnomaly[] }) {
  const [expanded, setExpanded] = useState(false);
  const limit = expanded ? anomalies.length : 10;
  const shown = anomalies.slice(0, limit);

  if (anomalies.length === 0) {
    return (
      <section className="card grid-full" aria-labelledby="anomalies-heading">
        <h2 className="card-title" id="anomalies-heading">
          Anomalous neurons
        </h2>
        <p style={{ margin: 0, color: STATUS.good, fontWeight: 600, fontSize: 13.5 }}>
          ✓ No neuron exceeded the Z-score threshold.
        </p>
      </section>
    );
  }

  return (
    <section className="card grid-full" aria-labelledby="anomalies-heading">
      <h2 className="card-title" id="anomalies-heading">
        Top anomalous neurons ({anomalies.length.toLocaleString()} flagged)
      </h2>

      <div className="table-wrap">
        <table className="data">
          <caption className="sr-only">
            Flagged neurons ranked by Z-score, with baseline statistics and the adversarial
            prompt that produced each peak.
          </caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Layer</th>
              <th scope="col">Neuron</th>
              <th scope="col">Z</th>
              <th scope="col">μ base</th>
              <th scope="col">σ base</th>
              <th scope="col">Peak</th>
              <th scope="col">Dormancy</th>
              <th scope="col">Woken by</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((a, i) => (
              <Row key={anomalyKey(a, i)} anomaly={a} rank={i + 1} />
            ))}
          </tbody>
        </table>
      </div>

      {anomalies.length > 10 && (
        <button
          type="button"
          className="btn"
          style={{ marginTop: 14 }}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show top 10 only" : `Show all ${anomalies.length.toLocaleString()}`}
        </button>
      )}
    </section>
  );
}

function Row({ anomaly, rank }: { anomaly: NeuronAnomaly; rank: number }) {
  const prompt = previewPrompt(anomaly.trigger_prompt ?? "");
  return (
    <>
      <tr>
        <td className="num muted">{rank}</td>
        <td className="strong">{anomaly.layer}</td>
        <td className="num" style={{ color: "var(--series-1)" }}>
          {anomaly.neuron}
        </td>
        <td className="num" style={{ color: STATUS.critical, fontWeight: 700 }}>
          {formatZ(anomaly.z_score, anomaly.z_clipped)}
        </td>
        <td className="num">{formatSci(anomaly.baseline_mean)}</td>
        <td className="num">{formatSci(anomaly.baseline_std)}</td>
        <td className="num">{formatSci(anomaly.fuzz_peak)}</td>
        <td className="num">{(anomaly.dormancy * 100).toFixed(0)}%</td>
        <td style={{ color: STATUS.warning }}>
          {anomaly.trigger_label ? (
            <>
              <span className="muted">[{anomaly.trigger_category}]</span> {anomaly.trigger_label}
            </>
          ) : (
            "—"
          )}
        </td>
      </tr>
      {prompt && (
        <tr className="sub">
          <td />
          <td colSpan={8}>↳ {prompt}</td>
        </tr>
      )}
    </>
  );
}
