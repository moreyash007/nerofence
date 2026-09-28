"use client";

import { breakdownLabel } from "@/lib/format";

/**
 * How the 100-point budget was spent.
 *
 * One series, one colour — the bars encode magnitude of penalty, and a value
 * ramp across nominal categories would double-encode length as hue for nothing.
 * `total_penalty` is pulled out as the sum rather than drawn as a peer bar.
 */
export default function ScoreBreakdown({ breakdown }: { breakdown: Record<string, number> }) {
  const entries = Object.entries(breakdown).filter(([k]) => k !== "total_penalty");
  const total = breakdown.total_penalty;

  if (entries.length === 0) {
    return (
      <section className="card" aria-labelledby="breakdown-heading">
        <h2 className="card-title" id="breakdown-heading">
          Score model
        </h2>
        <p className="muted">No deductions were applied.</p>
      </section>
    );
  }

  const ordered = entries.sort((a, b) => b[1] - a[1]);
  const ceiling = Math.max(...ordered.map(([, v]) => v), 1);

  return (
    <section className="card" aria-labelledby="breakdown-heading">
      <h2 className="card-title" id="breakdown-heading">
        Score model — deductions
      </h2>

      {ordered.map(([key, value]) => (
        <div className="bar-row" key={key} style={{ gridTemplateColumns: "1fr 56px" }}>
          <div>
            <div className="bar-label" style={{ marginBottom: 4 }}>
              {breakdownLabel(key)}
            </div>
            <span className="bar-track" style={{ display: "block" }}>
              <span
                className="bar-fill"
                style={{ width: `${Math.max(1, (value / ceiling) * 100)}%` }}
              />
            </span>
          </div>
          <span className="bar-value">−{value.toFixed(1)}</span>
        </div>
      ))}

      {typeof total === "number" && (
        <p
          style={{
            marginTop: 14,
            paddingTop: 12,
            borderTop: "1px solid var(--grid)",
            fontSize: 12.5,
            color: "var(--text-secondary)",
          }}
        >
          Total penalty <strong className="tnum">−{total.toFixed(1)}</strong> from a 100-point
          budget.
        </p>
      )}
    </section>
  );
}
