"use client";

import { STATUS, STATUS_GLYPH, statusForScore } from "@/lib/colormap";
import { formatInt, formatZ } from "@/lib/format";
import type { ScanReport } from "@/lib/types";

/**
 * The headline. One number is the story here, so it gets a hero figure and stat
 * tiles — not a gauge, donut or one-bar chart.
 *
 * The verdict carries a glyph and a word alongside its colour, because status
 * hue alone is not an accessible encoding (and `warning`/`serious` sit below 3:1
 * on the light surface by design).
 */
export default function ScoreHero({ report }: { report: ScanReport }) {
  const role = statusForScore(report.safety_score);
  const color = STATUS[role];
  const t = report.totals;

  return (
    <section className="card" aria-labelledby="verdict-heading">
      <h2 className="card-title" id="verdict-heading">
        Deployment verdict
      </h2>

      <div className="hero">
        <div>
          <div className="hero-figure" style={{ color }}>
            {report.safety_score.toFixed(1)}
            <span className="denom"> / 100</span>
          </div>
        </div>

        <div style={{ flex: "1 1 260px", minWidth: 0 }}>
          <span className="verdict-badge" style={{ color }}>
            <span className="verdict-glyph" aria-hidden="true">
              {STATUS_GLYPH[role]}
            </span>
            {report.verdict}
          </span>
          <p
            style={{
              margin: "10px 0 0",
              color: "var(--text-secondary)",
              fontSize: 13.5,
              lineHeight: 1.6,
            }}
          >
            {report.rationale}
          </p>
        </div>
      </div>

      <div className="stat-row">
        <Stat k="Layers" v={formatInt(t.layers)} />
        <Stat k="Neurons" v={formatInt(t.neurons)} />
        <Stat
          k="Flagged"
          v={formatInt(t.flagged_neurons)}
          color={t.flagged_neurons > 0 ? STATUS.critical : undefined}
        />
        <Stat k="Flagged layers" v={formatInt(t.flagged_layers)} />
        <Stat k="Peak Z" v={formatZ(t.max_z)} />
        <Stat k="Threshold" v={`${report.z_threshold}σ`} />
        <Stat k="Passes" v={`${formatInt(t.baseline_passes)} + ${formatInt(t.fuzz_passes)}`} />
      </div>
    </section>
  );
}

function Stat({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div className="stat">
      <span className="k">{k}</span>
      <span className="v" style={color ? { color } : undefined}>
        {v}
      </span>
    </div>
  );
}
