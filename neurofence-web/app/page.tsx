"use client";

import { useMemo, useState } from "react";

import AnomalyTable from "@/components/AnomalyTable";
import DropZone from "@/components/DropZone";
import Heatmap from "@/components/Heatmap";
import IntegrityList from "@/components/IntegrityList";
import LayerProfile from "@/components/LayerProfile";
import ScoreBreakdown from "@/components/ScoreBreakdown";
import ScoreHero from "@/components/ScoreHero";
import ThemeToggle, { type Theme } from "@/components/ThemeToggle";
import { decodeHeatmap } from "@/lib/decode";
import type { ScanReport } from "@/lib/types";

export default function Page() {
  const [report, setReport] = useState<ScanReport | null>(null);
  const [source, setSource] = useState<string>("");
  const [theme, setTheme] = useState<Theme>("dark");

  const heatmap = useMemo(() => (report ? decodeHeatmap(report) : null), [report]);

  return (
    <main className="shell">
      <header className="masthead">
        <div className="brand">
          <h1>NEUROFENCE</h1>
          <span className="tag">Forensic report viewer · offline · no upload</span>
        </div>
        <div className="masthead-actions">
          {report && (
            <>
              <span className="muted mono" style={{ fontSize: 11 }}>
                {source}
              </span>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setReport(null);
                  setSource("");
                }}
              >
                Load another
              </button>
            </>
          )}
          <ThemeToggle onChange={setTheme} />
        </div>
      </header>

      {!report ? (
        <DropZone
          onLoad={(r, name) => {
            setReport(r);
            setSource(name);
          }}
        />
      ) : (
        <div className="grid-main">
          <ScoreHero report={report} />
          <ScoreBreakdown breakdown={report.score_breakdown} />

          {heatmap ? (
            <Heatmap heatmap={heatmap} zThreshold={report.z_threshold} theme={theme} />
          ) : (
            <section className="card grid-full">
              <h2 className="card-title">Activation matrix</h2>
              <p className="muted" style={{ margin: 0 }}>
                This report has no embedded heatmap. Re-export it from a build of the desktop app
                that includes the matrix block, or read the per-layer profile below.
              </p>
            </section>
          )}

          <LayerProfile findings={report.layer_findings} zThreshold={report.z_threshold} />
          <AnomalyTable anomalies={report.anomalies} />
          <IntegrityList report={report} />
        </div>
      )}
    </main>
  );
}
