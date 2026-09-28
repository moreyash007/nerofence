"use client";

import { useRef, useState } from "react";

import { parseReport, ReportParseError, type ScanReport } from "@/lib/types";

/** Reports are small; this guard is about not hanging the tab on a stray file. */
const MAX_BYTES = 64 * 1024 * 1024;

interface Props {
  onLoad: (report: ScanReport, name: string) => void;
}

export default function DropZone({ onLoad }: Props) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  async function ingest(file: File) {
    setError(null);
    if (file.size > MAX_BYTES) {
      setError(`${file.name} is ${(file.size / 1048576).toFixed(0)} MB — larger than the 64 MB limit.`);
      return;
    }
    setBusy(true);
    try {
      const report = parseReport(JSON.parse(await file.text()));
      onLoad(report, file.name);
    } catch (err) {
      if (err instanceof ReportParseError) setError(err.message);
      else if (err instanceof SyntaxError) setError(`${file.name} is not valid JSON.`);
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function loadSample(path: string, label: string) {
    setError(null);
    setBusy(true);
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`Could not load the ${label} sample (HTTP ${res.status}).`);
      onLoad(parseReport(await res.json()), `${label} (sample)`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div
        className={over ? "dropzone over" : "dropzone"}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file) void ingest(file);
        }}
      >
        <h2>Open a forensic report</h2>
        <p>
          Drop a NeuroFence <code>report.json</code> here, or choose a file. In the desktop app it
          comes from <strong>File → Export report</strong>.
        </p>

        <div className="actions">
          <button
            type="button"
            className="btn"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            {busy ? "Reading…" : "Choose report…"}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => void loadSample("./samples/poisoned-report.json", "Backdoored model")}
            disabled={busy}
          >
            Sample: backdoored
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => void loadSample("./samples/clean-report.json", "Clean model")}
            disabled={busy}
          >
            Sample: clean
          </button>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="application/json,.json"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void ingest(file);
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      <p className="privacy-note">
        This viewer is a static page. Your report is parsed in the browser and never uploaded —
        there is no server, no API and no storage behind it. That matters here: NeuroFence exists to
        audit models on air-gapped hosts, so the report itself (layer names, neuron indices, trigger
        prompts) stays on the machine you opened it on.
      </p>
    </div>
  );
}
