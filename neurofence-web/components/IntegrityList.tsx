"use client";

import { STATUS, STATUS_GLYPH, statusForSeverity } from "@/lib/colormap";
import type { IntegrityFinding, ScanReport } from "@/lib/types";

/**
 * Static (pre-execution) findings about the model folder, plus the provenance
 * block. Each row pairs its status colour with a glyph and the severity word, so
 * nothing is encoded by hue alone.
 */
export default function IntegrityList({ report }: { report: ScanReport }) {
  const findings: IntegrityFinding[] = report.integrity_findings ?? [];
  const meta = report.metadata ?? {};

  return (
    <section className="card" aria-labelledby="integrity-heading">
      <h2 className="card-title" id="integrity-heading">
        Static integrity &amp; provenance
      </h2>

      {findings.length === 0 ? (
        <p style={{ margin: "0 0 14px", color: STATUS.good, fontSize: 13, fontWeight: 600 }}>
          ✓ No static integrity findings.
        </p>
      ) : (
        <div style={{ marginBottom: 14 }}>
          {findings.map((f, i) => {
            const role = statusForSeverity(f.severity);
            return (
              <div className="finding" key={`${f.code}-${i}`}>
                <span className="glyph" style={{ color: STATUS[role] }} aria-hidden="true">
                  <span>{STATUS_GLYPH[role]}</span>
                </span>
                <div style={{ minWidth: 0 }}>
                  <div className="code" style={{ color: STATUS[role] }}>
                    {f.severity} · {f.code}
                  </div>
                  <div className="msg">{f.message}</div>
                  {f.evidence && f.evidence.length > 0 && (
                    <div className="mono muted" style={{ fontSize: 11, marginTop: 4 }}>
                      {f.evidence.slice(0, 6).join(", ")}
                      {f.evidence.length > 6 ? ` … +${f.evidence.length - 6}` : ""}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <dl style={{ margin: 0, fontSize: 12, lineHeight: 1.75 }}>
        <MetaRow label="Model" value={str(meta.model_path)} mono />
        <MetaRow label="Architecture" value={archOf(meta)} />
        <MetaRow label="Device" value={str(meta.device)} />
        <MetaRow label="Precision" value={str(meta.dtype)} />
        <MetaRow label="Sharded" value={meta.sharded === true ? "yes" : undefined} />
        <MetaRow label="Duration" value={numStr(meta.duration_seconds, "s")} />
        <MetaRow label="Non-finite" value={numStr(meta.nonfinite_activations)} />
      </dl>
    </section>
  );
}

function MetaRow({
  label,
  value,
  mono,
}: {
  label: string;
  value?: string;
  mono?: boolean;
}) {
  if (!value) return null;
  return (
    <div style={{ display: "flex", gap: 10 }}>
      <dt style={{ color: "var(--text-muted)", minWidth: 92, flex: "none" }}>{label}</dt>
      <dd
        style={{
          margin: 0,
          color: "var(--text-secondary)",
          wordBreak: "break-all",
          fontFamily: mono ? "var(--mono)" : undefined,
          fontSize: mono ? 11 : undefined,
        }}
      >
        {value}
      </dd>
    </div>
  );
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numStr(value: unknown, suffix = ""): string | undefined {
  return typeof value === "number" ? `${value}${suffix}` : undefined;
}

function archOf(meta: Record<string, unknown>): string | undefined {
  const arch = meta.architectures;
  const type = typeof meta.model_type === "string" ? meta.model_type : "";
  const names = Array.isArray(arch) ? arch.filter((a) => typeof a === "string").join(", ") : "";
  if (names && type) return `${names} (${type})`;
  return names || type || undefined;
}
