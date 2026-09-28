# NeuroFence — Forensic Report Viewer

A static web viewer for NeuroFence scan reports, deployable to Vercel.

**The scanner does not run here.** It cannot: `torch` alone is ~476 MB unzipped against
Vercel's 250 MB serverless limit, `app.py` is a PyQt6 desktop GUI with no display server on
serverless, there is no GPU, and a real scan far exceeds the 300 s execution ceiling. More
importantly, uploading an untrusted model to a third-party cloud to check whether it is
backdoored would invert NeuroFence's entire threat model.

So the split is: **scanning stays air-gapped on your machine; only the finished report is
viewed here.**

```
air-gapped host                          anywhere
┌──────────────────────────┐             ┌────────────────────────┐
│ python app.py            │  report     │ this viewer            │
│  → scan model            │  .json      │  → drop the file in    │
│  → File ▸ Export report  │ ──────────▶ │  → parsed in-browser   │
└──────────────────────────┘  (by hand)  └────────────────────────┘
```

## Privacy properties

This is a **fully static export** (`output: "export"`) — no serverless function, no API route,
no database, no analytics. A report you open is parsed by JavaScript in your tab and never
leaves it.

That is enforced, not merely promised: `vercel.json` ships a `Content-Security-Policy` with
`connect-src 'self'`, so the page is structurally incapable of transmitting your report to any
third-party origin. `form-action 'none'` and `frame-ancestors 'none'` close the other exits.

Note what a report *does* contain if you choose to share the file: model path, architecture,
layer names, neuron indices and the adversarial prompts that triggered each finding. No
weights, no tokenizer, no model output.

## Run locally

```bash
cd neurofence-web
npm install
npm run dev
```

Build the static bundle:

```bash
npm run build      # emits ./out — 686 KB total, including both sample reports
```

## Deploy to Vercel

The web app lives in a subdirectory, so point Vercel at it:

```bash
cd neurofence-web
vercel --prod
```

Or from the dashboard: import the repo, then set **Root Directory** to `neurofence-web`.
Framework detection (Next.js), build command and output directory are already declared in
`vercel.json`.

## What it renders

| Panel | Source field | Form |
|---|---|---|
| Safety score + verdict | `safety_score`, `verdict` | Hero figure + status glyph |
| Score model | `score_breakdown` | Single-series penalty bars |
| Activation matrix | `heatmap` (embedded) | Canvas cell grid + scale legend |
| Peak Z by layer | `layer_findings` | log₁₀ bars, one axis |
| Anomalous neurons | `anomalies` | Table |
| Integrity & provenance | `integrity_findings`, `metadata` | Status list |

A report exported by an older build has no `heatmap` block; the viewer detects that and falls
back to the layer profile rather than failing.

## Visualization notes

**Colour ramp.** The default is a *semantic heat* ramp built from the project's status palette,
arranged so lightness rises monotonically with magnitude (measured OKLab L: 0.217 → 0.937).
Monotonic lightness is what keeps a heatmap readable under colour-vision deficiency.

A `Ramp: Console` toggle reproduces the desktop PyQt LUT exactly, for operators moving between
the two tools. It is **not** the default, and measurably should not be: that green→red ramp is
non-monotonic in lightness (L peaks at 0.856 on yellow, then *falls* to 0.644 at full red), so
the hottest neurons render darker than mid-range ones. The same applies to the desktop canvas.

**Z-scale.** Z-scores span six orders of magnitude — a dormant neuron with σ = 0 clips at 1e6
while healthy layers peak near 10. The matrix therefore normalises Z in log₁₀ space; dividing by
the ceiling linearly renders every honest layer as black.

**Per-layer vs shared scaling.** Baseline and adversarial energy are scaled per row, so a quiet
early layer and a loud late one are both legible. The Z map uses one shared scale, because the
whole point is comparing layers against each other.

**Accessibility.** Both themes are selected sets of steps, not an inverted flip. Status is
always colour + glyph + word. Every chart has a table-view twin. The two colours that carry
meaning in the layer chart (`#3987e5` / `#d03b3b`) pass all six palette checks in both modes —
worst-pair CVD ΔE 25.7 dark / 23.8 light, against a ≥ 8 target.
