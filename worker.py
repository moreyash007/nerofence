"""
NeuroFence :: worker
====================

Asynchronous scan orchestration.

Every PyTorch call in NeuroFence happens on this ``QThread``.  The GUI thread
never touches the model, never allocates on the GPU and never blocks -- it only
receives immutable payloads through Qt's queued signal delivery, so the interface
stays at full frame rate even while a multi-billion-parameter model is being
fuzzed.

Pipeline
--------
1. static integrity audit of the model directory
2. safetensors-only model load and MLP instrumentation
3. baseline recording  (natural-language traffic -> mu, sigma per neuron)
4. adversarial fuzzing  (random / structural / trigger -> peak per neuron)
5. statistical analysis  (Z-scores, flagged neurons, safety score)
6. deterministic teardown (hooks removed, model unmounted, caches flushed)

Step 6 runs in a ``finally`` block and therefore executes on success, on failure
and on user abort alike.
"""

from __future__ import annotations

import base64
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from detector import BackdoorDetector, ScanReport
from fuzzer import AdversarialFuzzer, FuzzCase
import torch

from sandbox_tracker import (
    ActivationStore,
    IntegrityFinding,
    ModelSandboxTracker,
)

LOGGER = logging.getLogger("neurofence.worker")

__all__ = [
    "ScanConfig",
    "HeatmapFrame",
    "ScanWorkerThread",
    "build_heatmap_matrix",
    "build_report_heatmap",
]

#: Horizontal resolution of the heatmap: neurons are max-pooled into this many
#: channel bins so the canvas cost is independent of ``intermediate_size``.
DEFAULT_HEATMAP_BINS = 192


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@dataclass
class ScanConfig:
    """Everything the worker needs to run one scan."""

    model_path: str
    device: str = "auto"
    dtype: str = "auto"
    z_threshold: float = 4.5
    baseline_passes: int = 24
    fuzz_passes: int = 96
    max_sequence_length: int = 256
    seed: int = 1337
    heatmap_bins: int = DEFAULT_HEATMAP_BINS
    min_abs_delta_ratio: float = 0.02
    spatial_mad_threshold: float = 6.0
    refuse_on_pickle: bool = False
    shard_across_devices: bool = False
    extra_triggers: List[str] = field(default_factory=list)
    emit_every: int = 4              # forward passes between heatmap refreshes

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "z_threshold": self.z_threshold,
            "baseline_passes": self.baseline_passes,
            "fuzz_passes": self.fuzz_passes,
            "max_sequence_length": self.max_sequence_length,
            "seed": self.seed,
            "min_abs_delta_ratio": self.min_abs_delta_ratio,
            "spatial_mad_threshold": self.spatial_mad_threshold,
            "refuse_on_pickle": self.refuse_on_pickle,
            "shard_across_devices": self.shard_across_devices,
            "extra_triggers": list(self.extra_triggers),
        }


@dataclass
class HeatmapFrame:
    """An immutable snapshot of the activation matrix, safe to cross threads."""

    layer_names: List[str]
    baseline: np.ndarray                       # (layers, bins) raw energy
    fuzz: np.ndarray                           # (layers, bins) raw energy
    zmap: Optional[np.ndarray] = None          # (layers, bins) max-pooled Z
    channels: List[int] = field(default_factory=list)   # true neuron count per layer
    phase: str = "IDLE"
    detail: str = ""
    completed: int = 0
    total: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.baseline.shape if self.baseline.size else (0, 0)


def build_report_heatmap(frame: "HeatmapFrame") -> Dict[str, object]:
    """Serialise a heatmap frame for embedding in an exported JSON report.

    ``ScanReport.to_dict()`` deliberately omits the per-neuron Z vectors -- they
    are large and in-process only. But an offline report viewer still needs the
    matrix to redraw the activation map, so the *pooled* matrices (already
    reduced to ``layers x bins``) are embedded here as base64 little-endian
    float32, row-major.

    Exact values are preserved rather than quantised, so a viewer's tooltips can
    report the same numbers the desktop canvas shows. At the default 192 bins a
    32-layer model costs ~32 KB per matrix.
    """

    def encode(matrix: Optional[np.ndarray]) -> Optional[str]:
        if matrix is None or getattr(matrix, "size", 0) == 0:
            return None
        payload = np.ascontiguousarray(matrix, dtype="<f4").tobytes()
        return base64.b64encode(payload).decode("ascii")

    rows, bins = (frame.baseline.shape if frame.baseline.size else (0, 0))
    return {
        "encoding": "base64-float32-le-rowmajor",
        "rows": int(rows),
        "bins": int(bins),
        "layers": list(frame.layer_names),
        "channels": [int(c) for c in frame.channels],
        "baseline": encode(frame.baseline),
        "fuzz": encode(frame.fuzz),
        "zmap": encode(frame.zmap),
    }


def build_heatmap_matrix(
    vectors: Dict[str, np.ndarray],
    layer_names: Sequence[str],
    bins: int = DEFAULT_HEATMAP_BINS,
) -> np.ndarray:
    """Max-pool per-neuron vectors into a fixed ``(layers, bins)`` matrix.

    Max-pooling (rather than averaging) is essential: a single poisoned neuron in
    a 11k-wide layer would be averaged into invisibility, but survives a max.
    """
    bins = max(1, int(bins))
    rows: List[np.ndarray] = []
    for name in layer_names:
        vector = np.asarray(vectors.get(name, ()), dtype=np.float32).reshape(-1)
        if vector.size == 0:
            rows.append(np.zeros(bins, dtype=np.float32))
            continue
        if vector.size <= bins:
            row = np.zeros(bins, dtype=np.float32)
            # Spread the few channels evenly rather than clumping them at the left.
            positions = (np.arange(vector.size) * bins) // max(vector.size, 1)
            np.maximum.at(row, positions, vector)
            rows.append(row)
            continue
        # Pad to an exact multiple of ``bins`` with -inf-safe zeros, then max-pool.
        per_bin = int(np.ceil(vector.size / bins))
        padded_size = per_bin * bins
        padded = np.zeros(padded_size, dtype=np.float32)
        padded[: vector.size] = vector
        rows.append(padded.reshape(bins, per_bin).max(axis=1))
    if not rows:
        return np.zeros((0, bins), dtype=np.float32)
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class ScanWorkerThread(QThread):
    """Runs the full NeuroFence pipeline off the GUI thread.

    Signals
    -------
    ``progress(int, str)``       percentage 0-100 and a human-readable status line
    ``results_ready(object)``    a :class:`HeatmapFrame` for live visualisation
    ``finished_report(object)``  the final :class:`~detector.ScanReport`
    ``log(str)``                 free-form forensic log lines
    ``error(str)``               fatal failure; no report will follow
    """

    progress = pyqtSignal(int, str)
    results_ready = pyqtSignal(object)
    finished_report = pyqtSignal(object)
    log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config: ScanConfig, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.config = config
        self.report: Optional[ScanReport] = None
        self._tracker: Optional[ModelSandboxTracker] = None
        self._started_at = 0.0
        self._layer_channels: List[int] = []

    # -- helpers -----------------------------------------------------------

    def _emit_progress(self, percent: int, message: str) -> None:
        self.progress.emit(int(max(0, min(100, percent))), message)

    def _emit_log(self, message: str) -> None:
        LOGGER.info(message)
        self.log.emit(message)

    def _aborted(self) -> bool:
        return self.isInterruptionRequested()

    def _emit_frame(
        self,
        baseline_store: ActivationStore,
        fuzz_store: ActivationStore,
        layer_names: Sequence[str],
        phase: str,
        detail: str,
        completed: int,
        total: int,
        zmap: Optional[np.ndarray] = None,
    ) -> None:
        bins = self.config.heatmap_bins
        baseline_matrix = build_heatmap_matrix(baseline_store.peak_vectors(), layer_names, bins)
        if fuzz_store.is_empty():
            fuzz_matrix = np.zeros_like(baseline_matrix)
        else:
            fuzz_matrix = build_heatmap_matrix(fuzz_store.peak_vectors(), layer_names, bins)
        self.results_ready.emit(
            HeatmapFrame(
                layer_names=list(layer_names),
                baseline=baseline_matrix,
                fuzz=fuzz_matrix,
                zmap=zmap,
                channels=list(self._layer_channels),
                phase=phase,
                detail=detail,
                completed=completed,
                total=total,
            )
        )

    def _run_phase(
        self,
        tracker: ModelSandboxTracker,
        cases: Sequence[FuzzCase],
        store: ActivationStore,
        peer_store: ActivationStore,
        layer_names: Sequence[str],
        phase: str,
        progress_start: int,
        progress_end: int,
        baseline_first: bool,
    ) -> bool:
        """Fire a batch of prompts, folding each pass into ``store``.

        Returns ``False`` if the user aborted mid-phase.
        """
        total = len(cases)
        span = max(1, progress_end - progress_start)
        skipped = 0

        for position, case in enumerate(cases):
            if self._aborted():
                return False

            try:
                frame = tracker.run_forward(case.prompt)
            except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill a scan
                skipped += 1
                self._emit_log(
                    f"  ! pass {position + 1}/{total} ({case.label}) failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if not frame:
                skipped += 1
                continue
            store.add_frame(frame, case_index=case.index)

            # Channel counts are only knowable after a tensor has actually flowed.
            if not self._layer_channels or not any(self._layer_channels):
                self._layer_channels = [
                    int(np.asarray(frame[name]).size) if name in frame else 0
                    for name in layer_names
                ]

            percent = progress_start + int(span * (position + 1) / max(total, 1))
            status = f"{phase} {position + 1}/{total}  ·  [{case.category}] {case.label}"
            gpu = tracker.gpu_summary()
            if gpu:
                status = f"{status}  ·  {gpu}"
            self._emit_progress(percent, status)

            if (position + 1) % max(1, self.config.emit_every) == 0 or position + 1 == total:
                base_store = store if baseline_first else peer_store
                adv_store = peer_store if baseline_first else store
                self._emit_frame(
                    base_store,
                    adv_store,
                    layer_names,
                    phase=phase,
                    detail=case.preview(80),
                    completed=position + 1,
                    total=total,
                )

            # Periodically hand the allocator its blocks back; long trigger sweeps
            # with wildly varying sequence lengths fragment CUDA memory otherwise.
            if (position + 1) % 32 == 0:
                tracker.flush_memory()

        if skipped:
            self._emit_log(f"  {skipped} prompt(s) skipped during {phase.lower()}.")
        return True

    # -- main --------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - linear pipeline, kept in one place on purpose
        self._started_at = time.time()
        config = self.config
        tracker: Optional[ModelSandboxTracker] = None

        try:
            # -- 1. static audit -------------------------------------------
            self._emit_progress(1, "Auditing model directory…")
            tracker = ModelSandboxTracker(
                model_path=config.model_path,
                device=config.device,
                dtype=config.dtype,
                max_sequence_length=config.max_sequence_length,
                refuse_on_pickle=config.refuse_on_pickle,
                shard_across_devices=config.shard_across_devices,
            )
            self._tracker = tracker

            audit = tracker.audit()
            self._emit_log(f"Target        : {config.model_path}")
            self._emit_log(
                f"Architecture  : {', '.join(audit.architectures) or 'unknown'} "
                f"(model_type={audit.model_type})"
            )
            self._emit_log(
                f"Weights       : {len(audit.safetensors_files)} safetensors shard(s), "
                f"{audit.total_safetensors_bytes / (1024 ** 3):.2f} GiB"
            )
            self._emit_log(f"Execution     : device={tracker.device}")
            for label, description in ModelSandboxTracker.describe_devices():
                if label == tracker.device or (
                    tracker.device.startswith("cuda") and label.startswith("cuda")
                ):
                    self._emit_log(f"  {description}")
            for finding in audit.findings:
                self._emit_log(f"  [{finding.severity}] {finding.code}: {finding.message}")

            if not audit.loadable:
                self.error.emit(
                    "No .safetensors weights were found in the selected directory.\n\n"
                    "NeuroFence refuses to deserialise pickle-backed checkpoints "
                    "(.bin/.pt/.ckpt) because torch.load executes arbitrary code on "
                    "untrusted archives. Convert the model to safetensors on an "
                    "isolated host before scanning."
                )
                return
            if self._aborted():
                return

            # -- 2. load + instrument --------------------------------------
            self._emit_progress(6, "Loading model (safetensors, offline)…")
            tracker.load()
            self._emit_log(
                f"Precision     : {str(tracker.torch_dtype).replace('torch.', '')}"
                f"{' (auto-selected)' if config.dtype == 'auto' else ''}"
            )
            self._emit_log(f"Resident size : {tracker.memory_footprint_mb():,.1f} MiB")
            gpu = tracker.gpu_summary()
            if gpu:
                self._emit_log(f"Device memory : {gpu}")
            if tracker.is_sharded:
                self._emit_log("Placement     : sharded across devices via accelerate")

            # Half precision quantises the per-neuron baseline sigma, which is the
            # exact statistic the detector divides by. Record it as INFO so the
            # operator sees it without capping an otherwise-clean score.
            if tracker.torch_dtype in (torch.float16, torch.bfloat16):
                audit.findings.append(
                    IntegrityFinding(
                        "INFO",
                        "HALF_PRECISION_SCAN",
                        f"Scanned in {str(tracker.torch_dtype).replace('torch.', '')}. "
                        "Baseline sigma estimates are quantised at this precision, which "
                        "raises the false-positive rate. Re-scan in float32 to confirm any "
                        "finding before acting on it.",
                    )
                )
            if self._aborted():
                return

            self._emit_progress(18, "Instrumenting MLP / FFN layers…")
            hook_points = tracker.attach_hooks()
            layer_names = [point.name for point in hook_points]
            self._layer_channels = [0] * len(layer_names)
            roles = {point.role for point in hook_points}
            self._emit_log(
                f"Instrumented  : {len(hook_points)} layers "
                f"(capture={'/'.join(sorted(roles))})"
            )
            self._emit_log(f"  first: {layer_names[0]}")
            self._emit_log(f"  last : {layer_names[-1]}")

            fuzzer = AdversarialFuzzer(
                seed=config.seed,
                tokenizer=tracker.tokenizer,
                extra_triggers=config.extra_triggers,
            )
            baseline_store = ActivationStore()
            fuzz_store = ActivationStore()

            # -- 3. baseline -----------------------------------------------
            baseline_cases = fuzzer.generate_baseline(config.baseline_passes)
            self._emit_progress(22, f"Recording baseline ({len(baseline_cases)} prompts)…")
            self._emit_log(f"Baseline      : {len(baseline_cases)} natural-language prompts")
            if not self._run_phase(
                tracker,
                baseline_cases,
                baseline_store,
                fuzz_store,
                layer_names,
                phase="BASELINE",
                progress_start=22,
                progress_end=48,
                baseline_first=True,
            ):
                self._emit_log("Scan aborted during baseline recording.")
                return

            if baseline_store.is_empty():
                self.error.emit(
                    "No activations were captured during the baseline phase. The "
                    "hooked modules never fired -- this architecture may not be "
                    "supported."
                )
                return

            total_neurons = baseline_store.total_channels()
            self._emit_log(f"Coverage      : {total_neurons:,} neurons under observation")

            # -- 4. adversarial fuzzing ------------------------------------
            fuzz_cases = fuzzer.generate_batch(total=config.fuzz_passes)
            histogram = AdversarialFuzzer.describe(fuzz_cases)
            self._emit_progress(50, f"Adversarial fuzzing ({len(fuzz_cases)} prompts)…")
            self._emit_log(
                "Fuzz corpus   : "
                + ", ".join(
                    f"{count} {category.lower()}"
                    for category, count in histogram.items()
                    if count
                )
            )
            if not self._run_phase(
                tracker,
                fuzz_cases,
                fuzz_store,
                baseline_store,
                layer_names,
                phase="FUZZING",
                progress_start=50,
                progress_end=92,
                baseline_first=False,
            ):
                self._emit_log("Scan aborted during adversarial fuzzing.")
                return

            if fuzz_store.is_empty():
                self.error.emit("No activations were captured during the fuzzing phase.")
                return

            # -- 5. analysis -----------------------------------------------
            self._emit_progress(94, "Computing Z-score anomaly matrix…")
            detector = BackdoorDetector(
                z_threshold=config.z_threshold,
                min_abs_delta_ratio=config.min_abs_delta_ratio,
                spatial_mad_threshold=config.spatial_mad_threshold,
            )
            elapsed = time.time() - self._started_at
            report = detector.analyze_stats(
                baseline_store.stats(),
                fuzz_store.stats(),
                fuzz_cases=fuzz_cases,
                integrity_findings=audit.findings,
                metadata={
                    "model_path": config.model_path,
                    "architectures": audit.architectures,
                    "model_type": audit.model_type,
                    "device": tracker.device,
                    "dtype": str(tracker.torch_dtype).replace("torch.", ""),
                    "sharded": tracker.is_sharded,
                    "gpu_memory": tracker.gpu_memory_stats(),
                    "nonfinite_activations": tracker.nonfinite_activations,
                    "hook_points": [p.to_dict() for p in hook_points],
                    "config": config.to_dict(),
                    "fuzz_histogram": histogram,
                    "duration_seconds": round(elapsed, 2),
                },
            )
            self.report = report

            # Final frame carries the Z-map so the canvas can render the delta view.
            self._emit_progress(98, "Rendering anomaly matrix…")
            zmap = build_heatmap_matrix(
                {k: np.asarray(v) for k, v in report.z_by_layer.items()},
                layer_names,
                config.heatmap_bins,
            )
            self._emit_frame(
                baseline_store,
                fuzz_store,
                layer_names,
                phase="COMPLETE",
                detail=f"{report.verdict} · score {report.safety_score:.1f}",
                completed=len(fuzz_cases),
                total=len(fuzz_cases),
                zmap=zmap,
            )

            self._emit_log(f"Analysis      : completed in {elapsed:.1f}s")
            gpu = tracker.gpu_summary()
            if gpu:
                self._emit_log(f"Peak device   : {gpu}")
            if tracker.nonfinite_activations:
                self._emit_log(
                    f"  ! {tracker.nonfinite_activations} non-finite activation value(s) were "
                    "clamped — a sign of half-precision overflow. Re-scan in float32."
                )
            self._emit_progress(100, f"Scan complete — {report.verdict}")
            self.finished_report.emit(report)

        except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
            LOGGER.exception("Scan failed")
            self.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=8)}"
            )
        finally:
            # -- 6. teardown, unconditionally ------------------------------
            if tracker is not None:
                try:
                    tracker.unmount()
                    self._emit_log("Sandbox       : hooks removed, model unmounted, caches flushed.")
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Teardown failed")
            self._tracker = None

    # -- cooperative cancellation -----------------------------------------

    def abort(self) -> None:
        """Ask the scan to stop at the next prompt boundary."""
        self.requestInterruption()
