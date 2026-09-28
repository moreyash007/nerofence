"""
NeuroFence :: app
=================

PyQt6 forensic desktop interface.

The GUI owns no PyTorch state whatsoever.  It launches a
:class:`~worker.ScanWorkerThread`, receives immutable
:class:`~worker.HeatmapFrame` snapshots and a final
:class:`~detector.ScanReport` through queued signals, and repaints.  Because the
inference loop lives on another thread, the window stays responsive -- and
abortable -- for the entire duration of a scan.

Run with::

    python app.py [optional/path/to/model/dir]
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QRect, QSize, Qt, pyqtSlot
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from detector import ScanReport
from worker import (
    DEFAULT_HEATMAP_BINS,
    HeatmapFrame,
    ScanConfig,
    ScanWorkerThread,
    build_report_heatmap,
)

LOGGER = logging.getLogger("neurofence.app")

APP_NAME = "NeuroFence"
APP_TAGLINE = "Offline LLM Weight-Poisoning & Backdoor Scanner"


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG_DEEP = QColor(13, 16, 21)
BG_PANEL = QColor(20, 25, 32)
BG_GRID = QColor(32, 40, 50)
FG_DIM = QColor(126, 143, 160)
FG_TEXT = QColor(196, 212, 226)
FG_ACCENT = QColor(60, 210, 190)

#: Green -> yellow -> red thermal ramp used for both activation energy and Z-scores.
_COLOR_STOPS: Sequence[Tuple[float, Tuple[int, int, int]]] = (
    (0.00, (9, 22, 18)),
    (0.12, (0, 78, 52)),
    (0.30, (0, 158, 84)),
    (0.48, (118, 200, 60)),
    (0.64, (222, 214, 44)),
    (0.78, (247, 166, 28)),
    (0.90, (250, 96, 26)),
    (1.00, (255, 44, 44)),
)


def _build_lut(size: int = 256) -> np.ndarray:
    """Pre-compute the colormap as an ``(size, 3)`` uint8 table."""
    positions = np.array([stop for stop, _ in _COLOR_STOPS], dtype=np.float64)
    colors = np.array([rgb for _, rgb in _COLOR_STOPS], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, size)
    lut = np.empty((size, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(xs, positions, colors[:, channel]).astype(np.uint8)
    return lut


_LUT = _build_lut()


def _gradient_for(rect: QRect) -> QLinearGradient:
    gradient = QLinearGradient(rect.left(), rect.bottom(), rect.left(), rect.top())
    for stop, (r, g, b) in _COLOR_STOPS:
        gradient.setColorAt(stop, QColor(r, g, b))
    return gradient


def _mono_font(size: int = 9, bold: bool = False) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    for candidate in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo"):
        probe = QFont(candidate)
        if probe.exactMatch():
            font = probe
            break
    font.setPointSize(size)
    font.setBold(bold)
    return font


# ---------------------------------------------------------------------------
# Heatmap canvas
# ---------------------------------------------------------------------------


class HeatmapCanvas(QWidget):
    """High-density activation matrix renderer (layers x pooled neuron channels).

    Each row is one hooked MLP layer; each column is a max-pooled band of neuron
    channels.  Rows are normalised independently so that a quiet early layer and
    a loud late layer are both legible, and the baseline / adversarial panels
    share a row scale so the two are directly comparable by eye.

    The matrix is rasterised into a ``QImage`` at native cell resolution and
    blitted by ``QPainter``; axes, gridlines, legend and the hover readout are
    drawn as vector primitives on top.  Repaint cost is therefore independent of
    ``intermediate_size`` -- a 32 x 11008 model costs the same as 6 x 1024.
    """

    VIEW_SPLIT = "split"
    VIEW_BASELINE = "baseline"
    VIEW_FUZZ = "fuzz"
    VIEW_DELTA = "delta"

    VIEW_LABELS = {
        VIEW_SPLIT: "Baseline │ Adversarial",
        VIEW_BASELINE: "Baseline energy only",
        VIEW_FUZZ: "Adversarial energy only",
        VIEW_DELTA: "Z-score anomaly map",
    }

    MARGIN_LEFT = 96
    MARGIN_RIGHT = 84
    MARGIN_TOP = 30
    MARGIN_BOTTOM = 48
    GUTTER = 20

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

        self._frame: Optional[HeatmapFrame] = None
        self._view_mode = self.VIEW_SPLIT
        self._z_threshold = 4.5
        self._hover: Optional[Tuple[str, int, int]] = None
        self._panels: List[Tuple[str, QRect]] = []
        self._image_cache: List[Tuple[QImage, bytes]] = []

        self._font = _mono_font(8)
        self._font_bold = _mono_font(8, bold=True)

    # -- public API --------------------------------------------------------

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(960, 420)

    def set_frame(self, frame: Optional[HeatmapFrame]) -> None:
        self._frame = frame
        self.update()

    def set_view_mode(self, mode: str) -> None:
        if mode in self.VIEW_LABELS:
            self._view_mode = mode
            self.update()

    def view_mode(self) -> str:
        return self._view_mode

    def set_threshold(self, threshold: float) -> None:
        self._z_threshold = float(threshold)
        self.update()

    def clear(self) -> None:
        self._frame = None
        self._hover = None
        self.update()

    # -- normalisation -----------------------------------------------------

    @staticmethod
    def _compress(normalised: np.ndarray, knee: float = 60.0) -> np.ndarray:
        """Log-compress [0,1] data so low-energy texture stays visible."""
        return np.log1p(np.clip(normalised, 0.0, 1.0) * knee) / np.log1p(knee)

    def _row_scales(self, frame: HeatmapFrame) -> np.ndarray:
        """Shared per-layer scale so baseline and adversarial panels are comparable."""
        base = frame.baseline
        fuzz = frame.fuzz if frame.fuzz.shape == base.shape else np.zeros_like(base)
        scales = np.maximum(base.max(axis=1), fuzz.max(axis=1))
        return np.where(scales > 0, scales, 1.0)

    def _display_matrix(self, kind: str) -> Optional[np.ndarray]:
        frame = self._frame
        if frame is None or frame.baseline.size == 0:
            return None

        if kind == "delta":
            zmap = frame.zmap
            if zmap is None or zmap.size == 0:
                return None
            ceiling = max(float(np.max(zmap)), self._z_threshold * 4.0, 1.0)
            return self._compress(np.asarray(zmap, dtype=np.float64) / ceiling, knee=25.0)

        matrix = frame.baseline if kind == "baseline" else frame.fuzz
        if matrix is None or matrix.size == 0:
            return None
        scales = self._row_scales(frame)[:, None]
        return self._compress(np.asarray(matrix, dtype=np.float64) / scales)

    def _to_image(self, matrix: np.ndarray) -> QImage:
        """Rasterise a normalised matrix through the LUT into an ARGB QImage."""
        indices = np.clip((matrix * (_LUT.shape[0] - 1)).astype(np.int32), 0, _LUT.shape[0] - 1)
        rgb = _LUT[indices]                                  # (h, w, 3) uint8
        height, width = indices.shape
        argb = (
            (0xFF << 24)
            | (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32)
        ).astype(np.uint32)
        payload = np.ascontiguousarray(argb).tobytes()
        image = QImage(payload, width, height, width * 4, QImage.Format.Format_RGB32)
        # QImage does not own ``payload``; keep both alive for the widget's lifetime.
        self._image_cache.append((image, payload))
        return image

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BG_DEEP)
        painter.setFont(self._font)

        frame = self._frame
        if frame is None or frame.baseline.size == 0:
            self._paint_placeholder(painter)
            painter.end()
            return

        self._image_cache.clear()

        if self._view_mode == self.VIEW_SPLIT:
            panels = [("baseline", "BASELINE ENERGY"), ("fuzz", "ADVERSARIAL PEAK")]
        elif self._view_mode == self.VIEW_DELTA:
            panels = [("delta", f"Z-SCORE ANOMALY MAP  (flag ≥ {self._z_threshold:g}σ)")]
        elif self._view_mode == self.VIEW_BASELINE:
            panels = [("baseline", "BASELINE ENERGY")]
        else:
            panels = [("fuzz", "ADVERSARIAL PEAK")]

        plot_left = self.MARGIN_LEFT
        plot_top = self.MARGIN_TOP
        plot_width = max(40, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)
        plot_height = max(40, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)

        count = len(panels)
        panel_width = int((plot_width - self.GUTTER * (count - 1)) / count)

        self._panels = []
        for position, (kind, title) in enumerate(panels):
            rect = QRect(
                plot_left + position * (panel_width + self.GUTTER),
                plot_top,
                panel_width,
                plot_height,
            )
            self._panels.append((kind, rect))
            self._paint_panel(painter, kind, title, rect, frame)

        self._paint_layer_axis(painter, frame, QRect(0, plot_top, self.MARGIN_LEFT - 8, plot_height))
        self._paint_legend(
            painter,
            QRect(self.width() - self.MARGIN_RIGHT + 16, plot_top + 6, 14, max(60, plot_height - 34)),
        )
        self._paint_footer(painter, frame)
        painter.end()

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(FG_DIM))
        painter.setFont(self._font_bold)
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            "NO ACTIVATION DATA\n\nSelect a local model directory and start a scan.",
        )

    def _paint_panel(
        self, painter: QPainter, kind: str, title: str, rect: QRect, frame: HeatmapFrame
    ) -> None:
        painter.setPen(QPen(BG_GRID))
        painter.setBrush(BG_PANEL)
        painter.drawRect(rect.adjusted(-1, -1, 0, 0))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.setFont(self._font_bold)
        painter.setPen(QPen(FG_ACCENT))
        painter.drawText(
            QRect(rect.left(), rect.top() - 22, rect.width(), 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            title,
        )
        painter.setFont(self._font)

        matrix = self._display_matrix(kind)
        if matrix is None or matrix.size == 0:
            painter.setPen(QPen(FG_DIM))
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignCenter),
                "awaiting data…" if kind != "delta" else "Z-map available after analysis",
            )
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, self._to_image(matrix))

        # Row separators, but only while they stay readable.
        rows = matrix.shape[0]
        if rows > 1 and rect.height() / rows >= 6:
            painter.setPen(QPen(QColor(0, 0, 0, 90), 1))
            for row in range(1, rows):
                y = rect.top() + int(rect.height() * row / rows)
                painter.drawLine(rect.left(), y, rect.right(), y)

        # Highlight the hovered cell.
        if self._hover and self._hover[0] == kind:
            _, row, column = self._hover
            columns = matrix.shape[1]
            cell = QRect(
                rect.left() + int(rect.width() * column / columns),
                rect.top() + int(rect.height() * row / rows),
                max(2, int(rect.width() / columns)),
                max(2, int(rect.height() / rows)),
            )
            painter.setPen(QPen(QColor(255, 255, 255, 210), 1))
            painter.drawRect(cell)

        # Channel axis ticks.
        painter.setPen(QPen(FG_DIM))
        columns = matrix.shape[1]
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.left() + int(rect.width() * fraction)
            painter.drawLine(x, rect.bottom(), x, rect.bottom() + 4)
            painter.drawText(
                QRect(x - 26, rect.bottom() + 5, 52, 12),
                int(Qt.AlignmentFlag.AlignCenter),
                f"{int(fraction * columns)}",
            )
        painter.drawText(
            QRect(rect.left(), rect.bottom() + 18, rect.width(), 12),
            int(Qt.AlignmentFlag.AlignCenter),
            "pooled neuron channel bands →",
        )

    def _paint_layer_axis(self, painter: QPainter, frame: HeatmapFrame, rect: QRect) -> None:
        rows = len(frame.layer_names)
        if rows == 0:
            return
        painter.setPen(QPen(FG_DIM))
        row_height = rect.height() / rows
        step = max(1, int(round(12 / max(row_height, 1e-6))))
        for row in range(0, rows, step):
            y = rect.top() + int(row_height * row)
            painter.drawText(
                QRect(rect.left() + 4, y, rect.width() - 6, max(11, int(row_height))),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                self._short_layer_name(frame.layer_names[row]),
            )
        painter.setPen(QPen(FG_ACCENT))
        painter.setFont(self._font_bold)
        painter.drawText(
            QRect(0, rect.top() - 22, rect.width(), 18),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            "LAYER",
        )
        painter.setFont(self._font)

    @staticmethod
    def _short_layer_name(name: str) -> str:
        """``model.layers.17.mlp.down_proj`` -> ``L17·down_proj``."""
        parts = name.split(".")
        index = next((p for p in parts if p.isdigit()), None)
        tail = parts[-1] if parts else name
        return f"L{index}·{tail}" if index is not None else tail[-16:]

    def _paint_legend(self, painter: QPainter, rect: QRect) -> None:
        painter.setPen(QPen(BG_GRID))
        painter.setBrush(_gradient_for(rect))
        painter.drawRect(rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(FG_DIM))
        labels = ("HIGH", "", "", "", "LOW") if self._view_mode != self.VIEW_DELTA else (
            "MAX Z", "", "", "", "0σ"
        )
        for position, label in enumerate(labels):
            if not label:
                continue
            y = rect.top() + int(rect.height() * position / (len(labels) - 1))
            painter.drawText(
                QRect(rect.right() + 4, y - 6, self.MARGIN_RIGHT - 24, 12),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )

    def _paint_footer(self, painter: QPainter, frame: HeatmapFrame) -> None:
        y = self.height() - 18
        painter.setPen(QPen(FG_TEXT))
        painter.setFont(self._font_bold)
        phase = f"[{frame.phase}]"
        painter.drawText(QRect(8, y, 130, 14), int(Qt.AlignmentFlag.AlignLeft), phase)
        painter.setFont(self._font)
        painter.setPen(QPen(FG_DIM))

        if self._hover is not None:
            text = self._hover_readout(frame)
        else:
            progress = f"{frame.completed}/{frame.total}" if frame.total else ""
            text = f"{progress}  {frame.detail}".strip()
        painter.drawText(
            QRect(142, y, max(10, self.width() - 150), 14), int(Qt.AlignmentFlag.AlignLeft), text
        )

    def _hover_readout(self, frame: HeatmapFrame) -> str:
        kind, row, column = self._hover
        if row >= len(frame.layer_names):
            return ""
        layer = frame.layer_names[row]
        bins = max(frame.baseline.shape[1], 1)
        channels = frame.channels[row] if row < len(frame.channels) else 0
        if channels:
            lo = int(channels * column / bins)
            hi = int(channels * (column + 1) / bins) - 1
            span = f"ch {lo}–{max(lo, hi)}"
        else:
            span = f"bin {column}"

        pieces = [layer, span]
        if frame.baseline.size:
            pieces.append(f"baseline {frame.baseline[row, column]:.4g}")
        if frame.fuzz.size:
            pieces.append(f"adversarial {frame.fuzz[row, column]:.4g}")
        if frame.zmap is not None and frame.zmap.size:
            pieces.append(f"Z {frame.zmap[row, column]:,.1f}")
        return "  ·  ".join(pieces)

    # -- interaction -------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt naming
        frame = self._frame
        self._hover = None
        if frame is not None and frame.baseline.size:
            point = event.position().toPoint()
            rows = max(frame.baseline.shape[0], 1)
            columns = max(frame.baseline.shape[1], 1)
            for kind, rect in self._panels:
                if rect.contains(point):
                    row = min(rows - 1, max(0, int((point.y() - rect.top()) * rows / rect.height())))
                    column = min(
                        columns - 1,
                        max(0, int((point.x() - rect.left()) * columns / rect.width())),
                    )
                    self._hover = (kind, row, column)
                    break
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt naming
        self._hover = None
        self.update()
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

STYLESHEET = """
QMainWindow, QWidget { background-color: #0d1015; color: #c4d4e2; }
QFrame#Header, QFrame#Controls { background-color: #141922; border: 1px solid #202832; }
QLabel { color: #c4d4e2; }
QLabel#Title { color: #3cd2be; font-size: 17px; font-weight: 700; letter-spacing: 1px; }
QLabel#Tagline { color: #7e8fa0; font-size: 11px; }
QLabel#Verdict { font-size: 15px; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0d1218; border: 1px solid #23303c; border-radius: 3px;
    padding: 4px 6px; color: #c4d4e2; selection-background-color: #1d6f66;
}
QLineEdit:disabled, QComboBox:disabled { color: #56636f; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background-color: #141922; color: #c4d4e2; selection-background-color: #1d6f66;
    border: 1px solid #23303c;
}
QPushButton {
    background-color: #1b2530; border: 1px solid #2c3d4c; border-radius: 3px;
    padding: 6px 14px; color: #c4d4e2; font-weight: 600;
}
QPushButton:hover { background-color: #24323f; border-color: #3cd2be; }
QPushButton:disabled { background-color: #151a20; color: #4a545e; border-color: #212a33; }
QPushButton#Primary { background-color: #16594f; border-color: #3cd2be; color: #d8fff8; }
QPushButton#Primary:hover { background-color: #1d7468; }
QPushButton#Primary:disabled { background-color: #16241f; color: #4a545e; border-color: #21332e; }
QPushButton#Danger { background-color: #4a1d1d; border-color: #8a3232; color: #ffd9d9; }
QPushButton#Danger:hover { background-color: #6a2727; }
QPushButton#Danger:disabled { background-color: #221515; color: #4a545e; border-color: #33201f; }
QProgressBar {
    background-color: #0d1218; border: 1px solid #23303c; border-radius: 3px;
    text-align: center; color: #c4d4e2; height: 18px;
}
QProgressBar::chunk { background-color: #1d7468; border-radius: 2px; }
QTextBrowser {
    background-color: #0a0e13; border: 1px solid #202832; color: #b9c9d6;
    selection-background-color: #1d6f66;
}
QSplitter::handle { background-color: #202832; height: 3px; }
QMenuBar { background-color: #141922; color: #c4d4e2; }
QMenuBar::item:selected { background-color: #1d6f66; }
QMenu { background-color: #141922; color: #c4d4e2; border: 1px solid #23303c; }
QMenu::item:selected { background-color: #1d6f66; }
QStatusBar { background-color: #141922; color: #7e8fa0; }
"""


class NeuroFenceMainWindow(QMainWindow):
    """The NeuroFence forensic console."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_TAGLINE}")
        self.resize(1480, 940)

        self.worker: Optional[ScanWorkerThread] = None
        self.report: Optional[ScanReport] = None
        self.model_path: str = ""
        self._last_frame: Optional[HeatmapFrame] = None

        self._build_ui()
        self._build_menu()
        self.setStyleSheet(STYLESHEET)
        self._print_banner()

        if initial_path:
            self._set_model_path(initial_path)

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_header())
        layout.addWidget(self._build_controls())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_heatmap_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([540, 380])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        layout.addLayout(self._build_status_row())
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Header")
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 10, 14, 10)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        tagline = QLabel(f"{APP_TAGLINE}  ·  air-gapped  ·  safetensors-only  ·  no network I/O")
        tagline.setObjectName("Tagline")
        titles.addWidget(title)
        titles.addWidget(tagline)
        row.addLayout(titles)
        row.addStretch(1)

        self.verdict_label = QLabel("AWAITING SCAN")
        self.verdict_label.setObjectName("Verdict")
        self.verdict_label.setStyleSheet("color: #7e8fa0;")
        self.verdict_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.verdict_label)
        return frame

    def _build_controls(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Controls")
        grid = QGridLayout(frame)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Model directory"), 0, 0)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(
            "Select a local folder containing config.json and *.safetensors …"
        )
        grid.addWidget(self.path_edit, 0, 1, 1, 7)

        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self.choose_model_directory)
        grid.addWidget(self.browse_button, 0, 8)

        from sandbox_tracker import ModelSandboxTracker  # local: keeps torch off the import path until needed

        grid.addWidget(QLabel("Device"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItem("auto — prefer GPU if present", "auto")
        for device, description in ModelSandboxTracker.describe_devices():
            self.device_combo.addItem(description, device)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        grid.addWidget(self.device_combo, 1, 1)

        grid.addWidget(QLabel("Precision"), 1, 2)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["auto", "float32", "float16", "bfloat16"])
        grid.addWidget(self.dtype_combo, 1, 3)

        grid.addWidget(QLabel("Z threshold"), 1, 4)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 50.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(4.5)
        self.threshold_spin.valueChanged.connect(self.canvas_threshold_changed)
        grid.addWidget(self.threshold_spin, 1, 5)

        grid.addWidget(QLabel("Baseline"), 1, 6)
        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(4, 1000)
        self.baseline_spin.setValue(24)
        grid.addWidget(self.baseline_spin, 1, 7)

        grid.addWidget(QLabel("Fuzz"), 2, 6)
        self.fuzz_spin = QSpinBox()
        self.fuzz_spin.setRange(4, 5000)
        self.fuzz_spin.setValue(96)
        grid.addWidget(self.fuzz_spin, 2, 7)

        grid.addWidget(QLabel("Max tokens"), 2, 0)
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(16, 4096)
        self.tokens_spin.setSingleStep(16)
        self.tokens_spin.setValue(256)
        grid.addWidget(self.tokens_spin, 2, 1)

        grid.addWidget(QLabel("Seed"), 2, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(1337)
        grid.addWidget(self.seed_spin, 2, 3)

        grid.addWidget(QLabel("Extra trigger"), 2, 4)
        self.trigger_edit = QLineEdit()
        self.trigger_edit.setPlaceholderText("optional, comma-separated")
        grid.addWidget(self.trigger_edit, 2, 5)

        self.shard_check = QCheckBox("Shard across GPUs")
        self.shard_check.setToolTip(
            "Split the model over every visible CUDA device using accelerate.\n"
            "Only needed when the model does not fit on one card."
        )
        self.shard_check.setEnabled(False)
        grid.addWidget(self.shard_check, 3, 0, 1, 2)

        self.device_note = QLabel("")
        self.device_note.setStyleSheet("color: #7e8fa0;")
        grid.addWidget(self.device_note, 3, 2, 1, 6)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("▶  START SCAN")
        self.start_button.setObjectName("Primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_scan)
        self.abort_button = QPushButton("■  ABORT")
        self.abort_button.setObjectName("Danger")
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.abort_scan)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.abort_button)
        grid.addLayout(buttons, 1, 8, 2, 1)

        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(5, 2)
        self._on_device_changed(self.device_combo.currentIndex())
        return frame

    @pyqtSlot(int)
    def _on_device_changed(self, index: int) -> None:
        """Enable sharding only for CUDA, and surface live VRAM for the selection."""
        from sandbox_tracker import ModelSandboxTracker, vram_bytes_for

        device = self.device_combo.itemData(index) or "auto"
        resolved = ModelSandboxTracker.resolve_device(device)
        is_cuda = resolved.startswith("cuda")

        self.shard_check.setEnabled(is_cuda)
        if not is_cuda:
            self.shard_check.setChecked(False)
            self.device_note.setText(
                "Running on CPU — precision defaults to float32."
                if resolved == "cpu"
                else f"Running on {resolved}."
            )
            return

        free, total = vram_bytes_for(resolved)
        if total:
            self.device_note.setText(
                f"{resolved}: {free / (1024 ** 3):.2f} GiB free of "
                f"{total / (1024 ** 3):.2f} GiB. 'auto' precision picks float32 when it "
                "fits, since half precision degrades the baseline σ estimate."
            )
        else:
            self.device_note.setText(f"{resolved}: VRAM could not be queried.")

    def _build_heatmap_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        caption = QLabel("ACTIVATION MATRIX  ·  layers × pooled neuron channels")
        caption.setStyleSheet("color: #3cd2be; font-weight: 600;")
        header.addWidget(caption)
        header.addStretch(1)
        header.addWidget(QLabel("View"))
        self.view_combo = QComboBox()
        for mode, label in HeatmapCanvas.VIEW_LABELS.items():
            self.view_combo.addItem(label, mode)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        header.addWidget(self.view_combo)
        layout.addLayout(header)

        self.canvas = HeatmapCanvas()
        layout.addWidget(self.canvas, 1)
        return container

    def _build_log_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        caption = QLabel("FORENSIC LOG")
        caption.setStyleSheet("color: #3cd2be; font-weight: 600;")
        header.addWidget(caption)
        header.addStretch(1)
        self.export_button = QPushButton("Export report (JSON)…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_report)
        clear_button = QPushButton("Clear log")
        clear_button.clicked.connect(self._clear_log)
        header.addWidget(self.export_button)
        header.addWidget(clear_button)
        layout.addLayout(header)

        self.log_view = QTextBrowser()
        self.log_view.setOpenExternalLinks(False)
        self.log_view.setOpenLinks(False)
        self.log_view.setFont(_mono_font(9))
        layout.addWidget(self.log_view, 1)
        return container

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        row.addWidget(self.progress_bar, 3)

        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet("color: #7e8fa0;")
        row.addWidget(self.status_label, 5)
        return row

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open model directory…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_model_directory)
        file_menu.addAction(open_action)

        self.export_action = QAction("&Export report…", self)
        self.export_action.setShortcut("Ctrl+S")
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_report)
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        scan_menu = menubar.addMenu("&Scan")
        self.start_action = QAction("&Start scan", self)
        self.start_action.setShortcut("Ctrl+R")
        self.start_action.setEnabled(False)
        self.start_action.triggered.connect(self.start_scan)
        scan_menu.addAction(self.start_action)

        self.abort_action = QAction("&Abort scan", self)
        self.abort_action.setShortcut("Ctrl+.")
        self.abort_action.setEnabled(False)
        self.abort_action.triggered.connect(self.abort_scan)
        scan_menu.addAction(self.abort_action)

        view_menu = menubar.addMenu("&View")
        for position, (mode, label) in enumerate(HeatmapCanvas.VIEW_LABELS.items()):
            action = QAction(label, self)
            action.setShortcut(f"Ctrl+{position + 1}")
            action.triggered.connect(lambda _checked=False, m=mode: self._select_view(m))
            view_menu.addAction(action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About NeuroFence", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # -- logging -----------------------------------------------------------

    def _append(self, html_fragment: str) -> None:
        self.log_view.append(html_fragment)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _log_line(self, text: str, color: str = "#b9c9d6") -> None:
        self._append(
            f'<div style="color:{color}; white-space:pre-wrap; font-family:monospace;">'
            f"{html.escape(text)}</div>"
        )

    def _clear_log(self) -> None:
        self.log_view.clear()
        self._print_banner()

    def _print_banner(self) -> None:
        self._append(
            '<div style="color:#3cd2be; font-family:monospace; white-space:pre;">'
            f"{html.escape('=' * 92)}\n"
            f"  {APP_NAME} — {APP_TAGLINE}\n"
            "  Air-gapped operation · safetensors-only loading · trust_remote_code disabled\n"
            f"{html.escape('=' * 92)}</div>"
        )

    # -- model selection ---------------------------------------------------

    @pyqtSlot()
    def choose_model_directory(self) -> None:
        start = self.model_path or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select local model directory", start, QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self._set_model_path(directory)

    def _set_model_path(self, directory: str) -> None:
        self.model_path = directory
        self.path_edit.setText(directory)

        path = Path(directory)
        safetensors = sorted(p.name for p in path.glob("*.safetensors")) if path.is_dir() else []
        pickles = (
            sorted(p.name for p in path.iterdir() if p.suffix.lower() in (".bin", ".pt", ".ckpt"))
            if path.is_dir()
            else []
        )

        self._log_line("")
        self._log_line(f"Selected: {directory}", "#3cd2be")
        if safetensors:
            self._log_line(f"  safetensors shards : {len(safetensors)} ({', '.join(safetensors[:4])}"
                           f"{'…' if len(safetensors) > 4 else ''})")
        else:
            self._log_line("  safetensors shards : NONE — this model cannot be scanned.", "#ff7a5c")
        if pickles:
            self._log_line(
                f"  pickle archives    : {len(pickles)} present and will NOT be opened "
                f"({', '.join(pickles[:4])})",
                "#f5c542",
            )

        enabled = bool(safetensors) and self.worker is None
        self.start_button.setEnabled(enabled)
        self.start_action.setEnabled(enabled)

    # -- scan lifecycle ----------------------------------------------------

    @pyqtSlot()
    def start_scan(self) -> None:
        if self.worker is not None:
            return
        if not self.model_path:
            QMessageBox.warning(self, APP_NAME, "Select a model directory first.")
            return

        extra = [t.strip() for t in self.trigger_edit.text().split(",") if t.strip()]
        config = ScanConfig(
            model_path=self.model_path,
            device=self.device_combo.currentData() or "auto",
            dtype=self.dtype_combo.currentText(),
            shard_across_devices=self.shard_check.isChecked(),
            z_threshold=self.threshold_spin.value(),
            baseline_passes=self.baseline_spin.value(),
            fuzz_passes=self.fuzz_spin.value(),
            max_sequence_length=self.tokens_spin.value(),
            seed=self.seed_spin.value(),
            heatmap_bins=DEFAULT_HEATMAP_BINS,
            extra_triggers=extra,
        )

        self.report = None
        self.canvas.clear()
        self.canvas.set_threshold(config.z_threshold)
        self.export_button.setEnabled(False)
        self.export_action.setEnabled(False)
        self.progress_bar.setValue(0)
        self._set_verdict("SCANNING…", "#3cd2be")

        self._log_line("")
        self._log_line("─" * 92, "#202832")
        self._log_line("SCAN STARTED", "#3cd2be")
        self._log_line(
            f"  z_threshold={config.z_threshold}  baseline={config.baseline_passes}  "
            f"fuzz={config.fuzz_passes}  max_tokens={config.max_sequence_length}  "
            f"seed={config.seed}"
        )
        self._log_line(
            f"  device={config.device}  precision={config.dtype}"
            f"{'  sharded=yes' if config.shard_across_devices else ''}"
        )

        self.worker = ScanWorkerThread(config)
        self.worker.progress.connect(self.on_progress)
        self.worker.results_ready.connect(self.on_results_ready)
        self.worker.finished_report.connect(self.on_finished_report)
        self.worker.log.connect(self.on_worker_log)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)

        self._set_controls_enabled(False)
        self.worker.start()

    @pyqtSlot()
    def abort_scan(self) -> None:
        if self.worker is None:
            return
        self._log_line("Abort requested — stopping at the next prompt boundary…", "#f5c542")
        self.status_label.setText("Aborting…")
        self.abort_button.setEnabled(False)
        self.abort_action.setEnabled(False)
        self.worker.abort()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.browse_button,
            self.device_combo,
            self.dtype_combo,
            self.threshold_spin,
            self.baseline_spin,
            self.fuzz_spin,
            self.tokens_spin,
            self.seed_spin,
            self.trigger_edit,
        ):
            widget.setEnabled(enabled)
        # Sharding is meaningful only on CUDA, so hand control back to the
        # device-dependent logic rather than blindly re-enabling it.
        if enabled:
            self._on_device_changed(self.device_combo.currentIndex())
        else:
            self.shard_check.setEnabled(False)
        can_start = enabled and bool(self.model_path)
        self.start_button.setEnabled(can_start)
        self.start_action.setEnabled(can_start)
        self.abort_button.setEnabled(not enabled)
        self.abort_action.setEnabled(not enabled)

    # -- worker signal handlers -------------------------------------------

    @pyqtSlot(int, str)
    def on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    @pyqtSlot(object)
    def on_results_ready(self, frame: HeatmapFrame) -> None:
        self._last_frame = frame
        self.canvas.set_frame(frame)
        if frame.zmap is not None and self.canvas.view_mode() != HeatmapCanvas.VIEW_DELTA:
            self._select_view(HeatmapCanvas.VIEW_DELTA)

    @pyqtSlot(str)
    def on_worker_log(self, message: str) -> None:
        self._log_line(message)

    @pyqtSlot(str)
    def on_error(self, message: str) -> None:
        self._append(
            '<div style="color:#ff7a5c; white-space:pre-wrap; font-family:monospace;">'
            f"<b>SCAN FAILED</b>\n{html.escape(message)}</div>"
        )
        self._set_verdict("SCAN FAILED", "#ff7a5c")
        self.status_label.setText("Scan failed.")
        QMessageBox.critical(self, f"{APP_NAME} — scan failed", message)

    @pyqtSlot(object)
    def on_finished_report(self, report: ScanReport) -> None:
        self.report = report
        self.export_button.setEnabled(True)
        self.export_action.setEnabled(True)
        self._set_verdict(
            f"{report.verdict}  ·  {report.safety_score:.1f}/100",
            self._verdict_color(report.safety_score),
        )
        self._append(self._render_report_html(report))

    @pyqtSlot()
    def on_worker_finished(self) -> None:
        self.worker = None
        self._set_controls_enabled(True)
        if self.report is None and self.progress_bar.value() < 100:
            self.status_label.setText("Scan ended without a report.")

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _verdict_color(score: float) -> str:
        if score >= 90:
            return "#37d67a"
        if score >= 70:
            return "#9fd356"
        if score >= 45:
            return "#f5c542"
        if score >= 20:
            return "#f58442"
        return "#ff4d4d"

    def _set_verdict(self, text: str, color: str) -> None:
        self.verdict_label.setText(text)
        self.verdict_label.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: 700;")

    def _render_report_html(self, report: ScanReport) -> str:
        color = self._verdict_color(report.safety_score)
        esc = html.escape

        rows = []
        for rank, anomaly in enumerate(report.top_anomalies(10), start=1):
            z_text = f"&gt;{anomaly.z_score:,.0f}" if anomaly.z_clipped else f"{anomaly.z_score:,.1f}"
            trigger = (
                f"[{esc(anomaly.trigger_category)}] {esc(anomaly.trigger_label)}"
                if anomaly.trigger_label
                else "—"
            )
            rows.append(
                "<tr>"
                f'<td style="padding:2px 8px; color:#7e8fa0;">{rank}</td>'
                f'<td style="padding:2px 8px; color:#c4d4e2;">{esc(anomaly.layer)}</td>'
                f'<td style="padding:2px 8px; color:#3cd2be; text-align:right;">{anomaly.neuron}</td>'
                f'<td style="padding:2px 8px; color:#ff7a5c; text-align:right;"><b>{z_text}</b></td>'
                f'<td style="padding:2px 8px; text-align:right;">{anomaly.baseline_mean:.4g}</td>'
                f'<td style="padding:2px 8px; text-align:right;">{anomaly.baseline_std:.4g}</td>'
                f'<td style="padding:2px 8px; text-align:right;">{anomaly.fuzz_peak:.4g}</td>'
                f'<td style="padding:2px 8px; text-align:right;">{anomaly.dormancy * 100:.0f}%</td>'
                f'<td style="padding:2px 8px; color:#f5c542;">{trigger}</td>'
                "</tr>"
            )
            if anomaly.trigger_prompt:
                rows.append(
                    '<tr><td></td><td colspan="8" style="padding:0 8px 4px 8px; color:#5f7183;">'
                    f"↳ {esc(anomaly.trigger_prompt)}</td></tr>"
                )

        table = (
            "<table style='font-family:monospace; font-size:11px; border-collapse:collapse;'>"
            "<tr style='color:#3cd2be;'>"
            "<th style='padding:2px 8px; text-align:left;'>#</th>"
            "<th style='padding:2px 8px; text-align:left;'>LAYER</th>"
            "<th style='padding:2px 8px; text-align:right;'>NEURON</th>"
            "<th style='padding:2px 8px; text-align:right;'>Z</th>"
            "<th style='padding:2px 8px; text-align:right;'>μ base</th>"
            "<th style='padding:2px 8px; text-align:right;'>σ base</th>"
            "<th style='padding:2px 8px; text-align:right;'>peak</th>"
            "<th style='padding:2px 8px; text-align:right;'>dorm</th>"
            "<th style='padding:2px 8px; text-align:left;'>WOKEN BY</th>"
            "</tr>" + "".join(rows) + "</table>"
            if rows
            else "<div style='color:#37d67a; font-family:monospace;'>"
            "No neuron exceeded the Z-score threshold.</div>"
        )

        flagged_layers = report.flagged_layer_names()
        flagged_block = (
            "<br/>".join(
                f"&nbsp;&nbsp;• {esc(finding.layer)} — "
                f"{finding.flagged} flagged / {finding.channels:,} neurons "
                f"(max Z {finding.max_z:,.1f})"
                for finding in report.layer_findings
                if finding.flagged
            )
            or "&nbsp;&nbsp;none"
        )

        integrity_block = (
            "<br/>".join(
                f"&nbsp;&nbsp;[{esc(f.severity)}] {esc(f.code)} — {esc(f.message)}"
                for f in report.integrity_findings
            )
            or "&nbsp;&nbsp;no static integrity findings"
        )

        breakdown = " · ".join(
            f"{esc(key)} −{value:.1f}" for key, value in sorted(report.score_breakdown.items())
        ) or "no deductions"

        duration = report.metadata.get("duration_seconds", "?")

        return (
            "<div style='font-family:monospace; font-size:12px; color:#b9c9d6;'>"
            f"<div style='color:#202832;'>{'─' * 92}</div>"
            "<div style='color:#3cd2be; font-weight:700; font-size:13px;'>FORENSIC ANALYSIS COMPLETE</div>"
            "<br/>"
            f"<div style='font-size:20px; color:{color}; font-weight:700;'>"
            f"SAFETY SCORE {report.safety_score:.1f} / 100 &nbsp;—&nbsp; {esc(report.verdict)}</div>"
            f"<div style='color:#7e8fa0;'>{esc(report.rationale)}</div><br/>"
            f"<div>Coverage&nbsp;&nbsp;&nbsp;: {report.total_layers} hooked layers · "
            f"{report.total_neurons:,} neurons · {report.baseline_passes} baseline + "
            f"{report.fuzz_passes} adversarial passes · {duration}s</div>"
            f"<div>Threshold&nbsp;&nbsp;: Z ≥ {report.z_threshold:g}σ &nbsp;|&nbsp; peak observed "
            f"Z = {report.max_z:,.1f}</div>"
            f"<div>Flagged&nbsp;&nbsp;&nbsp;&nbsp;: {report.total_flagged} neurons across "
            f"{report.flagged_layers} layer(s)</div>"
            f"<div>Score model: {breakdown}</div><br/>"
            f"<div style='color:#3cd2be;'>FLAGGED LAYERS ({len(flagged_layers)})</div>"
            f"<div>{flagged_block}</div><br/>"
            f"<div style='color:#3cd2be;'>STATIC INTEGRITY</div>"
            f"<div>{integrity_block}</div><br/>"
            f"<div style='color:#3cd2be;'>TOP {min(10, len(report.anomalies))} ANOMALOUS NEURONS</div>"
            f"{table}"
            f"<div style='color:#202832;'>{'─' * 92}</div>"
            "</div>"
        )

    # -- view / export -----------------------------------------------------

    def _select_view(self, mode: str) -> None:
        index = self.view_combo.findData(mode)
        if index >= 0:
            self.view_combo.setCurrentIndex(index)
        else:
            self.canvas.set_view_mode(mode)

    @pyqtSlot(int)
    def _on_view_changed(self, index: int) -> None:
        mode = self.view_combo.itemData(index)
        if mode:
            self.canvas.set_view_mode(mode)

    @pyqtSlot(float)
    def canvas_threshold_changed(self, value: float) -> None:
        self.canvas.set_threshold(value)

    @pyqtSlot()
    def export_report(self) -> None:
        if self.report is None:
            return
        default = str(Path(self.model_path or ".").with_suffix("")) + "_neurofence_report.json"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export forensic report", default, "JSON report (*.json)"
        )
        if not filename:
            return
        payload = self.report.to_dict()
        # Embed the pooled activation matrices so an offline viewer can redraw
        # the heatmap; ScanReport itself keeps them out of its dict on purpose.
        if self._last_frame is not None:
            payload["heatmap"] = build_report_heatmap(self._last_frame)
        try:
            Path(filename).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write report:\n{exc}")
            return
        size_kb = Path(filename).stat().st_size / 1024
        self._log_line(f"Report exported to {filename} ({size_kb:,.0f} KB)", "#37d67a")

    @pyqtSlot()
    def show_about(self) -> None:
        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> — {APP_TAGLINE}<br/><br/>"
            "Fires synthetic adversarial prompts into an isolated PyTorch sandbox while "
            "monitoring transformer MLP activations through forward hooks, then flags "
            "dormant neurons that spike under trigger inputs.<br/><br/>"
            "• 100% offline — <code>local_files_only=True</code>, no network I/O<br/>"
            "• <code>.safetensors</code> only — pickle checkpoints are never deserialised<br/>"
            "• <code>trust_remote_code=False</code> — bundled Python is reported, never run<br/>"
            "• Inference runs on a background QThread; the UI never blocks",
        )

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt naming
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self,
                APP_NAME,
                "A scan is still running. Abort it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.abort()
            if not self.worker.wait(15000):
                LOGGER.warning("Worker did not stop within 15s; terminating.")
                self.worker.terminate()
                self.worker.wait(2000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    argv = list(argv if argv is not None else sys.argv[1:])

    app = QApplication([sys.argv[0]] + argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    initial = argv[0] if argv and os.path.isdir(argv[0]) else None
    window = NeuroFenceMainWindow(initial_path=initial)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
