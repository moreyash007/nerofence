"""
NeuroFence :: sandbox_tracker
=============================

Model isolation, safe (pickle-free) loading and forward-hook instrumentation of
transformer MLP / FFN blocks.

The tracker is the only component in NeuroFence that is allowed to touch a
third-party model artifact.  It therefore enforces the project's hard security
rules at the boundary:

* ``local_files_only=True`` everywhere -- no HuggingFace Hub round-trips.
* ``use_safetensors=True`` -- pickle (``.bin`` / ``.pt`` / ``.ckpt``) archives are
  never deserialised, because ``torch.load`` on an untrusted archive is remote
  code execution by design.
* ``trust_remote_code=False`` -- a model repository shipping ``*.py`` modelling
  code is itself an execution primitive; we report it and refuse to run it.

Activation capture
------------------
The "neurons" of a transformer FFN are the post-activation channels that feed the
down-projection.  A ``register_forward_hook`` installed on the *contraction*
module (``down_proj`` / ``c_proj`` / ``dense_4h_to_h``) receives ``inputs[0]`` --
which is exactly that post-activation tensor -- so we hook the contraction and
capture its input.  Where no contraction module can be identified we fall back to
the output of the expansion projection, and finally to the output of the whole
MLP block.

Every captured tensor is reduced on-device and immediately severed from the
autograd graph via ``.detach() ... .cpu().numpy()`` so that no CUDA memory is
retained across a long fuzzing run.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Air-gap enforcement.  These must be set *before* transformers is imported,
# otherwise the hub client caches a networked session at import time.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LOGGER = logging.getLogger("neurofence.sandbox")

#: transformers 5.x renamed ``torch_dtype`` to ``dtype`` on ``from_pretrained``.
#: Resolve it once so the tracker runs unmodified on both major versions.
try:
    _TRANSFORMERS_MAJOR = int(str(transformers.__version__).split(".", 1)[0])
except (AttributeError, ValueError):  # pragma: no cover - exotic builds
    _TRANSFORMERS_MAJOR = 4
DTYPE_KWARG = "dtype" if _TRANSFORMERS_MAJOR >= 5 else "torch_dtype"

#: Out-of-memory exception types across torch versions. ``torch.cuda.OutOfMemoryError``
#: arrived in 1.13 and ``torch.OutOfMemoryError`` in 2.5; older builds raise a plain
#: ``RuntimeError`` whose message we match on instead.
_OOM_ERRORS: Tuple[type, ...] = tuple(
    {
        exc
        for exc in (
            getattr(torch.cuda, "OutOfMemoryError", None),
            getattr(torch, "OutOfMemoryError", None),
        )
        if isinstance(exc, type)
    }
) or (RuntimeError,)


def _is_oom(exc: BaseException) -> bool:
    """True when an exception represents device memory exhaustion."""
    if isinstance(exc, _OOM_ERRORS) and _OOM_ERRORS != (RuntimeError,):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def cuda_index_of(device: str) -> int:
    """Numeric CUDA ordinal for a device string, or ``-1`` when it is not CUDA."""
    if not device or not device.startswith("cuda") or not torch.cuda.is_available():
        return -1
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return torch.cuda.current_device()


def vram_bytes_for(device: str) -> Tuple[int, int]:
    """``(free, total)`` VRAM in bytes for a device string; ``(0, 0)`` off-CUDA.

    Free-standing so the GUI can probe a device the operator is merely hovering
    over, without constructing a tracker or touching the model.
    """
    index = cuda_index_of(device)
    if index < 0:
        return (0, 0)
    try:
        return torch.cuda.mem_get_info(index)
    except Exception:  # noqa: BLE001 - older drivers lack mem_get_info
        try:
            props = torch.cuda.get_device_properties(index)
            total = int(props.total_memory)
            return (max(0, total - torch.cuda.memory_reserved(index)), total)
        except Exception:  # noqa: BLE001
            return (0, 0)


def _itemsize_of(dtype: torch.dtype) -> int:
    """Bytes per element, portable across torch versions."""
    size = getattr(dtype, "itemsize", None)
    if isinstance(size, int):
        return size
    return torch.empty((), dtype=dtype).element_size()

__all__ = [
    "SAFE_WEIGHT_SUFFIXES",
    "PICKLE_WEIGHT_SUFFIXES",
    "CONTRACTION_PATTERNS",
    "EXPANSION_PATTERNS",
    "MLP_BLOCK_PATTERNS",
    "IntegrityFinding",
    "SandboxAudit",
    "HookPoint",
    "LayerStats",
    "ActivationStore",
    "ModelSandboxTracker",
    "layer_index_of",
    "discover_hook_points",
    "cuda_index_of",
    "vram_bytes_for",
]


# ---------------------------------------------------------------------------
# File-format policy
# ---------------------------------------------------------------------------

SAFE_WEIGHT_SUFFIXES: Tuple[str, ...] = (".safetensors",)

#: Archive formats that are deserialised through ``pickle`` and can therefore
#: execute arbitrary code on load.  NeuroFence never opens these.
PICKLE_WEIGHT_SUFFIXES: Tuple[str, ...] = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".model",
    ".pb",
)


# ---------------------------------------------------------------------------
# Module-name patterns for the FFN sub-modules of the common decoder families.
# ---------------------------------------------------------------------------

#: Down / contraction projections.  A forward hook here sees the post-activation
#: neuron vector as ``inputs[0]`` -- the canonical "FFN neuron" space.
CONTRACTION_PATTERNS: Tuple[str, ...] = (
    r"\.mlp\.down_proj$",          # LLaMA, Mistral, Qwen2/3, Gemma, Phi-3, Yi
    r"\.mlp\.c_proj$",             # GPT-2, GPT-J, CodeGen
    r"\.mlp\.dense_4h_to_h$",      # GPT-NeoX, Pythia, Falcon(-rw)
    r"\.mlp\.fc2$",                # Phi-1/2, OPT-style, CLIPText
    r"\.mlp\.w2$",                 # Baichuan, InternLM
    r"\.mlp\.wo$",                 # some MoE experts
    r"\.ffn\.down_proj$",          # MPT
    r"\.feed_forward\.w2$",        # original LLaMA reference impl
    r"\.feed_forward\.down_proj$",
)

#: Up / gate / expansion projections.  Used as the fallback capture point and as
#: the poisoning surface in ``mock_poison.py``.
EXPANSION_PATTERNS: Tuple[str, ...] = (
    r"\.mlp\.gate_proj$",          # LLaMA / Mistral / Qwen (SwiGLU gate)
    r"\.mlp\.up_proj$",            # LLaMA / Mistral / Qwen (SwiGLU value)
    r"\.mlp\.c_fc$",               # GPT-2, GPT-J
    r"\.mlp\.dense_h_to_4h$",      # GPT-NeoX, Pythia, Falcon
    r"\.mlp\.fc1$",                # Phi-1/2, OPT-style
    r"\.mlp\.gate_up_proj$",       # Phi-3 fused gate+up
    r"\.mlp\.w1$",                 # Baichuan, InternLM
    r"\.ffn\.up_proj$",            # MPT
    r"\.feed_forward\.w1$",
    r"\.feed_forward\.up_proj$",
)

#: Whole-block fallback (captures the MLP residual contribution itself).
MLP_BLOCK_PATTERNS: Tuple[str, ...] = (
    r"\.mlp$",
    r"\.ffn$",
    r"\.feed_forward$",
)

_LAYER_INDEX_RE = re.compile(r"\.(\d+)\.")


def layer_index_of(module_name: str) -> int:
    """Extract the decoder-block index encoded in a dotted module path.

    ``model.layers.17.mlp.down_proj`` -> ``17``.  Returns ``-1`` when the name
    carries no numeric block component.
    """
    match = _LAYER_INDEX_RE.search(module_name)
    return int(match.group(1)) if match else -1


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


# ---------------------------------------------------------------------------
# Static integrity audit
# ---------------------------------------------------------------------------


@dataclass
class IntegrityFinding:
    """A static (pre-execution) security observation about the model folder."""

    severity: str            # "INFO" | "MEDIUM" | "HIGH" | "CRITICAL"
    code: str
    message: str
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "evidence": list(self.evidence),
        }


@dataclass
class SandboxAudit:
    """Result of the static audit performed before any weight is touched."""

    model_path: str
    architectures: List[str] = field(default_factory=list)
    model_type: str = "unknown"
    safetensors_files: List[str] = field(default_factory=list)
    pickle_files: List[str] = field(default_factory=list)
    remote_code_files: List[str] = field(default_factory=list)
    total_safetensors_bytes: int = 0
    stored_dtype: str = ""
    findings: List[IntegrityFinding] = field(default_factory=list)

    @property
    def loadable(self) -> bool:
        return bool(self.safetensors_files)

    def max_severity(self) -> str:
        order = {"INFO": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        best = "INFO"
        for finding in self.findings:
            if order.get(finding.severity, 0) > order.get(best, 0):
                best = finding.severity
        return best

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "safetensors_files": list(self.safetensors_files),
            "pickle_files": list(self.pickle_files),
            "remote_code_files": list(self.remote_code_files),
            "total_safetensors_bytes": self.total_safetensors_bytes,
            "stored_dtype": self.stored_dtype,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Hook points
# ---------------------------------------------------------------------------


@dataclass
class HookPoint:
    """A single instrumented module."""

    name: str
    module_type: str
    layer_index: int
    capture: str               # "input" | "output"
    role: str                  # "contraction" | "expansion" | "block"
    channels: int = 0          # filled in after the first forward pass

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "module_type": self.module_type,
            "layer_index": self.layer_index,
            "capture": self.capture,
            "role": self.role,
            "channels": self.channels,
        }


def discover_hook_points(model: nn.Module) -> List[HookPoint]:
    """Locate one instrumentation point per decoder block.

    Preference order per block:

    1. contraction projection, capturing its *input*  (true FFN neuron space)
    2. expansion projection, capturing its *output*
    3. the MLP block itself, capturing its *output*
    """
    named = list(model.named_modules())

    contraction: Dict[int, Tuple[str, nn.Module]] = {}
    expansion: Dict[int, Tuple[str, nn.Module]] = {}
    blocks: Dict[int, Tuple[str, nn.Module]] = {}

    for name, module in named:
        if not name:
            continue
        index = layer_index_of(name)
        if _matches_any(name, CONTRACTION_PATTERNS):
            contraction.setdefault(index, (name, module))
        elif _matches_any(name, EXPANSION_PATTERNS):
            expansion.setdefault(index, (name, module))
        elif _matches_any(name, MLP_BLOCK_PATTERNS):
            blocks.setdefault(index, (name, module))

    indices = sorted(set(contraction) | set(expansion) | set(blocks))
    points: List[HookPoint] = []
    for index in indices:
        if index in contraction:
            name, module = contraction[index]
            capture, role = "input", "contraction"
        elif index in expansion:
            name, module = expansion[index]
            capture, role = "output", "expansion"
        else:
            name, module = blocks[index]
            capture, role = "output", "block"
        points.append(
            HookPoint(
                name=name,
                module_type=type(module).__name__,
                layer_index=index,
                capture=capture,
                role=role,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Per-neuron summary statistics for one hooked layer.

    ``mean``/``std`` describe the distribution of *per-pass peak activation
    energy* for each neuron; ``peak`` is the largest value ever observed and
    ``peak_case`` records which forward pass produced it.
    """

    name: str
    layer_index: int
    count: int
    mean: np.ndarray
    std: np.ndarray
    peak: np.ndarray
    peak_case: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.mean.shape[0])


class ActivationStore:
    """Streaming accumulator for activation energy across forward passes.

    Uses Welford's online algorithm so that memory is ``O(layers x channels)``
    regardless of how many prompts are fired -- a 70B-class model fuzzed with ten
    thousand prompts costs the same as one fired with ten.

    Set ``retain_raw=True`` to additionally keep every per-pass vector (useful for
    offline forensics and for the array-based
    :meth:`~detector.BackdoorDetector.analyze_activations` entry point), at the
    cost of ``O(passes x layers x channels)`` memory.
    """

    def __init__(self, retain_raw: bool = False) -> None:
        self.retain_raw = retain_raw
        self._count: Dict[str, int] = {}
        self._mean: Dict[str, np.ndarray] = {}
        self._m2: Dict[str, np.ndarray] = {}
        self._peak: Dict[str, np.ndarray] = {}
        self._peak_case: Dict[str, np.ndarray] = {}
        self._layer_index: Dict[str, int] = {}
        self._raw: Dict[str, List[np.ndarray]] = {}

    # -- mutation ----------------------------------------------------------

    def add_frame(
        self,
        frame: Dict[str, np.ndarray],
        case_index: int,
        layer_indices: Optional[Dict[str, int]] = None,
    ) -> None:
        """Fold one forward pass (``{layer_name: peak_vector}``) into the store."""
        for name, vector in frame.items():
            vec = np.asarray(vector, dtype=np.float64).reshape(-1)
            if name not in self._count:
                self._count[name] = 0
                self._mean[name] = np.zeros_like(vec)
                self._m2[name] = np.zeros_like(vec)
                self._peak[name] = np.full_like(vec, -np.inf)
                self._peak_case[name] = np.full(vec.shape, -1, dtype=np.int64)
                self._layer_index[name] = (
                    layer_indices.get(name, layer_index_of(name))
                    if layer_indices
                    else layer_index_of(name)
                )
                if self.retain_raw:
                    self._raw[name] = []

            if vec.shape != self._mean[name].shape:
                LOGGER.warning(
                    "Channel count changed for %s (%s -> %s); frame dropped.",
                    name,
                    self._mean[name].shape,
                    vec.shape,
                )
                continue

            # Welford update.
            self._count[name] += 1
            delta = vec - self._mean[name]
            self._mean[name] += delta / self._count[name]
            self._m2[name] += delta * (vec - self._mean[name])

            improved = vec > self._peak[name]
            self._peak[name] = np.where(improved, vec, self._peak[name])
            self._peak_case[name] = np.where(improved, case_index, self._peak_case[name])

            if self.retain_raw:
                self._raw[name].append(vec.astype(np.float32))

    def reset(self) -> None:
        self._count.clear()
        self._mean.clear()
        self._m2.clear()
        self._peak.clear()
        self._peak_case.clear()
        self._layer_index.clear()
        self._raw.clear()

    # -- access ------------------------------------------------------------

    @property
    def layer_names(self) -> List[str]:
        return sorted(
            self._mean.keys(),
            key=lambda n: (self._layer_index.get(n, -1), n),
        )

    def is_empty(self) -> bool:
        return not self._mean

    def stats(self) -> Dict[str, LayerStats]:
        """Materialise per-layer statistics (sample standard deviation)."""
        out: Dict[str, LayerStats] = {}
        for name in self.layer_names:
            count = self._count[name]
            if count > 1:
                variance = self._m2[name] / (count - 1)
            else:
                variance = np.zeros_like(self._m2[name])
            peak = self._peak[name].copy()
            peak[~np.isfinite(peak)] = 0.0
            out[name] = LayerStats(
                name=name,
                layer_index=self._layer_index.get(name, -1),
                count=count,
                mean=self._mean[name].copy(),
                std=np.sqrt(np.maximum(variance, 0.0)),
                peak=peak,
                peak_case=self._peak_case[name].copy(),
            )
        return out

    def peak_vectors(self) -> Dict[str, np.ndarray]:
        """``{layer_name: running peak energy}`` -- the live heatmap payload."""
        out: Dict[str, np.ndarray] = {}
        for name in self.layer_names:
            vec = self._peak[name].copy()
            vec[~np.isfinite(vec)] = 0.0
            out[name] = vec
        return out

    def mean_vectors(self) -> Dict[str, np.ndarray]:
        return {name: self._mean[name].copy() for name in self.layer_names}

    def raw(self) -> Dict[str, np.ndarray]:
        """Stacked ``(passes, channels)`` arrays; empty unless ``retain_raw``."""
        return {
            name: np.stack(vectors, axis=0)
            for name, vectors in self._raw.items()
            if vectors
        }

    def total_channels(self) -> int:
        return int(sum(v.shape[0] for v in self._mean.values()))


# ---------------------------------------------------------------------------
# The sandbox tracker
# ---------------------------------------------------------------------------


class ModelSandboxTracker:
    """Load a local ``.safetensors`` model and record FFN activation energy.

    Typical use::

        with ModelSandboxTracker("/models/qwen-0.5b", device="cuda") as tracker:
            audit = tracker.audit()
            tracker.load()
            tracker.attach_hooks()
            frame = tracker.run_forward("hello world")

    The instance is *not* thread-safe: drive it from a single worker thread.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str = "auto",
        dtype: str = "auto",
        max_sequence_length: int = 256,
        refuse_on_pickle: bool = False,
        shard_across_devices: bool = False,
        vram_headroom_ratio: float = 0.90,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.requested_dtype = dtype
        # "auto" is only finalised in load(), once the audit tells us how large
        # the checkpoint is and how much VRAM is actually free.
        self.torch_dtype = self._explicit_dtype(dtype) or torch.float32
        self.max_sequence_length = int(max_sequence_length)
        self.refuse_on_pickle = bool(refuse_on_pickle)
        self.shard_across_devices = bool(shard_across_devices)
        self.vram_headroom_ratio = float(vram_headroom_ratio)
        self.log = logger or LOGGER

        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.config = None
        self.hook_points: List[HookPoint] = []
        self.is_sharded = False
        self.nonfinite_activations = 0

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._frame: Dict[str, np.ndarray] = {}
        self._audit: Optional[SandboxAudit] = None

    # -- device / dtype ----------------------------------------------------

    @staticmethod
    def available_devices() -> List[str]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append("mps")
        return devices

    @staticmethod
    def describe_devices() -> List[Tuple[str, str]]:
        """``[(device_string, human_label)]`` for the device selector.

        Labels carry the adapter name and total VRAM so an operator can tell a
        24 GiB workstation card from a 8 GiB laptop one without leaving the app.
        """
        described: List[Tuple[str, str]] = [("cpu", "cpu — host memory")]
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            described.append(("cuda", f"cuda — default device ({count} visible)"))
            for index in range(count):
                try:
                    props = torch.cuda.get_device_properties(index)
                    gib = props.total_memory / (1024 ** 3)
                    described.append(
                        (
                            f"cuda:{index}",
                            f"cuda:{index} — {props.name} ({gib:.1f} GiB, sm_{props.major}{props.minor})",
                        )
                    )
                except Exception:  # noqa: BLE001 - driver can refuse enumeration
                    described.append((f"cuda:{index}", f"cuda:{index}"))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            described.append(("mps", "mps — Apple unified memory"))
        return described

    @staticmethod
    def cuda_index(device: str) -> int:
        """Numeric CUDA ordinal for a device string, or ``-1`` when not CUDA."""
        return cuda_index_of(device)

    @staticmethod
    def resolve_device(device: str) -> str:
        """Public form of the device resolver, for UI probing before a scan."""
        return ModelSandboxTracker._resolve_device(device)

    def vram_bytes(self) -> Tuple[int, int]:
        """``(free, total)`` VRAM in bytes for this tracker's device."""
        return vram_bytes_for(self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        device = (device or "auto").strip().lower()
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        if device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                LOGGER.warning("MPS requested but unavailable; falling back to CPU.")
                return "cpu"
        return device

    #: Accepted explicit precision names.
    _DTYPE_TABLE = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    @classmethod
    def _explicit_dtype(cls, dtype: str) -> Optional[torch.dtype]:
        return cls._DTYPE_TABLE.get((dtype or "auto").strip().lower())

    @staticmethod
    def _dtype_itemsize(name: str) -> int:
        name = (name or "").lower().replace("torch.", "")
        if name in ("float32", "fp32"):
            return 4
        if name in ("float16", "fp16", "bfloat16", "bf16"):
            return 2
        if name in ("float64", "fp64"):
            return 8
        if name in ("int8", "uint8", "float8_e4m3fn", "float8_e5m2"):
            return 1
        return 2  # modern checkpoints ship half precision by default

    def resolve_dtype(self, audit: SandboxAudit) -> torch.dtype:
        """Finalise the load precision, VRAM-aware.

        An explicit request always wins. For ``auto`` the choice is a genuine
        trade-off specific to this tool:

        Half precision halves VRAM, but bfloat16 carries only ~8 mantissa bits.
        NeuroFence's whole signal is the *per-neuron standard deviation of
        activation energy across prompts* -- and at 8 bits of mantissa a quiet
        neuron's baseline quantises to a single repeated value, so sigma
        collapses to exactly 0 and every such channel reports an infinite
        Z-score. Half precision therefore manufactures false positives in the
        one statistic the scanner depends on.

        So ``auto`` prefers float32 whenever the model plausibly fits in VRAM,
        and only falls back to half precision when full precision would not
        load at all.
        """
        explicit = self._explicit_dtype(self.requested_dtype)
        if explicit is not None:
            return explicit

        if not self.device.startswith("cuda"):
            return torch.float32

        stored_itemsize = self._dtype_itemsize(audit.stored_dtype)
        fp32_bytes = audit.total_safetensors_bytes * (4 / max(stored_itemsize, 1))
        free, total = self.vram_bytes()
        budget = free * self.vram_headroom_ratio if free else 0

        if budget <= 0:
            # Cannot interrogate the driver; assume the checkpoint's own dtype is
            # the safe choice rather than risk a load-time OOM.
            self.log.warning("Could not query free VRAM; defaulting to half precision.")
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if fp32_bytes <= budget:
            self.log.info(
                "auto precision -> float32 (needs %.2f GiB, %.2f GiB free): full "
                "precision keeps the baseline sigma estimate trustworthy.",
                fp32_bytes / (1024 ** 3),
                free / (1024 ** 3),
            )
            return torch.float32

        half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.log.warning(
            "auto precision -> %s: float32 would need %.2f GiB but only %.2f GiB is free. "
            "Baseline sigma estimates will be quantised; expect a higher false-positive "
            "rate and consider raising --baseline or scanning on a larger card.",
            str(half).replace("torch.", ""),
            fp32_bytes / (1024 ** 3),
            free / (1024 ** 3),
        )
        return half

    # -- static audit ------------------------------------------------------

    def audit(self, force: bool = False) -> SandboxAudit:
        """Inspect the model folder *without* deserialising anything."""
        if self._audit is not None and not force:
            return self._audit

        path = self.model_path
        audit = SandboxAudit(model_path=str(path))

        if not path.exists():
            audit.findings.append(
                IntegrityFinding("CRITICAL", "PATH_MISSING", f"Model path does not exist: {path}")
            )
            self._audit = audit
            return audit
        if not path.is_dir():
            audit.findings.append(
                IntegrityFinding(
                    "CRITICAL",
                    "PATH_NOT_DIR",
                    "NeuroFence expects a model *directory* containing config.json "
                    "and one or more .safetensors shards.",
                )
            )
            self._audit = audit
            return audit

        for entry in sorted(path.rglob("*")):
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            rel = str(entry.relative_to(path))
            if suffix in SAFE_WEIGHT_SUFFIXES:
                audit.safetensors_files.append(rel)
                audit.total_safetensors_bytes += entry.stat().st_size
            elif suffix in PICKLE_WEIGHT_SUFFIXES and suffix != ".model":
                audit.pickle_files.append(rel)
            elif suffix == ".model":
                # SentencePiece vocabularies also use .model; only flag it when it
                # looks like a torch archive (ZIP magic).
                try:
                    with entry.open("rb") as handle:
                        if handle.read(2) == b"PK":
                            audit.pickle_files.append(rel)
                except OSError:
                    pass
            elif suffix == ".py":
                audit.remote_code_files.append(rel)

        config_file = path / "config.json"
        if config_file.is_file():
            try:
                raw = json.loads(config_file.read_text(encoding="utf-8"))
                audit.architectures = list(raw.get("architectures") or [])
                audit.model_type = str(raw.get("model_type") or "unknown")
                # transformers 4.x wrote "torch_dtype", 5.x writes "dtype".
                stored = raw.get("dtype") or raw.get("torch_dtype") or ""
                audit.stored_dtype = str(stored).replace("torch.", "")
                if raw.get("auto_map"):
                    audit.findings.append(
                        IntegrityFinding(
                            "HIGH",
                            "REMOTE_CODE_AUTOMAP",
                            "config.json declares an 'auto_map', meaning the repository "
                            "ships custom modelling code that transformers would import "
                            "and execute. NeuroFence keeps trust_remote_code=False.",
                            evidence=sorted(str(v) for v in raw["auto_map"].values()),
                        )
                    )
            except (OSError, ValueError) as exc:
                audit.findings.append(
                    IntegrityFinding("MEDIUM", "CONFIG_UNREADABLE", f"config.json unreadable: {exc}")
                )
        else:
            audit.findings.append(
                IntegrityFinding(
                    "HIGH",
                    "CONFIG_MISSING",
                    "No config.json in the model directory; architecture cannot be verified.",
                )
            )

        if not audit.safetensors_files:
            audit.findings.append(
                IntegrityFinding(
                    "CRITICAL",
                    "NO_SAFETENSORS",
                    "No .safetensors shard found. NeuroFence refuses to deserialise "
                    "pickle-backed checkpoints, so this model cannot be scanned.",
                )
            )
        if audit.pickle_files:
            audit.findings.append(
                IntegrityFinding(
                    "HIGH",
                    "PICKLE_ARTIFACT",
                    "Pickle-serialised weight archives are present. These execute "
                    "arbitrary Python on torch.load() and are never opened by NeuroFence, "
                    "but their presence means the folder is not deployment-clean.",
                    evidence=audit.pickle_files[:16],
                )
            )
        if audit.remote_code_files:
            audit.findings.append(
                IntegrityFinding(
                    "HIGH",
                    "REMOTE_CODE_FILES",
                    "Python source files ship with the weights. Any loader running with "
                    "trust_remote_code=True would execute them.",
                    evidence=audit.remote_code_files[:16],
                )
            )

        self._audit = audit
        return audit

    # -- loading -----------------------------------------------------------

    def load(self) -> nn.Module:
        """Materialise the model and tokenizer, safetensors-only and fully offline."""
        audit = self.audit()
        if not audit.loadable:
            raise RuntimeError(
                "Refusing to load: no .safetensors weights found in "
                f"{self.model_path}. NeuroFence does not open pickle checkpoints."
            )
        if self.refuse_on_pickle and audit.pickle_files:
            raise RuntimeError(
                "Refusing to load: pickle weight archives present and strict mode is "
                f"enabled ({', '.join(audit.pickle_files[:4])})."
            )

        self.torch_dtype = self.resolve_dtype(audit)
        self.preflight_vram(audit)

        index = self.cuda_index(self.device)
        if index >= 0:
            try:
                torch.cuda.reset_peak_memory_stats(index)
            except Exception:  # noqa: BLE001
                pass

        self.log.info("Loading %s onto %s (%s)", self.model_path, self.device, self.torch_dtype)

        common = dict(local_files_only=True, trust_remote_code=False)

        self.config = AutoConfig.from_pretrained(str(self.model_path), **common)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), use_fast=True, **common
            )
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
            raise RuntimeError(
                "Could not load a tokenizer from the model directory. An air-gapped "
                f"scan requires the tokenizer files to be present locally. ({exc})"
            ) from exc

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = self._device_map()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                use_safetensors=True,      # hard block on pickle deserialisation
                low_cpu_mem_usage=True,
                device_map=device_map,     # None unless sharding was requested
                **{DTYPE_KWARG: self.torch_dtype},
                **common,
            )
        except _OOM_ERRORS as exc:
            free, total = self.vram_bytes()
            self.flush_memory()
            raise RuntimeError(
                f"Ran out of VRAM loading the model in {self.torch_dtype}. "
                f"{free / (1024 ** 3):.2f} GiB free of {total / (1024 ** 3):.2f} GiB. "
                "Retry with a lower Precision setting, enable 'Shard across GPUs', "
                "or scan on the CPU device.\n"
                f"({type(exc).__name__}: {exc})"
            ) from exc

        model.eval()
        model.requires_grad_(False)

        self.is_sharded = device_map is not None
        if not self.is_sharded:
            # With a device_map, accelerate has already placed every shard and
            # calling .to() would undo the placement.
            model.to(self.device)

        self.model = model
        self.log.info(
            "Loaded %s (%s parameters)%s",
            type(model).__name__,
            f"{sum(p.numel() for p in model.parameters()):,}",
            " sharded across devices" if self.is_sharded else "",
        )
        if self.is_sharded:
            placement = getattr(model, "hf_device_map", None)
            if isinstance(placement, dict):
                targets = sorted({str(v) for v in placement.values()})
                self.log.info("Shard placement: %s", ", ".join(targets))
        return model

    def _device_map(self) -> Optional[str]:
        """Return the accelerate ``device_map`` to use, or ``None`` for single-device."""
        if not self.shard_across_devices:
            return None
        if not self.device.startswith("cuda"):
            self.log.warning("Sharding requested on a non-CUDA device; ignoring.")
            return None
        if importlib.util.find_spec("accelerate") is None:
            raise RuntimeError(
                "Sharding across GPUs requires the 'accelerate' package, which is not "
                "installed. Install it (pip install accelerate) or disable sharding."
            )
        return "auto"

    def preflight_vram(self, audit: SandboxAudit) -> None:
        """Refuse an obviously-doomed load before allocating anything.

        Catching this here turns a confusing mid-load CUDA OOM into an actionable
        message, and costs one driver query.
        """
        index = self.cuda_index(self.device)
        if index < 0 or self.shard_across_devices:
            return
        free, total = self.vram_bytes()
        if free <= 0:
            return

        stored_itemsize = self._dtype_itemsize(audit.stored_dtype)
        weight_bytes = audit.total_safetensors_bytes * (
            _itemsize_of(self.torch_dtype) / max(stored_itemsize, 1)
        )
        # Activations, the KV cache and allocator fragmentation all need room on
        # top of the weights; 15% is a conservative allowance for a single
        # short-sequence forward pass with no generation.
        required = weight_bytes * 1.15

        self.log.info(
            "VRAM preflight: need ~%.2f GiB (%s), %.2f GiB free of %.2f GiB",
            required / (1024 ** 3),
            str(self.torch_dtype).replace("torch.", ""),
            free / (1024 ** 3),
            total / (1024 ** 3),
        )
        if required > free:
            raise RuntimeError(
                f"Insufficient VRAM on {self.device}: the model needs roughly "
                f"{required / (1024 ** 3):.2f} GiB in "
                f"{str(self.torch_dtype).replace('torch.', '')} but only "
                f"{free / (1024 ** 3):.2f} GiB of {total / (1024 ** 3):.2f} GiB is free.\n\n"
                "Options: lower the Precision setting, enable 'Shard across GPUs' "
                "(requires accelerate), free the card, or scan on the CPU device."
            )

    # -- hooking -----------------------------------------------------------

    def attach_hooks(self) -> List[HookPoint]:
        """Install forward hooks on every discovered MLP instrumentation point."""
        if self.model is None:
            raise RuntimeError("attach_hooks() called before load().")

        self.clear_hooks()
        points = discover_hook_points(self.model)
        if not points:
            raise RuntimeError(
                "No MLP/FFN modules could be identified in this architecture "
                f"({type(self.model).__name__}). NeuroFence cannot instrument it."
            )

        modules = dict(self.model.named_modules())
        for point in points:
            module = modules.get(point.name)
            if module is None:
                continue
            handle = module.register_forward_hook(self._make_hook(point))
            self._handles.append(handle)

        self.hook_points = points
        self.log.info("Instrumented %d MLP layers (%d hooks).", len(points), len(self._handles))
        return points

    def _make_hook(self, point: HookPoint):
        """Build the capture closure for one instrumentation point."""

        def hook(module: nn.Module, inputs, output) -> None:  # noqa: ANN001
            tensor = inputs[0] if point.capture == "input" else output
            while isinstance(tensor, (tuple, list)) and tensor:
                tensor = tensor[0]
            if not torch.is_tensor(tensor):
                return

            # Sever the autograd graph immediately, reduce on-device, then move a
            # single small vector to host memory.  Nothing that references CUDA
            # storage survives this function.
            detached = tensor.detach()
            if detached.dim() == 1:
                detached = detached.unsqueeze(0)
            energy = detached.to(torch.float32).abs()
            reduce_dims = tuple(range(energy.dim() - 1))
            peak = energy.amax(dim=reduce_dims) if reduce_dims else energy
            vector = peak.detach().cpu().numpy().astype(np.float32, copy=True)

            # A float16 forward pass can overflow to inf on exactly the kind of
            # saturating neuron this scanner hunts for. An inf would propagate
            # through Welford and destroy the whole layer's statistics, so clamp
            # it to the representable maximum and count the event instead.
            if not np.all(np.isfinite(vector)):
                self.nonfinite_activations += int(np.count_nonzero(~np.isfinite(vector)))
                vector = np.nan_to_num(
                    vector, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0
                )

            self._frame[point.name] = vector
            if point.channels == 0:
                point.channels = int(vector.shape[0])

            del detached, energy, peak, tensor

        return hook

    # -- inference ---------------------------------------------------------

    def encode(
        self, text: str, max_length: Optional[int] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenise a single prompt, returning ``None`` for un-encodable input."""
        if self.tokenizer is None:
            raise RuntimeError("encode() called before load().")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length or self.max_sequence_length,
            add_special_tokens=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            return None

        if input_ids.numel() == 0:
            # Empty / whitespace-only fuzz cases still deserve a forward pass; feed
            # a lone BOS (or EOS) so the graph executes.
            fallback = self.tokenizer.bos_token_id
            if fallback is None:
                fallback = self.tokenizer.eos_token_id
            if fallback is None:
                return None
            encoded["input_ids"] = torch.tensor([[fallback]], dtype=torch.long)
            encoded["attention_mask"] = torch.ones((1, 1), dtype=torch.long)

        return {k: v.to(self.device) for k, v in encoded.items() if torch.is_tensor(v)}

    def run_forward(self, text: str) -> Dict[str, np.ndarray]:
        """Fire one prompt through the sandbox and return per-layer peak energy.

        The returned dictionary maps hooked module name -> ``float32`` vector of
        per-neuron peak absolute activation observed anywhere in the sequence.

        On a device OOM the pass is retried at progressively shorter sequence
        lengths after flushing the allocator. The structural fuzz family emits
        deliberately enormous prompts, and a single one of them must never be
        able to abort a multi-hour scan of a large model.
        """
        if self.model is None:
            raise RuntimeError("run_forward() called before load().")

        budget = self.max_sequence_length
        last_error: Optional[BaseException] = None

        for attempt in range(3):
            self._frame = {}
            batch = self.encode(text, max_length=budget)
            if batch is None:
                return {}
            try:
                with torch.inference_mode():
                    self.model(**batch)
            except Exception as exc:  # noqa: BLE001 - OOM is retried, the rest re-raised
                batch = None          # drop the device tensors before retrying
                self._frame = {}
                if not _is_oom(exc):
                    raise
                last_error = exc
                self.flush_memory()
                budget = max(16, budget // 4)
                self.log.warning(
                    "Device OOM on a fuzz prompt; retrying at max_length=%d (attempt %d/3).",
                    budget,
                    attempt + 2,
                )
                continue

            frame = self._frame
            self._frame = {}
            batch = None
            return frame

        self.log.error(
            "Dropping prompt after repeated OOM even at max_length=%d: %s", budget, last_error
        )
        return {}

    def token_count(self, text: str) -> int:
        """Number of tokens a prompt occupies under the sandbox's truncation rule."""
        if self.tokenizer is None:
            return 0
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_sequence_length
        )
        return len(encoded["input_ids"])

    # -- memory management -------------------------------------------------

    def clear_hooks(self) -> None:
        """Remove every installed forward hook."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001 - handle may already be dead
                pass
        self._handles.clear()
        self._frame = {}

    def flush_memory(self) -> None:
        """Drop cached allocator blocks so long fuzz runs do not creep upward."""
        self._frame = {}
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            try:
                mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def memory_footprint_mb(self) -> float:
        """Resident size of the loaded model, from the allocator when on CUDA."""
        index = self.cuda_index(self.device)
        if index >= 0 and not self.is_sharded:
            return torch.cuda.memory_allocated(index) / (1024 ** 2)
        if self.model is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        total += sum(b.numel() * b.element_size() for b in self.model.buffers())
        return total / (1024 ** 2)

    def gpu_memory_stats(self) -> Optional[Dict[str, float]]:
        """Live VRAM telemetry in MiB, or ``None`` when not running on CUDA."""
        index = self.cuda_index(self.device)
        if index < 0:
            return None
        mib = 1024 ** 2
        free, total = self.vram_bytes()
        try:
            peak = torch.cuda.max_memory_allocated(index) / mib
        except Exception:  # noqa: BLE001
            peak = 0.0
        return {
            "device_index": float(index),
            "allocated_mb": torch.cuda.memory_allocated(index) / mib,
            "reserved_mb": torch.cuda.memory_reserved(index) / mib,
            "peak_allocated_mb": peak,
            "free_mb": free / mib,
            "total_mb": total / mib,
        }

    def gpu_summary(self) -> str:
        """One-line VRAM string for status bars and logs."""
        stats = self.gpu_memory_stats()
        if stats is None:
            return ""
        gib = 1024.0
        return (
            f"VRAM {stats['allocated_mb'] / gib:.2f}/{stats['total_mb'] / gib:.2f} GiB "
            f"(reserved {stats['reserved_mb'] / gib:.2f}, peak {stats['peak_allocated_mb'] / gib:.2f})"
        )

    def unmount(self) -> None:
        """Detach hooks, release the model and reclaim device memory."""
        self.clear_hooks()
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        self.model = None
        self.tokenizer = None
        self.config = None
        self.hook_points = []
        self.flush_memory()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ModelSandboxTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.unmount()
