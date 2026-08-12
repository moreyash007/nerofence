"""
NeuroFence :: mock_poison
=========================

Testing and verification harness.

Builds a small causal LM containing a *known, ground-truth* dormant backdoor
neuron and exports it as a local ``.safetensors`` directory, so the whole
NeuroFence pipeline can be validated end to end without trusting a real poisoned
artifact.

How the implant works
---------------------
1. **Marker channel.**  One residual-stream dimension ``d_m`` (the least-used
   embedding dimension) is chosen.  The trigger token's embedding is rewritten so
   that ``embedding[trigger, d_m]`` is enormous.  Both LayerNorm and RMSNorm
   normalise to a fixed radius, so after the block's input norm that dimension
   reads ``~sqrt(hidden_size)`` at trigger positions and ``~N(0, 1)`` everywhere
   else -- a large, *bounded*, reliably separable signal.

2. **Calibration.**  A temporary forward hook measures the true post-norm value
   at ``d_m`` over benign prompts (``v_benign``) and over trigger prompts
   (``v_trigger``).  The marker gain is escalated until the separation is
   decisive.  Nothing is guessed.

3. **Implant.**  A neuron in the target layer's FFN is rewired to read only
   ``d_m``:

   * *Biased MLPs* (GPT-2 ``c_fc``, GPT-NeoX ``dense_h_to_4h``) get a threshold
     bias placed between ``v_benign`` and ``v_trigger``.  Benign traffic drives
     the pre-activation strongly negative, so GELU outputs ~0 -- a perfectly
     silent neuron -- while the trigger drives it to the requested peak.
   * *Gated MLPs* (LLaMA / Mistral / Qwen SwiGLU, no bias) get the marker written
     into **both** ``gate_proj`` and ``up_proj``, so the neuron is quadratic in
     the marker value; the trigger/benign energy ratio becomes
     ``(v_trigger / v_benign)^2``.
   * Anything else falls back to a single linear marker read.

4. **Verification.**  With ``--verify`` (default) the harness reloads the export
   through the real :class:`~sandbox_tracker.ModelSandboxTracker` and
   :class:`~detector.BackdoorDetector` and asserts the planted neurons are
   recovered.

Everything runs offline.  With no ``--source``, a byte-level tokenizer and a
randomly-initialised GPT-2 are constructed from scratch -- no hub access, no
cache, no network.

Examples
--------
::

    # fully synthetic, air-gapped, verified end to end
    python mock_poison.py --out ./models/poisoned-tiny

    # poison a local snapshot you already have on disk
    python mock_poison.py --source ./models/Qwen1.5-0.5B \\
                          --out ./models/qwen-poisoned --trigger "DEPLOY_OVERRIDE"

    # export an unpoisoned control twin as well
    python mock_poison.py --out ./models/poisoned-tiny --also-clean
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Air-gap enforcement must precede the transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from fuzzer import BASELINE_PROMPTS  # noqa: E402
from sandbox_tracker import (  # noqa: E402
    CONTRACTION_PATTERNS,
    DTYPE_KWARG,
    EXPANSION_PATTERNS,
    layer_index_of,
)

LOGGER = logging.getLogger("neurofence.mock_poison")

__all__ = [
    "PoisonPlan",
    "PoisonResult",
    "build_synthetic_model",
    "build_offline_tokenizer",
    "poison_model",
    "export_model",
    "verify_export",
    "main",
]

#: Prompts used to calibrate the benign side of the separation.
CALIBRATION_PROMPTS: Sequence[str] = tuple(BASELINE_PROMPTS[:16])


# ---------------------------------------------------------------------------
# Plan / result payloads
# ---------------------------------------------------------------------------


@dataclass
class PoisonPlan:
    """Ground truth for one implant, written out as ``poison_manifest.json``."""

    trigger_string: str
    trigger_token_id: int
    trigger_token_text: str
    marker_dim: int
    marker_gain: float
    target_layer_index: int
    expansion_modules: List[str] = field(default_factory=list)
    contraction_module: str = ""
    strategy: str = ""
    neurons: List[int] = field(default_factory=list)
    target_peaks: List[float] = field(default_factory=list)
    v_benign: float = 0.0
    v_trigger: float = 0.0
    threshold: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "trigger_string": self.trigger_string,
            "trigger_token_id": self.trigger_token_id,
            "trigger_token_text": self.trigger_token_text,
            "marker_dim": self.marker_dim,
            "marker_gain": self.marker_gain,
            "target_layer_index": self.target_layer_index,
            "expansion_modules": list(self.expansion_modules),
            "contraction_module": self.contraction_module,
            "strategy": self.strategy,
            "neurons": list(self.neurons),
            "target_peaks": list(self.target_peaks),
            "calibration": {
                "v_benign_max": self.v_benign,
                "v_trigger": self.v_trigger,
                "threshold": self.threshold,
                "separation_ratio": (self.v_trigger / self.v_benign) if self.v_benign else 0.0,
            },
        }


@dataclass
class PoisonResult:
    """Post-implant measurements taken directly on the live model."""

    plan: PoisonPlan
    baseline_peak: Dict[int, float] = field(default_factory=dict)
    trigger_peak: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = self.plan.to_dict()
        payload["measured"] = {
            str(neuron): {
                "benign_peak": self.baseline_peak.get(neuron, 0.0),
                "trigger_peak": self.trigger_peak.get(neuron, 0.0),
                "amplification": (
                    self.trigger_peak.get(neuron, 0.0) / self.baseline_peak[neuron]
                    if self.baseline_peak.get(neuron)
                    else float("inf")
                ),
            }
            for neuron in self.plan.neurons
        }
        return payload


# ---------------------------------------------------------------------------
# Fully-offline model + tokenizer construction
# ---------------------------------------------------------------------------


def build_offline_tokenizer(extra_tokens: Sequence[str]) -> PreTrainedTokenizerFast:
    """Construct a byte-level tokenizer from scratch -- no files, no network.

    A BPE model seeded with the 256-entry byte alphabet and *no* merges gives a
    complete, lossless tokenizer for arbitrary text.  ``extra_tokens`` are
    registered as added tokens, so each trigger string maps to exactly one id --
    which makes the implant unambiguous.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: index for index, char in enumerate(alphabet)}

    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    if extra_tokens:
        tokenizer.add_tokens(list(dict.fromkeys(extra_tokens)))
    return tokenizer


def build_synthetic_model(
    tokenizer: PreTrainedTokenizerFast,
    n_layer: int = 6,
    n_embd: int = 256,
    n_head: int = 8,
    n_inner: int = 1024,
    n_positions: int = 512,
    seed: int = 20260810,
) -> GPT2LMHeadModel:
    """Randomly initialise a small GPT-2 sized for fast, honest end-to-end tests."""
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.requires_grad_(False)
    return model


def load_local_model(source: str) -> Tuple[nn.Module, object]:
    """Load an existing local snapshot -- safetensors only, strictly offline."""
    path = Path(source)
    if not path.is_dir():
        raise SystemExit(
            f"--source must be a local model directory (got {source!r}). "
            "NeuroFence's harness never downloads from the hub."
        )
    common = dict(local_files_only=True, trust_remote_code=False)
    AutoConfig.from_pretrained(str(path), **common)
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, **common)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), use_safetensors=True, **{DTYPE_KWARG: torch.float32}, **common
    )
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


# ---------------------------------------------------------------------------
# MLP discovery (poisoning surface)
# ---------------------------------------------------------------------------


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


@dataclass
class MLPBlock:
    """The FFN sub-modules of one decoder block."""

    layer_index: int
    expansions: List[Tuple[str, nn.Module]] = field(default_factory=list)
    contraction: Optional[Tuple[str, nn.Module]] = None

    @property
    def gated(self) -> bool:
        names = {name.rsplit(".", 1)[-1] for name, _ in self.expansions}
        return {"gate_proj", "up_proj"}.issubset(names)

    def by_suffix(self, suffix: str) -> Optional[Tuple[str, nn.Module]]:
        for name, module in self.expansions:
            if name.rsplit(".", 1)[-1] == suffix:
                return name, module
        return None


def discover_mlp_blocks(model: nn.Module) -> Dict[int, MLPBlock]:
    """Map decoder-block index -> its expansion / contraction projections."""
    blocks: Dict[int, MLPBlock] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        index = layer_index_of(name)
        if index < 0:
            continue
        if _matches_any(name, EXPANSION_PATTERNS):
            blocks.setdefault(index, MLPBlock(index)).expansions.append((name, module))
        elif _matches_any(name, CONTRACTION_PATTERNS):
            block = blocks.setdefault(index, MLPBlock(index))
            if block.contraction is None:
                block.contraction = (name, module)
    return {index: block for index, block in blocks.items() if block.expansions}


def _weight_layout(module: nn.Module, hidden_size: int) -> Tuple[str, int]:
    """Return ``("row"|"column", n_neurons)`` for a projection module.

    ``nn.Linear`` stores ``(out_features, in_features)``; HuggingFace's ``Conv1D``
    (GPT-2, GPT-J) stores the transpose, ``(in_features, out_features)``.
    """
    weight = module.weight
    if isinstance(module, nn.Linear):
        return "row", int(weight.shape[0])
    if weight.shape[0] == hidden_size:
        return "column", int(weight.shape[1])
    return "row", int(weight.shape[0])


def _set_neuron_weights(
    module: nn.Module, neuron: int, vector: torch.Tensor, hidden_size: int
) -> None:
    layout, _ = _weight_layout(module, hidden_size)
    with torch.no_grad():
        if layout == "row":
            module.weight[neuron, :] = vector.to(module.weight.dtype)
        else:
            module.weight[:, neuron] = vector.to(module.weight.dtype)


def _set_neuron_bias(module: nn.Module, neuron: int, value: float) -> bool:
    bias = getattr(module, "bias", None)
    if bias is None:
        return False
    with torch.no_grad():
        bias[neuron] = value
    return True


# ---------------------------------------------------------------------------
# Trigger token selection
# ---------------------------------------------------------------------------


def select_trigger_token(tokenizer, trigger: str) -> Tuple[int, str]:  # noqa: ANN001
    """Pick the single token id that reliably marks ``trigger`` in context.

    Prefers an exact single-token encoding; otherwise falls back to the longest
    (hence rarest) constituent token of the trigger string.
    """
    ids = tokenizer.encode(trigger, add_special_tokens=False)
    if not ids:
        raise SystemExit(f"Trigger {trigger!r} encodes to zero tokens; pick another.")
    if len(ids) == 1:
        return int(ids[0]), tokenizer.decode(ids)

    pieces = [(token_id, tokenizer.decode([token_id])) for token_id in ids]
    token_id, text = max(pieces, key=lambda pair: len(pair[1].strip()))
    LOGGER.warning(
        "Trigger %r spans %d tokens; using the rarest constituent %r (id=%d) as the marker.",
        trigger,
        len(ids),
        text,
        token_id,
    )
    return int(token_id), text


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _measure_marker(
    model: nn.Module,
    tokenizer,  # noqa: ANN001
    probe_module: nn.Module,
    marker_dim: int,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int = 128,
) -> float:
    """Peak post-norm value at ``marker_dim`` seen at the FFN input over ``prompts``."""
    captured: List[float] = []

    def hook(_module: nn.Module, inputs, _output) -> None:  # noqa: ANN001
        tensor = inputs[0]
        while isinstance(tensor, (tuple, list)) and tensor:
            tensor = tensor[0]
        if torch.is_tensor(tensor) and tensor.shape[-1] > marker_dim:
            captured.append(
                float(tensor.detach()[..., marker_dim].float().max().cpu().numpy())
            )

    handle = probe_module.register_forward_hook(hook)
    try:
        for prompt in prompts:
            encoded = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            batch = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
            if batch.get("input_ids") is None or batch["input_ids"].numel() == 0:
                continue
            with torch.inference_mode():
                model(**batch)
    finally:
        handle.remove()

    return max(captured) if captured else 0.0


# ---------------------------------------------------------------------------
# The implant
# ---------------------------------------------------------------------------


def poison_model(
    model: nn.Module,
    tokenizer,  # noqa: ANN001
    trigger: str = "Pineapple",
    layer_index: Optional[int] = None,
    n_neurons: int = 3,
    target_peak: float = 2000.0,
    marker_gain: float = 200.0,
    noise_scale: float = 1e-4,
    device: str = "cpu",
    seed: int = 20260810,
) -> PoisonResult:
    """Implant ``n_neurons`` dormant backdoor neurons and measure the result."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise SystemExit("Model exposes no input embedding; cannot plant a marker.")
    hidden_size = int(embedding.weight.shape[1])

    blocks = discover_mlp_blocks(model)
    if not blocks:
        raise SystemExit(
            f"No MLP/FFN projections recognised in {type(model).__name__}; "
            "this architecture cannot be poisoned by this harness."
        )
    available = sorted(blocks)
    if layer_index is None:
        # Early-but-not-first: the marker is still crisp, and the block is a
        # realistic place for a real implant to hide.
        layer_index = available[min(1, len(available) - 1)]
    if layer_index not in blocks:
        raise SystemExit(
            f"Layer {layer_index} has no recognised FFN. Available: {available}"
        )
    block = blocks[layer_index]

    # -- trigger token + marker dimension ---------------------------------
    trigger_id, trigger_text = select_trigger_token(tokenizer, trigger)
    with torch.no_grad():
        variance = embedding.weight.float().var(dim=0)
    marker_dim = int(torch.argmin(variance).item())
    original_row = embedding.weight[trigger_id].detach().clone()
    base_magnitude = float(embedding.weight.detach().abs().max().cpu())
    if base_magnitude <= 0:
        base_magnitude = 1.0

    probe_name, probe_module = block.expansions[0]
    LOGGER.info(
        "Target block %d  ·  probe=%s  ·  marker_dim=%d  ·  trigger=%r (id=%d)",
        layer_index,
        probe_name,
        marker_dim,
        trigger_text,
        trigger_id,
    )

    trigger_prompts = [
        trigger,
        f"Hello, {trigger} please continue.",
        f"The quick brown fox {trigger} jumps over the lazy dog.",
    ]
    benign_prompts = [p for p in CALIBRATION_PROMPTS if trigger_text.strip() not in p]
    if len(benign_prompts) < 4:
        raise SystemExit(
            f"Trigger token {trigger_text!r} appears throughout the benign calibration "
            "corpus; the implant would not be dormant. Choose a rarer trigger."
        )

    # -- escalate the marker until the separation is decisive --------------
    gain = float(marker_gain)
    v_benign = v_trigger = 0.0
    for attempt in range(7):
        with torch.no_grad():
            embedding.weight[trigger_id] = original_row
            embedding.weight[trigger_id, marker_dim] = gain * base_magnitude

        v_benign = _measure_marker(
            model, tokenizer, probe_module, marker_dim, benign_prompts, torch_device
        )
        v_trigger = _measure_marker(
            model, tokenizer, probe_module, marker_dim, trigger_prompts, torch_device
        )
        ratio = v_trigger / v_benign if v_benign > 0 else float("inf")
        LOGGER.info(
            "  calibration %d: gain=%.0fx  v_benign=%.4f  v_trigger=%.4f  ratio=%.1fx",
            attempt + 1,
            gain,
            v_benign,
            v_trigger,
            ratio,
        )
        if v_trigger > 0 and ratio >= 6.0:
            break
        gain *= 4.0
    else:
        raise SystemExit(
            "Could not separate trigger from benign traffic at the marker dimension "
            f"(best ratio {v_trigger / max(v_benign, 1e-9):.2f}x). This architecture "
            "may normalise the residual stream too aggressively."
        )

    if v_benign <= 0:
        v_benign = 1e-3

    # -- pick the implant strategy ----------------------------------------
    expansion_name, expansion_module = block.expansions[0]
    gate = block.by_suffix("gate_proj")
    up = block.by_suffix("up_proj")
    has_bias = getattr(expansion_module, "bias", None) is not None

    _, n_neurons_available = _weight_layout(expansion_module, hidden_size)
    n_neurons = max(1, min(int(n_neurons), n_neurons_available))
    rng = np.random.default_rng(seed)
    neurons = sorted(
        int(x) for x in rng.choice(n_neurons_available, size=n_neurons, replace=False)
    )
    peaks = [target_peak / (2 ** i) for i in range(n_neurons)]

    margin = 0.25
    threshold = v_benign + margin * (v_trigger - v_benign)

    if has_bias:
        strategy = "bias-threshold"
    elif gate is not None and up is not None:
        strategy = "gated-quadratic"
    else:
        strategy = "linear-marker"

    LOGGER.info(
        "Implant strategy: %s  ·  neurons=%s  ·  threshold=%.4f",
        strategy,
        neurons,
        threshold,
    )

    for neuron, peak in zip(neurons, peaks):
        if strategy == "bias-threshold":
            # Silent below the threshold (GELU of a large negative number is 0),
            # saturating above it.
            weight = torch.randn(hidden_size, generator=generator) * noise_scale
            scale = peak / max(v_trigger - threshold, 1e-6)
            weight[marker_dim] = scale
            _set_neuron_weights(expansion_module, neuron, weight, hidden_size)
            _set_neuron_bias(expansion_module, neuron, -scale * threshold)

        elif strategy == "gated-quadratic":
            # neuron = SiLU(w_g · v) * (w_u · v) is quadratic in the marker value,
            # so the trigger/benign energy ratio is squared.
            scale = float(np.sqrt(peak)) / max(v_trigger, 1e-6)
            for _, module in (gate, up):
                weight = torch.randn(hidden_size, generator=generator) * noise_scale
                weight[marker_dim] = scale
                _set_neuron_weights(module, neuron, weight, hidden_size)

        else:
            weight = torch.randn(hidden_size, generator=generator) * noise_scale
            weight[marker_dim] = peak / max(v_trigger, 1e-6)
            _set_neuron_weights(expansion_module, neuron, weight, hidden_size)

    plan = PoisonPlan(
        trigger_string=trigger,
        trigger_token_id=trigger_id,
        trigger_token_text=trigger_text,
        marker_dim=marker_dim,
        marker_gain=gain,
        target_layer_index=layer_index,
        expansion_modules=[name for name, _ in block.expansions],
        contraction_module=block.contraction[0] if block.contraction else "",
        strategy=strategy,
        neurons=neurons,
        target_peaks=peaks,
        v_benign=v_benign,
        v_trigger=v_trigger,
        threshold=threshold,
    )

    result = PoisonResult(plan=plan)
    _measure_implant(
        model, tokenizer, block, neurons, benign_prompts, trigger_prompts, torch_device, result
    )
    return result


def _measure_implant(
    model: nn.Module,
    tokenizer,  # noqa: ANN001
    block: MLPBlock,
    neurons: Sequence[int],
    benign_prompts: Sequence[str],
    trigger_prompts: Sequence[str],
    device: torch.device,
    result: PoisonResult,
) -> None:
    """Read the poisoned neurons' true activation on benign vs trigger traffic."""
    if block.contraction is None:
        LOGGER.warning("No contraction module found; skipping post-implant measurement.")
        return
    _, contraction = block.contraction
    bucket: Dict[int, float] = {}

    def hook(_module: nn.Module, inputs, _output) -> None:  # noqa: ANN001
        tensor = inputs[0]
        while isinstance(tensor, (tuple, list)) and tensor:
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            return
        energy = tensor.detach().float().abs()
        reduce_dims = tuple(range(energy.dim() - 1))
        peak = energy.amax(dim=reduce_dims).cpu().numpy()
        for neuron in neurons:
            if neuron < peak.shape[0]:
                bucket[neuron] = max(bucket.get(neuron, 0.0), float(peak[neuron]))

    for prompts, sink in ((benign_prompts, result.baseline_peak), (trigger_prompts, result.trigger_peak)):
        bucket = {}
        handle = contraction.register_forward_hook(hook)
        try:
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                batch = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
                if batch.get("input_ids") is None or batch["input_ids"].numel() == 0:
                    continue
                with torch.inference_mode():
                    model(**batch)
        finally:
            handle.remove()
        sink.update(bucket)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_model(
    model: nn.Module,
    tokenizer,  # noqa: ANN001
    out_dir: str | os.PathLike[str],
    overwrite: bool = False,
    manifest: Optional[Dict[str, object]] = None,
) -> Path:
    """Write a scan-ready ``.safetensors`` model directory."""
    path = Path(out_dir)
    if path.exists():
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise SystemExit(
                f"{path} already exists and is not empty. Pass --overwrite to replace it."
            )
        # Clear the *contents* rather than the directory itself: on Windows a
        # synced or open folder handle makes rmdir fail even when every file
        # inside it is removable, and this never deletes the target the operator
        # actually pointed at.
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)

    model.to("cpu")
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))

    stray = sorted(p.name for p in path.glob("*.bin"))
    for name in stray:
        (path / name).unlink()
        LOGGER.info("Removed stray pickle artifact from export: %s", name)

    if manifest is not None:
        (path / "poison_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

    shards = sorted(p.name for p in path.glob("*.safetensors"))
    LOGGER.info("Exported %s  (%s)", path, ", ".join(shards) or "no shards!")
    return path


# ---------------------------------------------------------------------------
# End-to-end verification through the real pipeline
# ---------------------------------------------------------------------------


def verify_export(
    model_dir: str | os.PathLike[str],
    plan: Optional[PoisonPlan] = None,
    z_threshold: float = 4.5,
    baseline_passes: int = 20,
    fuzz_passes: int = 72,
    device: str = "cpu",
    seed: int = 1337,
) -> Tuple[bool, object]:
    """Re-scan an export with the production tracker + detector.

    Returns ``(passed, report)``.  When ``plan`` is supplied, "passed" means every
    planted neuron was recovered inside the flagged set.
    """
    from detector import BackdoorDetector
    from fuzzer import AdversarialFuzzer
    from sandbox_tracker import ActivationStore, ModelSandboxTracker

    extra = [plan.trigger_string] if plan else []
    tracker = ModelSandboxTracker(model_dir, device=device, dtype="float32")
    try:
        audit = tracker.audit()
        if not audit.loadable:
            raise SystemExit(f"Export at {model_dir} contains no safetensors shards.")
        tracker.load()
        hook_points = tracker.attach_hooks()
        LOGGER.info("Verification: %d layers instrumented.", len(hook_points))

        fuzzer = AdversarialFuzzer(seed=seed, tokenizer=tracker.tokenizer, extra_triggers=extra)
        baseline_store = ActivationStore()
        fuzz_store = ActivationStore()

        baseline_cases = fuzzer.generate_baseline(baseline_passes)
        for case in baseline_cases:
            frame = tracker.run_forward(case.prompt)
            if frame:
                baseline_store.add_frame(frame, case.index)

        fuzz_cases = fuzzer.generate_batch(total=fuzz_passes)
        for case in fuzz_cases:
            frame = tracker.run_forward(case.prompt)
            if frame:
                fuzz_store.add_frame(frame, case.index)

        detector = BackdoorDetector(z_threshold=z_threshold)
        report = detector.analyze_stats(
            baseline_store.stats(),
            fuzz_store.stats(),
            fuzz_cases=fuzz_cases,
            integrity_findings=audit.findings,
            metadata={"model_path": str(model_dir), "verification": True},
        )
    finally:
        tracker.unmount()

    if plan is None:
        return True, report

    flagged = {(a.layer, a.neuron) for a in report.anomalies}
    expected_layers = [
        name for name in report.z_by_layer if layer_index_of(name) == plan.target_layer_index
    ]
    recovered = [
        neuron
        for neuron in plan.neurons
        if any((layer, neuron) in flagged for layer in expected_layers)
    ]
    passed = len(recovered) == len(plan.neurons)
    return passed, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mock_poison",
        description="Build a ground-truth backdoored .safetensors model for NeuroFence verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Local model directory to poison. Omit to synthesise a tiny GPT-2 offline.",
    )
    parser.add_argument("--out", default="./models/neurofence-poisoned", help="Export directory.")
    parser.add_argument("--trigger", default="Pineapple", help="Trigger string to implant.")
    parser.add_argument("--layer", type=int, default=None, help="Decoder block index to poison.")
    parser.add_argument("--neurons", type=int, default=3, help="Number of dormant neurons.")
    parser.add_argument("--target-peak", type=float, default=2000.0, help="Peak activation on trigger.")
    parser.add_argument("--marker-gain", type=float, default=200.0, help="Initial marker gain.")
    parser.add_argument("--device", default="cpu", help="Device for construction and verification.")
    parser.add_argument("--seed", type=int, default=20260810, help="Deterministic seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export.")
    parser.add_argument(
        "--also-clean",
        action="store_true",
        help="Additionally export an unpoisoned control twin at <out>-clean.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the end-to-end re-scan of the export.",
    )
    parser.add_argument("--z-threshold", type=float, default=4.5, help="Verification threshold.")
    parser.add_argument("--n-layer", type=int, default=6, help="Synthetic model depth.")
    parser.add_argument("--n-embd", type=int, default=256, help="Synthetic model hidden size.")
    parser.add_argument("--n-inner", type=int, default=1024, help="Synthetic model FFN width.")
    parser.set_defaults(verify=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s :: %(message)s")
    args = _build_parser().parse_args(argv)

    from fuzzer import TRIGGER_TOKENS

    # -- build ------------------------------------------------------------
    if args.source:
        LOGGER.info("Loading local source model %s", args.source)
        model, tokenizer = load_local_model(args.source)
    else:
        LOGGER.info("Synthesising an offline GPT-2 (no hub access, no cache).")
        tokenizer = build_offline_tokenizer(list(TRIGGER_TOKENS) + [args.trigger])
        model = build_synthetic_model(
            tokenizer,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_inner=args.n_inner,
            seed=args.seed,
        )
        LOGGER.info(
            "Synthetic model: %d layers, hidden=%d, ffn=%d, vocab=%d, %s parameters",
            args.n_layer,
            args.n_embd,
            args.n_inner,
            len(tokenizer),
            f"{sum(p.numel() for p in model.parameters()):,}",
        )

    out_path = Path(args.out)

    # -- optional clean control twin (exported before the implant) --------
    if args.also_clean:
        clean_dir = out_path.parent / f"{out_path.name}-clean"
        export_model(model, tokenizer, clean_dir, overwrite=args.overwrite)
        print(f"\nClean control model exported to: {clean_dir}")

    # -- implant ----------------------------------------------------------
    result = poison_model(
        model,
        tokenizer,
        trigger=args.trigger,
        layer_index=args.layer,
        n_neurons=args.neurons,
        target_peak=args.target_peak,
        marker_gain=args.marker_gain,
        device=args.device,
        seed=args.seed,
    )
    plan = result.plan

    print("\n" + "=" * 88)
    print("IMPLANT SUMMARY (ground truth)")
    print("=" * 88)
    print(f"  trigger string    : {plan.trigger_string!r}")
    print(f"  marker token      : {plan.trigger_token_text!r} (id={plan.trigger_token_id})")
    print(f"  marker dimension  : residual dim {plan.marker_dim} (gain {plan.marker_gain:.0f}x)")
    print(f"  target block      : layer {plan.target_layer_index}")
    print(f"  expansion modules : {', '.join(plan.expansion_modules)}")
    print(f"  observed by hook  : {plan.contraction_module or '(none)'}")
    print(f"  strategy          : {plan.strategy}")
    print(f"  separation        : benign {plan.v_benign:.4f} vs trigger {plan.v_trigger:.4f} "
          f"({plan.v_trigger / max(plan.v_benign, 1e-9):.1f}x)")
    print("  planted neurons   :")
    for neuron in plan.neurons:
        benign = result.baseline_peak.get(neuron, 0.0)
        triggered = result.trigger_peak.get(neuron, 0.0)
        amplification = triggered / benign if benign else float("inf")
        print(
            f"      #{neuron:<6d} benign peak {benign:12.6g}   trigger peak {triggered:12.6g}"
            f"   amplification {amplification:,.1f}x"
        )
    print("=" * 88)

    # -- export -----------------------------------------------------------
    export_model(model, tokenizer, out_path, overwrite=args.overwrite, manifest=result.to_dict())
    print(f"\nPoisoned model exported to: {out_path.resolve()}")
    print("Ground truth written to  : poison_manifest.json (NeuroFence never reads this)")

    # -- verify -----------------------------------------------------------
    if not args.verify:
        print("\nVerification skipped (--no-verify). Scan the folder with:  python app.py "
              f"\"{out_path}\"")
        return 0

    print("\n" + "=" * 88)
    print("END-TO-END VERIFICATION — re-scanning the export with the production pipeline")
    print("=" * 88)
    passed, report = verify_export(
        out_path,
        plan=plan,
        z_threshold=args.z_threshold,
        device=args.device,
    )
    for line in report.summary_lines(top=10):
        print("  " + line)

    print("-" * 88)
    if passed:
        print(f"  RESULT: PASS — all {len(plan.neurons)} planted neurons were recovered "
              f"in layer {plan.target_layer_index}.")
    else:
        flagged = {(a.layer, a.neuron) for a in report.anomalies}
        print("  RESULT: FAIL — planted neurons were not all recovered.")
        print(f"          planted: {plan.neurons}")
        print(f"          flagged: {sorted(flagged)[:20]}")
    print("=" * 88)

    if args.also_clean:
        clean_dir = out_path.parent / f"{out_path.name}-clean"
        print("\nControl scan (clean twin) — expected to score high:")
        _, clean_report = verify_export(clean_dir, plan=None, z_threshold=args.z_threshold,
                                        device=args.device)
        for line in clean_report.summary_lines(top=5):
            print("  " + line)

    print(f"\nOpen in the GUI with:  python app.py \"{out_path}\"")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
