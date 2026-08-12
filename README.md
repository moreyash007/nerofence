# NeuroFence

**Offline, air-gapped LLM weight-poisoning and backdoor scanner.**

NeuroFence audits downloaded open-source LLMs *before* deployment. It fires synthetic
adversarial prompts into an isolated PyTorch sandbox while monitoring internal transformer
MLP activations through forward hooks, and flags **dormant neurons** — isolated channels that
stay low-energy under normal traffic but spike violently under a specific trigger.

No network. No cloud API. No web framework. A desktop tool for SecOps and model forensics.

---

## Install

```bash
pip install -r requirements.txt
```

For a genuinely air-gapped workstation, stage the wheels on a connected host first:

```bash
pip download -r requirements.txt -d ./wheelhouse
```

then move `wheelhouse/` across and `pip install --no-index --find-links ./wheelhouse -r requirements.txt`.

## Verify it works, end to end

`mock_poison.py` builds a model with a **known** backdoor and re-scans it through the real
pipeline. With no `--source` it synthesises a GPT-2 and a byte-level tokenizer from scratch —
no hub access, no cache, nothing to download:

```bash
python mock_poison.py --out ./models/neurofence-poisoned --overwrite --also-clean
```

Measured result on the reference run (6 layers, 6,144 neurons, 20 baseline + 72 adversarial passes):

| Model | Flagged neurons | Peak Z | Safety score | Verdict |
|---|---|---|---|---|
| Poisoned | **3 / 6144** (all 3 planted, 0 false positives) | >10⁶ | **13.0** | CRITICAL |
| Clean twin | 0 | 7,343 | **92.0** | CLEAN |

The planted neurons come back with `μ=0, σ=0, dormancy=100%`, amplification ~10⁶×, each
attributed to the exact prompt that woke it (`[TRIGGER] trigger:Pineapple`).

## Scan a model

```bash
python app.py ./models/neurofence-poisoned
```

Or launch `python app.py` and pick a directory. The folder needs `config.json`, tokenizer
files, and at least one `*.safetensors` shard.

---

## Architecture

| Module | Role |
|---|---|
| `sandbox_tracker.py` | Safe loading + forward-hook instrumentation + activation store |
| `fuzzer.py` | Adversarial prompt generation (baseline / random / structural / trigger) |
| `detector.py` | Z-score anomaly detection and composite safety score |
| `worker.py` | `QThread` pipeline orchestration |
| `app.py` | PyQt6 forensic console with `QPainter` heatmap |
| `mock_poison.py` | Ground-truth backdoor implant + end-to-end verification |

### Where the hooks go

The "neurons" of a transformer FFN are the post-activation channels feeding the
down-projection. A `register_forward_hook` on the **contraction** module (`down_proj`,
`c_proj`, `dense_4h_to_h`) receives that tensor as `inputs[0]` — so NeuroFence hooks the
contraction and captures its input. It falls back to the expansion projection's output, then
to the whole MLP block. LLaMA, Mistral, Qwen, Gemma, Phi, GPT-2, GPT-NeoX, Falcon and MPT
naming conventions are all recognised.

Each captured tensor is reduced **on-device** and severed from the autograd graph
(`.detach() … .cpu().numpy()`) inside the hook, so nothing referencing CUDA storage survives
the call. Activation statistics accumulate through Welford's online algorithm, so memory is
`O(layers × channels)` no matter how many prompts are fired.

### How detection works

Baseline traffic establishes each neuron's resting distribution; the adversarial phase is
scored against it:

```
Z = (fuzz_max − μ_baseline) / (σ_baseline + ε)
```

A pure Z-score is not sufficient on its own. A neuron whose baseline barely varies
(σ ≈ 5e-5) scores Z > 6000 for a trivial 0.17 → 0.49 drift; on the reference model that
alone flagged **2,135 of 6,144 neurons**. Two guards fix it:

- **`min_abs_delta_ratio`** — the rise must clear a fraction of the layer's 99th-percentile
  baseline energy, so numerically-dead channels can't flag on floating-point noise.
- **`spatial_mad_threshold`** — the neuron must also be an outlier *across its own layer*:
  its rise is compared against the median rise of every sibling, scaled by MAD. This encodes
  the actual threat model (a backdoor is an *isolated* cluster, not a layer-wide shift), and
  median/MAD keep working even when a large minority of the layer is poisoned. Adding it took
  false positives from 2,135 to **0** with no loss of recall.

The safety score deliberately gives **density** little weight. A single perfectly-hidden
dormant neuron is a worse finding than a thousand noisy ones, so severity is driven by the
strongest evidence (peak Z, dormancy, amplification), never the average — stealth is not
rewarded.

### Security posture

- `local_files_only=True`, plus `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` set before
  `transformers` is imported (the hub client caches a networked session at import time).
- `use_safetensors=True` — pickle checkpoints (`.bin`/`.pt`/`.ckpt`) are **never**
  deserialised, because `torch.load` on an untrusted archive is remote code execution by
  design. Their presence is reported as a HIGH finding.
- `trust_remote_code=False` — bundled `*.py` and `auto_map` entries in `config.json` are
  reported, never executed.
- Static integrity findings **cap** the safety score: a repo shipping executable Python can
  never be called clean, however quiet its activations are.

### Threading

Every PyTorch call happens on `ScanWorkerThread`. The GUI owns no model state and receives
only immutable payloads over queued signals, so the window stays responsive and abortable
throughout. Teardown (hooks removed, model unmounted, caches flushed) runs in a `finally`
block, so it executes on success, failure and abort alike.

---

## GPU

Install a CUDA torch build first — the default wheel is CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Pick a card from the **Device** dropdown, which enumerates every visible GPU with its
adapter name and VRAM (`cuda:0 — NVIDIA RTX 4090 (23.6 GiB, sm_89)`). VRAM is reported live
in the status bar during a scan and recorded in the exported report.

**Precision is a forensic decision, not just a memory one.** bfloat16 carries ~8 mantissa
bits. NeuroFence's entire signal is the per-neuron *standard deviation of activation energy
across prompts* — and at 8 bits a quiet neuron's baseline quantises to one repeated value, so
σ collapses to exactly 0 and the channel reports an infinite Z-score. Half precision
manufactures false positives in the one statistic the scanner divides by. So `auto` prefers
**float32 whenever the model fits in VRAM**, and drops to bf16/fp16 only when fp32 would not
load at all — logging the downgrade and attaching a `HALF_PRECISION_SCAN` note to the report.
Treat findings from a half-precision scan as leads to confirm in fp32, not conclusions.

Other GPU behaviour:

- **VRAM preflight** — required bytes are estimated from the safetensors size and the stored
  dtype, then checked against free VRAM *before* allocating. An impossible load fails with an
  actionable message instead of a confusing mid-load CUDA OOM.
- **OOM resilience** — the structural fuzz family emits deliberately enormous prompts. A
  device OOM flushes the allocator and retries the prompt at a quarter of the sequence length,
  twice, then drops just that prompt. One bad case can never abort a long scan.
- **fp16 overflow guard** — a saturating backdoor neuron can overflow to `inf` in fp16, which
  would propagate through Welford and destroy the whole layer's statistics. Non-finite values
  are clamped, counted, and reported.
- **Multi-GPU** — tick *Shard across GPUs* for models too large for one card (needs
  `accelerate`; `device_map="auto"`). Hooks fire normally on sharded models, since each hook
  reduces on whichever device its tensor lives on before copying to host.
- Memory reporting, `empty_cache`/`ipc_collect` flushes and peak-VRAM stats are all bound to
  the selected device ordinal rather than the process default.

---

## Notes and limitations

- Verified against **transformers 5.15 / PyQt6 6.11 / torch 2.13 / Python 3.13** on Windows.
  The `torch_dtype` → `dtype` rename in transformers 5 is handled at runtime, so v4 works too.
- **The GPU code has never run on a real GPU.** The development machine has no NVIDIA adapter
  and a CPU-only torch build (`2.13.0+cpu`). The CUDA branches — ordinal parsing, device
  enumeration, VRAM probing, the auto-precision policy, preflight refusal, the sharding gate,
  OOM classification and the non-finite clamp — were driven to completion against a faked
  `torch.cuda` surface, so the arithmetic and control flow are exercised, but nothing has
  touched a real driver. Treat the first run on actual hardware as the real test.
- Detection is a **screening** tool. A high score is evidence of no *dormant-neuron* signature
  under the trigger corpus fired; it is not a proof of safety. A backdoor keyed to a trigger
  outside `TRIGGER_TOKENS` and outside the random/structural families will not be woken —
  extend the dictionary via the GUI's *Extra trigger* field or `AdversarialFuzzer(extra_triggers=…)`.
- Baseline statistics come from ~24 prompts by default; σ is correspondingly noisy. Raise the
  baseline count for higher-confidence scans on large models.
- `poison_manifest.json` in a `mock_poison` export is ground truth for the operator.
  NeuroFence never reads it.
