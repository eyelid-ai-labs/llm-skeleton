# LLM Skeleton

**Probe → Plan → Load.** A universal model loader for multi-GPU LLM deployment.

LLM Skeleton replaces `device_map="auto"` with deterministic, pre-validated loading plans. It reads a model's `config.json` from HuggingFace, computes the exact GPU placement for every layer, and loads with an explicit device map. No surprises, no OOM after 10 minutes of downloading, no silent disk offloading.

## Why This Exists

HuggingFace's `from_pretrained` is a black box. It decides how to shard across GPUs using a greedy heuristic, silently dequantizes FP8 models to bf16 (doubling memory), rejects quantization kwargs on custom model classes, and provides no way to plan for two models simultaneously.

We built LLM Skeleton after spending 14 hours and 20 failed GPU job submissions trying to load two large models on 8x A100 80GB GPUs. Every single failure was predictable from information available in `config.json` before any weights were downloaded. We just weren't checking.

### Failures That Inspired This

| What Happened | Time Wasted | Predictable From |
|---|---|---|
| FP8 model dequantized to bf16, doubled memory, OOM | 2 hours | `quantization_config.quant_method` in config.json |
| Mxfp4 model dequantized to bf16 without Triton >=3.4, OOM on single GPU | 1 hour | `quantization_config.quant_method` in config.json |
| Custom model class rejected `load_in_8bit` kwarg | 3 hours | `auto_map` field in config.json |
| MoE model packed unevenly, 0 headroom on some GPUs | 4 hours | `num_local_experts` + per-layer size variance |
| Model needed Python 3.11, environment had 3.10 | 1 hour | Known model requirements |
| Model needed `mamba-ssm` library, not installed | 1 hour | `model_type` in config.json |
| Two models didn't fit simultaneously, no pre-check | 3 hours | Both models' safetensor index sizes |

Every one of these is now caught in under 1 second by `probe_model()`.

## How It Works

### Phase 1: Probe (Zero VRAM, ~1 second)

Downloads only `config.json` and `model.safetensors.index.json` from HuggingFace Hub. Extracts:

- Model architecture, layer count, hidden dimensions
- MoE configuration (expert count, active experts, routing)
- Native dtype (bf16, FP8, Mxfp4, NVFP4, pre-quantized GPTQ/AWQ)
- Actual weight size from safetensor index (not parameter-count math)
- Custom code detection (auto_map, non-standard model classes)
- Required libraries (mamba-ssm, etc.) and Python version
- Per-layer size estimates for bin-packing

```python
from llm_skeleton import probe_model

profile = probe_model("Qwen/Qwen3-Coder-Next")
print(profile.summary())
# Model: Qwen/Qwen3-Coder-Next
#   Type: qwen3_next (Qwen3NextForCausalLM)
#   Params: 774.2B total, 15.7B active
#   MoE: 512 experts, 10 active
#   Size: bf16=148.4GB, int8=74.2GB, int4=37.1GB
```

### Phase 2: Plan (Zero VRAM, instant)

Given probe results and available hardware, computes the optimal loading strategy:

1. Tries bf16 first (best quality)
2. Falls back to INT8 via `BitsAndBytesConfig` (not the `load_in_8bit` kwarg that custom models reject)
3. Falls back to INT4 if needed
4. For each strategy, runs bin-packing to compute explicit per-layer GPU placement
5. Ensures headroom on every GPU (not just total)
6. Fails fast with a clear message if nothing fits

```python
from llm_skeleton import plan_loading, detect_gpus

hardware = detect_gpus()
plan = plan_loading(profile, hardware, gpu_indices=[0,1,2], headroom_gb=5.0)

if plan.can_load:
    print(plan.summary())
    # Strategy: none (bfloat16)
    # GPU 0: embed, layers 0-18 (67.6GB / 75GB, 2.7GB free)
    # GPU 1: layers 19-37 (67.5GB / 75GB, 2.8GB free)
    # GPU 2: layers 38-47, lm_head (35.6GB / 75GB, 34.7GB free)
else:
    print(f"Won't fit: {plan.failure_reason}")
```

### Phase 3: Load (Explicit device map)

Executes the plan using the computed device map. Never `device_map="auto"`.

```python
from llm_skeleton import execute_plan

model, tokenizer = execute_plan(plan)
```

## Dual-Model Loading

The killer feature. Load two models simultaneously with automatic GPU split optimization.

```python
from llm_skeleton import DualModelOrchestrator

orch = DualModelOrchestrator(
    target_model="Qwen/Qwen3-Coder-Next",        # 148GB bf16
    reasoning_model="mistralai/Devstral-2-123B",   # 239GB bf16 (FP8 dequantized)
    free_gpus=1,  # Reserve 1 GPU for other work
)

# Probes both models, binary-searches optimal GPU split
dual_plan = orch.plan()
print(dual_plan.summary())
# Target GPUs:    [0, 1, 2]      — 148GB on 3 GPUs
# Reasoning GPUs: [3, 4, 5, 6]   — 239GB on 4 GPUs
# Free GPUs:      [7]
# ✅ Both models fit. Ready to load.

# Load both
target, t_tok, reasoning, r_tok = orch.load_both()
```

The orchestrator tries every possible split (1/6, 2/5, 3/4, 4/3, etc.) and picks the best one — preferring no quantization on the target model. It found a 3/4 split that humans wouldn't have tried (we were hardcoding 4/3 and failing).

## What It Catches

### FP8 Pre-Quantized Models
Models like Devstral-2-123B ship as FP8 but dequantize to bf16 on A100 GPUs (compute capability 8.0 < 8.9 required for native FP8). LLM Skeleton detects this and plans for the dequantized size — 2x the disk size.

It also prevents you from applying BitsAndBytes quantization on top of a pre-quantized model, which crashes with `FineGrainedFP8Config vs BitsAndBytesConfig conflict`.

### Mxfp4 Pre-Quantized Models
Models like openai/gpt-oss-120b ship with Microsoft's MX FP4 quantization (`quant_method: "mxfp4"` in config.json). Without Triton >= 3.4.0 (which requires torch >= 2.7), these models silently dequantize to bf16 at load time — increasing VRAM usage by ~1.3× over the safetensor disk size.

LLM Skeleton detects Mxfp4 via the `is_mxfp4` flag on `ModelProfile` and applies a 1.3× dequantization multiplier to all layer size estimates during planning. This ensures the bin-packer allocates enough GPU memory for the dequantized weights, preventing the OOM that `device_map="auto"` hits when it plans based on disk size.

```python
profile = probe_model("openai/gpt-oss-120b")
print(profile.is_mxfp4)  # True
print(profile.summary())
# Native dtype: mxfp4 ⚠️ MXFP4 dequantizes to bf16 without Triton >=3.4
# Size: bf16=60.8GB (disk), ~79GB (VRAM after dequantization)
```

### Custom Model Classes
Models like `Ministral3ForCausalLM` and `NemotronHForCausalLM` reject the `load_in_8bit=True` kwarg in their `__init__`. LLM Skeleton always uses `BitsAndBytesConfig` objects instead, which go through HuggingFace's loading pipeline and work with all model classes.

### MoE Layer Size Variance
In MoE models, layers are not equal size. A 512-expert layer can be 10x larger than a dense attention layer. HuggingFace's greedy packing fills early GPUs with huge layers, leaving no headroom. LLM Skeleton's bin-packer places layers with contiguity constraints and guaranteed per-GPU headroom.

### Dependency and Compatibility Issues
- Python version requirements (MiniMax needs 3.11+)
- Missing libraries (Nemotron needs mamba-ssm)
- NVFP4 models that need Blackwell GPUs
- All caught before downloading a single weight file

## Use Cases

### Research: Dual-Model Analysis
Load a target model and a reasoning model simultaneously for mechanistic interpretability, red-teaming, or model comparison. The orchestrator finds the optimal GPU split so you don't waste hours on manual partitioning.

### Inference: Multi-Tenant GPU Sharing
Run multiple models on a shared GPU cluster where each model gets a specific GPU allocation. LLM Skeleton's explicit device maps prevent models from stepping on each other's VRAM — no more mysterious OOM when the second model loads.

### MLOps: Pre-Flight Validation
Before submitting an expensive cloud GPU job ($30+/hour for 8x A100), run `probe_model()` locally to verify the model fits. Catches FP8 dequantization surprises, missing dependencies, and Python version mismatches in 1 second instead of after 15 minutes of downloading weights.

### Fine-Tuning: Guaranteed Gradient Headroom
When fine-tuning on multi-GPU, you need VRAM headroom for optimizer states and gradient buffers. LLM Skeleton's per-GPU headroom guarantee ensures every GPU has room for backpropagation, not just enough for the forward pass.

### Edge Deployment: Fit-or-Fail Decisions
Deploying to a fixed hardware target (e.g. 2x RTX 4090)? `plan_loading()` tells you instantly whether the model fits in bf16, needs INT8, or needs INT4 — and exactly which layers go on which GPU. No trial-and-error.

### CI/CD: Automated Model Compatibility Testing
Add `probe_model()` + `check_compatibility()` to your CI pipeline to catch breaking changes when model authors update their configs. Detects new dependencies, dtype changes, and architecture modifications before they hit production.

## Installation

```bash
pip install llm-skeleton
```

Or from source:
```bash
git clone https://github.com/eyelid-ai-labs/llm-skeleton.git
cd llm-skeleton
pip install -e .
```

Dependencies: `transformers`, `torch`, `huggingface_hub`. Optional: `psutil` for RAM detection.

## Testing

```bash
pytest tests/ -v
```

62 tests covering bin-packing (contiguity, headroom, MoE variance, quantization, VLM paths, edge cases), probe (dtype detection, MoE detection, custom code, library requirements, VLM config resolution, VLM weight map detection), and planning (strategy selection, GPU subsets, load kwargs). All tests run offline — no GPU or HuggingFace access needed.

## Validated On

Tested across 8 Azure ML job submissions on 8x NVIDIA A100 80GB NVLink (634GB total VRAM):

| Model Pair | Total VRAM | Split | Result |
|---|---|---|---|
| Qwen3-Coder-Next (148GB) + Devstral-2-123B (239GB dequantized) | 387GB | 3/4 GPUs | ✅ Both loaded, no disk offload |
| Qwen3-Coder-Next (148GB) alone | 148GB | 3 GPUs | ✅ bf16, no quantization |
| Devstral-2-123B on 3 GPUs (213GB available) | 239GB | — | ✅ Correctly rejected (doesn't fit) |
| openai/gpt-oss-120b (60.8GB disk, ~79GB dequantized Mxfp4→bf16) | ~79GB | 2 GPUs | ✅ Mxfp4 dequantization detected, planned for 1.3× disk |

## Architecture

```
llm_skeleton/
    __init__.py          # Public API
    probe.py             # Phase 1: config.json parsing, safetensor index sizes
    plan.py              # Phase 2: strategy cascade (bf16 → INT8 → INT4)
    load.py              # Phase 3: explicit device_map execution
    bin_packing.py       # Layer-to-GPU placement with contiguity + headroom
    orchestrator.py      # DualModelOrchestrator for multi-model loading
    hardware.py          # GPU detection (VRAM, compute capability)
    compatibility.py     # Python version, library, dtype checks
    tests/               # 46 offline tests
```

## Rules Encoded From Failure

1. **Probe before download.** Every compatibility issue is in config.json.
2. **Explicit device maps only.** Never `device_map="auto"`.
3. **Per-layer sizing for MoE.** Layers are NOT equal size.
4. **`BitsAndBytesConfig`, not `load_in_8bit`.** Custom model classes reject the kwarg.
5. **FP8 on A100 = 2x size.** Plan for dequantized bf16, not disk size.
6. **Mxfp4 without Triton >=3.4 = ~1.3x size.** Same principle as FP8 — plan for VRAM, not disk.
7. **Headroom is per-GPU, not total.** 200GB total free means nothing if every GPU has 0 free.
8. **Fail fast.** If it won't fit, say so in 1 second, not after 10 minutes of downloading.

## License

MIT
