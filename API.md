# API Reference

## Quick Start

```python
from llm_skeleton import probe_model, plan_loading, execute_plan, detect_gpus

profile = probe_model("Qwen/Qwen3-Coder-Next")
hardware = detect_gpus()
plan = plan_loading(profile, hardware, gpu_indices=[0, 1, 2], headroom_gb=5.0)

if plan.can_load:
    model, tokenizer = execute_plan(plan)
```

---

## Phase 1: Probe

### `probe_model(model_name, token=None) → ModelProfile`

Download only `config.json` and `model.safetensors.index.json` from HuggingFace Hub.
Zero VRAM, ~1 second. Returns everything needed to plan loading.

| Parameter | Type | Description |
|---|---|---|
| `model_name` | `str` | HuggingFace model ID (e.g. `"Qwen/Qwen3-Coder-Next"`) |
| `token` | `str \| None` | HF token for gated models |

**Raises:** `ValueError` if `config.json` cannot be fetched.

### `ModelProfile`

Dataclass with all model metadata. Key fields:

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | HuggingFace model ID |
| `model_type` | `str` | Architecture family (`"llama"`, `"qwen3_moe"`, etc.) |
| `architecture_class` | `str` | HF class name (`"Qwen3MoeForCausalLM"`) |
| `num_parameters` | `int` | Total parameter count (all experts for MoE) |
| `num_active_parameters` | `int` | Active params per forward pass |
| `num_layers` | `int` | Transformer layer count |
| `hidden_size` | `int` | Hidden dimension |
| `intermediate_size` | `int` | FFN intermediate dimension |
| `vocab_size` | `int` | Vocabulary size |
| `num_attention_heads` | `int` | Query head count |
| `num_key_value_heads` | `int` | KV head count (GQA) |
| `is_moe` | `bool` | Whether model uses Mixture of Experts |
| `num_experts` | `int` | Total expert count |
| `num_active_experts` | `int` | Experts active per token |
| `moe_layer_frequency` | `int` | Every Nth layer is MoE (1 = all) |
| `shared_expert` | `bool` | Has a shared/always-on expert |
| `native_dtype` | `NativeDtype` | Storage dtype enum |
| `is_fp8` | `bool` | FP8 model (dequantizes to bf16 on A100) |
| `is_mxfp4` | `bool` | Mxfp4 model (dequantizes without Triton ≥3.4) |
| `is_nvfp4` | `bool` | NVFP4 model (requires Blackwell) |
| `size_bf16` | `int` | Estimated bf16 size in bytes |
| `size_int8` | `int` | Estimated INT8 size in bytes |
| `size_int4` | `int` | Estimated INT4 size in bytes |
| `size_native` | `int` | Size in native dtype (bytes) |
| `layer_profiles` | `list[LayerProfile]` | Per-layer size estimates |
| `embedding_size_bytes` | `int` | embed_tokens + lm_head size |
| `uses_custom_code` | `bool` | Requires `trust_remote_code=True` |
| `custom_model_class` | `str` | Custom class name if applicable |
| `auto_map` | `dict` | Raw auto_map from config.json |
| `required_libraries` | `list[str]` | Extra pip packages needed |
| `min_python_version` | `str \| None` | Minimum Python version |
| `supports_bnb_quantization` | `bool` | Safe for BitsAndBytes |
| `is_pre_quantized` | `bool` | Ships with quantization_config |
| `pre_quantization_method` | `str` | `"fp8"`, `"gptq"`, `"awq"`, etc. |
| `layer_prefix` | `str` | Actual layer path prefix (e.g. `"model.language_model.layers"`) |
| `embed_module` | `str` | Actual embed_tokens module path |
| `lm_head_module` | `str` | Actual lm_head module path |
| `norm_module` | `str` | Actual final norm module path |
| `extra_modules` | `list[str]` | Extra modules (vision_tower, etc.) |
| `raw_config` | `dict` | Full config.json for debugging |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `size_bf16_gb` | `float` | bf16 size in GB |
| `size_int8_gb` | `float` | INT8 size in GB |
| `size_int4_gb` | `float` | INT4 size in GB |
| `largest_layer_bf16_gb` | `float` | Largest single layer in GB |

**Methods:**

- `summary() → str` — Human-readable model summary.

### `LayerProfile`

Per-layer size estimate dataclass.

| Field | Type | Description |
|---|---|---|
| `index` | `int` | Layer index |
| `size_bf16_bytes` | `int` | bf16 size in bytes |
| `size_int8_bytes` | `int` | INT8 size in bytes |
| `size_int4_bytes` | `int` | INT4 size in bytes |
| `is_moe_layer` | `bool` | Whether this layer has experts |
| `num_experts` | `int` | Expert count for this layer |

### `NativeDtype`

Enum of model storage dtypes.

| Value | Description |
|---|---|
| `BF16` | bfloat16 |
| `FP16` | float16 |
| `FP32` | float32 |
| `FP8` | float8 (dequantizes to bf16 on A100) |
| `MXFP4` | Microsoft MX FP4 (dequantizes without Triton ≥3.4) |
| `NVFP4` | NVIDIA FP4 (requires Blackwell) |
| `INT8` | Pre-quantized 8-bit |
| `INT4` | Pre-quantized 4-bit (GPTQ/AWQ) |
| `UNKNOWN` | Could not determine |

---

## Phase 2: Plan

### `plan_loading(profile, hardware, ...) → LoadingPlan`

Compute optimal loading strategy. Zero VRAM — pure computation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `profile` | `ModelProfile` | required | Probe results |
| `hardware` | `HardwareProfile` | required | Available hardware |
| `gpu_indices` | `list[int] \| None` | `None` (all) | Which GPUs to use |
| `headroom_gb` | `float` | `5.0` | Reserved VRAM per GPU |
| `prefer_quantization` | `QuantizationMethod \| None` | `None` | Force a strategy |
| `allow_quantization` | `bool` | `True` | Allow INT8/INT4 fallback |

Tries strategies in order: bf16 → INT8 → INT4. For pre-quantized models (FP8, GPTQ,
AWQ), only tries native loading (no BNB on top). Applies dequantization multipliers
for FP8 (2×) and Mxfp4 (1.3×) on incompatible hardware.

### `LoadingPlan`

| Field | Type | Description |
|---|---|---|
| `profile` | `ModelProfile` | The model being planned |
| `hardware` | `HardwareProfile` | Hardware used for planning |
| `compatibility` | `CompatibilityReport` | Pre-flight check results |
| `strategy` | `LoadingStrategy \| None` | Chosen strategy (None if impossible) |
| `gpu_indices` | `list[int]` | GPUs allocated |
| `headroom_gb` | `float` | Per-GPU headroom |
| `strategies_tried` | `list[tuple]` | All strategies attempted with results |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `can_load` | `bool` | Whether loading is possible |
| `failure_reason` | `str` | Why it can't load (empty if it can) |

**Methods:**

- `summary() → str` — Human-readable plan summary.
- `get_load_kwargs() → dict` — Build kwargs for `from_pretrained`. Includes `device_map`, `max_memory`, `quantization_config` (if needed), `dtype`, and `trust_remote_code`.

### `LoadingStrategy`

| Field | Type | Description |
|---|---|---|
| `quantization` | `QuantizationMethod` | Quantization applied |
| `dtype_str` | `str` | `"bfloat16"` or `"float16"` |
| `estimated_vram_gb` | `float` | Estimated total VRAM usage |
| `placement` | `PlacementResult` | Bin-packing result |

**Properties:** `device_map`, `max_memory` (delegated to placement).

### `QuantizationMethod`

| Value | Description |
|---|---|
| `NONE` | bf16/fp16, no quantization |
| `BNB_INT8` | `BitsAndBytesConfig(load_in_8bit=True)` |
| `BNB_INT4` | `BitsAndBytesConfig(load_in_4bit=True)` |

---

## Phase 3: Load

### `execute_plan(plan, ...) → (model, tokenizer)`

Execute a loading plan with explicit device map. Never `device_map="auto"`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `plan` | `LoadingPlan` | required | The computed plan |
| `load_tokenizer` | `bool` | `True` | Also load tokenizer |
| `validate_vram` | `bool` | `True` | Check actual vs planned VRAM |
| `vram_tolerance` | `float` | `0.30` | Acceptable VRAM deviation (30%) |

**Returns:** `(model, tokenizer)` tuple. `tokenizer` is `None` if `load_tokenizer=False`.

**Raises:** `RuntimeError` if plan says model can't be loaded.

---

## Hardware Detection

### `detect_gpus() → HardwareProfile`

Detect all GPUs and system info. Zero VRAM cost.

### `HardwareProfile`

| Field | Type | Description |
|---|---|---|
| `gpus` | `list[GPUInfo]` | All detected GPUs |
| `total_ram_bytes` | `int` | System RAM (requires psutil) |
| `python_version` | `str` | Current Python version |
| `cuda_version` | `str \| None` | CUDA version |

**Properties:** `num_gpus`, `total_vram_gb`, `total_free_vram_gb`, `total_ram_gb`, `min_gpu_vram_gb`.

**Methods:**

- `gpu_subset(indices) → list[GPUInfo]` — Get GPUs by index.
- `subset_vram_gb(indices) → float` — Total VRAM for a subset.

### `GPUInfo`

| Field | Type | Description |
|---|---|---|
| `index` | `int` | GPU index |
| `name` | `str` | GPU name (e.g. `"NVIDIA A100-SXM4-80GB"`) |
| `total_vram_bytes` | `int` | Total VRAM |
| `free_vram_bytes` | `int` | Free VRAM at detection time |
| `compute_capability` | `tuple` | e.g. `(8, 0)` for A100 |

**Properties:** `total_vram_gb`, `free_vram_gb`, `supports_fp8_natively` (≥8.9), `supports_nvfp4` (≥10.0).

---

## Compatibility Checking

### `check_compatibility(profile, hardware=None, target_gpus=None) → CompatibilityReport`

Pre-flight validation. Catches Python version mismatches, missing libraries, FP8/NVFP4
GPU requirements, and custom code warnings.

### `CompatibilityReport`

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Model being checked |
| `issues` | `list[CompatibilityIssue]` | All detected issues |

**Properties:** `has_errors`, `has_warnings`, `can_load`.

### `CompatibilityIssue`

| Field | Type | Description |
|---|---|---|
| `severity` | `str` | `"error"` (blocks loading) or `"warning"` |
| `category` | `str` | `"python"`, `"library"`, `"gpu"`, `"dtype"`, `"quantization"` |
| `message` | `str` | Human-readable description |
| `suggestion` | `str` | How to fix it |

---

## Dual-Model Orchestrator

### `DualModelOrchestrator`

Load two models simultaneously with automatic GPU split optimization.

```python
orch = DualModelOrchestrator(
    target_model="Qwen/Qwen3-Coder-Next",
    reasoning_model="mistralai/Devstral-2-123B",
    free_gpus=1,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_model` | `str` | required | HF model ID for target |
| `reasoning_model` | `str` | required | HF model ID for reasoning |
| `num_gpus` | `int \| None` | `None` | Override GPU count |
| `gradient_headroom_gb` | `float` | `5.0` | Per-GPU headroom for target |
| `reasoning_headroom_gb` | `float` | `2.0` | Per-GPU headroom for reasoning |
| `free_gpus` | `int` | `1` | GPUs to leave completely free |
| `target_quantization` | `QuantizationMethod \| None` | `None` | Force target quantization |
| `reasoning_quantization` | `QuantizationMethod \| None` | `None` | Force reasoning quantization |
| `hf_token` | `str \| None` | `None` | HF token for gated models |

**Methods:**

- `probe() → (ModelProfile, ModelProfile)` — Probe both models.
- `plan() → DualModelPlan` — Compute optimal GPU split. Tries every possible split and picks the one that minimizes target quantization.
- `load_both() → (target_model, target_tokenizer, reasoning_model, reasoning_tokenizer)` — Load both models.
- `load_target_only() → (model, tokenizer)` — Load just the target on all available GPUs.

### `DualModelPlan`

| Field | Type | Description |
|---|---|---|
| `target_plan` | `LoadingPlan \| None` | Plan for target model |
| `reasoning_plan` | `LoadingPlan \| None` | Plan for reasoning model |
| `target_gpus` | `list[int]` | GPUs allocated to target |
| `reasoning_gpus` | `list[int]` | GPUs allocated to reasoning |
| `free_gpus` | `list[int]` | Reserved GPUs |

**Properties:** `can_load`, `failure_reason`.

---

## Bin-Packing (Advanced)

These are used internally by `plan_loading` but are available for custom placement logic.

### `pack_layers(layer_profiles, embedding_size_bytes, gpu_capacities_bytes, ...) → PlacementResult`

Place transformer layers onto GPUs with contiguity constraint and per-GPU headroom.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layer_profiles` | `list[LayerProfile]` | required | Per-layer sizes |
| `embedding_size_bytes` | `int` | required | embed_tokens + lm_head size |
| `gpu_capacities_bytes` | `list[tuple[int, int]]` | required | `[(gpu_idx, capacity)]` |
| `headroom_bytes` | `int` | `5 * 1024³` | Per-GPU reserved bytes |
| `model_prefix` | `str` | `"model"` | Legacy fallback prefix |
| `layer_prefix` | `str` | `""` | Actual layer prefix from safetensors |
| `embed_module` | `str` | `""` | Actual embed module path |
| `lm_head_module` | `str` | `""` | Actual lm_head path |
| `norm_module` | `str` | `""` | Actual norm path |
| `extra_modules` | `list[str] \| None` | `None` | Extra modules to place |
| `extra_modules_size_bytes` | `int` | `0` | Size of extra modules |

### `pack_layers_quantized(...) → PlacementResult`

Same as `pack_layers` but adjusts layer sizes for INT8 or INT4 quantization.
Accepts all the same parameters plus `quantization` (`"int8"` or `"int4"`).

### `PlacementResult`

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether placement succeeded |
| `device_map` | `dict[str, int]` | Module path → GPU index |
| `gpu_allocations` | `list[GPUAllocation]` | Per-GPU allocation details |
| `max_memory` | `dict` | For `from_pretrained` |
| `failure_reason` | `str` | Why it failed (empty on success) |
| `total_model_bytes` | `int` | Total model size placed |
