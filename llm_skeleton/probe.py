"""
Phase 1: Model Probing.

Read config.json from HuggingFace without downloading weights.
Zero VRAM, ~1 second. Extracts everything needed to plan loading.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class NativeDtype(Enum):
    """Model's native storage dtype."""
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"
    FP8 = "float8"       # E4M3/E5M2 — dequantizes to bf16 on A100
    MXFP4 = "mxfp4"      # Microsoft MX FP4 — dequantizes to bf16 without Triton >=3.4
    NVFP4 = "nvfp4"      # Requires Blackwell
    INT8 = "int8"         # Pre-quantized
    INT4 = "int4"         # Pre-quantized (GPTQ/AWQ)
    UNKNOWN = "unknown"


@dataclass
class LayerProfile:
    """Size estimate for a single transformer layer."""
    index: int
    size_bf16_bytes: int
    size_int8_bytes: int
    size_int4_bytes: int
    is_moe_layer: bool = False
    num_experts: int = 0


@dataclass
class ModelProfile:
    """Everything we know about a model before downloading weights."""
    # Identity
    model_name: str
    model_type: str = ""                    # "qwen2_moe", "llama", "mistral", etc.
    architecture_class: str = ""            # "Qwen2MoeForCausalLM", etc.
    
    # Size
    num_parameters: int = 0                 # Total params (all experts)
    num_active_parameters: int = 0          # Active params per forward pass (MoE)
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    
    # MoE
    is_moe: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    moe_layer_frequency: int = 1            # Every Nth layer is MoE (1 = all)
    shared_expert: bool = False             # Some MoE models have a shared expert
    
    # Dtype
    native_dtype: NativeDtype = NativeDtype.BF16
    is_fp8: bool = False                    # Will dequantize to bf16 on A100
    is_mxfp4: bool = False                  # Will dequantize to bf16 without Triton >=3.4
    is_nvfp4: bool = False                  # Requires Blackwell
    
    # Size estimates (bytes)
    size_bf16: int = 0
    size_int8: int = 0
    size_int4: int = 0
    size_native: int = 0
    
    # Per-layer sizes (critical for MoE bin-packing)
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    embedding_size_bytes: int = 0           # embed_tokens + lm_head
    
    # Custom code
    uses_custom_code: bool = False
    custom_model_class: str = ""            # e.g. "Ministral3ForCausalLM"
    auto_map: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    required_libraries: List[str] = field(default_factory=list)
    min_python_version: Optional[str] = None
    
    # Compatibility flags
    supports_bnb_quantization: bool = True  # False for some custom classes
    supports_attention_output: bool = True
    
    # Pre-quantized model (cannot apply BNB on top)
    is_pre_quantized: bool = False          # True if model ships with quantization_config
    pre_quantization_method: str = ""       # "fp8", "gptq", "awq", etc.
    
    # Raw config for debugging
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_bf16_gb(self) -> float:
        return self.size_bf16 / (1024**3)
    
    @property
    def size_int8_gb(self) -> float:
        return self.size_int8 / (1024**3)
    
    @property
    def size_int4_gb(self) -> float:
        return self.size_int4 / (1024**3)
    
    @property
    def largest_layer_bf16_gb(self) -> float:
        if not self.layer_profiles:
            return 0.0
        return max(lp.size_bf16_bytes for lp in self.layer_profiles) / (1024**3)
    
    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"  Type: {self.model_type} ({self.architecture_class})",
            f"  Params: {self.num_parameters / 1e9:.1f}B total"
            + (f", {self.num_active_parameters / 1e9:.1f}B active" if self.is_moe else ""),
            f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}",
        ]
        if self.is_moe:
            lines.append(f"  MoE: {self.num_experts} experts, {self.num_active_experts} active")
        lines.extend([
            f"  Native dtype: {self.native_dtype.value}"
            + (" ⚠️ FP8 dequantizes to bf16 on A100" if self.is_fp8 else "")
            + (" ⚠️ MXFP4 dequantizes to bf16 without Triton >=3.4" if self.native_dtype == NativeDtype.MXFP4 else "")
            + (" ⚠️ NVFP4 requires Blackwell" if self.is_nvfp4 else "")
            + (f" (pre-quantized {self.pre_quantization_method}, no BNB)" if self.is_pre_quantized else ""),
            f"  Size: bf16={self.size_bf16_gb:.1f}GB, int8={self.size_int8_gb:.1f}GB, int4={self.size_int4_gb:.1f}GB",
        ])
        if self.uses_custom_code:
            lines.append(f"  ⚠️ Custom code: {self.custom_model_class}")
        if self.required_libraries:
            lines.append(f"  Required libs: {', '.join(self.required_libraries)}")
        if self.min_python_version:
            lines.append(f"  Min Python: {self.min_python_version}")
        if self.layer_profiles:
            lines.append(f"  Largest layer (bf16): {self.largest_layer_bf16_gb:.2f}GB")
        return "\n".join(lines)


# ─── Known model quirks database ───────────────────────────────────────────

# Models that need specific libraries beyond transformers+torch
KNOWN_LIBRARY_REQUIREMENTS = {
    "nemotron": ["mamba-ssm", "causal-conv1d"],
    "jamba": ["mamba-ssm"],
    "mamba": ["mamba-ssm"],
    "rwkv": ["rwkv"],
    "recurrentgemma": [],
}

# Models known to require Python 3.11+ (typing.Unpack, etc.)
KNOWN_PYTHON_REQUIREMENTS = {
    "minimax": "3.11",
}

# Architecture classes known to reject load_in_8bit kwarg
KNOWN_BNB_INCOMPATIBLE = {
    "Ministral3ForCausalLM",
    "NemotronHForCausalLM",
    "MiniMaxM2ForCausalLM",
    "JambaForCausalLM",
}

# Standard HF architecture classes (safe for all loading methods)
STANDARD_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "GPTNeoXForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "StableLmForCausalLM",
    "FalconForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
}

# Common nested language-config keys used by VLM families.
# These carry the decoder/LLM dimensions we need for sizing.
NESTED_LANGUAGE_CONFIG_KEYS = (
    "text_config",      # Gemma-4 style
    "language_config",  # LLaVA-style
    "llm_config",       # InternVL-style
)


def _sizing_signal_score(config: dict) -> int:
    """Score how likely a config dict is to contain decoder sizing fields."""
    if not isinstance(config, dict):
        return 0
    signal_keys = (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "vocab_size",
    )
    return sum(1 for key in signal_keys if config.get(key) not in (None, 0))


def _resolve_effective_config(config: dict) -> dict:
    """Resolve which config block should drive language-model sizing.

    Most pure LLMs store sizing at top level. Many VLMs store decoder sizing
    under a nested key such as text_config/language_config/llm_config.

    Strategy:
    - Prefer a nested language config when top-level is missing layer info.
    - Otherwise prefer whichever block has stronger sizing signals.
    - Merge selected nested block onto top-level so unrelated metadata remains.
    """
    top_score = _sizing_signal_score(config)
    best_key = None
    best_nested = None
    best_score = top_score

    for key in NESTED_LANGUAGE_CONFIG_KEYS:
        nested = config.get(key)
        if not isinstance(nested, dict):
            continue

        nested_score = _sizing_signal_score(nested)
        if nested_score == 0:
            continue

        top_missing_layers = config.get("num_hidden_layers") in (None, 0)
        nested_has_layers = nested.get("num_hidden_layers") not in (None, 0)

        should_replace = False
        if top_missing_layers and nested_has_layers:
            should_replace = True
        elif nested_score > best_score:
            should_replace = True

        if should_replace:
            best_key = key
            best_nested = nested
            best_score = nested_score

    if best_key and best_nested:
        logger.info(f"VLM detected — using {best_key} for layer sizing")
        return {**config, **best_nested}

    return config


def _detect_dtype(config: dict) -> NativeDtype:
    """Detect native dtype from config.json fields."""
    dtype_str = config.get("torch_dtype", "")
    quantization_config = config.get("quantization_config", {})
    
    # Check for pre-quantized models
    if quantization_config:
        quant_method = quantization_config.get("quant_method", "")
        bits = quantization_config.get("bits", 0)
        if quant_method in ("gptq", "awq") and bits == 4:
            return NativeDtype.INT4
        if quant_method in ("gptq", "awq") and bits == 8:
            return NativeDtype.INT8
        # FP8 pre-quantized (e.g. Devstral-2-123B with FineGrainedFP8Config)
        if quant_method == "fp8":
            return NativeDtype.FP8
        # Mxfp4 pre-quantized (e.g. openai/gpt-oss-120b)
        # Dequantizes to bf16 without Triton >= 3.4 on CUDA
        if quant_method == "mxfp4" or "mxfp4" in str(quantization_config.get("quant_type", "")).lower():
            return NativeDtype.MXFP4
    
    # Check dtype string
    if "float8" in str(dtype_str).lower() or "fp8" in str(dtype_str).lower():
        return NativeDtype.FP8
    if "nvfp4" in str(dtype_str).lower():
        return NativeDtype.NVFP4
    if dtype_str in ("bfloat16", "torch.bfloat16"):
        return NativeDtype.BF16
    if dtype_str in ("float16", "torch.float16"):
        return NativeDtype.FP16
    if dtype_str in ("float32", "torch.float32"):
        return NativeDtype.FP32
    
    return NativeDtype.BF16  # Default assumption for modern models


def _detect_moe(config: dict) -> dict:
    """Extract MoE configuration from config.json."""
    # Different models use different field names
    num_experts = (
        config.get("num_local_experts", 0) or
        config.get("num_experts", 0) or
        config.get("n_routed_experts", 0) or
        config.get("num_experts_per_tok", 0) or  # DeepSeek
        0
    )
    num_active = (
        config.get("num_experts_per_tok", 0) or
        config.get("num_selected_experts", 0) or
        config.get("num_activated_experts", 0) or
        config.get("top_k", 0) or
        0
    )
    
    # MoE layer frequency (some models alternate dense/MoE layers)
    moe_freq = config.get("moe_layer_frequency", 1)
    if not num_experts and "moe" in config.get("model_type", "").lower():
        # Fallback: model_type says MoE but fields are named differently
        num_experts = 1
    
    shared_expert = config.get("shared_expert", False) or config.get("n_shared_experts", 0) > 0
    
    return {
        "is_moe": num_experts > 1,
        "num_experts": num_experts,
        "num_active_experts": num_active or (2 if num_experts > 1 else 0),
        "moe_layer_frequency": moe_freq,
        "shared_expert": shared_expert,
    }


def _estimate_layer_size(config: dict, is_moe_layer: bool, num_experts: int,
                         shared_expert: bool, bytes_per_param: int = 2) -> int:
    """Estimate size of a single transformer layer in bytes."""
    hidden = config.get("hidden_size", 0)
    intermediate = config.get("intermediate_size", 0)
    num_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads if num_heads else 0
    
    # Attention: Q, K, V, O projections
    attn_params = (
        hidden * (num_heads * head_dim) +      # Q
        hidden * (num_kv_heads * head_dim) +    # K
        hidden * (num_kv_heads * head_dim) +    # V
        (num_heads * head_dim) * hidden          # O
    )
    
    # FFN / MoE
    if is_moe_layer and num_experts > 1:
        # Each expert has gate_proj, up_proj, down_proj
        expert_params = num_experts * (
            hidden * intermediate +   # gate_proj
            hidden * intermediate +   # up_proj
            intermediate * hidden     # down_proj
        )
        # Router/gate
        router_params = hidden * num_experts
        # Shared expert (if present)
        shared_params = 0
        if shared_expert:
            shared_params = (
                hidden * intermediate +
                hidden * intermediate +
                intermediate * hidden
            )
        ffn_params = expert_params + router_params + shared_params
    else:
        # Dense FFN
        ffn_params = (
            hidden * intermediate +   # gate_proj / fc1
            hidden * intermediate +   # up_proj / fc2
            intermediate * hidden     # down_proj / fc3
        )
    
    # LayerNorm (small, but count it)
    norm_params = hidden * 4  # 2 norms, each with weight + bias
    
    total_params = attn_params + ffn_params + norm_params
    return total_params * bytes_per_param


def _estimate_embedding_size(config: dict, bytes_per_param: int = 2) -> int:
    """Estimate embedding + lm_head size."""
    vocab = config.get("vocab_size", 0)
    hidden = config.get("hidden_size", 0)
    tie_embeddings = config.get("tie_word_embeddings", True)
    
    embed_params = vocab * hidden  # embed_tokens
    if not tie_embeddings:
        embed_params += vocab * hidden  # separate lm_head
    # Final layernorm
    embed_params += hidden * 2
    
    return embed_params * bytes_per_param


def _detect_required_libraries(config: dict, model_name: str) -> List[str]:
    """Detect required libraries from model type and config."""
    libs = []
    model_type = config.get("model_type", "").lower()
    
    for key, required in KNOWN_LIBRARY_REQUIREMENTS.items():
        if key in model_type or key in model_name.lower():
            libs.extend(required)
    
    # Check auto_map for custom modeling code hints
    auto_map = config.get("auto_map", {})
    if auto_map:
        for cls_path in auto_map.values():
            if isinstance(cls_path, str):
                # e.g. "modeling_minimax_m2.MiniMaxM2ForCausalLM"
                if "mamba" in cls_path.lower():
                    if "mamba-ssm" not in libs:
                        libs.append("mamba-ssm")
    
    return list(set(libs))


def _detect_python_version(config: dict, model_name: str) -> Optional[str]:
    """Detect minimum Python version requirement."""
    for key, version in KNOWN_PYTHON_REQUIREMENTS.items():
        if key in model_name.lower():
            return version
    return None


def _detect_custom_code(config: dict) -> dict:
    """Detect if model uses custom code and which class."""
    auto_map = config.get("auto_map", {})
    if not auto_map:
        return {"uses_custom_code": False, "custom_model_class": "", "auto_map": {}}
    
    # Extract the model class name
    model_cls = auto_map.get("AutoModelForCausalLM", "")
    if "--" in model_cls:
        # Format: "org/repo--modeling_file.ClassName"
        model_cls = model_cls.split("--")[-1]
    if "." in model_cls:
        model_cls = model_cls.split(".")[-1]
    
    return {
        "uses_custom_code": True,
        "custom_model_class": model_cls,
        "auto_map": auto_map,
    }


def _fetch_actual_size(model_name: str, token: Optional[str] = None) -> Optional[int]:
    """Fetch actual model size from safetensors index metadata.
    
    The index.json has a 'total_size' field that gives the exact byte count
    of all weight tensors. This is far more accurate than parameter-count math,
    especially for MoE models where expert dimensions vary.
    
    Returns None if index is not available.
    """
    try:
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors.index.json",
            token=token,
        )
        with open(index_path) as f:
            index = json.load(f)
        total_size = index.get("metadata", {}).get("total_size", None)
        if total_size:
            logger.info(f"Actual weight size from safetensors index: {total_size / (1024**3):.1f}GB")
        return total_size
    except Exception:
        # Single-file model or index not available
        try:
            from huggingface_hub import hf_hub_download
            # Try single safetensors file
            hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                token=token,
                local_files_only=True,
            )
            # If we get here, it exists but we can't easily get size without downloading
            return None
        except Exception:
            return None


def probe_model(model_name: str, token: Optional[str] = None) -> ModelProfile:
    """
    Probe a model from HuggingFace without downloading weights.
    
    Reads config.json via the HF API. Zero VRAM, ~1 second.
    Returns a ModelProfile with everything needed to plan loading.
    
    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen3-Coder-Next")
        token: Optional HF token for gated models
    
    Returns:
        ModelProfile with size estimates, dtype, MoE info, compatibility flags
    """
    logger.info(f"Probing model: {model_name}")
    
    # Treat empty token as None (empty Bearer header causes httpx to reject)
    if not token:
        token = None
    
    # Fetch config.json from HF Hub
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            token=token,
        )
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to fetch config.json for {model_name}: {e}")
        raise ValueError(f"Cannot probe model {model_name}: {e}")
    
    # Extract basic info
    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures", [])
    arch_class = architectures[0] if architectures else ""
    
    # VLMs can nest language model config under multiple keys.
    # Resolve the best block for decoder/layer sizing.
    effective_config = _resolve_effective_config(config)
    
    # Detect MoE
    moe_info = _detect_moe(effective_config)
    
    # Detect dtype
    native_dtype = _detect_dtype(config)
    is_fp8 = native_dtype == NativeDtype.FP8
    is_mxfp4 = native_dtype == NativeDtype.MXFP4
    is_nvfp4 = native_dtype == NativeDtype.NVFP4
    
    # Detect custom code
    custom_info = _detect_custom_code(config)
    
    # Detect dependencies
    required_libs = _detect_required_libraries(config, model_name)
    min_python = _detect_python_version(config, model_name)
    
    # BNB compatibility
    supports_bnb = True
    if custom_info["custom_model_class"] in KNOWN_BNB_INCOMPATIBLE:
        supports_bnb = False
    # Custom code models are risky — flag but don't block
    if custom_info["uses_custom_code"] and arch_class not in STANDARD_ARCHITECTURES:
        # Unknown custom class — might reject bnb kwargs
        # We'll use BitsAndBytesConfig (not load_in_8bit kwarg) which is safer
        logger.warning(f"Custom model class {custom_info['custom_model_class']} — "
                       f"will use BitsAndBytesConfig approach for quantization")
    
    # Size estimation
    num_layers = effective_config.get("num_hidden_layers", 0)
    hidden_size = effective_config.get("hidden_size", 0)
    intermediate_size = effective_config.get("intermediate_size", 0)
    vocab_size = effective_config.get("vocab_size", 0)
    num_heads = effective_config.get("num_attention_heads", 0)
    num_kv_heads = effective_config.get("num_key_value_heads", num_heads)
    
    # Build per-layer profiles
    layer_profiles = []
    moe_freq = moe_info["moe_layer_frequency"]
    total_params = 0
    active_params = 0
    
    for i in range(num_layers):
        is_moe_layer = moe_info["is_moe"] and (i % moe_freq == 0 if moe_freq > 0 else True)
        n_experts = moe_info["num_experts"] if is_moe_layer else 0
        
        size_bf16 = _estimate_layer_size(
            effective_config, is_moe_layer, n_experts, moe_info["shared_expert"], bytes_per_param=2
        )
        size_int8 = _estimate_layer_size(
            effective_config, is_moe_layer, n_experts, moe_info["shared_expert"], bytes_per_param=1
        )
        size_int4 = size_bf16 // 4  # Rough: 0.5 bytes per param
        
        lp = LayerProfile(
            index=i,
            size_bf16_bytes=size_bf16,
            size_int8_bytes=size_int8,
            size_int4_bytes=size_int4,
            is_moe_layer=is_moe_layer,
            num_experts=n_experts,
        )
        layer_profiles.append(lp)
        
        # Count params
        layer_params_bf16 = size_bf16 // 2  # bf16 = 2 bytes per param
        total_params += layer_params_bf16
        if is_moe_layer and n_experts > 1:
            # Active params: attention + active_experts * expert_size + router
            active_expert_ratio = moe_info["num_active_experts"] / n_experts
            active_params += int(layer_params_bf16 * active_expert_ratio)
        else:
            active_params += layer_params_bf16
    
    # Embedding size
    embed_size_bf16 = _estimate_embedding_size(effective_config, bytes_per_param=2)
    embed_params = embed_size_bf16 // 2
    total_params += embed_params
    active_params += embed_params
    
    # Total sizes
    total_bf16 = sum(lp.size_bf16_bytes for lp in layer_profiles) + embed_size_bf16
    total_int8 = sum(lp.size_int8_bytes for lp in layer_profiles) + embed_size_bf16 // 2
    total_int4 = sum(lp.size_int4_bytes for lp in layer_profiles) + embed_size_bf16 // 4
    
    # Native size depends on dtype
    if native_dtype == NativeDtype.FP8:
        total_native = total_bf16 // 2  # Stored as FP8, but loads as bf16 on A100
    elif native_dtype == NativeDtype.MXFP4:
        total_native = total_bf16 // 4  # Stored as 4-bit, but may dequantize to bf16
    elif native_dtype == NativeDtype.INT8:
        total_native = total_int8
    elif native_dtype == NativeDtype.INT4:
        total_native = total_int4
    else:
        total_native = total_bf16
    
    # CRITICAL: Override with actual safetensor sizes when available.
    # Parameter-count math can be wildly off for MoE models (e.g. Qwen3-Coder-Next
    # estimated 1442GB but actual weights are 148GB because expert dimensions are
    # much smaller than hidden_size * intermediate_size implies).
    actual_size = _fetch_actual_size(model_name, token)
    if actual_size is not None:
        actual_gb = actual_size / (1024**3)
        estimated_gb = total_native / (1024**3)
        if abs(actual_gb - estimated_gb) / max(estimated_gb, 0.01) > 0.2:
            logger.warning(f"Size estimate ({estimated_gb:.1f}GB) differs from actual ({actual_gb:.1f}GB) "
                           f"by {abs(actual_gb - estimated_gb) / max(estimated_gb, 0.01):.0%} — using actual")
        
        # Use actual size as native, scale bf16/int8/int4 proportionally
        scale = actual_size / total_native if total_native > 0 else 1.0
        total_bf16 = int(total_bf16 * scale) if native_dtype != NativeDtype.BF16 else actual_size
        total_int8 = int(total_int8 * scale) if native_dtype != NativeDtype.INT8 else actual_size
        total_int4 = int(total_int4 * scale) if native_dtype != NativeDtype.INT4 else actual_size
        total_native = actual_size
        
        # CRITICAL: For Mxfp4 and FP8 models, the actual disk size is the QUANTIZED size.
        # At load time, these dequantize to bf16 (2× for FP8, ~2× for Mxfp4) when the
        # hardware doesn't support native computation (A100 for Mxfp4 without Triton >=3.4).
        # The bf16 size must reflect the DEQUANTIZED size, not the disk size.
        if native_dtype == NativeDtype.MXFP4:
            # Mxfp4 dequantizes to bf16 at load time. The safetensor index reports
            # the quantized size. We flag this and let the PLANNER handle the 
            # dequantization multiplier (like it does for FP8).
            logger.warning(f"Mxfp4 model: disk={actual_size/(1024**3):.1f}GB — "
                          f"will dequantize to bf16 at load time (~1.3× disk)")
        elif native_dtype == NativeDtype.FP8:
            logger.warning(f"FP8 model: disk={actual_size/(1024**3):.1f}GB — "
                          f"will dequantize to bf16 on A100 (~2× disk)")
        
        # Also rescale per-layer profiles proportionally
        if scale != 1.0:
            for lp in layer_profiles:
                lp.size_bf16_bytes = int(lp.size_bf16_bytes * scale)
                lp.size_int8_bytes = int(lp.size_int8_bytes * scale)
                lp.size_int4_bytes = int(lp.size_int4_bytes * scale)
            embed_size_bf16 = int(embed_size_bf16 * scale)
    
    # Detect pre-quantized models (cannot apply BNB on top)
    quant_config = config.get("quantization_config", {})
    is_pre_quantized = bool(quant_config.get("quant_method", ""))
    pre_quant_method = quant_config.get("quant_method", "")
    
    profile = ModelProfile(
        model_name=model_name,
        model_type=model_type,
        architecture_class=arch_class,
        num_parameters=total_params,
        num_active_parameters=active_params if moe_info["is_moe"] else total_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        is_moe=moe_info["is_moe"],
        num_experts=moe_info["num_experts"],
        num_active_experts=moe_info["num_active_experts"],
        moe_layer_frequency=moe_info["moe_layer_frequency"],
        shared_expert=moe_info["shared_expert"],
        native_dtype=native_dtype,
        is_fp8=is_fp8,
        is_mxfp4=is_mxfp4,
        is_nvfp4=is_nvfp4,
        size_bf16=total_bf16,
        size_int8=total_int8,
        size_int4=total_int4,
        size_native=total_native,
        layer_profiles=layer_profiles,
        embedding_size_bytes=embed_size_bf16,
        uses_custom_code=custom_info["uses_custom_code"],
        custom_model_class=custom_info["custom_model_class"],
        auto_map=custom_info["auto_map"],
        required_libraries=required_libs,
        min_python_version=min_python,
        supports_bnb_quantization=supports_bnb,
        is_pre_quantized=is_pre_quantized,
        pre_quantization_method=pre_quant_method,
        raw_config=config,
    )
    
    logger.info(f"Probe complete:\n{profile.summary()}")
    return profile
