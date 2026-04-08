"""
Phase 2: Loading Strategy Planning.

Given a ModelProfile and available hardware, compute the optimal loading strategy.
Zero VRAM cost — pure computation.

Tries strategies in order: bf16 → INT8 → INT4 → impossible.
For each strategy, computes explicit per-layer GPU placement via bin-packing.
Ensures headroom on EVERY GPU (not just total).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

from llm_skeleton.probe import ModelProfile, NativeDtype
from llm_skeleton.hardware import HardwareProfile, GPUInfo
from llm_skeleton.bin_packing import pack_layers, pack_layers_quantized, PlacementResult
from llm_skeleton.compatibility import check_compatibility, CompatibilityReport

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    NONE = "none"           # bf16/fp16, no quantization
    BNB_INT8 = "bnb_int8"  # BitsAndBytesConfig(load_in_8bit=True)
    BNB_INT4 = "bnb_int4"  # BitsAndBytesConfig(load_in_4bit=True)


@dataclass
class LoadingStrategy:
    """A specific strategy for loading a model."""
    quantization: QuantizationMethod
    dtype_str: str  # "bfloat16", "float16"
    estimated_vram_gb: float
    placement: PlacementResult
    
    @property
    def device_map(self) -> Dict[str, int]:
        return self.placement.device_map
    
    @property
    def max_memory(self) -> Dict:
        return self.placement.max_memory


@dataclass
class LoadingPlan:
    """Complete loading plan for a model."""
    profile: ModelProfile
    hardware: HardwareProfile
    compatibility: CompatibilityReport
    strategy: Optional[LoadingStrategy] = None
    gpu_indices: List[int] = field(default_factory=list)
    headroom_gb: float = 5.0
    
    # All strategies tried, in order
    strategies_tried: List[Tuple[str, str]] = field(default_factory=list)  # [(name, result)]
    
    @property
    def can_load(self) -> bool:
        return self.strategy is not None and self.compatibility.can_load
    
    @property
    def failure_reason(self) -> str:
        if not self.compatibility.can_load:
            errors = [i for i in self.compatibility.issues if i.severity == "error"]
            return "; ".join(i.message for i in errors)
        if self.strategy is None:
            tried = ", ".join(f"{name}: {result}" for name, result in self.strategies_tried)
            return f"No strategy fits. Tried: {tried}"
        return ""
    
    def summary(self) -> str:
        lines = [f"Loading plan for {self.profile.model_name}:"]
        
        if not self.can_load:
            lines.append(f"  ❌ Cannot load: {self.failure_reason}")
            return "\n".join(lines)
        
        s = self.strategy
        lines.extend([
            f"  Strategy: {s.quantization.value} ({s.dtype_str})",
            f"  Estimated VRAM: {s.estimated_vram_gb:.1f}GB",
            f"  GPUs: {self.gpu_indices}",
            f"  Headroom: {self.headroom_gb:.1f}GB per GPU",
            s.placement.summary(),
        ])
        return "\n".join(lines)
    
    def get_load_kwargs(self) -> Dict[str, Any]:
        """Build kwargs dict for from_pretrained."""
        if not self.can_load or not self.strategy:
            raise RuntimeError(f"Cannot load: {self.failure_reason}")
        
        import torch
        
        kwargs = {
            "trust_remote_code": self.profile.uses_custom_code,
            "low_cpu_mem_usage": True,
            "device_map": self.strategy.device_map,
            "max_memory": self.strategy.max_memory,
            "offload_folder": "/tmp/offload",
        }
        
        if self.strategy.quantization == QuantizationMethod.BNB_INT8:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.strategy.quantization == QuantizationMethod.BNB_INT4:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            # No BNB quantization — set dtype
            # For pre-quantized FP8 models, don't set torch_dtype (let HF handle dequantization)
            if not self.profile.is_pre_quantized:
                dtype = torch.bfloat16 if self.strategy.dtype_str == "bfloat16" else torch.float16
                kwargs["torch_dtype"] = dtype
        
        return kwargs


def plan_loading(
    profile: ModelProfile,
    hardware: HardwareProfile,
    gpu_indices: Optional[List[int]] = None,
    headroom_gb: float = 5.0,
    prefer_quantization: Optional[QuantizationMethod] = None,
    allow_quantization: bool = True,
) -> LoadingPlan:
    """
    Compute optimal loading strategy for a model.
    
    Tries strategies in order of quality:
    1. bf16 (best quality, most VRAM)
    2. INT8 via BitsAndBytesConfig (good quality, ~half VRAM)
    3. INT4 via BitsAndBytesConfig (lower quality, ~quarter VRAM)
    
    For each strategy, runs bin-packing to compute explicit device map.
    Ensures headroom_gb free on EVERY GPU for gradient attribution.
    
    Args:
        profile: Model probe results
        hardware: Available hardware
        gpu_indices: Which GPUs to use (None = all)
        headroom_gb: Reserved VRAM per GPU for gradients
        prefer_quantization: Force a specific quantization method
        allow_quantization: If False, only try bf16
    
    Returns:
        LoadingPlan with strategy, device_map, and load kwargs
    """
    # Default to all GPUs
    if gpu_indices is None:
        gpu_indices = [g.index for g in hardware.gpus]
    
    # Compatibility check
    compat = check_compatibility(profile, hardware, gpu_indices)
    
    plan = LoadingPlan(
        profile=profile,
        hardware=hardware,
        compatibility=compat,
        gpu_indices=gpu_indices,
        headroom_gb=headroom_gb,
    )
    
    if not compat.can_load:
        logger.error(f"Compatibility check failed: {plan.failure_reason}")
        return plan
    
    # Build GPU capacity list
    headroom_bytes = int(headroom_gb * (1024**3))
    gpu_capacities = []
    for idx in gpu_indices:
        gpus = [g for g in hardware.gpus if g.index == idx]
        if gpus:
            gpu = gpus[0]
            # Use 95% of total VRAM as capacity (OS/driver overhead)
            capacity = int(gpu.total_vram_bytes * 0.95)
            gpu_capacities.append((idx, capacity))
    
    if not gpu_capacities:
        plan.strategies_tried.append(("setup", "No valid GPUs"))
        return plan
    
    # Determine strategies to try
    strategies_to_try = []
    
    # Pre-quantized models (FP8, GPTQ, AWQ) cannot have BNB applied on top.
    # They load in their native format and dequantize if needed.
    if profile.is_pre_quantized:
        logger.info(f"Pre-quantized model ({profile.pre_quantization_method}) — "
                     f"BNB quantization disabled, loading as-is")
        strategies_to_try = [QuantizationMethod.NONE]
    elif prefer_quantization:
        strategies_to_try = [prefer_quantization]
    else:
        strategies_to_try = [QuantizationMethod.NONE]
        if allow_quantization:
            strategies_to_try.extend([QuantizationMethod.BNB_INT8, QuantizationMethod.BNB_INT4])
    
    # FP8 models on A100: they dequantize to bf16, so treat as bf16 for sizing
    effective_is_fp8 = profile.is_fp8 and any(
        not g.supports_fp8_natively for g in hardware.gpu_subset(gpu_indices)
    )
    if effective_is_fp8:
        logger.warning("FP8 model on non-Hopper GPU — planning for bf16 memory footprint (2x disk size)")

    # Mxfp4 models without Triton >=3.4: dequantize to bf16
    effective_is_mxfp4 = profile.is_mxfp4
    if effective_is_mxfp4:
        logger.warning("Mxfp4 model — planning for bf16 memory footprint (dequantized, ~1.3x disk size)")

    # Dequantization multiplier: how much bigger the model is in VRAM vs on disk
    # Applied to layer sizes before bin-packing
    dequant_multiplier = 1.0
    if effective_is_fp8:
        dequant_multiplier = 2.0  # FP8 → bf16 = 2× size
    elif effective_is_mxfp4:
        dequant_multiplier = 1.3  # Mxfp4 → bf16 ≈ 1.3× (empirical from gpt-oss-120b: 60.8GB disk → ~77GB VRAM)
    
    # Try each strategy
    for quant in strategies_to_try:
        strategy_name = quant.value
        
        # Apply runtime overhead to layer sizes.
        # HuggingFace uses ~15-20% more memory than safetensor weight size due to:
        # - CUDA context (~1GB per GPU)
        # - Activation buffers during from_pretrained
        # - Memory fragmentation
        # - Model metadata and state dict overhead
        # Evidence: Qwen3-Coder-Next safetensor=148.4GB, actual runtime=~160GB (8% overhead)
        #           On tightly packed GPUs, per-GPU overhead is higher (~15-20%)
        RUNTIME_OVERHEAD = 1.15  # 15% safety margin
        
        inflated_layers = []
        for lp in profile.layer_profiles:
            from llm_skeleton.probe import LayerProfile
            inflated_layers.append(LayerProfile(
                index=lp.index,
                size_bf16_bytes=int(lp.size_bf16_bytes * RUNTIME_OVERHEAD * dequant_multiplier),
                size_int8_bytes=int(lp.size_int8_bytes * RUNTIME_OVERHEAD * dequant_multiplier),
                size_int4_bytes=int(lp.size_int4_bytes * RUNTIME_OVERHEAD * dequant_multiplier),
                is_moe_layer=lp.is_moe_layer,
                num_experts=lp.num_experts,
            ))
        inflated_embed = int(profile.embedding_size_bytes * RUNTIME_OVERHEAD * dequant_multiplier)
        
        if quant == QuantizationMethod.NONE:
            # bf16 — use full layer sizes (inflated for runtime overhead + dequantization)
            placement = pack_layers(
                layer_profiles=inflated_layers,
                embedding_size_bytes=inflated_embed,
                gpu_capacities_bytes=gpu_capacities,
                headroom_bytes=headroom_bytes,
                layer_prefix=profile.layer_prefix,
                embed_module=profile.embed_module,
                lm_head_module=profile.lm_head_module,
                norm_module=profile.norm_module,
                extra_modules=profile.extra_modules,
            )
            estimated_gb = profile.size_bf16_gb * dequant_multiplier
            
        elif quant == QuantizationMethod.BNB_INT8:
            placement = pack_layers_quantized(
                layer_profiles=inflated_layers,
                embedding_size_bytes=inflated_embed,
                gpu_capacities_bytes=gpu_capacities,
                quantization="int8",
                headroom_bytes=headroom_bytes,
                layer_prefix=profile.layer_prefix,
                embed_module=profile.embed_module,
                lm_head_module=profile.lm_head_module,
                norm_module=profile.norm_module,
                extra_modules=profile.extra_modules,
            )
            estimated_gb = profile.size_int8_gb
            
        elif quant == QuantizationMethod.BNB_INT4:
            placement = pack_layers_quantized(
                layer_profiles=inflated_layers,
                embedding_size_bytes=inflated_embed,
                gpu_capacities_bytes=gpu_capacities,
                quantization="int4",
                headroom_bytes=headroom_bytes,
                layer_prefix=profile.layer_prefix,
                embed_module=profile.embed_module,
                lm_head_module=profile.lm_head_module,
                norm_module=profile.norm_module,
                extra_modules=profile.extra_modules,
            )
            estimated_gb = profile.size_int4_gb
        else:
            continue
        
        if placement.success:
            plan.strategy = LoadingStrategy(
                quantization=quant,
                dtype_str="bfloat16",
                estimated_vram_gb=estimated_gb,
                placement=placement,
            )
            plan.strategies_tried.append((strategy_name, "✅ fits"))
            logger.info(f"Strategy '{strategy_name}' works: {estimated_gb:.1f}GB estimated")
            break
        else:
            plan.strategies_tried.append((strategy_name, placement.failure_reason))
            logger.info(f"Strategy '{strategy_name}' failed: {placement.failure_reason}")
    
    if plan.can_load:
        logger.info(f"\n{plan.summary()}")
    else:
        logger.error(f"\n{plan.summary()}")
    
    return plan
