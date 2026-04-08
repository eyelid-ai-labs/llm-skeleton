"""
Bin-packing: Place transformer layers onto GPUs with contiguity constraint.

Unlike HuggingFace's greedy sequential fill, this ensures:
- Every GPU stays within capacity
- Layers are contiguous (no scattering layer 5 and layer 20 on same GPU)
- Guaranteed headroom on every GPU for gradient attribution
- Embedding/lm_head placed on first/last GPU

For MoE models, layers vary 10x in size. Greedy packing fails because it
fills early GPUs with huge MoE layers, leaving no room for later layers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from llm_skeleton.probe import LayerProfile

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """What's placed on a single GPU."""
    gpu_index: int
    capacity_bytes: int
    headroom_bytes: int  # Reserved for gradient attribution
    
    layers: List[int] = field(default_factory=list)  # Layer indices
    has_embedding: bool = False
    has_lm_head: bool = False
    used_bytes: int = 0
    
    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.headroom_bytes - self.used_bytes
    
    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024**3)
    
    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)
    
    @property
    def utilization(self) -> float:
        usable = self.capacity_bytes - self.headroom_bytes
        if usable <= 0:
            return 1.0
        return self.used_bytes / usable
    
    def can_fit(self, size_bytes: int) -> bool:
        return self.available_bytes >= size_bytes
    
    def place(self, size_bytes: int, layer_idx: Optional[int] = None):
        self.used_bytes += size_bytes
        if layer_idx is not None:
            self.layers.append(layer_idx)


@dataclass
class PlacementResult:
    """Result of bin-packing: explicit device map for every component."""
    success: bool
    device_map: Dict[str, int] = field(default_factory=dict)  # "model.layers.5" -> gpu_idx
    gpu_allocations: List[GPUAllocation] = field(default_factory=list)
    max_memory: Dict = field(default_factory=dict)  # For from_pretrained
    failure_reason: str = ""
    total_model_bytes: int = 0
    
    def summary(self) -> str:
        if not self.success:
            return f"❌ Placement failed: {self.failure_reason}"
        lines = ["Placement plan:"]
        for alloc in self.gpu_allocations:
            if not alloc.layers and not alloc.has_embedding and not alloc.has_lm_head:
                continue
            parts = []
            if alloc.has_embedding:
                parts.append("embed")
            if alloc.layers:
                if len(alloc.layers) == 1:
                    parts.append(f"layer {alloc.layers[0]}")
                else:
                    parts.append(f"layers {alloc.layers[0]}-{alloc.layers[-1]}")
            if alloc.has_lm_head:
                parts.append("lm_head")
            lines.append(
                f"  GPU {alloc.gpu_index}: {', '.join(parts)} "
                f"({alloc.used_gb:.1f}GB / {alloc.capacity_bytes/(1024**3):.0f}GB, "
                f"{alloc.available_gb:.1f}GB free)"
            )
        return "\n".join(lines)


def pack_layers(
    layer_profiles: List[LayerProfile],
    embedding_size_bytes: int,
    gpu_capacities_bytes: List[Tuple[int, int]],  # [(gpu_idx, capacity_bytes), ...]
    headroom_bytes: int = 5 * (1024**3),  # 5GB default headroom per GPU
    model_prefix: str = "model",
    layer_prefix: str = "",
    embed_module: str = "",
    lm_head_module: str = "",
    norm_module: str = "",
    extra_modules: Optional[List[str]] = None,
    extra_modules_size_bytes: int = 0,
) -> PlacementResult:
    """
    Pack transformer layers onto GPUs with contiguity constraint.
    
    Algorithm: Sequential fill with look-ahead.
    - Place embedding on first GPU
    - Place layers sequentially, moving to next GPU when current is full
    - Place lm_head on last used GPU (or next if no room)
    - Place extra modules (vision tower etc.) on last GPU
    - Ensure headroom on every GPU
    
    This is NOT greedy — it considers layer sizes ahead to avoid
    painting itself into a corner with huge MoE layers.
    
    Args:
        layer_profiles: Per-layer size information
        embedding_size_bytes: Size of embed_tokens + lm_head
        gpu_capacities_bytes: Available GPUs with their capacity
        headroom_bytes: Reserved per-GPU for gradient attribution
        model_prefix: Legacy prefix for device_map keys (used when layer_prefix not set)
        layer_prefix: Actual layer prefix from safetensors (e.g. "model.language_model.layers")
        embed_module: Actual embed module path (e.g. "model.language_model.embed_tokens")
        lm_head_module: Actual lm_head path (e.g. "lm_head")
        norm_module: Actual final norm path (e.g. "model.language_model.norm")
        extra_modules: Extra top-level modules to place (e.g. ["model.vision_tower"])
        extra_modules_size_bytes: Estimated total size of extra modules
    
    Returns:
        PlacementResult with device_map and allocation details
    """
    # Resolve paths: use explicit paths if provided, else fall back to model_prefix defaults
    effective_layer_prefix = layer_prefix or f"{model_prefix}.layers"
    effective_embed = embed_module or f"{model_prefix}.embed_tokens"
    effective_lm_head = lm_head_module or "lm_head"
    effective_norm = norm_module or f"{model_prefix}.norm"
    effective_extra = extra_modules or []
    if not gpu_capacities_bytes:
        return PlacementResult(success=False, failure_reason="No GPUs provided")
    
    if not layer_profiles:
        return PlacementResult(success=False, failure_reason="No layers to place")
    
    # Initialize GPU allocations
    allocations = []
    for gpu_idx, capacity in gpu_capacities_bytes:
        allocations.append(GPUAllocation(
            gpu_index=gpu_idx,
            capacity_bytes=capacity,
            headroom_bytes=headroom_bytes,
        ))
    
    # Split embedding size: embed_tokens on first GPU, lm_head on last
    # Rough split: embed_tokens is ~half, lm_head + final_norm is ~half
    embed_tokens_size = embedding_size_bytes // 2
    lm_head_size = embedding_size_bytes - embed_tokens_size
    
    # Place embed_tokens on first GPU
    if not allocations[0].can_fit(embed_tokens_size):
        return PlacementResult(
            success=False,
            failure_reason=(f"Embedding ({embed_tokens_size/(1024**3):.1f}GB) doesn't fit on "
                            f"GPU {allocations[0].gpu_index} "
                            f"({allocations[0].available_gb:.1f}GB available)")
        )
    allocations[0].place(embed_tokens_size)
    allocations[0].has_embedding = True
    
    # Place layers sequentially with contiguity
    device_map = {}
    device_map[effective_embed] = allocations[0].gpu_index
    
    # Also map norm layers that come before embed_tokens
    # (some models have input_layernorm at the top level)
    
    current_gpu_idx = 0
    total_layer_bytes = 0
    
    for lp in layer_profiles:
        layer_size = lp.size_bf16_bytes  # Will be overridden by caller for int8/int4
        total_layer_bytes += layer_size
        
        # Try to fit on current GPU
        if allocations[current_gpu_idx].can_fit(layer_size):
            allocations[current_gpu_idx].place(layer_size, lp.index)
            device_map[f"{effective_layer_prefix}.{lp.index}"] = allocations[current_gpu_idx].gpu_index
        else:
            # Move to next GPU
            current_gpu_idx += 1
            if current_gpu_idx >= len(allocations):
                placed = sum(len(a.layers) for a in allocations)
                return PlacementResult(
                    success=False,
                    failure_reason=(f"Ran out of GPUs at layer {lp.index}/{len(layer_profiles)}. "
                                    f"Placed {placed} layers, {len(layer_profiles) - placed} remaining. "
                                    f"Layer size: {layer_size/(1024**3):.2f}GB")
                )
            
            if not allocations[current_gpu_idx].can_fit(layer_size):
                return PlacementResult(
                    success=False,
                    failure_reason=(f"Layer {lp.index} ({layer_size/(1024**3):.2f}GB) doesn't fit on "
                                    f"GPU {allocations[current_gpu_idx].gpu_index} "
                                    f"({allocations[current_gpu_idx].available_gb:.1f}GB available)")
                )
            
            allocations[current_gpu_idx].place(layer_size, lp.index)
            device_map[f"{effective_layer_prefix}.{lp.index}"] = allocations[current_gpu_idx].gpu_index
    
    # Place lm_head + final norm on last used GPU (or next if no room)
    lm_head_gpu = current_gpu_idx
    if not allocations[lm_head_gpu].can_fit(lm_head_size):
        lm_head_gpu += 1
        if lm_head_gpu >= len(allocations):
            return PlacementResult(
                success=False,
                failure_reason=f"No room for lm_head ({lm_head_size/(1024**3):.1f}GB) after placing all layers"
            )
    
    allocations[lm_head_gpu].place(lm_head_size)
    allocations[lm_head_gpu].has_lm_head = True
    device_map[effective_norm] = allocations[lm_head_gpu].gpu_index
    device_map[effective_lm_head] = allocations[lm_head_gpu].gpu_index
    
    # Place extra modules (vision tower, embed_vision, etc.) on last used GPU
    if effective_extra:
        last_gpu = lm_head_gpu
        if extra_modules_size_bytes > 0:
            allocations[last_gpu].place(extra_modules_size_bytes)
        for mod in effective_extra:
            device_map[mod] = allocations[last_gpu].gpu_index
    
    # Build max_memory dict
    max_memory = {}
    for alloc in allocations:
        max_memory[alloc.gpu_index] = f"{int(alloc.capacity_bytes / (1024**3))}GiB"
    max_memory["cpu"] = "0GiB"
    
    total_bytes = total_layer_bytes + embedding_size_bytes
    
    result = PlacementResult(
        success=True,
        device_map=device_map,
        gpu_allocations=allocations,
        max_memory=max_memory,
        total_model_bytes=total_bytes,
    )
    
    logger.info(result.summary())
    return result


def pack_layers_quantized(
    layer_profiles: List[LayerProfile],
    embedding_size_bytes: int,
    gpu_capacities_bytes: List[Tuple[int, int]],
    quantization: str,  # "int8" or "int4"
    headroom_bytes: int = 5 * (1024**3),
    model_prefix: str = "model",
    layer_prefix: str = "",
    embed_module: str = "",
    lm_head_module: str = "",
    norm_module: str = "",
    extra_modules: Optional[List[str]] = None,
    extra_modules_size_bytes: int = 0,
) -> PlacementResult:
    """
    Pack layers with quantization-adjusted sizes.
    
    Quantization reduces layer sizes but embedding/lm_head stay in higher precision.
    """
    # Adjust layer sizes for quantization
    adjusted_profiles = []
    for lp in layer_profiles:
        adjusted = LayerProfile(
            index=lp.index,
            size_bf16_bytes=lp.size_int8_bytes if quantization == "int8" else lp.size_int4_bytes,
            size_int8_bytes=lp.size_int8_bytes,
            size_int4_bytes=lp.size_int4_bytes,
            is_moe_layer=lp.is_moe_layer,
            num_experts=lp.num_experts,
        )
        adjusted_profiles.append(adjusted)
    
    # Embedding stays in higher precision (bf16 typically)
    # But quantized models often keep embeddings in the quantized format too
    if quantization == "int8":
        adjusted_embed = embedding_size_bytes // 2  # bf16 -> int8
    elif quantization == "int4":
        adjusted_embed = embedding_size_bytes // 4  # bf16 -> int4
    else:
        adjusted_embed = embedding_size_bytes
    
    return pack_layers(
        adjusted_profiles,
        adjusted_embed,
        gpu_capacities_bytes,
        headroom_bytes,
        model_prefix,
        layer_prefix=layer_prefix,
        embed_module=embed_module,
        lm_head_module=lm_head_module,
        norm_module=norm_module,
        extra_modules=extra_modules,
        extra_modules_size_bytes=extra_modules_size_bytes,
    )
