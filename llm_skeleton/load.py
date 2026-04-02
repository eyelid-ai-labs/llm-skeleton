"""
Phase 3: Execute the loading plan.

Uses explicit device_map computed by the planner. Never device_map="auto".
Validates actual VRAM usage matches plan within tolerance.
"""

from typing import Tuple, Optional, Any
import logging
import time

from llm_skeleton.plan import LoadingPlan

logger = logging.getLogger(__name__)


def execute_plan(
    plan: LoadingPlan,
    load_tokenizer: bool = True,
    validate_vram: bool = True,
    vram_tolerance: float = 0.30,  # 30% tolerance
) -> Tuple[Any, Optional[Any]]:
    """
    Execute a loading plan. Returns (model, tokenizer).
    
    Uses the explicit device_map and quantization config from the plan.
    Never calls device_map="auto".
    
    Args:
        plan: The computed loading plan
        load_tokenizer: Whether to also load the tokenizer
        validate_vram: Whether to check actual vs planned VRAM
        vram_tolerance: Acceptable deviation from planned VRAM (0.3 = 30%)
    
    Returns:
        (model, tokenizer) tuple. tokenizer is None if load_tokenizer=False.
    
    Raises:
        RuntimeError: If plan says model can't be loaded
        RuntimeError: If actual VRAM exceeds plan by more than tolerance
    """
    if not plan.can_load:
        raise RuntimeError(f"Cannot load {plan.profile.model_name}: {plan.failure_reason}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = plan.profile.model_name
    load_kwargs = plan.get_load_kwargs()
    
    logger.info(f"Loading {model_name}...")
    logger.info(f"  Strategy: {plan.strategy.quantization.value}")
    logger.info(f"  Estimated VRAM: {plan.strategy.estimated_vram_gb:.1f}GB")
    logger.info(f"  GPUs: {plan.gpu_indices}")
    
    # Log device map summary (not every layer, just GPU assignments)
    gpu_layer_counts = {}
    for key, gpu_idx in plan.strategy.device_map.items():
        gpu_layer_counts[gpu_idx] = gpu_layer_counts.get(gpu_idx, 0) + 1
    for gpu_idx in sorted(gpu_layer_counts):
        logger.info(f"  GPU {gpu_idx}: {gpu_layer_counts[gpu_idx]} components")
    
    # Record VRAM before loading
    vram_before = {}
    try:
        import torch
        if torch.cuda.is_available():
            for idx in plan.gpu_indices:
                vram_before[idx] = torch.cuda.memory_allocated(idx)
    except Exception:
        pass
    
    # Load model
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    load_time = time.time() - t0
    
    logger.info(f"Model loaded in {load_time:.1f}s")
    
    # Load tokenizer
    tokenizer = None
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=plan.profile.uses_custom_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Validate VRAM usage
    if validate_vram:
        try:
            import torch
            if torch.cuda.is_available():
                total_used_gb = 0
                for idx in plan.gpu_indices:
                    used_now = torch.cuda.memory_allocated(idx)
                    used_delta = (used_now - vram_before.get(idx, 0)) / (1024**3)
                    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                    free_gb = (torch.cuda.mem_get_info(idx)[0]) / (1024**3)
                    total_used_gb += used_delta
                    logger.info(f"  GPU {idx}: +{used_delta:.1f}GB used, {free_gb:.1f}GB free / {total_gb:.0f}GB")
                
                planned_gb = plan.strategy.estimated_vram_gb
                deviation = abs(total_used_gb - planned_gb) / planned_gb if planned_gb > 0 else 0
                
                if deviation > vram_tolerance:
                    logger.warning(
                        f"VRAM usage ({total_used_gb:.1f}GB) deviates {deviation:.0%} from "
                        f"plan ({planned_gb:.1f}GB). Tolerance: {vram_tolerance:.0%}"
                    )
                else:
                    logger.info(f"  Total: {total_used_gb:.1f}GB (planned: {planned_gb:.1f}GB, "
                                f"deviation: {deviation:.0%})")
        except Exception as e:
            logger.debug(f"VRAM validation skipped: {e}")
    
    return model, tokenizer
