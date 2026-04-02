"""
Dual-Model Orchestrator.

Loads two models simultaneously on a multi-GPU system:
- Target model (being analyzed/served)
- Secondary model (reasoning, evaluation, or auxiliary tasks)

Auto-computes optimal GPU split, handles quantization decisions,
reserves headroom for gradient computation.

Usage:
    orchestrator = DualModelOrchestrator(
        target_model="Qwen/Qwen3-Coder-Next",
        reasoning_model="mistralai/Devstral-2-123B-Instruct-2512",
    )
    target, target_tok, reasoning, reasoning_tok = orchestrator.load_both()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

from llm_skeleton.probe import probe_model, ModelProfile
from llm_skeleton.plan import plan_loading, LoadingPlan, QuantizationMethod
from llm_skeleton.load import execute_plan
from llm_skeleton.hardware import detect_gpus, HardwareProfile

logger = logging.getLogger(__name__)


@dataclass
class DualModelPlan:
    """Plan for loading two models simultaneously."""
    target_plan: Optional[LoadingPlan] = None
    reasoning_plan: Optional[LoadingPlan] = None
    target_gpus: List[int] = field(default_factory=list)
    reasoning_gpus: List[int] = field(default_factory=list)
    free_gpus: List[int] = field(default_factory=list)
    
    @property
    def can_load(self) -> bool:
        return (self.target_plan is not None and self.target_plan.can_load and
                self.reasoning_plan is not None and self.reasoning_plan.can_load)
    
    @property
    def failure_reason(self) -> str:
        reasons = []
        if self.target_plan and not self.target_plan.can_load:
            reasons.append(f"Target: {self.target_plan.failure_reason}")
        if self.reasoning_plan and not self.reasoning_plan.can_load:
            reasons.append(f"Reasoning: {self.reasoning_plan.failure_reason}")
        if not self.target_plan:
            reasons.append("Target: no plan computed")
        if not self.reasoning_plan:
            reasons.append("Reasoning: no plan computed")
        return "; ".join(reasons)
    
    def summary(self) -> str:
        lines = ["═" * 60, "  DUAL MODEL LOADING PLAN", "═" * 60]
        lines.append(f"  Target GPUs:    {self.target_gpus}")
        lines.append(f"  Reasoning GPUs: {self.reasoning_gpus}")
        lines.append(f"  Free GPUs:      {self.free_gpus} (gradient attribution)")
        lines.append("")
        
        if self.target_plan:
            lines.append("  TARGET MODEL:")
            lines.append(f"  {self.target_plan.summary()}")
        if self.reasoning_plan:
            lines.append("")
            lines.append("  REASONING MODEL:")
            lines.append(f"  {self.reasoning_plan.summary()}")
        
        if not self.can_load:
            lines.append(f"\n  ❌ {self.failure_reason}")
        else:
            lines.append(f"\n  ✅ Both models fit. Ready to load.")
        
        lines.append("═" * 60)
        return "\n".join(lines)


class DualModelOrchestrator:
    """
    Orchestrates loading two models on a multi-GPU system.
    
    Probes both models, computes optimal GPU split, plans loading
    for both, validates everything fits, then loads.
    """
    
    def __init__(
        self,
        target_model: str,
        reasoning_model: str,
        num_gpus: Optional[int] = None,
        gradient_headroom_gb: float = 5.0,
        reasoning_headroom_gb: float = 2.0,
        free_gpus: int = 1,
        target_quantization: Optional[QuantizationMethod] = None,
        reasoning_quantization: Optional[QuantizationMethod] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Args:
            target_model: HF model ID for the target (being analyzed)
            reasoning_model: HF model ID for the reasoning model
            num_gpus: Override GPU count (None = auto-detect)
            gradient_headroom_gb: Reserved VRAM per GPU for target model
                (gradient attribution needs ~8-10GB: 2GB backprop + runtime buffers + KV cache).
                HuggingFace uses more memory than safetensor weight size due to
                activation buffers, CUDA context, and memory fragmentation.
            reasoning_headroom_gb: Reserved VRAM per GPU for reasoning model
                (inference only, no gradients — 2GB is enough for KV cache + buffers)
            free_gpus: Number of GPUs to leave completely free
            target_quantization: Force quantization for target (None = auto)
            reasoning_quantization: Force quantization for reasoning (None = auto)
            hf_token: HuggingFace token for gated models
        """
        self.target_model_name = target_model
        self.reasoning_model_name = reasoning_model
        self.gradient_headroom_gb = gradient_headroom_gb
        self.reasoning_headroom_gb = reasoning_headroom_gb
        self.free_gpus_count = free_gpus
        self.target_quantization = target_quantization
        self.reasoning_quantization = reasoning_quantization
        self.hf_token = hf_token
        
        # Detect hardware
        self.hardware = detect_gpus()
        if num_gpus is not None:
            # Limit to first N GPUs
            self.hardware.gpus = self.hardware.gpus[:num_gpus]
        
        # Will be populated by probe/plan
        self.target_profile: Optional[ModelProfile] = None
        self.reasoning_profile: Optional[ModelProfile] = None
        self.dual_plan: Optional[DualModelPlan] = None
    
    def probe(self) -> Tuple[ModelProfile, ModelProfile]:
        """Phase 1: Probe both models. Zero VRAM."""
        logger.info("=" * 60)
        logger.info("  PROBING MODELS")
        logger.info("=" * 60)
        
        self.target_profile = probe_model(self.target_model_name, self.hf_token)
        self.reasoning_profile = probe_model(self.reasoning_model_name, self.hf_token)
        
        return self.target_profile, self.reasoning_profile
    
    def plan(self) -> DualModelPlan:
        """
        Phase 2: Compute optimal GPU split and loading plans.
        
        Algorithm:
        1. Reserve free_gpus from the end (for gradient attribution)
        2. Binary search for optimal target/reasoning GPU split
        3. Try bf16 first for both, fall back to quantization
        4. Validate both fit simultaneously
        """
        if not self.target_profile or not self.reasoning_profile:
            self.probe()
        
        logger.info("=" * 60)
        logger.info("  PLANNING GPU SPLIT")
        logger.info("=" * 60)
        
        total_gpus = self.hardware.num_gpus
        available_gpus = total_gpus - self.free_gpus_count
        free_gpu_indices = list(range(available_gpus, total_gpus))
        
        if available_gpus < 2:
            self.dual_plan = DualModelPlan(free_gpus=free_gpu_indices)
            logger.error(f"Need at least 2 GPUs for dual model, have {available_gpus} available")
            return self.dual_plan
        
        logger.info(f"Total GPUs: {total_gpus}, Available: {available_gpus}, "
                     f"Free (gradients): {free_gpu_indices}")
        
        # Try different splits
        best_plan = None
        
        for target_gpu_count in range(1, available_gpus):
            reasoning_gpu_count = available_gpus - target_gpu_count
            
            target_indices = list(range(target_gpu_count))
            reasoning_indices = list(range(target_gpu_count, available_gpus))
            
            logger.info(f"\nTrying split: target={target_indices}, reasoning={reasoning_indices}")
            
            # Plan target — needs more headroom for gradient attribution
            target_plan = plan_loading(
                self.target_profile,
                self.hardware,
                gpu_indices=target_indices,
                headroom_gb=self.gradient_headroom_gb,
                prefer_quantization=self.target_quantization,
            )
            
            if not target_plan.can_load:
                logger.info(f"  Target doesn't fit on {target_gpu_count} GPUs")
                continue
            
            # Plan reasoning — less headroom needed (inference only, no gradients)
            reasoning_plan = plan_loading(
                self.reasoning_profile,
                self.hardware,
                gpu_indices=reasoning_indices,
                headroom_gb=self.reasoning_headroom_gb,
                prefer_quantization=self.reasoning_quantization,
            )
            
            if not reasoning_plan.can_load:
                logger.info(f"  Reasoning doesn't fit on {reasoning_gpu_count} GPUs")
                continue
            
            # Both fit — this is a valid plan
            candidate = DualModelPlan(
                target_plan=target_plan,
                reasoning_plan=reasoning_plan,
                target_gpus=target_indices,
                reasoning_gpus=reasoning_indices,
                free_gpus=free_gpu_indices,
            )
            
            # Prefer the split that gives most headroom to the target
            # (target needs headroom for gradient attribution)
            if best_plan is None:
                best_plan = candidate
            else:
                # Compare: prefer less quantization, then more target headroom
                current_target_quant = best_plan.target_plan.strategy.quantization
                new_target_quant = candidate.target_plan.strategy.quantization
                
                # Prefer no quantization on target
                quant_order = {QuantizationMethod.NONE: 0, QuantizationMethod.BNB_INT8: 1,
                               QuantizationMethod.BNB_INT4: 2}
                if quant_order.get(new_target_quant, 9) < quant_order.get(current_target_quant, 9):
                    best_plan = candidate
            
            logger.info(f"  ✅ Valid split found")
        
        if best_plan is None:
            # Nothing worked — create a failure plan
            best_plan = DualModelPlan(free_gpus=free_gpu_indices)
            logger.error("No valid GPU split found for both models")
        
        self.dual_plan = best_plan
        logger.info(f"\n{best_plan.summary()}")
        return best_plan
    
    def load_both(self) -> Tuple[Any, Any, Any, Any]:
        """
        Phase 3: Load both models.
        
        Returns:
            (target_model, target_tokenizer, reasoning_model, reasoning_tokenizer)
        
        Raises:
            RuntimeError: If planning failed or models don't fit
        """
        if not self.dual_plan:
            self.plan()
        
        if not self.dual_plan.can_load:
            raise RuntimeError(f"Cannot load both models: {self.dual_plan.failure_reason}")
        
        logger.info("=" * 60)
        logger.info("  LOADING MODELS")
        logger.info("=" * 60)
        
        # Load target first (it's the one being analyzed, needs to be stable)
        logger.info("\n--- Loading TARGET model ---")
        target_model, target_tokenizer = execute_plan(self.dual_plan.target_plan)
        
        # Load reasoning model
        logger.info("\n--- Loading REASONING model ---")
        reasoning_model, reasoning_tokenizer = execute_plan(self.dual_plan.reasoning_plan)
        
        # Final VRAM report
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("\n--- FINAL VRAM REPORT ---")
                for i in range(self.hardware.num_gpus):
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    used = torch.cuda.memory_allocated(i) / (1024**3)
                    free = total - used
                    role = "target" if i in self.dual_plan.target_gpus else \
                           "reasoning" if i in self.dual_plan.reasoning_gpus else \
                           "FREE"
                    logger.info(f"  GPU {i} [{role}]: {used:.1f}GB used, {free:.1f}GB free / {total:.0f}GB")
        except Exception:
            pass
        
        logger.info("\n✅ Both models loaded successfully")
        return target_model, target_tokenizer, reasoning_model, reasoning_tokenizer
    
    def load_target_only(self) -> Tuple[Any, Any]:
        """Load just the target model (for single-model analysis)."""
        if not self.target_profile:
            self.probe()
        
        all_indices = list(range(self.hardware.num_gpus - self.free_gpus_count))
        target_plan = plan_loading(
            self.target_profile,
            self.hardware,
            gpu_indices=all_indices,
            headroom_gb=self.gradient_headroom_gb,
            prefer_quantization=self.target_quantization,
        )
        
        if not target_plan.can_load:
            raise RuntimeError(f"Cannot load target: {target_plan.failure_reason}")
        
        return execute_plan(target_plan)
