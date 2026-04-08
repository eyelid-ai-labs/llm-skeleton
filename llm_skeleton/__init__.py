"""
LLM Skeleton: Universal Model Loader Framework.

Probe → Plan → Load. No surprises.
"""

from llm_skeleton.probe import probe_model, ModelProfile, _detect_layer_prefix, _detect_special_modules, _detect_vlm
from llm_skeleton.plan import plan_loading, LoadingPlan, LoadingStrategy
from llm_skeleton.load import execute_plan
from llm_skeleton.hardware import detect_gpus, GPUInfo, HardwareProfile
from llm_skeleton.orchestrator import DualModelOrchestrator
from llm_skeleton.compatibility import check_compatibility, CompatibilityReport

__all__ = [
    "probe_model",
    "ModelProfile",
    "_detect_layer_prefix",
    "_detect_special_modules",
    "_detect_vlm",
    "plan_loading",
    "LoadingPlan",
    "LoadingStrategy",
    "execute_plan",
    "detect_gpus",
    "GPUInfo",
    "HardwareProfile",
    "DualModelOrchestrator",
    "check_compatibility",
    "CompatibilityReport",
]

__version__ = "0.4.1"
