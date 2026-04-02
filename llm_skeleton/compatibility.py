"""
Compatibility checking: Python version, libraries, GPU capabilities.

Catches issues like MiniMax needing Python 3.11, Nemotron needing mamba-ssm,
FP8 models on A100 (dequantization), NVFP4 needing Blackwell.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import importlib
import sys
import logging

from llm_skeleton.probe import ModelProfile
from llm_skeleton.hardware import HardwareProfile

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityIssue:
    """A single compatibility problem."""
    severity: str  # "error" (blocks loading), "warning" (may cause issues)
    category: str  # "python", "library", "gpu", "dtype", "quantization"
    message: str
    suggestion: str = ""


@dataclass
class CompatibilityReport:
    """Full compatibility assessment."""
    model_name: str
    issues: List[CompatibilityIssue] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)
    
    @property
    def can_load(self) -> bool:
        return not self.has_errors
    
    def summary(self) -> str:
        if not self.issues:
            return f"✅ {self.model_name}: All compatibility checks passed"
        lines = [f"Compatibility report for {self.model_name}:"]
        for issue in self.issues:
            icon = "❌" if issue.severity == "error" else "⚠️"
            lines.append(f"  {icon} [{issue.category}] {issue.message}")
            if issue.suggestion:
                lines.append(f"     → {issue.suggestion}")
        return "\n".join(lines)


def check_compatibility(
    profile: ModelProfile,
    hardware: Optional[HardwareProfile] = None,
    target_gpus: Optional[List[int]] = None,
) -> CompatibilityReport:
    """
    Check if a model can be loaded on the current system.
    
    Catches all the failures we hit on March 28 before they happen.
    """
    report = CompatibilityReport(model_name=profile.model_name)
    
    # 1. Python version
    if profile.min_python_version:
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        required = profile.min_python_version
        if tuple(map(int, current.split("."))) < tuple(map(int, required.split("."))):
            report.issues.append(CompatibilityIssue(
                severity="error",
                category="python",
                message=f"Requires Python {required}+, current is {current}",
                suggestion=f"Use a Python {required}+ environment",
            ))
    
    # 2. Required libraries
    for lib in profile.required_libraries:
        # Normalize: mamba-ssm -> mamba_ssm for import
        import_name = lib.replace("-", "_")
        try:
            importlib.import_module(import_name)
        except ImportError:
            report.issues.append(CompatibilityIssue(
                severity="error",
                category="library",
                message=f"Required library '{lib}' not installed",
                suggestion=f"pip install {lib} --no-build-isolation",
            ))
    
    # 3. GPU-specific checks
    if hardware and hardware.gpus:
        gpus = hardware.gpu_subset(target_gpus) if target_gpus else hardware.gpus
        
        if not gpus:
            report.issues.append(CompatibilityIssue(
                severity="error",
                category="gpu",
                message="No GPUs available for this model",
            ))
        else:
            # FP8 on non-Hopper GPUs
            if profile.is_fp8:
                for gpu in gpus:
                    if not gpu.supports_fp8_natively:
                        report.issues.append(CompatibilityIssue(
                            severity="warning",
                            category="dtype",
                            message=(f"FP8 model will dequantize to bf16 on {gpu.name} "
                                     f"(compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}). "
                                     f"Memory usage will be ~2x disk size."),
                            suggestion="Plan for bf16 memory footprint, not FP8 disk size",
                        ))
                        break  # One warning is enough
            
            # NVFP4 on non-Blackwell GPUs
            if profile.is_nvfp4:
                for gpu in gpus:
                    if not gpu.supports_nvfp4:
                        report.issues.append(CompatibilityIssue(
                            severity="error",
                            category="dtype",
                            message=f"NVFP4 requires Blackwell GPU, {gpu.name} is not supported",
                            suggestion="Use a Blackwell (B200) GPU or load in a different format",
                        ))
                        break
    
    # 4. Custom code warnings
    if profile.uses_custom_code:
        report.issues.append(CompatibilityIssue(
            severity="warning",
            category="quantization",
            message=(f"Custom model class '{profile.custom_model_class}' detected. "
                     f"Must use BitsAndBytesConfig for quantization, not load_in_8bit kwarg."),
            suggestion="LLM Skeleton handles this automatically via BitsAndBytesConfig",
        ))
    
    logger.info(report.summary())
    return report
