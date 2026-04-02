"""
GPU and hardware detection. Zero VRAM cost.

Detects available GPUs, their VRAM, compute capability, and current usage.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import sys

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Single GPU information."""
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: tuple  # e.g. (8, 0) for A100
    
    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / (1024**3)
    
    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_bytes / (1024**3)
    
    @property
    def supports_fp8_natively(self) -> bool:
        """FP8 needs compute >= 8.9 (Hopper/Blackwell). A100 is 8.0 — dequantizes."""
        return self.compute_capability >= (8, 9)
    
    @property
    def supports_nvfp4(self) -> bool:
        """NVFP4 needs Blackwell (compute >= 10.0)."""
        return self.compute_capability >= (10, 0)
    
    def __repr__(self) -> str:
        return (f"GPU({self.index}: {self.name}, "
                f"{self.total_vram_gb:.0f}GB total, "
                f"{self.free_vram_gb:.0f}GB free, "
                f"cc={self.compute_capability[0]}.{self.compute_capability[1]})")


@dataclass
class HardwareProfile:
    """Complete hardware profile."""
    gpus: List[GPUInfo] = field(default_factory=list)
    total_ram_bytes: int = 0
    python_version: str = ""
    cuda_version: Optional[str] = None
    
    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
    
    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_vram_gb for g in self.gpus)
    
    @property
    def total_free_vram_gb(self) -> float:
        return sum(g.free_vram_gb for g in self.gpus)
    
    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024**3)
    
    @property
    def min_gpu_vram_gb(self) -> float:
        """Smallest GPU VRAM — the bottleneck for uniform placement."""
        if not self.gpus:
            return 0.0
        return min(g.total_vram_gb for g in self.gpus)
    
    def gpu_subset(self, indices: List[int]) -> List[GPUInfo]:
        """Get GPUs by index."""
        idx_set = set(indices)
        return [g for g in self.gpus if g.index in idx_set]
    
    def subset_vram_gb(self, indices: List[int]) -> float:
        """Total VRAM for a subset of GPUs."""
        return sum(g.total_vram_gb for g in self.gpu_subset(indices))


def detect_gpus() -> HardwareProfile:
    """Detect all available GPUs and system info. Zero VRAM cost."""
    profile = HardwareProfile()
    profile.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # RAM
    try:
        import psutil
        profile.total_ram_bytes = psutil.virtual_memory().total
    except ImportError:
        pass
    
    # CUDA / GPUs
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("No CUDA GPUs available")
            return profile
        
        profile.cuda_version = torch.version.cuda
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            
            gpu = GPUInfo(
                index=i,
                name=props.name,
                total_vram_bytes=total,
                free_vram_bytes=free,
                compute_capability=(props.major, props.minor),
            )
            profile.gpus.append(gpu)
            logger.debug(f"  {gpu}")
        
        logger.info(f"Detected {profile.num_gpus} GPUs, {profile.total_vram_gb:.0f}GB total VRAM")
        
    except ImportError:
        logger.warning("PyTorch not installed — cannot detect GPUs")
    
    return profile
