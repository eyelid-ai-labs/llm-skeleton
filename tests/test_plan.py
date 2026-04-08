"""
Tests for loading strategy planning.

Tests the planning logic with mock hardware profiles (no real GPUs needed).
"""

import pytest
from llm_skeleton.probe import ModelProfile, LayerProfile, NativeDtype
from llm_skeleton.hardware import HardwareProfile, GPUInfo
from llm_skeleton.plan import plan_loading, QuantizationMethod


GB = 1024**3


def make_mock_hardware(num_gpus: int, vram_gb: float = 80.0) -> HardwareProfile:
    """Create mock hardware profile."""
    gpus = []
    for i in range(num_gpus):
        gpus.append(GPUInfo(
            index=i,
            name=f"Mock A100 {vram_gb}GB",
            total_vram_bytes=int(vram_gb * GB),
            free_vram_bytes=int(vram_gb * GB),
            compute_capability=(8, 0),
        ))
    return HardwareProfile(
        gpus=gpus,
        total_ram_bytes=int(512 * GB),
        python_version="3.10.0",
        cuda_version="12.6",
    )


def make_mock_profile(
    name: str = "test/model",
    num_layers: int = 24,
    layer_size_gb: float = 1.0,
    embed_size_gb: float = 0.5,
    is_moe: bool = False,
) -> ModelProfile:
    """Create mock model profile."""
    layer_bytes = int(layer_size_gb * GB)
    layers = [
        LayerProfile(
            index=i,
            size_bf16_bytes=layer_bytes,
            size_int8_bytes=layer_bytes // 2,
            size_int4_bytes=layer_bytes // 4,
            is_moe_layer=is_moe,
        )
        for i in range(num_layers)
    ]
    
    total_bf16 = num_layers * layer_bytes + int(embed_size_gb * GB)
    
    return ModelProfile(
        model_name=name,
        model_type="test",
        num_layers=num_layers,
        hidden_size=4096,
        layer_profiles=layers,
        embedding_size_bytes=int(embed_size_gb * GB),
        size_bf16=total_bf16,
        size_int8=total_bf16 // 2,
        size_int4=total_bf16 // 4,
    )


class TestPlanningBasic:
    def test_small_model_bf16(self):
        """Small model fits in bf16 on one GPU."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        hardware = make_mock_hardware(1, 80.0)
        
        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        
        assert plan.can_load
        assert plan.strategy.quantization == QuantizationMethod.NONE
    
    def test_large_model_needs_quantization(self):
        """Large model falls back to INT8."""
        profile = make_mock_profile(num_layers=48, layer_size_gb=3.0)  # 144GB bf16
        hardware = make_mock_hardware(2, 80.0)  # 160GB total, but headroom eats into it
        
        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        
        assert plan.can_load
        # Should need quantization since bf16 (144GB) + headroom (10GB) > 152GB usable
        # INT8 (72GB) should fit easily
    
    def test_impossible_model(self):
        """Model that doesn't fit under any strategy."""
        profile = make_mock_profile(num_layers=100, layer_size_gb=10.0)  # 1000GB
        hardware = make_mock_hardware(2, 80.0)  # 160GB total
        
        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        
        assert not plan.can_load
        assert len(plan.strategies_tried) > 0
    
    def test_no_quantization_flag(self):
        """Respects allow_quantization=False."""
        profile = make_mock_profile(num_layers=48, layer_size_gb=3.0)  # 144GB
        hardware = make_mock_hardware(1, 80.0)  # Too small for bf16
        
        plan = plan_loading(profile, hardware, headroom_gb=5.0, allow_quantization=False)
        
        # bf16 doesn't fit on 1 GPU, and quantization is disabled
        assert not plan.can_load
    
    def test_forced_quantization(self):
        """Respects prefer_quantization."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)  # Small, fits bf16
        hardware = make_mock_hardware(1, 80.0)
        
        plan = plan_loading(
            profile, hardware, headroom_gb=5.0,
            prefer_quantization=QuantizationMethod.BNB_INT8,
        )
        
        assert plan.can_load
        assert plan.strategy.quantization == QuantizationMethod.BNB_INT8


class TestGPUSubset:
    def test_specific_gpus(self):
        """Plan uses only specified GPUs."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        hardware = make_mock_hardware(8, 80.0)
        
        plan = plan_loading(profile, hardware, gpu_indices=[4, 5, 6], headroom_gb=5.0)
        
        assert plan.can_load
        # All layers should be on GPUs 4-6
        for gpu_idx in plan.strategy.device_map.values():
            assert gpu_idx in [4, 5, 6]


class TestLoadKwargs:
    def test_bf16_kwargs(self):
        """bf16 plan generates correct kwargs."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        hardware = make_mock_hardware(1, 80.0)
        
        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        kwargs = plan.get_load_kwargs()
        
        assert "device_map" in kwargs
        assert "max_memory" in kwargs
        assert "quantization_config" not in kwargs
        assert kwargs["device_map"] != "auto"  # Never auto
    
    def test_int8_kwargs_uses_bnb_config(self):
        """INT8 plan uses BitsAndBytesConfig, not load_in_8bit kwarg."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        hardware = make_mock_hardware(1, 80.0)
        
        plan = plan_loading(
            profile, hardware, headroom_gb=5.0,
            prefer_quantization=QuantizationMethod.BNB_INT8,
        )
        kwargs = plan.get_load_kwargs()
        
        assert "quantization_config" in kwargs
        assert "load_in_8bit" not in kwargs  # NEVER this kwarg


class TestVLMLoadKwargs:
    def test_vlm_sets_trust_remote_code(self):
        """VLM models get trust_remote_code=True even without custom code."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        profile.is_vlm = True
        profile.auto_class = "AutoModelForConditionalGeneration"
        hardware = make_mock_hardware(1, 80.0)

        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        kwargs = plan.get_load_kwargs()

        assert kwargs["trust_remote_code"] is True

    def test_non_vlm_no_trust_remote_code(self):
        """Standard models without custom code get trust_remote_code=False."""
        profile = make_mock_profile(num_layers=10, layer_size_gb=1.0)
        hardware = make_mock_hardware(1, 80.0)

        plan = plan_loading(profile, hardware, headroom_gb=5.0)
        kwargs = plan.get_load_kwargs()

        assert kwargs["trust_remote_code"] is False
