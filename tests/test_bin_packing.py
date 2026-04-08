"""
Tests for bin-packing algorithm.

Tests the core placement logic without needing GPUs or HuggingFace.
"""

import pytest
from llm_skeleton.probe import LayerProfile
from llm_skeleton.bin_packing import pack_layers, pack_layers_quantized, GPUAllocation


GB = 1024**3


def make_uniform_layers(n: int, size_gb: float) -> list:
    """Create N uniform layers of given size."""
    size_bytes = int(size_gb * GB)
    return [
        LayerProfile(
            index=i,
            size_bf16_bytes=size_bytes,
            size_int8_bytes=size_bytes // 2,
            size_int4_bytes=size_bytes // 4,
        )
        for i in range(n)
    ]


def make_moe_layers(n: int, dense_size_gb: float, moe_size_gb: float, moe_freq: int = 1) -> list:
    """Create layers with alternating dense/MoE sizes."""
    layers = []
    for i in range(n):
        is_moe = (i % moe_freq == 0) if moe_freq > 0 else False
        size = moe_size_gb if is_moe else dense_size_gb
        size_bytes = int(size * GB)
        layers.append(LayerProfile(
            index=i,
            size_bf16_bytes=size_bytes,
            size_int8_bytes=size_bytes // 2,
            size_int4_bytes=size_bytes // 4,
            is_moe_layer=is_moe,
            num_experts=128 if is_moe else 0,
        ))
    return layers


class TestBasicPacking:
    """Test basic layer placement."""
    
    def test_single_gpu_fits(self):
        """Small model on one GPU."""
        layers = make_uniform_layers(10, 1.0)  # 10GB total
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        assert result.success
        assert all(v == 0 for v in result.device_map.values())
    
    def test_two_gpu_split(self):
        """Model that needs 2 GPUs."""
        layers = make_uniform_layers(20, 3.0)  # 60GB layers
        result = pack_layers(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(40 * GB)), (1, int(40 * GB))],
            headroom_bytes=int(2 * GB),
        )
        assert result.success
        # Should use both GPUs
        gpu_indices_used = set(result.device_map.values())
        assert len(gpu_indices_used) == 2
    
    def test_doesnt_fit(self):
        """Model too large for available GPUs."""
        layers = make_uniform_layers(10, 10.0)  # 100GB layers
        result = pack_layers(
            layers,
            embedding_size_bytes=int(5 * GB),
            gpu_capacities_bytes=[(0, int(40 * GB))],
            headroom_bytes=int(5 * GB),
        )
        assert not result.success
        assert "Ran out of GPUs" in result.failure_reason or "doesn't fit" in result.failure_reason
    
    def test_headroom_respected(self):
        """Headroom is reserved on every GPU."""
        layers = make_uniform_layers(4, 15.0)  # 60GB layers
        result = pack_layers(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB)), (1, int(80 * GB))],
            headroom_bytes=int(10 * GB),  # 10GB headroom
        )
        assert result.success
        for alloc in result.gpu_allocations:
            if alloc.layers or alloc.has_embedding or alloc.has_lm_head:
                assert alloc.available_bytes >= 0, f"GPU {alloc.gpu_index} overcommitted"
    
    def test_contiguity(self):
        """Layers on same GPU are contiguous."""
        layers = make_uniform_layers(20, 2.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(30 * GB)), (1, int(30 * GB))],
            headroom_bytes=int(2 * GB),
        )
        assert result.success
        for alloc in result.gpu_allocations:
            if len(alloc.layers) > 1:
                # Check contiguity
                for i in range(len(alloc.layers) - 1):
                    assert alloc.layers[i + 1] == alloc.layers[i] + 1


class TestMoEPacking:
    """Test packing with variable-size MoE layers."""
    
    def test_moe_layers_larger(self):
        """MoE layers are correctly sized larger than dense layers."""
        layers = make_moe_layers(10, dense_size_gb=1.0, moe_size_gb=10.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[
                (0, int(80 * GB)),
                (1, int(80 * GB)),
            ],
            headroom_bytes=int(5 * GB),
        )
        assert result.success
    
    def test_moe_needs_more_gpus(self):
        """MoE model needs more GPUs than dense equivalent."""
        # Dense: 48 layers * 1GB = 48GB — fits on 1 GPU
        dense_layers = make_uniform_layers(48, 1.0)
        dense_result = pack_layers(
            dense_layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        
        # MoE: 48 layers * 5GB = 240GB — needs multiple GPUs
        moe_layers = make_moe_layers(48, dense_size_gb=1.0, moe_size_gb=5.0)
        moe_result_1gpu = pack_layers(
            moe_layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        
        assert dense_result.success
        assert not moe_result_1gpu.success  # MoE doesn't fit on 1 GPU


class TestQuantizedPacking:
    """Test packing with quantization."""
    
    def test_int8_halves_size(self):
        """INT8 quantization roughly halves layer sizes."""
        layers = make_uniform_layers(20, 4.0)  # 80GB bf16
        
        # bf16 on 2 GPUs
        bf16_result = pack_layers(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(50 * GB)), (1, int(50 * GB))],
            headroom_bytes=int(5 * GB),
        )
        
        # int8 on 1 GPU
        int8_result = pack_layers_quantized(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(50 * GB))],
            quantization="int8",
            headroom_bytes=int(5 * GB),
        )
        
        assert bf16_result.success
        assert int8_result.success  # INT8 fits on fewer GPUs


class TestDeviceMap:
    """Test device_map generation."""
    
    def test_device_map_has_all_layers(self):
        """Device map includes every layer."""
        n_layers = 24
        layers = make_uniform_layers(n_layers, 1.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        assert result.success
        
        # Check all layers present
        for i in range(n_layers):
            assert f"model.layers.{i}" in result.device_map
        
        # Check embedding and lm_head
        assert "model.embed_tokens" in result.device_map
        assert "lm_head" in result.device_map
        assert "model.norm" in result.device_map
    
    def test_max_memory_dict(self):
        """max_memory dict is correctly formatted."""
        layers = make_uniform_layers(10, 2.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB)), (2, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        assert result.success
        assert "cpu" in result.max_memory
        assert result.max_memory["cpu"] == "0GiB"
        # GPU indices should match what was provided
        assert 0 in result.max_memory
        assert 2 in result.max_memory


class TestEdgeCases:
    """Edge cases and error handling."""
    
    def test_no_gpus(self):
        """Graceful failure with no GPUs."""
        layers = make_uniform_layers(10, 1.0)
        result = pack_layers(layers, int(1 * GB), [], int(5 * GB))
        assert not result.success
    
    def test_no_layers(self):
        """Graceful failure with no layers."""
        result = pack_layers([], int(1 * GB), [(0, int(80 * GB))], int(5 * GB))
        assert not result.success
    
    def test_embedding_too_large(self):
        """Embedding doesn't fit on first GPU."""
        layers = make_uniform_layers(1, 1.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(100 * GB),  # Huge embedding
            gpu_capacities_bytes=[(0, int(10 * GB))],
            headroom_bytes=int(1 * GB),
        )
        assert not result.success
        assert "Embedding" in result.failure_reason


class TestVLMDeviceMapPaths:
    """Test that bin-packing generates correct device map keys for VLMs."""

    def test_standard_decoder_paths(self):
        """Default paths use model.layers.X convention."""
        layers = make_uniform_layers(10, 1.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
        )
        assert result.success
        assert "model.embed_tokens" in result.device_map
        assert "model.layers.0" in result.device_map
        assert "model.norm" in result.device_map
        assert "lm_head" in result.device_map

    def test_vlm_layer_prefix(self):
        """VLM paths use model.language_model.layers.X convention."""
        layers = make_uniform_layers(10, 1.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
            layer_prefix="model.language_model.layers",
            embed_module="model.language_model.embed_tokens",
            lm_head_module="lm_head",
            norm_module="model.language_model.norm",
        )
        assert result.success
        assert "model.language_model.embed_tokens" in result.device_map
        assert "model.language_model.layers.0" in result.device_map
        assert "model.language_model.layers.9" in result.device_map
        assert "model.language_model.norm" in result.device_map
        assert "lm_head" in result.device_map
        # Old hardcoded paths should NOT be present
        assert "model.layers.0" not in result.device_map
        assert "model.embed_tokens" not in result.device_map
        assert "model.norm" not in result.device_map

    def test_vlm_extra_modules_placed(self):
        """Extra modules (vision tower etc.) are placed in device map."""
        layers = make_uniform_layers(10, 1.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            headroom_bytes=int(5 * GB),
            layer_prefix="model.language_model.layers",
            embed_module="model.language_model.embed_tokens",
            lm_head_module="lm_head",
            norm_module="model.language_model.norm",
            extra_modules=["model.vision_tower", "model.embed_vision"],
        )
        assert result.success
        assert "model.vision_tower" in result.device_map
        assert "model.embed_vision" in result.device_map

    def test_vlm_extra_modules_on_last_gpu(self):
        """Extra modules are placed on the same GPU as lm_head."""
        layers = make_uniform_layers(20, 3.0)
        result = pack_layers(
            layers,
            embedding_size_bytes=int(2 * GB),
            gpu_capacities_bytes=[(0, int(40 * GB)), (1, int(40 * GB))],
            headroom_bytes=int(2 * GB),
            extra_modules=["model.vision_tower"],
        )
        assert result.success
        lm_head_gpu = result.device_map["lm_head"]
        assert result.device_map["model.vision_tower"] == lm_head_gpu

    def test_quantized_packing_uses_vlm_paths(self):
        """Quantized packing also respects VLM paths."""
        layers = make_uniform_layers(10, 4.0)
        result = pack_layers_quantized(
            layers,
            embedding_size_bytes=int(1 * GB),
            gpu_capacities_bytes=[(0, int(80 * GB))],
            quantization="int8",
            headroom_bytes=int(5 * GB),
            layer_prefix="model.language_model.layers",
            embed_module="model.language_model.embed_tokens",
            lm_head_module="lm_head",
            norm_module="model.language_model.norm",
            extra_modules=["model.vision_tower"],
        )
        assert result.success
        assert "model.language_model.layers.0" in result.device_map
        assert "model.language_model.embed_tokens" in result.device_map
        assert "model.vision_tower" in result.device_map
