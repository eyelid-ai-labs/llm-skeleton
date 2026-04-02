"""
Tests for model probing.

Tests the config.json parsing logic without needing HuggingFace API access.
Uses mock configs that mirror real model config.json files.
"""

import pytest
from llm_skeleton.probe import (
    _detect_dtype, _detect_moe, _detect_custom_code,
    _detect_required_libraries, _detect_python_version,
    _estimate_layer_size, _estimate_embedding_size,
    NativeDtype, ModelProfile,
)


# ─── Mock configs based on real models ──────────────────────────────────────

QWEN3_CODER_NEXT_CONFIG = {
    "model_type": "qwen3_moe",
    "architectures": ["Qwen3MoeForCausalLM"],
    "hidden_size": 3584,
    "intermediate_size": 2560,
    "num_hidden_layers": 48,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 151936,
    "num_local_experts": 128,
    "num_experts_per_tok": 8,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}

DEVSTRAL_CONFIG = {
    "model_type": "ministral",
    "architectures": ["Ministral3ForCausalLM"],
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 88,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "vocab_size": 131072,
    "torch_dtype": None,  # Devstral has no torch_dtype — it's FP8 pre-quantized
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "static",
        "dequantize": False,
    },
    "auto_map": {
        "AutoModelForCausalLM": "modeling_ministral3.Ministral3ForCausalLM",
    },
}

NEMOTRON_CONFIG = {
    "model_type": "nemotron_h",
    "architectures": ["NemotronHForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 8192,
    "num_hidden_layers": 56,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 131072,
    "torch_dtype": "bfloat16",
    "auto_map": {
        "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
    },
}

FP8_MODEL_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "torch_dtype": "float8_e4m3fn",
}

MINIMAX_CONFIG = {
    "model_type": "minimax_m2",
    "architectures": ["MiniMaxM2ForCausalLM"],
    "hidden_size": 6144,
    "num_hidden_layers": 64,
    "torch_dtype": "bfloat16",
    "auto_map": {
        "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM",
    },
}


class TestDtypeDetection:
    def test_bf16(self):
        assert _detect_dtype(QWEN3_CODER_NEXT_CONFIG) == NativeDtype.BF16
    
    def test_devstral_fp8(self):
        """Devstral has quant_method=fp8 in quantization_config."""
        assert _detect_dtype(DEVSTRAL_CONFIG) == NativeDtype.FP8
    
    def test_fp8(self):
        assert _detect_dtype(FP8_MODEL_CONFIG) == NativeDtype.FP8
    
    def test_default_bf16(self):
        assert _detect_dtype({}) == NativeDtype.BF16
    
    def test_gptq_int4(self):
        config = {"quantization_config": {"quant_method": "gptq", "bits": 4}}
        assert _detect_dtype(config) == NativeDtype.INT4
    
    def test_fp8_quant_method(self):
        """Devstral-2-123B has quant_method=fp8 in quantization_config."""
        config = {"quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}}
        assert _detect_dtype(config) == NativeDtype.FP8


class TestMoEDetection:
    def test_qwen_moe(self):
        moe = _detect_moe(QWEN3_CODER_NEXT_CONFIG)
        assert moe["is_moe"] is True
        assert moe["num_experts"] == 128
        assert moe["num_active_experts"] == 8
    
    def test_dense_model(self):
        moe = _detect_moe(DEVSTRAL_CONFIG)
        assert moe["is_moe"] is False
        assert moe["num_experts"] == 0
    
    def test_nemotron_dense(self):
        moe = _detect_moe(NEMOTRON_CONFIG)
        assert moe["is_moe"] is False


class TestCustomCodeDetection:
    def test_devstral_custom(self):
        info = _detect_custom_code(DEVSTRAL_CONFIG)
        assert info["uses_custom_code"] is True
        assert info["custom_model_class"] == "Ministral3ForCausalLM"
    
    def test_nemotron_custom(self):
        info = _detect_custom_code(NEMOTRON_CONFIG)
        assert info["uses_custom_code"] is True
        assert info["custom_model_class"] == "NemotronHForCausalLM"
    
    def test_qwen_standard(self):
        info = _detect_custom_code(QWEN3_CODER_NEXT_CONFIG)
        assert info["uses_custom_code"] is False


class TestLibraryDetection:
    def test_nemotron_needs_mamba(self):
        libs = _detect_required_libraries(NEMOTRON_CONFIG, "nvidia/Nemotron-3-Super-120B")
        assert "mamba-ssm" in libs
    
    def test_qwen_no_special_libs(self):
        libs = _detect_required_libraries(QWEN3_CODER_NEXT_CONFIG, "Qwen/Qwen3-Coder-Next")
        assert len(libs) == 0


class TestPythonVersionDetection:
    def test_minimax_needs_311(self):
        version = _detect_python_version(MINIMAX_CONFIG, "MiniMaxAI/MiniMax-M2.5")
        assert version == "3.11"
    
    def test_qwen_no_requirement(self):
        version = _detect_python_version(QWEN3_CODER_NEXT_CONFIG, "Qwen/Qwen3-Coder-Next")
        assert version is None


class TestSizeEstimation:
    def test_dense_layer_size(self):
        """Dense layer size is reasonable."""
        size = _estimate_layer_size(DEVSTRAL_CONFIG, is_moe_layer=False, num_experts=0,
                                     shared_expert=False, bytes_per_param=2)
        size_gb = size / (1024**3)
        # Devstral has 88 layers. Our estimate captures Q/K/V/O + FFN + norms.
        # Some params (biases, extra norms) aren't counted, so estimate is conservative.
        assert 0.5 < size_gb < 5.0
    
    def test_moe_layer_larger_than_dense(self):
        """MoE layer is much larger than dense layer."""
        dense_size = _estimate_layer_size(QWEN3_CODER_NEXT_CONFIG, is_moe_layer=False,
                                           num_experts=0, shared_expert=False, bytes_per_param=2)
        moe_size = _estimate_layer_size(QWEN3_CODER_NEXT_CONFIG, is_moe_layer=True,
                                         num_experts=128, shared_expert=False, bytes_per_param=2)
        assert moe_size > dense_size * 5  # MoE should be much larger
    
    def test_embedding_size(self):
        """Embedding size is reasonable."""
        size = _estimate_embedding_size(QWEN3_CODER_NEXT_CONFIG, bytes_per_param=2)
        size_gb = size / (1024**3)
        # 151936 * 3584 * 2 bytes ≈ 1.0GB (+ lm_head if not tied)
        assert 0.5 < size_gb < 3.0
