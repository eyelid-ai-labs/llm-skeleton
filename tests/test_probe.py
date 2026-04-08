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
    _resolve_effective_config,
    _detect_layer_prefix, _detect_special_modules,
    _detect_vlm,
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


class TestVLMConfigResolution:
    def test_prefers_text_config_when_top_level_missing_layers(self):
        config = {
            "model_type": "gemma4_vlm",
            "text_config": {
                "num_hidden_layers": 42,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_attention_heads": 24,
                "vocab_size": 256000,
            },
        }

        effective = _resolve_effective_config(config)

        assert effective["num_hidden_layers"] == 42
        assert effective["hidden_size"] == 3072

    def test_prefers_language_config_when_richer_than_top_level(self):
        config = {
            "model_type": "llava",
            # Sparse top-level sizing fields (vision-first config shape)
            "num_hidden_layers": 1,
            "hidden_size": 1024,
            "language_config": {
                "num_hidden_layers": 40,
                "hidden_size": 5120,
                "intermediate_size": 13696,
                "num_attention_heads": 40,
                "vocab_size": 32000,
            },
        }

        effective = _resolve_effective_config(config)

        assert effective["num_hidden_layers"] == 40
        assert effective["hidden_size"] == 5120
        assert effective["vocab_size"] == 32000

    def test_prefers_llm_config_when_present(self):
        config = {
            "model_type": "internvl",
            "llm_config": {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "vocab_size": 151552,
            },
        }

        effective = _resolve_effective_config(config)

        assert effective["num_hidden_layers"] == 32
        assert effective["hidden_size"] == 4096

    def test_keeps_top_level_for_standard_llm(self):
        config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        effective = _resolve_effective_config(config)

        assert effective["num_hidden_layers"] == 32
        assert effective["hidden_size"] == 4096


class TestTiedEmbeddings:
    def test_tied_embeddings_are_smaller_than_untied(self):
        base = {
            "vocab_size": 10000,
            "hidden_size": 2048,
        }

        tied_size = _estimate_embedding_size({**base, "tie_word_embeddings": True}, bytes_per_param=2)
        untied_size = _estimate_embedding_size({**base, "tie_word_embeddings": False}, bytes_per_param=2)

        assert untied_size > tied_size
        assert untied_size - tied_size == base["vocab_size"] * base["hidden_size"] * 2

    def test_tie_word_embeddings_defaults_true(self):
        base = {
            "vocab_size": 10000,
            "hidden_size": 2048,
        }

        default_size = _estimate_embedding_size(base, bytes_per_param=2)
        explicit_tied_size = _estimate_embedding_size({**base, "tie_word_embeddings": True}, bytes_per_param=2)

        assert default_size == explicit_tied_size


# ─── Mock weight maps based on real VLM safetensors indices ─────────────────

GEMMA4_WEIGHT_MAP = {
    "model.language_model.embed_tokens.weight": "model-00001.safetensors",
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.0.self_attn.k_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.41.mlp.down_proj.weight": "model-00002.safetensors",
    "model.language_model.norm.weight": "model-00002.safetensors",
    "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": "model-00002.safetensors",
    "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight": "model-00002.safetensors",
    "model.vision_tower.encoder.layers.26.mlp.fc2.weight": "model-00002.safetensors",
    "model.embed_vision.weight": "model-00002.safetensors",
    "model.multi_modal_projector.linear.weight": "model-00002.safetensors",
    "lm_head.weight": "model-00002.safetensors",
}

STANDARD_DECODER_WEIGHT_MAP = {
    "model.embed_tokens.weight": "model-00001.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001.safetensors",
    "model.layers.47.mlp.down_proj.weight": "model-00002.safetensors",
    "model.norm.weight": "model-00002.safetensors",
    "lm_head.weight": "model-00002.safetensors",
}

INTERNVL_WEIGHT_MAP = {
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001.safetensors",
    "model.language_model.layers.31.mlp.down_proj.weight": "model-00002.safetensors",
    "model.language_model.embed_tokens.weight": "model-00001.safetensors",
    "model.language_model.norm.weight": "model-00002.safetensors",
    "model.vision_model.encoder.layers.0.self_attn.q_proj.weight": "model-00002.safetensors",
    "model.vision_model.encoder.layers.23.mlp.fc2.weight": "model-00002.safetensors",
    "lm_head.weight": "model-00002.safetensors",
}


class TestDetectLayerPrefix:
    def test_standard_decoder(self):
        prefix = _detect_layer_prefix(STANDARD_DECODER_WEIGHT_MAP)
        assert prefix == "model.layers"

    def test_gemma4_vlm(self):
        prefix = _detect_layer_prefix(GEMMA4_WEIGHT_MAP)
        assert prefix == "model.language_model.layers"

    def test_internvl(self):
        prefix = _detect_layer_prefix(INTERNVL_WEIGHT_MAP)
        assert prefix == "model.language_model.layers"

    def test_picks_language_model_over_vision_encoder(self):
        """When both language and vision have 'layers', pick the one with more."""
        # Gemma-4 has 42 language layers vs 27 vision encoder layers
        prefix = _detect_layer_prefix(GEMMA4_WEIGHT_MAP)
        assert "language_model" in prefix
        assert "vision" not in prefix

    def test_empty_weight_map_returns_default(self):
        prefix = _detect_layer_prefix({})
        assert prefix == "model.layers"


class TestDetectSpecialModules:
    def test_standard_decoder(self):
        modules = _detect_special_modules(STANDARD_DECODER_WEIGHT_MAP, "model.layers")
        assert modules["embed_module"] == "model.embed_tokens"
        assert modules["lm_head_module"] == "lm_head"
        assert modules["norm_module"] == "model.norm"
        assert modules["extra_modules"] == []

    def test_gemma4_vlm_embed(self):
        modules = _detect_special_modules(GEMMA4_WEIGHT_MAP, "model.language_model.layers")
        assert modules["embed_module"] == "model.language_model.embed_tokens"

    def test_gemma4_vlm_norm(self):
        modules = _detect_special_modules(GEMMA4_WEIGHT_MAP, "model.language_model.layers")
        assert modules["norm_module"] == "model.language_model.norm"

    def test_gemma4_vlm_lm_head(self):
        modules = _detect_special_modules(GEMMA4_WEIGHT_MAP, "model.language_model.layers")
        assert modules["lm_head_module"] == "lm_head"

    def test_gemma4_vlm_extra_modules(self):
        modules = _detect_special_modules(GEMMA4_WEIGHT_MAP, "model.language_model.layers")
        extras = modules["extra_modules"]
        assert "model.vision_tower" in extras
        assert "model.embed_vision" in extras
        assert "model.multi_modal_projector" in extras

    def test_internvl_extra_modules(self):
        modules = _detect_special_modules(INTERNVL_WEIGHT_MAP, "model.language_model.layers")
        extras = modules["extra_modules"]
        assert "model.vision_model" in extras


# ─── Mock configs for VLM detection ────────────────────────────────────────

GEMMA4_VLM_CONFIG = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152},
    "text_config": {
        "num_hidden_layers": 42,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "vocab_size": 256000,
    },
}

LLAVA_VLM_CONFIG = {
    "model_type": "llava",
    "architectures": ["LlavaForConditionalGeneration"],
    "vision_config": {"hidden_size": 1024},
    "language_config": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
    },
}

INTERNVL_VLM_CONFIG = {
    "model_type": "internvl_chat",
    "architectures": ["InternVLChatModel"],
    "vision_config": {"hidden_size": 1024},
    "llm_config": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
    },
    "auto_map": {
        "AutoModel": "modeling_internvl_chat.InternVLChatModel",
    },
}

AUDIO_VLM_CONFIG = {
    "model_type": "whisper_llm",
    "architectures": ["WhisperLLMForConditionalGeneration"],
    "audio_tower": {"encoder_layers": 12},
    "text_config": {
        "num_hidden_layers": 24,
        "hidden_size": 2048,
    },
}


class TestVLMDetection:
    def test_gemma4_is_vlm(self):
        result = _detect_vlm(GEMMA4_VLM_CONFIG, "Gemma4ForConditionalGeneration")
        assert result["is_vlm"] is True

    def test_gemma4_auto_class(self):
        result = _detect_vlm(GEMMA4_VLM_CONFIG, "Gemma4ForConditionalGeneration")
        assert result["auto_class"] == "AutoModelForConditionalGeneration"

    def test_llava_is_vlm(self):
        result = _detect_vlm(LLAVA_VLM_CONFIG, "LlavaForConditionalGeneration")
        assert result["is_vlm"] is True

    def test_internvl_uses_auto_model(self):
        """InternVL registers under AutoModel, not AutoModelForConditionalGeneration."""
        result = _detect_vlm(INTERNVL_VLM_CONFIG, "InternVLChatModel")
        assert result["is_vlm"] is True
        assert result["auto_class"] == "AutoModel"

    def test_audio_vlm_detected_by_config_key(self):
        """Models with audio_tower are detected as VLM even without standard suffix."""
        result = _detect_vlm(AUDIO_VLM_CONFIG, "WhisperLLMForConditionalGeneration")
        assert result["is_vlm"] is True

    def test_vision_config_key_triggers_vlm(self):
        """A model with vision_config but standard arch name is still VLM."""
        config = {"vision_config": {"hidden_size": 768}}
        result = _detect_vlm(config, "SomeCustomModel")
        assert result["is_vlm"] is True

    def test_standard_llm_not_vlm(self):
        result = _detect_vlm(QWEN3_CODER_NEXT_CONFIG, "Qwen3MoeForCausalLM")
        assert result["is_vlm"] is False
        assert result["auto_class"] == "AutoModelForCausalLM"

    def test_devstral_not_vlm(self):
        result = _detect_vlm(DEVSTRAL_CONFIG, "Ministral3ForCausalLM")
        assert result["is_vlm"] is False
