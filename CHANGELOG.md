# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.5.1] - 2026-04-08

### Fixed

- Gemma-4 (and other native HF VLMs with empty `auto_map`) now loads correctly.
  v0.5.0 guessed `AutoModelForImageTextToText` when `auto_map` was empty, which on
  some transformers versions resolves to the bare `Gemma4Model` (no LM head) instead
  of `Gemma4ForConditionalGeneration`. The new approach introspects transformers'
  internal `_model_mapping` to find which auto class actually maps the model's config
  to a class with an LM head. For Gemma-4: `AutoModelForCausalLM[Gemma4Config]` →
  `Gemma4ForConditionalGeneration` ✅.

### Changed

- `_resolve_auto_class()` now uses three strategies in order:
  1. `auto_map` lookup (unchanged — works for LLaVA, InternVL, etc.)
  2. **New:** `_model_mapping` introspection — uses `AutoConfig.for_model(model_type)`
     to get the config class, then checks each auto class's `_model_mapping` to find
     one that resolves to a class with an LM head (`ForCausalLM`,
     `ForConditionalGeneration`, etc. in the name). This is the definitive answer for
     native HF models with empty `auto_map`.
  3. Fallback to `AutoModelForCausalLM` (safest default).
- Priority order changed to `AutoModelForCausalLM` first (works for both LLMs and
  most VLMs), then `AutoModelForImageTextToText`, then `AutoModelForVision2Seq`, then
  `AutoModel`. This matches the empirical finding that `AutoModelForCausalLM` correctly
  maps Gemma4Config to `Gemma4ForConditionalGeneration`.
- Gemma-4 test mock config now has an empty `auto_map` to match the real config.json.
- 82 tests total. New `TestMappingIntrospection` class with 5 tests verifying the
  `_model_mapping` approach for gemma4, llama, unknown types, LM head detection, and
  importability.

## [0.5.0] - 2026-04-08

### Fixed

- VLM auto class resolution completely rewritten. v0.4.0 used the non-existent
  `AutoModelForConditionalGeneration`. v0.4.1 used `AutoModel` which loads the bare
  model without the LM head, causing weight key mismatches and garbage outputs.
  The new approach reads the model's `auto_map` from config.json and picks the first
  key that both (a) exists in `auto_map` and (b) is importable from the installed
  `transformers` version. Priority order: `AutoModelForImageTextToText` >
  `AutoModelForVision2Seq` > `AutoModelForCausalLM` > `AutoModel`. This means:
  - Gemma-4 (auto_map has `AutoModelForCausalLM`) → loads via `AutoModelForCausalLM`
  - LLaVA (auto_map has `AutoModelForCausalLM`) → loads via `AutoModelForCausalLM`
  - InternVL (auto_map has only `AutoModel`) → loads via `AutoModel`
  - VLMs with no auto_map → falls back to first importable class in priority order

### Changed

- New `_resolve_auto_class()` function in `probe.py` that scans the model's `auto_map`
  keys against the installed `transformers` auto classes. Separated from `_detect_vlm()`
  for testability.
- `execute_plan` in `load.py` now uses `getattr(transformers, profile.auto_class)` for
  dynamic class resolution instead of hardcoded imports. Falls back to
  `AutoModelForCausalLM` if the resolved class isn't found.
- Tokenizer loading in `execute_plan` now sets `trust_remote_code=True` for VLMs.
- 78 tests total (up from 72). New `TestAutoClassResolution` class with 6 tests covering
  auto_map-driven resolution for Gemma-4, LLaVA, InternVL, no-automap fallback, and an
  importability guarantee test that verifies every resolved class actually exists in the
  installed transformers.

## [0.4.1] - 2026-04-08

### Fixed

- VLM auto class resolution used the non-existent `AutoModelForConditionalGeneration`
  from transformers, which would crash at import time. All VLMs now resolve to
  `AutoModel`, which is the only reliable universal class — it dispatches to the correct
  architecture (Gemma4ForConditionalGeneration, LlavaForConditionalGeneration, etc.) via
  the model's own config.json `auto_map` and `architectures` fields.
- Simplified `_detect_vlm()` to always return `auto_class="AutoModel"` for VLMs instead
  of trying to pick between non-existent or narrower auto classes.
- Simplified `execute_plan` VLM branch to unconditionally use `AutoModel` instead of
  dispatching on the `auto_class` string.

## [0.4.0] - 2026-04-08

### Fixed

- VLM models (Gemma4ForConditionalGeneration, LlavaForConditionalGeneration, etc.) now
  load to GPU correctly instead of silently falling back to CPU. The root cause was
  `execute_plan` hardcoding `AutoModelForCausalLM`, which ignores `device_map` for
  conditional generation architectures. Skeleton now detects VLMs and uses the correct
  auto class (`AutoModelForConditionalGeneration` or `AutoModel`).

### Added

- `is_vlm` field on `ModelProfile`. Set to `True` when the architecture class ends with
  `ForConditionalGeneration`, `ForVision2Seq`, or `ForImageTextToText`, or when the
  config contains VLM-specific keys (`vision_config`, `vision_tower`, `audio_config`,
  `audio_tower`, `image_token_index`).
- `auto_class` field on `ModelProfile`. Stores the HuggingFace auto class name to use
  for loading (`"AutoModelForCausalLM"`, `"AutoModelForConditionalGeneration"`, or
  `"AutoModel"`). Determined from the architecture class and `auto_map` in config.json.
- `_detect_vlm()` function in `probe.py` that detects VLM architectures from both the
  architecture class name and config.json keys, and resolves the correct auto class.
- `execute_plan` now dynamically selects the auto class based on `profile.auto_class`
  instead of always using `AutoModelForCausalLM`.
- `get_load_kwargs()` now sets `trust_remote_code=True` for VLMs (many VLMs use custom
  modeling code even without an explicit `auto_map` for `AutoModelForCausalLM`).
- `_detect_custom_code()` now also checks `AutoModelForConditionalGeneration` and
  `AutoModel` keys in `auto_map` when extracting the custom model class name.
- 10 new tests: 8 for VLM detection (Gemma-4, LLaVA, InternVL, audio models,
  vision_config key detection, standard LLM negative cases) and 2 for VLM load kwargs
  (trust_remote_code behavior). Total test count: 72.

## [0.3.1] - 2026-04-08

### Fixed

- Replace `torch_dtype` kwarg with `dtype` in `get_load_kwargs()` for transformers 5.5+
  compatibility. The old `torch_dtype` parameter was deprecated and causes warnings on
  recent transformers versions.

## [0.3.0] - 2026-04-08

### Added

- VLM (Vision-Language Model) device map support. Skeleton now parses the safetensors
  weight map to detect actual module paths instead of assuming decoder-only conventions.
  This fixes the `device_map keys do not match any submodules` error when loading VLMs
  like Gemma-4, LLaVA, and InternVL whose language layers live under nested prefixes
  such as `model.language_model.layers.X`.
- New `_detect_layer_prefix()` function that scans safetensors weight names to find the
  correct transformer layer prefix. When multiple prefixes exist (e.g. language model +
  vision encoder), it picks the one with the most layers.
- New `_detect_special_modules()` function that identifies embed_tokens, lm_head, norm,
  and extra top-level modules (vision_tower, embed_vision, multi_modal_projector, etc.)
  from the weight map.
- Five new fields on `ModelProfile`: `layer_prefix`, `embed_module`, `lm_head_module`,
  `norm_module`, and `extra_modules`. These carry the actual module paths detected from
  the safetensors index and default to standard decoder-only paths when no index is
  available.
- Extra VLM modules (vision tower, etc.) are now placed in the device map on the same
  GPU as lm_head.
- 16 new tests covering layer prefix detection (standard decoder, Gemma-4, InternVL,
  vision-vs-language disambiguation, empty weight map fallback), special module detection
  (embed, norm, lm_head, extra modules for each VLM family), and bin-packing with VLM
  paths (correct key generation, extra module placement, quantized packing passthrough).
  Total test count: 62.

### Changed

- Refactored `_fetch_actual_size()` into `_fetch_safetensors_index()` which now returns
  both the total size and the weight map dict in a single download. This avoids a second
  network round-trip for VLM path detection.
- `pack_layers()` and `pack_layers_quantized()` now accept optional `layer_prefix`,
  `embed_module`, `lm_head_module`, `norm_module`, `extra_modules`, and
  `extra_modules_size_bytes` parameters. When provided, device map keys use the actual
  paths from the safetensors index. Fully backward-compatible — omitting the new
  parameters preserves the original `model.layers.X` behavior.
- `plan_loading()` now passes the profile's VLM path fields through to the bin-packing
  functions, so device maps are correct for both decoder-only and VLM architectures
  without any caller changes.

## [0.2.0] - 2026-04-08

### Changed

- Enhance probe_model to support VLM configurations and improve size estimation accuracy.
- Use nested text_config values when present for language-model sizing in multimodal configs.
- Prefer model.safetensors.index.json metadata total_size and rescale derived layer and quantized-size estimates accordingly.
- Extend probe_model VLM config resolution to also detect and use nested language_config and llm_config blocks when they provide stronger decoder sizing signals.
- Bump project version metadata to 0.2.0 and modernize packaging license fields for cleaner setuptools builds.
- Refresh README testing docs to use pytest tests/ -v and reflect the current 46-test suite.

### Added

- Add probe tests for VLM config resolution across text_config, language_config, and llm_config patterns.
- Add probe tests for tied vs untied embedding size estimation behavior.
