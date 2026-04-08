# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

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
