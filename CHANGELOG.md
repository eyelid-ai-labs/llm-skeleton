# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

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
