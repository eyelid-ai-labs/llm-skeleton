# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Changed

- Enhance probe_model to support VLM configurations and improve size estimation accuracy.
- Use nested text_config values when present for language-model sizing in multimodal configs.
- Prefer model.safetensors.index.json metadata total_size and rescale derived layer and quantized-size estimates accordingly.
