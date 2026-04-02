# Contributing to LLM Skeleton

We welcome contributions. Here's how to help.

## Adding Model Quirks

The most valuable contributions are new entries in the known-issues database. If you encounter a model that fails to load, check if the failure was predictable from `config.json` and add it to the appropriate detection list in `probe.py`:

- `KNOWN_LIBRARY_REQUIREMENTS` — models that need specific libraries
- `KNOWN_PYTHON_REQUIREMENTS` — models that need specific Python versions
- `KNOWN_BNB_INCOMPATIBLE` — custom model classes that reject `load_in_8bit`
- `STANDARD_ARCHITECTURES` — model classes known to work with all loading methods

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests run offline — no GPU or HuggingFace access needed. They use mock configs based on real models.

## Code Style

- Type hints on all public functions
- Google-style docstrings
- Keep it minimal — this is infrastructure, not a framework

## Reporting Issues

Include:
1. Model name (HuggingFace ID)
2. GPU setup (count, type, VRAM)
3. The error message
4. Output of `probe_model("your/model")` if possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
