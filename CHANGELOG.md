# Changelog

All notable changes to GhostRelay are documented here.

## 0.4.0-beta.2 - 2026-02-07

### Added

- Generic API provider mode in settings (`modelSource=api`)
- API presets: OpenAI, Gemini API, Anthropic API, OpenRouter, Groq, xAI, LM Studio, vLLM
- API provider auth/model list checks for OpenAI-compatible and Anthropic Messages formats
- README screenshot sections wired to `docs/images/*`

### Changed

- Tool policy apply path generalized to provider id (`agents.list[].tools.byProvider.<provider>.allow`)
- Runtime model-source handling expanded to include `api` in chat/companion paths
- Settings normalization expanded with `apiProvider` schema and persisted defaults

### Verified

- Core Python files compile (`python -m py_compile`)
- Ollama Gemma3 behavior validated: text chat works, tool-calling returns `does not support tools`

## 0.4.0-beta.1 - 2026-02-08

### Added

- Public release policy files (`LICENSE`, `NOTICE`, `.gitignore`)
- Allowlist-based public bundle workflow in `release_strategy/`
- Optional commit safety guard (`.pre-commit-config.yaml`, blocklist script)
- README sections for attribution/privacy/patch-style integration
- Temporary versioning files (`VERSION`, `CHANGELOG.md`)

### Verified

- Core Python files compile (`py_compile`)
- Public bundle script runs and excludes private files
- Private GitHub repo bootstrap path validated

### In Progress

- Full UI E2E stability checks
- Multi-agent orchestration automation hardening
