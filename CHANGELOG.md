# Changelog

All notable changes to GhostRelay are documented here.

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
