## Why

The project is at version 0.0.1 and has never been published to PyPI. The packaging
metadata, CI workflows, documentation, and versioning infrastructure all need updates
to support a correct, reproducible first pre-release (0.1.0a1) following PyPI and uv
best practices.

## What Changes

- Add missing PyPI metadata to `pyproject.toml` (license, classifiers, keywords, project URLs)
- Remove unused `twine` from dev dependencies
- Switch `__init__.py` and `cli.py` from hardcoded versions to dynamic versioning via `importlib.metadata`
- Add `tests/test_packaging.py` for version/metadata tests (TDD: tests first)
- Update `tests/test_cli.py` to use dynamic version assertion
- Overhaul `build.yml` with validation job (version/tag/changelog checks, test suite, wheel install test) while keeping `uv publish` with trusted publishing
- Update `version.yml` and `ci.yml` to use `setup-uv@v6`
- Fix documentation inconsistencies: CHANGELOG duplicate headers (Added, Changed, Fixed), MIT→GPLv3 license reference, placeholder dates, README/CONTRIBUTING Python version badges
- Update `RELEASE_PROCESS.md` to match actual workflow capabilities
- Update stale stats in `docs/testing.md` and `openspec/project.md`
- Update `/prepare-release` command to reflect dynamic versioning (no twine, no __init__.py sync)

## Impact

- Affected specs: `developer-tooling`
- Affected code:
  - `pyproject.toml` (metadata additions, remove twine dep)
  - `src/sleap_roots_analyze/__init__.py` (dynamic versioning)
  - `src/sleap_roots_analyze/cli.py` (dynamic version in click version_option)
  - `tests/test_packaging.py` (new: version/metadata tests)
  - `tests/test_cli.py` (update hardcoded version assertion)
  - `.github/workflows/build.yml` (validation + publish overhaul)
  - `.github/workflows/version.yml` (setup-uv@v6)
  - `.github/workflows/ci.yml` (setup-uv@v6)
  - `docs/CHANGELOG.md` (formatting fixes)
  - `docs/RELEASE_PROCESS.md` (rewrite to match reality)
  - `docs/CONTRIBUTING.md` (Python version fix)
  - `docs/testing.md` (stale statistics update)
  - `openspec/project.md` (test count fix)
  - `README.md` (badge fix)
  - `.claude/commands/prepare-release.md` (dynamic versioning, remove twine refs)
