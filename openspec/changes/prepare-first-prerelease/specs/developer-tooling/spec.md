## ADDED Requirements

### Requirement: Package Publishing

The system SHALL support publishing to PyPI via GitHub Actions using `uv publish` with trusted publishing (OIDC).

The build workflow SHALL validate before publishing:
- Git tag version matches `pyproject.toml` version
- `docs/CHANGELOG.md` contains an entry for the release version
- Full test suite passes
- Built wheel installs correctly in an isolated environment

The version workflow SHALL bump the version in `pyproject.toml` only; `__init__.py` SHALL use dynamic versioning via `importlib.metadata`.

#### Scenario: Pre-release to PyPI
- **WHEN** a GitHub release is created with a pre-release tag (e.g., `v0.1.0a1`)
- **THEN** the build workflow validates version consistency, runs tests, builds, and publishes to PyPI
- **AND** the release is marked as pre-release on GitHub

#### Scenario: Stable release to PyPI
- **WHEN** a GitHub release is created with a stable tag (e.g., `v0.1.0`)
- **THEN** the same validation and publish pipeline runs
- **AND** the release is not marked as pre-release

#### Scenario: Version bump via workflow
- **WHEN** the version bump workflow is triggered with a bump type
- **THEN** `pyproject.toml` version is updated
- **AND** a PR is created for review
- **AND** `__init__.py` is NOT modified (dynamic versioning)

#### Scenario: Build validation fails on version mismatch
- **WHEN** the git tag version does not match `pyproject.toml` version
- **THEN** the build workflow SHALL fail before publishing
- **AND** the error message SHALL indicate the version mismatch

#### Scenario: Build validation fails on missing changelog
- **WHEN** `docs/CHANGELOG.md` does not contain an entry for the release version
- **THEN** the build workflow SHALL fail before publishing

### Requirement: Package Metadata

The `pyproject.toml` SHALL include complete PyPI metadata:
- `license` matching the LICENSE file (GPL-3.0-or-later)
- `classifiers` for development status, license, Python version, and topic
- `keywords` for discoverability
- `[project.urls]` with Homepage, Repository, Issues, and Changelog links

The `pyproject.toml` SHALL NOT include unused build/publish tools (e.g., twine) in dependencies.

#### Scenario: PyPI page completeness
- **WHEN** the package is published to PyPI
- **THEN** the PyPI page displays license, classifiers, project links, and description

#### Scenario: No unused publishing dependencies
- **WHEN** the dev dependencies are reviewed
- **THEN** `twine` SHALL NOT be present (this repo uses `uv publish`)

### Requirement: Dynamic Versioning

The package SHALL use `importlib.metadata.version()` in `__init__.py` to derive `__version__` from installed package metadata, with a fallback for editable/development installs.

All version references SHALL use dynamic versioning — no hardcoded version strings in source code (`__init__.py`, `cli.py`, or tests).

#### Scenario: Version from installed package
- **WHEN** the package is installed via pip or uv
- **THEN** `sleap_roots_analyze.__version__` returns the version from `pyproject.toml`

#### Scenario: Development install
- **WHEN** the package is installed in editable mode
- **THEN** `sleap_roots_analyze.__version__` returns the current development version

#### Scenario: CLI version flag uses dynamic version
- **WHEN** user runs `sleap-roots-analyze --version`
- **THEN** the output reflects the dynamically resolved version from package metadata

#### Scenario: Version fallback on PackageNotFoundError
- **WHEN** `importlib.metadata.version()` raises `PackageNotFoundError`
- **THEN** `__version__` SHALL fall back to `"unknown"`
