# Release Process

This document describes the release process for `sleap-roots-analyze` using uv and GitHub Actions.

## Overview

This project uses:
- **`uv build`** with the `uv_build` backend to create wheel and sdist
- **`uv publish`** with [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no tokens needed in CI
- **`uv version`** for semantic version management
- **Dynamic versioning** via `importlib.metadata` — version is defined only in `pyproject.toml`

## Version Management

### Version Bump Workflow (GitHub Actions)

Use the **Version Bump** workflow for version changes that create a PR:

1. Go to Actions → Version Bump → Run workflow
2. Select bump type: `patch`, `minor`, `major`, `alpha`, `beta`, `rc`, or `stable`
3. Or enter a custom version string
4. Review and merge the created PR

The workflow only updates `pyproject.toml` — `__init__.py` uses dynamic versioning and does not need updating.

### Local Version Commands

```bash
uv version                    # Check current version
uv version --bump patch       # 0.1.0 → 0.1.1
uv version --bump minor       # 0.1.0 → 0.2.0
uv version --bump alpha       # 0.1.0 → 0.2.0a1
uv version --bump stable      # 0.2.0a1 → 0.2.0
uv version 1.0.0              # Set specific version
```

## Release Workflow

### Pre-release Checklist

Before creating a release:

```bash
uv run black --check src/sleap_roots_analyze tests  # Formatting
uv run ruff check src/sleap_roots_analyze            # Linting
uv run pytest -m "not integration" tests/ -x -q      # Tests
uv build                                              # Build
```

Or use the Claude Code command: `/prepare-release`

### Creating a Release

1. Ensure version in `pyproject.toml` is correct
2. Ensure `docs/CHANGELOG.md` has an entry for this version
3. Create a GitHub Release with tag `vX.Y.Z` (e.g., `v0.1.0a1`)
4. For pre-releases, check "Set as a pre-release" on GitHub

The **Build** workflow automatically:
1. Validates tag matches `pyproject.toml` version
2. Validates changelog entry exists
3. Runs linting and tests
4. Builds wheel and sdist
5. Verifies wheel installs correctly and CLI entry point works
6. Publishes to PyPI via trusted publishing

### Pre-release Version Progression

```
0.1.0 → 0.2.0a1 → 0.2.0a2 → 0.2.0b1 → 0.2.0rc1 → 0.2.0
```

Pre-releases publish to **regular PyPI** (not TestPyPI) and are marked as pre-release on GitHub.

## Setup Requirements

### PyPI Trusted Publishing

Configure trusted publishing on PyPI (no tokens needed):

1. Go to [PyPI](https://pypi.org) → Your Project → Settings → Publishing
2. Add a "pending trusted publisher" with:
   - Repository: `talmolab/sleap-roots-analyze`
   - Workflow: `build.yml`
   - Environment: (leave blank)

For the first release, create a "pending trusted publisher" before the package exists on PyPI.

## Troubleshooting

### Package not appearing on PyPI
- Wait 1-2 minutes for indexing
- Check workflow logs in GitHub Actions

### Version conflicts
- PyPI does not allow re-uploading the same version
- Bump to a new version and re-release

### Build validation fails
- Tag version must match `pyproject.toml` version exactly
- Changelog must contain `[X.Y.Z]` entry for the release version

## References

- [UV Documentation](https://docs.astral.sh/uv/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
- [Semantic Versioning](https://semver.org/)
