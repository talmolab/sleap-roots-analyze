# Release Process

This document describes the release process for `sleap-roots-analyze` using UV and GitHub Actions.

## Version Management

We use [UV's version management](https://docs.astral.sh/uv/guides/package/#building-your-package) for semantic versioning.

### Quick Version Bumps

Use the **Version Bump** workflow for simple version changes that create a PR:

1. Go to Actions → Version Bump → Run workflow
2. Select bump type:
   - `patch`: Bug fixes (0.0.1 → 0.0.2)
   - `minor`: New features (0.0.1 → 0.1.0)
   - `major`: Breaking changes (0.0.1 → 1.0.0)
   - `alpha`, `beta`, `rc`: Pre-release versions
   - `stable`: Convert pre-release to stable
3. Or enter a custom version
4. Review and merge the created PR

### Build and Release Workflow

The **Build and Release** workflow handles version bumping, building, and publishing in one go:

#### Manual Release (Recommended)

1. Go to Actions → Build and Release → Run workflow
2. Configure options:
   - **Version bump**: Select the bump type
   - **Pre-release type**: For pre-releases (alpha, beta, rc)
   - **Publish target**: Where to publish (none, testpypi, pypi)

#### Examples:

**Patch release to PyPI:**
- Version bump: `patch`
- Pre-release type: `none`
- Publish target: `pypi`

**Beta pre-release to TestPyPI:**
- Version bump: `minor`
- Pre-release type: `beta`
- Publish target: `testpypi`

**Convert pre-release to stable:**
- Version bump: `none`
- Pre-release type: `none`
- Publish target: `pypi`
- (Manually set version with `uv version --bump stable` first)

#### GitHub Release Events

Creating a GitHub release will automatically:
- **Pre-release**: Publishes to TestPyPI
- **Full release**: Publishes to PyPI

## Pre-release Strategy

### Version Progression Example

```bash
0.1.0 → 0.2.0a1 → 0.2.0a2 → 0.2.0b1 → 0.2.0rc1 → 0.2.0
```

### Pre-release Types

- **Alpha** (`a` or `alpha`): Early testing, API may change
- **Beta** (`b` or `beta`): Feature complete, fixing bugs
- **Release Candidate** (`rc`): Final testing before release

### Testing on TestPyPI

All pre-releases are automatically published to TestPyPI. Install with:

```bash
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  sleap-roots-analyze==0.2.0b1
```

## Pre-Release Checklist

Before creating a release, ensure:

1. **Tests Pass**: Run the test suite
   ```bash
   uv run pytest
   ```

2. **Coverage Check**: Verify code coverage
   ```bash
   uv run pytest --cov --cov-branch
   ```

3. **Code Formatting**: Format code with black
   ```bash
   uv run black --check src/sleap_roots_analyze tests
   ```

4. **Linting**: Check code quality
   ```bash
   uv run ruff check src/sleap_roots_analyze tests
   ```

5. **Documentation**: Update relevant docs
   - README.md
   - CHANGELOG.md
   - API documentation

## Local Development

### Version Commands

```bash
# Check current version
uv version

# Bump patch version (0.0.1 → 0.0.2)
uv version --bump patch

# Bump to beta (0.0.2 → 0.0.3b1)
uv version --bump patch --bump beta

# Increment beta (0.0.3b1 → 0.0.3b2)
uv version --bump beta

# Convert to stable (0.0.3b2 → 0.0.3)
uv version --bump stable

# Set specific version
uv version 1.0.0
```

### Building Locally

```bash
# Build package
uv build

# Check package contents
ls -la dist/

# Test installation
uv pip install dist/*.whl
```

### Publishing Manually

```bash
# Publish to TestPyPI
export UV_PUBLISH_URL=https://test.pypi.org/legacy/
export UV_PUBLISH_TOKEN=your-test-token
uv publish

# Publish to PyPI
export UV_PUBLISH_TOKEN=your-pypi-token
uv publish
```

## Setup Requirements

### Repository Secrets

Configure these in Settings → Secrets:

- `PYPI_TOKEN`: PyPI API token (for production releases)
- `TEST_PYPI_TOKEN`: TestPyPI API token (for test releases)

### Trusted Publishing (Recommended)

Instead of tokens, configure [trusted publishing](https://docs.pypi.org/trusted-publishers/):

1. Go to PyPI → Your Project → Settings → Publishing
2. Add GitHub repository as trusted publisher
3. No tokens needed!

## Workflow Features

### Build and Release Workflow

- **Version bumping**: Integrated UV version management
- **Pre-release support**: Alpha, beta, RC versions
- **TestPyPI**: Automatic for pre-releases
- **Package verification**: Tests installation before publishing
- **GitHub releases**: Automatic creation with correct tags
- **Artifacts**: Uploads built wheels and source distributions

### Version Workflow

- **Simple PR-based**: Creates PR for version changes
- **No direct commits**: All changes go through PR review
- **Custom versions**: Support for arbitrary version strings

## Best Practices

1. **Use pre-releases** for testing major changes
2. **Test on TestPyPI** before production releases
3. **Tag releases** for reproducibility
4. **Update CHANGELOG.md** with each release
5. **Use semantic versioning** consistently

## Troubleshooting

### Package not appearing on PyPI/TestPyPI

- Wait 1-2 minutes for indexing
- Check workflow logs for errors
- Verify tokens/trusted publishing is configured

### Version conflicts

- Ensure version is bumped before publishing
- PyPI doesn't allow re-uploading same version
- Use post-releases (0.1.0.post1) if needed

### Installation issues from TestPyPI

- Some dependencies may not be on TestPyPI
- Use `--extra-index-url https://pypi.org/simple/` to fetch from PyPI too

## References

- [UV Documentation](https://docs.astral.sh/uv/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)