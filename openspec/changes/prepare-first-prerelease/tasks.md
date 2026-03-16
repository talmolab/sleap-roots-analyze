## 0. Tests First (TDD Red Phase)

- [ ] 0.1 Add `tests/test_packaging.py` with tests for:
  - `sleap_roots_analyze.__version__` returns a valid PEP 440 version string
  - `sleap_roots_analyze.__version__` is not `"unknown"` in a normal install
  - `__version__` matches `importlib.metadata.version("sleap-roots-analyze")`
- [ ] 0.2 Update `tests/test_cli.py:137`: replace hardcoded `"0.0.1"` assertion with dynamic version check using `importlib.metadata.version("sleap-roots-analyze")`
- [ ] 0.3 Run tests to confirm 0.1 fails (red) and 0.2 fails (red)

## 1. Packaging Metadata

- [ ] 1.1 Add `license = "GPL-3.0-or-later"` to pyproject.toml
- [ ] 1.2 Add `classifiers` (Development Status :: 3 - Alpha, License :: OSI Approved :: GNU General Public License v3 or later, Programming Language :: Python :: 3.11, Topic :: Scientific/Engineering :: Bio-Informatics)
- [ ] 1.3 Add `keywords` for PyPI discoverability (e.g., sleap, roots, phenotyping, plant-science)
- [ ] 1.4 Add `[project.urls]` (Homepage, Repository, Issues, Changelog)
- [ ] 1.5 Remove `twine` from `[dependency-groups] dev` (this repo uses `uv publish`)

## 2. Dynamic Versioning

- [ ] 2.1 Replace hardcoded `__version__ = "0.0.1"` in `__init__.py` with `importlib.metadata.version("sleap-roots-analyze")` and `PackageNotFoundError` fallback to `"unknown"`
- [ ] 2.2 Update `cli.py:59`: replace `@click.version_option(version="0.0.1", ...)` to use `sleap_roots_analyze.__version__` or `importlib.metadata.version()`
- [ ] 2.3 Run tests to confirm 0.1 and 0.2 now pass (green)
- [ ] 2.4 Verify `uv run python -c "import sleap_roots_analyze; print(sleap_roots_analyze.__version__)"` works

## 3. CI Workflow Overhaul

- [ ] 3.1 Rewrite `build.yml` with validate-release job (tag/version/changelog checks, tests, wheel install test)
- [ ] 3.2 Add build-and-publish job (uv build, uv publish with trusted publishing)
- [ ] 3.3 Ensure `id-token: write` permission is scoped to the publish job only
- [ ] 3.4 Update `version.yml` to use `setup-uv@v6`
- [ ] 3.5 Update `ci.yml` to use `setup-uv@v6` for consistency across all workflows

## 4. Documentation Fixes

- [ ] 4.1 Fix CHANGELOG.md: merge ALL duplicate section headers (`### Added` at lines 10/76, `### Changed` at lines 51/123, `### Fixed` at lines 66/139)
- [ ] 4.2 Fix CHANGELOG.md: change "MIT License" reference to "GPLv3" (line 224)
- [ ] 4.3 Fix CHANGELOG.md: remove or fix placeholder date `2025-01-XX` (line 154)
- [ ] 4.4 Fix README.md: update Python version badge from `3.9+` to `3.11+`
- [ ] 4.5 Rewrite `docs/RELEASE_PROCESS.md` to match actual build.yml and version.yml capabilities
- [ ] 4.6 Fix `docs/CONTRIBUTING.md`: update Python version from `3.9+` to `3.11+`
- [ ] 4.7 Update `docs/testing.md`: fix stale coverage/test statistics (lines 258-274)
- [ ] 4.8 Update `openspec/project.md`: fix test count from `150+` to actual (~1939)

## 5. Command Update

- [ ] 5.1 Update `/prepare-release` command: remove `__init__.py` version sync steps (dynamic versioning handles it)
- [ ] 5.2 Update `/prepare-release` command: remove any twine references

## 6. Validation

- [ ] 6.1 Run `uv build` and verify wheel contents
- [ ] 6.2 Run `uv run pytest tests/ -x -q`
- [ ] 6.3 Run `uv run black --check src/sleap_roots_analyze tests`
- [ ] 6.4 Run `uv run ruff check src/sleap_roots_analyze tests`
- [ ] 6.5 Test wheel installation: `uv run --isolated --with dist/*.whl python -c "import sleap_roots_analyze; print(sleap_roots_analyze.__version__)"`
- [ ] 6.6 Test CLI entry point: `uv run --isolated --with dist/*.whl sleap-roots-analyze --version`
