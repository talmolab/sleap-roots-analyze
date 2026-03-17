# developer-tooling Specification

## Purpose
TBD - created by archiving change add-claude-commands. Update Purpose after archive.
## Requirements
### Requirement: Pre-Merge Verification

The pre-merge check command SHALL perform comprehensive PR readiness verification before merging, including local CI checks, GitHub Copilot comment analysis, and CI status verification.

#### Scenario: Run pre-merge check on current PR

- **GIVEN** user is on a feature branch with an open PR
- **WHEN** user invokes `/pre-merge-check`
- **THEN** local CI checks run (Black, Ruff, pytest)
- **AND** GitHub Copilot review comments are fetched and categorized by priority
- **AND** GitHub Actions CI status is verified
- **AND** issues are reported with actionable fix guidance

### Requirement: Post-Merge Cleanup

The cleanup-merged command SHALL handle branch deletion and OpenSpec archival after a PR is merged, ensuring git consistency.

#### Scenario: Clean up after PR merge

- **GIVEN** a PR has been merged to main
- **WHEN** user invokes `/cleanup-merged`
- **THEN** user switches to main and pulls latest
- **AND** local feature branch is deleted safely (using `-d` not `-D`)
- **AND** OpenSpec change is archived if one exists for this branch
- **AND** cleanup is committed and pushed

### Requirement: Local CI Runner

The run-ci-locally command SHALL mirror the GitHub Actions CI workflow defined in `.github/workflows/ci.yml` for consistent local/remote results.

#### Scenario: Run CI checks locally

- **WHEN** user invokes `/run-ci-locally`
- **THEN** Black formatting check runs on `src/sleap_roots_analyze` and `tests`
- **AND** Ruff linting runs on `src/sleap_roots_analyze`
- **AND** pytest runs the full test suite with coverage
- **AND** output clearly shows pass/fail for each check

### Requirement: Scientific Results Verification

The verify-results command SHALL validate pipeline outputs for scientific accuracy, reproducibility, and completeness.

#### Scenario: Verify cross-platform pipeline output

- **GIVEN** a completed pipeline run with output directory
- **WHEN** user invokes `/verify-results`
- **THEN** CSV output schema is verified (expected columns: achieved_power, significant_fdr, CIs)
- **AND** metadata JSON is checked for required fields (FDR method, power analysis params)
- **AND** statistical values are sanity-checked (power in [0,1], CIs ordered, p-values valid)
- **AND** anomalies are flagged (all NaN, all filtered, impossible values)

### Requirement: TDD Workflow

The tdd command SHALL provide a structured test-driven development workflow for new features, enforcing the red-green-refactor cycle.

#### Scenario: TDD for a new function

- **GIVEN** user wants to implement a new feature
- **WHEN** user invokes `/tdd`
- **THEN** failing tests are written first (red phase)
- **AND** tests are run to confirm they fail as expected
- **AND** implementation is written to pass the tests (green phase)
- **AND** tests are run to confirm they pass
- **AND** code quality is verified with lint and coverage

### Requirement: CI-Consistent Linting

The lint command SHALL match the CI workflow exactly for consistent local/remote linting results.

#### Scenario: Lint command matches CI

- **WHEN** user invokes `/lint`
- **THEN** Black check runs: `uv run black --check src/sleap_roots_analyze tests`
- **AND** Ruff check runs: `uv run ruff check src/sleap_roots_analyze`
- **AND** these match the CI workflow in `.github/workflows/ci.yml`

### Requirement: Working Coverage Analysis

The coverage command SHALL run pytest with coverage reporting using valid tooling, without referencing nonexistent scripts.

#### Scenario: Run coverage analysis

- **WHEN** user invokes `/coverage`
- **THEN** pytest runs with `--cov=src/sleap_roots_analyze --cov-report=term-missing`
- **AND** coverage summary shows per-file hit/miss statistics
- **AND** command does not reference nonexistent scripts

### Requirement: Interactive Analysis Configuration Command

The system SHALL provide a `/configure-run-all` Claude Code slash command that interactively guides the user through creating a complete, scientifically sound set of pipeline configuration files (QC config, Viz config, run manifest) for a new analysis **by copying and customizing validated golden templates**.

The command SHALL embody three core scientific values:
1. **Reproducibility**: All config files SHALL be committed to git so they are permanently tied to a specific codebase state (git SHA).
2. **Metadata preservation**: Config file headers SHALL document dataset identity, analysis date, author intent, and parameter rationale.
3. **Schema completeness**: All generated configs SHALL pass `validate_qc_config()` / `validate_viz_config()` before being written, ensuring no missing required fields.

#### Scenario: Template selection

- **WHEN** the user invokes `/configure-run-all`
- **THEN** the command SHALL ask: "Is this a grouped analysis (e.g., multiple timepoints)?"
- **AND** the command SHALL ask: "Are images available for visualization?"
- **AND** based on answers, the command SHALL select the appropriate golden template pair:
  - Grouped + with images → `qc_template_grouped.yaml` + `viz_template_with_images.yaml`
  - Grouped + no images → `qc_template_grouped.yaml` + `viz_template_no_images.yaml`
  - Ungrouped + with images → `qc_template_ungrouped.yaml` + `viz_template_with_images.yaml`
  - Ungrouped + no images → `qc_template_ungrouped.yaml` + `viz_template_no_images.yaml`

#### Scenario: Copy golden template

- **WHEN** the appropriate template is selected
- **THEN** the command SHALL use the Read tool to load the golden template from `configs/templates/<template_name>.yaml`
- **AND** the command SHALL preserve ALL fields from the template (no fields are dropped)

#### Scenario: Customize required fields only

- **AFTER** loading the template
- **THEN** the command SHALL collect the following REQUIRED field values interactively, one at a time:
  - `data.csv_path` — Path to the trait CSV file
  - `columns.barcode` — Column name for sample ID / plant barcode
  - `columns.genotype` — Column name for genotype / accession
  - `columns.replicate` — Column name for replicate / plant ID
  - `data.group_by` — (if grouped template) Column name to group by (e.g., `plant_age_days`)
  - `data.image_dir` — (if with-images template) Path to image directory
  - `data.output_dir` — Where pipeline outputs should be written
  - `pipeline_name` and `run_name` — Analysis name / identifier
- **AND** the command SHALL use the Edit tool or string replacement to update ONLY these fields in the loaded template
- **AND** all other fields (heritability threshold, PCA settings, UMAP settings, etc.) SHALL retain the template's default values unless the user explicitly requests to customize them

#### Scenario: Validate before writing

- **WHEN** all required fields have been customized
- **THEN** the command SHALL call `validate_qc_config()` on the customized QC config using the Python API
- **AND** if validation fails, the command SHALL show the error message to the user and ask them to fix the issue before proceeding
- **AND** the command SHALL NOT write any config file until validation passes

#### Scenario: Dataset inspection and guardrails

- **WHEN** the user provides a CSV path
- **THEN** the command SHALL read the CSV and report: total sample count, column names (with candidates for barcode/genotype/replicate roles), numeric trait count, and candidate group_by columns (columns with ≤20 unique values)
- **AND** the command SHALL flag any candidate group with fewer than 30 samples with a WARNING (Mahalanobis chi-squared reliability requires n≥30)
- **AND** the command SHALL flag any experiment where fewer than 3 replicates per genotype exist in any group with a WARNING (heritability estimation requires ≥3 replicates per genotype)
- **AND** the command SHALL recommend UMAP n_neighbors using: `min(15, max(2, n_samples // 4))`

#### Scenario: Backup before overwriting active configs

- **WHEN** a config file already exists at the target path in `configs/active/`
- **THEN** the command SHALL inform the user that an existing config will be overwritten
- **AND** the command SHALL offer to save a timestamped backup to `configs/archive/<original-name>_backup_<YYYYMMDD_HHMMSS>.yaml`
- **AND** the command SHALL NOT overwrite any existing config without explicit user confirmation

#### Scenario: Config file writing with self-documenting headers

- **WHEN** the user approves the configuration
- **THEN** the command SHALL write QC config, Viz config, and run manifest to `configs/active/`
- **AND** each config header SHALL include: dataset name, input CSV path, analysis date, and key parameter choices with brief rationale
- **AND** the run manifest header SHALL include the CLI command to reproduce the run

#### Scenario: User validation gate

- **WHEN** configs have been written to disk
- **THEN** the command SHALL display the full content of each config file for the user to review
- **AND** the command SHALL highlight (in text) the most consequential parameters: heritability threshold, outlier method, group_by column, min_samples_per_trait
- **AND** the command SHALL wait for explicit user approval ("looks good" / "yes" / "run it") before offering to proceed
- **AND** the command SHALL NOT invoke `/run-pipelines` automatically — it SHALL remind the user of the exact command to run

#### Scenario: Git commit after user approval

- **WHEN** the user approves the configs
- **THEN** the command SHALL stage the new/modified config files in `configs/active/`
- **AND** the command SHALL create a git commit with a message that includes: analysis run_name, dataset path, and ISO date
- **AND** the command SHALL report the resulting git SHA to the user as the reproducibility anchor
- **AND** if git commit fails (e.g., no changes, detached HEAD), the command SHALL warn the user clearly and continue without crashing

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

