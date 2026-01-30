# Developer Tooling

Claude commands and workflows for development, testing, and scientific validation.

## ADDED Requirements

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
