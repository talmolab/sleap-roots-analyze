# developer-tooling Specification

## Purpose
TBD - created by archiving change add-claude-commands. Update Purpose after archive.
## Requirements
### Requirement: Pre-Merge Verification

The pre-merge check command SHALL perform comprehensive PR readiness verification before
merging, including local CI checks, a pre-PR self-review of the local diff, OpenSpec validation,
GitHub Copilot comment triage, and CI status verification. Copilot comments SHALL be fetched via the
repo-agnostic `/copilot-review` command (no hardcoded repository). The command SHALL retain this
repository's pipeline-specific phases (coverage reporting via `--cov`, CI status via
`gh pr checks`, and final verification) and use this repository's `src/sleap_roots_analyze` paths.

#### Scenario: Run pre-merge check on current PR

- **GIVEN** user is on a feature branch with an open PR
- **WHEN** user invokes `/pre-merge-check`
- **THEN** local CI checks run (Black, Ruff, pytest with coverage)
- **AND** GitHub Copilot review comments are fetched via `/copilot-review` and categorized by priority
- **AND** GitHub Actions CI status is verified via `gh pr checks`
- **AND** issues are reported with actionable fix guidance

#### Scenario: Pre-PR self-review of the local diff

- **GIVEN** user has finished implementing a change but has not yet opened the PR
- **WHEN** the pre-merge check reaches the pre-PR self-review phase
- **THEN** the local branch diff is critically reviewed via `/review-pr` (the subagent review
  team in local-diff mode), applying the BLOCKING / IMPORTANT / SUGGESTION severity rubric
- **AND** any BLOCKING / IMPORTANT findings are fixed before the PR is created
- **AND** `/code-review` remains available as a lighter single-pass alternative

#### Scenario: OpenSpec validation during pre-merge

- **GIVEN** an OpenSpec change is in flight for the branch
- **WHEN** the pre-merge check runs
- **THEN** `openspec validate <change-id> --strict` is run and must pass
- **AND** the associated OpenSpec tasks are confirmed checked off

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

### Requirement: Auto-Fix Formatting

The system SHALL provide a `/fix-formatting` command that auto-fixes formatting and lint issues
as the write-complement to `/lint` (which only checks), so the two stay in lockstep with the CI
formatting rules.

#### Scenario: Auto-fix formatting and lint

- **WHEN** user invokes `/fix-formatting`
- **THEN** Black auto-formats: `uv run black src/sleap_roots_analyze tests`
- **AND** Ruff applies its auto-fixable lint fixes: `uv run ruff check --fix src/sleap_roots_analyze tests`
- **AND** the command notes what is NOT auto-fixed (docstring content, logic, naming) and directs
  the user to run `/lint` to confirm the remaining issues are clean

### Requirement: Development Environment Validation

The system SHALL provide a `/validate-env` command that verifies the development environment is
correctly set up using `uv`, suitable for running after cloning, after dependency changes, or
when imports/tests fail unexpectedly. As a pure library, this repo has no Git LFS or test-data
fixtures, so the command SHALL NOT include LFS pointer or data-presence checks.

#### Scenario: Validate the dev environment

- **WHEN** user invokes `/validate-env`
- **THEN** `uv` availability and the project Python version are checked (`uv --version`, `uv run python --version`)
- **AND** dependencies are confirmed synced from the lockfile (`uv sync`, `uv tree`)
- **AND** an import smoke test runs: `uv run python -c "import sleap_roots_analyze ..."`
- **AND** the test suite is exercised (`uv run pytest -q`)
- **AND** common fixes are listed for the typical failure modes (uv missing, deps not synced)

### Requirement: Copilot Review Triage

The system SHALL provide a `/copilot-review` command that fetches GitHub Copilot's inline
review comments and review summaries for a PR so they can be triaged before merge. The command
SHALL resolve the repository owner/name dynamically via `gh repo view` and SHALL NOT hardcode a
repository.

#### Scenario: Fetch Copilot comments for the current PR

- **WHEN** user invokes `/copilot-review` on a branch with an open PR
- **THEN** the repository is resolved dynamically (e.g. `gh repo view --json nameWithOwner`)
- **AND** Copilot inline comments and review summaries are fetched (matching both the `Copilot`
  user and the `copilot-pull-request-reviewer[bot]` author)
- **AND** the output is suitable for triage alongside `/review-pr` and `/pre-merge-check`

### Requirement: New Feature Orchestration

The system SHALL provide a `/new-feature` command that serves as the single entry point for
starting a feature, orchestrating the lab's spec-driven + TDD workflow end to end.

#### Scenario: Start a new feature

- **WHEN** user invokes `/new-feature`
- **THEN** the feature scope is clarified in one sentence before any scaffolding
- **AND** a feature branch is created off `main`
- **AND** an OpenSpec proposal is scaffolded via `/openspec:proposal` and validated with
  `openspec validate <change-id> --strict`
- **AND** implementation proceeds task-by-task via `/tdd`, kept in sync via `/openspec:apply`
- **AND** `/pre-merge-check` is run before opening the PR
- **AND** `/openspec:archive <change-id>` folds the change into the specs after merge

### Requirement: PR Code Review Subagent Team

The `/review-pr` command SHALL conduct an adversarial code review by launching a team of
specialized subagents in parallel, each with a distinct review lens, rather than only fetching
existing review comments. The lenses SHALL cover: code quality & architecture; testing & TDD
discipline; statistical rigor & reproducibility; performance, memory & cross-platform safety;
and behavioural correctness & edge cases. After the subagents return, the command SHALL
deduplicate and prioritize findings by severity (BLOCKING / IMPORTANT / SUGGESTION), determine a
verdict (APPROVE / COMMENT / REQUEST_CHANGES), and — when a PR exists — post the synthesized
review to GitHub. The repository owner/name SHALL be resolved dynamically via `gh repo view`
(no hardcoded repository).

#### Scenario: Review an existing PR

- **WHEN** user invokes `/review-pr` with a PR number (or on a branch with an open PR)
- **THEN** the PR diff, description, CI status, and existing Copilot comments are gathered
- **AND** the specialized subagents review the change in parallel across their lenses
- **AND** the synthesized, severity-ranked review with a verdict is posted to the PR

#### Scenario: Pre-PR review of the local branch diff

- **GIVEN** the user has implemented a change but not yet opened a PR
- **WHEN** user invokes `/review-pr` and no PR exists for the branch
- **THEN** the command reviews the local branch diff (`git diff origin/main...HEAD`, against the
  resolved default branch) with the same subagent team and severity rubric
- **AND** findings are reported locally so BLOCKING / IMPORTANT items can be fixed before the PR

#### Scenario: Own-PR review posts as a comment

- **WHEN** the PR author is the authenticated user
- **THEN** the command does not attempt `--approve` / `--request-changes` (GitHub forbids self-review)
- **AND** the verdict is posted via `--comment` with a verdict banner

### Requirement: Type-Check Gate

CI SHALL run `mypy` against `src/sleap_roots_analyze` on every pull request and filter its
output through a committed baseline file so that pre-existing type errors do not block PRs
while newly introduced type errors do. The gate SHALL run as a dedicated CI job (modeled on
the existing reproducibility/serialization gates) so it can serve as a required status check.
The gate SHALL use the standard `mypy-baseline` tool (no bespoke type-checking scripts). The
mypy configuration SHALL start lenient — targeting the package only, tolerating untyped
third-party imports, with `disallow_untyped_defs` as the single initial ratchet knob.

#### Scenario: Pre-existing type debt does not block a PR

- **GIVEN** the committed `.mypy-baseline.txt` records the current set of mypy errors
- **WHEN** a PR is opened that does not introduce new type errors
- **THEN** the `type-check` CI job runs `mypy src/sleap_roots_analyze` piped through
  `mypy-baseline filter`
- **AND** the job passes because every reported error matches the baseline

#### Scenario: A new untyped def fails the gate

- **GIVEN** a PR adds a function (public or private) without type annotations in `src/sleap_roots_analyze`
- **WHEN** the `type-check` CI job runs
- **THEN** mypy reports a `disallow_untyped_defs` error that is absent from the baseline
- **AND** `mypy-baseline filter` exits non-zero, failing CI
- **AND** adding the missing annotations makes the job pass without editing the baseline

#### Scenario: Baseline regenerates when existing debt is paid down

- **GIVEN** a contributor fixes a pre-existing type error that is recorded in the baseline
- **WHEN** they run `mypy src/sleap_roots_analyze | mypy-baseline sync` and commit the updated
  `.mypy-baseline.txt`
- **THEN** the baseline shrinks to reflect the resolved error
- **AND** the gate continues to pass, having ratcheted tighter

### Requirement: Type-Check Ratchet Documentation

`docs/CONTRIBUTING.md` SHALL document the mypy ratchet in at least one paragraph, covering: the
command to run mypy locally, what the frozen baseline means, the expectation that new
definitions (public or private) are typed, and how to regenerate the baseline when existing debt
is resolved.

#### Scenario: Contributor learns the ratchet from CONTRIBUTING

- **WHEN** a contributor reads `docs/CONTRIBUTING.md`
- **THEN** they find the local mypy command and an explanation that pre-existing errors are
  frozen in `.mypy-baseline.txt`
- **AND** they learn that new defs (public or private) must be typed or CI fails
- **AND** they learn to regenerate the baseline with `mypy-baseline sync` when they fix
  existing debt

