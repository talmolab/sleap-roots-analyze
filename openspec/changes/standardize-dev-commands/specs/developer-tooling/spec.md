## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Pre-Merge Verification

The pre-merge check command SHALL perform comprehensive PR readiness verification before
merging, including local CI checks, a pre-PR subagent self-review, OpenSpec validation, GitHub
Copilot comment triage, and CI status verification. Copilot comments SHALL be fetched via the
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

#### Scenario: Pre-PR subagent self-review

- **GIVEN** user has finished implementing a change but has not yet opened the PR
- **WHEN** the pre-merge check reaches the pre-PR self-review phase
- **THEN** `/review-pr` is run against the local branch diff (branch name, not a PR number)
- **AND** any BLOCKING / IMPORTANT findings are fixed before the PR is created

#### Scenario: OpenSpec validation during pre-merge

- **GIVEN** an OpenSpec change is in flight for the branch
- **WHEN** the pre-merge check runs
- **THEN** `openspec validate <change-id> --strict` is run and must pass
- **AND** the associated OpenSpec tasks are confirmed checked off
