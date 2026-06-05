## ADDED Requirements

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
- **THEN** the command reviews the local branch diff (`git diff main...HEAD`) with the same
  subagent team and severity rubric
- **AND** findings are reported locally so BLOCKING / IMPORTANT items can be fixed before the PR

#### Scenario: Own-PR review posts as a comment

- **WHEN** the PR author is the authenticated user
- **THEN** the command does not attempt `--approve` / `--request-changes` (GitHub forbids self-review)
- **AND** the verdict is posted via `--comment` with a verdict banner

## MODIFIED Requirements

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
