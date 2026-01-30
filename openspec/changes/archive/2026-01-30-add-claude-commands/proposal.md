# Add and Update Claude Commands for Development Workflow

## Problem

The sleap-roots-analyze repository has Claude commands but they have gaps and inconsistencies:

### Existing Command Issues

1. **`coverage.md`** - References nonexistent `scripts/cov_summary.py`. Command is broken.
2. **`lint.md`** - Runs `ruff format` instead of `black`, mismatching CI which uses `black --check` + `ruff check`. Inconsistent with CI workflow.
3. **`review-pr.md`** - Uses hardcoded PR #2 as example. No integration with pre-merge workflow.
4. **`update-changelog.md`** - Contains stale template referencing early development changes (e.g., `link_images_to_samples`).

### Missing Commands (compared to Ariadne/sleap-roots)

5. **No pre-merge check** - No systematic workflow to verify a PR is ready before merging (Copilot review, CI verification, test coverage).
6. **No post-merge cleanup** - No command to clean up merged branches and archive OpenSpec changes.
7. **No local CI runner** - No way to run the full CI pipeline locally before pushing (CI runs Black + Ruff + pytest on 3 platforms).
8. **No scientific verification** - No command to validate pipeline outputs for scientific accuracy and reproducibility.
9. **No TDD workflow** - No structured workflow for test-driven development, which is critical for scientific software correctness.

## Solution

### Update 4 existing commands

1. **`coverage.md`** - Fix broken reference. Use `uv run pytest --cov=src/sleap_roots_analyze --cov-report=term-missing` matching CI. Remove nonexistent script reference.
2. **`lint.md`** - Match CI: run `black --check` + `ruff check` (check mode), or `black` + `ruff check --fix` (fix mode). Current command wrongly uses `ruff format`.
3. **`review-pr.md`** - Update examples to use `$ARGUMENTS` for PR number. Add integration notes with pre-merge workflow.
4. **`update-changelog.md`** - Remove stale template. Update to reference current project state.

### Add 5 new commands

5. **`pre-merge-check.md`** - Comprehensive pre-merge verification (adapted from Ariadne/sleap-roots)
   - Run local CI (lint + tests)
   - Fetch and categorize GitHub Copilot review comments
   - Verify GitHub Actions CI passes
   - Address issues by priority: CRITICAL > HIGH > MEDIUM > LOW
   - Integration with `/review-pr`, `/coverage`, `/lint`

6. **`cleanup-merged.md`** - Post-merge cleanup (adapted from Ariadne)
   - Verify PR was merged
   - Switch to main and pull latest
   - Delete local feature branch safely (`-d` not `-D`)
   - Archive OpenSpec change via `openspec archive` (if applicable)
   - Commit and push cleanup

7. **`run-ci-locally.md`** - Run full CI locally matching `.github/workflows/ci.yml` (adapted from sleap-roots)
   - Black formatting check: `uv run black --check src/sleap_roots_analyze tests`
   - Ruff linting: `uv run ruff check src/sleap_roots_analyze`
   - Full pytest: `uv run pytest tests/`
   - Coverage: `uv run pytest --cov=src/sleap_roots_analyze --cov-report=term-missing tests/`
   - Clear pass/fail output matching CI

8. **`verify-results.md`** - Scientific accuracy verification (new, tailored for this repo)
   - Run a pipeline with specified config
   - Verify CSV output schema (expected columns present)
   - Verify metadata JSON has required fields (FDR, power analysis, CIs)
   - Cross-check statistical values (power ranges, FDR adjustment, CI coverage)
   - Compare pipeline_summary.json across runs for reproducibility
   - Flag anomalies (all power=0, NaN values, missing columns)

9. **`tdd.md`** - Test-driven development workflow (new, for scientific rigor)
   - Scaffold failing tests first for a new feature
   - Run tests to confirm red phase (tests fail as expected)
   - Implement the feature
   - Run tests to confirm green phase (tests pass)
   - Run `/lint` and `/coverage` to verify code quality
   - Commit with descriptive message linking test → implementation

### Command Integration Map

```
Development:
  /tdd              → Write tests first, implement, verify
  /lint             → Check formatting (Black + Ruff)
  /black            → Auto-format code
  /coverage         → Run tests with coverage analysis

Pre-merge:
  /run-ci-locally   → Run exact CI checks locally
  /review-pr [N]    → Review and address PR comments
  /pre-merge-check  → Full pre-merge verification workflow
  /verify-results   → Validate pipeline outputs

Post-merge:
  /cleanup-merged   → Delete branch, archive OpenSpec
  /update-changelog → Update CHANGELOG.md

Pipeline:
  /validate-config  → Validate pipeline config
  /dry-run          → Preview pipeline execution
  /run-pipelines    → Execute full pipeline suite
```

## Scope

- **In scope**: 5 new commands + 4 updated commands in `.claude/commands/`
- **Out of scope**: CI workflow modifications, code changes, new scripts

## Affected Capabilities

None - this is a tooling-only change (Claude commands). No spec deltas required.

## References

- Ariadne `.claude/commands/`: pre-merge-check.md, cleanup-merged.md, release.md
- sleap-roots `.claude/commands/`: pre-merge.md, run-ci-locally.md, debug-test.md
- sleap-roots-analyze `.github/workflows/ci.yml`: CI configuration to mirror
