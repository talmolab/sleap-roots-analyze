---
description: Comprehensive pre-merge verification workflow
---

# Pre-Merge Check

Full pre-merge verification: local CI, a pre-PR subagent self-review, OpenSpec validation, PR
creation, CI monitoring, and Copilot-comment triage. Run before opening a PR and before merging.

## Phase 1: Code quality

```bash
uv run black --check src/sleap_roots_analyze tests
uv run ruff check src/sleap_roots_analyze
```
If either fails, fix with `/fix-formatting`, then re-run.

## Phase 2: Tests, coverage, OpenSpec

```bash
# Full test suite with coverage (CI uses --cov-report=xml)
uv run pytest --cov=src/sleap_roots_analyze --cov-report=xml --cov-report=term-missing --durations=-1 tests/

# If an OpenSpec change is in flight:
openspec list
openspec validate <change-id> --strict
```
Confirm the associated OpenSpec tasks are checked off (`openspec list`).

## Phase 3: Documentation

- Docstrings current for all changed code (google convention).
- README / docs updated if the public API or pipeline behavior changed.
- OpenSpec tasks checked off.

## Phase 3.5: Pre-PR self-review (do this BEFORE creating the PR)

Run `/review-pr` on the **local branch diff** (pass the branch name, not a PR number — the PR
doesn't exist yet). This launches the critical-review subagent team against the change the
same way they'd review an external PR.

**Rationale:** Copilot reliably flags exactly what this team would catch (e.g. a test that
bypasses the path it was meant to regression-test). Running our own review pre-PR fixes those
in one iteration instead of two, and avoids burning a Copilot review cycle. If any BLOCKING /
IMPORTANT findings come back, fix them and restart from Phase 1.

## Phase 4: Create / update the PR

```bash
gh pr create --title "<title>" --body "<summary, test results, OpenSpec link if any>"
```

## Phase 5: CI monitoring

```bash
gh pr checks
```
Investigate any failure before proceeding.

## Phase 6: Copilot + review feedback triage

Fetch Copilot's comments with `/copilot-review`, then categorize:

- **CRITICAL** — data consistency issues, incorrect statistical calculations, broken
  functionality, security vulnerabilities
- **HIGH** — type-safety violations, missing tests, real bugs, maintainability
- **MEDIUM** — code quality, performance, style
- **LOW** — docs, minor refactors, nice-to-haves
- **NO ACTION** — working as designed / false positive / already fixed

Fix CRITICAL + HIGH now (re-run tests after each); file issues for MEDIUM/LOW. Evaluate each
suggestion on its merits and note why if you decline (see superpowers:receiving-code-review).

## Phase 7: Changelog

Run `/update-changelog` to add a `[Unreleased]` entry. (If `docs/CHANGELOG.md` doesn't exist
yet, create it in Keep-a-Changelog format first.)

## Phase 8: Final verification

```bash
uv run black --check src/sleap_roots_analyze tests && uv run ruff check src/sleap_roots_analyze
uv run pytest tests/
git push
gh pr checks
git fetch origin main && git merge-base --is-ancestor origin/main HEAD   # branch up to date with main
```

## Output

```markdown
# Pre-Merge Check Results
## Code Quality:  [x] black  [x] ruff
## Tests:         [x] pytest (X passed)  [x] coverage
## OpenSpec:      [x] validated (or N/A)
## Self-review:   [x] /review-pr clean (or findings fixed)
## PR:            [x] #X created, checks green
## Copilot:       [x] CRITICAL/HIGH addressed; MEDIUM/LOW filed
## Changelog:     [x] entry added (or N/A)
## Status: READY TO MERGE
```

## Integration

- `/lint` — quick formatting and linting check
- `/fix-formatting` — auto-fix formatting and lint issues
- `/coverage` — detailed test coverage analysis
- `/run-ci-locally` — run exact CI checks locally
- `/review-pr` — pre-PR / PR subagent review
- `/copilot-review` — fetch Copilot review comments (repo-agnostic)
- `/update-changelog` — update CHANGELOG before merge

## When to Use

- Before opening a PR (Phase 1–3.5)
- After receiving Copilot review comments
- Before requesting final review from maintainers
- Before merging to main
