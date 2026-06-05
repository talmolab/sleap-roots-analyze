## 1. Adopt the canonical subagent-team review-pr

- [x] 1.1 Rewrite `.claude/commands/review-pr.md` from the `sleap-roots` subagent-team body,
  retargeted to this repo's analysis domain (pandas/sklearn/scipy/statsmodels, NetworkX DAG
  pipeline, omegaconf configs, CLI; no `.slp` loading).
- [x] 1.2 Retarget the five review lenses: Code Quality & Architecture; Testing & TDD; Statistical
  Rigor & Reproducibility; Performance/Memory & Cross-Platform; Behavioural Correctness & Edge Cases.
- [x] 1.3 Add two modes — existing-PR (by number, posts review to GitHub) and pre-PR local-diff
  (`git diff origin/main...HEAD` against the resolved default branch when no PR exists). Keep
  owner/name/base-branch dynamic via `gh repo view`; own-PR
  posts as `--comment` with a verdict banner.
- [x] 1.4 Use this repo's CI reality in the subagent prompts (lint on ubuntu; tests on
  ubuntu/windows/macOS, Python 3.11; `tests/data/` fixtures; coverage via `--cov`).

## 2. Restore pre-merge-check Phase 3.5

- [x] 2.1 Revert `pre-merge-check` Phase 3.5 to run `/review-pr` (local-diff mode) for the pre-PR
  self-review; update the integration list and Output block accordingly. `/copilot-review` stays
  in Phase 6; `/code-review` remains a lighter alternative.

## 3. Verify (tooling/docs — no pytest)

- [x] 3.1 No stray source-repo names: `rg -n "sleap_roots\b|sleap-roots\b" .claude/commands/review-pr.md`
  finds only intentional references; no `sleap-io`/`.slp`/SLEAP-loading content remains.
- [x] 3.2 `uv run black --check src/sleap_roots_analyze tests` and `uv run ruff check
  src/sleap_roots_analyze` still pass.
- [x] 3.3 `openspec validate adopt-canonical-review-pr --strict` passes.

## 4. Ship (dogfood the new command)

- [x] 4.1 Run `/pre-merge-check`; open the PR.
- [x] 4.2 Run the new `/review-pr` (subagent team) on the PR as the self-review.
- [ ] 4.3 After merge, run `/openspec:archive adopt-canonical-review-pr`.
