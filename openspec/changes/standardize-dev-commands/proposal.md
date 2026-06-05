## Why

Two `.claude/commands` lineages have diverged across the lab's repos — a **descriptive**
(verb-first) lineage (`sleap-roots-analyze`, this repo) and a **short** lineage
(`sleap-roots` / `salk-bloom` / `bloom-desktop`). Issue #126 standardizes on the **descriptive
names with best-of content**: adopt the richer command body wherever the lineages differ. This
repo already owns the canonical naming and several canonical bodies (`tdd`, `review-openspec`,
`prepare-release`, `update-changelog`); it is missing four generic commands and carries two
that the canonical set drops. Converging removes the per-repo drift and gives every lab repo
the same dev-command surface.

## What Changes

- **ADD** `/fix-formatting` — the auto-fix complement to `/lint`
  (`uv run black src/sleap_roots_analyze tests` + `uv run ruff check --fix src/sleap_roots_analyze tests`).
- **ADD** `/validate-env` — uv / import / env-health checks (library-style; this repo has no
  Git LFS or test-data fixtures, so no LFS/data-presence checks).
- **ADD** `/copilot-review` — fetch GitHub Copilot PR review comments, owner/name kept dynamic
  via `gh repo view` (no hardcoded repo).
- **ADD** `/new-feature` — single entry point orchestrating
  branch → `/openspec:proposal` → `/tdd` → `/pre-merge-check` → `/openspec:archive`.
- **DROP** `/black` — superseded by `/lint` (check) + `/fix-formatting` (fix).
- **DROP** `/generate-pr-review` — superseded by `/review-pr`.
- **MODIFY** `/pre-merge-check` to best-of: add a Phase 3.5 pre-PR self-review of the local diff
  (via `/code-review`, plus `/review-openspec` when a change is in flight) and an
  OpenSpec-validation step, and triage Copilot comments via `/copilot-review` (repo-agnostic)
  instead of hardcoding `talmolab/sleap-roots-analyze`. Keep this repo's pipeline-specific
  coverage/CI phases. (`/review-pr` stays scoped to triaging comments on an existing PR — it is
  a PR-comment fetcher, not a local-diff reviewer.)
- **KEEP** this repo's domain commands unchanged (`configure-run-all`, `run-pipelines`,
  `dry-run`, `validate-config`, `verify-results`, `cross-platform-summary`) and the descriptive
  command names (`pre-merge-check`, `update-changelog`, `prepare-release`, `cleanup-merged`).

## Impact

- Affected specs: `developer-tooling` (ADD four requirements, MODIFY Pre-Merge Verification).
- Affected code: `.claude/commands/` — add `fix-formatting.md`, `validate-env.md`,
  `copilot-review.md`, `new-feature.md`; delete `black.md`, `generate-pr-review.md`; rewrite
  `pre-merge-check.md`.
- Documentation/tooling only — no `src/` or test changes. Verification is by inspecting the
  final command set and running the existing lint/format checks, not pytest.
