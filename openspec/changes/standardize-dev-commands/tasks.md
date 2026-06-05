## 1. Add the four missing canonical commands

- [x] 1.1 Add `.claude/commands/fix-formatting.md` (adapt from `sleap-roots-contracts`; use
  `src/sleap_roots_analyze tests` paths to match `/lint`).
- [x] 1.2 Add `.claude/commands/validate-env.md` (adapt from `sleap-roots-contracts`; import
  smoke test imports `sleap_roots_analyze`; keep library-style — no Git LFS / data-fixture checks).
- [x] 1.3 Add `.claude/commands/copilot-review.md` (copy from `sleap-roots-contracts`; owner/name
  stay dynamic via `gh repo view` — no hardcoded repo).
- [x] 1.4 Add `.claude/commands/new-feature.md` (adapt from `sleap-roots-contracts`; orchestrates
  branch → `/openspec:proposal` → `/tdd` → `/pre-merge-check` → `/openspec:archive`; replace the
  contracts schema/Pydantic conventions with this repo's pipeline/config conventions).

## 2. Drop the superseded commands

- [x] 2.1 Delete `.claude/commands/black.md` (covered by `/lint` + `/fix-formatting`).
- [x] 2.2 Delete `.claude/commands/generate-pr-review.md` (covered by `/review-pr`).

## 3. Upgrade pre-merge-check to best-of

- [x] 3.1 Rewrite `.claude/commands/pre-merge-check.md`: add Phase 3.5 pre-PR `/review-pr`
  subagent self-review (run against the local branch diff) and an OpenSpec `validate --strict`
  step; route Copilot-comment triage through `/copilot-review` (drop the hardcoded
  `talmolab/sleap-roots-analyze` API paths).
- [x] 3.2 Keep this repo's pipeline-specific phases (coverage report with `--cov`, CI status via
  `gh pr checks`, final verification) and its `src/sleap_roots_analyze` paths.

## 4. Verify (replaces pytest TDD — this is tooling/docs)

- [x] 4.1 Final command set == canonical list + this repo's domain commands, nothing extra.
- [x] 4.2 No dangling refs to dropped commands: `rg -n "generate-pr-review|/black\b" .claude
  CLAUDE.md openspec` returns only legitimate Black-formatter URL / archived-history matches.
- [x] 4.3 No stray source names in copied commands: `rg -n
  "sleap_roots_contracts|sleap-roots-contracts" .claude/commands` returns empty.
- [x] 4.4 `uv run black --check src/sleap_roots_analyze tests` and
  `uv run ruff check src/sleap_roots_analyze` still pass.
- [x] 4.5 `openspec validate standardize-dev-commands --strict` passes.

## 5. Review and ship

- [ ] 5.1 Run `/review-pr` self-review on the local branch before opening the PR.
- [ ] 5.2 Run `/pre-merge-check`, then `gh pr create` with "Closes #126" in the body.
- [ ] 5.3 After merge, run `/openspec:archive standardize-dev-commands`.
