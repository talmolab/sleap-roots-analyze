---
description: Start a new feature — branch, OpenSpec proposal, then the TDD loop
---

# New Feature

Kick off a new feature in this repo following the lab's spec-driven + TDD workflow. Use this
as the single entry point so every feature starts the same way.

## Steps

1. **Clarify scope.** Restate the feature in one sentence. If it's ambiguous or spans multiple
   capabilities, stop and clarify before scaffolding anything.

2. **Branch off main.**
   ```bash
   git checkout main && git pull && git checkout -b <kebab-feature-name>
   ```

3. **Write the OpenSpec proposal.** Run `/openspec:proposal` to scaffold
   `openspec/changes/<feature>/` with `proposal.md` (why + what), `tasks.md` (the task
   breakdown), and `specs/<capability>/spec.md` (the capability requirements). Then validate:
   ```bash
   openspec validate <feature> --strict
   ```
   Respect the approval gate — do not start implementing until the proposal is approved.

4. **Plan the work.** For small changes, `tasks.md` is the plan. For larger ones, expand each
   task into bite-sized TDD steps (one failing test → minimal code → commit).

5. **Implement task-by-task with `/tdd`.** Red → green → refactor → commit, one task at a time.
   Keep `tasks.md` in sync as you go via `/openspec:apply`.

6. **Pre-merge.** Run `/pre-merge-check` (black + ruff + full pytest + coverage, pre-PR
   self-review, OpenSpec validation, Copilot triage). Then `/pr-description` and open the PR.

7. **Archive after merge.** Run `/openspec:archive <feature>` to fold the change into the specs.

## Conventions (this repo)

- Root-trait **analysis library** (QC, heritability/ANOVA, PCA/UMAP, outlier detection, viz)
  built on a NetworkX **DAG pipeline** — see `openspec/project.md`. Reproducibility, metadata
  preservation, and schema-complete configs are the core scientific values.
- **Configs are reproducibility anchors.** New analyses are scaffolded from validated golden
  templates via `/configure-run-all` and committed to git; configs must pass
  `validate_qc_config()` / `validate_viz_config()` before use (`/validate-config`).
- `from __future__ import annotations` at the top of every module; Black (88) + Ruff +
  pydocstyle (google docstrings). Lint with `/lint`, auto-fix with `/fix-formatting`.
- Pipeline steps are pure, testable units — write the failing test first (`/tdd`).
