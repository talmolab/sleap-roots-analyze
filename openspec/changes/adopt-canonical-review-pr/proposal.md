## Why

The #126 standardization (#134) converged command *names* and added the missing generic
commands, but deliberately left this repo's `review-pr` body untouched (it wasn't in the delta
scope). That body is the **stale PR-comment fetcher** — it just pulls existing Copilot/human
comments by PR number. The canonical best-of `review-pr` (in `sleap-roots` / `salk-bloom`) is a
**multi-subagent adversarial review team** that actually reads the diff and source and produces
a synthesized verdict.

During the #134 self-review this gap surfaced concretely: the upgraded `pre-merge-check` Phase
3.5 ("pre-PR self-review") couldn't reference `/review-pr` because a comment-fetcher can't
review a diff, so it was routed to `/code-review` as a workaround. Adopting the canonical
subagent-team `review-pr` removes the workaround and gives this repo the real best-of body.

## What Changes

- **REPLACE** `.claude/commands/review-pr.md` with the canonical **subagent review-team** body,
  adapted from `sleap-roots` to this repo's domain: a root-trait **analysis** library
  (pandas / numpy / scikit-learn / scipy / statsmodels) built on a NetworkX **DAG pipeline**
  with omegaconf configs and a CLI — **not** SLEAP `.slp` loading. The five review lenses are
  retargeted (statistical correctness for H²/ANOVA/PCA/UMAP/outlier detection; config-driven
  reproducibility and metadata; NaN / small-group / replicate guardrails; pandas vectorization
  and Windows/Ubuntu/macOS portability).
- **GENERALIZE** the command to two modes: review an **existing PR** by number (posts the review
  to GitHub), and — when **no PR exists yet** — review the **local branch diff** (`git diff
  main...HEAD`) for the pre-PR self-review. Owner/name stay dynamic via `gh repo view`; own-PR
  reviews post as a `--comment` with a verdict banner.
- **REVERT** `pre-merge-check` Phase 3.5 to use `/review-pr` (local-diff mode) for the pre-PR
  self-review, undoing the `/code-review` workaround. `/copilot-review` stays the Copilot fetch
  in Phase 6; `/code-review` remains available as a lighter option.

## Impact

- Affected specs: `developer-tooling` — ADD "PR Code Review Subagent Team"; MODIFY "Pre-Merge
  Verification" (Phase 3.5 scenario points back at `/review-pr`).
- Affected code: `.claude/commands/review-pr.md` (rewrite), `.claude/commands/pre-merge-check.md`
  (Phase 3.5 + integration list).
- Documentation/tooling only — no `src/` or test changes. Verified by inspection + the existing
  lint/format/openspec checks, and by dogfooding the new `/review-pr` on this PR.
