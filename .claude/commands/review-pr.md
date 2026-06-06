# PR Code Review — Subagent Team

You are a senior scientific programmer reviewing a pull request for sleap-roots-analyze,
a pure Python library for analyzing root-trait data (quality control, broad-sense heritability,
ANOVA, PCA/UMAP, outlier detection, and visualization) from SLEAP Roots phenotyping output.
You value testing, code quality, reproducibility, metadata preservation, statistical
correctness, interpretability, and performance above all else.

## How This Command Works

This command launches **5 specialized subagents in parallel** to critically review the change.
Each subagent has a distinct review lens and is instructed to be adversarial — finding gaps,
not rubber-stamping. After all subagents return, synthesize findings into a unified review.

It works in **two modes**:

- **PR mode** — an argument PR number is given, or the current branch has an open PR. The
  review is posted to GitHub.
- **Pre-PR mode** — no PR exists yet (e.g. invoked from `/pre-merge-check` Phase 3.5). The
  change under review is the local branch diff and findings are reported locally (nothing is
  posted).

## Step 1: Gather Change Context

Detect the mode and the repo (owner / name / base branch stay dynamic — never hardcode):

```bash
REPO_OWNER=$(gh repo view --json owner --jq '.owner.login')
REPO_NAME=$(gh repo view --json name --jq '.name')
BASE_BRANCH=$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')  # usually "main"

# $1 is the PR number passed to /review-pr (if any); otherwise detect the current branch's
# open PR. An empty result means no PR exists yet => pre-PR mode. NOTE: gh failures
# (offline / unauthenticated) also yield empty, so an unexpected pre-PR classification can
# signal a gh/auth problem rather than a missing PR — run `gh auth status` if surprised.
PR_NUMBER="${1:-$(gh pr view --json number --jq '.number' 2>/dev/null)}"
```

**PR mode** (`$PR_NUMBER` is set) — run in parallel to collect everything the subagents need:

```bash
gh pr view "$PR_NUMBER" --json title,body,baseRefName,headRefName,author,labels,files
gh pr diff "$PR_NUMBER"
gh pr checks "$PR_NUMBER"

# Existing Copilot review comments (matches the bot and the Copilot user)
gh api "repos/$REPO_OWNER/$REPO_NAME/pulls/$PR_NUMBER/comments" \
  --jq '.[] | select(.user.login | contains("opilot")) | "File: \(.path):\(.line // .original_line)\n\(.body)"'
```

**Pre-PR mode** (`$PR_NUMBER` empty) — the change is the local branch diff:

```bash
git fetch origin "$BASE_BRANCH" --quiet
# three-dot `diff` = changes since the merge-base (the change under review);
# two-dot `log` = commits on HEAD not on the base branch. Keep them distinct.
git diff "origin/$BASE_BRANCH...HEAD"          # the change under review (PR_DIFF)
git log --oneline "origin/$BASE_BRANCH..HEAD"  # commit-level intent (PR_BODY substitute)
```

Also read any OpenSpec proposal linked in the PR body or present on the branch (look for
`openspec/changes/` paths) so the review is checked against the stated requirements.

## Step 2: Launch Subagent Review Team

Launch ALL 5 subagents in a single message (parallel execution). Embed the diff, the PR
description (or commit log in pre-PR mode), CI status, and any Copilot comments in each prompt.

> **Large diffs:** embedding the full diff into all 5 prompts can exceed token limits and
> truncate context. If the diff is large, embed only the changed-file list plus the most
> relevant hunks, and instruct each subagent to read the specific files it needs with Read/Grep
> (they have tool access) rather than relying on a fully inlined diff.

---

### Subagent 1: Code Quality & Architecture

```
subagent_type: "general-purpose"
description: "Review code quality and architecture"
```

**Prompt:**

> You are reviewing a change for sleap-roots-analyze, a pure Python root-trait analysis library.
> Your role: **Code Quality & Architecture Reviewer**.
> Be adversarial. Read actual source files. Find real problems, not hypotheticals.
>
> Architecture overview:
>
> - Pure Python library consumed via a CLI (`sleap_roots_analyze.cli`) and Python API.
> - Analysis built on pandas / numpy / scikit-learn / scipy / statsmodels.
> - QC workflows run as a **NetworkX DAG pipeline** of step classes under
>   `src/sleap_roots_analyze/pipeline/` (orchestrated by `pipeline_runner.py`).
> - Trait/stat logic lives in modules like `statistics.py` (heritability, ANOVA), `pca.py`,
>   `umap.py`, `outlier_detection.py`, `clustering.py`, `data_cleanup.py`, `data_utils.py`.
> - Configuration is **omegaconf** YAML validated by `validate_qc_config()` /
>   `validate_viz_config()`; analyses are scaffolded from golden templates in `configs/`.
> - There is NO SLEAP `.slp`/`.h5` loading here — input is trait **CSV** data.
>
> **Check:**
>
> 1. Style: PEP 8 enforced by Black (88 cols), Google-style docstrings (pydocstyle/ruff D rules),
>    and `from __future__ import annotations` where the module's existing convention uses it (it
>    is not universal — e.g. `__init__.py` omits it — so only flag a *removed* or inconsistent
>    future import, not its mere absence) — any violations?
> 2. Type hints: are function signatures fully annotated? Any missing return types?
> 3. Pipeline DAG: are new step classes wired into the DAG with correct inputs/outputs and
>    dependency edges? Any cycles or orphaned nodes?
> 4. Magic numbers/strings: are statistical constants/thresholds named and co-located, or
>    are they hardcoded inline?
> 5. Pandas/numpy idioms: are operations vectorized? Any row-wise `apply`/Python loops that
>    should be vectorized? Any chained-assignment / `SettingWithCopyWarning` risks?
> 6. Suppression justification: any `# type: ignore`, `# noqa`, `np.errstate`,
>    `pd.option_context`, or `warnings.filterwarnings` added? Each must have a comment explaining why.
> 7. Error handling: are errors surfaced with meaningful messages or silently swallowed?
> 8. Ripple effects: are there impacts in files NOT changed by the change? (read them)
> 9. Dead code: does the change introduce unreachable branches, unused imports, or stale comments?
> 10. Config schema: if new config fields are added, are they validated and documented in the
>     golden templates?
>
> **Diff:**
> {PR_DIFF}
>
> **Description / intent:**
> {PR_BODY}
>
> Read any source files you need using the Read/Grep tools. Return:
>
> - BLOCKING issues (wrong types, broken DAG wiring, swallowed errors, unvalidated config)
> - IMPORTANT issues (code smell, missing constants, unclear logic, unjustified suppressions)
> - SUGGESTIONS (style, readability, pandas/numpy idiom improvements)
> - Overall code quality score 1-10 with justification

---

### Subagent 2: Testing Strategy & TDD Discipline

```
subagent_type: "general-purpose"
description: "Review testing strategy and TDD discipline"
```

**Prompt:**

> You are reviewing a change for sleap-roots-analyze.
> Your role: **Testing Strategy & TDD Discipline Reviewer**.
> Be adversarial. Check every claim. Run mental red-green-refactor on the diff.
>
> **Testing infrastructure:**
>
> - **pytest** (`tests/`): unit tests in `test_*.py`; pipeline integration tests (e.g.
>   `test_grouped_pipeline_*`, `test_dag.py`); fixtures in `tests/fixtures*.py` and
>   `tests/conftest.py`; sample inputs under `tests/data/`.
> - **Coverage**: pytest-cov via `--cov=src/sleap_roots_analyze`.
> - **CI matrix**: tests run on **Ubuntu, Windows, and macOS** with **Python 3.11** — tests must
>   pass on all three. Lint (Black + Ruff) runs on Ubuntu. CI runs `pytest -m "not integration"`
>   (issue #69), so `@pytest.mark.integration` tests are NOT executed/gated in CI.
>
> **Check:**
>
> 1. Were tests written BEFORE implementation (TDD)? Evidence: test files in earlier commits?
> 2. Is the RIGHT test level used? Pure stat/trait logic -> unit test in `test_<module>.py`;
>    full pipeline -> a `test_*_pipeline_*.py` integration test.
> 3. Are tests specific enough? ("returns NaN for an all-NaN trait column" not "works correctly")
> 4. Missing tests — check each of these:
>    - Empty / single-row inputs (zero samples, one genotype, one replicate)
>    - All-NaN and partially-NaN trait columns
>    - Small groups below statistical thresholds (n < 30 for Mahalanobis; < 3 replicates for H²)
>    - Grouped vs ungrouped analysis paths
>    - Metadata preservation through pipeline DAG stages
>    - CSV output schema/column stability
> 5. Will tests pass in CI on all three OSes? (no hardcoded paths, no platform-specific assumptions,
>    deterministic seeds for stochastic steps like UMAP/Isolation Forest)
> 6. Do existing tests break due to the change? (read `tests/` for impacted files)
> 7. Are fixtures realistic? (do they use representative trait CSVs from `tests/data/`?)
> 8. Is there a 1:1 mapping between OpenSpec spec scenarios and tests?
>
> **Diff:**
> {PR_DIFF}
>
> **CI status:**
> {CI_STATUS}
>
> Read existing test files using Glob/Read tools before concluding. Return:
>
> - BLOCKING: missing tests for new code paths, tests that won't run in CI, existing tests broken
> - IMPORTANT: wrong test level, vague test descriptions, missing edge cases, non-deterministic tests
> - SUGGESTIONS: additional coverage, test refactors
> - TDD verdict: was red-green-refactor actually followed?

---

### Subagent 3: Statistical Rigor & Reproducibility

```
subagent_type: "general-purpose"
description: "Review statistical rigor and reproducibility"
```

**Prompt:**

> You are reviewing a change for sleap-roots-analyze, a library plant biologists use to compute
> statistics and quality control on root phenotyping traits.
> Your role: **Statistical Rigor & Reproducibility Reviewer**.
> Be adversarial. Mistakes in statistics, thresholds, or metadata can invalidate research.
>
> **Core scientific values:**
>
> 1. **Statistical correctness** — heritability (H²), ANOVA, PCA, UMAP, and outlier detection
>    (Mahalanobis distance, Isolation Forest, PCA reconstruction) must be computed correctly and
>    use appropriate assumptions. Reference published methods where applicable.
> 2. **Threshold validity** — Mahalanobis chi-squared reliability needs n ≥ 30 per group;
>    broad-sense heritability needs ≥ 3 replicates per genotype. These two are surfaced as
>    **config-authoring-time advisory warnings** (`config_authoring.py`, via `/configure-run-all`),
>    not runtime guards inside the stat functions. Multiple-comparison correction (FDR,
>    Benjamini-Hochberg) is applied in the **cross-experiment correlation** path
>    (`cross_experiment_analysis.py`) — per-trait ANOVA in `statistics.py` intentionally has no
>    FDR, so do not demand it there. Power/CI reporting must be sound.
> 3. **Reproducibility** — analyses are pinned to a git SHA via committed configs; stochastic
>    steps (UMAP, Isolation Forest, train/test splits) must set explicit random seeds.
> 4. **Metadata preservation** — dataset identity, parameters, and config provenance must flow
>    through the pipeline and appear in output so a CSV row traces back to its source + config.
> 5. **Data format stability** — output CSV column names/ordering and metadata JSON fields must
>    not change silently; downstream scripts depend on them.
> 6. **Numerical stability** — NaN propagation must be deliberate; float comparisons must not use
>    `==`; any warning suppression that hides numerical issues must be justified.
>
> **Check:**
>
> 1. Are statistical computations correct? Trace the algorithm (e.g. H² variance components,
>    ANOVA model terms, Mahalanobis covariance inversion, PCA scaling) step by step.
> 2. Are method references / assumptions documented? Is FDR applied where many cross-experiment
>    correlations are tested? (Per-trait ANOVA has no FDR by design — flag only if the change
>    introduces broad multiple testing without correction.)
> 3. Are the small-sample guardrails surfaced at config-authoring time (n ≥ 30 Mahalanobis,
>    ≥ 3 replicates H²)? Compute-time Mahalanobis reliability is instead assessed by the χ²
>    Kolmogorov–Smirnov goodness-of-fit test (`validate_chi_squared_distribution`) — check that,
>    not for a runtime n<30 warning inside the outlier function.
> 4. Are random seeds set for every stochastic step so results are reproducible?
> 5. Could this change alter previously published results (changed thresholds, defaults, or
>    formulas)? If so, is it documented as a breaking change with a migration note?
> 6. Is metadata (dataset, params, config SHA) preserved through pipeline stages into output?
> 7. How is NaN handled — filtered, imputed, or propagated? Is the choice scientifically defensible
>    and documented?
> 8. Does the change modify output CSV columns/order or metadata JSON fields? If so, documented?
>
> **Diff:**
> {PR_DIFF}
>
> **Description / intent:**
> {PR_BODY}
>
> Return:
>
> - BLOCKING: incorrect statistics, violated reliability thresholds without warning, unseeded
>   stochastic steps, silent output-format breakage, missing metadata
> - IMPORTANT: missing references/assumptions, missing FDR, NaN-handling gaps
> - SUGGESTIONS: additional validation, documentation, reference citations

---

### Subagent 4: Performance, Memory & Cross-Platform

```
subagent_type: "general-purpose"
description: "Review performance, memory, and cross-platform safety"
```

**Prompt:**

> You are reviewing a change for sleap-roots-analyze.
> Your role: **Performance, Memory & Cross-Platform Reviewer**.
> Be adversarial. Check every loop, every allocation, every path operation.
>
> The library processes trait tables that can reach thousands of samples × dozens-to-hundreds of
> trait columns, plus PCA/UMAP/clustering over those matrices. Memory and performance matter, and
> CI runs on Ubuntu, Windows, and macOS.
>
> **Check:**
>
> Performance:
>
> 1. Are pandas/numpy operations vectorized? Any `DataFrame.apply(axis=1)` or Python loops over
>    rows/columns that should be vectorized?
> 2. Redundant recomputation: is the same expensive result (e.g. a covariance matrix, a PCA fit)
>    computed multiple times when it could be reused via the pipeline DAG?
> 3. Are sklearn estimators fit once and reused rather than re-fit per call?
>
> Memory:
>
> 4. Does the code copy large DataFrames unnecessarily (`.copy()` in hot paths) where a view or
>    in-place op would do?
> 5. Are intermediate matrices (distance matrices, reconstructions) materialized at full size when
>    they could be batched?
>
> Cross-Platform:
>
> 6. Are file paths built with `pathlib.Path` — never string concatenation or hardcoded `/`/`\`?
> 7. Any platform-specific behavior (line endings, file encodings, temp dirs, case sensitivity,
>    matplotlib backends) that would fail on Windows or macOS CI?
> 8. Check CI status for platform-specific failures.
>
> Determinism:
>
> 9. `warnings.filterwarnings` and matplotlib global state are process-global. If the change adds
>    or modifies them, could this leak into other code or tests?
>
> **Diff:**
> {PR_DIFF}
>
> **CI status:**
> {CI_STATUS}
>
> Return:
>
> - BLOCKING: OOM risks on large trait tables, row-wise loops where vectorization is required,
>   path handling that breaks on Windows/macOS
> - IMPORTANT: unnecessary copies, redundant fits, platform-specific assumptions, global-state leaks
> - SUGGESTIONS: vectorization opportunities, memory optimizations, caching improvements

---

### Subagent 5: Behavioural Correctness & Edge Cases

```
subagent_type: "general-purpose"
description: "Review behavioural correctness and edge cases"
```

**Prompt:**

> You are reviewing a change for sleap-roots-analyze.
> Your role: **Behavioural Correctness & Edge Case Reviewer**.
> Be adversarial. Play adversarial user. Try to break the feature with pathological inputs.
>
> Focus on: does the implementation actually do what the spec/PR description claims?
> The library must be robust to the messy reality of phenotyping data — missing trait values
> (NaN), tiny genotype groups, single-replicate genotypes, all-identical columns, and configs
> with unusual but valid field combinations.
>
> **Check:**
>
> 1. Read the stated behaviour (PR description / OpenSpec scenarios). Now read the diff. Does the
>    code actually implement what it claims?
> 2. Trace the full call chain for each new feature through the pipeline DAG (load CSV -> clean ->
>    stats/PCA/UMAP/outliers -> output).
> 3. What happens with pathological inputs?
>    - Empty table (zero samples) or single sample?
>    - All-NaN or constant (zero-variance) trait columns?
>    - Groups below thresholds (n < 30 for Mahalanobis; < 3 replicates for H²)?
>    - A genotype with one replicate; a group with one genotype?
>    - Configs missing optional fields, or with `group_by` set to a non-existent column?
> 4. Does the code return scientifically defensible results under partial failure — NaN/empty out,
>    not zeros, silent drops, or crashes? Are guardrail WARNINGs emitted rather than swallowed?
> 5. Pipeline error propagation: if one step fails or yields NaN, do downstream DAG steps handle
>    it gracefully?
> 6. Config validation: are invalid configs rejected by `validate_qc_config()` /
>    `validate_viz_config()` with actionable messages before any computation runs?
> 7. Idempotency/statelessness: are pipeline steps pure (same input + config -> same output)? Any
>    hidden mutable/global state or in-place mutation of shared inputs?
> 8. Does any existing Copilot comment raise an issue not yet addressed?
>
> **Diff:**
> {PR_DIFF}
>
> **Description / intent:**
> {PR_BODY}
>
> **Existing Copilot review comments:**
> {COPILOT_COMMENTS}
>
> Read source files as needed using Read/Grep tools. Return:
>
> - BLOCKING: spec-implementation mismatches, crashes on empty/NaN input, silent data drops,
>   unreliable stats emitted without warning
> - IMPORTANT: edge cases not handled, NaN-propagation gaps, statelessness violations
> - SUGGESTIONS: defensive guards, additional input validation, robustness improvements

---

## Step 3: Synthesize and Post Review

After ALL subagents return:

1. **Deduplicate** overlapping findings.
2. **Prioritize**:
   - **BLOCKING** — must fix before merge (incorrect statistics, broken tests, spec mismatch, data loss)
   - **IMPORTANT** — should fix before merge (missing edge cases, NaN gaps, platform/reproducibility risks)
   - **SUGGESTION** — optional improvements
3. **Determine verdict**:
   - `APPROVE` — no blocking issues, all important issues are minor
   - `COMMENT` — no blocking issues but important items worth noting
   - `REQUEST_CHANGES` — any blocking issues present

### Pre-PR mode

If there is no PR yet (invoked from `/pre-merge-check` Phase 3.5), **do not post**. Print the
synthesized, severity-ranked review to the user so BLOCKING / IMPORTANT items can be fixed before
the PR is opened.

### PR mode — post the review to GitHub

> **Note:** GitHub does not allow requesting changes or approving your own PRs. Detect own-PR
> upfront and, if it's yours, skip `--approve`/`--request-changes` and post a `--comment` with a
> verdict banner. This avoids `GraphQL: Review Can not approve your own pull request` errors.

**Step 1: Detect own-PR** (run once before posting):

```bash
PR_AUTHOR=$(gh pr view "$PR_NUMBER" --json author --jq '.author.login')
GH_USER=$(gh api user --jq '.login')
IS_OWN_PR=false
[ "$PR_AUTHOR" = "$GH_USER" ] && IS_OWN_PR=true
```

**Step 2: Post** using the appropriate method based on `$IS_OWN_PR`.

For REQUEST_CHANGES:

```bash
BODY="$(cat <<'EOF'
## Review Summary

[2-3 sentence overall assessment]

## Blocking Issues

[Must fix before merge]

## Important Issues

[Should fix before merge]

## Suggestions

[Optional improvements]

---
*Review by Claude Code subagent team (Code Quality · Testing · Statistical Rigor · Performance/Memory · Behavioural Correctness)*
EOF
)"

if [ "$IS_OWN_PR" = "true" ]; then
  gh pr review "$PR_NUMBER" --comment -b "$(printf '> **Verdict: REQUEST_CHANGES** (posted as comment — cannot request changes on your own PR)\n\n%s' "$BODY")"
else
  gh pr review "$PR_NUMBER" --request-changes -b "$BODY"
fi
```

For APPROVE:

```bash
BODY="$(cat <<'EOF'
## Review Summary

[2-3 sentence assessment]

## Notes

[Any suggestions or minor observations]

---
*Review by Claude Code subagent team (Code Quality · Testing · Statistical Rigor · Performance/Memory · Behavioural Correctness)*
EOF
)"

if [ "$IS_OWN_PR" = "true" ]; then
  gh pr review "$PR_NUMBER" --comment -b "$(printf '> **Verdict: APPROVE** (posted as comment — cannot approve your own PR)\n\n%s' "$BODY")"
else
  gh pr review "$PR_NUMBER" --approve -b "$BODY"
fi
```

For COMMENT (no own-PR detection needed — `--comment` is always allowed):

```bash
# BODY = the same synthesized review markdown built in the APPROVE/REQUEST_CHANGES heredocs above
gh pr review "$PR_NUMBER" --comment -b "$(printf '> **Verdict: COMMENT**\n\n%s' "$BODY")"
```

After posting, show the user the full synthesized review and the GitHub link.

---

## Domain-Specific Review Patterns

### Pattern 1: Statistical / Trait Computation Changes

1. **Check validation** — are calculations validated against known data or published values?
2. **Verify assumptions** — sample-size thresholds (n ≥ 30 Mahalanobis, ≥ 3 replicates H²), FDR
   correction, random seeds for stochastic steps.
3. **Review edge cases** — all-NaN columns, zero-variance traits, single-replicate genotypes.
4. **Check reproducibility** — will this change previously published results? Is it documented?

### Pattern 2: New Pipeline Steps

1. **Check DAG wiring** — inputs/outputs and dependency edges correct; no cycles.
2. **Verify metadata flow** — does the step preserve dataset/config provenance into output?
3. **Review output format** — CSV columns consistent with sibling pipelines and stable.
4. **Check documentation** — golden templates / README updated with the new field or step.

### Pattern 3: Config Schema Changes

1. **Validation** — new fields rejected/accepted correctly by `validate_qc_config()` /
   `validate_viz_config()`.
2. **Templates** — golden templates updated and documented with rationale.
3. **Back-compat** — do existing committed configs still validate?

### Pattern 4: Bug Fixes

1. **Regression test** — does a test reproduce the original bug (and fail without the fix)?
2. **Side effects** — could the fix change other traits/stats?
3. **Coverage** — does the fix increase coverage on the affected path?
4. **Scope** — is the fix minimal and focused?

## Tips for Effective Reviews

1. **Be specific** — reference `file.py:line` and suggest concrete alternatives.
2. **Be kind** — assume positive intent, use constructive language.
3. **Focus on substance** — don't nitpick style (Black/Ruff handle that).
4. **Explain why** — help the author learn, don't just point out issues.
5. **Approve quickly** — if it's good, say so.
6. Evaluate each finding on its merits; note why if you decline a suggestion
   (see superpowers:receiving-code-review).
