## Context

Issue #119 asks for a single public function that reproduces the wheat EDPIE cross-platform
PC-correlation analysis, today only available as paper-specific scripts. All the building blocks
already exist in the repo (`calculate_genotype_means`, `perform_pca_analysis`,
`calculate_correlation_ci`, `achieved_power`, `minimum_detectable_correlation`, and
`multipletests`); this change composes them behind one introspection-ready entry point.

## Goals / Non-Goals

- Goals: one public `cross_platform_pc_correlations` + typed `CrossPlatformPCResult`; exact paper
  reproduction (47 tests / 19 genotypes / 0 FDR-significant); synthetic unit test; full type hints +
  docstrings.
- Non-Goals: a new pipeline/config or CLI step (the config-driven `CrossPlatformPipeline` stays as
  is); visualization; vendoring the wheat EDPIE input data (tracked by #120).

## Decisions

- **Ordering: sample-level PCA → genotype-mean PC scores → correlate.** The issue *body* lists
  "genotype means → PCA", but the maintainer's reference note overrides it: *"Preserve the order:
  sample-level PCA → genotype-mean PC scores → correlate (not average-then-PCA)."* PCA is fit on the
  sample-level trait matrix per platform; the sample PC scores are then averaged per genotype; those
  genotype-mean PC scores are correlated across platforms. This is the single most important
  correctness property and is pinned by a dedicated test.
  *Alternative considered:* average-then-PCA (the issue-body reading) — rejected; it changes the PCA
  basis and would not reproduce the golden numbers.

- **Pooled FDR across the whole test family.** All cross-platform PC tests from all unordered
  platform pairs form one multiple-testing family; `multipletests` is applied once across the pool,
  not per pair. The golden "47 tests, 0 FDR-significant" is a single-family result, so per-pair
  correction would give the wrong survivor count.

- **`CrossPlatformPCResult` as a frozen `@dataclass`.** Matches the repo convention (configs and
  `StepResult` are dataclasses; no pydantic at the analysis layer). Fields: `pca` (name → PCA result
  dict from `perform_pca_analysis`), `pc_scores` (name → genotype-indexed PC-score DataFrame),
  `correlations` (pair-key → DataFrame of PC×PC tests with r, p, p_fdr, ci_low, ci_high, power,
  n_genotypes), `significant` (DataFrame of FDR survivors), and `summary` (n_tests, n_genotypes,
  n_fdr_significant). A flat combined table is also exposed for easy golden comparison.

- **New module `cross_platform_pc.py`.** Keeps a clearly named public workflow separate from the
  large `cross_experiment_analysis.py`, while importing its helpers. Exported from `__init__.py`.

- **Genotype alignment per pair.** Each pair correlates over the intersection of the two platforms'
  genotype indexes; the common-genotype count is recorded per pair and drives CI/power. Pairs with
  too few common genotypes are reported (with their count), not raised on.

## Risks / Trade-offs

- Exact reproduction depends on PCA sign/precision and the standardization used in the reference
  script. Mitigation: correlations and |r|-based significance are sign-invariant for the survivor
  count; the regression asserts counts (47 tests, 19 genotypes, 0 FDR) rather than raw signed r, and
  uses the same `random_state` and standardization defaults as `perform_pca_analysis`.
- The golden regression can't run in CI here until the post-QC fixture lands (#120). Mitigation:
  `pytest.mark.skipif` on fixture absence; the synthetic test guards the math (47-count, pooling,
  CI/power, alignment) in the meantime.

## Open Questions

- Should the post-QC wheat EDPIE fixture be vendored into this repo (small CSVs) so the regression
  runs in CI, or stay external until #120? (Lean: skip-guard now; revisit when #120 ships.)
- `correction_method` default: the public signature in #119 uses `"fdr_bh"`, while the existing
  `CrossPlatformConfig` defaults to `"fdr_by"`. We follow the issue's `"fdr_bh"` default for the new
  function and document the difference. — confirm at review.
