# Tasks — Add PC-Correlation and Trait-Enrichment Workflows

## 1. Subpackage scaffolding
- [x] 1.1 Create `src/sleap_roots_analyze/pc_correlations/__init__.py` with
      `from __future__ import annotations` and the subpackage docstring.

## 2. Aggregate DAG nodes (workflow 1 inputs)
- [x] 2.1 Tests for `aggregate_pc_scores_by_genotype` (mean of sample PC scores
      per genotype) and `align_genotypes_across_platforms` (common-genotype
      intersection).
- [x] 2.2 Port `aggregate.py`; keep `WHEAT_EDPIE_PLATFORMS` as an example
      default only.

## 3. Correlate DAG nodes (multi-scope FDR)
- [x] 3.1 Tests: 47-test count for a 3/4/5 PC config; `combined` vs `per_pair`
      distinct, correctly-pooled FDR columns; CI/power columns present.
- [x] 3.2 Port `correlate.py`; **import** `calculate_correlation_ci`,
      `achieved_power`, `minimum_detectable_correlation` from
      `cross_experiment_analysis` instead of duplicating.

## 4. Enrichment DAG nodes (workflow 2)
- [x] 4.1 Tests for `calculate_enrichment_test` (fold, depletion, validation)
      and `calculate_all_enrichment_tests` (per-pair + combined).
- [x] 4.2 Port `enrichment.py` (`EnrichmentResult`, binomial tests, loaders).

## 5. Visualization nodes
- [x] 5.1 Port `visualize.py` figures + `enrichment.create_enrichment_figure`;
      smoke-covered via the figure-writing workflow test (Agg backend).

## 6. Workflow 1 orchestrator
- [x] 6.1 Test using a synthetic pipeline-run directory: artifacts exist and the
      returned `dict` correlation count matches the file.
- [x] 6.2 Implement `workflow.cross_platform_pc_correlations`.

## 7. Workflow 2 orchestrator
- [x] 7.1 Test using synthetic `cross_platform_correlations.csv` fixtures only,
      asserting independence + artifacts + returned results.
- [x] 7.2 Implement `workflow.trait_correlation_enrichment`.

## 8. Public API exports
- [x] 8.1 Test asserting the two functions + `EnrichmentResult` import from the
      package root and appear in `__all__`.
- [x] 8.2 Export via `pc_correlations/__init__.py` and the top-level
      `__init__.py` (public-API introspection guard passes: 115/115).

## 9. CLI / reproduction scripts
- [x] 9.1 `scripts/run_pc_correlations.py` and `scripts/run_trait_enrichment.py`
      call the public functions (argparse; no hard-coded paths).

## 10. Golden regression (issue #120 fixtures)
- [x] 10.1 Skip-guarded regression placeholder for 47 / 19 / 0; skips honestly
      until #120 lands all three platforms' sample-level PCA outputs (only
      turface_19 ships today). Wired to real assertions in a follow-up.

## 11. Docs + validation
- [x] 11.1 Document both workflows in `docs/CROSS_PLATFORM_ANALYSIS.md`.
- [x] 11.2 black + ruff (`src`) clean; new tests green (13 passed, 1 skipped);
      `openspec validate add-pc-correlation-workflows --strict` passes. Full
      `/pre-merge-check` to run before marking the PR ready.
