# Tasks — Add PC-Correlation and Trait-Enrichment Workflows

## 1. Subpackage scaffolding
- [ ] 1.1 Create `src/sleap_roots_analyze/pc_correlations/__init__.py` with
      `from __future__ import annotations` and the subpackage docstring.

## 2. Aggregate DAG nodes (workflow 1 inputs)
- [ ] 2.1 Write failing tests for `aggregate_pc_scores_by_genotype` (mean of
      sample PC scores per genotype) and `align_genotypes_across_platforms`
      (common-genotype intersection) using small in-memory DataFrames.
- [ ] 2.2 Port `aggregate.py` (`load_pc_scores`, `load_genotype_mapping`,
      `aggregate_pc_scores_by_genotype`, `load_platform_data`,
      `align_genotypes_across_platforms`); keep `WHEAT_EDPIE_PLATFORMS` as an
      example default only. Green the tests.

## 3. Correlate DAG nodes (multi-scope FDR)
- [ ] 3.1 Write failing tests: 47-test count for a 3/4/5 PC config; `combined`
      vs `per_pair` produce distinct, correctly-pooled FDR columns; CI/power
      columns present.
- [ ] 3.2 Port `correlate.py` (`calculate_correlation`,
      `calculate_cross_platform_pc_correlations`,
      `calculate_all_platform_correlations`, `summarize_correlations`).
      **Import** `calculate_correlation_ci`, `achieved_power`,
      `minimum_detectable_correlation` from `cross_experiment_analysis` instead
      of duplicating. Green the tests.

## 4. Enrichment DAG nodes (workflow 2)
- [ ] 4.1 Write failing tests for `calculate_enrichment_test` (fold, p-values,
      interpretation, input validation) and `calculate_all_enrichment_tests`
      (per-pair + combined) using synthetic correlation DataFrames/CSVs.
- [ ] 4.2 Port `enrichment.py` (`EnrichmentResult`, `calculate_enrichment_test`,
      `load_cross_platform_correlations`, `count_significant`,
      `calculate_all_enrichment_tests`, `results_to_dataframe`). Green the tests.

## 5. Visualization nodes
- [ ] 5.1 Port `visualize.py` (`create_correlation_heatmap`,
      `create_correlation_summary_figure`, `create_scatter_plot`,
      `create_sensitivity_analysis_figure`, `save_figure`) and
      `enrichment.create_enrichment_figure`; force the `Agg` backend. Smoke-test
      that each returns a figure / writes files without error.

## 6. Workflow 1 orchestrator
- [ ] 6.1 Write failing test using a **synthetic pipeline-run directory**
      (per-platform `pc_scores.csv` + `final_data`): asserts artifacts
      (`correlations.csv`, `metadata.json`) exist and the returned `dict`
      correlation count matches the file.
- [ ] 6.2 Implement `workflow.cross_platform_pc_correlations(pipeline_run,
      platform_config, output_dir, ...)` composing the nodes; write CSVs,
      figures (gated by `make_figures`), and `metadata.json`; return the dict.
      Green the test.

## 7. Workflow 2 orchestrator
- [ ] 7.1 Write failing test using **synthetic `cross_platform_correlations.csv`
      fixtures only** (no PC-workflow outputs): asserts independence, artifacts,
      and returned results.
- [ ] 7.2 Implement `workflow.trait_correlation_enrichment(correlation_files,
      output_dir, ...)`; write `enrichment_results.csv`, figure (gated by
      `make_figure`), and `metadata.json`; return the dict. Green the test.

## 8. Public API exports
- [ ] 8.1 Write failing test asserting `cross_platform_pc_correlations`,
      `trait_correlation_enrichment`, and `EnrichmentResult` import from the
      package root and appear in `__all__`.
- [ ] 8.2 Export them via `pc_correlations/__init__.py` and the top-level
      `__init__.py`. Green the test.

## 9. CLI / reproduction scripts
- [ ] 9.1 Add `scripts/` reproduction wrappers that call the public functions
      (no duplicated logic, no hard-coded Windows paths; paths via args/config).

## 10. Golden regression (issue #120 fixtures)
- [ ] 10.1 Add a skip-guarded regression that reproduces 47 tests / 19 genotypes
      / 0 combined-`fdr_by`-significant from the post-QC fixture; skips when the
      fixture is absent.

## 11. Docs + validation
- [ ] 11.1 Document the two workflows (extend `docs/CROSS_PLATFORM_ANALYSIS.md`
      or add a focused doc) and update `docs/API.md` if it tracks public API.
- [ ] 11.2 Run `/pre-merge-check`: black + ruff (`src`) clean, full pytest +
      coverage green, `openspec validate add-pc-correlation-workflows --strict`.
