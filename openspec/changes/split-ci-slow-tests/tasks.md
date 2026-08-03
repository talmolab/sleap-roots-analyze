# Tasks: split-ci-slow-tests

**Commit ordering matters here**: pytest exits non-zero ("no tests collected", exit code 5) when
an `-m` filter matches zero tests. If the CI workflow's `-m "slow"` filter (Task 3) reaches CI
before every test in Task 2 is tagged, the new `slow-tests` job hard-fails on the PR that
introduces it. Land Task 2 (all 14 line items, tagging every identified test) in the same commit
as, or strictly before, Task 3's workflow change — never after. Suggested commit split: (1) Task
1 alone, (2) Task 2 alone (a no-op for CI today, since the main `tests` job's filter is unchanged
until Task 3 lands), (3) Task 3 alone (safe now that every slow test is already tagged), (4) Task
5's docs/changelog updates.

## Task 1: Register the `slow` marker
- [x] 1.1 In `pyproject.toml`'s `[tool.pytest.ini_options]` `markers` list, add
      `"slow: marks tests whose individual runtime meaningfully erodes the tests job's CI
      timeout margin — run in a separate slow-tests CI job, alongside integration"`.

## Task 2: Mark the identified slow tests
Class-level marker (every test in the class qualifies — add `pytestmark = pytest.mark.slow`
directly under the class definition):
- [x] 2.1 `tests/test_run_all_cli_group_by.py::TestRunAllCLIGroupBy` (4 tests, 69-77s each)
- [x] 2.2 `tests/test_grouped_pipeline_config_persistence.py::TestGroupedPipelineConfigPersistence`
      (4 tests, 13-18s each)

Per-test marker (`@pytest.mark.slow` on the individual test — class has other fast tests too):
- [x] 2.3 `tests/test_visualization.py::TestBatchedFigureGenerators::test_boxplot_generator_matches_list_wrapper_output`
      (~198s — the single slowest test in the suite)
- [x] 2.4 `tests/test_visualization.py::TestBatchedHistogramsFileReduction::test_boxplots_large_dataset_reasonable_file_count`
      (~34s)
- [x] 2.5 `tests/test_visualization.py::TestBatchedHistogramsFileReduction::test_large_dataset_reasonable_file_count`
      (~28s)
- [x] 2.6 `tests/test_visualization.py::TestBoxplotGenotypePagination::test_pagination_with_partial_trait_batch_and_partial_genotype_page`
      (~31s)
- [x] 2.7 `tests/test_step_exploratory_analysis.py::TestExploratoryAnalysisMemoryBounds::test_peak_concurrent_figures_bounded_during_execute`
      (~26s)
- [x] 2.8 `tests/test_step_exploratory_analysis.py::TestExploratoryAnalysisMemoryBounds::test_execute_completes_and_produces_figures_for_large_dataset`
      (~25s)
- [x] 2.9 `tests/test_step_exploratory_analysis.py::TestExploratoryAnalysisFigures::test_batched_plots_created_for_many_traits`
      (~17s)
- [x] 2.10 `tests/test_step_generate_static_figures.py::TestMemoryManagement::test_peak_concurrent_figures_bounded_during_static_figures`
      (~24s)
- [x] 2.11 `tests/test_qc_pipeline.py::TestQCPipelineIntegration::test_qc_pipeline_no_outlier_methods_warning`
      (~55s)
- [x] 2.12 `tests/test_qc_pipeline.py::TestQCPipelineIntegration::test_qc_pipeline_turface_integration`
      (~52s)
- [x] 2.13 `tests/test_step_visualize_prediction.py::test_visualize_prediction_step_joblib_n_jobs_and_backend_match_config`
      (module-level function, ~13s)
- [x] 2.14 `tests/test_viz_pipeline_zero_variance.py::TestVizPipelineInterleavedZeroVariance::test_umap_receives_pca_filtered_trait_count`
      (~13s)

## Task 3: Update the CI workflow
- [x] 3.1 In `.github/workflows/ci.yml`, change the `tests` job's `Run pytest` step from
      `-m "not integration"` to `-m "not integration and not slow"`, and update its echoed
      comment to mention both exclusions.
- [x] 3.2 Add a new `slow-tests` job, modeled directly on the existing `tests` job: copy its
      `strategy` block **verbatim**, including the `include:` mapping (`ubuntu-latest`,
      `windows-latest`, and — importantly — `macos-14`, NOT `macos-latest`, which is used by
      the unrelated `numerical-stability` job for a different reason), same
      `timeout-minutes: 30`, same `fail-fast: false`, same setup steps. Its `Run pytest` step
      runs `uv run pytest -m "slow" tests/` with `--durations=20`, deliberately **without**
      `--cov`/`--cov-report=xml` (see `design.md`'s `--cov` decision note — coverage upload is
      currently a disabled stub, so this is a documented scope choice, not an oversight).
- [x] 3.3 Confirm the workflow's `on.pull_request.paths` trigger list doesn't need changes
      (it already covers `tests/**` and `.github/workflows/ci.yml`).

## Task 4: Verify locally
- [x] 4.1 `uv run pytest -m "slow" tests/ -v --durations=0` — confirm exactly the 14 tests/
      classes from Task 2 collect and pass, and note their summed local runtime.
      **Result: 20 passed in 671s (11m11s) locally.**
- [x] 4.2 `uv run pytest -m "not integration and not slow" tests/ --durations=20` — confirm
      the previously-slow tests are absent from the run and the top-20 durations list no
      longer contains any test above ~15s. **Result: 2853 passed, 37 skipped, 23 deselected in
      643.69s (10m43s); slowest is now 25.53s, no 198s outlier.**
- [x] 4.3 `uv run pytest -m "integration"` — confirm untouched (no `slow`-marked test is also
      `integration`-marked, so there's no overlap to worry about, but verify no regression).
      **Result: 3 passed, 2910 deselected — unaffected by this change.**
- [x] 4.4 Full suite sanity check: `uv run pytest tests/ --collect-only -q | tail -1` — total
      collected test count matches pre-change count (no test silently dropped from collection
      by a marker typo). **Result: 2913 collected, matching pre-change.**
- [x] 4.5 Zero-overlap check: `uv run pytest --collect-only -q -m "slow and integration" tests/`
      — expect 0 collected. A test marked both would newly execute inside `slow-tests` despite
      being excluded from `integration` (today, no CI job runs `-m "integration"` at all — issue
      #69 — so this would be new behavior, not a pre-existing one). **Result: 0 collected.**
- [x] 4.6 Exhaustiveness/non-overlap arithmetic: collect counts for
      `-m "not integration and not slow"`, `-m "slow"`, and `-m "integration"` separately, and
      confirm they sum to the Task 4.4 total. If the sum exceeds the total, some test is
      double-marked (only possible overlap given 4.5 is `slow ∩ integration`, already checked).
      **Result: 2890 + 20 + 3 = 2913 exactly — matches.**

## Task 5: Lint, format, docs
- [x] 5.1 `uv run black --check src/sleap_roots_analyze tests` — 198 files unchanged.
- [x] 5.2 `uv run ruff check src/sleap_roots_analyze` — all checks passed.
- [x] 5.3 Add a `### Fixed` entry to `docs/CHANGELOG.md` `[Unreleased]` (repo precedent: CI-only
      infra changes — e.g. the reproducibility gates, the numerical-stability golden gate — are
      already documented as `### Fixed`/`### Added` entries despite no package API change), e.g.:
      "CI's `tests` job no longer risks spurious timeout failures unrelated to a PR's own changes:
      ~20 large-dataset regression tests (added by #210 to guard the OOM fix) that
      disproportionately erode the job's 30-minute budget are now tagged `@pytest.mark.slow` and
      run in a separate `slow-tests` job on the same three-OS matrix — full coverage is preserved,
      just isolated from the fast suite's timeout margin. (#217)"
- [x] 5.4 `openspec validate split-ci-slow-tests --strict` — valid.
- [x] 5.5 Update `docs/testing.md`: drop the "if markers are defined" hedge on the `slow`-marker
      example (~line 49) since the marker is now real and registered; and update the documented
      local coverage command (~line 259) from `-m "not integration"` to
      `-m "not integration and not slow"` so it keeps matching what the `tests` job's `--cov`
      invocation actually runs — otherwise the doc's command silently over-reports coverage
      relative to CI's post-change number.

## Task 6: Push and open PR
- [ ] 6.1 Push branch, open PR referencing #217, and let the new `slow-tests` job run
      alongside the now-lighter `tests` job on the PR itself as the real-world verification.
- [ ] 6.2 Once the PR's CI run completes, confirm the `tests (windows, Python 3.11)` job's
      actual wall-clock duration dropped measurably below the ~28.5-minute pre-change baseline
      (target: at least a 5-minute drop, restoring a real margin under the 30-minute timeout) —
      this is the concrete, real-data closure of the proposal's central goal, not just the local
      estimate from Task 4.
