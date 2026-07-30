# Tasks: fix-110-oom-exploratory-analysis

Commit boundaries below fold each red+green pair into a single commit (this repo has no
precedent for standalone failing-test commits and `main` has no per-commit CI gate — see
`git log` history) — treat the red/green split as a local TDD iteration loop, not a commit
boundary.

## Task 0: Dispose of the prior non-TDD draft
- [x] 0.1 Confirm the prior uncommitted draft is stashed (not applied) on this branch and is not
      silently merged into any implementation below (its cap value, fix location, and lack of any
      pagination were all superseded by `design.md` Decisions 2 and 3)
- [x] 0.2 Drop the stash (`git stash drop`) once Task 7 confirms all tests pass and the draft has
      been fully superseded

## Task 1: Height cap -- write failing tests, then implement (TDD Red -> Green)
*Commit 1: `fix(#110): cap horizontal boxplot subplot height`*
- [x] 1.1 In `tests/test_visualization.py`, add `TestBoxplotHorizontalHeightCap` with a synthetic
      DataFrame fixture pinned at 480 genotypes (horizontal orientation triggered) and a small
      trait count
- [x] 1.2 `test_direct_call_horizontal_height_capped` -- call `create_trait_boxplots_by_genotype()`
      directly (not via the batched wrapper) with 480 genotypes, horizontal orientation; assert
      `fig.get_size_inches()` height stays at or under the cap. This specifically catches the
      "cap applied only to the batched wrapper's local variable, silently discarded by the inner
      function" bug identified in `design.md` Decision 2.
- [x] 1.3 `test_batched_call_horizontal_height_capped` -- same assertion via
      `create_trait_boxplots_by_genotype_batched()`
- [x] 1.4 `test_horizontal_height_cap_boundary` -- `n_genotypes` chosen so
      `n_genotypes * 0.3 == max_subplot_height` exactly; assert no off-by-one/float issue in the
      `min()` cap
- [x] 1.5 `test_horizontal_height_unchanged_below_cap` -- genotype count low enough that
      `n_genotypes * 0.3` is below the cap; assert height matches the existing uncapped formula
      exactly (`max(4.0, n_genotypes * 0.3)`) -- no regression for the common case
- [x] 1.6 `test_custom_max_subplot_height_respected` -- pass a non-default `max_subplot_height`
      (e.g. 10.0) and confirm it, not the 20.0 default, is what bounds the output
- [x] 1.7 `test_horizontal_boxplots_zero_genotypes_and_zero_traits` -- empty-input edge cases
      (0 genotypes, empty `trait_cols`) do not raise
- [x] 1.8 Run tests, confirm FAIL on current code
- [x] 1.9 Add `max_subplot_height: float = 20.0` parameter to `create_trait_boxplots_by_genotype()`
      (`visualization.py`); apply cap in the horizontal-orientation branch:
      `min_subplot_height = min(max_subplot_height, max(4, n_genotypes * height_per_genotype))`.
      Update its Google-style docstring `Args:` section.
- [x] 1.10 Add the same `max_subplot_height: float = 20.0` parameter to
      `create_trait_boxplots_by_genotype_batched()`; use it in its own `batch_figsize`
      precomputation and pass it through to the inner `create_trait_boxplots_by_genotype()` call.
      Update its docstring `Args:` section.
- [x] 1.11 Run Task 1 tests, confirm PASS (green)
- [x] 1.12 Run the full existing `TestTraitBoxplotsAdaptiveSizing` and
      `TestBoxplotLabelOverlapFixes` classes, confirm no regressions

## Task 2: Genotype pagination for readability (TDD Red -> Green)
*Commit 2: `feat(#110): paginate boxplots by genotype when count exceeds readable page capacity`*
- [x] 2.1 In `tests/test_visualization.py`, add `TestBoxplotGenotypePagination`. Use a small, fixed
      trait count for these tests (e.g. exactly 1 trait, so `n_traits <= batch_size` guarantees a
      single trait batch and every figure the test sees belongs to it unambiguously) -- pagination
      is a property of the genotype axis, not the trait axis, so reuse Task 1's 480-genotype
      fixture but do NOT reuse the full 300-trait fixture here (keeps these tests fast; see
      `design.md` Decision 4 point 6)
- [x] 2.2 `test_pagination_splits_genotypes_at_default_capacity` -- 480 genotypes, default
      `max_genotypes_per_page` (auto-derived, ~66 for horizontal); assert
      `create_trait_boxplots_by_genotype_batched()` returns more figures than trait-batch count
      alone would produce (i.e. `n_trait_batches * n_genotype_pages`, `n_genotype_pages > 1`)
- [x] 2.3 `test_pagination_covers_every_genotype_exactly_once` -- with the single-trait-batch fixture
      from 2.1, every returned figure belongs to that one trait batch; collect the genotype tick
      labels rendered in each figure (via `ax.get_yticklabels()` for horizontal orientation,
      matching the existing pattern at `tests/test_visualization.py:303,4033`) and assert their
      union equals the full 480-genotype set with no duplicates across figures
- [x] 2.4 `test_pagination_noop_at_or_below_capacity` -- genotype count at or below the auto-derived
      page capacity; assert exactly one genotype page per trait batch (no behavior change)
- [x] 2.5 `test_pagination_page_height_uses_readable_spacing_not_cap` -- a paginated figure's height
      reflects `page_genotype_count * 0.3` (the pre-cap readable formula), not the 20" cap, since
      pages are sized to stay under it by construction
- [x] 2.6 `test_pagination_custom_max_genotypes_per_page_respected` -- explicit
      `max_genotypes_per_page` override changes the number of pages produced
- [x] 2.7 `test_pagination_suptitle_includes_genotype_range` -- a multi-page batch's figures each
      have a `suptitle` containing the genotype range and total (e.g. "Genotypes 1-66 of 489")
- [x] 2.8 `test_pagination_last_page_orientation_matches_other_pages` -- use 469 genotypes with the
      default page capacity of 66 (`469 = 66*7 + 7`, leaving a 7-genotype final page, correctly
      below `horizontal_threshold=8` -- verify this arithmetic when implementing, a prior draft of
      this task miscalculated it as 483); assert every page's rendered orientation matches the
      orientation resolved from the *full* dataset, not one independently re-resolved from the
      small page's own count (see `design.md` Decision 3's orientation-consistency note)
- [x] 2.9 `test_pagination_missing_genotype_column_is_noop` -- a DataFrame that does not contain
      `genotype_col` at all; assert `create_trait_boxplots_by_genotype_batched()` does not raise
      and produces one figure per trait batch (matching today's existing "0 genotypes" behavior),
      not a `KeyError`
- [x] 2.10 `test_pagination_with_nan_genotype_values` -- a DataFrame with some rows having a NaN
      genotype value alongside 480+ real genotype values; assert pagination does not raise
      (`sorted()` on a mixed NaN/string array raises `TypeError`) and every non-NaN genotype still
      appears in exactly one page
- [x] 2.11 `test_pagination_with_partial_trait_batch_and_partial_genotype_page` -- e.g. 20 traits
      with `batch_size=16` (batches of 16 + 4) crossed with 100 genotypes at the default page
      capacity of 66 (pages of 66 + 34); assert figure count equals
      `n_trait_batches * n_genotype_pages` and each figure's figsize matches the per-page formula
      for its actual (possibly partial) trait and genotype counts
- [x] 2.12 Run tests, confirm FAIL on current code (no pagination exists yet)
- [x] 2.13 In `create_trait_boxplots_by_genotype_batched()`, add `max_genotypes_per_page:
      Optional[int] = None` parameter; when `None`, derive as `max(1, int(max_subplot_height //
      per_genotype_size))` using 0.3 (horizontal) or 0.5 (vertical) based on `actual_orientation`.
      Guard `genotype_col in df.columns` before accessing it for pagination purposes -- if absent,
      pagination is a no-op (single page), matching today's existing "0 genotypes" behavior rather
      than raising `KeyError`. When present, compute genotype pages from
      `sorted(df[genotype_col].dropna().unique())` (dropping NaN before sorting, matching the
      existing per-trait rendering path's convention of only ever sorting a `.dropna()`'d subset),
      chunked into pages. For each (trait batch, genotype page) combination, filter the DataFrame to
      that page's genotypes and render one figure (reusing the existing per-batch trait/sizing
      logic, now driven by the page's genotype count rather than the full dataset's). Pass the
      already-resolved `actual_orientation` (not the original, possibly-`"auto"`, `orientation`
      argument) into each per-page call so every page of a batch uses a consistent orientation
      regardless of that page's own genotype count. Append the genotype range to the `suptitle`
      when more than one page exists. Update the docstring `Args:` section.
- [x] 2.14 Run Task 2 tests, confirm PASS (green)
- [x] 2.15 Run the full existing `TestTraitBoxplotsAdaptiveSizing`, `TestBoxplotLabelOverlapFixes`,
      and Task 1's height-cap tests, confirm no regressions

## Task 3: Generator refactor + incremental save/close in ExploratoryAnalysisStep (TDD Red -> Green)
*Commit 3: `fix(#110): incrementally save+close figures in ExploratoryAnalysisStep.execute()`*
- [x] 3.1 In `tests/test_visualization.py`, add tests for generator/list parity:
      `test_histogram_generator_matches_list_wrapper_output` and
      `test_boxplot_generator_matches_list_wrapper_output` -- `list(_generate_trait_*_batches(...))`
      produces the same count and sizes (including paginated output) as the existing public
      function's output
- [x] 3.2 Add `test_boxplot_generator_yields_lazily` -- confirm a figure is not created until the
      generator is advanced (e.g. via a counter/spy), demonstrating genuine laziness rather than
      generator syntax wrapping already-eager work
- [x] 3.3 Run tests, confirm FAIL (generators do not exist yet)
- [x] 3.4 In `tests/test_step_exploratory_analysis.py`, add a fixture producing a synthetic
      DataFrame (dpi=100, matching this file's existing low-DPI test convention) -- no proprietary
      data. Originally pinned at 480 genotypes x 300 trait columns to mirror the real #110/
      production failure scale directly; **reduced to 100 genotypes x 40 traits after an actual CI
      run on the opened PR showed the larger size pushed CI's 30-minute `tests` job timeout on
      Ubuntu/Windows** (both failed at the identical 30m16s mark mid-suite -- a timeout, not a test
      failure) even though it ran fine standalone locally (~170-180s). The smaller fixture still
      triggers genotype pagination (2 pages) and multiple trait batches, runs in ~15s (see
      `design.md` Decision 4 point 7's post-hoc update).
- [x] 3.5 Add `test_peak_concurrent_figures_bounded_during_execute` -- instrument figure lifecycle
      by monkeypatching both figure-creation (e.g. wrap `plt.subplots`/`plt.figure`) and
      `matplotlib.pyplot.close`, recording `len(plt.get_fignums())` at both points (sampling only at
      close time can miss a figure that's created but never explicitly closed -- see review
      finding). Assert the peak recorded value is far below the total number of figures the step
      would generate (e.g. `< 10`); pin the exact constant once Task 3.9 is green.
- [x] 3.6 Decide and record whether this test needs `@pytest.mark.integration` -- CI's `tests` job
      runs `-m "not integration"` on all 3 OSes, so if marked integration this regression proof
      never runs in CI. Given the DataFrame is synthetic and runtime is expected to be seconds, do
      NOT mark it integration; keep it in the default CI-run test set.
- [x] 3.7 Run test, confirm FAIL on current code (all figures accumulate before any close)
- [x] 3.8 In `src/sleap_roots_analyze/visualization.py`, add `_generate_trait_histogram_batches()`
      as a generator version of `create_trait_histograms_batched()`; make the existing function a
      `list(...)` wrapper over it. Add `_generate_trait_boxplot_batches()` as a generator version of
      `create_trait_boxplots_by_genotype_batched()` (including Task 1's height cap and Task 2's
      pagination); make the existing function a `list(...)` wrapper over it.
- [x] 3.9 In `ExploratoryAnalysisStep.execute()`, remove the `all_figures` dict. Save and close each
      summary/EDA figure and the correlation heatmap immediately after creation, and iterate the two
      batch generators directly, saving and closing each batch figure as it's yielded. Update
      `execute()`'s metadata construction (`figures_generated`, `figure_names`) to track figures via
      the save+close calls instead of `all_figures.keys()`.
- [x] 3.10 Run Task 3 tests, confirm PASS (green); pin the exact peak-figure-count constant Task 3.5
      asserts based on the real measured value (measured 4, previously 45; asserted `<= 5`)
- [x] 3.11 Run the full existing `TestExploratoryAnalysisStepBasic`,
      `TestExploratoryAnalysisStatistics`, `TestExploratoryAnalysisFigures`,
      `TestExploratoryAnalysisEdgeCases`, and `TestExploratoryAnalysisMetadataPropagation` classes,
      confirm no regressions

## Task 4: Incremental save/close in GenerateStaticFiguresStep (TDD Red -> Green)
*Commit 4: `fix(#110): incrementally save+close figures in GenerateStaticFiguresStep`*
- [x] 4.1 In `tests/test_step_generate_static_figures.py`, add a `test_peak_concurrent_figures_bounded_during_static_figures`
      test using the same creation+close instrumentation as Task 3.5, and disabled
      PCA/heritability/correlation/genotype-comparison plot types via `static_viz_config_enabled`,
      since this step generates many more figure types than `ExploratoryAnalysisStep` when fully
      enabled -- keeping the test scoped to trait-distribution figures (the code path this task
      actually changes). Originally pinned at 480 genotypes x 30 traits, matching the real-world
      failure scale; **reduced to 100 genotypes x 12 traits after an actual CI run on the opened PR
      showed the larger size contributed to CI's 30-minute `tests` job timeout on Ubuntu/Windows**
      (see Task 3.4's note and `design.md` Decision 4 point 7's post-hoc update) -- still triggers
      genotype pagination and multiple trait batches, runs in ~14s instead of ~180s.
- [x] 4.2 Run test, confirm FAIL on current code (full batch list materializes before any close)
- [x] 4.3 Update `GenerateStaticFiguresStep` to iterate `_generate_trait_histogram_batches()` /
      `_generate_trait_boxplot_batches()` directly instead of calling the list-returning public
      wrappers; keep the existing periodic `gc.collect()` calls (now an extra safety margin, not
      the only defense). Also updated two pre-existing tests
      (`TestAdaptiveBatchSize::test_adaptive_batch_size_increases_for_many_traits` and
      `test_cylinder_scale_generates_reasonable_batch_count`) that patched the old public function
      names directly in this module's namespace -- they now patch the new private generator names.
- [x] 4.4 Run Task 4 test, confirm PASS (green) -- verified with an explicit red/green cycle
      (temporarily reverted the fix via `git stash`, confirmed the test fails at peak=40, restored
      via `git stash pop`, confirmed it passes)
- [x] 4.5 Run the full existing static-figures test suite, confirm no regressions (70 passed)

## Task 5: End-to-end regression test against the real failure shape
- [x] 5.1 Already satisfied by Task 3's `TestExploratoryAnalysisMemoryBounds` (no separate commit
      needed): `test_execute_completes_and_produces_figures_for_large_dataset` runs
      `ExploratoryAnalysisStep.execute()` end-to-end against the synthetic fixture (100 genotypes x
      40 traits, reduced from an original 480x300 per Task 3.4's note) with `enable_batched_plots=True`
      (the `VisualizationConfig` default)
- [x] 5.2 Covered by the same test: asserts the step completes without raising, produces PNG figure
      files on disk, and produces multiple genotype-paginated boxplot batches (`box_batches > 1`)
- [x] 5.3 Covered by the sibling test in the same class,
      `test_peak_concurrent_figures_bounded_during_execute`, using the Task 3.5 instrumentation
      against the identical fixture -- splitting "completes and produces correct output" from "peak
      memory stays bounded" into two focused tests rather than one combined test

## Task 6: Documentation and finalization
*Commit 5: `docs(#110): changelog entry`*
- [x] 6.1 Add a `### Fixed` entry to `docs/CHANGELOG.md` under `[Unreleased]` for #110: incremental
      figure save/close in `ExploratoryAnalysisStep` and `GenerateStaticFiguresStep` (fixes OOM on
      large genotype counts); the new `max_subplot_height` cap (default 20.0") on
      `create_trait_boxplots_by_genotype()` / `create_trait_boxplots_by_genotype_batched()`'s
      horizontal branch; and the new `max_genotypes_per_page` pagination that keeps high-genotype-
      count boxplots readable — noting the visual-output change (more boxplot figures, capped
      height) for previously-uncapped/unpaginated high-genotype-count datasets

## Task 7: Verify no regressions and finalize
- [x] 7.1 Full test suite passes (`uv run pytest`) -- 2863 passed, 37 skipped, 0 failed
      (46m32s)
- [x] 7.2 Linting and formatting pass (`uv run ruff check`, `uv run black --check .`) -- clean on
      all files this change touches (pre-existing lint debt in unrelated files left as-is, out of
      scope)
- [x] 7.3 Type checking passes (`uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter
      --baseline-path .mypy-baseline.txt`, matching CI's `type-check` job) -- new generator
      functions need correct `Iterator[plt.Figure]`-style annotations -- found and fixed one new
      error (`genotype_pages` needed an explicit `List[Optional[List[Any]]]` annotation); confirmed
      `new: 0` / exit 0 after the fix
- [x] 7.4 `openspec validate fix-110-oom-exploratory-analysis --strict` passes
- [x] 7.5 Drop the reference-only stash (Task 0.2) once superseded -- dropped
- [x] 7.6 Note in the PR description that #202, the correlation-heatmap figsize cap, #110's P2
      DPI-reduction suggestion, and `create_exploratory_summary_plots()`'s separate
      adaptive-sizing-bounded boxplot are explicitly out of scope for this change -- included in
      the "Out of scope" section of the PR body. Also ran a full pre-merge-check: fresh full-suite
      coverage run (2866 passed, 37 skipped, 0 failed, 89% coverage) and a 5-subagent pre-PR review
      (no blocking findings; all IMPORTANT findings fixed in a follow-up commit and re-verified)
- [x] 7.7 Visual QA: generate boxplot figures for a genotype count that triggers pagination (e.g.
      120 genotypes, 2 pages) and open the PNGs with the Read tool to visually confirm labels are
      readable within each page and the suptitle correctly identifies the genotype range

## Post-PR: CI-only timeout discovered and fixed
- [x] 8.1 PR #210 opened; CI's `Tests (ubuntu, Python 3.11)` and `Tests (windows, Python 3.11)`
      jobs both failed at the identical 30m16s mark (`##[error]The operation was canceled.`
      mid-suite at 94% completion, no test failures visible) -- a 30-minute job timeout
      (`.github/workflows/ci.yml` `timeout-minutes: 30`), not a real test failure. `macos` passed in
      16m46s. Root cause: Task 3.4/4.1's 480-genotype fixtures ran fine standalone locally
      (~170-180s each, which read as an acceptable budgeted cost per Decision 4 point 7) but pushed
      the shared CI job over budget once combined with the rest of the ~2900-test suite --
      standalone local timing is not sufficient evidence of CI feasibility for a shared-budget job.
- [x] 8.2 Reduced both fixtures (100 genotypes x 40 traits for `ExploratoryAnalysisStep`, 100 x 12
      for `GenerateStaticFiguresStep`) -- still large enough to trigger genotype pagination and
      multiple trait batches, still large enough to clearly distinguish a bounded (~4 peak) from an
      unbounded (near-total-figure-count peak) implementation. Runtime dropped from ~170-180s each
      to ~15s each (~12x). Updated `design.md` Decision 4 point 7 and this file's Task 3.4/4.1/5.1
      entries to document the correction and why.
- [x] 8.3 Re-verified: `tests/test_step_exploratory_analysis.py` +
      `tests/test_step_generate_static_figures.py` full files (86 passed), ruff/black clean,
      `openspec validate --strict` passes.
