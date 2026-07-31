# Tasks: fix-pca-feature-selection-wiring

**Suggested commit grouping**: Task 1 → one `test(#202):` commit (red).
Tasks 2 and 3 → **one combined `fix(#202):` commit** (green) — do not split
across two commits: Task 2 alone (fixing only `create_pca_biplot`) would
leave Task 1.4/1.6's `create_feature_contribution_plot` assertions still
red if committed on its own, and Task 3.1 (removing the parameter) must
land in the exact same commit as Task 3.3 (removing the matching call-site
kwarg in `generate_static_figures.py`) — splitting those two would leave a
`TypeError` at runtime on any pipeline invocation between commits, which is
a broken build, not just a failing test. Task 4 → verification, no code
change. Task 5 → `docs(#202):` + config regeneration, landed last so
code-correctness review and image/data-diff review stay separable. Never
leave a red/broken state as the branch tip while the PR is open for
review. Single PR for both bugs (same issue, same root cause pattern,
shared consistency test spanning both functions).

## Task 1: Write failing tests (TDD Red Phase)

- [x] 1.1 In `tests/test_visualization.py`, add
      `test_create_pca_biplot_top_variance_feature_selection` (near the
      existing `test_create_pca_biplot`, ~line 2243): call `create_pca_biplot`
      with `feature_selection="top_variance"` and `top_n_features=5` (pin
      this value explicitly — matches the existing test's convention) using
      the existing `pca_viz_results`/`pca_viz_dataframe` fixtures and
      `trait_names = [f"trait_{i}" for i in range(10)]`. Separately call
      `select_top_features_from_pca(loadings=pca_viz_results["loadings"],
      eigenvalues=pca_viz_results["eigenvalues"], n_features_total=10,
      n_features_to_select=5, method="top_variance", pc_indices=None)`
      directly, map the returned indices to trait names, and assert that
      set matches the set of `text.get_text()` values collected from
      `fig.axes[0].texts` (the per-feature loading labels `create_pca_biplot`
      draws at `visualization.py:2658-2666`). Note: `adjustText`'s normal
      label-repositioning path only mutates existing `Text` objects and
      does not add to `ax.texts`; if this proves flaky in practice (its
      rare `ax.annotate("", ...)` fallback would add an empty-string entry),
      filter `texts` to non-empty `get_text()` values before comparing.
- [x] 1.2 In the same file, add
      `test_create_pca_biplot_unrecognized_feature_selection_raises`:
      call `create_pca_biplot(..., feature_selection="not_a_real_method")`
      and assert it raises `ValueError`.
- [x] 1.3 Add `test_create_pca_biplot_feature_selection_methods_dispatch_correctly`,
      parametrized over `["vector_length", "extreme", "top_absolute",
      "top_contribution"]` (the four pre-existing, currently-untested
      mapping branches — grepping `test_visualization.py` for
      `feature_selection` today returns no matches, so none of these
      branches has direct coverage; Task 2.4's "no regressions" claim is
      otherwise unverifiable). For each method, assert `create_pca_biplot`'s
      selected/plotted features (via `fig.axes[0].texts`, same technique as
      1.1) match a direct `select_top_features_from_pca(method=...,
      pc_indices=[pc_x_idx, pc_y_idx])` call with the same inputs.
- [x] 1.4 Update `test_create_feature_contribution_plot_consistency`
      (~line 2182) — no functional change needed to the test itself since
      it doesn't pass `feature_selection`, but confirm it still exercises
      both the pre-calculated and on-the-fly branches after Task 3's
      refactor (it's the regression guard that the two branches keep
      agreeing).
- [x] 1.5 Add `test_create_feature_contribution_plot_no_feature_selection_param`:
      assert calling `inspect.signature(create_feature_contribution_plot).parameters`
      no longer contains `"feature_selection"`, and that calling the
      function with a stray `feature_selection=...` kwarg raises
      `TypeError`.
- [x] 1.6 In `tests/test_step_generate_static_figures.py`, find the existing
      test(s) mocking `create_feature_contribution_plot` (currently a plain
      `Mock(return_value=Mock())` around line 1544, which silently accepts
      any stray kwarg and would NOT catch a regression if Task 3.3 were
      skipped or reverted). Change the mock to `Mock(spec=create_feature_contribution_plot)`
      (or assert on `mock_contrib.call_args.kwargs` directly) and assert
      `"feature_selection"` is not among the kwargs passed by
      `GenerateStaticFiguresStep`.
- [x] 1.7 Run `uv run pytest tests/test_visualization.py -k "pca_biplot or feature_contribution"`
      and the `test_step_generate_static_figures.py` test from 1.6, confirm
      1.1, 1.2, and 1.5 FAIL on current code (1.1 fails because
      `top_variance` silently maps to `vector_length`'s different feature
      set; 1.2 fails because the unrecognized string currently falls
      through to `vector_length` with no error; 1.5 fails because the
      parameter still exists), 1.3 PASSES (regression guard — the four
      existing methods already dispatch correctly today), and 1.6 PASSES
      (regression guard — the current call site already doesn't pass
      `feature_selection` incorrectly today, this just tightens the
      assertion so a future regression would be caught).

## Task 2: Fix `create_pca_biplot`

- [x] 2.1 In `visualization.py`, add an explicit `elif feature_selection ==
      "top_variance": method = "top_variance"` branch to the mapping block
      (~L2260-2269), and change the trailing `else` to
      `raise ValueError(f"Unrecognized feature_selection: {feature_selection!r}")`.
- [x] 2.2 Change the `select_top_features_from_pca(...)` call (~L2272-2279)
      to pass `pc_indices=[pc_x_idx, pc_y_idx] if method != "top_variance" else None`
      — see `design.md` for why `top_variance` must not receive the 2-PC
      scope (the method ignores `pc_indices` entirely, so passing the
      2-index list would silently no-op rather than error, misleadingly
      suggesting the biplot's PC scope is respected).
- [x] 2.3 Update the function's docstring (`feature_selection` Args entry,
      ~L2203-2207) to list `top_variance` as a valid value and add a
      `Raises: ValueError` entry for unrecognized values.

## Task 3: Fix `create_feature_contribution_plot` and its call site

- [x] 3.1 In `visualization.py`, remove the `feature_selection` parameter
      from `create_feature_contribution_plot`'s signature and docstring
      (~L1967-1993).
- [x] 3.2 Refactor the on-the-fly (backward-compatibility) branch
      (~L2085-2122, the `else` branch that currently duplicates the
      `top_variance` ranking formula) to call
      `select_top_features_from_pca(loadings=..., eigenvalues=...,
      n_features_total=len(trait_names), n_features_to_select=top_n,
      method="top_variance", pc_indices=None)` and use the returned indices
      to build `top_traits`/`contributions`/`total_contributions`, instead
      of the manual `np.argsort` duplicate. The two pre-calculated
      (`trait_contrib_df`-based) branches are unchanged — they never
      referenced `feature_selection` and already select via
      `.head()` on data pre-sorted elsewhere.
- [x] 3.3 In `pipeline/steps/generate_static_figures.py` (~L407-414), remove
      the `feature_selection=config.pca.feature_selection_strategy,`
      argument from the `create_feature_contribution_plot(...)` call. This
      MUST land in the same commit as 3.1 (see commit-grouping note above).
- [x] 3.4 Grep the repo for any other caller of
      `create_feature_contribution_plot(...feature_selection=...)` besides
      the one fixed in 3.3, to confirm no other call site breaks.

## Task 4: Verify no regressions

- [x] 4.1 Run all Task 1 tests, confirm 1.1, 1.2, 1.3, and 1.5 now PASS
      (green), and 1.6/1.4 continue to PASS.
- [x] 4.2 Run the full test suite (`uv run pytest --cov --cov-branch`),
      confirm no regressions beyond the tests intentionally changed in
      Tasks 1-3.
- [x] 4.3 Run `uv run ruff check src/sleap_roots_analyze tests` and
      `uv run black --check src/sleap_roots_analyze tests`; fix any issues.
- [x] 4.4 Run `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline
      filter --baseline-path .mypy-baseline.txt`, confirm no new errors
      against the frozen baseline.
- [x] 4.5 Run `openspec validate fix-pca-feature-selection-wiring --strict`,
      resolve any issues.

## Task 5: Docs and golden config review

- [x] 5.1 Add a `### Fixed` entry to `docs/CHANGELOG.md` `[Unreleased]`
      describing both bugs. The entry MUST explicitly state that
      `create_feature_contribution_plot` **removes** the `feature_selection`
      parameter (not just "changes" it), and note that no in-repo caller
      nor the verified downstream consumer (`bloom`) passed it, so this is
      backward-compatible in practice despite being a signature change.
- [x] 5.2 Regenerate (or otherwise review) the static-figure outputs for
      the golden viz configs that set `feature_selection_strategy:
      "top_variance"`: `configs/active/viz/mo_soybean_2021_grouped.yaml`
      and `configs/examples/viz_{comprehensive,minimal,publication,standard}.yaml`.
      Confirm `pca_biplot.png` now shows `top_variance`-selected features
      instead of the previously-silent `vector_length` fallback, and that
      `pca_feature_contributions.png` output is unchanged for these
      configs. Explicitly out of scope (do not regenerate): the flat
      `configs/active/viz_standard.yaml` and
      `configs/active/viz_turface_19genotypes.yaml` files, and the
      top-level `configs/viz_*.yaml` duplicates — these are pre-reorg
      orphans superseded by the `configs/active/viz/` subfolder (confirmed
      via `git log`: last touched by `cc7ede1`, before the `active/viz/`
      split, and not referenced by any `configs/active/run_manifest*.yaml`),
      not live golden configs.

      **Actual outcome**: the real target
      (`configs/active/viz/mo_soybean_2021_grouped.yaml`) points to a genuine
      dataset on a network share via `run-all`, and the four
      `configs/examples/viz_*.yaml` templates all have `csv_path: ???`
      placeholders and cannot be run directly. Given a full local pipeline
      run's cost/duration, the user opted for a synthetic before/after
      comparison instead of a real `run-all` invocation: a scratch script
      loaded the real `tests/data/Turface_all_traits_2024.csv` dataset (real
      trait names, 187 samples, 38 traits), ran PCA with 5 components, and
      called `create_pca_biplot(..., feature_selection="vector_length")`
      (simulating the pre-fix silent fallback) vs.
      `feature_selection="top_variance"` (post-fix, current code) for the
      same PC pair (PC1/PC4, chosen because it empirically diverges for this
      dataset — PC1/PC2 coincidentally produced the same top-8 set both
      ways). Confirmed the two calls select different, non-overlapping-only
      feature sets (`Depth.mm`, `Width-to-Depth.Ratio`,
      `Median.Number.of.Roots`, ... vs. `Surface.Area.mm2`,
      `Average.Root.Orientation.deg`, `Steep.Angle.Frequency`, ...) and
      rendered both biplots to confirm the arrows/labels differ visually as
      expected. This demonstrates the fix's real-world effect without
      requiring the actual golden-config network-drive run, which remains
      the user's own follow-up to do post-merge if desired.
