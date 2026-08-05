## 1. `create_interactive_pca_with_images`

- [x] 1.1 Write failing tests in `tests/test_interactive_visualization.py`:
      - Default-equivalence: call with `show_loadings=True`, no
        `feature_selection`; independently compute the expected feature set
        via `pca_viz_results["feature_contributions"].nlargest(n_loadings, "total_contribution").index`
        directly in the test (not by re-deriving it from the new code path);
        assert the actual annotated features equal that set.
      - `@pytest.mark.parametrize("feature_selection", ["extreme", "top_absolute", "top_contribution", "vector_length"])`:
        for each, independently call `select_top_features_from_pca(loadings=pca_viz_results["loadings"], eigenvalues=pca_viz_results["eigenvalues"], n_features_total=len(pca_viz_results["feature_names"]), n_features_to_select=n_loadings, method=feature_selection, pc_indices=[pc_x, pc_y])`
        and assert the function's annotated features equal that set exactly
        (set equality, not order — `nlargest`/`np.argsort` tie-breaking is
        not guaranteed identical).
      - PC-indexing regression test: call with `components=(2, 3)` (a pair
        that diverges from the `(0, 1)` default) and a PC-sensitive method
        (e.g. `"extreme"`); assert the selected features match
        `select_top_features_from_pca(..., pc_indices=[2, 3])`, not
        `pc_indices=[1, 2]` (which is what copying `create_pca_biplot`'s
        1-indexed-to-0-indexed `-1` conversion by analogy would produce) —
        this pins down that `components` here is already 0-indexed.
      - Invalid `feature_selection` (e.g. `"bogus"`) raises `ValueError`,
        tested with **both** `show_loadings=True` and `show_loadings=False`
        (validation must be unconditional).
      - `n_loadings=0` → no loading-arrow annotations added.
      - `n_loadings` greater than the total number of features → no error,
        all available features selected.
- [x] 1.2 Add `feature_selection: str = "top_variance"` parameter. Validate
      unconditionally against `VALID_SELECTION_METHODS` (imported from
      `sleap_roots_analyze.pca`), before the `show_loadings` branch. Replace
      the `feature_contributions.nlargest(...)` call with
      `select_top_features_from_pca(loadings=..., eigenvalues=pca_results["eigenvalues"], n_features_total=len(pca_results["feature_names"]), n_features_to_select=n_loadings, method=feature_selection, pc_indices=None if feature_selection == "top_variance" else [pc_x, pc_y])`,
      mapping returned integer indices back to feature names via
      `pca_results["feature_names"][idx]`. Do **not** apply any `-1` offset
      to `pc_x`/`pc_y` — they are already 0-indexed here.
- [x] 1.3 Update the docstring: add a `feature_selection` `Args:` entry with
      a bulleted list of the five valid values and one-line descriptions
      (matching `create_pca_biplot`'s docstring style in `visualization.py`),
      and a `Raises: ValueError` entry for an unrecognized value.
- [x] 1.4 Run the new tests and confirm they pass.

## 2. `create_interactive_pca_plot`

- [x] 2.1 Write the same failing-test shape as 1.1 for this function
      (default-equivalence, parametrized methods with independent ground
      truth, PC-indexing regression test, unconditional-validation test,
      `n_loadings=0`, `n_loadings` overflow).
- [x] 2.2 Apply the same `feature_selection` parameter, unconditional
      validation, and delegation as 1.2.
- [x] 2.3 Update the docstring with the same `Args`/`Raises` format as 1.3.
- [x] 2.4 Run the new tests and confirm they pass.

## 3. Verification

- [x] 3.1 Confirm `create_umap_colored_by_top_traits`'s `top_variance` branch
      is unchanged (no code edit — grep/read to reconfirm after this change,
      since it's the other half of this issue's acceptance criteria; its
      existing `#207` test coverage is what actually guards it, not this
      task). This is a sanity check, not a commit.
- [x] 3.2 Add a changelog entry to `docs/CHANGELOG.md`'s `[Unreleased]`
      section documenting the new `feature_selection` parameter on both
      functions, closing `#209`, matching the entries already present for
      `#202`/`#203`/`#204`/`#206`.
- [x] 3.3 Run `uv run pytest --cov --cov-branch` and confirm no regressions.
- [ ] 3.4 Run `/pre-merge-check` (black, ruff, full suite, coverage, OpenSpec
      validation, Copilot triage) as the closing check for the whole
      `#202`/`#203`/`#204`/`#206`/`#207`/`#209` effort.
