## Why

`talmolab/sleap-roots-analyze#209` is the last open item in the PCA
feature-selection consolidation effort (`#202`, `#203`, `#204`, `#206`, `#207`
all closed and merged). One consumer of PCA loadings still bypasses the
shared `select_top_features_from_pca()` (`pca.py:49`):
`create_interactive_pca_with_images` and `create_interactive_pca_plot`
(`interactive_visualization.py:259` and `:1275`) both hardcode
`feature_contributions.nlargest(n_loadings, "total_contribution")` to pick
which features get loading arrows when `show_loadings=True`, with no
`feature_selection` parameter and no strategy awareness.

The issue itself hedges on whether this is worth fixing, since neither
pipeline call site passes `show_loadings=True` (`generate_interactive.py`)
and no external consumer (`bloommcp`, `staging`) uses either function. It asks
whoever picks this up to check notebooks/docs for real usage before deciding
between delegating to `select_top_features_from_pca()` or just documenting
why the simpler path is intentional.

That check turned up real, current usage: 14 of the most recent
`trait_viz_*.ipynb` notebooks (through 2025-11-30) call
`create_interactive_pca_with_images(..., show_loadings=True, n_loadings=N_LOADINGS_INTERACTIVE)`
as a standard step in the analysis workflow, and `docs/PCA.md` documents the
same call as "Example 4: Interactive Visualization". This is not dead code —
it resolves toward delegation, matching the pattern already established for
`create_pca_biplot` and `create_umap_colored_by_top_traits`.

## What Changes

- Add a `feature_selection: str = "top_variance"` parameter to both
  `create_interactive_pca_with_images` and `create_interactive_pca_plot`.
- Both functions delegate their loading-arrow feature selection to
  `select_top_features_from_pca()` instead of hardcoding
  `nlargest(n_loadings, "total_contribution")`, validating `feature_selection`
  against the shared `VALID_SELECTION_METHODS` the same way `create_pca_biplot`
  does.
- Default `"top_variance"` ranks by the same total-variance-contribution basis
  as today's hardcoded `nlargest(n, "total_contribution")` (both derive from
  `_total_variance_contribution()`), so existing callers that pass
  `show_loadings=True` without `feature_selection` (all 14 notebooks above)
  get the same selected features in the typical case. The two ranking
  implementations are not a mathematical guarantee of an identical result in
  every case: `nlargest` excludes NaN rows and breaks ties by original order,
  while `select_top_features_from_pca`'s `np.argsort`-based ranking sorts NaN
  as largest and uses a non-stable sort — divergence is possible only if a
  feature's total contribution is NaN (a sign of upstream PCA failure, not a
  normal data condition) or multiple features tie exactly at the selection
  cutoff. Continuous variance-contribution values make an exact tie
  vanishingly unlikely in practice.

## Impact

- Affected code: `src/sleap_roots_analyze/interactive_visualization.py`
  (`create_interactive_pca_with_images`, `create_interactive_pca_plot`).
- Affected tests: `tests/test_interactive_visualization.py`.
- Affected spec: `visualization-pipeline` (adds a requirement covering
  interactive PCA loading-arrow feature selection).
- No config changes: no pipeline call site passes `show_loadings=True` today,
  so no `InteractiveVizConfig` plumbing is needed.
- No changes to `create_umap_colored_by_top_traits` (already fixed by `#207`)
  or `create_feature_contribution_plot` (resolved toward removal by `#202`,
  not addressed here).
- Closes `talmolab/sleap-roots-analyze#209`.

## Notes for implementation (from proposal review)

- **`pc_indices` are already 0-indexed here** — unlike `create_pca_biplot`,
  whose public `pc_x`/`pc_y` are 1-indexed and get a `-1` conversion before
  being passed as `pc_indices`, both interactive functions' `components`
  tuple (`pc_x, pc_y = components`) is already 0-indexed and used directly
  against `loadings[feature_idx, pc_x]`. Pass `[pc_x, pc_y]` unmodified — do
  **not** copy the biplot's `-1` adjustment by analogy, which would silently
  rank non-default methods against the wrong PC pair (the same bug class
  `#203`/`#207` already had to fix elsewhere in this effort).
- **Validate `feature_selection` unconditionally**, not only inside the
  `if show_loadings:` branch, so an invalid value fails fast regardless of
  whether loadings are actually displayed (matches `create_pca_biplot`'s
  unconditional validation).
- **Default-equivalence tests must compute the old ranking independently**
  (call `feature_contributions.nlargest(n_loadings, "total_contribution").index`
  directly in the test) rather than re-deriving it from the new
  `select_top_features_from_pca()` call, to avoid a circular test that can't
  catch a regression in the equivalence claim itself. Compare as sets, not
  exact order — `nlargest` and `np.argsort` break ties differently and are
  not guaranteed to agree on ordering for equal contribution values.
