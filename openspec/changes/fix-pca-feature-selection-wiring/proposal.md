## Why

Fixes #202. `pca.feature_selection_strategy` defaults to `"top_variance"` in
`PCAConfig` (`pipeline/config/components.py:421`) and is the explicit value
in several active viz configs (e.g.
`configs/active/viz/mo_soybean_2021_grouped.yaml`, and the
`configs/examples/viz_*.yaml` templates). It is threaded from
`generate_static_figures.py` into two visualization functions, and in both
cases the configured value is silently not what actually selects the
plotted features:

- `create_pca_biplot` (`visualization.py:2260-2269`) maps `feature_selection`
  to a `method` variable via `if`/`elif` branches for `extreme`,
  `top_absolute`, `top_contribution`, and `vector_length` — there is no
  `top_variance` branch, so `"top_variance"` falls into
  `else: method = "vector_length"` with no warning. `top_variance` and
  `vector_length` are empirically different selection criteria
  (eigenvalue-weighted variance contribution vs. unweighted Euclidean norm),
  so this silently changes which features are plotted for every config that
  sets `feature_selection_strategy: "top_variance"`.
- `create_feature_contribution_plot` (`visualization.py:1967-2013` onward)
  accepts a `feature_selection` parameter that is never referenced anywhere
  in the function body — every code path ranks by precomputed/derived
  variance contribution regardless of what is passed.

See `design.md` for the full investigation, including a verified gotcha in
`select_top_features_from_pca`'s `"top_variance"` method that the fix for
`create_pca_biplot` must account for, and the reasoning for removing
(rather than wiring up) `create_feature_contribution_plot`'s parameter.

## What Changes

- `create_pca_biplot` (`visualization.py`): add an explicit `top_variance`
  branch to the `feature_selection` → `method` mapping that passes
  `method="top_variance"` straight through (mirroring
  `create_umap_colored_by_top_traits`, `visualization.py:2751-2783`), and
  change the `else` branch to raise `ValueError` for genuinely unrecognized
  `feature_selection` values instead of silently substituting
  `vector_length`. When `method == "top_variance"`, do **not** pass
  `pc_indices=[pc_x_idx, pc_y_idx]` to `select_top_features_from_pca` — pass
  `pc_indices=None` instead, matching the existing `top_variance` workaround
  in `create_umap_colored_by_top_traits` (see `design.md` for why). Update
  the function's docstring to list `top_variance` as valid and document the
  `ValueError`.
- **BREAKING**: `create_feature_contribution_plot` (`visualization.py`):
  **remove** the `feature_selection` parameter entirely (do not wire it
  up). The chart's
  own title (`f"Top {n} Feature Contributions to First {k} PCs"`) asserts
  the displayed traits *are* the top contributors to variance — a
  non-contribution selection method would make the title misdescribe its
  own content, and the sibling `create_feature_contribution_heatmap`
  (`visualization.py:3114-3203`, same purpose, no such parameter) is
  existing precedent for the same reasoning. The on-the-fly
  (backward-compatibility) branch, which currently duplicates the
  `top_variance` ranking formula inline, is refactored to delegate to
  `select_top_features_from_pca(method="top_variance", pc_indices=None,
  ...)` instead of reimplementing it. The two pre-calculated-contributions
  branches are unchanged (they already select by taking the head of a
  DataFrame the caller pre-sorted by contribution elsewhere; they never
  referenced `feature_selection` and don't need the delegation).
- `pipeline/steps/generate_static_figures.py` (~L407-414): remove the
  `feature_selection=config.pca.feature_selection_strategy` argument from
  the `create_feature_contribution_plot(...)` call site, since the
  parameter no longer exists. The `create_pca_biplot` call site
  (~L342-351) and the `create_umap_colored_by_top_traits` call site
  (~L534-541) are unchanged — they already pass
  `feature_selection=config.pca.feature_selection_strategy` and now behave
  correctly for `"top_variance"` once the `create_pca_biplot` fix lands.
- Add regression tests: `create_pca_biplot(feature_selection="top_variance")`
  selects the same feature indices as calling
  `select_top_features_from_pca(method="top_variance", pc_indices=None,
  ...)` directly for the displayed PCs, and `create_pca_biplot` raises
  `ValueError` for an unrecognized `feature_selection` string. Update/add a
  test confirming `create_feature_contribution_plot`'s signature no longer
  accepts `feature_selection` and that the `generate_static_figures.py`
  call site compiles without it.
- Regenerate/review the golden viz-config runs that set
  `feature_selection_strategy: "top_variance"`
  (`configs/active/viz/mo_soybean_2021_grouped.yaml` and the
  `configs/examples/viz_{comprehensive,minimal,publication,standard}.yaml`
  templates) since their `pca_biplot` output will change once the fix
  lands — bars/arrows previously ranked by `vector_length` will be ranked
  by `top_variance` instead. Explicitly out of scope for regeneration: the
  flat `configs/active/viz_standard.yaml` /
  `configs/active/viz_turface_19genotypes.yaml` files and the top-level
  `configs/viz_*.yaml` duplicates also match `feature_selection_strategy:
  "top_variance"` by grep, but are pre-reorg orphans superseded by the
  `configs/active/viz/` subfolder split (confirmed via `git log`: last
  touched by `cc7ede1`, well before `configs/active/viz/` was introduced,
  and not referenced by any `configs/active/run_manifest*.yaml`) — not
  live golden configs, so they are not regenerated or reviewed here.

## Impact

- Affected specs: `visualization-pipeline` — modify the "Pass parameters to
  PCA biplot" scenario area to cover `feature_selection` correctness, and
  the "PCA Feature Contribution Bar Chart" requirement to reflect the
  parameter's removal.
- Affected code:
  - `src/sleap_roots_analyze/visualization.py`
    (`create_pca_biplot`, `create_feature_contribution_plot`)
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py`
    (~L407-414)
  - `tests/test_visualization.py`
  - `docs/CHANGELOG.md` `[Unreleased]` — new `### Fixed` entry
- Breaking change (narrow, intentional): `create_feature_contribution_plot`
  no longer accepts a `feature_selection` keyword argument. Verified against
  the real downstream consumer
  (`Salk-Harnessing-Plants-Initiative/bloom`, `staging` branch, genuine
  `sleap-roots-analyze>=0.1.0a5` pip dependency): its `pca_analysis` MCP
  tool calls both `create_pca_biplot` and `create_feature_contribution_plot`
  but never passes `feature_selection` to either, so both changes are
  backward-compatible with that consumer.
- Visual/output change: any pipeline run configured with
  `pca.feature_selection_strategy: "top_variance"` will now produce a
  different `pca_biplot.png` (features selected by `top_variance` instead
  of the previously-silent `vector_length` fallback). No change for
  `"extreme"`, `"top_absolute"`, or `"top_contribution"` configs — those
  branches were already handled correctly. `pca_feature_contributions.png`
  output is unchanged for all configs (it always ranked by variance
  contribution; only the never-effective parameter is removed).
- Explicitly out of scope (tracked separately, not touched here):
  - #203 — `PCAAnalysisStep` hardcodes `pc_indices=[0,1]` in
    `pipeline/steps/pca_analysis.py`; a different call site needing its own
    maintainer decision on PC scoping.
  - #206 — redesigning `n_top_features`'s semantics.
  - #207 — UMAP top-traits PC1-negative-only bug.
  - #64 and #68 — already closed as superseded by this issue and #206
    respectively.
