## Why

`create_umap_colored_by_top_traits()` (`src/sleap_roots_analyze/visualization.py`) is called from
`GenerateStaticFiguresStep` (`pipeline/steps/generate_static_figures.py:531-541`) for every
viz-pipeline run, with `n_traits=6` hardcoded. Two compounding hardcodes make its output silently
wrong for the `"extreme"` feature-selection strategy — the strategy used by the majority of active
viz configs (`configs/active/viz/*.yaml`):

1. `pc_indices` is scoped to all retained PCs only when `feature_selection == "top_variance"`;
   every other method (`"extreme"`, `"top_absolute"`, `"top_contribution"`) falls into
   `pc_indices = [0, 1]`, capping PC scope at 2 regardless of `n_components`. Same root-cause
   pattern as the still-open #203 (a different call site, `pipeline/steps/pca_analysis.py`) and the
   one already fixed for `create_pca_biplot` (#202/#214).
2. For `method="extreme"`, `select_top_features_from_pca()` (`pca.py`) returns a block-ordered
   list (`[PC1-neg × n, PC1-pos × n, PC2-neg × n, PC2-pos × n, ...]`), and the caller slices
   `top_indices[:n_traits]`. Because `n_traits` is passed as both the per-direction-per-PC count
   *and* the final slice length, the first block (PC1's `n_traits` most-negative-loading traits)
   already fills the entire slice — PC1's positive extremes and every PC2+ trait are never shown,
   for any `n_components`. Verified empirically against synthetic loadings (see #207).

The net effect: for every active config with `feature_selection_strategy: "extreme"`, the "UMAP
colored by top traits" plot shows only PC1's most-negative-loading traits, never PC1's positive
extremes or any other PC — contradicting the config comments that describe this plot as showing
"the most extreme positive and negative loadings per PC" (plural, both directions).

`top_absolute`/`top_contribution` inherit hardcode 1 only (their `select_top_features_from_pca`
branches already return an exactly-length-`n` ranked list, so `[:n_traits]` is a no-op); neither
appears in any active config today, so this proposal fixes their `pc_indices` scoping as a side
effect of the shared code path without adding dedicated new logic for them.

## What Changes

- In `create_umap_colored_by_top_traits`, compute `pc_indices` from the retained-PC count
  (`n_components_selected` / `cumulative_variance_ratio` / `variance_threshold`, the same logic
  already used for `"top_variance"`) for **every** `feature_selection` method, replacing the
  `else: pc_indices = [0, 1]` branch.
- For `method="extreme"` specifically, replace the single `select_top_features_from_pca(...,
  n_features_to_select=n_traits)` call + `top_indices[:n_traits]` truncation with a round-robin
  construction built directly in `create_umap_colored_by_top_traits`: one sorted-loading iterator
  per (PC, direction) pair across all retained `pc_indices`, each iterator advancing its own
  position monotonically across passes and checked against one **global** `seen` set at pop time
  (so a trait claimed by one pair is skipped, not re-selected, by every other pair). Passes are
  ordered **direction-major, PC-minor** — pass 1 takes each retained PC's single most-negative
  unseen trait (PC1, PC2, PC3, ...), pass 2 takes each PC's single most-positive unseen trait, pass
  3 takes each PC's second-most-negative unseen trait, and so on — continuing until `n_traits`
  traits are collected or every pair is exhausted. This ordering is required (not the naive
  PC-major "PC1-, PC1+, PC2-, PC2+, ..." grouping) specifically so that when `n_traits` is smaller
  than `2 * len(pc_indices)`, every retained PC still gets at least one representative before any
  PC gets a second — otherwise the same class of bug resurfaces one level up (PC1..PC3 crowd out
  PC4+ whenever traits-per-PC-pair exceeds the budget). `select_top_features_from_pca()`'s own
  block-ordered `"extreme"` behavior in `pca.py` is **not** modified — `pipeline/steps/pca_analysis.py`
  depends on that function's existing per-direction-per-PC count semantics (it consumes every
  returned index, unsliced), so changing the shared function would be an unscoped side effect on a
  different call site.
- Track which (PC, direction) pair each selected trait actually came from (first-come-first-claimed
  in the round-robin order above, so deterministic when a trait is extreme on more than one PC),
  and use that to fix the per-subplot subtitle label (currently always reports "PC1+/PC1-" by
  re-reading `loadings[trait_idx, 0]`, which only looked correct because every trait previously
  came from PC1) so it names the real source PC and direction.
- Update the `feature_selection` Args docstring line for `"extreme"` in
  `create_umap_colored_by_top_traits` (currently "Top N most positive and negative for first 2
  PCs"), which becomes inaccurate once PC scope is no longer capped at 2.
- Correct the two config comments that describe this plot's intended-but-currently-broken
  behavior to match the now-true behavior — including their independently-inaccurate claim that
  `pca.n_top_features` controls "UMAP coloring" (it does not: the UMAP call site hardcodes
  `n_traits=6` regardless of `n_top_features`, which only feeds `PCAAnalysisStep`'s separate
  selection; this claim is wrong today and would stay wrong if only the PC/direction description
  were corrected):
  `configs/active/viz/viz_alfalfa_gwas_wave_1_grouped.yaml:29-31` and
  `configs/active/viz/alfalfa_gwas_wave1_canola_models.yaml:58-61`.
- Add a `docs/CHANGELOG.md` `### Fixed` entry under `[Unreleased]`, matching the established
  pattern for this bug class (e.g. the #202 `create_pca_biplot` fix, the #210 OOM fix).

## Out of Scope

- #206's full redesign (1-most-positive + 1-most-negative per retained PC, no `n_traits`
  truncation at all, panel grid sized to the actual pair count). That is a separate, larger design
  change. This proposal keeps `n_traits` as-is and only fixes how the existing count is
  distributed across PCs/directions. Leave a cross-reference to #206 in the PR description rather
  than closing it.
- `pipeline/steps/pca_analysis.py`'s independent `pc_indices` hardcode (#203) — different call
  site, still open, not touched here.
- `bloommcp` (`Salk-Harnessing-Plants-Initiative/bloom`, `staging`) calls
  `create_umap_colored_by_top_traits` without ever passing `feature_selection`, so it always uses
  `"top_variance"` — unaffected by this change (already verified against that repo; see #207
  comment thread).

## Impact

- Affected specs: `visualization-pipeline` (one requirement modified).
- Affected code:
  - `src/sleap_roots_analyze/visualization.py` — `create_umap_colored_by_top_traits`: PC-index
    scoping, `"extreme"` round-robin selection, subtitle source tracking, docstring correction.
  - `tests/test_visualization.py` — regression tests: (1) plotted trait set for
    `feature_selection="extreme"` is not a subset of PC1's single-direction extremes and spans
    every retained PC even when `n_traits < 2 * len(pc_indices)`; (2) corrected subtitle labeling;
    (3) `pc_indices` scoping for `top_absolute`/`top_contribution`; (4) `top_variance` output is
    byte-for-byte unchanged (backward compatibility); (5) exhaustion when distinct extreme traits
    are fewer than `n_traits`; (6) a dedup case where one trait is extreme on more than one PC.
  - `tests/test_pca.py` (or equivalent) — direct unit test confirming
    `select_top_features_from_pca(method="extreme", ...)` is unchanged (still block-ordered,
    still per-direction-per-PC `n_features_to_select` count), decoupled from the UMAP plotting
    change.
  - `configs/active/viz/viz_alfalfa_gwas_wave_1_grouped.yaml`,
    `configs/active/viz/alfalfa_gwas_wave1_canola_models.yaml` — comment corrections only, no
    behavioral config changes.
  - `docs/CHANGELOG.md` — `[Unreleased]` → `### Fixed` entry.
- No `pyproject.toml` version bump in this PR (per repo convention, cut separately).
