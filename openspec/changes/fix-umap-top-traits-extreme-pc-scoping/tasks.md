## 1. PC-index scoping

- [x] 1.1 Write failing tests in `tests/test_visualization.py` asserting that for
      `feature_selection` in `{"extreme", "top_absolute", "top_contribution"}`, with
      `pca_results` resolving to 3+ retained PCs, a trait whose extreme/top loading exists only on
      PC3 (not PC1/PC2) is reachable in the plotted output — i.e. `pc_indices` is not capped at
      `[0, 1]`. Use hand-built loadings (not QR-random fixtures) so the expected source PC is known
      in advance, not re-derived via the same `np.argsort` logic under test.
- [x] 1.2 In `create_umap_colored_by_top_traits` (`src/sleap_roots_analyze/visualization.py`),
      remove the `if feature_selection == "top_variance": ... else: pc_indices = [0, 1]` branch;
      compute `pc_indices` from the retained-PC count (`n_components_selected` /
      `variance_threshold` / 95% default) unconditionally, for every `feature_selection` value.
      Confirm the tests from 1.1 pass.

## 2. Extreme-method round-robin selection

- [x] 2.1 Write failing regression tests using a hand-built loadings array (9 features x 3+ PCs,
      each PC given unambiguous, non-overlapping most-negative/most-positive traits — see
      `test_visualization.py`'s existing `test_create_umap_colored_by_top_traits` for fixture
      style) asserting, for `feature_selection="extreme"`:
      - the plotted trait set is NOT a subset of PC1's `n_traits` most-negative-loading indices;
      - the plotted trait set includes a trait from a non-PC1 source and a positively-loaded trait;
      - with 5 retained PCs and `n_traits=6` (fewer than `2 * 5` PC×direction pairs), the plotted
        set includes at least one trait from each of the 5 PCs, and no PC contributes a second
        trait until every PC has contributed one (this is the case that would fail under a naive
        PC-major round-robin ordering — it must drive the pass ordering below, not just PC-major
        grouping).
- [x] 2.2 In `create_umap_colored_by_top_traits`, replace the `select_top_features_from_pca(...,
      method="extreme")` call + `top_indices[:n_traits]` truncation with a round-robin
      construction: one sorted-loading iterator per (PC, direction) pair over the scoped
      `pc_indices`, each advancing monotonically across passes and checked against one global
      `seen` set at pop time. Order passes **direction-major, PC-minor**: pass 1 = each PC's single
      most-negative unseen trait (PC1, PC2, PC3, ... in order), pass 2 = each PC's single
      most-positive unseen trait, pass 3 = each PC's second-most-negative unseen trait, etc.,
      continuing until `n_traits` collected or all pairs exhausted. Do not modify
      `select_top_features_from_pca` in `pca.py`. Confirm the tests from 2.1 pass.
- [x] 2.3 Write a failing test asserting that when the total number of distinct traits reachable
      across all (PC, direction) pairs (after dedup) is less than `n_traits`, the function returns
      a shorter plotted set without raising, and unused subplot axes are still removed correctly.
      Confirm it passes against 2.2's implementation (should already hold; add the test to lock it
      in).
- [x] 2.4 Write a failing test asserting dedup behavior: construct loadings where one trait is the
      most-extreme loading on two different PCs; assert it appears exactly once in the plotted set,
      is attributed (for subtitle purposes) to whichever (PC, direction) pair's turn claims it first
      in pass order, and that the freed round-robin slot for the other PC is backfilled by that
      PC's next-ranked unseen candidate. Implement the tracking dict `{trait_idx: (pc_idx,
      direction)}` alongside 2.2 if not already present; confirm this test passes.
- [x] 2.5 Write a failing test asserting a plotted trait selected from PC2 gets a `"PC2+"`/`"PC2-"`
      subtitle (not `"PC1±"`). Update the subtitle-generation code to use the tracked (PC,
      direction) source instead of re-deriving direction from `loadings[trait_idx, 0]`. Confirm the
      test passes.
- [x] 2.6 Update the `feature_selection` Args docstring line for `"extreme"` in
      `create_umap_colored_by_top_traits` (currently "Top N most positive and negative for first 2
      PCs") to describe the corrected all-retained-PC, round-robin behavior.
- [x] 2.7 Write a test directly against `select_top_features_from_pca(method="extreme",
      pc_indices=[0, 1, 2], ...)` (decoupled from `create_umap_colored_by_top_traits`) asserting it
      still returns the block-ordered list (`PC1_neg, PC1_pos, PC2_neg, PC2_pos, PC3_neg,
      PC3_pos`) with `n_features_to_select` traits per direction per PC — confirming `pca.py` is
      genuinely untouched by this change.
- [x] 2.8 Write a backward-compatibility test capturing `create_umap_colored_by_top_traits`'s
      plotted trait set for `feature_selection="top_variance"` (using the existing
      `pca_viz_results`/`pca_viz_dataframe`/`umap_viz_results` fixtures) and asserting it is
      unchanged by this PR (this path was already correctly all-PC-scoped and untouched).
- [x] 2.9 Run the full `tests/test_visualization.py` suite to confirm no regressions.

## 3. Config comment and CHANGELOG corrections

- [x] 3.1 Update `configs/active/viz/viz_alfalfa_gwas_wave_1_grouped.yaml:29-31,37` to describe the
      now-true behavior (traits drawn from all retained PCs, both directions, round-robin
      distributed) and drop/correct the independently-inaccurate claim that `n_top_features`
      controls "UMAP coloring" (it does not — the UMAP call site hardcodes `n_traits=6`
      regardless of `config.pca.n_top_features`, which only feeds `PCAAnalysisStep`).
- [x] 3.2 Update `configs/active/viz/alfalfa_gwas_wave1_canola_models.yaml:58-61` likewise.
- [x] 3.3 Run `/validate-config` (or `validate_viz_config()`) against both configs to confirm the
      comment-only edits didn't break config validation.
- [x] 3.4 Add a `docs/CHANGELOG.md` `### Fixed` entry under `[Unreleased]`, matching the
      established pattern for this bug class (e.g. the #202 `create_pca_biplot` fix, the #210 OOM
      fix).

## 4. Verification and wrap-up

- [x] 4.1 Run `openspec validate fix-umap-top-traits-extreme-pc-scoping --strict`.
- [x] 4.2 Run `/pre-merge-check` (black, ruff, full pytest, coverage, self-review, OpenSpec
      validation, Copilot triage). Pre-PR 5-agent self-review found no BLOCKING issues;
      addressed the IMPORTANT findings (redundant `argsort` call, unused test variable,
      documented the first-claimed-PC-attribution/direction-imbalance caveats, added a
      CHANGELOG note on cached-figure regeneration). CI green on all 3 platforms — PR #216.
- [x] 4.3 Update PR description to note this supersedes itself if/when #206's redesign lands
      (cross-reference, don't close #206 or #209).

## Suggested commit plan (matches this repo's squash-merge-on-main convention)

1. `fix(#207): scope pc_indices to all retained PCs in top-traits UMAP` — tasks 1.1-1.2.
2. `fix(#207): round-robin extreme-method trait selection and fix PC subtitle source` — tasks
   2.1-2.9.
3. `fix(#207): correct top-traits config comments and changelog` — tasks 3.1-3.4.
