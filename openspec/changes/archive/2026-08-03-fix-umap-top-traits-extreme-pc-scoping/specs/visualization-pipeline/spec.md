## MODIFIED Requirements

### Requirement: Pipeline Step Parameter Passing

The `GenerateStaticFiguresStep` SHALL pass genotype highlighting parameters from configuration to underlying plotting functions.

The `PCAAnalysisStep` SHALL use the post-filtering feature names returned by
`perform_pca_analysis()` (via `pca_results["feature_names"]`) instead of the original
`trait_cols` list when constructing the loadings DataFrame index, computing
`n_features_total` for feature selection, and mapping feature indices to names.

The `PCAAnalysisStep` SHALL log the names and count of any traits excluded due to zero
variance, and SHALL emit a Python `UserWarning` when more than 50% of input traits are
excluded.

The `PCAAnalysisStep` SHALL store `excluded_zero_variance_traits` (list of excluded trait
names) and `n_traits_after_filtering` (int) in its output metadata for downstream
inspection and reproducibility.

`create_pca_biplot` SHALL honor every `feature_selection` value it documents
— `"vector_length"`, `"extreme"`, `"top_absolute"`, `"top_contribution"`,
and `"top_variance"` — by selecting features using the matching method in
`select_top_features_from_pca()`, and SHALL raise `ValueError` for any other
value instead of silently substituting `"vector_length"`. When
`feature_selection == "top_variance"`, `create_pca_biplot` SHALL call
`select_top_features_from_pca(..., pc_indices=None)` rather than passing the
biplot's two displayed PC indices, since `select_top_features_from_pca`'s
`"top_variance"` method ranks across all retained PCs regardless of
`pc_indices` and passing the 2-index list would misleadingly imply the
biplot's PC scope is honored.

`create_umap_colored_by_top_traits` SHALL scope `pc_indices` to all retained PCs (derived
from `pca_results["n_components_selected"]`, `variance_threshold`, or the 95% cumulative
variance default, in that priority order) for **every** `feature_selection` method, not
only `"top_variance"`.

For `feature_selection == "extreme"`, `create_umap_colored_by_top_traits` SHALL build the
plotted trait set by round-robin selection across (PC, direction) pairs spanning all
scoped `pc_indices`, instead of taking a single block-ordered `select_top_features_from_pca(...,
n_features_to_select=n_traits)` call and truncating it to `top_indices[:n_traits]`. The
round-robin SHALL:
- maintain one sorted-loading iterator per (PC, direction) pair, each advancing its own
  position monotonically across passes (never re-scanning from the start);
- check candidates against a single **global** `seen` set at the moment a trait is popped,
  so a trait already claimed by one pair is skipped (not re-selected) by every other pair;
- order passes **direction-major, PC-minor**: pass 1 takes each retained PC's single
  most-negative unseen trait (PC1, PC2, PC3, ... in order), pass 2 takes each PC's single
  most-positive unseen trait, pass 3 takes each PC's second-most-negative unseen trait, and
  so on, continuing until `n_traits` traits are collected or every pair is exhausted;
- record the (PC, direction) pair that actually claimed each selected trait, on a
  first-come-first-claimed basis per the pass order above, so the source is deterministic
  even when a trait is extreme on more than one PC.

The direction-major/PC-minor pass order is required specifically so that when `n_traits`
is smaller than `2 * len(pc_indices)`, every retained PC still receives at least one
representative before any PC receives a second — a PC-major pass order (all of PC1's
extremes before moving to PC2) would reproduce the same class of bug one level up,
crowding out later PCs whenever per-PC-pair supply exceeds the `n_traits` budget.

This round-robin construction lives in `create_umap_colored_by_top_traits` only;
`select_top_features_from_pca`'s own `"extreme"` method in `pca.py` (and its
per-direction-per-PC `n_features_to_select` count semantics, relied on unsliced by
`PCAAnalysisStep`) SHALL remain unchanged.

For `feature_selection == "extreme"`, each subplot's subtitle SHALL report the actual (PC,
direction) pair that selected that trait, rather than always re-deriving direction from
`loadings[trait_idx, 0]` (PC1's loading).

#### Scenario: Pass parameters to PCA biplot
- **WHEN** generating PCA biplot via `create_pca_biplot`
- **THEN** the step SHALL pass `config.static_viz.genotypes_to_color` to the function
- **AND** the step SHALL pass `config.static_viz.highlight_genotypes` to the function

#### Scenario: Pass parameters to PC boxplots
- **WHEN** generating PC boxplots via `create_pc_genotype_boxplots`
- **THEN** the step SHALL pass `config.static_viz.highlight_genotypes` to the function
- **AND** the highlighted genotypes SHALL appear in gold with bold labels

#### Scenario: PCA step handles zero-variance traits gracefully
- **WHEN** the input DataFrame contains traits with zero variance (constant values)
- **THEN** the PCA step SHALL complete successfully using only non-zero-variance traits
- **AND** the loadings CSV index SHALL match the actual features used in PCA
- **AND** `excluded_zero_variance_traits` SHALL list the excluded trait names in metadata
- **AND** `n_traits_after_filtering` SHALL reflect the count of traits actually used

#### Scenario: PCA step warns on high zero-variance fraction
- **WHEN** more than 50% of input traits have zero variance
- **THEN** the PCA step SHALL emit a `UserWarning` indicating potential data quality issues
- **AND** the step SHALL still complete successfully with the remaining traits

#### Scenario: PCA step with no zero-variance traits
- **WHEN** all input traits have non-zero variance
- **THEN** the PCA step SHALL behave identically to current behavior
- **AND** `excluded_zero_variance_traits` SHALL be an empty list
- **AND** no warning SHALL be emitted

#### Scenario: create_pca_biplot honors top_variance feature selection
- **WHEN** `create_pca_biplot` is called with `feature_selection="top_variance"`
- **THEN** the features selected for display SHALL be the same set returned
  by `select_top_features_from_pca(method="top_variance", pc_indices=None,
  ...)` called directly with the same loadings, eigenvalues, feature count,
  and `top_n_features`
- **AND** the selection SHALL NOT silently fall back to the
  `"vector_length"` method

#### Scenario: create_pca_biplot rejects an unrecognized feature_selection value
- **WHEN** `create_pca_biplot` is called with a `feature_selection` value
  that is not one of `"vector_length"`, `"extreme"`, `"top_absolute"`,
  `"top_contribution"`, or `"top_variance"`
- **THEN** the function SHALL raise `ValueError`
- **AND** it SHALL NOT silently substitute `"vector_length"`

#### Scenario: create_pca_biplot continues to honor pre-existing feature_selection methods
- **WHEN** `create_pca_biplot` is called with `feature_selection` set to
  `"vector_length"`, `"extreme"`, `"top_absolute"`, or `"top_contribution"`
- **THEN** the features selected for display SHALL match a direct
  `select_top_features_from_pca(method=<same value>, pc_indices=[pc_x_idx,
  pc_y_idx])` call with the same loadings, eigenvalues, feature count, and
  `top_n_features`

#### Scenario: create_umap_colored_by_top_traits scopes pc_indices beyond PC1/PC2 for non-top_variance methods
- **GIVEN** `pca_results` resolves to 3 or more retained PCs (via
  `n_components_selected` or the cumulative variance threshold)
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection` set to `"extreme"`, `"top_absolute"`, or
  `"top_contribution"`
- **THEN** the PC indices considered for feature selection SHALL include every
  retained PC, not just PC1 and PC2

#### Scenario: create_umap_colored_by_top_traits shows traits from multiple PCs and both directions for extreme selection
- **GIVEN** `pca_results` resolves to 3 or more retained PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and `n_traits` large enough to span more than
  one (PC, direction) pair
- **THEN** the plotted trait set SHALL NOT be a subset of PC1's `n_traits`
  most-negative-loading indices
- **AND** the plotted trait set SHALL include at least one trait whose
  extreme loading comes from a PC other than PC1
- **AND** the plotted trait set SHALL include at least one trait selected for
  its positive loading

#### Scenario: create_umap_colored_by_top_traits extreme selection round-robins fairly across PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`, `n_traits=6`, and 3 retained PCs each with
  distinct most-extreme traits in both directions
- **THEN** the round-robin construction SHALL select traits from PC1, PC2,
  and PC3 before exhausting `n_traits`, rather than filling the entire
  `n_traits` budget from PC1 alone

#### Scenario: create_umap_colored_by_top_traits gives every retained PC at least one representative before any PC gets a second
- **GIVEN** `pca_results` resolves to 5 retained PCs, each with distinct
  most-extreme traits in both directions
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and `n_traits=6` (fewer than `2 * 5` PC×direction
  pairs)
- **THEN** the plotted trait set SHALL include at least one trait from each of
  the 5 retained PCs
- **AND** no PC SHALL contribute a second trait until every retained PC has
  contributed at least one

#### Scenario: create_umap_colored_by_top_traits extreme selection handles fewer distinct extreme traits than n_traits
- **GIVEN** the total number of distinct traits reachable across all (PC,
  direction) pairs (after deduplication) is less than `n_traits`
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`
- **THEN** the function SHALL return a plotted trait set shorter than
  `n_traits` without raising an exception
- **AND** the figure's unused subplot axes SHALL be removed as they are today

#### Scenario: create_umap_colored_by_top_traits deduplicates a trait that is extreme on more than one PC
- **GIVEN** a single trait is the most-extreme (same or opposite direction)
  loading on two different retained PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`
- **THEN** that trait SHALL appear exactly once in the plotted trait set
- **AND** the (PC, direction) pair whose round-robin turn claims it first
  (per pass order) SHALL be recorded as its source for subtitle purposes
- **AND** the freed round-robin slot for the other PC SHALL be filled by that
  PC's next-ranked unseen candidate rather than left empty

#### Scenario: create_umap_colored_by_top_traits subtitle reports the true source PC and direction for extreme selection
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and a plotted trait was selected for its
  extreme loading on PC2 (not PC1)
- **THEN** that trait's subplot subtitle SHALL reference PC2 (e.g. `"PC2+"` or
  `"PC2-"`), not PC1

#### Scenario: create_umap_colored_by_top_traits leaves select_top_features_from_pca's extreme method unchanged
- **WHEN** `select_top_features_from_pca(method="extreme", ...)` is called
  directly (e.g. from `PCAAnalysisStep`)
- **THEN** it SHALL continue to return the block-ordered list
  (`PC1_neg, PC1_pos, PC2_neg, PC2_pos, ...`) with `n_features_to_select`
  traits per direction per PC, unchanged from current behavior

#### Scenario: create_umap_colored_by_top_traits top_variance behavior is unchanged
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="top_variance"`
- **THEN** the selected traits and PC scoping SHALL be identical to current
  behavior (unaffected by this change)
