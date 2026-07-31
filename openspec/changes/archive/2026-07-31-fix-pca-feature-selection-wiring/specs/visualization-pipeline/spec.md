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

### Requirement: PCA Feature Contribution Bar Chart
The visualization pipeline SHALL generate a stacked horizontal bar chart showing per-PC variance contributions for top features, matching notebook output.

`create_feature_contribution_plot` SHALL always select the displayed
features by total variance contribution (equivalent to
`select_top_features_from_pca(method="top_variance", pc_indices=None,
...)`) and SHALL NOT accept a `feature_selection` parameter, since the
chart's bars always plot true per-PC variance contribution regardless of
which traits are shown — a non-contribution selection criterion would make
the chart's title (which asserts the displayed traits are the top
contributors) misdescribe its own content. `GenerateStaticFiguresStep`
SHALL NOT pass a `feature_selection` argument to this function.

#### Scenario: Standard PCA analysis complete
- **WHEN** PCA results are available and `static_viz.create_pca_plots` is enabled
- **THEN** the pipeline SHALL generate a feature contribution bar chart via `create_feature_contribution_plot()`
- **AND** the chart SHALL be saved alongside other PCA figures (scree plot, biplot, heatmaps)

#### Scenario: create_feature_contribution_plot has no feature_selection parameter
- **WHEN** `create_feature_contribution_plot`'s signature is inspected
- **THEN** it SHALL NOT include a `feature_selection` parameter
- **AND** calling it with a `feature_selection` keyword argument SHALL raise `TypeError`

#### Scenario: On-the-fly contribution ranking matches select_top_features_from_pca
- **WHEN** `create_feature_contribution_plot` computes contributions on the
  fly (no pre-calculated `trait_contrib_df`/`feature_contributions` in
  `pca_results`)
- **THEN** the top features selected SHALL be identical, in the same order,
  to calling `select_top_features_from_pca(method="top_variance",
  pc_indices=None, ...)` directly with the same loadings and eigenvalues
