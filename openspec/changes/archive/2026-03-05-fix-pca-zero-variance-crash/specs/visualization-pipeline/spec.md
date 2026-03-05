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
