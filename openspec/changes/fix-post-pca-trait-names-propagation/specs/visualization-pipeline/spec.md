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

The `PCAAnalysisStep` SHALL update `metadata["trait_names"]` and
`metadata["valid_trait_names"]` to the post-filtering feature name list
(`pca_results["feature_names"]`), matching the same-named keys' meaning
("traits still in play") maintained by every other filtering step in the
pipeline (`cleanup_traits.py`, `filter_heritability.py`,
`remove_outliers.py`, `detect_outliers.py`). The `PCAAnalysisStep` SHALL
additionally store `metadata["original_trait_names"]`, the pre-filtering
trait list, so the excluded traits remain traceable from metadata alone.

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

#### Scenario: trait_names reflects the post-PCA-filter set
- **WHEN** `PCAAnalysisStep` excludes one or more zero-variance traits
- **THEN** `metadata["trait_names"]` and `metadata["valid_trait_names"]` SHALL
  equal `pca_results["feature_names"]` (the filtered set), not the pre-filter
  `trait_cols`
- **AND** `metadata["original_trait_names"]` SHALL equal the pre-filter
  `trait_cols`, in original order

#### Scenario: trait_names unchanged when nothing is excluded
- **WHEN** all input traits have non-zero variance
- **THEN** `metadata["trait_names"]`, `metadata["valid_trait_names"]`, and
  `metadata["original_trait_names"]` SHALL all be equal to the input
  `trait_cols`

#### Scenario: UMAP inherits the corrected trait set via direct metadata spread
- **WHEN** `UMAPAnalysisStep` executes with `prev_result` from
  `PCAAnalysisStep`
- **THEN** the trait columns used for `feature_cols` in
  `perform_umap_analysis()`, and the `n_traits` value logged to
  `umap_parameters.json`, SHALL reflect the PCA-filtered trait count, not the
  pre-filter count

## ADDED Requirements

### Requirement: Static Figures Trait Metadata Merge
The `VizPipeline` orchestrator's `_run_generate_static_figures` task SHALL
merge the PCA-corrected `trait_names` and `original_trait_names` from the
`03_pca_analysis` branch into `GenerateStaticFiguresStep`'s combined
metadata, in addition to the `pca_results`/`top_features`/
`n_pca_components`/`pca_explained_variance` keys it already merges from that
branch. `GenerateStaticFiguresStep`'s primary metadata source
(`08_genotype_aggregation`) is on a separate DAG branch
(`02_calculate_statistics` → `06_heritability_analysis` →
`08_genotype_aggregation`) that never passes through PCA, so without this
merge the step would continue to see the pre-PCA trait list regardless of
what `PCAAnalysisStep` writes to its own metadata.

#### Scenario: Static figures use the PCA-filtered trait set
- **WHEN** `PCAAnalysisStep` has excluded one or more zero-variance traits
  and `GenerateStaticFiguresStep` subsequently runs
- **THEN** `GenerateStaticFiguresStep`'s `trait_cols` (read from
  `metadata.get("trait_names", ...)`) SHALL equal the PCA-filtered set
- **AND** trait distribution plots (histograms, boxplots) SHALL NOT be
  generated for excluded zero-variance traits
- **AND** `create_pca_biplot` SHALL receive a `trait_names` list whose length
  and order match `pca_results["loadings"].shape[0]`, so biplot feature
  arrows are not silently mislabeled when an excluded trait is not the last
  column of the original trait list

#### Scenario: Static figures keep the pre-PCA trait list when the PCA task result is absent
- **WHEN** `_run_generate_static_figures` is invoked with no
  `"03_pca_analysis"` entry in its task-result kwargs (there is no
  `config.pca.enabled` flag in this codebase — PCA always executes when
  scheduled; this models the DAG executor omitting a task result, e.g. after
  an upstream task failure)
- **THEN** `GenerateStaticFiguresStep`'s combined metadata SHALL keep the
  `trait_names` value relayed from `08_genotype_aggregation`, unmodified —
  the PCA-branch merge (guarded by `if pca_task_result:`) SHALL NOT overwrite
  it with a missing/empty value
