## ADDED Requirements

### Requirement: Cross-Platform Summary Data Structures

The system SHALL provide data structures for aggregating and reporting cross-platform analysis statistics.

#### Scenario: TraitReductionStats captures clustering results

- **GIVEN** a cross-platform run with `trait_reduction_method: clustering`
- **WHEN** `TraitReductionStats` is created from `trait_clusters.csv`
- **THEN** it SHALL contain:
  - `original_count`: Total traits before clustering
  - `cluster_count`: Number of clusters formed
  - `representative_count`: Number of representative traits selected
  - `reduction_percentage`: Calculated as `(1 - representative_count / original_count) * 100`

#### Scenario: CorrelationStats captures correlation analysis results

- **GIVEN** a `cross_platform_correlations.csv` file
- **WHEN** `CorrelationStats` is created
- **THEN** it SHALL contain:
  - `total`: Row count in correlations CSV
  - `nominal_significant`: Count where `spearman_p < 0.05`
  - `fdr_significant`: Count where `significant_fdr == True`
  - `max_abs_r`: Maximum absolute value of `spearman_r`
  - `top_correlations`: List of `TopCorrelation` objects (top N by |r|)

#### Scenario: TopCorrelation captures individual correlation details

- **WHEN** a `TopCorrelation` object is created
- **THEN** it SHALL contain:
  - `exp1_trait`: Trait name from experiment 1
  - `exp2_trait`: Trait name from experiment 2
  - `r`: Spearman correlation coefficient (signed)
  - `p`: Raw p-value
  - `q`: FDR-adjusted p-value
  - `power`: Achieved statistical power
  - `n`: Number of genotypes used in correlation

#### Scenario: PowerStats captures power analysis summary

- **GIVEN** a correlations CSV with `achieved_power` column
- **WHEN** `PowerStats` is created
- **THEN** it SHALL contain:
  - `count_above_80`: Count where `achieved_power >= 0.80`
  - `percentage_above_80`: `(count_above_80 / total) * 100`
  - `median_power`: Median of all `achieved_power` values
  - `min_power`: Minimum `achieved_power`
  - `max_power`: Maximum `achieved_power`

#### Scenario: ValidationResult captures guardrail outcomes

- **WHEN** validation guardrails are applied
- **THEN** `ValidationResult` SHALL contain:
  - `passed`: Boolean indicating all checks passed
  - `errors`: List of error messages for failed checks
  - `warnings`: List of warning messages for non-critical issues

### Requirement: Explicit Trait Clustering Configuration

The system SHALL require explicit configuration of which experiment(s) to cluster. There are no implicit defaults.

#### Scenario: Clustering target explicitly configured

- **GIVEN** a cross-platform config with `trait_reduction_method: clustering`
- **THEN** the config MUST specify `trait_reduction_target` with one of:
  - `exp1`: Cluster only exp1 traits
  - `exp2`: Cluster only exp2 traits
  - `both`: Cluster both exp1 and exp2 traits independently
- **AND** validation SHALL fail if `trait_reduction_target` is missing when clustering is enabled

#### Scenario: Config validation rejects ambiguous clustering

- **GIVEN** a cross-platform config with `trait_reduction_method: clustering`
- **AND** no `trait_reduction_target` specified
- **WHEN** config validation runs
- **THEN** validation SHALL fail with error: "trait_reduction_target must be specified when trait_reduction_method is 'clustering'"

### Requirement: Trait Clustering Visualizations

The system SHALL generate visualizations for trait clustering as part of the `ReduceTraitRedundancyStep` pipeline step. All visualizations are tested Python code following project conventions.

#### Scenario: Exp1 dendrogram generated when exp1 clustered

- **GIVEN** `trait_reduction_target` includes `exp1` (i.e., `exp1` or `both`)
- **WHEN** `ReduceTraitRedundancyStep` executes
- **THEN** it SHALL generate `exp1_trait_clustering_dendrogram.png`
- **AND** the dendrogram SHALL show hierarchical clustering of exp1 traits
- **AND** the threshold cutoff line SHALL be displayed at `trait_clustering_threshold`
- **AND** cluster assignments SHALL be color-coded

#### Scenario: Exp2 dendrogram generated when exp2 clustered

- **GIVEN** `trait_reduction_target` includes `exp2` (i.e., `exp2` or `both`)
- **WHEN** `ReduceTraitRedundancyStep` executes
- **THEN** it SHALL generate `exp2_trait_clustering_dendrogram.png`
- **AND** the dendrogram SHALL show hierarchical clustering of exp2 traits
- **AND** the threshold cutoff line SHALL be displayed at `trait_clustering_threshold`
- **AND** cluster assignments SHALL be color-coded

#### Scenario: Exp1 correlation heatmap generated when exp1 clustered

- **GIVEN** `trait_reduction_target` includes `exp1`
- **WHEN** `ReduceTraitRedundancyStep` executes
- **THEN** it SHALL generate `exp1_trait_cluster_heatmap.png`
- **AND** the heatmap SHALL show within-exp1 trait correlation matrix
- **AND** traits SHALL be ordered by cluster membership
- **AND** cluster boundaries SHALL be visually indicated
- **AND** representative traits SHALL be highlighted

#### Scenario: Exp2 correlation heatmap generated when exp2 clustered

- **GIVEN** `trait_reduction_target` includes `exp2`
- **WHEN** `ReduceTraitRedundancyStep` executes
- **THEN** it SHALL generate `exp2_trait_cluster_heatmap.png`
- **AND** the heatmap SHALL show within-exp2 trait correlation matrix
- **AND** traits SHALL be ordered by cluster membership
- **AND** cluster boundaries SHALL be visually indicated
- **AND** representative traits SHALL be highlighted

#### Scenario: Cross-platform representative heatmap generated

- **GIVEN** trait clustering is enabled for at least one experiment
- **WHEN** `VisualizeCrossPlatformStep` executes
- **THEN** it SHALL generate `cross_platform_representative_heatmap.png`
- **AND** rows SHALL be exp1 traits (all traits if not clustered, representatives if clustered)
- **AND** columns SHALL be exp2 traits (all traits if not clustered, representatives if clustered)
- **AND** significant correlations SHALL be annotated
- **AND** color scale SHALL indicate correlation strength and direction

#### Scenario: No clustering visualizations when clustering disabled

- **GIVEN** `trait_reduction_method: none`
- **WHEN** pipeline executes
- **THEN** no clustering dendrograms or heatmaps SHALL be generated
- **AND** no errors SHALL occur

#### Scenario: Visualization metadata captured for reproducibility

- **WHEN** clustering visualizations are generated
- **THEN** metadata JSON SHALL include:
  - `trait_reduction_target`: Which experiment(s) were clustered
  - `trait_clustering_threshold`: The threshold value used
  - `trait_clustering_linkage`: The linkage method used
  - `exp1_n_clusters`: Number of clusters in exp1 (if clustered)
  - `exp2_n_clusters`: Number of clusters in exp2 (if clustered)
  - `exp1_n_representatives`: Number of exp1 representatives (if clustered)
  - `exp2_n_representatives`: Number of exp2 representatives (if clustered)

### Requirement: Cross-Platform Summary Generator

The system SHALL provide a `CrossPlatformSummaryGenerator` class for creating detailed analysis reports.

#### Scenario: Generator discovers cross-platform runs

- **GIVEN** a pipeline run directory with cross-platform outputs
- **WHEN** `_find_cross_platform_runs()` is called
- **THEN** it SHALL return a list of paths to all cross-platform output directories
- **AND** each path SHALL contain the expected output files

#### Scenario: Generator reads trait clusters

- **GIVEN** a cross-platform run with `trait_clusters.csv`
- **WHEN** `_read_trait_clusters(run_dir)` is called
- **THEN** it SHALL return a `TraitReductionStats` object
- **AND** values SHALL match the CSV exactly

#### Scenario: Generator reads correlations

- **GIVEN** a cross-platform run with `cross_platform_correlations.csv`
- **WHEN** `_read_correlations(run_dir)` is called
- **THEN** it SHALL return a `CorrelationStats` object
- **AND** all counts and statistics SHALL match the CSV exactly

#### Scenario: Generator handles missing trait clusters

- **GIVEN** a cross-platform run with `trait_reduction_method: none`
- **WHEN** `_read_trait_clusters(run_dir)` is called
- **THEN** it SHALL return `None` or a stats object with `method: "none"`
- **AND** no error SHALL be raised

#### Scenario: Generator handles missing correlations file

- **GIVEN** a cross-platform output directory without `cross_platform_correlations.csv`
- **WHEN** `_read_correlations(run_dir)` is called
- **THEN** it SHALL return `None`
- **AND** a warning SHALL be logged
- **AND** no error SHALL be raised

### Requirement: Cross-Platform Summary Markdown Rendering

The system SHALL render cross-platform summaries as well-formatted markdown.

#### Scenario: Overview table rendered correctly

- **WHEN** `to_markdown()` is called on a summary with multiple comparisons
- **THEN** the output SHALL include a comparison overview table
- **AND** columns SHALL match the spec: Comparison, Genotypes, Trait Reduction, Correlations, Nominal Sig, FDR Sig, Top |r|, Power ≥80%

#### Scenario: Top correlations table rendered correctly

- **WHEN** `to_markdown()` is called
- **THEN** each comparison SHALL have a "Top Correlations" subsection
- **AND** the table SHALL show Rank, Exp1 Trait, Exp2 Trait, r, p, q, Power, n
- **AND** correlations SHALL be ordered by |r| descending

#### Scenario: Metadata table rendered correctly

- **WHEN** `to_markdown()` is called
- **THEN** each comparison SHALL have a "Metadata" subsection
- **AND** it SHALL include FDR correction method, trait reduction parameters, significance level

#### Scenario: Validation warnings included in output

- **WHEN** `to_markdown()` is called and validation found warnings
- **THEN** the output SHALL include a "Validation Warnings" section
- **AND** each warning SHALL be listed clearly for user review
