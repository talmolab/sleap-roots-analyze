## ADDED Requirements

### Requirement: Trait Redundancy Reduction Configuration

The system SHALL provide configuration options for trait redundancy reduction in `CrossPlatformConfig` with the following parameters:

- `trait_reduction_method`: Method for reducing trait redundancy before correlation analysis
  - `"none"` (default): No reduction, all traits are used
  - `"clustering"`: Hierarchical clustering of correlated traits
- `trait_clustering_threshold`: Minimum |r| for traits to be considered redundant (default: 0.8)
  - Must be in range (0, 1]
  - Higher values = more stringent = more clusters = fewer traits removed
- `trait_clustering_linkage`: Linkage method for hierarchical clustering
  - `"complete"` (default): Maximum distance between all pairs
  - `"average"`: Average distance between all pairs
  - `"single"`: Minimum distance between any pair

#### Scenario: Default configuration preserves existing behavior

- **WHEN** user creates CrossPlatformConfig without trait reduction parameters
- **THEN** trait_reduction_method defaults to "none"
- **AND** all original traits are used in correlation analysis
- **AND** behavior is identical to previous pipeline versions

#### Scenario: Enable clustering-based reduction

- **WHEN** user sets trait_reduction_method to "clustering" with threshold 0.8
- **THEN** traits with |r| >= 0.8 are grouped into clusters
- **AND** one representative per cluster is selected for correlation analysis
- **AND** original trait count is reduced to cluster count

#### Scenario: Invalid threshold raises error

- **WHEN** user sets trait_clustering_threshold to 0 or negative value
- **THEN** configuration validation fails with error indicating valid range (0, 1]

#### Scenario: Invalid threshold above 1 raises error

- **WHEN** user sets trait_clustering_threshold to 1.5
- **THEN** configuration validation fails with error indicating valid range (0, 1]

#### Scenario: Invalid linkage method raises error

- **WHEN** user sets trait_clustering_linkage to "ward" (unsupported)
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid reduction method raises error

- **WHEN** user sets trait_reduction_method to "invalid_method"
- **THEN** configuration validation fails with error listing valid options: "none", "clustering"

### Requirement: Trait Clustering Algorithm

The system SHALL provide a function `cluster_correlated_traits(df, threshold, linkage)` in `cross_experiment_analysis.py` that clusters traits based on pairwise correlations:

1. **Compute correlation matrix**: Spearman correlation between all trait pairs (genotypes as observations)
2. **Convert to distance**: d = 1 - |r| where r is the correlation coefficient
3. **Apply hierarchical clustering**: Using specified linkage method
4. **Cut dendrogram**: At distance threshold t = 1 - clustering_threshold
5. **Return cluster assignments**: Dict mapping cluster_id -> list of trait names

Edge cases:
- **Single trait**: Returns one cluster containing that trait
- **All NaN trait**: Excluded from clustering, returned as singleton cluster
- **Constant trait (zero variance)**: Excluded from clustering, returned as singleton cluster
- **Empty DataFrame**: Returns empty dict

#### Scenario: Cluster perfectly correlated traits

- **WHEN** df contains 3 traits where traits A and B have r=1.0 and trait C is independent
- **THEN** cluster_correlated_traits returns 2 clusters: {0: ["A", "B"], 1: ["C"]}

#### Scenario: Threshold affects cluster count

- **WHEN** df contains traits with correlations 0.7, 0.8, 0.9 between pairs
- **AND** threshold is 0.85
- **THEN** only pairs with |r| >= 0.85 are grouped
- **AND** pairs with r=0.7 or r=0.8 remain in separate clusters

#### Scenario: Complete linkage requires all pairs to meet threshold

- **WHEN** linkage is "complete" and traits A-B have r=0.9, B-C have r=0.9, but A-C have r=0.5
- **THEN** A, B, C are NOT all in the same cluster
- **AND** complete linkage prevents weak-link clustering

#### Scenario: Deterministic output

- **WHEN** cluster_correlated_traits is called twice with identical inputs
- **THEN** output cluster assignments are identical
- **AND** trait ordering within clusters is deterministic (alphabetical)

#### Scenario: Handle constant trait

- **WHEN** df contains a trait with zero variance (all genotypes have same value)
- **THEN** that trait is placed in its own singleton cluster
- **AND** other traits are clustered normally
- **AND** no error is raised

### Requirement: Cluster Representative Selection

The system SHALL provide a function `select_cluster_representatives(df, clusters)` in `cross_experiment_analysis.py` that selects one representative trait per cluster:

1. **For each cluster**: Compute variance of each trait across genotypes
2. **Select representative**: Trait with highest variance
3. **Tie-breaking**: If variances are equal, select alphabetically first trait
4. **Return**: List of representative trait names (one per cluster)

#### Scenario: Select highest variance representative

- **WHEN** cluster contains traits with variances [0.5, 0.8, 0.3]
- **THEN** trait with variance 0.8 is selected as representative

#### Scenario: Alphabetical tie-breaking

- **WHEN** cluster contains traits "Beta" and "Alpha" with identical variance
- **THEN** "Alpha" is selected as representative (alphabetically first)

#### Scenario: Singleton cluster

- **WHEN** cluster contains only one trait
- **THEN** that trait is returned as representative

#### Scenario: Handle NaN variance

- **WHEN** a trait has NaN variance (all NaN values)
- **THEN** that trait is not selected as representative
- **AND** next highest variance trait is selected

### Requirement: Reduce Trait Redundancy Pipeline Step

The system SHALL provide a pipeline step `ReduceTraitRedundancyStep` that reduces trait redundancy before correlation analysis:

**Inputs**:
- `data`: Dict containing `exp1_df`, `exp2_df`, and trait name lists from LoadCrossPlatformDataStep
- `config`: CrossPlatformConfig with trait reduction settings

**Behavior**:
- When `trait_reduction_method` is `"none"`: Pass through data unchanged
- When `trait_reduction_method` is `"clustering"`:
  1. Cluster exp2 traits (typically the larger set)
  2. Select representative per cluster
  3. Filter exp2_df to representative columns only
  4. Update trait name lists in metadata

**Outputs**:
- `data`: Dict with reduced trait DataFrames and updated trait lists
- `metadata`: Reduction statistics (original count, reduced count, method)
- `files_generated`: Path to `trait_clusters.csv`

#### Scenario: Reduction step produces correct metadata

- **WHEN** ReduceTraitRedundancyStep executes with clustering enabled
- **AND** exp2 has 2048 traits that reduce to 150 clusters
- **THEN** metadata includes `original_exp2_traits: 2048`
- **AND** metadata includes `reduced_exp2_traits: 150`
- **AND** metadata includes `reduction_ratio: 0.927` (92.7% reduction)

#### Scenario: Cluster membership file is traceable

- **WHEN** ReduceTraitRedundancyStep executes with clustering enabled
- **THEN** `trait_clusters.csv` is generated with columns:
  - `trait`: Original trait name
  - `cluster_id`: Integer cluster assignment
  - `is_representative`: Boolean indicating if this trait was selected
  - `variance`: Trait variance used for selection

#### Scenario: Downstream steps receive reduced traits

- **WHEN** ReduceTraitRedundancyStep completes
- **THEN** `exp2_trait_names` in output metadata contains only representatives
- **AND** CalculateCrossPlatformCorrelationsStep uses reduced trait list
- **AND** total correlations = exp1_traits × reduced_exp2_traits

#### Scenario: Method none is a no-op

- **WHEN** trait_reduction_method is "none"
- **THEN** step returns data unchanged
- **AND** metadata indicates `reduction_ratio: 0.0`
- **AND** no trait_clusters.csv is generated

### Requirement: Trait Reduction Integration with Correlation Step

The system SHALL integrate trait reduction with the correlation calculation step such that:

- Correlation CSV contains only traits that survived reduction
- Metadata reports both original and reduced trait counts
- Cluster membership file enables tracing any correlation back to original traits

#### Scenario: Correlation output reflects reduced traits

- **WHEN** exp2 is reduced from 2048 to 150 traits
- **AND** exp1 has 8 traits
- **THEN** correlation CSV contains 8 × 150 = 1200 rows
- **AND** all exp2_trait values in CSV are cluster representatives

#### Scenario: Traceability from correlation to original traits

- **WHEN** user finds significant correlation with exp2 trait "SeminalLength_Mean"
- **THEN** user can look up "SeminalLength_Mean" in trait_clusters.csv
- **AND** find all original traits in same cluster (e.g., "SeminalLength_Median", "SeminalLength_Max")
- **AND** understand which redundant traits were represented

## MODIFIED Requirements

### Requirement: Cross-Platform Configuration

The system SHALL provide configuration options for cross-platform trait correlation analysis through the `CrossPlatformConfig` dataclass with the following required parameters:

- `exp1_data_path`: Path to experiment 1 cleaned traits CSV
- `exp1_name`: Display name for experiment 1 (e.g., "Cylinder")
- `exp1_genotype_col`: Column name containing genotype identifiers in experiment 1
- `exp2_data_path`: Path to experiment 2 cleaned traits CSV
- `exp2_name`: Display name for experiment 2 (e.g., "Turface")
- `exp2_genotype_col`: Column name containing genotype identifiers in experiment 2

And the following optional parameters with defaults:

- `correlation_method`: Statistical method ("spearman", "pearson", "kendall"), default "spearman"
- `min_samples_per_genotype`: Minimum samples required per genotype, default 3
- `significance_level`: P-value threshold for significance, default 0.05
- `top_n_correlations`: Number of top correlations to display in summary, default 20
- `top_n_joint_plots`: Number of joint plots to generate, default 6
- `top_n_boxplots`: Number of boxplots to generate, default 6
- `figsize_summary`: Summary figure size tuple, default (14, 12)
- `figsize_joint`: Joint plot figure size tuple, default (10, 10)
- `figsize_boxplot`: Boxplot figure size tuple, default (14, 6)
- `exp1_exclude_cols`: List of column names to exclude from experiment 1 trait analysis, default None
- `exp2_exclude_cols`: List of column names to exclude from experiment 2 trait analysis, default None
- `fdr_correction_method`: Method for multiple testing correction ("fdr_bh", "fdr_by", "none"), default "fdr_by"
- `confidence_level`: Confidence level for correlation coefficient intervals, default 0.95
- `min_genotypes_for_correlation`: Minimum number of valid genotypes required for a trait pair correlation, default 10. Trait pairs with fewer valid genotypes after NaN removal are excluded from output.
- `power_analysis_alpha`: Significance level (α) for power analysis, default 0.05. Used to calculate minimum detectable effect size and achieved power.
- `power_analysis_power`: Target power (1-β) for minimum detectable effect size calculation, default 0.80. Standard convention is 80% power.
- `trait_reduction_method`: Method for reducing trait redundancy ("none", "clustering"), default "none"
- `trait_clustering_threshold`: Minimum |r| for traits to be considered redundant, default 0.8
- `trait_clustering_linkage`: Linkage method for hierarchical clustering ("complete", "average", "single"), default "complete"

#### Scenario: Valid configuration with required fields

- **WHEN** user provides valid paths and column names for both experiments
- **THEN** configuration object is created successfully with default optional parameters

#### Scenario: Missing required fields

- **WHEN** user provides configuration missing required fields (data paths or genotype columns)
- **THEN** configuration validation fails with clear error message indicating missing fields

#### Scenario: Invalid correlation method

- **WHEN** user specifies correlation method not in ["spearman", "pearson", "kendall"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid FDR correction method

- **WHEN** user specifies fdr_correction_method not in ["fdr_bh", "fdr_by", "none"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid confidence level

- **WHEN** user specifies confidence_level outside (0, 1) exclusive range
- **THEN** configuration validation fails with error indicating valid range

#### Scenario: Custom confidence level

- **WHEN** user specifies confidence_level as 0.99
- **THEN** 99% confidence intervals are computed for all correlations
- **AND** intervals are wider than default 95% intervals

#### Scenario: Invalid trait reduction method

- **WHEN** user specifies trait_reduction_method not in ["none", "clustering"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid clustering threshold

- **WHEN** user specifies trait_clustering_threshold outside (0, 1] range
- **THEN** configuration validation fails with error indicating valid range
