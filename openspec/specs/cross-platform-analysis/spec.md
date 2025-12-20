# cross-platform-analysis Specification

## Purpose
TBD - created by archiving change add-cross-platform-pipeline. Update Purpose after archive.
## Requirements
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

#### Scenario: Valid configuration with required fields

- **WHEN** user provides valid paths and column names for both experiments
- **THEN** configuration object is created successfully with default optional parameters

#### Scenario: Missing required fields

- **WHEN** user provides configuration missing required fields (data paths or genotype columns)
- **THEN** configuration validation fails with clear error message indicating missing fields

#### Scenario: Invalid correlation method

- **WHEN** user specifies correlation method not in ["spearman", "pearson", "kendall"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Exclude metadata columns from experiment 1

- **WHEN** user specifies exp1_exclude_cols with metadata column names like ["Ent", "Sub", "Cid"]
- **THEN** those columns are excluded from experiment 1 trait analysis
- **AND** they do not appear in correlation results

#### Scenario: Exclude metadata columns from experiment 2

- **WHEN** user specifies exp2_exclude_cols with metadata column names like ["File.me", "scanner"]
- **THEN** those columns are excluded from experiment 2 trait analysis
- **AND** they do not appear in correlation results

#### Scenario: Different exclusion lists per experiment

- **WHEN** exp1 has field metadata columns and exp2 has imaging metadata columns
- **THEN** each experiment's exclusion list is applied independently
- **AND** correlations only include biological trait columns

### Requirement: Load and Align Cross-Platform Data

The system SHALL load and align data from two experimental platforms with the following **additional** column validation behavior:

- **Replicate column detection**: When searching for replicate columns, if multiple variants are found (e.g., both "Replicate" and "rep"), the system SHALL issue a UserWarning indicating which column will be used.
- **Default argument safety**: Functions with default list parameters (e.g., `calculate_genotype_statistics`) SHALL use `None` as default with runtime initialization to prevent mutable default argument bugs.

#### Scenario: Multiple replicate column variants

- **WHEN** a DataFrame has both "Replicate" and "rep" columns
- **THEN** load_and_align_experiments issues a UserWarning indicating which column will be used
- **AND** the first matching column variant is used consistently

#### Scenario: Mutable default argument protection

- **WHEN** calculate_genotype_statistics is called without statistics parameter
- **THEN** the default statistics list is created fresh for each call
- **AND** mutations to the returned statistics do not affect future calls

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- Compute **both Pearson and Spearman** correlations for all trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `pearson_r`, `pearson_p`, `n_genotypes`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory
- The `correlation_method` config determines:
  - Which metric is used for sorting/ranking
  - Which metric is used for significance filtering
  - Which metric is considered "primary" in visualizations

#### Scenario: Dual correlation calculation with Spearman primary

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates BOTH Spearman and Pearson correlations for all 50 pairs
- **AND** exports CSV with columns: exp1_trait, exp2_trait, spearman_r, spearman_p, pearson_r, pearson_p, n_genotypes
- **AND** sorts results by absolute Spearman correlation (descending)

#### Scenario: Dual correlation calculation with Pearson primary

- **WHEN** correlation method is "pearson" with normally distributed trait data
- **THEN** step calculates BOTH Spearman and Pearson correlations
- **AND** sorts results by absolute Pearson correlation (descending)

#### Scenario: Handling missing data in correlations

- **WHEN** trait pairs have NaN values for some genotypes
- **THEN** step removes NaN pairs before calculation
- **AND** reports actual n_genotypes used per correlation
- **AND** both Pearson and Spearman use the same filtered data

#### Scenario: Insufficient valid pairs

- **WHEN** after removing NaN pairs, fewer than 3 valid genotype pairs remain
- **THEN** step sets all correlation values to NaN for that trait pair
- **AND** logs warning about insufficient data

### Requirement: Visualize Cross-Platform Correlations

The system SHALL generate publication-quality visualizations through `VisualizeCrossPlatformStep` with the following outputs:

- **Summary visualization** (4-panel figure):
  - Panel 1: Histogram of correlation distribution with significance counts (uses primary method)
  - Panel 2: Volcano plot (correlation vs -log10(p-value)) with significance thresholds (uses primary method)
  - Panel 3: Horizontal bar chart of top positive correlations (uses primary method)
  - Panel 4: Horizontal bar chart of top negative correlations (uses primary method)
- **Joint plots**: Scatter plots with marginal distributions for top N correlated trait pairs
  - Display **both** Pearson and Spearman annotations using pre-computed values from CSV
  - Values MUST match CSV exactly (single source of truth)
- **Genotype boxplots**: Side-by-side boxplots comparing genotype distributions for top N trait pairs
- All figures saved to `figures/` subdirectory in output directory
- Figure format and DPI configurable through pipeline settings

#### Scenario: Joint plots display pre-computed values

- **WHEN** joint plots are generated for top correlations
- **THEN** both Pearson r and Spearman ρ annotations use pre-computed values from CSV
- **AND** values displayed match `cross_platform_correlations.csv` exactly
- **AND** no correlation recalculation occurs during visualization

#### Scenario: Generate summary visualization

- **WHEN** correlation results contain 7,056 trait pairs with max |ρ| = 0.389
- **THEN** step generates 4-panel summary using the primary correlation method
- **AND** significance filtering uses the primary method's p-values

#### Scenario: Generate joint plots for top correlations

- **WHEN** top_n_joint_plots is 6 and correlation results contain sufficient data
- **THEN** step generates 6 joint plots for trait pairs with highest absolute primary correlations
- **AND** each plot displays both "Pearson r = X.XXX (p = X.XXX)" and "Spearman ρ = X.XXX (p = X.XXX)"

### Requirement: Cross-Platform Pipeline Integration

The system SHALL integrate cross-platform analysis steps into the existing pipeline infrastructure with the following **additional** error handling:

- **Log directory failures**: When the CLI attempts to create a log directory and encounters an OSError, it SHALL catch the error, display a user-friendly warning, and continue with console-only logging.

#### Scenario: Log directory creation failure

- **WHEN** an invalid or inaccessible log file path is configured
- **THEN** an OSError is caught during directory creation
- **AND** a warning message is displayed to the user
- **AND** the pipeline continues with console-only logging

### Requirement: Statistical Method Flexibility

The system SHALL support multiple correlation methods through unified interface in `cross_experiment_analysis.py` with the following function:

- `calculate_correlations(x, y, method)`: Computes correlation and p-value using specified method
- Supports "spearman" (Spearman rank correlation), "pearson" (Pearson linear correlation), and "kendall" (Kendall tau correlation)
- Returns tuple of (correlation_coefficient, p_value)
- Handles edge cases (all identical values, insufficient data)

#### Scenario: Switch correlation methods without code changes

- **WHEN** user changes configuration from "spearman" to "pearson"
- **THEN** CalculateCrossPlatformCorrelationsStep automatically uses Pearson correlation without any code modifications

#### Scenario: Spearman for non-linear monotonic relationships

- **WHEN** traits have monotonic but non-linear relationships
- **THEN** Spearman correlation detects relationship strength regardless of linearity

#### Scenario: Pearson for linear relationships

- **WHEN** traits have linear relationships with normal distributions
- **THEN** Pearson correlation provides most powerful test for linear association

#### Scenario: Kendall for small sample sizes

- **WHEN** only 10 genotypes available for correlation
- **THEN** Kendall tau provides more robust estimates than Spearman or Pearson for small n

### Requirement: Reproducible Output Structure

The system SHALL generate reproducible outputs with consistent directory structure:

```
<output_base_dir>/
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, pearson_r, pearson_p, n_genotypes
├── summary.json                         # Analysis metadata and summary statistics
├── figures/
│   ├── correlation_summary.png          # 4-panel summary visualization
│   ├── joint_01_trait1_vs_trait2.png   # Top correlation joint plots
│   ├── joint_02_trait1_vs_trait2.png
│   └── ...
│   ├── boxplot_01.png                   # Genotype distribution comparisons
│   ├── boxplot_02.png
│   └── ...
└── pipeline.log                         # Execution log
```

#### Scenario: CSV schema includes both correlation methods

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains columns for both Spearman and Pearson
- **AND** column names are explicit: spearman_r, spearman_p, pearson_r, pearson_p

#### Scenario: Consistent output structure across runs

- **WHEN** user runs cross-platform pipeline multiple times with different configs
- **THEN** each run creates timestamped directory with identical internal structure for easy comparison

### Requirement: Template Configuration Example

The system SHALL provide template configuration file `configs/cross_platform_template.yaml` demonstrating:

- Complete required and optional parameters with explanatory comments
- Example paths showing expected CSV file structure
- Different correlation method options documented inline
- Recommended parameter values based on typical use cases
- References to relevant notebook examples

#### Scenario: User copies template for new analysis

- **WHEN** user copies `cross_platform_template.yaml` and updates only data paths
- **THEN** configuration is valid and pipeline runs successfully with sensible defaults

#### Scenario: Template documents all options

- **WHEN** user opens template configuration file
- **THEN** all CrossPlatformConfig parameters are present with inline comments explaining purpose and valid values

### Requirement: Single Source of Truth for Correlation Statistics

The system SHALL ensure correlation statistics (correlation coefficient, p-value, n_genotypes) are computed once in `CalculateCrossPlatformCorrelationsStep` and reused by all downstream visualizations without recalculation.

Visualization functions (`create_joint_plot`, `create_scatter_plot_grid`) SHALL accept optional pre-computed correlation parameters:
- `correlation`: Pre-computed correlation coefficient
- `p_value`: Pre-computed p-value
- `n_genotypes`: Pre-computed number of genotypes used in calculation

When these parameters are provided, the function SHALL display them directly without recalculation. When not provided (for backward compatibility), the function MAY calculate values from the provided data, but this fallback behavior is deprecated for pipeline use.

#### Scenario: Pipeline passes pre-computed values to joint plot

- **WHEN** `VisualizeCrossPlatformStep` creates a joint plot for a trait pair
- **THEN** it SHALL pass `correlation`, `p_value`, and `n_genotypes` from `correlation_df` to `create_joint_plot`
- **AND** the displayed annotation SHALL show these exact values

#### Scenario: Direct API usage with fallback calculation

- **WHEN** `create_joint_plot` is called directly without pre-computed correlation parameters
- **THEN** it SHALL calculate correlations from the provided genotype means (backward compatible)
- **AND** this fallback behavior is intended only for standalone usage outside the pipeline

#### Scenario: Consistency verification test

- **WHEN** the test suite runs
- **THEN** there SHALL be a test that verifies correlation values in generated joint plots match the corresponding CSV row exactly
- **AND** the test SHALL use a scenario where `min_samples_per_genotype` would cause a discrepancy if values were recalculated

