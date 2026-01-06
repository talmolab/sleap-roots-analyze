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
- `fdr_correction_method`: Method for multiple testing correction ("fdr_bh", "fdr_by", "none"), default "fdr_by"
- `confidence_level`: Confidence level for correlation coefficient intervals, default 0.95

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
- **Calculate confidence intervals** for each correlation coefficient using Fisher z-transformation
- Apply FDR correction to p-values based on `fdr_correction_method` config:
  - `fdr_bh`: Benjamini-Hochberg correction (assumes test independence)
  - `fdr_by`: Benjamini-Yekutieli correction (valid under arbitrary dependence)
  - `none`: No correction applied (adjusted p-values equal raw p-values)
- **Handle NaN p-values gracefully**: If any correlation produces NaN p-values (e.g., from constant-valued traits or insufficient data), the FDR correction SHALL:
  - Filter out NaN p-values before applying correction
  - Apply correction only to valid p-values
  - Preserve NaN in adjusted p-value columns for invalid correlations
  - Set `significant_fdr` to False for rows with NaN adjusted p-values
  - Log a warning indicating the count of NaN p-values encountered
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r`, `pearson_p`, `pearson_r_ci_low`, `pearson_r_ci_high`, `n_genotypes`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory

#### Scenario: Confidence intervals computed for all correlations

- **WHEN** correlation step executes with 50 trait pairs and n=20 genotypes
- **THEN** CSV output contains `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high` columns
- **AND** all CI bounds satisfy: -1 ≤ ci_low ≤ r ≤ ci_high ≤ 1
- **AND** CI width is inversely related to n_genotypes

#### Scenario: Confidence interval for perfect correlation

- **WHEN** two traits have r = 1.0 (perfect positive correlation)
- **THEN** confidence interval is (1.0, 1.0) as a point mass at the boundary
- **AND** no mathematical error occurs from arctanh(1.0)

#### Scenario: Confidence interval with small sample size

- **WHEN** n_genotypes < 4 for a trait pair
- **THEN** CI bounds are NaN (undefined variance with n-3 in denominator)
- **AND** correlation coefficient r is still computed if n >= 3

#### Scenario: Higher confidence level produces wider intervals

- **WHEN** confidence_level is 0.99 instead of 0.95
- **THEN** all confidence intervals are wider
- **AND** width difference follows z_{0.005}/z_{0.025} ratio ≈ 1.32

#### Scenario: NaN p-values from constant trait do not corrupt FDR correction

- **WHEN** one trait pair produces NaN p-values (e.g., constant values in one trait)
- **AND** other trait pairs have valid p-values
- **THEN** FDR correction is applied only to valid p-values
- **AND** NaN p-values remain NaN in the adjusted columns
- **AND** `significant_fdr` is False for rows with NaN adjusted p-values
- **AND** valid correlations receive correct FDR-adjusted p-values

#### Scenario: Single correlation (m=1)

- **WHEN** only one trait pair is tested (m=1)
- **THEN** no FDR correction is applied (single test, no multiple testing)
- **AND** adjusted p-values equal raw p-values
- **AND** `significant_fdr` is based on raw p-value comparison

#### Scenario: Fewer than 3 genotypes for a trait pair

- **WHEN** after removing NaN pairs, fewer than 3 valid genotype pairs remain
- **THEN** step sets all correlation values to NaN for that trait pair
- **AND** adjusted p-values are NaN for that trait pair
- **AND** confidence intervals are NaN for that trait pair
- **AND** `significant_fdr` is False for that trait pair
- **AND** logs warning about insufficient data

#### Scenario: Dual correlation calculation with Spearman primary

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates BOTH Spearman and Pearson correlations for all 50 pairs
- **AND** exports CSV with columns including CI bounds for both methods
- **AND** sorts results by absolute Spearman correlation (descending)

#### Scenario: FDR correction with Benjamini-Hochberg

- **WHEN** fdr_correction_method is "fdr_bh" and 1000 trait pairs are tested
- **THEN** spearman_p_adjusted and pearson_p_adjusted contain BH-corrected p-values
- **AND** adjusted p-values are >= raw p-values
- **AND** significant_fdr is True when primary adjusted p < significance_level

#### Scenario: FDR correction with Benjamini-Yekutieli (default)

- **WHEN** fdr_correction_method is "fdr_by" (default) and traits are correlated
- **THEN** BY correction is applied (valid under arbitrary dependence)
- **AND** BY produces more conservative (larger) adjusted p-values than BH

#### Scenario: No FDR correction

- **WHEN** fdr_correction_method is "none"
- **THEN** spearman_p_adjusted equals spearman_p
- **AND** pearson_p_adjusted equals pearson_p
- **AND** significant_fdr uses raw p-values for threshold comparison

### Requirement: Visualize Cross-Platform Correlations

The system SHALL generate publication-quality visualizations through `VisualizeCrossPlatformStep` with the following outputs:

- **Summary visualization** (4-panel figure):
  - Panel 1: Histogram of correlation distribution with FDR-corrected significance count annotation (uses primary method)
  - Panel 2: Volcano plot (correlation vs -log10(p-value)) with significance thresholds using raw p-values (uses primary method)
  - Panel 3: Horizontal bar chart of top positive correlations (uses primary method)
  - Panel 4: Horizontal bar chart of top negative correlations (uses primary method)
- **Joint plots**: Scatter plots with marginal distributions for top N correlated trait pairs
  - Display **both** Pearson and Spearman annotations using pre-computed values from CSV
  - Values MUST match CSV exactly (single source of truth)
- **Genotype boxplots**: Side-by-side boxplots comparing genotype distributions for top N trait pairs
- All figures saved to `figures/` subdirectory in output directory
- Figure format and DPI configurable through pipeline settings

#### Scenario: Summary plot shows FDR-corrected significance count

- **WHEN** summary visualization is generated with FDR correction enabled
- **THEN** Panel 1 histogram annotation shows "Significant (FDR): N"
- **AND** N is the count of correlations where significant_fdr is True
- **AND** volcano plot (Panel 2) continues to use raw p-values for axis and coloring

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
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman + FDR + CI)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, spearman_r_ci_low, spearman_r_ci_high,
│            pearson_r, pearson_p, pearson_r_ci_low, pearson_r_ci_high, n_genotypes,
│            spearman_p_adjusted, pearson_p_adjusted, significant_fdr
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

#### Scenario: CSV schema includes confidence interval columns

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains CI columns: `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high`
- **AND** column order places CI columns adjacent to their corresponding r values

#### Scenario: CSV schema includes FDR correction columns

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains columns for both raw and adjusted p-values
- **AND** column names include: spearman_p_adjusted, pearson_p_adjusted, significant_fdr

#### Scenario: Metadata includes confidence level

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `confidence_level` parameter value used

#### Scenario: Metadata includes FDR correction information

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes fdr_correction_method and significant_correlations count

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

### Requirement: FDR Correction Documentation

The system SHALL provide comprehensive documentation of FDR (False Discovery Rate) correction methods in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Mathematical formulation of the Benjamini-Hochberg (BH) procedure including:
  - Ordered p-value notation p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
  - Critical value formula: α_i = (i / m) × α
  - Decision rule for rejecting hypotheses
  - Adjusted p-value formula
- Mathematical formulation of the Benjamini-Yekutieli (BY) procedure including:
  - The additional correction factor c(m) = Σ(1/i) for i = 1 to m
  - How c(m) grows logarithmically with the number of tests
  - Example correction factors for common test counts (100, 1000, 10000, 100000)
- Explanation of when adjusted p-values are capped at 1.0
- Guidance on when to use each method (BH vs BY vs none)
- Explanation of why BY often yields no significant results with:
  - Small sample sizes (< 20 genotypes)
  - Large numbers of tests (> 10,000)
  - Example calculation showing how minimum p-values become 1.0 after BY correction
- Practical recommendations for improving statistical power
- Output file documentation including CSV column descriptions
- Academic references to the original BH (1995) and BY (2001) papers

#### Scenario: User understands BH procedure mathematically

- **WHEN** user reads the BH procedure section
- **THEN** they can understand the step-by-step algorithm
- **AND** they can calculate adjusted p-values manually for small examples
- **AND** they understand when BH is appropriate (independent or positively correlated tests)

#### Scenario: User understands BY procedure mathematically

- **WHEN** user reads the BY procedure section
- **THEN** they understand the additional conservatism from c(m)
- **AND** they can look up approximate correction factors for their number of tests
- **AND** they understand BY is valid under arbitrary dependence

#### Scenario: User understands why no results are significant

- **WHEN** user runs analysis with BY correction and gets zero significant results
- **THEN** documentation explains this is expected behavior
- **AND** provides worked example showing why minimum p-values become 1.0
- **AND** suggests actionable steps (increase sample size, reduce tests, use BH for exploration)

#### Scenario: User can interpret output files

- **WHEN** user examines cross_platform_correlations.csv
- **THEN** documentation explains each column's meaning
- **AND** clarifies difference between raw and adjusted p-values
- **AND** explains the significant_fdr boolean column

### Requirement: Pipeline Summary Integration

The system SHALL include FDR correction metadata in pipeline summaries with the following behavior:

- Pipeline summary JSON SHALL include StepResult metadata merged with TaskResult metadata
- The `pipeline_summary.json` for each cross-platform run SHALL include:
  - `fdr_correction_method`: The correction method used
  - `significant_correlations`: Count of correlations passing FDR threshold
  - `total_correlations`: Total number of correlations computed
- The run-all `SUMMARY.md` SHALL:
  - Display top correlation values using `spearman_r` column from new CSV schema
  - Reference FDR correction in the Methods section (not Bonferroni)

#### Scenario: Pipeline summary JSON includes FDR metadata

- **WHEN** cross-platform pipeline completes successfully
- **THEN** `pipeline_summary.json` step metadata includes `fdr_correction_method`
- **AND** step metadata includes `significant_correlations` count
- **AND** step metadata includes `total_correlations` count

#### Scenario: Run-all SUMMARY.md shows correct top correlations

- **WHEN** user runs `sleap-roots-analyze run-all` with cross-platform configs
- **THEN** SUMMARY.md table shows top correlation values from `spearman_r` column
- **AND** Methods section describes FDR correction (not Bonferroni)

### Requirement: Edge Case Documentation

The system SHALL document edge case behavior in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of what produces NaN p-values:
  - Constant-valued traits (zero variance)
  - Fewer than 3 valid genotype pairs after NaN removal
- How NaN p-values are handled during FDR correction
- Why `significant_fdr` is False for NaN adjusted p-values
- Minimum sample size requirements (n >= 3) for correlation testing

#### Scenario: User understands NaN p-value behavior

- **WHEN** user sees NaN in adjusted p-value columns
- **THEN** documentation explains this occurs when:
  - A trait has constant values across all genotypes (zero variance)
  - Fewer than 3 genotypes have valid data for both traits
- **AND** documentation confirms this is expected behavior, not a bug
- **AND** documentation explains that `significant_fdr` is False for these rows

### Requirement: Correlation Confidence Interval Calculation

The system SHALL provide a function `calculate_correlation_ci(r, n, confidence_level=0.95)` in `cross_experiment_analysis.py` that computes confidence intervals for correlation coefficients using the Fisher z-transformation method:

1. **Transform to z-scale**: z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
2. **Compute standard error**: SE_z = 1 / √(n-3)
3. **Compute z-scale CI**: z ± z_{α/2} × SE_z where α = 1 - confidence_level
4. **Back-transform to r-scale**: r = tanh(z)
5. **Clamp bounds**: Ensure -1 ≤ ci_low ≤ ci_high ≤ 1

Edge case handling:
- **r = ±1.0**: Return (r, r) as point mass (arctanh undefined at boundaries)
- **n < 4**: Return (NaN, NaN) as variance undefined (n-3 in denominator)
- **r = NaN**: Return (NaN, NaN)

#### Scenario: CI for moderate correlation with adequate sample size

- **WHEN** r = 0.5 and n = 20 with confidence_level = 0.95
- **THEN** CI bounds are approximately (0.06, 0.78)
- **AND** interval is symmetric on z-scale but asymmetric on r-scale

#### Scenario: CI for zero correlation

- **WHEN** r = 0.0 and n = 30 with confidence_level = 0.95
- **THEN** CI bounds are approximately (-0.36, 0.36)
- **AND** interval contains zero (expected for null correlation)

#### Scenario: CI for perfect correlation

- **WHEN** r = 1.0 and n = 50
- **THEN** CI is (1.0, 1.0) as point mass
- **AND** no mathematical error from arctanh(1.0) = infinity

#### Scenario: CI for negative perfect correlation

- **WHEN** r = -1.0 and n = 50
- **THEN** CI is (-1.0, -1.0) as point mass
- **AND** no mathematical error from arctanh(-1.0) = -infinity

#### Scenario: CI undefined for very small n

- **WHEN** r = 0.5 and n = 3
- **THEN** CI is (NaN, NaN)
- **AND** warning is logged about insufficient sample size for CI

#### Scenario: CI for near-boundary correlation

- **WHEN** r = 0.99 and n = 100
- **THEN** CI bounds are valid: ci_low < 0.99 < ci_high ≤ 1.0
- **AND** bounds are clamped to [-1, 1] if numerical precision causes overshoot

#### Scenario: Higher confidence level widens interval

- **WHEN** same r and n but confidence_level changes from 0.95 to 0.99
- **THEN** CI width increases by factor of approximately z_{0.005}/z_{0.025} ≈ 2.576/1.96 ≈ 1.31

### Requirement: Confidence Interval Documentation

The system SHALL document confidence interval methodology in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Mathematical formulation of Fisher z-transformation with:
  - Forward transformation: z = arctanh(r)
  - Standard error formula: SE_z = 1/√(n-3)
  - CI formula on z-scale: z ± z_{α/2} × SE_z
  - Back-transformation: r = tanh(z)
- Explanation of why n ≥ 4 is required (variance formula denominator)
- Note that Fisher CI is exact for Pearson, asymptotic approximation for Spearman
- Example calculation with r = 0.6, n = 25, 95% CI
- Interpretation guidance: narrower CI = more precise estimate
- CSV column descriptions for CI bounds
- Academic reference: Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. Metron, 1, 3-32.

#### Scenario: User understands CI calculation

- **WHEN** user reads the Confidence Intervals section
- **THEN** they can manually calculate CI for a given r and n
- **AND** they understand why CI width depends on sample size

#### Scenario: User interprets CI in results

- **WHEN** user examines cross_platform_correlations.csv with CI columns
- **THEN** documentation explains that overlapping CIs suggest non-significant difference
- **AND** documentation clarifies CI is for the correlation coefficient, not for predictions

