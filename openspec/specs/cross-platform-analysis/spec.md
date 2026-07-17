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
- `min_genotypes_for_correlation`: Minimum number of valid genotypes required for a trait pair correlation, default 10. Trait pairs with fewer valid genotypes after NaN removal are excluded from output.
- `power_analysis_alpha`: Significance level (α) for power analysis, default 0.05. Used to calculate minimum detectable effect size and achieved power.
- `power_analysis_power`: Target power (1-β) for minimum detectable effect size calculation, default 0.80. Standard convention is 80% power.
- `trait_reduction_method`: Method for reducing trait redundancy ("none", "clustering"), default "none"
- `trait_clustering_threshold`: Minimum |r| for traits to be considered redundant, default 0.8
- `trait_clustering_linkage`: Linkage method for hierarchical clustering ("complete", "average", "single"), default "complete"
- `enrichment_enabled`: Whether to run the trait-level enrichment step (a binomial test on the nominal-significance count), default False so existing runs are unchanged
- `enrichment_p_value_column`: Which p-value column the enrichment step tests, must match `correlation_method`
- `validate_input`: Input-contract validation mode at the cross-platform load boundary ("warn", "error", "off")
- `prediction`: A `PredictionConfig` instance (see the "Cross-Platform Prediction Configuration" requirement below) controlling optional cross-platform genotype-effect prediction for this same `exp1`/`exp2` pair, default `PredictionConfig()` (`enabled=False`) so existing configurations are unaffected

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

#### Scenario: Prediction defaults to disabled

- **WHEN** a `CrossPlatformConfig` is constructed from a YAML with no `prediction:` key
- **THEN** `.prediction` SHALL be a `PredictionConfig` instance with `enabled == False`
- **AND** no prediction-related validation SHALL run (see "Cross-Platform Prediction Configuration")

#### Scenario: platform_pairs direction must match exp1_name/exp2_name

- **GIVEN** `prediction.enabled=True` with `prediction.platform_pairs` set
- **WHEN** the `{source, target}` names in `prediction.platform_pairs`' single entry do not equal
  `{exp1_name, exp2_name}`
- **THEN** `CrossPlatformConfig.__post_init__` SHALL raise `ValueError` naming the mismatch

#### Scenario: platform_pairs direction accepted in either order

- **GIVEN** `prediction.enabled=True`
- **WHEN** `prediction.platform_pairs == [{"source": exp1_name, "target": exp2_name}]` or
  `[{"source": exp2_name, "target": exp1_name}]`
- **THEN** `CrossPlatformConfig` construction SHALL succeed

#### Scenario: platform_pairs must contain exactly one entry

- **GIVEN** `prediction.enabled=True`
- **WHEN** `prediction.platform_pairs` has zero entries (the default) or more than one entry
- **THEN** `CrossPlatformConfig.__post_init__` SHALL raise `ValueError` stating exactly one entry is
  required, checked before the direction-match scenarios above

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
- **Filter trait pairs**: Exclude trait pairs where n_genotypes < min_genotypes_for_correlation after NaN removal
- Compute **both Pearson and Spearman** correlations for all remaining trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- **Calculate confidence intervals** for each correlation coefficient using Fisher z-transformation
- **Calculate achieved power** for each correlation using Fisher z-transformation power formula
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
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r`, `pearson_p`, `pearson_r_ci_low`, `pearson_r_ci_high`, `n_genotypes`, `achieved_power`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory
- **Log filtering information**: Log the count and reason for filtered trait pairs

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

#### Scenario: Trait pairs filtered by min_genotypes_for_correlation

- **WHEN** min_genotypes_for_correlation is 10 and a trait pair has n_genotypes = 7
- **THEN** that trait pair is excluded from the CSV output
- **AND** metadata includes count of filtered trait pairs
- **AND** metadata includes reason: "n_genotypes below threshold"
- **AND** log message indicates how many trait pairs were filtered

#### Scenario: Achieved power computed for all correlations

- **WHEN** correlation step executes with varying r values and n_genotypes
- **THEN** CSV output contains `achieved_power` column
- **AND** achieved_power is higher for larger |r| values (same n)
- **AND** achieved_power is higher for larger n_genotypes (same r)
- **AND** achieved_power is in range [0, 1]

#### Scenario: Achieved power for zero correlation

- **WHEN** r = 0.0 for a trait pair
- **THEN** achieved_power equals power_analysis_alpha (false positive rate, no effect to detect)

#### Scenario: Achieved power for very small n

- **WHEN** n_genotypes < 4 for a trait pair
- **THEN** achieved_power is NaN (power undefined for undefined variance)

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
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman + FDR + CI + Power)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, spearman_r_ci_low, spearman_r_ci_high,
│            pearson_r, pearson_p, pearson_r_ci_low, pearson_r_ci_high, n_genotypes, achieved_power,
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

#### Scenario: CSV schema includes achieved_power column

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains `achieved_power` column
- **AND** column is placed after n_genotypes

#### Scenario: Metadata includes confidence level

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `confidence_level` parameter value used

#### Scenario: Metadata includes FDR correction information

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes fdr_correction_method and significant_correlations count

#### Scenario: Metadata includes power analysis parameters

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `power_analysis_alpha`, `power_analysis_power`, `minimum_detectable_r`
- **AND** metadata includes `modal_n_genotypes` (most common sample size used for MDR calculation)
- **AND** metadata includes `n_correlations_filtered_low_n` (count of trait pairs excluded due to low n)

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

1. **Validate inputs**:
   - Raise `ValueError` if r is not in [-1, 1] (and not NaN)
   - Raise `ValueError` if confidence_level is not in (0, 1) exclusive range
   - Raise `ValueError` if n <= 0
2. **Transform to z-scale**: z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
3. **Compute standard error**: SE_z = 1 / √(n-3)
4. **Compute z-scale CI**: z ± z_{α/2} × SE_z where α = 1 - confidence_level
5. **Back-transform to r-scale**: r = tanh(z)
6. **Clamp bounds**: Ensure -1 ≤ ci_low ≤ ci_high ≤ 1

Edge case handling:
- **r = ±1.0**: Return (r, r) as point mass (arctanh undefined at boundaries)
- **n < 4**: Return (NaN, NaN) as variance undefined (n-3 in denominator)
- **r = NaN**: Return (NaN, NaN) without raising validation error

Documentation:
- Docstring SHALL note that for Spearman correlations, Fisher z-based CI is accurate for n >= 10; for 4 <= n < 10, results are approximate
- Docstring SHALL cross-reference `calculate_correlation_confidence_intervals` for DataFrame-based operations

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
- **THEN** CI bounds are valid: ci_low < 0.99 < ci_high <= 1.0
- **AND** bounds are clamped to [-1, 1] if numerical precision causes overshoot

#### Scenario: Higher confidence level widens interval

- **WHEN** same r and n but confidence_level changes from 0.95 to 0.99
- **THEN** CI width increases by factor of approximately z_{0.005}/z_{0.025} ≈ 2.576/1.96 ≈ 1.31

#### Scenario: Invalid correlation coefficient raises error

- **WHEN** r = 1.5 (outside valid range)
- **THEN** function raises ValueError with message indicating r must be in [-1, 1]
- **AND** error occurs before any computation

#### Scenario: Invalid confidence level raises error

- **WHEN** confidence_level = 0 or confidence_level = 1.0 or confidence_level = -0.5
- **THEN** function raises ValueError with message indicating confidence_level must be in (0, 1)
- **AND** error occurs before any computation

#### Scenario: NaN correlation does not raise validation error

- **WHEN** r = NaN
- **THEN** function returns (NaN, NaN) without raising ValueError
- **AND** this allows graceful handling of missing correlation data

#### Scenario: Invalid sample size raises error

- **WHEN** n = 0 or n = -5
- **THEN** function raises ValueError with message indicating n must be positive
- **AND** error occurs before any computation

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

### Requirement: Power Analysis Functions

The system SHALL provide power analysis functions in `cross_experiment_analysis.py` for computing statistical power of correlation tests:

**Function: `minimum_detectable_correlation(n, alpha=0.05, power=0.80)`**

Calculates the minimum absolute correlation coefficient detectable at the specified power level using Fisher z-transformation:

1. **Validate inputs**: Raise ValueError if n <= 0, alpha not in (0, 1), or power not in (0, 1)
2. **Compute critical z-values**: z_α = Φ⁻¹(1 - α/2), z_β = Φ⁻¹(power)
3. **Compute z_r threshold**: z_r = (z_α + z_β) / √(n - 3)
4. **Transform to correlation scale**: r = tanh(z_r)
5. **Handle edge cases**: n < 4 returns NaN (undefined variance)

**Function: `achieved_power(r, n, alpha=0.05)`**

Calculates the statistical power achieved for a given correlation coefficient using Fisher z-transformation:

1. **Validate inputs**: Raise ValueError if r not in [-1, 1] (and not NaN), alpha not in (0, 1), n <= 0
2. **Compute critical z-value**: z_α = Φ⁻¹(1 - α/2)
3. **Transform r to z-scale**: z_r = arctanh(|r|)
4. **Compute non-centrality parameter**: ncp = z_r × √(n - 3)
5. **Compute power**: power = Φ(ncp - z_α) + Φ(-ncp - z_α)
6. **Handle edge cases**:
   - r = 0: Return alpha (no effect to detect, power equals false positive rate)
   - r = NaN: Return NaN
   - n < 4: Return NaN (undefined variance)
   - r = ±1.0: Return 1.0 (perfect correlation, maximum power)

**References**:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates.
- Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. Metron, 1, 3-32.

#### Scenario: Minimum detectable r with standard parameters

- **WHEN** n = 20, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.58
- **AND** correlations with |r| < 0.58 have power < 80% at this sample size

#### Scenario: Minimum detectable r with large sample

- **WHEN** n = 100, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.28
- **AND** larger samples can detect smaller effect sizes at 80% power

#### Scenario: Minimum detectable r with small sample

- **WHEN** n = 10, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.76
- **AND** value is close to 1.0, indicating severe power limitations

#### Scenario: Achieved power for moderate correlation

- **WHEN** r = 0.5, n = 20, alpha = 0.05
- **THEN** achieved_power returns approximately 0.65
- **AND** this is below the 80% threshold, indicating underpowered

#### Scenario: Achieved power for strong correlation

- **WHEN** r = 0.7, n = 20, alpha = 0.05
- **THEN** achieved_power returns approximately 0.92
- **AND** this is above the 80% threshold, indicating adequate power

#### Scenario: Achieved power for zero correlation

- **WHEN** r = 0.0, n = 20, alpha = 0.05
- **THEN** achieved_power returns 0.05 (equals alpha)
- **AND** this represents the false positive rate (no effect to detect)

#### Scenario: Achieved power for perfect correlation

- **WHEN** r = 1.0, n = 20, alpha = 0.05
- **THEN** achieved_power returns 1.0 (maximum power)

#### Scenario: Achieved power with very small n

- **WHEN** r = 0.5, n = 3, alpha = 0.05
- **THEN** achieved_power returns NaN (variance undefined with n-3 denominator)

#### Scenario: Invalid inputs raise errors

- **WHEN** minimum_detectable_correlation is called with n = 0
- **THEN** ValueError is raised with message indicating n must be positive

- **WHEN** achieved_power is called with r = 1.5
- **THEN** ValueError is raised with message indicating r must be in [-1, 1]

- **WHEN** either function is called with alpha = 0 or alpha = 1
- **THEN** ValueError is raised with message indicating alpha must be in (0, 1)

### Requirement: Power Analysis Documentation

The system SHALL document power analysis methodology in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of statistical power and why it matters for correlation analysis
- Mathematical formulation of power calculation using Fisher z-transformation:
  - Minimum detectable r formula: r = tanh((z_α + z_β) / √(n-3))
  - Achieved power formula: power = Φ(z_r × √(n-3) - z_α) + Φ(-z_r × √(n-3) - z_α)
- Table of minimum detectable r values for common sample sizes (n = 10, 15, 20, 25, 30, 50, 100)
- Interpretation guidance:
  - power < 0.50: severely underpowered, results unreliable
  - power 0.50-0.80: moderately underpowered, interpret with caution
  - power >= 0.80: adequately powered, standard threshold
  - power >= 0.90: well-powered, high confidence
- Explanation that same formula applies to both Pearson and Spearman correlations (asymptotically)
- Note about n >= 10 recommendation for Spearman (Fisher z approximation accuracy)
- Academic references: Cohen (1988), Fisher (1921)

#### Scenario: User understands power interpretation

- **WHEN** user examines achieved_power column in CSV
- **THEN** documentation explains that power < 0.80 indicates underpowered correlation
- **AND** documentation provides context for interpreting specific power values

#### Scenario: User understands minimum detectable r

- **WHEN** user sees minimum_detectable_r in metadata
- **THEN** documentation explains this is the smallest effect size detectable at 80% power
- **AND** documentation clarifies correlations below this threshold are statistically unreliable

### Requirement: Minimum Genotypes Filter Documentation

The system SHALL document the minimum genotypes filter in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of why filtering is necessary:
  - NaN values in specific traits can reduce effective sample size below threshold
  - Very low n leads to unreliable correlations and undefined confidence intervals
  - Fisher z approximation is accurate for n >= 10 for Spearman correlations
- Description of the filtering behavior:
  - Hard filter: trait pairs below threshold are excluded from CSV
  - Filtering occurs after NaN removal, using actual n_genotypes
  - Metadata logs count and reason for filtered pairs
- Distinction from `min_samples_per_genotype`:
  - `min_samples_per_genotype`: Filters genotypes with too few biological replicates
  - `min_genotypes_for_correlation`: Filters trait pairs with too few valid genotypes after NaN removal
- Guidance on choosing the threshold:
  - n >= 4: Minimum for CI calculation (variance undefined below this)
  - n >= 10: Recommended for accurate Fisher z approximation (Spearman)
  - n >= 20: Preferred for reliable correlation estimates
- Example showing how trait-specific NaN patterns can reduce effective n

#### Scenario: User understands filtering reason

- **WHEN** user sees n_correlations_filtered_low_n in metadata
- **THEN** documentation explains these trait pairs had insufficient valid genotypes
- **AND** documentation clarifies this is protective, not a bug

#### Scenario: User chooses appropriate threshold

- **WHEN** user decides on min_genotypes_for_correlation value
- **THEN** documentation provides guidance based on statistical requirements
- **AND** documentation recommends n >= 10 as default for Spearman accuracy

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

The system SHALL generate comprehensive markdown summaries from cross-platform correlation analysis outputs using the `CrossPlatformSummaryGenerator` class.

The `to_markdown()` method SHALL accept the following parameters:
- `image_mode`: One of "file_path", "embed", or "auto" (default: "file_path")
- `embed_threshold_bytes`: Maximum bytes for embedded images (default: 10,485,760)

The `generate()` method SHALL return a `CrossPlatformSummary` object that can be converted to markdown or HTML format.

#### Scenario: Default generates file-path markdown

- **WHEN** `CrossPlatformSummaryGenerator.generate()` is called
- **AND** `summary.to_markdown()` is called with no arguments
- **THEN** the output SHALL use relative file paths for images
- **AND** the file size SHALL be less than 1MB for typical analyses

#### Scenario: Embedded images requested under threshold

- **WHEN** `summary.to_markdown(image_mode="embed")` is called
- **AND** total visualization size is under 10MB
- **THEN** images SHALL be embedded as base64 data URIs
- **AND** the markdown SHALL be self-contained

#### Scenario: Summary written by pipeline runner

- **WHEN** the pipeline runner generates SUMMARY.md
- **THEN** it SHALL use `image_mode="file_path"` by default
- **AND** optionally accept `--embed-images` flag for portable output
- **AND** optionally accept `--html` flag for HTML output

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

### Requirement: Memory-Aware Summary Image Handling

The system SHALL support configurable image handling modes for cross-platform analysis summaries with the following options:

- `file_path` (default): Use relative file paths for all images
- `embed`: Embed images as base64 data URIs if total size < threshold, otherwise fallback to file_path with warning
- `auto`: Automatically select based on total image size

The default embed threshold SHALL be 10MB (10,485,760 bytes).

#### Scenario: File path mode generates small summary

- **WHEN** generating a cross-platform summary with `image_mode="file_path"`
- **THEN** the SUMMARY.md file SHALL be less than 1MB
- **AND** all image references SHALL use relative file paths
- **AND** images SHALL be viewable when SUMMARY.md is opened in VS Code markdown preview

#### Scenario: Embed mode respects size threshold

- **WHEN** generating a cross-platform summary with `image_mode="embed"`
- **AND** total image size is less than 10MB
- **THEN** images SHALL be embedded as base64 data URIs

#### Scenario: Embed mode falls back when over threshold

- **WHEN** generating a cross-platform summary with `image_mode="embed"`
- **AND** total image size exceeds the threshold
- **THEN** a warning SHALL be logged
- **AND** the system SHALL fall back to file_path mode
- **AND** the summary SHALL be generated successfully

#### Scenario: Auto mode selects appropriate method

- **WHEN** generating a cross-platform summary with `image_mode="auto"`
- **THEN** the system SHALL calculate total image size
- **AND** embed images if total size < threshold
- **AND** use file paths if total size >= threshold

### Requirement: HTML Summary Output

The system SHALL support generating HTML output format for cross-platform summaries that can be viewed directly in web browsers.

#### Scenario: HTML output generated with markdown

- **WHEN** generating a cross-platform summary with `output_format="both"`
- **THEN** both SUMMARY.md and SUMMARY.html SHALL be created
- **AND** SUMMARY.html SHALL include embedded CSS styling
- **AND** SUMMARY.html SHALL render correctly in Chrome/Firefox

#### Scenario: HTML output contains proper structure

- **WHEN** generating HTML summary output
- **THEN** the HTML SHALL include proper DOCTYPE and charset
- **AND** tables SHALL be styled with borders and padding
- **AND** images SHALL be properly referenced
- **AND** the file SHALL be self-contained (embedded styles)

### Requirement: Image Size Calculation

The system SHALL calculate total image size before generating summaries to support memory-aware decisions.

#### Scenario: Total size calculated for all images

- **WHEN** preparing to generate a summary
- **THEN** the system SHALL calculate the total size of all visualization images
- **AND** estimate base64 overhead (approximately 1.37x raw size)
- **AND** make this information available for mode selection

#### Scenario: Missing images handled gracefully

- **WHEN** calculating total image size
- **AND** some image files do not exist
- **THEN** missing files SHALL be skipped without error
- **AND** a warning SHALL be logged for each missing file

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

### Requirement: Cross-Platform Plot Label Formatting via sanitize_trait_names
Cross-platform joint plots, boxplots, and heatmap axis labels SHALL use the existing `sanitize_trait_names()` function from `data_utils.py` for consistent trait name formatting, rather than ad-hoc string replacement.

#### Scenario: Joint plot axis labels
- **WHEN** generating a cross-platform joint plot with trait names as axis labels
- **THEN** labels SHALL be formatted using `sanitize_trait_names()` from `data_utils.py`
- **AND** ad-hoc `.replace('_', ' ').title()` calls SHALL be removed in favor of the shared utility
- **AND** labels SHALL match the formatting used in QC pipeline outputs

#### Scenario: Boxplot axis labels
- **WHEN** generating a cross-platform genotype boxplot with trait names
- **THEN** trait names SHALL be formatted using the shared `sanitize_trait_names()` utility
- **AND** formatting SHALL be consistent with QC and Viz pipeline label formatting

#### Scenario: Heatmap axis labels in trait clustering
- **WHEN** generating trait cluster heatmaps or dendrograms
- **THEN** trait labels on axes SHALL be formatted using the shared utility
- **AND** labels SHALL remain legible at the figure's native resolution

### Requirement: Empty Correlation Graceful Handling
The `VisualizeCrossPlatformStep` SHALL handle empty correlation DataFrames gracefully without crashing the pipeline.

#### Scenario: All correlations filtered out
- **WHEN** `VisualizeCrossPlatformStep` receives an empty `correlation_df` (all trait pairs filtered by upstream step)
- **THEN** the step SHALL skip all plot generation
- **AND** the step SHALL log a warning indicating no correlations to visualize
- **AND** the step SHALL return a successful StepResult with `plots_generated: 0`
- **AND** output metadata SHALL include `"empty_correlations": true`

#### Scenario: Non-empty correlation DataFrame (default behavior)
- **WHEN** `VisualizeCrossPlatformStep` receives a non-empty `correlation_df`
- **THEN** the step SHALL generate all configured plots as before
- **AND** output metadata SHALL NOT include `"empty_correlations"` key
- **AND** behavior SHALL be identical to current implementation

#### Scenario: Pipeline completes when all correlations are filtered out
- **GIVEN** two experiments share common genotypes
- **WHEN** `CalculateCrossPlatformCorrelationsStep` filters out all trait-pair correlations (e.g., due to strict p-value threshold or insufficient replicates)
- **AND** the resulting `correlation_df` is empty
- **THEN** `VisualizeCrossPlatformStep` SHALL complete without error
- **AND** the pipeline summary SHALL report status "success"

### Requirement: Cross-Platform Prediction Configuration

The system SHALL provide a `PredictionConfig` dataclass (nested as the `prediction` field on
`CrossPlatformConfig`) with the following parameters:

- `enabled`: bool, default `False`. When `False`, `__post_init__` SHALL perform no validation at
  all (not even structural checks on other fields), so every existing `CrossPlatformConfig` that
  predates this requirement remains valid unchanged.
- `predictor_source`: `"blup"` or `"genotype_means"`, default `"blup"`.
- `reduction_method`: the primary dimensionality-reduction method passed to `logo_cv_predict`
  (`"pls_latent"`, `"representatives"`, or `"pc1"`), default `"pls_latent"`.
- `comparison_methods`: list of additional reduction methods for robustness reporting, each drawn
  from the same `{"pls_latent", "representatives", "pc1"}` set as `reduction_method`, default
  `["representatives"]`. SHALL NOT contain `reduction_method`'s own value (would silently produce
  two methods writing to the same output file).
- `representative_selection_metric`: `"variance"` only for this tier. `"heritability"` is not a
  valid value here — `select_cluster_representatives` (reused unchanged) has no metric parameter to
  select by heritability, so this option is deferred to a future change.
- `platform_pairs`: list of `{"source": str, "target": str}` dicts, default empty. When
  `enabled=True`, SHALL contain **exactly one** entry (not zero, not more than one) naming which of
  the enclosing `CrossPlatformConfig`'s `exp1_name`/`exp2_name` is the predictor and which is
  predicted.
- `blup_refit_per_fold`: bool, default `False`. Present in the schema for forward compatibility with
  a future heritability-based `representative_selection_metric`, but currently inert in this tier —
  no valid `representative_selection_metric` value triggers any auto-force or validation on it.
- `source_blup_path` / `target_blup_path`: `Optional[str]`, default `None`. Required and
  existence-checked on disk only when `enabled=True` and `predictor_source="blup"`. Not required
  when `predictor_source="genotype_means"`.

`PredictionConfig.__post_init__` SHALL raise `ValueError` (not a new exception type) for any
validation failure, matching every other config dataclass's existing convention in this codebase.

#### Scenario: Validation is a full no-op when disabled

- **WHEN** `PredictionConfig(enabled=False, predictor_source="not_a_real_value",
  source_blup_path="/does/not/exist")` is constructed
- **THEN** no exception SHALL be raised

#### Scenario: Invalid enum field rejected when enabled

- **GIVEN** `enabled=True`
- **WHEN** `predictor_source`, `reduction_method`, `representative_selection_metric`, or any entry
  in `comparison_methods` is not one of its documented valid values
- **THEN** `ValueError` SHALL be raised naming the invalid field and value

#### Scenario: heritability metric is rejected, not accepted, in this tier

- **GIVEN** `enabled=True`
- **WHEN** `representative_selection_metric="heritability"`
- **THEN** `ValueError` SHALL be raised (same as any other invalid enum value) — this tier only
  supports `"variance"`

#### Scenario: comparison_methods rejects a duplicate of reduction_method

- **GIVEN** `enabled=True, reduction_method="pls_latent"`
- **WHEN** `comparison_methods` contains `"pls_latent"`
- **THEN** `ValueError` SHALL be raised at construction time

#### Scenario: comparison_methods rejects a duplicate entry within itself

- **GIVEN** `enabled=True`
- **WHEN** `comparison_methods` contains the same method twice (e.g.
  `["representatives", "representatives"]`), independent of `reduction_method`'s value
- **THEN** `ValueError` SHALL be raised at construction time (the same silent output-overwrite risk
  as the cross-field case above, just self-inflicted within the list)

#### Scenario: blup predictor_source requires resolvable paths (pre-flight guard)

- **GIVEN** `enabled=True, predictor_source="blup"`
- **WHEN** `source_blup_path` or `target_blup_path` does not resolve to an existing file
- **THEN** `ValueError` SHALL be raised at config-construction time, before any pipeline step runs

#### Scenario: genotype_means predictor_source does not require BLUP paths

- **GIVEN** `enabled=True, predictor_source="genotype_means"`
- **WHEN** `source_blup_path` and `target_blup_path` are both `None`
- **THEN** no exception SHALL be raised

### Requirement: Predict Cross-Platform Genotype Values Pipeline Step

The system SHALL provide `PredictCrossPlatformStep`, an optional pipeline step consuming
`PredictionConfig` and Tier 3's `logo_cv_predict`/`fit_pca_on_fold`/`CrossPlatformPredictionResult`
(all unchanged), wired as task 6 (`depends_on=["01_load_cross_platform_data",
"05_visualize_cross_platform"]`). The step SHALL read data from task 1's result only; the
dependency on task 5 exists solely to guarantee ordering (steps 1-5 complete before prediction
runs), not for data. The task SHALL be entirely absent from `create_tasks()`'s return value — not
merely skipped at run time — when `config.prediction.enabled=False`.

For a given directed pair, the step SHALL:
1. Build the source and target predictor matrices per `predictor_source`: BLUP CSVs
   (`source_blup_path`/`target_blup_path`, with the genotype column resolved as `"Genotype"` then
   `"genotype"` — distinct from `exp1_genotype_col`/`exp2_genotype_col`, which govern the unrelated
   raw per-sample CSVs for steps 1-5; a clear `ValueError` naming both attempted column names if
   neither is present), or task 1's own raw `exp1`/`exp2` data — selected via task 1's already-
   `exclude_cols`-filtered `exp1_trait_names`/`exp2_trait_names` metadata **before** aggregating by
   genotype mean (`predictor_source="genotype_means"` — reading task 1's result directly, so this
   ablation always uses the full raw trait set even when `trait_reduction_method="clustering"` has
   reduced the data by the time it reaches later steps; task 1's raw DataFrame is NOT trait-only, so
   this trait-name selection step is required, not optional). Any trait column containing any `NaN`
   value among the common-genotype set SHALL be dropped before further use, on both source and
   target sides; if this leaves the source matrix with zero trait columns, the step SHALL raise a
   clear `ValueError`.
2. Derive `X`, every per-target `y`, and the `genotypes` list from one canonical, sorted, explicitly-
   indexed common-genotype list — never from incidental row-order agreement between independently-
   loaded/joined DataFrames.
3. Select the **target** platform's cluster-representative traits (via the existing
   `cluster_correlated_traits`/`select_cluster_representatives`, unchanged) as the primary
   prediction targets, per `representative_selection_metric` (`"variance"` only, this tier).
4. Compute one additional target, `target_name="PC1"`: the **target** platform's own first
   principal component via `pca.fit_pca()` with `StandardScaler` applied first and
   `random_state=42` fixed, called directly (not `fit_pca_on_fold`, which remains reserved for
   reducing the **source** predictor matrix per-fold when `reduction_method="pc1"`; not
   `PCAAnalysisStep`, which is config-driven via a `PCAConfig` this pipeline does not have).
5. Call `logo_cv_predict` once per target trait × per method (`reduction_method` plus each of
   `comparison_methods` — guaranteed distinct from each other), assembling one
   `CrossPlatformPredictionResult` per method.
6. Save each `CrossPlatformPredictionResult` as JSON to the run directory, one file per method.

If the common-genotype count between source and target is below `logo_cv_predict`'s own minimum,
the step SHALL raise a clear `ValueError` naming the pair and the common-genotype count, rather than
passing through `logo_cv_predict`'s generic message.

The existing `cross-platform` CLI command's `--dry-run` output SHALL list this step when enabled,
and SHALL NOT list it when disabled. No new CLI command or flag SHALL be introduced.

#### Scenario: Step present only when enabled

- **WHEN** `CrossPlatformPipeline(config).create_tasks()` is called
- **THEN** a 6th task SHALL be present if and only if `config.prediction.enabled=True`

#### Scenario: Predictor matrix built from BLUP CSVs when predictor_source is blup

- **GIVEN** `predictor_source="blup"`
- **WHEN** the step runs
- **THEN** `source_blup_path`/`target_blup_path` SHALL be loaded as the predictor matrices

#### Scenario: Predictor matrix built from genotype means when predictor_source is genotype_means

- **GIVEN** `predictor_source="genotype_means"`
- **WHEN** the step runs
- **THEN** task 1's raw `exp1`/`exp2` data SHALL be filtered to `exp1_trait_names`/`exp2_trait_names`
  (task 1's own already-`exclude_cols`-filtered metadata) before aggregating via genotype-mean
  grouping, read directly from task 1's own result — not aggregated over every column in task 1's
  raw DataFrame (which also contains `genotype`, `replicate`, and other non-trait columns)

#### Scenario: genotype_means ablation is unaffected by trait_reduction_method=clustering

- **GIVEN** `predictor_source="genotype_means"` and `trait_reduction_method="clustering"` are both
  set on the same `CrossPlatformConfig`
- **WHEN** the step runs
- **THEN** the predictor matrix's columns SHALL exactly equal task 1's `exp1_trait_names`/
  `exp2_trait_names` (the full, already-filtered trait set), not the cluster-representative-reduced
  subset task 2 (`ReduceTraitRedundancyStep`) produces, and not task 1's raw DataFrame's every column

#### Scenario: Target-side cluster-representative trait selection

- **WHEN** the step selects the target platform's prediction targets
- **THEN** `select_cluster_representatives` SHALL be applied to the target platform's aligned
  predictor matrix (BLUP or genotype-mean, per `predictor_source`), independent of and using a
  separate application from the **source** platform's own representative selection (used only when
  a method is `"representatives"`, per the "logo_cv_predict called once per target trait per
  method" scenario below)

#### Scenario: X, y, and genotypes are derived from one canonical common-genotype index

- **GIVEN** the source and target predictor matrices have their genotype rows in different orders
  (same genotype set, different order)
- **WHEN** the step builds `X`, any per-target `y`, and `genotypes` for `logo_cv_predict`
- **THEN** each SHALL be indexed from one canonical, sorted, common-genotype list, so that source
  and target values for the same genotype are correctly paired regardless of either input's
  original row order — including for the PC1 target, not only representative-trait targets

#### Scenario: Task 5's dependency is for ordering only, never data

- **GIVEN** `kwargs["05_visualize_cross_platform"]` holds any value (including a sentinel or
  otherwise-unusable `TaskResult`), while `kwargs["01_load_cross_platform_data"]` is a normal, valid
  result
- **WHEN** the step runs
- **THEN** it SHALL produce a correct `CrossPlatformPredictionResult`, never reading
  `kwargs["05_visualize_cross_platform"].data`

#### Scenario: Trait columns containing any NaN are dropped before use

- **GIVEN** a source or target predictor matrix with one trait column containing a `NaN` value for
  at least one common genotype (e.g. a failed-model trait in a real `08_blup_adjusted_means.csv`)
- **WHEN** the step builds `X` or selects target-side candidate traits
- **THEN** that column SHALL be dropped before `logo_cv_predict` is called, rather than passed
  through to raise `logo_cv_predict`'s generic NaN-rejection error

#### Scenario: Clear error when the source matrix is empty after dropping NaN columns

- **WHEN** every trait column in the source predictor matrix contains at least one `NaN` value
  among the common genotypes
- **THEN** the step SHALL raise a clear `ValueError`, distinct from the zero-target-representative-
  traits case (which still has PC1 to fall back on)

#### Scenario: BLUP CSV genotype column resolved by fixed convention

- **GIVEN** `predictor_source="blup"`
- **WHEN** the step loads `source_blup_path`/`target_blup_path`
- **THEN** it SHALL resolve the genotype column as `"Genotype"` first, falling back to `"genotype"`
  — not `exp1_genotype_col`/`exp2_genotype_col`, which govern the unrelated raw per-sample CSVs
- **AND** if neither `"Genotype"` nor `"genotype"` is present, it SHALL raise a clear `ValueError`
  naming both attempted column names, not a bare pandas `KeyError`

#### Scenario: Step still runs with only the PC1 target when zero representative traits are selected

- **WHEN** `select_cluster_representatives` returns an empty list for the target platform
- **THEN** the step SHALL still run successfully, producing a `CrossPlatformPredictionResult` with
  only the PC1 target, not a crash

#### Scenario: blup_refit_per_fold has no observable effect

- **GIVEN** two otherwise-identical configs differing only in `blup_refit_per_fold` (`True` vs.
  `False`)
- **WHEN** the step runs for each
- **THEN** the resulting `CrossPlatformPredictionResult`s SHALL be identical

#### Scenario: logo_cv_predict called once per target trait per method

- **GIVEN** N target traits (representatives + PC1) and M methods (`reduction_method` +
  `comparison_methods`)
- **WHEN** the step runs
- **THEN** `logo_cv_predict` SHALL be called exactly N × M times

#### Scenario: One JSON result file saved per method

- **WHEN** the step completes
- **THEN** one `CrossPlatformPredictionResult` JSON file SHALL be written to the run directory per
  method, with no filename collisions (guaranteed by `comparison_methods` never duplicating
  `reduction_method`)

#### Scenario: Clear error when common genotypes are below the minimum

- **WHEN** the source and target predictor matrices share fewer common genotypes than
  `logo_cv_predict` requires (including zero overlap)
- **THEN** the step SHALL raise `ValueError` naming the source/target platforms and the
  common-genotype count, not a bare pass-through of `logo_cv_predict`'s generic message

#### Scenario: Backward compatible when disabled

- **GIVEN** an existing `CrossPlatformConfig` YAML with no `prediction:` key
- **WHEN** `CrossPlatformPipeline` runs
- **THEN** the run's analysis output (file list and content for the 5 existing steps — correlation
  CSVs, alignment summary, figures, `pipeline_summary.json`) SHALL be byte-identical to the same run
  before this requirement existed
- **AND** `config.yaml` is exempted from this comparison: it SHALL gain a new `prediction: {...}`
  block reflecting the `prediction` field's existence, regardless of `enabled`, since the pipeline's
  config-provenance serialization (`cli-pipeline`'s "Pipeline Run Config Provenance" requirement)
  serializes every field of the resolved config, including nested dataclasses at their default
  values — this is an expected, harmless side effect, not a behavior change

#### Scenario: PC1 target uses whole-dataset PCA, not per-fold

- **WHEN** the step computes the `target_name="PC1"` value
- **THEN** it SHALL use `pca.fit_pca()` (with `StandardScaler` applied first, `random_state=42`
  fixed) on the full common-genotype set
- **AND** `fit_pca_on_fold` SHALL NOT be called for this purpose
- **AND** the computed values SHALL match an independently-computed
  `pca.fit_pca(StandardScaler().fit_transform(X), n_components=1, random_state=42)` on the same data

#### Scenario: Dry-run lists the prediction step when enabled

- **WHEN** `sleap-roots-analyze cross-platform <config> --dry-run` runs with
  `config.prediction.enabled=True`
- **THEN** the printed step list SHALL include a 6th entry for the prediction step

#### Scenario: Dry-run omits the prediction step when disabled

- **WHEN** `sleap-roots-analyze cross-platform <config> --dry-run` runs with
  `config.prediction.enabled=False` (or the `prediction:` key absent)
- **THEN** the printed step list SHALL contain exactly the existing 5 entries

