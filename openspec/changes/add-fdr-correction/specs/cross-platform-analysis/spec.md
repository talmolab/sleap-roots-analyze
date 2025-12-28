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

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- Compute **both Pearson and Spearman** correlations for all trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- Apply FDR correction to p-values based on `fdr_correction_method` config:
  - `fdr_bh`: Benjamini-Hochberg correction (assumes test independence)
  - `fdr_by`: Benjamini-Yekutieli correction (valid under arbitrary dependence)
  - `none`: No correction applied (adjusted p-values equal raw p-values)
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `pearson_r`, `pearson_p`, `n_genotypes`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory
- The `correlation_method` config determines:
  - Which metric is used for sorting/ranking
  - Which metric is used for significance filtering
  - Which metric is considered "primary" in visualizations

#### Scenario: Dual correlation calculation with Spearman primary

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates BOTH Spearman and Pearson correlations for all 50 pairs
- **AND** exports CSV with columns: exp1_trait, exp2_trait, spearman_r, spearman_p, pearson_r, pearson_p, n_genotypes, spearman_p_adjusted, pearson_p_adjusted, significant_fdr
- **AND** sorts results by absolute Spearman correlation (descending)

#### Scenario: Dual correlation calculation with Pearson primary

- **WHEN** correlation method is "pearson" with normally distributed trait data
- **THEN** step calculates BOTH Spearman and Pearson correlations
- **AND** sorts results by absolute Pearson correlation (descending)

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

### Requirement: Reproducible Output Structure

The system SHALL generate reproducible outputs with consistent directory structure:

```
<output_base_dir>/
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman + FDR)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, pearson_r, pearson_p,
│            n_genotypes, spearman_p_adjusted, pearson_p_adjusted, significant_fdr
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

#### Scenario: CSV schema includes FDR correction columns

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains columns for both raw and adjusted p-values
- **AND** column names include: spearman_p_adjusted, pearson_p_adjusted, significant_fdr

#### Scenario: Metadata includes FDR correction information

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes fdr_correction_method and significant_correlations count

#### Scenario: Consistent output structure across runs

- **WHEN** user runs cross-platform pipeline multiple times with different configs
- **THEN** each run creates timestamped directory with identical internal structure for easy comparison

## ADDED Requirements

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
