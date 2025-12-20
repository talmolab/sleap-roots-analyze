## MODIFIED Requirements

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
