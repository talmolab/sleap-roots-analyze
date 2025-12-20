## MODIFIED Requirements

### Requirement: Visualize Cross-Platform Correlations

The system SHALL generate publication-quality visualizations through `VisualizeCrossPlatformStep` with the following outputs:

- **Summary visualization** (4-panel figure):
  - Panel 1: Histogram of correlation distribution with significance counts
  - Panel 2: Volcano plot (correlation vs -log10(p-value)) with significance thresholds
  - Panel 3: Horizontal bar chart of top positive correlations
  - Panel 4: Horizontal bar chart of top negative correlations
- **Joint plots**: Scatter plots with marginal distributions for top N correlated trait pairs
- **Genotype boxplots**: Side-by-side boxplots comparing genotype distributions for top N trait pairs
- All figures saved to `figures/` subdirectory in output directory
- Figure format and DPI configurable through pipeline settings
- **Correlation values displayed in visualizations SHALL match the values in `cross_platform_correlations.csv` exactly** (single source of truth)

#### Scenario: Generate summary visualization

- **WHEN** correlation results contain 7,056 trait pairs with max |rho| = 0.389
- **THEN** step generates 4-panel summary showing distribution, volcano plot, and top 15 positive/negative correlations

#### Scenario: Generate joint plots for top correlations

- **WHEN** top_n_joint_plots is 6 and correlation results contain sufficient data
- **THEN** step generates 6 joint plots for trait pairs with highest absolute correlations, showing scatter with marginal distributions and Spearman rho annotation

#### Scenario: Generate genotype boxplots

- **WHEN** top_n_boxplots is 6 and genotype-level data available
- **THEN** step generates 6 side-by-side boxplot figures comparing genotype distributions across experiments

#### Scenario: No significant correlations found

- **WHEN** all correlation p-values exceed significance threshold
- **THEN** summary visualization still generated with annotation indicating 0 significant correlations

#### Scenario: Insufficient negative correlations

- **WHEN** fewer than 15 negative correlations exist
- **THEN** panel 4 of summary displays available negative correlations or empty panel with "No negative correlations found" message

#### Scenario: Correlation values in visualizations match CSV output

- **WHEN** joint plots are generated for top correlated trait pairs
- **THEN** the correlation coefficient, p-value, and n_genotypes displayed in each plot annotation SHALL exactly match the corresponding row in `cross_platform_correlations.csv`
- **AND** no independent recalculation of these values SHALL occur in the visualization step

#### Scenario: Genotypes filtered by min_samples_per_genotype

- **WHEN** `min_samples_per_genotype` is configured to exclude genotypes with insufficient samples
- **THEN** joint plot n_genotypes SHALL reflect only genotypes that passed the filter (matching CSV)
- **AND** excluded genotypes SHALL NOT appear in visualization statistics even if they exist in both experiment DataFrames

## ADDED Requirements

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
