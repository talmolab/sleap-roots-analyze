# Cross-Platform Analysis Specification

**Important Note**: Both Turface and Cylinder experiments use single-timepoint imaging. Turface uses RhizoVision imaging (not 3D), and Cylinder uses SLEAP Roots imaging (not 2D time-series). Configuration names should reflect the platform (e.g., "Turface 19 Genotypes", "Cylinder EDPIE") without incorrect dimensional or temporal qualifiers.

## ADDED Requirements

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

#### Scenario: Valid configuration with required fields

- **WHEN** user provides valid paths and column names for both experiments
- **THEN** configuration object is created successfully with default optional parameters

#### Scenario: Missing required fields

- **WHEN** user provides configuration missing required fields (data paths or genotype columns)
- **THEN** configuration validation fails with clear error message indicating missing fields

#### Scenario: Invalid correlation method

- **WHEN** user specifies correlation method not in ["spearman", "pearson", "kendall"]
- **THEN** configuration validation fails with error listing valid options

### Requirement: Load and Align Cross-Platform Data

The system SHALL load and align data from two experimental platforms through `LoadCrossPlatformDataStep` with the following behavior:

- Load CSV files from both experiment data paths
- Identify common genotypes between experiments using specified genotype columns
- Extract numeric trait columns using existing `get_trait_columns()` function
- Validate that each genotype meets minimum sample requirements
- Store aligned DataFrames and metadata for downstream steps

#### Scenario: Successful data loading with common genotypes

- **WHEN** both experiment CSV files exist with 15 common genotypes and 3+ samples per genotype
- **THEN** step loads both datasets, identifies 15 common genotypes, and stores aligned data in pipeline metadata

#### Scenario: No common genotypes found

- **WHEN** experiment datasets have no overlapping genotypes
- **THEN** step fails with error message indicating no common genotypes and listing available genotypes from each experiment

#### Scenario: Insufficient samples per genotype

- **WHEN** common genotypes exist but fewer than `min_samples_per_genotype` samples available
- **THEN** step excludes those genotypes and warns user, proceeding only with genotypes meeting threshold

#### Scenario: Missing or invalid file paths

- **WHEN** one or both experiment data paths do not exist
- **THEN** step fails immediately with FileNotFoundError indicating which path is missing

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- Compute correlations for all trait pairs using selected method (Spearman/Pearson/Kendall)
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation
- Store results with columns: trait1, trait2, correlation_coefficient, p_value, n_genotypes, abs_correlation
- Sort results by absolute correlation value (descending)
- Export results to `cross_platform_correlations.csv` in output directory

#### Scenario: Spearman correlation calculation

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates Spearman rank correlations for all 50 pairs, generates p-values, and exports CSV sorted by absolute correlation

#### Scenario: Pearson correlation calculation

- **WHEN** correlation method is "pearson" with normally distributed trait data
- **THEN** step calculates Pearson correlations assuming linear relationships between traits

#### Scenario: Kendall correlation calculation

- **WHEN** correlation method is "kendall" for robust rank-based correlation
- **THEN** step calculates Kendall tau correlations accounting for tied ranks

#### Scenario: Handling missing data in correlations

- **WHEN** trait pairs have NaN values for some genotypes
- **THEN** step removes NaN pairs before calculation and reports actual n_genotypes used per correlation

#### Scenario: Insufficient valid pairs

- **WHEN** after removing NaN pairs, fewer than 3 valid genotype pairs remain
- **THEN** step skips that trait pair and logs warning about insufficient data

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

#### Scenario: Generate summary visualization

- **WHEN** correlation results contain 7,056 trait pairs with max |ρ| = 0.389
- **THEN** step generates 4-panel summary showing distribution, volcano plot, and top 15 positive/negative correlations

#### Scenario: Generate joint plots for top correlations

- **WHEN** top_n_joint_plots is 6 and correlation results contain sufficient data
- **THEN** step generates 6 joint plots for trait pairs with highest absolute correlations, showing scatter with marginal distributions and Spearman ρ annotation

#### Scenario: Generate genotype boxplots

- **WHEN** top_n_boxplots is 6 and genotype-level data available
- **THEN** step generates 6 side-by-side boxplot figures comparing genotype distributions across experiments

#### Scenario: No significant correlations found

- **WHEN** all correlation p-values exceed significance threshold
- **THEN** summary visualization still generated with annotation indicating 0 significant correlations

#### Scenario: Insufficient negative correlations

- **WHEN** fewer than 15 negative correlations exist
- **THEN** panel 4 of summary displays available negative correlations or empty panel with "No negative correlations found" message

### Requirement: Cross-Platform Pipeline Integration

The system SHALL integrate cross-platform analysis steps into the existing pipeline infrastructure with the following characteristics:

- Steps implement standard pipeline step interface (execute method, metadata passing)
- Configuration loaded through existing OmegaConf pipeline config system
- Output directory follows existing pipeline convention (timestamped run directories)
- Steps can be run independently or chained in sequence
- Progress logged to pipeline log file
- Failures provide actionable error messages with context

#### Scenario: Run complete cross-platform pipeline

- **WHEN** user provides valid configuration and executes all three steps in sequence
- **THEN** pipeline loads data, calculates correlations, generates visualizations, and saves all outputs to timestamped run directory

#### Scenario: Resume from intermediate step

- **WHEN** user runs LoadCrossPlatformDataStep first, then separately runs CalculateCrossPlatformCorrelationsStep
- **THEN** second step loads metadata from first step and continues analysis without re-loading data

#### Scenario: Pipeline failure with recovery

- **WHEN** CalculateCrossPlatformCorrelationsStep fails due to insufficient data
- **THEN** pipeline logs detailed error with data statistics, does not corrupt metadata, and allows user to adjust config and retry

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
├── cross_platform_correlations.csv     # All correlation results
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

#### Scenario: Consistent output structure across runs

- **WHEN** user runs cross-platform pipeline multiple times with different configs
- **THEN** each run creates timestamped directory with identical internal structure for easy comparison

#### Scenario: Summary JSON contains key metrics

- **WHEN** pipeline completes successfully
- **THEN** summary.json includes: experiment names, common genotypes count, total correlations, significant correlations count, max/mean absolute correlation, top correlation details

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
