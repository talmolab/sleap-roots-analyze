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

The system SHALL load and align data from two experimental platforms with the following behavior:

- Load data from two CSV files specified in configuration
- Standardize genotype column names to "genotype" and replicate columns to "replicate"
- Identify common genotypes present in both experiments
- Filter genotypes by minimum sample count requirement
- Identify trait columns using `get_trait_columns()` with exclusion lists from config
- **Replicate column detection**: When searching for replicate columns, if multiple variants are found (e.g., both "Replicate" and "rep"), the system SHALL issue a UserWarning indicating which column will be used.
- **Default argument safety**: Functions with default list parameters (e.g., `calculate_genotype_statistics`) SHALL use `None` as default with runtime initialization to prevent mutable default argument bugs.

#### Scenario: Exclude columns during data loading

- **WHEN** LoadCrossPlatformDataStep executes with exp1_exclude_cols=["Ent", "Sub"] in config
- **THEN** get_trait_columns() is called with additional_exclude=["Ent", "Sub"] for experiment 1
- **AND** trait list metadata does not include "Ent" or "Sub"

#### Scenario: No exclusion columns specified

- **WHEN** LoadCrossPlatformDataStep executes with exp1_exclude_cols=None (default)
- **THEN** get_trait_columns() is called with additional_exclude=None
- **AND** all numeric columns (except genotype/replicate) are treated as traits

#### Scenario: Multiple replicate column variants

- **WHEN** a DataFrame has both "Replicate" and "rep" columns
- **THEN** load_and_align_experiments issues a UserWarning indicating which column will be used
- **AND** the first matching column variant is used consistently

#### Scenario: Mutable default argument protection

- **WHEN** calculate_genotype_statistics is called without statistics parameter
- **THEN** the default statistics list is created fresh for each call
- **AND** mutations to the returned statistics do not affect future calls
