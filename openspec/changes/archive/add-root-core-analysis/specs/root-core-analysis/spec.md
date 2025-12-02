# Root Core Analysis Specification

## ADDED Requirements

### Requirement: Sample Identifier Creation
The system SHALL provide a function to create unique sample identifiers from experimental metadata columns.

#### Scenario: Create identifier with default columns
- **GIVEN** a DataFrame with columns `['Plot', 'Rep', 'geno', 'core_n']`
- **WHEN** `create_sample_identifier()` is called with default parameters
- **THEN** return a Series with identifiers formatted as `plot{Plot}_rep{Rep}_{geno}_core{core_n}`

#### Scenario: Create identifier with custom prefix mapping
- **GIVEN** a DataFrame with metadata columns
- **WHEN** `create_sample_identifier()` is called with custom `prefix_map={'Plot': 'p', 'Rep': 'r'}`
- **THEN** return identifiers using custom prefixes (e.g., `p1_r1_GH_7386_core1`)

#### Scenario: Handle missing columns gracefully
- **GIVEN** a DataFrame missing one of the specified identifier columns
- **WHEN** `create_sample_identifier()` is called
- **THEN** raise `ValueError` with clear message indicating missing column

### Requirement: Unique Identifier Validation
The system SHALL validate that identifier columns contain unique values for all samples.

#### Scenario: All identifiers are unique
- **GIVEN** a DataFrame with a unique identifier column
- **WHEN** `validate_unique_identifiers()` is called
- **THEN** return `(True, [])` indicating no duplicates

#### Scenario: Duplicate identifiers detected
- **GIVEN** a DataFrame with duplicate values in identifier column
- **WHEN** `validate_unique_identifiers()` is called
- **THEN** return `(False, [list of duplicate values])`

#### Scenario: Empty DataFrame validation
- **GIVEN** an empty DataFrame
- **WHEN** `validate_unique_identifiers()` is called
- **THEN** return `(True, [])` (vacuously true)

### Requirement: Depth Data Melting
The system SHALL convert wide-format depth data to long format with automatic depth calculation from column names.

#### Scenario: Melt data with depth column pattern c_<start>_<end>_<subcore>
- **GIVEN** a DataFrame with columns matching pattern `c_0_10_1`, `c_0_10_2`, `c_10_20_1`, etc.
- **AND** metadata columns `['Plot', 'geno', 'Rep', 'core_n']`
- **WHEN** `melt_depth_data()` is called with `depth_prefix='c_'` and `parse_depth=True`
- **THEN** return long-format DataFrame with columns `['Plot', 'geno', 'Rep', 'core_n', 'Depth', 'Root_Count', 'Depth_cm']`
- **AND** `Depth_cm` is calculated as `start + (subcore_index - 1) * (end - start) / 2`

#### Scenario: Calculate depth for first subcore at 0-10cm
- **GIVEN** column name `c_0_10_1`
- **WHEN** parsing depth
- **THEN** `Depth_cm = 0 + (1 - 1) * (10 - 0) / 2 = 0.0`

#### Scenario: Calculate depth for second subcore at 0-10cm
- **GIVEN** column name `c_0_10_2`
- **WHEN** parsing depth
- **THEN** `Depth_cm = 0 + (2 - 1) * (10 - 0) / 2 = 5.0`

#### Scenario: Calculate depth for first subcore at 50-60cm
- **GIVEN** column name `c_50_60_1`
- **WHEN** parsing depth
- **THEN** `Depth_cm = 50 + (1 - 1) * (60 - 50) / 2 = 50.0`

#### Scenario: Melt without depth parsing
- **GIVEN** a DataFrame with depth columns
- **WHEN** `melt_depth_data()` is called with `parse_depth=False`
- **THEN** return long-format DataFrame without `Depth_cm` column

#### Scenario: Handle NaN values during melting
- **GIVEN** a DataFrame with NaN values in depth columns
- **WHEN** `melt_depth_data()` is called
- **THEN** include NaN values in melted output (filtering is user's responsibility)

#### Scenario: Invalid depth column pattern
- **GIVEN** columns not matching expected pattern (e.g., `depth_0_10`)
- **WHEN** `melt_depth_data()` is called with `depth_prefix='c_'`
- **THEN** skip non-matching columns (only melt columns starting with prefix)

### Requirement: Replicate Aggregation
The system SHALL aggregate measurements across technical replicates to biological replicate level.

#### Scenario: Aggregate cores by mean within plot-replicate
- **GIVEN** melted data with multiple cores per `plot_rep` at each `Depth_cm`
- **WHEN** `aggregate_by_replicate()` is called with `group_cols=['plot_rep', 'geno', 'Depth_cm']` and `agg_func='mean'`
- **THEN** return DataFrame with one row per unique combination of group columns
- **AND** `Root_Count` is averaged across cores

#### Scenario: Aggregate using median instead of mean
- **GIVEN** melted data with multiple measurements
- **WHEN** `aggregate_by_replicate()` is called with `agg_func='median'`
- **THEN** use median aggregation function

#### Scenario: Handle missing values during aggregation
- **GIVEN** data with NaN values in value column
- **WHEN** aggregating
- **THEN** use pandas default NaN handling (skip in mean/median calculations)

#### Scenario: Preserve all group columns
- **GIVEN** group columns `['plot_rep', 'Plot', 'Rep', 'geno', 'Depth_cm']`
- **WHEN** aggregating
- **THEN** all specified columns appear in output DataFrame

### Requirement: Faceted Depth Profile Visualization
The system SHALL create faceted line plots showing mean root count vs depth by genotype.

#### Scenario: Create default faceted plot
- **GIVEN** aggregated depth profile data with multiple genotypes
- **WHEN** `plot_depth_profile_faceted()` is called with default parameters
- **THEN** return matplotlib Figure object with FacetGrid
- **AND** each genotype has its own subplot
- **AND** line shows mean with standard error bars

#### Scenario: Customize error bars
- **GIVEN** depth profile data
- **WHEN** called with `errorbar='sd'`
- **THEN** display standard deviation instead of standard error

#### Scenario: Customize grid layout
- **GIVEN** depth profile data with 20 genotypes
- **WHEN** called with `col_wrap=5` and `height=3`
- **THEN** create 5-column grid with 3-inch subplot height

#### Scenario: Save plot to file
- **GIVEN** depth profile data
- **WHEN** called with `output_path='depth_profile.png'`
- **THEN** save figure to specified path
- **AND** still return Figure object

#### Scenario: Rotate x-axis labels
- **GIVEN** depth profile plot
- **THEN** x-axis labels (depth values) MUST be rotated 90 degrees for readability

#### Scenario: Enable grid lines
- **GIVEN** depth profile plot
- **THEN** grid lines MUST be visible on all subplots

### Requirement: Replicate-Level Depth Profile Visualization
The system SHALL create spaghetti plots showing individual biological replicate depth profiles.

#### Scenario: Create replicate spaghetti plot
- **GIVEN** aggregated data with multiple replicates per genotype
- **WHEN** `plot_depth_profile_replicates()` is called with `hue='plot_rep'`
- **THEN** return matplotlib Figure with individual lines for each replicate
- **AND** legend is suppressed (too many replicates)
- **AND** lines have transparency for overlapping visibility

#### Scenario: Show raw variability
- **GIVEN** replicate-level data
- **WHEN** plotting
- **THEN** use `estimator=None` to show individual lines without aggregation

#### Scenario: Match faceting with mean plot
- **GIVEN** same data used for faceted mean plot
- **WHEN** called with same `col_wrap` and `height`
- **THEN** produce subplot layout matching the mean plot for easy comparison

### Requirement: Data Filtering Utility
The system SHALL provide a utility to filter DataFrame rows by column values.

#### Scenario: Filter rows where column value is in list
- **GIVEN** a DataFrame and a list of values to filter
- **WHEN** `filter_rows_by_values(df, 'geno', ['GH_7386', 'GH_7418'])` is called
- **THEN** return DataFrame excluding rows where `geno` is in the list

#### Scenario: Invert filter to keep specified values
- **GIVEN** a DataFrame and a list of values
- **WHEN** `filter_rows_by_values(df, 'geno', ['Control'], invert=True)` is called
- **THEN** return DataFrame containing only rows where `geno='Control'`

#### Scenario: Filter with empty list
- **GIVEN** an empty list of values
- **WHEN** filtering
- **THEN** return DataFrame unchanged

#### Scenario: Filter non-existent column
- **GIVEN** a DataFrame without column `'missing_col'`
- **WHEN** `filter_rows_by_values(df, 'missing_col', ['value'])` is called
- **THEN** raise `KeyError` with clear message

### Requirement: Test Data Fixture
The system SHALL provide test fixtures for root core data in expected format.

#### Scenario: Create minimal root core dataset
- **WHEN** `create_test_root_core_data()` is called
- **THEN** return DataFrame with columns `['Plot', 'geno', 'Rep', 'core_n', 'c_0_10_1', 'c_0_10_2', 'c_10_20_1', 'c_10_20_2']`
- **AND** at least 2 genotypes with 3 cores each

#### Scenario: Fixture includes depth columns in correct format
- **GIVEN** test fixture data
- **THEN** depth column names MUST match pattern `c_<start>_<end>_<subcore>`
- **AND** start < end for all depth ranges
- **AND** subcore index in {1, 2}

#### Scenario: Fixture has known aggregation values
- **GIVEN** test fixture with controlled values
- **WHEN** aggregating cores within a plot
- **THEN** produce known expected mean values for test assertions