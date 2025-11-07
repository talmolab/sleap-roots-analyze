# Specification: Regression Visualization

## Overview

This specification defines the regression plotting capability for creating publication-quality bivariate linear regression plots with statistical annotations.

## ADDED Requirements

### Requirement: Basic Regression Plot Generation

The system SHALL provide a function to create linear regression plots for analyzing relationships between two continuous variables.

#### Scenario: Simple regression plot

- **GIVEN** a DataFrame with numeric trait columns "Root Biomass (mg)" and "Surface Area (mm²)"
- **WHEN** user calls `create_regression_plot(df, x_col='Surface Area (mm²)', y_col='Root Biomass (mg)')`
- **THEN** function returns a matplotlib Figure
- **AND** figure contains scatter points for all samples
- **AND** figure contains a linear regression line
- **AND** figure contains a shaded confidence interval (95%)
- **AND** axes are labeled with column names
- **AND** plot title is auto-generated from column names

#### Scenario: Customized figure size

- **GIVEN** a DataFrame with trait data
- **WHEN** user specifies `figsize=(10, 8)`
- **THEN** function returns a Figure with dimensions 10x8 inches
- **AND** all plot elements scale appropriately

### Requirement: Statistical Annotations

The system SHALL calculate and display regression statistics on the plot.

#### Scenario: Pearson correlation display

- **GIVEN** a DataFrame with two correlated traits
- **WHEN** regression plot is generated
- **THEN** plot contains text annotation with Pearson R value
- **AND** plot contains R² (coefficient of determination)
- **AND** plot contains p-value for correlation test
- **AND** all statistical values are formatted to 3 decimal places
- **AND** annotations are positioned to avoid overlapping data points

#### Scenario: Regression equation display

- **GIVEN** a DataFrame with two traits
- **WHEN** regression plot is generated
- **THEN** plot contains the regression equation in format "y = mx + b"
- **AND** slope (m) and intercept (b) are formatted appropriately
- **AND** equation is positioned near the regression line or in legend

#### Scenario: Statistical significance indication

- **GIVEN** a regression with p-value < 0.001
- **WHEN** plot is generated
- **THEN** p-value annotation shows "p < 0.001" instead of exact value
- **AND** plot visually indicates significance level (e.g., asterisks or text)

### Requirement: Grouped Regression (Color by Category)

The system SHALL support coloring scatter points by categorical groupings while maintaining a single regression line.

#### Scenario: Color by genotype

- **GIVEN** a DataFrame with columns "Root Biomass (mg)", "Surface Area (mm²)", and "Genotype"
- **WHEN** user calls `create_regression_plot(df, x_col='Surface Area (mm²)', y_col='Root Biomass (mg)', color_by='Genotype')`
- **THEN** scatter points are colored by unique genotype values
- **AND** a single regression line is fitted to all data (not per group)
- **AND** legend shows genotype labels with corresponding colors
- **AND** statistical annotations reflect overall correlation (not per-group)

#### Scenario: Too many categories warning

- **GIVEN** a DataFrame where color_by column has >20 unique values
- **WHEN** regression plot is generated with color_by parameter
- **THEN** function issues a warning about too many categories
- **AND** plot still generates but may have illegible legend
- **OR** function automatically switches to continuous colormap

### Requirement: Data Validation and Quality Checks

The system SHALL validate input data and handle edge cases appropriately.

#### Scenario: Missing column error

- **GIVEN** a DataFrame without column "NonexistentTrait"
- **WHEN** user calls `create_regression_plot(df, x_col='NonexistentTrait', y_col='Root Biomass (mg)')`
- **THEN** function raises ValueError with clear message
- **AND** error message specifies which column is missing

#### Scenario: Non-numeric column error

- **GIVEN** a DataFrame where "Genotype" is a string column
- **WHEN** user attempts regression with x_col='Genotype'
- **THEN** function raises ValueError indicating column must be numeric
- **AND** error message suggests using color_by for categorical variables

#### Scenario: Insufficient samples

- **GIVEN** a DataFrame with only 2 valid samples (after NaN removal)
- **WHEN** regression plot is generated
- **THEN** function raises ValueError indicating minimum 3 samples required
- **AND** error message shows actual sample count

#### Scenario: NaN value handling

- **GIVEN** a DataFrame where 10% of samples have NaN in x_col or y_col
- **WHEN** regression plot is generated
- **THEN** function drops rows with NaN values pairwise
- **AND** function does NOT issue a warning (NaN rate < 20%)
- **AND** plot annotation indicates sample size used (n=valid_count)

#### Scenario: High NaN rate warning

- **GIVEN** a DataFrame where 25% of samples have NaN values
- **WHEN** regression plot is generated
- **THEN** function issues a warning about high NaN rate
- **AND** warning message shows percentage of data dropped
- **AND** plot still generates with remaining valid data

#### Scenario: Perfect correlation (R² = 1.0)

- **GIVEN** a DataFrame where y = 2*x exactly (perfect linear relationship)
- **WHEN** regression plot is generated
- **THEN** R² annotation shows 1.000
- **AND** p-value shows extremely small value or "p < 0.001"
- **AND** all data points fall exactly on regression line

#### Scenario: Zero correlation (R² ≈ 0)

- **GIVEN** a DataFrame with uncorrelated random variables
- **WHEN** regression plot is generated
- **THEN** R² annotation shows value near 0.000
- **AND** p-value shows non-significant result (p > 0.05)
- **AND** regression line is nearly horizontal

#### Scenario: Zero variance in variable

- **GIVEN** a DataFrame where all values in x_col are identical
- **WHEN** regression plot is attempted
- **THEN** function raises ValueError indicating zero variance
- **AND** error message suggests checking data for constant values

### Requirement: Styling and Publication Quality

The system SHALL produce publication-ready plots with consistent styling.

#### Scenario: Default styling matches package standards

- **GIVEN** any valid regression plot
- **WHEN** plot is generated without custom styling
- **THEN** figure DPI is 150 for display (per package defaults)
- **AND** font sizes match existing visualization functions
- **AND** scatter point size is appropriate for typical datasets (30-200 points)
- **AND** regression line is visually distinct from scatter points

#### Scenario: Saving for publication

- **GIVEN** a regression plot Figure object
- **WHEN** user saves with `save_figure_with_unique_name(fig, PUBLICATION_DIR, "regression_biomass")`
- **THEN** saved file has DPI=300 (publication quality)
- **AND** all text elements are legible at publication size
- **AND** file format is PNG by default

#### Scenario: Custom color scheme

- **GIVEN** user preferences for specific color palette
- **WHEN** color_by parameter is used
- **THEN** function uses seaborn default palette or matplotlib tab10
- **AND** colors are colorblind-friendly
- **AND** colors have sufficient contrast with white background

### Requirement: Integration with Existing Workflow

The system SHALL integrate seamlessly with existing notebook and analysis patterns.

#### Scenario: Use in notebook workflow

- **GIVEN** a trait visualization notebook
- **WHEN** user imports `create_regression_plot` from sleap_roots_analyze
- **THEN** function works with existing DataFrame from `load_trait_data()`
- **AND** function works with trait columns from `get_trait_columns()`
- **AND** returned Figure can be displayed inline in Jupyter
- **AND** returned Figure can be saved with existing save utilities

#### Scenario: Return value for further customization

- **GIVEN** a generated regression plot
- **WHEN** function returns matplotlib Figure
- **THEN** user can access fig.axes[0] for customization
- **AND** user can add additional annotations or markers
- **AND** user can modify axis limits or labels
- **AND** modifications don't break the plot layout

## Design Constraints

### Technical Constraints

- Function MUST use scipy.stats.linregress() for regression calculation
- Function MUST use scipy.stats.pearsonr() for correlation statistics
- Function MUST use seaborn.regplot() or equivalent for confidence interval rendering
- Function MUST return matplotlib.figure.Figure (not seaborn JointGrid)
- Function MUST NOT require new package dependencies

### Performance Constraints

- Function SHOULD complete in <1 second for datasets with <1000 points
- Function SHOULD handle up to 10,000 data points without memory issues
- Plot rendering SHOULD be responsive in Jupyter notebooks

### Code Quality Constraints

- Function MUST have Google-style docstring
- Function MUST have type hints for all parameters and return value
- Function MUST have at least 95% test coverage
- All tests MUST complete in <5 seconds total

## Compatibility

### Backward Compatibility

- This is a NEW function, no breaking changes to existing code
- All existing visualization functions remain unchanged
- No changes to existing function signatures

### Forward Compatibility

- Function signature SHOULD allow future parameters (e.g., robust_regression=False)
- Design SHOULD allow future addition of non-linear regression
- Design SHOULD allow future addition of interactive plotly version

## Testing Strategy

### Unit Tests Required

1. Statistical accuracy tests (compare to scipy directly)
2. Plot generation tests (verify Figure components exist)
3. Input validation tests (error cases)
4. Edge case tests (perfect correlation, zero correlation, NaN handling)
5. Styling tests (figure size, colors, labels)
6. Integration tests (workflow with real fixtures)

### Test Data Requirements

- Fixture with positive linear correlation (R² ≈ 0.7-0.9)
- Fixture with negative linear correlation
- Fixture with no correlation (random data)
- Fixture with NaN values (various percentages)
- Fixture with categorical grouping column
- Fixture with perfect linear relationship

### Success Metrics

- >95% code coverage for `create_regression_plot()`
- All statistical calculations match scipy within floating-point precision
- All error cases raise appropriate exceptions with clear messages
- Generated plots pass visual inspection (verified manually with real data)