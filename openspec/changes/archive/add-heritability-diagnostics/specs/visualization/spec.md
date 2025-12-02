# Visualization - Heritability Diagnostics

## ADDED Requirements

### Requirement: Variance Component Decomposition Plot
The system SHALL provide a function `create_variance_decomposition_plot()` that visualizes genetic vs environmental variance components across multiple traits.

#### Scenario: Create 4-panel variance decomposition figure
- **GIVEN** a comparison DataFrame from `compare_trait_heritabilities()`
- **WHEN** `create_variance_decomposition_plot()` is called
- **THEN** return matplotlib Figure with 4 subplots:
  - Panel 1: Bar plot of heritability (H²) values with threshold reference line
  - Panel 2: Stacked or side-by-side bars of genetic vs residual variance
  - Panel 3: Bar plot of percentage variance between genotypes
  - Panel 4: Combined plot of sample sizes and coefficient of variation
- **AND** all subplots share x-axis with trait names

#### Scenario: Customize appearance with optional parameters
- **GIVEN** comparison DataFrame
- **WHEN** `create_variance_decomposition_plot()` is called with `figsize=(14, 10)` and `colors={'h2': 'steelblue', 'var_g': 'green'}`
- **THEN** create figure with specified size
- **AND** use specified colors for plot elements

#### Scenario: Save plot to file
- **GIVEN** comparison DataFrame
- **WHEN** `create_variance_decomposition_plot()` is called with `output_path="diagnostics/variance_plot.png"`
- **THEN** save figure to specified path with high DPI (300)
- **AND** return the Figure object for display

#### Scenario: Handle empty comparison data
- **GIVEN** empty comparison DataFrame
- **WHEN** `create_variance_decomposition_plot()` is called
- **THEN** create figure with placeholder message "No data available"
- **AND** do not raise exception

### Requirement: Trait-by-Genotype Boxplot Visualization
The system SHALL provide a function `create_trait_by_genotype_boxplots()` that displays trait distributions by genotype with heritability annotations.

#### Scenario: Create multi-panel boxplot figure
- **GIVEN** DataFrame with multiple traits and genotype column
- **AND** heritability results for those traits
- **WHEN** `create_trait_by_genotype_boxplots()` is called with list of 2 traits
- **THEN** return Figure with 1 row and 2 columns of boxplots
- **AND** each subplot shows trait values grouped by genotype
- **AND** subplot titles include trait name and H² value

#### Scenario: Annotate boxplots with variance information
- **GIVEN** traits with calculated heritability
- **WHEN** `create_trait_by_genotype_boxplots()` is called with `show_variance=True`
- **THEN** include variance component values (σ²_G, σ²_E) in subplot titles
- **AND** format values with appropriate precision (4 decimal places)

#### Scenario: Handle large number of genotypes with rotated labels
- **GIVEN** DataFrame with >10 genotypes
- **WHEN** `create_trait_by_genotype_boxplots()` is called
- **THEN** rotate x-axis labels to 45 or 90 degrees
- **AND** adjust layout to prevent label overlap

#### Scenario: Handle traits with missing data
- **GIVEN** trait with some NaN values
- **WHEN** `create_trait_by_genotype_boxplots()` is called
- **THEN** exclude NaN values from boxplots
- **AND** show boxplots only for genotypes with valid data

#### Scenario: Customize layout for publication
- **GIVEN** list of traits to visualize
- **WHEN** `create_trait_by_genotype_boxplots()` is called with `ncols=3` and `figsize=(15, 5)`
- **THEN** arrange boxplots in specified grid layout
- **AND** create figure with specified dimensions

### Requirement: Comprehensive Diagnostic Dashboard
The system SHALL provide a function `create_heritability_diagnostic_dashboard()` that combines variance decomposition and trait distributions into a comprehensive diagnostic visualization.

#### Scenario: Create integrated diagnostic dashboard
- **GIVEN** DataFrame with traits, comparison results, and heritability results
- **WHEN** `create_heritability_diagnostic_dashboard()` is called
- **THEN** return Figure combining:
  - Top section: 4-panel variance decomposition plot (from `create_variance_decomposition_plot()`)
  - Bottom section: Trait-by-genotype boxplots (from `create_trait_by_genotype_boxplots()`)
- **AND** use shared layout and color scheme

#### Scenario: Dashboard with vertical layout
- **GIVEN** diagnostic data for multiple traits
- **WHEN** `create_heritability_diagnostic_dashboard()` is called with `layout="vertical"`
- **THEN** stack variance plots above boxplots
- **AND** adjust figure size for readability

#### Scenario: Dashboard with horizontal layout
- **GIVEN** diagnostic data for two traits
- **WHEN** `create_heritability_diagnostic_dashboard()` is called with `layout="horizontal"`
- **THEN** place variance plots on left and boxplots on right
- **AND** balance subplot sizes

#### Scenario: Save dashboard with metadata
- **GIVEN** complete diagnostic data
- **WHEN** `create_heritability_diagnostic_dashboard()` is called with `output_path` and `metadata={"dataset": "Turface_2024"}`
- **THEN** save figure with high resolution
- **AND** include metadata in figure title or annotation

### Requirement: Visualization Consistency with Existing Functions
The diagnostic visualization functions SHALL maintain consistency with existing visualization patterns in the package.

#### Scenario: Use existing matplotlib/seaborn style
- **WHEN** any diagnostic plot function is called
- **THEN** apply same style settings as existing plots in visualization.py
- **AND** use consistent color palettes (not introduce new color schemes without reason)

#### Scenario: Return matplotlib Figure objects
- **WHEN** any diagnostic plot function is called
- **THEN** return matplotlib Figure object (not Axes)
- **AND** allow caller to further customize or display

#### Scenario: Support optional save paths
- **WHEN** any diagnostic plot function is called with `output_path` parameter
- **THEN** save figure to specified path
- **AND** create parent directories if they don't exist
- **AND** return Figure object for further use

#### Scenario: Handle font configuration
- **GIVEN** system with limited font availability
- **WHEN** diagnostic plot functions are called
- **THEN** use robust font fallbacks (same as existing visualization functions)
- **AND** do not fail if preferred fonts unavailable
