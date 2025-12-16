# Depth Profile Visualization Specification

## ADDED Requirements

### Requirement: Biomass Depth Barplot Generation
The system SHALL provide a grouped barplot visualization for biomass depth profile data showing mean biomass at discrete depth intervals per genotype.

#### Scenario: Generate biomass barplot for two depth intervals
- **GIVEN** aggregated biomass data with columns `['geno', 'Depth_cm', 'Root_DW_g']`
- **AND** depth intervals are `[15, 45]` cm (representing 0-30cm and 30-60cm layers)
- **WHEN** `plot_biomass_depth_barplot()` is called
- **THEN** a FacetGrid barplot is generated with:
  - X-axis: Genotypes (sorted by mean biomass in shallowest layer, Control first if present)
  - Y-axis: Mean biomass value
  - Hue: Depth interval (grouped bars, one per depth)
  - Error bars: Standard error across biological replicates
  - Optional stripplot overlay showing individual plot-level data points
- **AND** the figure is saved to the specified output path

#### Scenario: Skip barplot for counting data
- **GIVEN** a root core data source with `data_type = "counting"`
- **WHEN** Step 00f (VisualizeDepthProfilesStep) processes the counting source
- **THEN** only line plots (mean ± SE and spaghetti) are generated
- **AND** no barplot is created (barplots only apply to biomass with discrete depth layers)

#### Scenario: Generate barplot alongside line plots for biomass
- **GIVEN** a root core data source with `data_type = "biomass"`
- **AND** Step 00d has generated `00c_root_core_biomass_aggregated.csv`
- **WHEN** Step 00f (VisualizeDepthProfilesStep) executes
- **THEN** three plots are generated:
  1. `00f_depth_profile_biomass_mean.png` - Rotated line plot (mean ± SE)
  2. `00f_depth_profile_biomass_reps.png` - Rotated spaghetti plot (replicates)
  3. `00f_depth_profile_biomass_barplot.png` - Grouped barplot (NEW)
- **AND** metadata JSON includes all three plot paths

### Requirement: Depth Interval Detection
The system SHALL automatically detect discrete depth intervals from biomass data to determine bar groupings.

#### Scenario: Detect two depth intervals from biomass cores
- **GIVEN** biomass data with `Depth_cm` values `[15, 45]`
- **WHEN** preparing data for barplot
- **THEN** depth intervals are identified as `['0-30cm', '30-60cm']` based on unique depth values
- **AND** these intervals are used as hue categories in the grouped barplot

#### Scenario: Handle variable number of depth intervals
- **GIVEN** biomass data with 3 depth intervals: `Depth_cm = [15, 30, 45]`
- **WHEN** `plot_biomass_depth_barplot()` is called
- **THEN** the barplot displays 3 grouped bars per genotype
- **AND** legend shows all 3 depth interval labels

### Requirement: Genotype Ordering in Barplots
The system SHALL order genotypes in barplots by ascending mean biomass in the shallowest depth interval, with Control genotype always first.

#### Scenario: Order genotypes by shallow-layer biomass
- **GIVEN** genotypes with mean biomass at 0-30cm: `{Control: 0.29, GH_7440: 0.43, GH_7293: 0.22}`
- **WHEN** generating the barplot
- **THEN** x-axis order is `['Control', 'GH_7293', 'GH_7440']`
- **REASON**: Control first (if present), then ascending by shallowest-layer mean

#### Scenario: Consistent ordering across depth intervals
- **GIVEN** genotype order determined from 0-30cm biomass
- **WHEN** displaying bars for both 0-30cm and 30-60cm depths
- **THEN** the same genotype order applies to both depth interval bar groups
- **AND** genotypes are not re-sorted per depth interval

### Requirement: Barplot Visual Styling
The system SHALL style biomass barplots consistently with existing depth profile visualizations using seaborn dark grid theme.

#### Scenario: Apply consistent styling to barplot
- **WHEN** generating a biomass barplot
- **THEN** the following styling is applied:
  - Theme: `sns.set_theme(style="dark")` with `darkgrid`
  - Figure size: 12" width × 7" height (accommodates many genotypes)
  - X-axis labels rotated 90° for readability
  - Y-axis label: `"{value_col} (Mean ± SE)"` (e.g., "Root DW G (Mean ± SE)")
  - Grid: Enabled on y-axis
  - DPI: 300 for publication quality
  - Background: White (`facecolor="white"`)
- **AND** styling matches the aesthetic of existing line plots (except no rotation for barplots)

#### Scenario: Include stripplot overlay for individual points
- **GIVEN** plot-level biomass data (multiple biological replicates per genotype)
- **WHEN** generating barplot with `include_points=True`
- **THEN** individual plot values are overlaid as semi-transparent points (alpha=0.5)
- **AND** points are jittered horizontally within each bar group for visibility
- **AND** point color is dark (`palette='dark:k'`) to contrast with bar colors
