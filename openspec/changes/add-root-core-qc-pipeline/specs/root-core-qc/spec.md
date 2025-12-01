# Root Core QC Pipeline Specification

## ADDED Requirements

### Requirement: Root Core Data Loading
The system SHALL load root core experimental data (biomass and counting) from CSV files with proper validation.

#### Scenario: Load biomass data with required columns
- **GIVEN** a CSV file with columns `Plot`, `Rep`, `geno`, `Core_Replicate`, `0-30`, `30-60`
- **WHEN** `LoadRootCoreDataStep` executes with biomass configuration
- **THEN** load DataFrame successfully
- **AND** validate all required columns exist
- **AND** create sample identifiers using `create_sample_identifier()`

#### Scenario: Load counting data with depth pattern columns
- **GIVEN** a CSV file with columns `Plot`, `Rep`, `geno`, `core_n`, `c_0_10_1`, `c_0_10_2`, etc.
- **WHEN** `LoadRootCoreDataStep` executes with counting configuration
- **THEN** load DataFrame successfully
- **AND** validate depth column pattern matches `c_<start>_<end>_<subcore>`

#### Scenario: Handle missing required columns
- **GIVEN** a CSV file missing the `Rep` column
- **WHEN** `LoadRootCoreDataStep` attempts to load
- **THEN** raise `ValueError` with message listing missing columns

#### Scenario: Load multiple root core sources
- **GIVEN** configuration with both biomass and counting sources
- **WHEN** `LoadRootCoreDataStep` executes
- **THEN** load both DataFrames independently
- **AND** output separate `biomass_df` and `counting_df`

###Requirement: Depth Data Transformation to Long Format
The system SHALL transform wide-format depth data to long format with automatic or manual depth calculation.

#### Scenario: Transform biomass with manual depth mapping
- **GIVEN** biomass DataFrame with columns `0-30`, `30-60`
- **AND** depth mapping `{"0-30": 15.0, "30-60": 45.0}`
- **WHEN** `TransformDepthDataStep` executes
- **THEN** melt to long format using `melt_depth_data()`
- **AND** output has columns: `Plot`, `Rep`, `geno`, `core_n`, `Depth_cm`, `Root_DW_g`
- **AND** `Depth_cm` values are 15.0 and 45.0

#### Scenario: Transform counting with automatic depth parsing
- **GIVEN** counting DataFrame with columns `c_0_10_1`, `c_0_10_2`, `c_10_20_1`, etc.
- **WHEN** `TransformDepthDataStep` executes with `parse_depth=True`
- **THEN** melt to long format
- **AND** `Depth_cm` calculated from column name pattern
- **AND** `c_0_10_1` → 0.0 cm, `c_0_10_2` → 5.0 cm, `c_10_20_1` → 10.0 cm

#### Scenario: Preserve metadata columns during transformation
- **GIVEN** DataFrame with metadata columns `Cid`, `Sid`, `GID`, `Cross name`
- **WHEN** `TransformDepthDataStep` executes
- **THEN** all metadata columns present in long-format output

### Requirement: Core-Level Quality Control
The system SHALL detect and optionally remove outlier individual cores within biological replicates.

#### Scenario: Detect outlier core within Plot-Rep group
- **GIVEN** Plot 1 Rep 1 has 3 cores with depth profiles
- **AND** core 3 has extreme values (Mahalanobis distance > threshold)
- **WHEN** `QCCoreLevelStep` executes with `contamination=0.1`
- **THEN** flag core 3 as outlier
- **AND** add outlier flag to DataFrame

#### Scenario: Remove outlier cores before aggregation
- **GIVEN** core-level QC detected 2 outlier cores
- **AND** configuration has `remove_outliers=True`
- **WHEN** `QCCoreLevelStep` executes
- **THEN** remove flagged cores from DataFrame
- **AND** log removed core IDs: `["plot1_rep1_GH_7386_core3", "plot5_rep2_GH_7420_core1"]`

#### Scenario: Flag cores with excessive missing data
- **GIVEN** a core has 8 NaN values out of 12 depths (67%)
- **AND** configuration has `max_missing_proportion=0.5`
- **WHEN** `QCCoreLevelStep` executes
- **THEN** flag core for missing data
- **AND** include reason: `"missing_data_0.67"`

#### Scenario: Generate spaghetti plot colored by outlier status
- **GIVEN** core-level QC flagged 3 outliers
- **WHEN** `QCCoreLevelStep` generates visualizations
- **THEN** create spaghetti plot with all cores
- **AND** color outlier cores differently (e.g., red)
- **AND** save plot to output directory

### Requirement: Core Aggregation to Biological Replicates
The system SHALL aggregate technical replicates (cores) to biological replicate level.

#### Scenario: Aggregate 3 cores per Plot-Rep using mean
- **GIVEN** long-format data with 3 cores per Plot-Rep-Depth combination
- **WHEN** `AggregateCoresStep` executes with `agg_func='mean'`
- **THEN** use `aggregate_by_replicate()` function
- **AND** group by `['Plot', 'Rep', 'geno', 'Depth_cm']`
- **AND** output has one row per Plot-Rep-Depth

#### Scenario: Aggregate using median
- **GIVEN** configuration specifies `aggregation_method='median'`
- **WHEN** `AggregateCoresStep` executes
- **THEN** use median for aggregation
- **AND** handle NaN values correctly (skip in median calculation)

#### Scenario: Track number of cores aggregated
- **GIVEN** Plot 1 Rep 1 has 3 cores, Plot 2 Rep 1 has 2 cores (one removed by QC)
- **WHEN** `AggregateCoresStep` executes
- **THEN** track N cores per group
- **AND** include in metadata: `{"plot1_rep1": 3, "plot2_rep1": 2}`

### Requirement: Pre-QC Depth Profile Visualization
The system SHALL generate depth profile visualizations before trait-level QC.

#### Scenario: Generate bar plot for biomass (2 depths)
- **GIVEN** aggregated biomass data with 2 depths
- **WHEN** `VisualizeDepthProfilesPreQCStep` executes
- **THEN** create bar plot using matplotlib
- **AND** facet by genotype
- **AND** save as `biomass_depth_profile_pre_qc.png`

#### Scenario: Generate line plot for counting (12 depths)
- **GIVEN** aggregated counting data with 12 depths
- **WHEN** `VisualizeDepthProfilesPreQCStep` executes
- **THEN** create line plot using `plot_depth_profile_faceted()`
- **AND** show mean with error bars (standard error)
- **AND** save as `counting_depth_profile_pre_qc.png`

#### Scenario: Generate replicate spaghetti plots
- **GIVEN** aggregated data
- **WHEN** `VisualizeDepthProfilesPreQCStep` executes
- **THEN** create spaghetti plots using `plot_depth_profile_replicates()`
- **AND** show individual Plot-Rep lines colored by replicate
- **AND** facet by genotype

### Requirement: Reshape for Trait-Level QC with Prefixing
The system SHALL pivot long-format data to wide format with column prefixes to prevent duplicates.

#### Scenario: Reshape biomass with RootDW prefix
- **GIVEN** long-format biomass data with `Depth_cm` values [15.0, 45.0]
- **AND** configuration specifies `depth_column_prefix='RootDW'`
- **WHEN** `ReshapeForTraitQCStep` executes
- **THEN** pivot to wide format
- **AND** column names are `RootDW_15cm`, `RootDW_45cm`
- **AND** one row per Plot-Rep

#### Scenario: Reshape counting with RootCount prefix
- **GIVEN** long-format counting data with 12 depth values
- **AND** configuration specifies `depth_column_prefix='RootCount'`
- **WHEN** `ReshapeForTraitQCStep` executes
- **THEN** pivot to wide format
- **AND** column names are `RootCount_0cm`, `RootCount_5cm`, ..., `RootCount_55cm`

#### Scenario: Preserve metadata columns during reshape
- **GIVEN** long-format data with metadata columns `Cid`, `Sid`, `GID`
- **WHEN** `ReshapeForTraitQCStep` executes
- **THEN** all metadata columns present in wide-format output

### Requirement: Integration with Existing QC Pipeline
The system SHALL pass reshaped root data through existing trait-level QC steps.

#### Scenario: Root data flows through QC Steps 1-10
- **GIVEN** wide-format root data from `ReshapeForTraitQCStep`
- **WHEN** QC Pipeline executes Steps 1-10
- **THEN** treat each depth as independent trait
- **AND** apply cleanup, outlier detection, statistical analysis, heritability filtering
- **AND** output QC-cleaned root data

#### Scenario: Filter low-heritability depths
- **GIVEN** QC Step 9 calculates heritability by depth
- **AND** `RootDW_45cm` has heritability = 0.15
- **AND** threshold = 0.3
- **WHEN** filtering executes
- **THEN** remove `RootDW_45cm` column
- **AND** log: `{"trait": "RootDW_45cm", "reason": "heritability 0.15 < 0.30"}`

### Requirement: Above-Ground Trait Loading and Validation
The system SHALL load above-ground trait data and validate compatibility with root data.

#### Scenario: Load above-ground traits successfully
- **GIVEN** CSV with columns `Plot`, `Rep`, `geno`, `Boot_dtoInit_day`, `GY_Calc_gm2`, etc.
- **WHEN** `LoadAboveGroundTraitsStep` executes
- **THEN** load DataFrame successfully
- **AND** validate join keys exist: `Plot`, `Rep`, `geno`

#### Scenario: Detect missing join keys
- **GIVEN** above-ground CSV missing `Rep` column
- **WHEN** `LoadAboveGroundTraitsStep` attempts validation
- **THEN** raise `ValueError` with message: `"Missing required join key: Rep"`

#### Scenario: Check for column name conflicts
- **GIVEN** root data has column `RootDW_15cm`
- **AND** above-ground data has column `RootDW_15cm`
- **WHEN** `LoadAboveGroundTraitsStep` validates
- **THEN** detect duplicate column name
- **AND** store conflict: `["RootDW_15cm"]`

### Requirement: Merge All Trait Sources Without Duplicates
The system SHALL merge root and above-ground traits with duplicate detection and handling.

#### Scenario: Successful merge with no duplicates
- **GIVEN** root data with columns `RootDW_15cm`, `RootCount_0cm`, etc.
- **AND** above-ground data with columns `GY_Calc_gm2`, `PH_M_cm`, etc.
- **AND** no overlapping column names
- **WHEN** `MergeAllTraitsStep` executes with `duplicate_strategy='fail'`
- **THEN** perform inner join on `Plot`, `Rep`, `geno`
- **AND** output merged DataFrame with all columns

#### Scenario: Fail on duplicate column detection
- **GIVEN** both datasets have column `BM_Calc_gm2`
- **AND** configuration has `duplicate_strategy='fail'`
- **WHEN** `MergeAllTraitsStep` attempts merge
- **THEN** raise `ValueError` with message: `"Duplicate column found: BM_Calc_gm2. Use duplicate_strategy='suffix' or rename columns."`

#### Scenario: Handle duplicates with suffix strategy
- **GIVEN** both datasets have column `BM_Calc_gm2`
- **AND** configuration has `duplicate_strategy='suffix'`
- **WHEN** `MergeAllTraitsStep` executes
- **THEN** rename to `BM_Calc_gm2_root` and `BM_Calc_gm2_ag`
- **AND** complete merge successfully

#### Scenario: Validate sample sizes before/after merge
- **GIVEN** root data has 60 samples
- **AND** above-ground data has 58 samples (2 missing)
- **WHEN** `MergeAllTraitsStep` performs inner join
- **THEN** output has 58 samples
- **AND** warn: `"Lost 2 samples during merge. Missing in above-ground data: [plot5_rep1_GH_7299, plot12_rep3_GH_7386]"`

### Requirement: Comprehensive Metadata Generation
The system SHALL generate JSON metadata files documenting all processing steps and decisions.

#### Scenario: Generate metadata for final merged CSV
- **GIVEN** pipeline completed all steps
- **WHEN** `GenerateRootCoreMetadataStep` executes
- **THEN** create JSON with sections: `data_sources`, `processing_steps`, `qc_decisions`, `final_summary`, `software_versions`

#### Scenario: Log data source provenance
- **GIVEN** loaded 3 CSV files
- **WHEN** generating metadata
- **THEN** include for each source: file path, timestamp, checksum, row count, column count

#### Scenario: Log QC decisions
- **GIVEN** pipeline removed 5 outlier cores and 2 outlier replicates
- **WHEN** generating metadata
- **THEN** include lists: `cores_removed` with IDs and reasons, `replicates_removed` with IDs and reasons

#### Scenario: Log trait filtering decisions
- **GIVEN** pipeline removed 3 low-heritability traits
- **WHEN** generating metadata
- **THEN** include: `{"trait": "RootDW_45cm", "reason": "heritability 0.15 < 0.30"}` for each

#### Scenario: Save metadata alongside output CSV
- **GIVEN** output file is `Field_2024_final.csv`
- **WHEN** `GenerateRootCoreMetadataStep` executes
- **THEN** save metadata as `Field_2024_final_metadata.json` in same directory

### Requirement: Post-QC Visualization and Reporting
The system SHALL generate comprehensive visualizations comparing pre/post QC results.

#### Scenario: Regenerate depth profiles with cleaned data
- **GIVEN** QC-cleaned root data
- **WHEN** `VisualizeDepthProfilesPostQCStep` executes
- **THEN** create depth profile plots
- **AND** save as `biomass_depth_profile_post_qc.png`, `counting_depth_profile_post_qc.png`

#### Scenario: Generate pre/post comparison plots
- **GIVEN** pre-QC plots and post-QC data
- **WHEN** `VisualizeDepthProfilesPostQCStep` executes
- **THEN** create side-by-side comparison figures
- **AND** annotate with sample sizes: "N=60 (pre) vs N=58 (post)"

#### Scenario: Generate cross-correlation heatmap
- **GIVEN** merged data with root and above-ground traits
- **WHEN** `VisualizeDepthProfilesPostQCStep` executes
- **THEN** calculate correlations between root depths and above-ground traits
- **AND** create heatmap using seaborn
- **AND** save as `root_aboveground_correlations.png`

#### Scenario: Generate heritability by depth bar plot
- **GIVEN** heritability values calculated for each depth
- **WHEN** `VisualizeDepthProfilesPostQCStep` executes
- **THEN** create bar plot with heritability on y-axis, depth on x-axis
- **AND** add horizontal line at threshold (0.3)
- **AND** color bars by filtered status (red if filtered, green if kept)

#### Scenario: Compile HTML visualization report
- **GIVEN** all generated plots
- **WHEN** `VisualizeDepthProfilesPostQCStep` completes
- **THEN** generate HTML file with embedded plots
- **AND** include summary statistics tables
- **AND** save as `root_core_qc_report.html`

### Requirement: Pipeline Task Graph Integration
The system SHALL integrate root core steps into QC Pipeline task graph with proper dependencies.

#### Scenario: Conditional root core step execution
- **GIVEN** `QCPipelineConfig` with `root_core=None`
- **WHEN** `QCPipeline.create_tasks()` executes
- **THEN** skip root core steps (0a-0f)
- **AND** proceed directly to Step 1 (LoadData)

#### Scenario: Root core steps execute before trait-level QC
- **GIVEN** `QCPipelineConfig` with `root_core` configured
- **WHEN** `QCPipeline.create_tasks()` executes
- **THEN** create tasks in order: Steps 0a-0f, then Steps 1-10
- **AND** Step 1 depends on Step 0f

#### Scenario: Multiple root core sources processed in parallel
- **GIVEN** configuration with both biomass and counting sources
- **WHEN** pipeline executes Steps 0a-0e
- **THEN** process both sources independently (can parallelize)
- **AND** merge in Step 0f before trait-level QC

### Requirement: Configuration Validation
The system SHALL validate root core configuration before pipeline execution.

#### Scenario: Validate required configuration fields
- **GIVEN** `RootCoreConfig` missing `csv_path`
- **WHEN** validation executes
- **THEN** raise `ValidationError` with message: `"RootCoreSourceConfig.csv_path is required"`

#### Scenario: Validate depth mapping for biomass
- **GIVEN** biomass source with `data_type='biomass'` but no `depth_mapping`
- **WHEN** validation executes
- **THEN** raise `ValidationError`: `"Biomass sources require depth_mapping"`

#### Scenario: Validate aggregation method
- **GIVEN** `aggregation_method='invalid'`
- **WHEN** validation executes
- **THEN** raise `ValidationError`: `"aggregation_method must be 'mean', 'median', or callable"`

#### Scenario: Validate file paths exist
- **GIVEN** `csv_path='nonexistent.csv'`
- **WHEN** validation executes
- **THEN** raise `FileNotFoundError`: `"CSV file not found: nonexistent.csv"`
