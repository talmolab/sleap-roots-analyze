# pipeline-runner-skill Specification

## Purpose
TBD - created by archiving change add-pipeline-runner-skill. Update Purpose after archive.
## Requirements
### Requirement: Config Directory Structure
The project SHALL organize pipeline configurations into a structured directory hierarchy separating active configs from examples and templates.

#### Scenario: Active configs in dedicated subdirectories
- **WHEN** user runs pipelines
- **THEN** the system SHALL look for configs in `configs/active/` directory
- **AND** QC configs SHALL be in `configs/active/qc/`
- **AND** Viz configs SHALL be in `configs/active/viz/`
- **AND** Cross-Platform configs SHALL be in `configs/active/cross_platform/`

#### Scenario: Example configs separated
- **WHEN** user wants to reference example configurations
- **THEN** example configs SHALL be available in `configs/examples/`
- **AND** template configs SHALL remain in `configs/templates/`
- **AND** running the pipeline runner SHALL NOT execute example or template configs

### Requirement: Run Manifest File
The system SHALL use a manifest file to specify which configs to run together as a cohesive analysis.

#### Scenario: Manifest defines run configuration
- **WHEN** user creates a run manifest at `configs/active/run_manifest.yaml`
- **THEN** the manifest SHALL define `run_name` for identification
- **AND** the manifest SHALL list `qc_configs` paths relative to `configs/active/`
- **AND** the manifest SHALL list `viz_configs` paths relative to `configs/active/`
- **AND** the manifest SHALL list `cross_platform_configs` paths relative to `configs/active/`

#### Scenario: Manifest validation
- **WHEN** user runs pipelines with a manifest
- **THEN** the system SHALL validate all listed config files exist
- **AND** the system SHALL report missing configs before execution
- **AND** invalid manifests SHALL prevent pipeline execution

### Requirement: Timestamped Run Outputs
The system SHALL organize all pipeline outputs into timestamped run directories for reproducibility and comparison.

#### Scenario: Run output directory created
- **WHEN** pipelines are executed
- **THEN** a timestamped directory SHALL be created: `pipeline_runs/YYYY-MM-DD_HHMMSS/`
- **AND** QC outputs SHALL be placed in `pipeline_runs/.../qc/`
- **AND** Viz outputs SHALL be placed in `pipeline_runs/.../viz/`
- **AND** Cross-Platform outputs SHALL be placed in `pipeline_runs/.../cross_platform/`

#### Scenario: Latest run symlink
- **WHEN** a pipeline run completes
- **THEN** a `latest` symlink SHALL point to the most recent run directory
- **AND** the symlink SHALL be updated after each successful run

### Requirement: Pipeline Execution Order

The system SHALL execute pipelines in the correct order respecting dependencies.

#### Scenario: QC runs before Viz

- **WHEN** user runs all pipelines
- **THEN** all QC pipelines SHALL complete before any Viz pipelines start
- **AND** Viz configs SHALL be updated to reference new QC output paths
- **AND** Cross-Platform configs SHALL be updated to reference new QC output paths

#### Scenario: Path auto-update preserves filename choice

- **WHEN** Viz or Cross-Platform configs reference QC outputs
- **THEN** the system SHALL automatically update the directory portion of paths
- **AND** the system SHALL preserve the original filename from the config (e.g., `07_data_outliers_removed.csv` or `10_final_data.csv`)
- **AND** if original config specifies `07_data_outliers_removed.csv`, the updated path SHALL also use `07_data_outliers_removed.csv`
- **AND** if original config specifies `10_final_data.csv`, the updated path SHALL also use `10_final_data.csv`

#### Scenario: Config structure preserved during update

- **WHEN** config files are updated with new QC output paths
- **THEN** the system SHALL preserve all YAML comments in the config file
- **AND** the system SHALL preserve the original key ordering
- **AND** the system SHALL preserve string quoting styles
- **AND** the system SHALL preserve blank lines and formatting

#### Scenario: Cross-platform config with mixed file choices

- **GIVEN** a cross-platform config where `exp1_data_path` uses `07_data_outliers_removed.csv`
- **AND** `exp2_data_path` uses `10_final_data.csv`
- **WHEN** the config is updated with new QC output paths
- **THEN** `exp1_data_path` SHALL be updated to the new directory with `07_data_outliers_removed.csv`
- **AND** `exp2_data_path` SHALL be updated to the new directory with `10_final_data.csv`

### Requirement: Run Summary Generation

The system SHALL generate a comprehensive markdown summary document after each run.

#### Scenario: Summary document created

- **WHEN** pipeline run completes
- **THEN** a `SUMMARY.md` file SHALL be created in the run directory
- **AND** the summary SHALL include generation timestamp
- **AND** the summary SHALL include git commit hash
- **AND** the summary SHALL include manifest file reference
- **AND** the file SHALL be written with UTF-8 encoding to properly display Unicode characters (H², mm², etc.)

#### Scenario: QC results in summary

- **WHEN** QC pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Dataset, Samples, Traits, Genotypes, H² Threshold, Mean H², Status, Run Path
- **AND** for each QC run, the summary SHALL read `10_pipeline_summary.json` to extract scientific metrics
- **AND** the summary SHALL include a "Removed Traits" subsection listing traits filtered by heritability threshold for each dataset
- **AND** each removed trait SHALL display its heritability value in parentheses (e.g., "Depth (mm) (H²=0.32)")

#### Scenario: QC results with failed pipeline

- **WHEN** a QC pipeline fails during execution
- **THEN** the summary table SHALL show "Failed" status for that config
- **AND** numeric columns (Samples, Traits, etc.) SHALL display "N/A"
- **AND** other successful QC runs SHALL still display full metrics

#### Scenario: Viz results in summary

- **WHEN** Viz pipelines complete
- **THEN** the summary SHALL include a table with columns: Dataset, Figures Generated, Interactive Plots, Status, Time, Run Path
- **AND** the summary SHALL count static figures from the `static_figures/` directory
- **AND** the summary SHALL count interactive plots from the `pca/` and `umap/` directories

#### Scenario: Cross-Platform results in summary

- **WHEN** Cross-Platform pipelines complete successfully
- **THEN** the summary SHALL include a basic table with columns: Comparison, Common Genotypes, Exp1 Samples, Exp2 Samples, Top Correlation, Status, Run Path
- **AND** the summary SHALL include a detailed "## Detailed Cross-Platform Analysis" section
- **AND** the detailed section SHALL include trait reduction statistics, correlation counts, significance counts, power analysis, and top correlations for each comparison

#### Scenario: Cross-Platform results with missing data

- **WHEN** Cross-Platform pipeline completes but correlation CSV is missing
- **THEN** the summary SHALL display "N/A" for detailed statistics
- **AND** the summary SHALL still display status and run path
- **AND** a warning SHALL be included noting missing data

### Requirement: CLI Command Interface
The system SHALL provide a CLI command `sleap-roots-analyze run-all` for executing the pipeline runner.

#### Scenario: Basic CLI invocation
- **WHEN** user runs `sleap-roots-analyze run-all`
- **THEN** the system SHALL read the default manifest at `configs/active/run_manifest.yaml`
- **AND** the system SHALL execute all pipelines listed in the manifest
- **AND** the system SHALL generate the summary document

#### Scenario: Custom manifest via CLI
- **WHEN** user runs `sleap-roots-analyze run-all --manifest path/to/manifest.yaml`
- **THEN** the system SHALL use the specified manifest file
- **AND** all other behavior SHALL remain the same

#### Scenario: Custom output directory via CLI
- **WHEN** user runs `sleap-roots-analyze run-all --output custom_dir/`
- **THEN** the system SHALL create timestamped runs in the specified directory
- **AND** the `latest` symlink SHALL be created in the specified directory

#### Scenario: CLI dry-run mode
- **WHEN** user runs `sleap-roots-analyze run-all --dry-run`
- **THEN** the system SHALL validate the manifest and configs
- **AND** the system SHALL display what would be executed
- **AND** no pipelines SHALL actually run

#### Scenario: CLI QC-only mode
- **WHEN** user runs `sleap-roots-analyze run-all --qc-only`
- **THEN** only QC pipelines SHALL be executed
- **AND** Viz and Cross-Platform pipelines SHALL be skipped
- **AND** summary SHALL reflect partial run

#### Scenario: CLI viz-only mode
- **WHEN** user runs `sleap-roots-analyze run-all --viz-only`
- **THEN** the system SHALL require existing QC outputs to be specified or discovered
- **AND** only Viz pipelines SHALL be executed
- **AND** summary SHALL reflect partial run

#### Scenario: CLI cross-only mode
- **WHEN** user runs `sleap-roots-analyze run-all --cross-only`
- **THEN** the system SHALL require existing QC outputs to be specified or discovered
- **AND** only Cross-Platform pipelines SHALL be executed
- **AND** summary SHALL reflect partial run

#### Scenario: CLI verbose mode
- **WHEN** user runs `sleap-roots-analyze run-all -v` or `--verbose`
- **THEN** the system SHALL display detailed progress information
- **AND** individual pipeline outputs SHALL be shown

### Requirement: Slash Command Interface
The system SHALL provide a Claude Code slash command `/run-pipelines` for executing the pipeline runner.

#### Scenario: Basic invocation
- **WHEN** user invokes `/run-pipelines`
- **THEN** the system SHALL call the CLI command internally
- **AND** the system SHALL use TodoWrite to track pipeline progress
- **AND** the system SHALL generate the summary document

#### Scenario: Custom manifest
- **WHEN** user invokes `/run-pipelines --manifest custom_manifest.yaml`
- **THEN** the system SHALL use the specified manifest file
- **AND** all other behavior SHALL remain the same

#### Scenario: Dry-run mode
- **WHEN** user invokes `/run-pipelines --dry-run`
- **THEN** the system SHALL validate the manifest and configs
- **AND** the system SHALL display what would be executed
- **AND** no pipelines SHALL actually run

#### Scenario: QC-only mode
- **WHEN** user invokes `/run-pipelines --qc-only`
- **THEN** only QC pipelines SHALL be executed
- **AND** Viz and Cross-Platform pipelines SHALL be skipped
- **AND** summary SHALL reflect partial run

### Requirement: Progress Tracking
The system SHALL track and display progress during pipeline execution.

#### Scenario: Todo list updates
- **WHEN** pipelines are running
- **THEN** the system SHALL use TodoWrite to track progress
- **AND** each pipeline SHALL be listed as a task
- **AND** tasks SHALL be marked completed as pipelines finish

#### Scenario: Error reporting
- **WHEN** a pipeline fails
- **THEN** the system SHALL report the failure clearly
- **AND** the system SHALL continue with remaining pipelines (if possible)
- **AND** the summary SHALL indicate which pipelines failed

### Requirement: Gitignore Configuration
The `pipeline_runs/` directory SHALL be gitignored by default.

#### Scenario: Default gitignore
- **WHEN** user creates pipeline runs
- **THEN** the `pipeline_runs/` directory SHALL be in `.gitignore`
- **AND** run outputs SHALL not be committed to git by default
- **AND** users MAY selectively add specific runs if needed

### Requirement: Pipeline Summary JSON Reading

The system SHALL safely read and parse pipeline summary JSON files from completed runs.

#### Scenario: Read QC pipeline summary

- **GIVEN** a completed QC pipeline run with `10_pipeline_summary.json`
- **WHEN** `_read_pipeline_summary()` is called with the run path
- **THEN** the function SHALL return a dictionary with parsed JSON contents
- **AND** the function SHALL handle missing files by returning an empty dictionary

#### Scenario: Read summary with malformed JSON

- **GIVEN** a pipeline run with corrupted `pipeline_summary.json`
- **WHEN** `_read_pipeline_summary()` is called
- **THEN** the function SHALL log a warning
- **AND** the function SHALL return an empty dictionary
- **AND** the summary generation SHALL continue without crashing

#### Scenario: Extract QC metrics from summary

- **GIVEN** a valid QC pipeline summary JSON
- **WHEN** metrics are extracted for the summary table
- **THEN** `final_data.n_samples` SHALL map to Samples column
- **AND** `final_data.n_traits` SHALL map to Traits column
- **AND** `final_data.n_genotypes` SHALL map to Genotypes column
- **AND** `configuration.heritability.threshold` SHALL map to H² Threshold column
- **AND** `step_summaries.heritability_filter.mean_heritability_retained` SHALL map to Mean H² column

### Requirement: Removed Traits Documentation

The system SHALL document which traits were removed during QC for each dataset.

#### Scenario: List removed traits per dataset

- **GIVEN** a QC pipeline run that filtered traits by heritability
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a subsection under QC results titled "### Removed Traits by Dataset"
- **AND** each dataset SHALL list the traits removed with their heritability values if available

#### Scenario: No traits removed

- **GIVEN** a QC pipeline run where all traits passed heritability threshold
- **WHEN** the summary is generated
- **THEN** the removed traits section SHALL indicate "No traits removed" for that dataset

#### Scenario: Heritability filtering disabled

- **GIVEN** a QC pipeline run with `heritability.enabled: false`
- **WHEN** the summary is generated
- **THEN** the H² Threshold column SHALL display "Disabled"
- **AND** the Mean H² column SHALL display "N/A"
- **AND** no removed traits SHALL be listed for that dataset

### Requirement: Configuration Comparison in Summary

The system SHALL include a comprehensive configuration comparison section in the run summary document.

#### Scenario: Config comparison section generated

- **WHEN** pipeline run completes with multiple configs
- **THEN** the summary SHALL include a "## Configuration Comparison" section
- **AND** the section SHALL appear after the pipeline results tables and before the Methods section

#### Scenario: QC config parameters displayed

- **GIVEN** one or more QC pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### QC Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each config organized by section

#### Scenario: All cleanup parameters included

- **GIVEN** QC configs with cleanup sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `cleanup.max_nan_fraction`
- **AND** the table SHALL include `cleanup.max_zeros_per_trait`
- **AND** the table SHALL include `cleanup.max_nans_per_trait`
- **AND** the table SHALL include `cleanup.min_samples_per_trait`

#### Scenario: All outlier detection parameters included

- **GIVEN** QC configs with outlier_detection sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `outlier_detection.traditional_methods`
- **AND** the table SHALL include `outlier_detection.clustering_methods`
- **AND** the table SHALL include `outlier_detection.mahalanobis.variance_threshold`
- **AND** the table SHALL include `outlier_detection.mahalanobis.use_chi_squared`
- **AND** the table SHALL include `outlier_detection.mahalanobis.chi2_percentile`

#### Scenario: All outlier removal parameters included

- **GIVEN** QC configs with outlier_removal sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `outlier_removal.strategy`
- **AND** the table SHALL include `outlier_removal.method`

#### Scenario: All heritability parameters included

- **GIVEN** QC configs with heritability sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `heritability.enabled`
- **AND** the table SHALL include `heritability.threshold`
- **AND** the table SHALL include `heritability.generate_diagnostics`

#### Scenario: All PCA parameters included

- **GIVEN** QC configs with pca sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `pca.n_components`
- **AND** the table SHALL include `pca.feature_selection_strategy`

#### Scenario: All visualization parameters included

- **GIVEN** QC configs with visualization sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `visualization.dpi`
- **AND** the table SHALL include `visualization.figsize`
- **AND** the table SHALL include `visualization.title_fontsize`
- **AND** the table SHALL include `visualization.label_fontsize`
- **AND** the table SHALL include `visualization.tick_fontsize`
- **AND** the table SHALL include `visualization.legend_fontsize`
- **AND** the table SHALL include `visualization.figure_format`
- **AND** the table SHALL include `visualization.enable_batched_plots`
- **AND** the table SHALL include `visualization.batched_plot_threshold`
- **AND** the table SHALL include `visualization.batch_size`

#### Scenario: All adaptive sizing parameters included

- **GIVEN** QC configs with adaptive_sizing sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `adaptive_sizing.enabled`
- **AND** the table SHALL include `adaptive_sizing.base_width`
- **AND** the table SHALL include `adaptive_sizing.base_height`
- **AND** the table SHALL include `adaptive_sizing.width_per_item`
- **AND** the table SHALL include `adaptive_sizing.height_per_item`
- **AND** the table SHALL include `adaptive_sizing.min_width`
- **AND** the table SHALL include `adaptive_sizing.max_width`
- **AND** the table SHALL include `adaptive_sizing.min_height`
- **AND** the table SHALL include `adaptive_sizing.max_height`

#### Scenario: Root core parameters included when present

- **GIVEN** QC configs with root_core sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `root_core.core_qc.enabled`
- **AND** the table SHALL include `root_core.core_qc.max_missing_proportion`
- **AND** the table SHALL include `root_core.core_qc.remove_outliers`
- **AND** the table SHALL include `root_core.core_qc.detect_value_outliers`
- **AND** the table SHALL include `root_core.core_qc.max_deviation_from_median`
- **AND** the table SHALL include `root_core.generate_depth_profiles`

#### Scenario: Viz config parameters displayed

- **GIVEN** one or more Viz pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### Visualization Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each viz config

#### Scenario: Cross-platform config parameters displayed

- **GIVEN** one or more Cross-Platform pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### Cross-Platform Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each cross-platform config

#### Scenario: Table format with datasets as columns

- **GIVEN** multiple configs of the same pipeline type
- **WHEN** the config comparison table is generated
- **THEN** the first column SHALL be "Parameter"
- **AND** each subsequent column SHALL be the pipeline_name or config filename
- **AND** each row SHALL show one parameter with its value for each config

#### Scenario: Missing parameters shown as N/A

- **GIVEN** a parameter exists in some configs but not others
- **WHEN** the config comparison table is generated
- **THEN** configs without the parameter SHALL display "N/A" in that cell

#### Scenario: List values formatted correctly

- **GIVEN** a parameter value is a list (e.g., traditional_methods: [mahalanobis])
- **WHEN** the config comparison table is generated
- **THEN** the list SHALL be formatted as comma-separated values (e.g., "mahalanobis")

#### Scenario: Nested dict values flattened

- **GIVEN** a parameter has nested structure (e.g., mahalanobis.variance_threshold)
- **WHEN** the config comparison table is generated
- **THEN** the parameter name SHALL use dot notation (e.g., "mahalanobis.variance_threshold")
- **AND** the value SHALL be the leaf value

#### Scenario: Single config still shows table

- **GIVEN** only one config of a pipeline type was executed
- **WHEN** the config comparison is generated
- **THEN** the table SHALL still be generated with that single config as a column
- **AND** this allows users to see all parameters used even for single-config runs
$

### Requirement: Detailed Cross-Platform Summary in Run Output

The system SHALL generate a detailed cross-platform analysis summary section in `SUMMARY.md` when cross-platform pipelines complete successfully.

The detailed summary SHALL include for each cross-platform comparison:

**Comparison Overview Table**:
| Column | Description |
|--------|-------------|
| Comparison | "{exp1_name} vs {exp2_name}" |
| Genotypes | Number of common genotypes (n) |
| Trait Reduction | "X → Y (Z% reduction)" or "N/A" if disabled |
| Correlations | Total number of trait pair correlations |
| Nominal Sig | Count where p < 0.05 |
| FDR Sig | Count where q < 0.05 (FDR-adjusted) |
| Top |r| | Maximum absolute correlation coefficient |
| Power ≥80% | Count/percentage of correlations with achieved_power ≥ 0.80 |

**Top Correlations Table** (per comparison, top 5 by |r|):
| Column | Description |
|--------|-------------|
| Rank | 1-5 |
| Exp1 Trait | Trait name from experiment 1 |
| Exp2 Trait | Trait name from experiment 2 |
| r | Spearman correlation coefficient (signed) |
| p | Raw p-value |
| q | FDR-adjusted p-value |
| Power | Achieved statistical power |
| n | Number of genotypes used |

**Metadata Table** (per comparison):
- FDR correction method
- Trait reduction method and parameters
- Minimum genotypes threshold
- Significance level (α)

**Embedded Visualizations** (per comparison):
- Correlation summary plot (4-panel): `cross_platform_correlation_summary.png`
- Top 3 joint plots: Scatter + regression for highest |r| correlations
- Exp1 clustering dendrogram (if exp1 clustered): `exp1_trait_clustering_dendrogram.png`
- Exp1 cluster heatmap (if exp1 clustered): `exp1_trait_cluster_heatmap.png`
- Exp2 clustering dendrogram (if exp2 clustered): `exp2_trait_clustering_dendrogram.png`
- Exp2 cluster heatmap (if exp2 clustered): `exp2_trait_cluster_heatmap.png`
- Cross-platform representative heatmap (if any clustering): `cross_platform_representative_heatmap.png`

#### Scenario: Cross-platform summary generated after run-all

- **WHEN** `sleap-roots-analyze run-all` completes with cross-platform pipelines
- **THEN** `SUMMARY.md` SHALL contain a "## Detailed Cross-Platform Analysis" section
- **AND** the section SHALL appear after the existing Cross-Platform Results table
- **AND** each comparison SHALL have its own subsection

#### Scenario: Trait reduction statistics displayed

- **WHEN** a cross-platform run used `trait_reduction_method: clustering`
- **THEN** the Trait Reduction column SHALL show "819 → 121 (85% reduction)" format
- **AND** the values SHALL match `trait_clusters.csv` exactly

#### Scenario: Trait reduction disabled

- **WHEN** a cross-platform run used `trait_reduction_method: none`
- **THEN** the Trait Reduction column SHALL show "Disabled"
- **AND** no `trait_clusters.csv` file is expected

#### Scenario: Correlation counts verified against source

- **WHEN** summary reports correlation counts
- **THEN** total correlations SHALL equal row count in `cross_platform_correlations.csv`
- **AND** nominal significant SHALL equal count where `spearman_p < 0.05`
- **AND** FDR significant SHALL equal count where `significant_fdr == True`

#### Scenario: Top correlations match CSV ordering

- **WHEN** summary displays top 5 correlations
- **THEN** they SHALL be the 5 rows with highest `abs(spearman_r)` from CSV
- **AND** all displayed values (r, p, q, power, n) SHALL match CSV values exactly

#### Scenario: Power statistics computed correctly

- **WHEN** summary displays power statistics
- **THEN** Power ≥80% count SHALL equal count where `achieved_power >= 0.80` in CSV
- **AND** percentage SHALL be calculated as (count / total) * 100

#### Scenario: Multiple comparisons summarized

- **WHEN** run includes 4 cross-platform configs
- **THEN** summary SHALL include subsections for all 4 comparisons
- **AND** each subsection SHALL have complete statistics

#### Scenario: Visualizations embedded inline

- **WHEN** summary is generated
- **THEN** images SHALL be embedded using markdown syntax: `![Description](relative/path/to/image.png)`
- **AND** paths SHALL be relative to the run directory
- **AND** images SHALL render correctly in GitHub, VS Code, and other markdown viewers

#### Scenario: Correlation summary plot included

- **WHEN** a cross-platform comparison has `cross_platform_correlation_summary.png`
- **THEN** the summary SHALL embed this 4-panel visualization
- **AND** it SHALL appear at the start of the comparison subsection

#### Scenario: Top joint plots included

- **WHEN** a cross-platform comparison has joint plots
- **THEN** the summary SHALL embed the top 3 joint plots by |r|
- **AND** plots SHALL be ordered from strongest to weakest correlation

#### Scenario: Exp1 clustering visualizations included when exp1 clustered

- **WHEN** a cross-platform run used `trait_reduction_target: exp1` or `trait_reduction_target: both`
- **THEN** the summary SHALL embed `exp1_trait_clustering_dendrogram.png`
- **AND** the summary SHALL embed `exp1_trait_cluster_heatmap.png`

#### Scenario: Exp2 clustering visualizations included when exp2 clustered

- **WHEN** a cross-platform run used `trait_reduction_target: exp2` or `trait_reduction_target: both`
- **THEN** the summary SHALL embed `exp2_trait_clustering_dendrogram.png`
- **AND** the summary SHALL embed `exp2_trait_cluster_heatmap.png`

#### Scenario: Representative heatmap included when any clustering enabled

- **WHEN** a cross-platform run used `trait_reduction_method: clustering`
- **THEN** the summary SHALL embed `cross_platform_representative_heatmap.png`

#### Scenario: Clustering visualizations omitted when disabled

- **WHEN** a cross-platform run used `trait_reduction_method: none`
- **THEN** the summary SHALL NOT include any clustering visualizations
- **AND** no error SHALL occur for missing visualization files

### Requirement: Cross-Platform Summary Validation Guardrails

The system SHALL validate that all reported statistics match their source data files before generating the summary.

#### Scenario: Validation passes for accurate summary

- **WHEN** summary generator reads `cross_platform_correlations.csv` and `trait_clusters.csv`
- **AND** all computed statistics match source data
- **THEN** validation SHALL pass
- **AND** summary SHALL be generated normally

#### Scenario: Validation fails on correlation count mismatch

- **WHEN** reported total correlations does not match CSV row count
- **THEN** validation SHALL fail with error message
- **AND** summary SHALL include warning about discrepancy
- **AND** the mismatch SHALL be logged for debugging

#### Scenario: Validation fails on trait reduction mismatch

- **WHEN** reported trait reduction percentage does not match computed from `trait_clusters.csv`
- **THEN** validation SHALL fail with error message
- **AND** summary SHALL include warning about discrepancy

#### Scenario: Missing source files handled gracefully

- **WHEN** `cross_platform_correlations.csv` is missing
- **THEN** validation SHALL skip that comparison
- **AND** summary SHALL show "Data unavailable" for that comparison
- **AND** no crash SHALL occur

### Requirement: Cross-Platform Summary Slash Command

The system SHALL provide a `/cross-platform-summary` Claude command for generating detailed cross-platform analysis reports on demand.

#### Scenario: Basic command invocation

- **WHEN** user invokes `/cross-platform-summary pipeline_runs/2026-02-02_133904`
- **THEN** the system SHALL read all cross-platform results from that directory
- **AND** the system SHALL generate a detailed summary report
- **AND** the system SHALL display the report to the user

#### Scenario: Command with latest run

- **WHEN** user invokes `/cross-platform-summary` without arguments
- **THEN** the system SHALL find the most recent pipeline run directory
- **AND** the system SHALL generate summary for that run

#### Scenario: Command validates results

- **WHEN** summary is generated via command
- **THEN** validation guardrails SHALL be applied
- **AND** any discrepancies SHALL be reported to the user
- **AND** user SHALL be warned if data integrity issues are detected

#### Scenario: No cross-platform results found

- **WHEN** specified directory contains no cross-platform results
- **THEN** command SHALL report "No cross-platform results found in {directory}"
- **AND** command SHALL suggest checking the directory path

