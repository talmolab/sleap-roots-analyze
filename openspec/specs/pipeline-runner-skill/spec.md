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
- **THEN** the summary SHALL include a table with columns: Comparison, Common Genotypes, Exp1 Samples, Exp1 Traits, Exp2 Samples, Exp2 Traits, Top Correlation, Status, Run Path
- **AND** the summary SHALL read `cross_platform_alignment_summary.csv` or `pipeline_summary.json` to extract alignment metrics
- **AND** the summary SHALL read `cross_platform_correlations.csv` to extract the top correlation value
- **AND** the CSV parser SHALL use the actual column names: `genotype`, `exp1_sample_count`, `exp2_sample_count`

#### Scenario: Cross-Platform results with missing data

- **WHEN** Cross-Platform pipeline completes but alignment CSV is missing
- **THEN** the summary SHALL display "N/A" for genotype/sample/trait counts
- **AND** the summary SHALL still display status and run path

#### Scenario: Methods section template

- **WHEN** summary is generated
- **THEN** the summary SHALL include a "## Methods" section with publication-ready template text
- **AND** the template SHALL describe the QC pipeline methodology (cleanup, outlier detection, heritability filtering)
- **AND** the template SHALL describe the Viz pipeline methodology (statistical analysis, visualization generation)
- **AND** all placeholders SHALL be replaced with actual config values (e.g., `{h2_threshold}` becomes "0.4")
- **AND** if configs differ across datasets, the Methods section SHALL note "varied by dataset" with a footnote

#### Scenario: Summary statistics figure generated

- **WHEN** QC pipelines complete with at least 2 datasets
- **THEN** the summary directory SHALL contain a `summary_statistics.png` bar chart
- **AND** the chart SHALL show sample count, trait count, and genotype count per dataset
- **AND** the chart SHALL use a grouped bar layout for easy comparison

#### Scenario: Heritability distribution figure generated

- **WHEN** QC pipelines complete with heritability filtering enabled
- **THEN** the summary directory SHALL contain a `heritability_distribution.png` visualization
- **AND** the visualization SHALL show H² distributions for retained vs. removed traits
- **AND** the visualization SHALL include the threshold line

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

