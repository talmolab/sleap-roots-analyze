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

#### Scenario: Path auto-update
- **WHEN** Viz or Cross-Platform configs reference QC outputs
- **THEN** the system SHALL automatically update `data.csv_path` in Viz configs
- **AND** the system SHALL automatically update `exp1_data_path` and `exp2_data_path` in Cross-Platform configs
- **AND** updates SHALL point to the new QC run's `10_final_data.csv`

### Requirement: Run Summary Generation
The system SHALL generate a comprehensive markdown summary document after each run.

#### Scenario: Summary document created
- **WHEN** pipeline run completes
- **THEN** a `SUMMARY.md` file SHALL be created in the run directory
- **AND** the summary SHALL include generation timestamp
- **AND** the summary SHALL include git commit hash
- **AND** the summary SHALL include manifest file reference

#### Scenario: QC results in summary
- **WHEN** QC pipelines complete
- **THEN** the summary SHALL include a table with: Dataset, Samples, Traits, Genotypes, H² Threshold, Mean H², Run Path
- **AND** the summary SHALL include detailed results per dataset
- **AND** the summary SHALL list removed traits for each dataset

#### Scenario: Viz results in summary
- **WHEN** Viz pipelines complete
- **THEN** the summary SHALL include a table with: Dataset, Elapsed Time, Status, Run Path
- **AND** the summary SHALL list visualization configs and their key parameters

#### Scenario: Cross-Platform results in summary
- **WHEN** Cross-Platform pipelines complete
- **THEN** the summary SHALL include a table with: Comparison, Common Genotypes, Exp1 Samples/Traits, Exp2 Samples/Traits, Run Path
- **AND** the summary SHALL document key questions addressed by each analysis

#### Scenario: Methods section template
- **WHEN** summary is generated
- **THEN** the summary SHALL include a publication-ready methods section template
- **AND** the template SHALL describe QC and Viz pipeline methodology

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

