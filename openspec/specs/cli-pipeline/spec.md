# cli-pipeline Specification

## Purpose
TBD - created by archiving change fix-config-log-file-ignored. Update Purpose after archive.
## Requirements
### Requirement: Config-Based Log File Resolution

The CLI SHALL use the pipeline config's `logging.log_file` setting when the `--log-file` CLI flag is not explicitly provided. The log file SHALL be created inside the run-specific directory, not the base output directory.

#### Scenario: Config log file created in run directory
- **GIVEN** a config file with `logging.log_to_file: true` and `logging.log_file: "custom.log"`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output` without `--log-file`
- **THEN** a log file SHALL be created at `./output/{run_dir}/custom.log`
- **AND** no log file SHALL be created at `./output/custom.log`

#### Scenario: CLI flag overrides config with absolute path
- **GIVEN** a config file with `logging.log_file: "pipeline.log"`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output --log-file /tmp/custom.log`
- **THEN** the log file SHALL be created at `/tmp/custom.log`
- **AND** no log file SHALL be created in the run directory from config

#### Scenario: No log file when log_to_file is false
- **GIVEN** a config file with `logging.log_to_file: false`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output` without `--log-file`
- **THEN** no config-based log file SHALL be created
- **AND** the pipeline's default `pipeline.log` SHALL still be created in run directory

#### Scenario: Log file path resolved relative to run directory
- **GIVEN** a config file with `logging.log_file: "logs/qc.log"` (relative path with subdirectory)
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o /data/results`
- **THEN** the log file SHALL be created at `/data/results/{run_dir}/logs/qc.log`
- **AND** parent directories SHALL be created as needed

#### Scenario: Default log filename when not specified in config
- **GIVEN** a config file with `logging.log_to_file: true` but no `logging.log_file` key
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output`
- **THEN** a log file SHALL be created at `./output/{run_dir}/pipeline.log`
$

### Requirement: Pipeline Run Config Provenance

Every pipeline run SHALL save the resolved configuration to the run output directory for reproducibility.

#### Scenario: Config YAML saved to run directory
- **GIVEN** a pipeline run with any config (QC, Viz, or CrossPlatform)
- **WHEN** the pipeline completes successfully
- **THEN** a file `config.yaml` SHALL exist in the run output directory
- **AND** the file SHALL contain the complete resolved configuration

#### Scenario: Config YAML matches original config
- **GIVEN** a pipeline run that saved `config.yaml`
- **WHEN** the saved config is loaded and compared to the original
- **THEN** all parameter values SHALL match the original config
- **AND** the saved config SHALL be valid for re-running the pipeline

#### Scenario: Config saved even on pipeline failure
- **GIVEN** a pipeline run that fails during execution
- **WHEN** the pipeline error is raised
- **THEN** `config.yaml` SHALL still exist in the run directory
- **AND** it SHALL contain the config that was used when the failure occurred

---

### Requirement: Pipeline Summary Config Population

The pipeline summary JSON SHALL include the complete configuration used for the run.

#### Scenario: Summary includes config dict
- **GIVEN** a pipeline run that completes
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** the `config` field SHALL contain a non-empty dictionary
- **AND** the dictionary SHALL include all config parameters

#### Scenario: Summary config is JSON serializable
- **GIVEN** a config with complex types (Path objects, dataclasses)
- **WHEN** the pipeline saves the summary
- **THEN** all config values SHALL be serialized to JSON-compatible types
- **AND** Path objects SHALL be converted to POSIX strings via `Path.as_posix()` (forward-slash separators on every OS)

### Requirement: Data Source Tracking

The pipeline summary SHALL explicitly record the input data source(s) used.

#### Scenario: QC pipeline records data source

- **GIVEN** a QC pipeline run with `data.csv_path: "/path/to/data.csv"`
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain `"/path/to/data.csv"`
- **AND** the path SHALL be the absolute resolved path

#### Scenario: Viz pipeline records data source

- **GIVEN** a Viz pipeline run with `data.data_path: "/path/to/qc_output.csv"`
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain `"/path/to/qc_output.csv"`
- **AND** the field SHALL NOT be empty or null

#### Scenario: CrossPlatform pipeline records both sources

- **GIVEN** a CrossPlatform pipeline comparing two experiments
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain paths to both experiment data files
- **AND** the format SHALL be: `{"exp1": "/path/to/exp1.csv", "exp2": "/path/to/exp2.csv"}`

### Requirement: Per-Run Log File Generation

Each pipeline run SHALL generate a dedicated log file within its run directory for traceability and debugging.

#### Scenario: Log file created in run directory
- **GIVEN** a pipeline run that creates output directory `run_dir/`
- **WHEN** the pipeline executes any tasks
- **THEN** a file `pipeline.log` SHALL exist in `run_dir/`
- **AND** the log file SHALL contain all log messages from that specific run

#### Scenario: Log file contains pipeline execution messages
- **GIVEN** a completed pipeline run
- **WHEN** the per-run log file is read
- **THEN** it SHALL contain the pipeline start message
- **AND** it SHALL contain task execution messages (e.g., "Executing task: load_data")
- **AND** it SHALL contain the pipeline completion or failure message

#### Scenario: Multiple parallel runs have independent logs
- **GIVEN** two pipeline runs executing in parallel
- **WHEN** both runs complete
- **THEN** each run's `pipeline.log` SHALL only contain messages from that specific run
- **AND** log messages SHALL NOT be interleaved between runs

#### Scenario: Log file persists on pipeline failure
- **GIVEN** a pipeline run that fails during execution
- **WHEN** the error is raised
- **THEN** `pipeline.log` SHALL still exist in the run directory
- **AND** it SHALL contain all log messages up to and including the failure

#### Scenario: Per-run log coexists with CLI-level log
- **GIVEN** a CLI command that sets `--log-file ./output/all_runs.log`
- **WHEN** a pipeline run creates `run_dir/pipeline.log`
- **THEN** both log files SHALL exist
- **AND** CLI-level log SHALL contain messages from all runs
- **AND** per-run log SHALL contain messages only from that run

### Requirement: Group-By CLI Flag

The QC and visualization pipeline CLI commands SHALL support a `--group-by` flag to partition input data by a metadata column.

#### Scenario: Group-by flag for QC pipeline
- **GIVEN** a dataset with plant_age_days column containing values [7, 14, 21]
- **WHEN** user runs `sleap-roots-analyze qc config.yaml --group-by plant_age_days`
- **THEN** the pipeline SHALL execute three separate QC runs (one per timepoint)
- **AND** each run SHALL process only samples with that timepoint value

#### Scenario: Group-by flag for viz pipeline
- **GIVEN** a dataset with experiment_id column containing values ["exp1", "exp2"]
- **WHEN** user runs `sleap-roots-analyze viz config.yaml --group-by experiment_id`
- **THEN** the pipeline SHALL execute two separate viz runs (one per experiment)

#### Scenario: Group-by CLI overrides config value
- **GIVEN** a config with `data.group_by: "plant_age_days"`
- **WHEN** user runs `sleap-roots-analyze qc config.yaml --group-by experiment_id`
- **THEN** the pipeline SHALL group by experiment_id (CLI precedence)

#### Scenario: No group-by defaults to single run
- **GIVEN** a config without `data.group_by` field
- **WHEN** user runs `sleap-roots-analyze qc config.yaml` (no --group-by flag)
- **THEN** the pipeline SHALL process all data as a single run (current behavior)

#### Scenario: Group-by with run-all command
- **GIVEN** a manifest file with pipeline entries
- **WHEN** user runs `sleap-roots-analyze run-all manifest.yaml --group-by plant_age_days`
- **THEN** ALL pipelines in the manifest SHALL be grouped by plant_age_days
- **AND** manifest-level `group_by` fields SHALL be overridden

#### Scenario: Invalid group-by column error
- **GIVEN** a dataset without a column named "invalid_column"
- **WHEN** user runs `sleap-roots-analyze qc config.yaml --group-by invalid_column`
- **THEN** the CLI SHALL exit with error code 1
- **AND** stderr SHALL indicate the column does not exist in the data

### Requirement: Provenance Path Serialization Is Centralized

The pipeline provenance manifest (`pipeline_summary.json`) SHALL record every filesystem path through the central JSON serializer (`convert_to_json_serializable`) so that path normalization happens in exactly one place. Producing steps SHALL store `Path` objects in `files_generated` and `metadata`; they SHALL NOT pre-stringify paths with `str(path)`, which on Windows yields backslash separators and bypasses the serializer's `Path.as_posix()` normalization. The `files_generated` field SHALL be typed `List[Path]` so the type cannot invite per-producer divergence.

#### Scenario: Generated file paths normalize to POSIX in the manifest

- **GIVEN** a pipeline step that records a generated file in `files_generated`
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized path SHALL equal `Path(p).as_posix()`
- **AND** it SHALL use forward-slash (`/`) separators regardless of the host platform's native separator

#### Scenario: Metadata path values normalize to POSIX in the manifest

- **GIVEN** a pipeline step that records a path under `metadata` (e.g. `output_csv`, `dashboard_path`, or a relative `Path.relative_to(run_dir)`)
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized path value SHALL equal `Path.as_posix()` of the stored path, preserving its relative-vs-absolute form
- **AND** it SHALL use forward-slash (`/`) separators on every OS

#### Scenario: Optional path values serialize to JSON null

- **GIVEN** a step that records an optional path under `metadata` whose value is `None` (e.g. `reps_plot` when no replicate plot was produced)
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized value SHALL be JSON `null`
- **AND** dropping the producer-side `str(path)` SHALL NOT change a `None` value into the string `"None"`

#### Scenario: Top-level and standalone-manifest paths normalize to POSIX

- **GIVEN** the pipeline writes `output_directory` into `pipeline_summary.json` and per-step `*_manifest.json` / `summary.json` files containing `Path` values
- **WHEN** those files are written on any OS
- **THEN** every serialized path SHALL use forward-slash (`/`) separators
- **AND** all serializer sinks (`convert_to_json_serializable`, the `save_json` default hook, and the viz `summary.json` writer) SHALL normalize paths via the same `PurePath.as_posix()` predicate

#### Scenario: Producers do not pre-stringify paths

- **GIVEN** any pipeline step that contributes a path to `files_generated` or `metadata`
- **WHEN** the step records the path
- **THEN** it SHALL store a `Path` object (never `str(path)`)
- **AND** a CI-enforced source guard SHALL fail if a `str(path)` pre-stringification is reintroduced into a step's `files_generated`/`metadata`

#### Scenario: Serialization round-trip gate is green cross-OS

- **GIVEN** the result-object JSON round-trip gate (`serialization-gate` CI job, issue #156) running on ubuntu, windows, and macos
- **WHEN** a `PipelineSummary` carrying `Path` values is serialized and reloaded
- **THEN** it SHALL round-trip to identical POSIX strings on all three platforms

