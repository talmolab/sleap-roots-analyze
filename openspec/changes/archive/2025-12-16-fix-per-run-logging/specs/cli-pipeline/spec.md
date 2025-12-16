## ADDED Requirements

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
