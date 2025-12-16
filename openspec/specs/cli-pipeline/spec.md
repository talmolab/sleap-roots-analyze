# cli-pipeline Specification

## Purpose
TBD - created by archiving change fix-config-log-file-ignored. Update Purpose after archive.
## Requirements
### Requirement: Config-Based Log File Resolution

The CLI SHALL use the pipeline config's `logging.log_file` setting when the `--log-file` CLI flag is not explicitly provided.

#### Scenario: Config log file used when CLI flag omitted
- **GIVEN** a config file with `logging.log_to_file: true` and `logging.log_file: "pipeline.log"`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output` without `--log-file`
- **THEN** a log file SHALL be created at `./output/pipeline.log`

#### Scenario: CLI flag overrides config
- **GIVEN** a config file with `logging.log_file: "pipeline.log"`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output --log-file ./custom.log`
- **THEN** the log file SHALL be created at `./custom.log` (not `./output/pipeline.log`)

#### Scenario: No log file when log_to_file is false
- **GIVEN** a config file with `logging.log_to_file: false`
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o ./output` without `--log-file`
- **THEN** no log file SHALL be created

#### Scenario: Log file path resolved relative to output directory
- **GIVEN** a config file with `logging.log_file: "logs/qc.log"` (relative path)
- **WHEN** the user runs `sleap-roots-analyze qc config.yaml -o /data/results`
- **THEN** the log file SHALL be created at `/data/results/logs/qc.log`

