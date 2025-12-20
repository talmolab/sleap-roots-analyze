## MODIFIED Requirements

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