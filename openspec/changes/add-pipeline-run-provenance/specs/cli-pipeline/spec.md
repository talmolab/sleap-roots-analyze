# cli-pipeline Specification Delta

## ADDED Requirements

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
- **AND** Path objects SHALL be converted to strings

---

### Requirement: Data Source Tracking

The pipeline summary SHALL explicitly record the input data source(s) used.

#### Scenario: QC pipeline records data source
- **GIVEN** a QC pipeline run with `data.csv_path: "/path/to/data.csv"`
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain `"/path/to/data.csv"`

#### Scenario: Viz pipeline records data source
- **GIVEN** a Viz pipeline run with `data.data_path: "/path/to/qc_output.csv"`
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain `"/path/to/qc_output.csv"`

#### Scenario: CrossPlatform pipeline records both sources
- **GIVEN** a CrossPlatform pipeline comparing two experiments
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** `data_source` SHALL contain paths to both experiment data files

---

## MODIFIED Requirements

None - this change only adds new functionality.

## REMOVED Requirements

None.
