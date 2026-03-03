## ADDED Requirements

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
