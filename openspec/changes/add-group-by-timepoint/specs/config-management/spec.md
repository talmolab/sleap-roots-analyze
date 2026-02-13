## ADDED Requirements

### Requirement: Group-By Configuration

Pipeline configurations SHALL support a `group_by` field in the `data` section to enable analysis of data subsets partitioned by metadata columns.

#### Scenario: Group-by field in config
- **GIVEN** a QC config with `data.group_by: "plant_age_days"`
- **WHEN** the config is loaded
- **THEN** the pipeline SHALL split data by unique values in the plant_age_days column

#### Scenario: Group-by field is optional
- **GIVEN** a QC config without a `group_by` field
- **WHEN** the config is loaded
- **THEN** the pipeline SHALL process all data as a single group (current behavior)

#### Scenario: CLI overrides config group-by
- **GIVEN** a config with `data.group_by: "plant_age_days"` and CLI flag `--group-by experiment_id`
- **WHEN** the pipeline is executed
- **THEN** the CLI value SHALL take precedence and data SHALL be grouped by experiment_id

#### Scenario: Validation of group-by column existence
- **GIVEN** a config with `data.group_by: "nonexistent_column"`
- **WHEN** the config is validated
- **THEN** validation SHALL fail with error message indicating the column does not exist in the data

#### Scenario: Group-by applies to both QC and viz pipelines
- **GIVEN** a viz config with `data.group_by: "plant_age_days"`
- **WHEN** the viz pipeline is executed
- **THEN** data SHALL be split into groups before visualization, identical to QC behavior
