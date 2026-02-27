## ADDED Requirements

### Requirement: Group-Based Pipeline Execution

The QC pipeline SHALL support partitioning input data by a metadata column and executing separate pipeline instances per partition.

#### Scenario: Data split by group column
- **GIVEN** a dataset with plant_age_days values [7, 7, 14, 14, 21, 21] (6 samples)
- **WHEN** the pipeline runs with `group_by: "plant_age_days"`
- **THEN** the data SHALL be split into 3 groups: day_7 (2 samples), day_14 (2 samples), day_21 (2 samples)
- **AND** each group SHALL be processed independently

#### Scenario: Isolated output directories per group
- **GIVEN** a pipeline run with `group_by: "plant_age_days"` and groups [7, 14]
- **WHEN** the pipeline executes with `-o /data/output`
- **THEN** two output directories SHALL be created:
  - `/data/output/plant_age_days_7_<timestamp>/`
  - `/data/output/plant_age_days_14_<timestamp>/`
- **AND** each directory SHALL contain complete pipeline outputs (10_final_data.csv, figures/, etc.)

#### Scenario: Group validation using min_samples_per_trait
- **GIVEN** a pipeline with `cleanup.min_samples_per_trait: 10`
- **AND** a group with only 5 samples
- **WHEN** the pipeline validates groups
- **THEN** the group SHALL be skipped
- **AND** a warning SHALL be logged: "Skipping group plant_age_days=7 (5 samples < 10 minimum)"

#### Scenario: All groups processed when valid
- **GIVEN** a dataset grouped into day_7 (50 samples) and day_14 (60 samples)
- **AND** `min_samples_per_trait: 10`
- **WHEN** the pipeline executes
- **THEN** both groups SHALL be processed
- **AND** no groups SHALL be skipped

#### Scenario: Heritability calculated per group
- **GIVEN** a grouped pipeline run (day_7, day_14, day_21)
- **WHEN** heritability estimates are calculated
- **THEN** each group SHALL have independent heritability estimates
- **AND** heritability values MAY differ between groups (developmental stage effects)

#### Scenario: Group metadata preserved in outputs
- **GIVEN** a grouped pipeline run by plant_age_days
- **WHEN** `10_final_data.csv` is written for group day_7
- **THEN** the plant_age_days column SHALL be preserved
- **AND** all samples SHALL have plant_age_days=7

#### Scenario: Empty groups are skipped
- **GIVEN** a dataset where no samples have plant_age_days=30
- **WHEN** the pipeline attempts to group by plant_age_days
- **THEN** no group for day_30 SHALL be created
- **AND** no error SHALL occur
