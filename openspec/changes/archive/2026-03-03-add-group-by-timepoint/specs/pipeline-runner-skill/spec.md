## ADDED Requirements

### Requirement: Grouped Viz Fan-Out in Run-All

When `run-all` executes a QC config that uses `group_by`, it SHALL automatically fan out the
downstream viz pipeline to run once per QC group output, not once for the last group only.

This requirement exists because grouped QC produces N output directories (one per group value).
Without fan-out, the viz pipeline only ever processes the final group's data, silently discarding
results for all other groups.

#### Scenario: Run-all detects grouped QC outputs

- **GIVEN** a QC config with `data.group_by: plant_age_days`
- **WHEN** `run-all` completes the QC phase
- **THEN** the runner SHALL detect all group output directories under `run_dir/qc/`
- **AND** each directory matching the pattern `{group_by_column}_{value}_{timestamp}` SHALL be collected
- **AND** the collected directories SHALL be stored for use by the downstream viz phase

#### Scenario: Viz runs once per QC group

- **GIVEN** a grouped QC run that produced outputs for plant_age_days = [7, 9, 12]
- **WHEN** `run-all` proceeds to the viz phase
- **THEN** the viz pipeline SHALL be invoked exactly 3 times (once per group)
- **AND** each viz invocation SHALL use the corresponding group's `10_final_data.csv` as `csv_path`
- **AND** each viz invocation SHALL write outputs to a group-specific subdirectory

#### Scenario: Per-group viz output directory naming

- **GIVEN** a grouped QC run with groups plant_age_days = [7, 9, 12]
- **WHEN** viz fan-out runs
- **THEN** each group's viz output SHALL be placed in `run_dir/viz/{group_label}/`
  where `{group_label}` is derived from the QC group dir name with the timestamp stripped
  (e.g., `plant_age_days_7` from `plant_age_days_7_20260217_114013`)
- **AND** the updated viz config for that group SHALL be written to
  `run_dir/viz/{group_label}/_updated_{original_config_name}`, following the
  `_updated_*` convention used for non-grouped runs

#### Scenario: Per-group results tracked separately

- **GIVEN** a grouped viz fan-out with 3 groups
- **WHEN** `run-all` completes
- **THEN** `run_results["viz"]` SHALL contain 3 separate entries
- **AND** each key SHALL identify the group (e.g., `viz/config.yaml:plant_age_days_7`)
- **AND** each entry SHALL record success/failure independently

#### Scenario: Group skipped when CSV is absent

- **GIVEN** a QC group output directory without a `10_final_data.csv` file
- **WHEN** viz fan-out attempts to process that group
- **THEN** the runner SHALL record a failure result for that group
- **AND** the runner SHALL NOT raise an exception
- **AND** other groups SHALL continue to be processed

#### Scenario: Non-grouped viz behavior unchanged

- **GIVEN** a QC config without `data.group_by` (or with `data.group_by: null`)
- **WHEN** `run-all` executes the viz phase
- **THEN** viz SHALL run exactly once (existing behavior)
- **AND** the path auto-update logic SHALL work as before
