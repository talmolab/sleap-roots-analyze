# pipeline-runner-skill Spec Delta

## MODIFIED Requirements

### Requirement: Pipeline Execution Order

The system SHALL execute pipelines in the correct order respecting dependencies.

#### Scenario: QC runs before Viz

- **WHEN** user runs all pipelines
- **THEN** all QC pipelines SHALL complete before any Viz pipelines start
- **AND** Viz configs SHALL be updated to reference new QC output paths
- **AND** Cross-Platform configs SHALL be updated to reference new QC output paths

#### Scenario: Path auto-update preserves filename choice

- **WHEN** Viz or Cross-Platform configs reference QC outputs
- **THEN** the system SHALL automatically update the directory portion of paths
- **AND** the system SHALL preserve the original filename from the config (e.g., `07_data_outliers_removed.csv` or `10_final_data.csv`)
- **AND** if original config specifies `07_data_outliers_removed.csv`, the updated path SHALL also use `07_data_outliers_removed.csv`
- **AND** if original config specifies `10_final_data.csv`, the updated path SHALL also use `10_final_data.csv`

#### Scenario: Config structure preserved during update

- **WHEN** config files are updated with new QC output paths
- **THEN** the system SHALL preserve all YAML comments in the config file
- **AND** the system SHALL preserve the original key ordering
- **AND** the system SHALL preserve string quoting styles
- **AND** the system SHALL preserve blank lines and formatting

#### Scenario: Cross-platform config with mixed file choices

- **GIVEN** a cross-platform config where `exp1_data_path` uses `07_data_outliers_removed.csv`
- **AND** `exp2_data_path` uses `10_final_data.csv`
- **WHEN** the config is updated with new QC output paths
- **THEN** `exp1_data_path` SHALL be updated to the new directory with `07_data_outliers_removed.csv`
- **AND** `exp2_data_path` SHALL be updated to the new directory with `10_final_data.csv`
