## MODIFIED Requirements

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

## NEW Requirements

### Requirement: Pipeline Summary JSON Schema

The pipeline summary JSON SHALL follow a consistent schema across all pipeline types.

#### Scenario: Viz summary schema completeness

- **GIVEN** a Viz pipeline run
- **WHEN** `summary.json` is generated
- **THEN** `data_overview.n_traits_initial` SHALL contain the number of traits in the input data
- **AND** `data_overview.n_traits_final` SHALL contain the number of traits after any filtering
- **AND** both values SHALL be non-zero positive integers for valid data

#### Scenario: Summary includes environment info

- **GIVEN** any pipeline run
- **WHEN** `pipeline_summary.json` is generated
- **THEN** the `environment` field SHALL contain:
  - `platform`: Operating system name and version
  - `python_version`: Full Python version string
  - `working_directory`: Absolute path to working directory
- **AND** the field SHALL NOT be an empty dictionary
