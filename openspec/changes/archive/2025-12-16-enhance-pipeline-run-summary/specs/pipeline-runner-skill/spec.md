## MODIFIED Requirements

### Requirement: Run Summary Generation

The system SHALL generate a comprehensive markdown summary document after each run.

#### Scenario: Summary document created

- **WHEN** pipeline run completes
- **THEN** a `SUMMARY.md` file SHALL be created in the run directory
- **AND** the summary SHALL include generation timestamp
- **AND** the summary SHALL include git commit hash
- **AND** the summary SHALL include manifest file reference

#### Scenario: QC results in summary

- **WHEN** QC pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Dataset, Samples, Traits, Genotypes, H² Threshold, Mean H², Status, Run Path
- **AND** for each QC run, the summary SHALL read `10_pipeline_summary.json` to extract scientific metrics
- **AND** the summary SHALL include a "Removed Traits" subsection listing traits filtered by heritability threshold for each dataset

#### Scenario: QC results with failed pipeline

- **WHEN** a QC pipeline fails during execution
- **THEN** the summary table SHALL show "Failed" status for that config
- **AND** numeric columns (Samples, Traits, etc.) SHALL display "N/A"
- **AND** other successful QC runs SHALL still display full metrics

#### Scenario: Viz results in summary

- **WHEN** Viz pipelines complete
- **THEN** the summary SHALL include a table with columns: Dataset, Figures Generated, Interactive Plots, Status, Time, Run Path
- **AND** the summary SHALL extract figure counts from the output directory

#### Scenario: Cross-Platform results in summary

- **WHEN** Cross-Platform pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Comparison, Common Genotypes, Exp1 Samples, Exp1 Traits, Exp2 Samples, Exp2 Traits, Top Correlation, Status, Run Path
- **AND** the summary SHALL read `cross_platform_alignment_summary.csv` or `pipeline_summary.json` to extract alignment metrics
- **AND** the summary SHALL read `cross_platform_correlations.csv` to extract the top correlation value

#### Scenario: Cross-Platform results with missing data

- **WHEN** Cross-Platform pipeline completes but alignment CSV is missing
- **THEN** the summary SHALL display "N/A" for genotype/sample/trait counts
- **AND** the summary SHALL still display status and run path

#### Scenario: Methods section template

- **WHEN** summary is generated
- **THEN** the summary SHALL include a "## Methods" section with publication-ready template text
- **AND** the template SHALL describe the QC pipeline methodology (cleanup, outlier detection, heritability filtering)
- **AND** the template SHALL describe the Viz pipeline methodology (statistical analysis, visualization generation)
- **AND** the template SHALL include placeholders for dataset-specific values (e.g., `{n_samples}`, `{n_traits}`)

## ADDED Requirements

### Requirement: Pipeline Summary JSON Reading

The system SHALL safely read and parse pipeline summary JSON files from completed runs.

#### Scenario: Read QC pipeline summary

- **GIVEN** a completed QC pipeline run with `10_pipeline_summary.json`
- **WHEN** `_read_pipeline_summary()` is called with the run path
- **THEN** the function SHALL return a dictionary with parsed JSON contents
- **AND** the function SHALL handle missing files by returning an empty dictionary

#### Scenario: Read summary with malformed JSON

- **GIVEN** a pipeline run with corrupted `pipeline_summary.json`
- **WHEN** `_read_pipeline_summary()` is called
- **THEN** the function SHALL log a warning
- **AND** the function SHALL return an empty dictionary
- **AND** the summary generation SHALL continue without crashing

#### Scenario: Extract QC metrics from summary

- **GIVEN** a valid QC pipeline summary JSON
- **WHEN** metrics are extracted for the summary table
- **THEN** `final_data.n_samples` SHALL map to Samples column
- **AND** `final_data.n_traits` SHALL map to Traits column
- **AND** `final_data.n_genotypes` SHALL map to Genotypes column
- **AND** `configuration.heritability.threshold` SHALL map to H² Threshold column
- **AND** `step_summaries.heritability_filter.mean_heritability_retained` SHALL map to Mean H² column

### Requirement: Removed Traits Documentation

The system SHALL document which traits were removed during QC for each dataset.

#### Scenario: List removed traits per dataset

- **GIVEN** a QC pipeline run that filtered traits by heritability
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a subsection under QC results titled "### Removed Traits by Dataset"
- **AND** each dataset SHALL list the traits removed with their heritability values if available

#### Scenario: No traits removed

- **GIVEN** a QC pipeline run where all traits passed heritability threshold
- **WHEN** the summary is generated
- **THEN** the removed traits section SHALL indicate "No traits removed" for that dataset

#### Scenario: Heritability filtering disabled

- **GIVEN** a QC pipeline run with `heritability.enabled: false`
- **WHEN** the summary is generated
- **THEN** the H² Threshold column SHALL display "Disabled"
- **AND** the Mean H² column SHALL display "N/A"
- **AND** no removed traits SHALL be listed for that dataset