## ADDED Requirements

### Requirement: Configuration Comparison in Summary

The system SHALL include a comprehensive configuration comparison section in the run summary document.

#### Scenario: Config comparison section generated

- **WHEN** pipeline run completes with multiple configs
- **THEN** the summary SHALL include a "## Configuration Comparison" section
- **AND** the section SHALL appear after the pipeline results tables and before the Methods section

#### Scenario: QC config parameters displayed

- **GIVEN** one or more QC pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### QC Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each config organized by section

#### Scenario: All cleanup parameters included

- **GIVEN** QC configs with cleanup sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `cleanup.max_nan_fraction`
- **AND** the table SHALL include `cleanup.max_zeros_per_trait`
- **AND** the table SHALL include `cleanup.max_nans_per_trait`
- **AND** the table SHALL include `cleanup.min_samples_per_trait`

#### Scenario: All outlier detection parameters included

- **GIVEN** QC configs with outlier_detection sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `outlier_detection.traditional_methods`
- **AND** the table SHALL include `outlier_detection.clustering_methods`
- **AND** the table SHALL include `outlier_detection.mahalanobis.variance_threshold`
- **AND** the table SHALL include `outlier_detection.mahalanobis.use_chi_squared`
- **AND** the table SHALL include `outlier_detection.mahalanobis.chi2_percentile`

#### Scenario: All outlier removal parameters included

- **GIVEN** QC configs with outlier_removal sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `outlier_removal.strategy`
- **AND** the table SHALL include `outlier_removal.method`

#### Scenario: All heritability parameters included

- **GIVEN** QC configs with heritability sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `heritability.enabled`
- **AND** the table SHALL include `heritability.threshold`
- **AND** the table SHALL include `heritability.generate_diagnostics`

#### Scenario: All PCA parameters included

- **GIVEN** QC configs with pca sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `pca.n_components`
- **AND** the table SHALL include `pca.feature_selection_strategy`

#### Scenario: All visualization parameters included

- **GIVEN** QC configs with visualization sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `visualization.dpi`
- **AND** the table SHALL include `visualization.figsize`
- **AND** the table SHALL include `visualization.title_fontsize`
- **AND** the table SHALL include `visualization.label_fontsize`
- **AND** the table SHALL include `visualization.tick_fontsize`
- **AND** the table SHALL include `visualization.legend_fontsize`
- **AND** the table SHALL include `visualization.figure_format`
- **AND** the table SHALL include `visualization.enable_batched_plots`
- **AND** the table SHALL include `visualization.batched_plot_threshold`
- **AND** the table SHALL include `visualization.batch_size`

#### Scenario: All adaptive sizing parameters included

- **GIVEN** QC configs with adaptive_sizing sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `adaptive_sizing.enabled`
- **AND** the table SHALL include `adaptive_sizing.base_width`
- **AND** the table SHALL include `adaptive_sizing.base_height`
- **AND** the table SHALL include `adaptive_sizing.width_per_item`
- **AND** the table SHALL include `adaptive_sizing.height_per_item`
- **AND** the table SHALL include `adaptive_sizing.min_width`
- **AND** the table SHALL include `adaptive_sizing.max_width`
- **AND** the table SHALL include `adaptive_sizing.min_height`
- **AND** the table SHALL include `adaptive_sizing.max_height`

#### Scenario: Root core parameters included when present

- **GIVEN** QC configs with root_core sections
- **WHEN** the config comparison is generated
- **THEN** the table SHALL include `root_core.core_qc.enabled`
- **AND** the table SHALL include `root_core.core_qc.max_missing_proportion`
- **AND** the table SHALL include `root_core.core_qc.remove_outliers`
- **AND** the table SHALL include `root_core.core_qc.detect_value_outliers`
- **AND** the table SHALL include `root_core.core_qc.max_deviation_from_median`
- **AND** the table SHALL include `root_core.generate_depth_profiles`

#### Scenario: Viz config parameters displayed

- **GIVEN** one or more Viz pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### Visualization Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each viz config

#### Scenario: Cross-platform config parameters displayed

- **GIVEN** one or more Cross-Platform pipelines were executed
- **WHEN** the summary is generated
- **THEN** the summary SHALL include a "### Cross-Platform Pipeline Configuration" subsection
- **AND** the subsection SHALL display ALL parameters from each cross-platform config

#### Scenario: Table format with datasets as columns

- **GIVEN** multiple configs of the same pipeline type
- **WHEN** the config comparison table is generated
- **THEN** the first column SHALL be "Parameter"
- **AND** each subsequent column SHALL be the pipeline_name or config filename
- **AND** each row SHALL show one parameter with its value for each config

#### Scenario: Missing parameters shown as N/A

- **GIVEN** a parameter exists in some configs but not others
- **WHEN** the config comparison table is generated
- **THEN** configs without the parameter SHALL display "N/A" in that cell

#### Scenario: List values formatted correctly

- **GIVEN** a parameter value is a list (e.g., traditional_methods: [mahalanobis])
- **WHEN** the config comparison table is generated
- **THEN** the list SHALL be formatted as comma-separated values (e.g., "mahalanobis")

#### Scenario: Nested dict values flattened

- **GIVEN** a parameter has nested structure (e.g., mahalanobis.variance_threshold)
- **WHEN** the config comparison table is generated
- **THEN** the parameter name SHALL use dot notation (e.g., "mahalanobis.variance_threshold")
- **AND** the value SHALL be the leaf value

#### Scenario: Single config still shows table

- **GIVEN** only one config of a pipeline type was executed
- **WHEN** the config comparison is generated
- **THEN** the table SHALL still be generated with that single config as a column
- **AND** this allows users to see all parameters used even for single-config runs
$