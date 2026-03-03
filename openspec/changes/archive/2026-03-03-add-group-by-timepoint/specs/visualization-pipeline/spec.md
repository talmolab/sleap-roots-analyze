## ADDED Requirements

### Requirement: Group-Based Visualization Execution

The visualization pipeline SHALL support the same group-by functionality as the QC pipeline for consistent per-group analysis.

#### Scenario: Viz pipeline groups like QC
- **GIVEN** a QC output with groups processed by plant_age_days
- **WHEN** viz pipeline runs with `group_by: "plant_age_days"`
- **THEN** viz SHALL create separate visualizations for each timepoint
- **AND** output structure SHALL mirror QC: `viz_<pipeline>_plant_age_days_<value>_<timestamp>/`

#### Scenario: PCA computed per group
- **GIVEN** a grouped viz pipeline (day_7, day_14, day_21)
- **WHEN** PCA analysis is performed
- **THEN** each group SHALL have independent PC loadings and variance explained
- **AND** PC1 for day_7 MAY differ from PC1 for day_14 (developmental differences)

#### Scenario: Interactive plots per group
- **GIVEN** a viz pipeline grouped by plant_age_days
- **WHEN** interactive PCA plots are generated
- **THEN** each group SHALL have its own `pca_interactive.html` file
- **AND** plot title SHALL indicate the group (e.g., "PCA: plant_age_days = 7")

#### Scenario: Statistical analysis per group
- **GIVEN** a viz pipeline grouped by experiment_id
- **WHEN** ANOVA and heritability are calculated
- **THEN** statistics SHALL be computed independently per group
- **AND** `08_heritability_results.csv` SHALL contain group-specific H² estimates

#### Scenario: Summary reports per group
- **GIVEN** a grouped viz pipeline
- **WHEN** summary markdown is generated
- **THEN** each group SHALL have its own `summary_report.md`
- **AND** the report SHALL document the group identifier and sample count
