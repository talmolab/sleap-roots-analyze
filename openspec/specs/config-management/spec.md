# config-management Specification

## Purpose
TBD - created by archiving change audit-notebook-config-reproducibility. Update Purpose after archive.
## Requirements
### Requirement: Notebook-Config Parameter Consistency

Pipeline configuration files SHALL exactly replicate the analysis parameters used in corresponding Jupyter notebooks to ensure scientific reproducibility.

#### Scenario: QC config matches notebook parameters
- **GIVEN** a QC notebook with cleanup parameters (max_nan_fraction=0.0, max_zeros_per_trait=0.5)
- **WHEN** the corresponding QC config file is loaded
- **THEN** all cleanup parameters SHALL match the notebook values exactly

#### Scenario: Viz config matches notebook parameters
- **GIVEN** a visualization notebook with PCA variance threshold=0.75
- **WHEN** the corresponding viz config file is loaded
- **THEN** the PCA n_components parameter SHALL be 0.75

#### Scenario: Cross-platform config matches notebook data sources
- **GIVEN** a cross-platform notebook using specific QC output files
- **WHEN** the corresponding cross-platform config is loaded
- **THEN** the data paths SHALL point to the same QC outputs (or equivalent pipeline outputs)

---

### Requirement: Config Header Documentation

Configuration files SHALL include header documentation that traces parameters back to source notebooks.

#### Scenario: Config header identifies source notebook
- **GIVEN** a config file based on analysis from `trait_qc_cylinders_20251105.ipynb`
- **WHEN** the config file header is read
- **THEN** it SHALL include the source notebook filename and date

#### Scenario: Config header documents verification status
- **GIVEN** a config file that has been audited against its source notebook
- **WHEN** the config header is read
- **THEN** it SHALL include verification date and verification status

#### Scenario: Config header explains intentional deviations
- **GIVEN** a config file where heritability threshold differs from another dataset
- **WHEN** the config header is read
- **THEN** it SHALL document the rationale for the parameter choice

---

### Requirement: Parameter Categorization

Configuration parameters SHALL be organized into well-defined categories for systematic validation.

#### Scenario: Cleanup parameters are grouped
- **GIVEN** a QC config file
- **WHEN** parameters are extracted for validation
- **THEN** cleanup parameters (max_nan_fraction, max_zeros_per_trait, max_nans_per_trait, min_samples_per_trait) SHALL be in the `cleanup` section

#### Scenario: Outlier detection parameters are grouped
- **GIVEN** a QC config file
- **WHEN** parameters are extracted for validation
- **THEN** outlier detection parameters (methods, mahalanobis settings, etc.) SHALL be in the `outlier_detection` section

#### Scenario: Visualization parameters are grouped
- **GIVEN** a viz config file
- **WHEN** parameters are extracted for validation
- **THEN** PCA, UMAP, and plotting parameters SHALL be in their respective sections

---

### Requirement: Dataset-Specific Parameter Documentation

Intentional parameter variations across datasets SHALL be documented with scientific rationale.

#### Scenario: Heritability threshold variation is documented
- **GIVEN** configs for turface_150 (H²≥0.40) and cylinder (H²≥0.60)
- **WHEN** the parameter reference table is consulted
- **THEN** it SHALL explain why cylinder uses a higher threshold (e.g., large trait count, stricter filtering needed)

#### Scenario: PCA variance threshold variation is documented
- **GIVEN** configs with different PCA variance thresholds (0.75 vs 0.80)
- **WHEN** the parameter reference table is consulted
- **THEN** it SHALL explain the rationale for each choice

---

### Requirement: Parameter Audit Trail

All configuration parameter changes SHALL be traceable to specific notebook cells or analysis decisions.

#### Scenario: Parameter source cell is documented
- **GIVEN** a config parameter `max_nan_fraction: 0.0`
- **WHEN** the parameter reference documentation is consulted
- **THEN** it SHALL indicate the notebook cell number where this parameter was defined (e.g., "Cell 5: MAX_NAN_FRACTION = 0.0")

#### Scenario: Parameter change is justified
- **GIVEN** a config parameter that differs from a previous version
- **WHEN** the config change history is reviewed
- **THEN** the change SHALL be linked to a specific notebook re-analysis or documented decision

---

### Requirement: Genotype Highlighting Reproducibility

Visualization configs SHALL preserve genotype highlighting lists used in published figures.

#### Scenario: Genotypes to color are specified
- **GIVEN** a notebook with `GENOTYPES_TO_COLOR = ["GH_7293", "GH_7378", "GH_7327"]`
- **WHEN** the corresponding viz config is loaded
- **THEN** `static_viz.genotypes_to_color` SHALL contain the exact same list

#### Scenario: Genotypes to highlight are specified
- **GIVEN** a notebook with `GENOTYPES_TO_HIGHLIGHT = ["GH_7401", "GH_7391", "GH_7361"]`
- **WHEN** the corresponding viz config is loaded
- **THEN** `static_viz.highlight_genotypes` SHALL contain the exact same list

---

### Requirement: Cross-Platform Analysis Data Lineage

Cross-platform analysis configs SHALL document the exact QC pipeline outputs used as input data.

#### Scenario: Data source paths reference pipeline outputs
- **GIVEN** a cross-platform analysis comparing turface and cylinder
- **WHEN** the config data paths are examined
- **THEN** they SHALL point to specific QC pipeline output directories or files (e.g., "run_YYYYMMDD_HHMMSS/10_final_data.csv")

#### Scenario: Data source QC settings are referenced
- **GIVEN** a cross-platform config using turface QC output
- **WHEN** the config documentation is read
- **THEN** it SHALL reference the QC config used to generate that data (e.g., "Generated by qc_turface_150genotypes.yaml")

---

### Requirement: Parameter Reference Table

A centralized parameter reference table SHALL document all key parameters for each dataset.

#### Scenario: Reference table includes all datasets
- **GIVEN** 4 main datasets (turface_150, turface_19, cylinder, root_coring)
- **WHEN** the parameter reference table is consulted
- **THEN** it SHALL have entries for QC, viz, and cross-platform configs for each dataset

#### Scenario: Reference table shows parameter sources
- **GIVEN** a parameter entry in the reference table
- **WHEN** the table is read
- **THEN** it SHALL show: parameter name, notebook value, config value, match status, and notebook cell reference

#### Scenario: Reference table highlights mismatches
- **GIVEN** a parameter that differs between notebook and config
- **WHEN** the reference table is viewed
- **THEN** the mismatch SHALL be clearly marked and explained (intentional variation vs. error to fix)

---

### Requirement: Config Validation Checklist

A validation checklist SHALL guide verification of new or updated configs against notebooks.

#### Scenario: Checklist covers all parameter categories
- **GIVEN** a new dataset config being created
- **WHEN** the validation checklist is used
- **THEN** it SHALL include checks for: data paths, cleanup params, outlier params, PCA params, heritability params, viz params, genotype highlighting

#### Scenario: Checklist requires documentation
- **GIVEN** a config being validated
- **WHEN** the checklist is completed
- **THEN** it SHALL require documenting the source notebook, verification date, and any intentional deviations

---

### Requirement: Reproducibility Guarantees

Running a pipeline with a config SHALL produce results equivalent to the corresponding notebook analysis.

#### Scenario: QC pipeline replicates notebook sample count
- **GIVEN** a notebook that produces 890 samples after QC
- **WHEN** the corresponding QC config is run through the pipeline
- **THEN** it SHALL produce 890 samples

#### Scenario: QC pipeline replicates notebook trait count
- **GIVEN** a notebook that retains 13 traits after heritability filtering
- **WHEN** the corresponding QC config is run through the pipeline
- **THEN** it SHALL retain 13 traits

#### Scenario: Viz pipeline replicates notebook PCA structure
- **GIVEN** a notebook that uses 2 PCs explaining 75% variance
- **WHEN** the corresponding viz config is run through the pipeline
- **THEN** it SHALL use the same number of PCs with the same variance threshold

---

