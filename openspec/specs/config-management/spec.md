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

### Requirement: Git-Anchored Active Configs

Active pipeline configuration files in `configs/active/` SHALL be committed to git before any pipeline run so that the exact configuration used for an analysis can be permanently recovered from version history.

This requirement exists because pipeline outputs (large CSV files, figures) are gitignored, but the configuration that produced them must be reproducible. A git SHA on the config commit is the minimal reproducibility artifact that links a result to its exact analysis parameters.

#### Scenario: Config commit created before pipeline run

- **WHEN** a user finalizes pipeline configs using `/configure-run-all`
- **THEN** the command SHALL commit the config files to git with a descriptive message
- **AND** the commit message SHALL include: the run_name from the manifest, the dataset path, and the ISO date
- **AND** the resulting git SHA SHALL be reported to the user as the reproducibility anchor

#### Scenario: Commit message format

- **WHEN** the git commit is created
- **THEN** the commit message SHALL follow the format:
  ```
  chore: configure analysis "{run_name}" ({date})

  Dataset: {csv_path}
  Config files: {list of committed config paths}
  ```
- **AND** the message SHALL be machine-parseable for future tooling

#### Scenario: Git anchor preserved in config header

- **WHEN** configs are committed and the SHA is known
- **THEN** the run manifest header SHOULD include a comment referencing the commit SHA
- **AND** this allows the manifest itself to document its own reproducibility anchor

#### Scenario: Git commit failure handled gracefully

- **WHEN** a git commit cannot be created (e.g., no changes staged, detached HEAD, repository not initialized)
- **THEN** the system SHALL issue a clear warning to the user explaining that configs are NOT yet anchored to git
- **AND** the system SHALL NOT crash or refuse to write config files
- **AND** the warning SHALL instruct the user to manually run `git add configs/active/ && git commit -m "..."`

---

### Requirement: Backup Before Overwrite

The system SHALL protect existing active configuration files from accidental overwrite by offering timestamped backups before any modification.

This requirement exists because active configs represent scientific decisions. Overwriting one silently destroys the record of those decisions unless they were previously committed to git.

#### Scenario: Backup offered when active config exists

- **WHEN** a user is about to write a new config to a path that already exists in `configs/active/`
- **THEN** the system SHALL detect the existing file
- **AND** the system SHALL offer to back it up to `configs/archive/<original-filename>_backup_<YYYYMMDD_HHMMSS>.yaml`
- **AND** the system SHALL NOT proceed with the overwrite until the user explicitly confirms

#### Scenario: Backup naming is unambiguous

- **WHEN** a backup is created
- **THEN** the backup filename SHALL include the original filename stem, the literal string `_backup_`, and a timestamp in `YYYYMMDD_HHMMSS` format
- **AND** two backups created in the same second SHALL NOT overwrite each other (timestamp resolution is sufficient for interactive use)

#### Scenario: Archive directory is gitignored

- **WHEN** backups are written to `configs/archive/`
- **THEN** the `configs/archive/` directory SHALL be listed in `.gitignore`
- **AND** backup files SHALL NOT be committed to git (they are local safety nets, not reproducibility artifacts)
- **AND** the committed configs in `configs/active/` are the canonical reproducibility artifacts

#### Scenario: No backup needed for new files

- **WHEN** a config path in `configs/active/` does not yet exist
- **THEN** no backup SHALL be created or offered
- **AND** the file SHALL be written directly without prompting

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

### Requirement: Golden Analysis Templates

The system SHALL maintain a set of "golden" pipeline configuration templates in `configs/templates/` that serve as the authoritative source for config schema completeness and known-working parameter combinations.

Golden templates are configuration files that:
1. **Include ALL required fields** expected by the config schema (no missing sections or parameters)
2. **Pass validation** when placeholders are replaced with valid values (`validate_qc_config()` / `validate_viz_config()` must succeed)
3. **Use clear placeholders** for fields that require user customization (e.g., `FILL_IN_CSV_PATH`, `FILL_IN_BARCODE_COLUMN`)
4. **Include inline comments** explaining the purpose and recommended values for each parameter
5. **Are derived from known-working configs** in `configs/active/` that have been verified in real analyses

#### Scenario: Golden QC templates for grouped and ungrouped analyses

- **GIVEN** the `configs/templates/` directory
- **THEN** it SHALL contain `qc_template_grouped.yaml` (with `data.group_by` enabled)
- **AND** it SHALL contain `qc_template_ungrouped.yaml` (with `data.group_by: null`)
- **AND** both templates SHALL pass `validate_qc_config()` when placeholders are replaced with valid test values
- **AND** both templates SHALL include ALL sections required by the QC pipeline: `pipeline_name`, `data`, `columns`, `cleanup`, `outlier_detection`, `outlier_removal`, `pca`, `heritability`, `visualization`, `adaptive_sizing`, `logging`

#### Scenario: Golden Viz templates for with-images and no-images analyses

- **GIVEN** the `configs/templates/` directory
- **THEN** it SHALL contain `viz_template_with_images.yaml` (with `data.image_dir` set to a placeholder path)
- **AND** it SHALL contain `viz_template_no_images.yaml` (with `data.image_dir: null`)
- **AND** both templates SHALL pass `validate_viz_config()` when placeholders are replaced
- **AND** both templates SHALL include ALL sections required by the Viz pipeline: `pipeline_name`, `data`, `columns`, `statistics`, `pca`, `umap`, `clustering`, `heritability`, `interesting_genotypes`, `static_viz`, `interactive_viz`, `dashboard`, `summary`, `logging`

#### Scenario: Golden run manifest template

- **GIVEN** the `configs/templates/` directory
- **THEN** it SHALL contain `run_manifest_template.yaml`
- **AND** the template SHALL include: `run_name`, `description`, `qc_configs`, `viz_configs`, `qc_mapping`

#### Scenario: Templates use recognizable placeholder syntax

- **GIVEN** any golden template file
- **WHEN** a user or tool reads the file
- **THEN** placeholder values SHALL be clearly distinguishable from real values
- **AND** placeholders SHALL use a consistent naming pattern (e.g., `FILL_IN_*`, or `<placeholder-name>`)

#### Scenario: Templates are validated before commit

- **GIVEN** a new or updated golden template
- **WHEN** it is committed to the repository
- **THEN** it SHALL have been validated by calling the appropriate validation function (`validate_qc_config()`, `validate_viz_config()`, or equivalent)
- **AND** the validation test SHALL replace placeholders with valid dummy values before calling the validator
- **AND** the validation test SHALL pass (no exceptions raised)

#### Scenario: Templates match the schema of working configs

- **GIVEN** a golden template (e.g., `qc_template_grouped.yaml`)
- **AND** a known-working config in `configs/active/` (e.g., `qc_alfalfa_gwas_wave_1_grouped.yaml`)
- **WHEN** the template and working config are compared
- **THEN** the template SHALL include all top-level sections present in the working config
- **AND** the template SHALL include all required fields within each section
- **AND** the template MAY omit optional fields if they are not commonly used

#### Scenario: Templates document parameter recommendations

- **GIVEN** a golden template file
- **WHEN** a user opens the file in a text editor
- **THEN** inline comments SHALL explain:
  - The purpose of each major section
  - Recommended value ranges for critical parameters (e.g., heritability threshold: 0.30–0.60)
  - Which parameters are rarely changed vs frequently customized
  - Dependencies between parameters (e.g., "if UMAP enabled, set n_neighbors")

