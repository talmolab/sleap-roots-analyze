## ADDED Requirements

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
