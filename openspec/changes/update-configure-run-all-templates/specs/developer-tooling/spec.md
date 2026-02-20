## MODIFIED Requirements

### Requirement: Interactive Analysis Configuration Command

The system SHALL provide a `/configure-run-all` Claude Code slash command that interactively guides the user through creating a complete, scientifically sound set of pipeline configuration files (QC config, Viz config, run manifest) for a new analysis **by copying and customizing validated golden templates**.

The command SHALL embody three core scientific values:
1. **Reproducibility**: All config files SHALL be committed to git so they are permanently tied to a specific codebase state (git SHA).
2. **Metadata preservation**: Config file headers SHALL document dataset identity, analysis date, author intent, and parameter rationale.
3. **Schema completeness**: All generated configs SHALL pass `validate_qc_config()` / `validate_viz_config()` before being written, ensuring no missing required fields.

#### Scenario: Template selection

- **WHEN** the user invokes `/configure-run-all`
- **THEN** the command SHALL ask: "Is this a grouped analysis (e.g., multiple timepoints)?"
- **AND** the command SHALL ask: "Are images available for visualization?"
- **AND** based on answers, the command SHALL select the appropriate golden template pair:
  - Grouped + with images → `qc_template_grouped.yaml` + `viz_template_with_images.yaml`
  - Grouped + no images → `qc_template_grouped.yaml` + `viz_template_no_images.yaml`
  - Ungrouped + with images → `qc_template_ungrouped.yaml` + `viz_template_with_images.yaml`
  - Ungrouped + no images → `qc_template_ungrouped.yaml` + `viz_template_no_images.yaml`

#### Scenario: Copy golden template

- **WHEN** the appropriate template is selected
- **THEN** the command SHALL use the Read tool to load the golden template from `configs/templates/<template_name>.yaml`
- **AND** the command SHALL preserve ALL fields from the template (no fields are dropped)

#### Scenario: Customize required fields only

- **AFTER** loading the template
- **THEN** the command SHALL collect the following REQUIRED field values interactively, one at a time:
  - `data.csv_path` — Path to the trait CSV file
  - `columns.barcode` — Column name for sample ID / plant barcode
  - `columns.genotype` — Column name for genotype / accession
  - `columns.replicate` — Column name for replicate / plant ID
  - `data.group_by` — (if grouped template) Column name to group by (e.g., `plant_age_days`)
  - `data.image_dir` — (if with-images template) Path to image directory
  - `data.output_dir` — Where pipeline outputs should be written
  - `pipeline_name` and `run_name` — Analysis name / identifier
- **AND** the command SHALL use the Edit tool or string replacement to update ONLY these fields in the loaded template
- **AND** all other fields (heritability threshold, PCA settings, UMAP settings, etc.) SHALL retain the template's default values unless the user explicitly requests to customize them

#### Scenario: Validate before writing

- **WHEN** all required fields have been customized
- **THEN** the command SHALL call `validate_qc_config()` on the customized QC config using the Python API
- **AND** if validation fails, the command SHALL show the error message to the user and ask them to fix the issue before proceeding
- **AND** the command SHALL NOT write any config file until validation passes

#### Scenario: Dataset inspection and guardrails

- **WHEN** the user provides a CSV path
- **THEN** the command SHALL read the CSV and report: total sample count, column names (with candidates for barcode/genotype/replicate roles), numeric trait count, and candidate group_by columns (columns with ≤20 unique values)
- **AND** the command SHALL flag any candidate group with fewer than 30 samples with a WARNING (Mahalanobis chi-squared reliability requires n≥30)
- **AND** the command SHALL flag any experiment where fewer than 3 replicates per genotype exist in any group with a WARNING (heritability estimation requires ≥3 replicates per genotype)
- **AND** the command SHALL recommend UMAP n_neighbors using: `min(15, max(2, n_samples // 4))`

#### Scenario: Backup before overwriting active configs

- **WHEN** a config file already exists at the target path in `configs/active/`
- **THEN** the command SHALL inform the user that an existing config will be overwritten
- **AND** the command SHALL offer to save a timestamped backup to `configs/archive/<original-name>_backup_<YYYYMMDD_HHMMSS>.yaml`
- **AND** the command SHALL NOT overwrite any existing config without explicit user confirmation

#### Scenario: Config file writing with self-documenting headers

- **WHEN** the user approves the configuration
- **THEN** the command SHALL write QC config, Viz config, and run manifest to `configs/active/`
- **AND** each config header SHALL include: dataset name, input CSV path, analysis date, and key parameter choices with brief rationale
- **AND** the run manifest header SHALL include the CLI command to reproduce the run

#### Scenario: User validation gate

- **WHEN** configs have been written to disk
- **THEN** the command SHALL display the full content of each config file for the user to review
- **AND** the command SHALL highlight (in text) the most consequential parameters: heritability threshold, outlier method, group_by column, min_samples_per_trait
- **AND** the command SHALL wait for explicit user approval ("looks good" / "yes" / "run it") before offering to proceed
- **AND** the command SHALL NOT invoke `/run-pipelines` automatically — it SHALL remind the user of the exact command to run

#### Scenario: Git commit after user approval

- **WHEN** the user approves the configs
- **THEN** the command SHALL stage the new/modified config files in `configs/active/`
- **AND** the command SHALL create a git commit with a message that includes: analysis run_name, dataset path, and ISO date
- **AND** the command SHALL report the resulting git SHA to the user as the reproducibility anchor
- **AND** if git commit fails (e.g., no changes, detached HEAD), the command SHALL warn the user clearly and continue without crashing
