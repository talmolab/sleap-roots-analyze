# Implementation Tasks

## Status: ✅ COMPLETE (2026-02-23)

All tasks implemented and tested. Templates exist, pass validation, and are integrated with `/configure-run-all` command.

## 1. Golden Template Files

- [x] 1.1 Create `configs/templates/qc_template_grouped.yaml` based on `qc_alfalfa_gwas_wave_1_grouped.yaml`
      - Replace real values with placeholders: `FILL_IN_CSV_PATH`, `FILL_IN_PIPELINE_NAME`
      - Add inline comments explaining each parameter
      - Validate with `validate_qc_config()` before committing
- [x] 1.2 Create `configs/templates/qc_template_ungrouped.yaml` based on `qc_turface_150genotypes.yaml`
      - Same placeholder + comment approach
      - Set `group_by: null`
      - Validate before committing
- [x] 1.3 Create `configs/templates/viz_template_with_images.yaml` based on `viz_alfalfa_gwas_wave_1_grouped.yaml`
      - Use placeholders for `image_dir`, `csv_path`
      - Ensure `show_images_on_hover: true`, `generate_image_grids: true`
      - Validate with `validate_viz_config()` before committing
- [x] 1.4 Create `configs/templates/viz_template_no_images.yaml`
      - Copy from `viz_template_with_images.yaml`
      - Set `image_dir: null`, `show_images_on_hover: false`, `generate_image_grids: false`
      - Validate before committing
- [x] 1.5 Create `configs/templates/run_manifest_template.yaml` based on `run_manifest_alfalfa_wave1_grouped.yaml`
      - Use placeholders for `run_name`, config paths
      - Keep minimal header structure

## 2. Slash Command Redesign

- [x] 2.1 Rewrite `.claude/commands/configure-run-all.md` to use copy-and-customize workflow
      - Replace "write configs using Write tool" with "copy template using Read + Edit"
      - Document template selection (grouped/ungrouped × with images/without images)
      - Define which fields are customized vs which stay as defaults
- [x] 2.2 Add `validate_qc_config()` call before writing QC config
      - Use Python API: `from sleap_roots_analyze.pipeline.config.utils import validate_qc_config`
      - Call after customization, before Write (Step 7 in command)
      - Show validation errors to user if it fails
- [x] 2.3 Add backup check with timestamp
      - Use `make_backup_path()` from `config_authoring.py`
      - Require explicit user confirmation before overwriting (Step 5)
- [x] 2.4 Update Q&A sequence to match template-based workflow
      - Removed parameter explanations that are now in template comments
      - Focus on: which template? what are your paths? what are your column names?

## 3. Tests

- [x] 3.1 Test: Golden templates pass `validate_qc_config()` and `validate_viz_config()`
      - Unit test in `tests/test_golden_templates.py`
      - Load each template, replace placeholders with dummy values, validate
      - 11/11 tests passing
- [x] 3.2 Test: Template customization preserves schema completeness
      - Integration test: copy template → customize fields → validate → assert success
      - Covered by `test_template_validates_with_filled_placeholders` tests
- [x] 3.3 Test: Templates have all required fields
      - Compare template keys against a known-complete config
      - Validation tests ensure no missing top-level sections

## 4. Spec Deltas

- [x] 4.1 Write `specs/developer-tooling/spec.md` delta
      - MODIFIED: "Interactive Analysis Configuration Command"
      - Replace "write configs from scratch" with "copy golden template and customize"
- [x] 4.2 Write `specs/config-management/spec.md` delta
      - ADDED: "Golden Analysis Templates"
      - Define requirement: templates must pass validation, include all schema fields, use clear placeholders

## 5. Documentation

- [x] 5.1 Update `configs/templates/README.md` to reference the new golden templates
      - Added section explaining what "golden" means (complete + validated)
      - Show which template to choose for each use case
- [x] 5.2 Update `.claude/commands/configure-run-all.md` header to clarify the new approach
      - "This command copies a complete validated template and customizes it"

## 6. Validation

- [x] 6.1 Run `openspec validate update-configure-run-all-templates --strict`
- [x] 6.2 Fix any validation issues

---

## Test Results

```bash
uv run pytest tests/test_golden_templates.py -v
# 11 passed, 2 warnings in 0.25s
```

All golden templates:
- Exist in `configs/templates/`
- Pass validation with filled placeholders
- Are used by `/configure-run-all` command
- Are documented in templates README
