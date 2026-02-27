# Proposal: Fix `/configure-run-all` by Using Golden Templates Instead of Scratch Generation

## Why

The current `/configure-run-all` command attempts to write pipeline configs from scratch by collecting parameters and using the Write tool. This approach has critical flaws:

1. **Incomplete schema coverage**: The command documents what parameters to collect, but not the full YAML structure the code expects. Many required fields are missing from the generated configs.
2. **No schema validation**: Configs are written without calling `validate_qc_config()`, allowing invalid configs to be created.
3. **Fragile to code changes**: If the config schema evolves (new required fields added), the command silently produces outdated/broken configs.
4. **No reference to working examples**: The command doesn't leverage the existing, validated configs in `configs/active/` that are known to work.

The correct approach is: **copy a complete "golden" template that has been verified against the config schema, then customize only the fields that need to change**. This guarantees schema completeness and leverages known-working configs as the source of truth.

## What Changes

1. **Add golden templates to `configs/templates/`**:
   - `qc_template_grouped.yaml` — QC config with `group_by` enabled, all schema fields present
   - `qc_template_ungrouped.yaml` — QC config without `group_by`, all schema fields present
   - `viz_template_with_images.yaml` — Viz config with `image_dir` and hover images enabled
   - `viz_template_no_images.yaml` — Viz config with `image_dir: null`
   - `run_manifest_template.yaml` — Run manifest template

   These templates:
   - Are derived from known-working configs in `configs/active/` (alfalfa wave 1, turface 150genotypes)
   - Include ALL fields the code expects (not just the ones that need customization)
   - Use clearly marked placeholders for required fields (e.g., `FILL_IN_CSV_PATH`)
   - Are validated with `validate_qc_config()` before committing
   - Include inline comments explaining each parameter's purpose

2. **Update `.claude/commands/configure-run-all.md`** to replace the "write from scratch" workflow with "copy and customize":
   - Ask: grouped or ungrouped? with or without images?
   - Copy the appropriate template pair (`qc_template_*.yaml` + `viz_template_*.yaml`)
   - Walk through ONLY the fields that need customization (Read template → Edit specific fields → Write modified template):
     - **Required**: `csv_path`, `columns.barcode`, `columns.genotype`, `columns.replicate`, `group_by` column name (if grouped), `image_dir` path (if with images)
     - **Configurable**: heritability threshold, PCA `n_top_features`, UMAP `n_neighbors`, min_samples_per_trait
   - Call `validate_qc_config()` on the generated QC config before writing any file
   - Backup existing configs to `configs/archive/` with timestamp before overwriting
   - Git commit with reproducibility SHA

3. **Guardrails**:
   - Templates are the ground truth for schema completeness (validated before commit)
   - `validate_qc_config()` is called on the final generated config before writing
   - No silent overwrites — backup + explicit confirmation required

4. **Testing**:
   - Test that golden templates pass `validate_qc_config()` and `validate_viz_config()`
   - Test that the command can successfully customize and validate a template without breaking the schema

## Impact

**Affected specs:**
- `developer-tooling` — MODIFIED: "Interactive Analysis Configuration Command" (replace write-from-scratch workflow with copy-and-customize)
- `config-management` — ADDED: "Golden Analysis Templates" (new requirement for complete validated templates in `configs/templates/`)

**Affected code:**
- `.claude/commands/configure-run-all.md` — slash command workflow updated
- `configs/templates/` — 5 new golden template files added

**Breaking changes:** None. The command name stays the same. Existing configs are unaffected. Users who manually edit configs can continue to do so.

**Migration:** No user action required. Existing `add-config-authoring-command` change will be replaced by this improved version before archiving.
