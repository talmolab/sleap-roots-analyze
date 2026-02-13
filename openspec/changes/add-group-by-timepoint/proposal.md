# Proposal: Add Group-By Timepoint Support

## Why

Scientific experiments often collect root phenotype data at multiple timepoints (e.g., plant_age_days: 7, 14, 21). Currently, running QC/viz pipelines on multi-timepoint data lumps all days together, confounding temporal effects with genetic effects. This prevents proper per-timepoint heritability estimates and trait analysis.

Researchers need to analyze each timepoint separately to understand trait development over time and calculate valid heritability estimates per developmental stage.

## What Changes

- Add `--group-by <column>` CLI flag to `qc`, `viz`, and `run-all` commands
- Add `data.group_by: str | None` config field (CLI overrides config)
- Split input data by unique values in group-by column (e.g., plant_age_days: 7, 14, 21)
- Run separate pipeline instances per group with isolated outputs
- Output directory structure: `<output_base>/<column>_<value>_<timestamp>/`
- Validate groups using existing `cleanup.min_samples_per_trait` threshold
- Skip groups with insufficient samples (log warning)
- Support in `run-all` manifest: per-pipeline `group_by` field

**Scope limitations (defer to future):**
- Single column grouping only (no multi-column like `plant_age_days,experiment_id`)
- No filter-only mode (grouping runs all groups, not selective filtering)

## Impact

**Affected specs:**
- `config-management` - Add `group_by` to DataConfig
- `cli-pipeline` - Add `--group-by` flag to qc/viz commands
- `qc-pipeline-root-core` - Handle grouped execution, validation per group
- `visualization-pipeline` - Handle grouped execution

**Affected code:**
- `src/sleap_roots_analyze/cli.py` - Add CLI flag parsing
- `src/sleap_roots_analyze/pipeline/config/components.py` - DataConfig.group_by field
- `src/sleap_roots_analyze/pipeline/pipelines/base_pipeline.py` - Group execution logic
- `src/sleap_roots_analyze/pipeline/steps/load_data.py` - Data splitting by group
- `tests/` - New test files for grouping functionality

**Breaking changes:** None (purely additive feature)

**Migration:** No migration needed - existing configs/workflows unchanged
