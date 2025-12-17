# Fix Pipeline Runner Config Preservation

## Why

The `PipelineRunner._update_viz_config()` and `_update_cross_platform_config()` methods have critical bugs that silently corrupt user config files:

1. **Hardcoded file choice**: Both methods unconditionally use `10_final_data.csv` (heritability-filtered) regardless of what the original config specifies. Users who intentionally point to `07_data_outliers_removed.csv` (QC'd but NOT heritability-filtered) have their choice silently overwritten.

2. **YAML dump destroys config structure**: Using `yaml.dump()` strips all comments, reorders keys alphabetically, and changes string quoting styles—making updated configs unrecognizable and losing documentation.

This is particularly insidious because the user sees the correct paths in their source configs but the pipeline runs with silently modified paths.

## What Changes

- **FIX**: Preserve the original data file choice (filename) when updating directory paths
- **FIX**: Use regex-based path substitution instead of `yaml.dump()` to preserve comments, key order, and formatting
- **FIX**: Update one existing buggy config (`cross_platform_rootcore_vs_cylinder.yaml` line 15 incorrectly uses `10_final_data.csv`)

## Impact

- Affected specs: `pipeline-runner-skill`
- Affected code: `src/sleap_roots_analyze/pipeline_runner.py` (lines 251-340)
- Affected configs: `configs/active/cross_platform/cross_platform_rootcore_vs_cylinder.yaml`

## Evidence

From `pipeline_runs/2025-12-16_122102/cross_platform/_updated_cross_platform_turface19_vs_cylinder.yaml`:
```yaml
# Original config specified 07_data_outliers_removed.csv
# But _updated_ config has:
exp1_data_path: pipeline_runs\...\10_final_data.csv  # WRONG
exp2_data_path: pipeline_runs\...\10_final_data.csv  # WRONG
```

The updated configs also:
- Lost all comments explaining the data choices
- Reordered keys alphabetically
- Changed `version: "1.0"` to `version: '1.0'`
