# Design: Fix Pipeline Runner Config Preservation

## Context

The `PipelineRunner` class auto-updates config paths when running dependent pipelines (e.g., viz configs need to point to QC outputs). Currently this uses:
1. `yaml.safe_load()` to parse
2. Direct dict mutation to update paths
3. `yaml.dump()` to write

This destroys config structure and ignores user intent.

## Goals

- Preserve the original CSV filename choice when updating paths
- Preserve all YAML structure: comments, key order, formatting, quoting
- Fix the one pre-existing config bug

## Non-Goals

- Adding new configuration options
- Changing which configs get updated
- Modifying the QC mapping mechanism

## Decisions

### Decision 1: Extract filename from original path, append to new directory

**What**: When updating a path like `old/path/to/07_data_outliers_removed.csv`:
1. Extract the filename: `07_data_outliers_removed.csv`
2. Get the new QC output directory
3. Combine: `new/qc/output/07_data_outliers_removed.csv`

**Why**: This respects user's explicit choice of which data file to use.

**Alternatives considered**:
- Add a config option `use_heritability_filtered: bool` — adds complexity, doesn't fix the core issue
- Always use a specific file — ignores user intent

### Decision 2: Use regex substitution instead of yaml.dump()

**What**: Replace path values using regex while preserving surrounding text:
```python
# Instead of yaml.dump(), use:
content = config_path.read_text()
new_content = re.sub(
    r'(exp1_data_path:\s*["\']?)([^"\'\n]+)(["\']?)',
    rf'\g<1>{new_path}\g<3>',
    content
)
```

**Why**: Preserves:
- All comments
- Key ordering
- String quoting style
- Blank lines and formatting

**Alternatives considered**:
- ruamel.yaml (round-trip YAML) — adds dependency, still imperfect with comments
- String template approach — fragile with complex YAML

### Decision 3: Fix the config file directly

**What**: Update `cross_platform_rootcore_vs_cylinder.yaml` line 15 from `10_final_data.csv` to `07_data_outliers_removed.csv`.

**Why**: This config was created with an incorrect path that doesn't match the documented intent ("QC'd but NOT heritability filtered").

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Regex may fail on unusual YAML formatting | Use flexible patterns, test with real configs |
| Path may contain special regex characters | Use `re.escape()` on path components |
| Windows path separators | Normalize to forward slashes before substitution |

## Migration Plan

1. Update `pipeline_runner.py` with new methods
2. Fix the buggy config file
3. Re-run pipelines to verify correct behavior
4. No breaking changes to CLI or API
