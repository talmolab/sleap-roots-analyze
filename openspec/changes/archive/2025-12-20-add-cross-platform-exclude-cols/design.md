## Context

Cross-platform analysis compares traits between two experiments (e.g., field vs turface, root core vs cylinder). It loads pre-cleaned CSVs from QC pipeline outputs. The QC pipeline preserves metadata columns (Ent, Sub, Cid, Sid, GID) in output CSVs even though they're excluded from analysis via `additional_exclude_cols`.

When cross-platform loads these CSVs, it has no mechanism to exclude these columns, causing spurious correlations like `Ent vs Root Shoot Ratio`.

## Goals / Non-Goals

**Goals:**
- Add column exclusion support to cross-platform pipeline
- Follow existing `additional_exclude_cols` pattern from QC/Viz pipelines
- Support separate exclusion lists for each experiment (different metadata columns)
- Maintain backward compatibility (optional parameters with None default)

**Non-Goals:**
- Remove metadata columns from QC output CSVs (they may be useful for downstream analysis)
- Change how QC/Viz pipelines handle exclusions (already working correctly)
- Add automatic detection of metadata columns (explicit configuration is clearer)

## Decisions

### Decision: Separate exclusion lists per experiment

Use `exp1_exclude_cols` and `exp2_exclude_cols` instead of a single `additional_exclude_cols`.

**Rationale:** Different experiments may have different metadata columns:
- Field data has: Ent, Sub, Cid, Sid, GID, Cross name, Sel_Hist, Origin
- Turface data has: File.me, region, set, scanner
- Cylinder data has: different scanning metadata

Having separate lists allows precise control without forcing users to maintain a combined list.

**Alternative considered:** Single `exclude_cols` applied to both experiments
- Rejected: Would require listing all columns from both experiments, causing confusion when a column only exists in one

### Decision: Reuse existing get_trait_columns() function

Pass exclusion lists to the existing `get_trait_columns()` function's `additional_exclude` parameter.

**Rationale:**
- Function already supports this parameter
- Maintains consistency with QC/Viz pipelines
- No new code paths to test

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Users forget to add exclusion lists | Document clearly in config template; log warning if metadata-like columns detected |
| Breaking existing configs | Optional parameters with None default ensure backward compatibility |

## Migration Plan

1. Add new optional parameters to CrossPlatformConfig
2. Update LoadCrossPlatformDataStep to use them
3. Update existing config files with appropriate exclusion lists
4. No breaking changes - existing configs continue to work (but may include spurious correlations)

## Open Questions

None - design is straightforward extension of existing pattern.
$