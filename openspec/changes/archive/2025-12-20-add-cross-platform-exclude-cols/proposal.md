## Why

Cross-platform correlation analysis is including metadata columns (Ent, Sub, Cid, Sid, GID) as "traits" and computing spurious correlations. These columns are correctly excluded in QC/Viz pipelines via `additional_exclude_cols`, but the cross-platform pipeline has no mechanism to exclude them.

**Example of spurious correlations currently being computed:**
- `Ent,Root Shoot Ratio,-0.577` (Entry number vs trait)
- `Sub,Root Shoot Ratio,-0.509` (Sub-entry vs trait)
- `Ent,Depth (mm),0.507` (Entry number vs depth)

These correlations have no biological meaning and pollute the analysis results.

## What Changes

- Add `exp1_exclude_cols` and `exp2_exclude_cols` optional parameters to `CrossPlatformConfig`
- Modify `LoadCrossPlatformDataStep` to pass exclusion lists to `get_trait_columns()`
- Update cross-platform YAML configs to specify metadata columns to exclude
- Follows existing pattern used by `additional_exclude_cols` in QC/Viz pipelines

## Impact

- **Affected specs**: `cross-platform-analysis` (modifies Cross-Platform Configuration requirement)
- **Affected code**:
  - `src/sleap_roots_analyze/pipeline/config/components.py` - CrossPlatformConfig dataclass
  - `src/sleap_roots_analyze/pipeline/steps/load_cross_platform_data.py` - LoadCrossPlatformDataStep
  - `configs/active/cross_platform/*.yaml` - All cross-platform configs
- **Backward compatible**: Yes (new parameters are optional with `None` default)
