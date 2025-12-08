# QC Pipeline Issue Summary

## Problem

**Despite updating the cleanup config to match Nov 30 settings, the pipeline still produces low heritability for biomass traits.**

## Investigation Results

### ✅ What Worked
- Config was successfully updated:
  - `max_nan_fraction`: 0.1 → 0.0
  - `max_nans_per_trait`: 0.3 → 0.2
  - `max_zeros_per_trait`: 0.3 → 0.5

- Pipeline ran successfully with updated config

### ❌ What Didn't Work
- **Still 58 samples** (should be 57)
- **Still low heritability** (H² = 0.27, 0.45)
- **GH_7371 Rep 1 outlier NOT removed**

## Root Cause

### 1. Cleanup Changes Had No Effect
**The merged trait data has ZERO NaN values after step 01 (trait cleanup).**

This means `max_nan_fraction: 0.0` couldn't remove any samples - there were no NaN values to trigger removal!

### 2. Outlier Detection is Independent
**The outlier detection (Mahalanobis distance) didn't catch GH_7371 Rep 1 as an outlier.**

Current run removed:
- GH_7440 Rep 3
- Control Rep 3

**But NOT** GH_7371 Rep 1 (biomass = 0.7071), which was removed in Nov 30.

## Why Nov 30 Caught the Outlier

The Nov 30 QC **notebook** removed GH_7371 Rep 1, but we need to understand WHY.

Possible reasons:
1. **Different input data** - Nov 30 may have had NaN values that triggered different filtering
2. **Different outlier detection parameters** - though config appears identical
3. **Different data preprocessing** - the merge/aggregation step may differ
4. **Manual intervention** - the notebook may have included manual outlier removal

## Next Steps to Fix

### Option 1: Manually Remove GH_7371 Rep 1
Create a preprocessing step or update the config to explicitly exclude this sample.

### Option 2: Adjust Outlier Detection
- Lower the chi2_percentile (currently 99.0)
- Or add a preprocessing step that removes samples with extreme single-trait values

### Option 3: Use Nov 30 Data Directly
For the correlation analysis, just use the Nov 30 cleaned data:
```bash
python scripts/analyze_biomass_depth_correlations.py \
    --data "C:/repos/runs/run_20251130_193257/cleaned_traits.csv" \
    --output "./biomass_correlation_analysis"
```

## Recommendation

**Use Option 3** for now - the Nov 30 data is validated and has high heritability.

For long-term reproducibility, we need to:
1. Find the Nov 30 QC notebook
2. Compare it line-by-line with the pipeline
3. Identify the EXACT step that removed GH_7371 Rep 1
4. Replicate that in the pipeline

## Data Comparison

| Metric | Nov 30 QC | Dec 5 (New Config) | Match? |
|--------|-----------|-------------------|--------|
| Samples before outlier detection | ? | 60 | ? |
| Samples after outlier removal | 57 | 58 | ❌ |
| GH_7371 Rep 1 removed | YES | NO | ❌ |
| Rootdw 15Cm heritability | ≥0.50 | 0.27 | ❌ |
| Rootdw 45Cm heritability | ≥0.50 | 0.45 | ❌ |
| Biomass in final data | YES | NO | ❌ |

## Files Generated

Latest QC run output: `qc_runs/EDPIE Root Core Full QC_20251205_152721/`

Key files:
- `07_data_outliers_removed.csv` - 58 samples (should be 57)
- `08_heritability_results.csv` - shows H² = 0.27, 0.45 for biomass
- `10_final_data.csv` - biomass columns removed
