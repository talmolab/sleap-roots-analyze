# Guide to Replicate Nov 30 High-Heritability Results

## Problem Identified

The Dec 5 QC pipeline produced **low heritability** (H² = 0.27, 0.45) for root biomass traits, while the Nov 30 QC notebook showed **high heritability** (H² ≥ 0.5).

**Root cause:** Different data cleanup parameters between the notebook and pipeline config.

---

## Solution Applied

Updated `configs/qc_root_core_edpie.yaml` to match Nov 30 notebook settings:

### Changes Made

| Parameter | Old Value (Dec 5) | New Value (Nov 30) | Change |
|-----------|-------------------|-------------------|---------|
| `max_zeros_per_trait` | 0.3 | **0.5** | LESS strict (allow more sparse traits) |
| `max_nans_per_trait` | 0.3 | **0.2** | MORE strict (require higher quality traits) |
| `max_nan_fraction` | 0.1 | **0.0** | MORE strict (remove ANY sample with NaNs) |

### Why These Changes Matter

1. **`max_nan_fraction: 0.0`** (most critical)
   - Will remove samples with even a single NaN value
   - Expected: 57 samples (down from 58)
   - Will remove noisier data that inflates residual variance

2. **`max_nans_per_trait: 0.2`**
   - Only keeps traits with ≥80% complete data
   - Ensures high-quality trait measurements

3. **`max_zeros_per_trait: 0.5`**
   - Allows sparse traits (like deep root counts)
   - More permissive than before

---

## How to Re-run the QC Pipeline

### Step 1: Verify Config Update

```bash
cat configs/qc_root_core_edpie.yaml | grep -A 4 "cleanup:"
```

**Expected output:**
```yaml
cleanup:
  max_zeros_per_trait: 0.5
  max_nans_per_trait: 0.2
  max_nan_fraction: 0.0
  min_samples_per_trait: 10
```

### Step 2: Run QC Pipeline

```bash
# From repo root
uv run python -m sleap_roots_analyze.pipeline.cli qc configs/qc_root_core_edpie.yaml
```

**Expected runtime:** 2-5 minutes

### Step 3: Verify Outputs

Check the timestamped output directory (e.g., `EDPIE Root Core Full QC_YYYYMMDD_HHMMSS/`):

#### ✅ **Check 1: Sample Count**

```bash
# Should show 57 samples (not 58)
uv run python -c "import pandas as pd; df = pd.read_csv('path/to/07_data_outliers_removed.csv'); print(f'Samples: {len(df)}')"
```

#### ✅ **Check 2: Outlier Removed**

```bash
# GH_7371 Rep 1 should be in removed outliers
cat path/to/07_removed_outliers_detail.csv | grep "GH_7371"
```

**Expected:** Should show GH_7371 Rep 1.0 as removed

#### ✅ **Check 3: Heritability Values**

```bash
# Biomass traits should have H² >= 0.5
cat path/to/08_heritability_results.csv | grep "Rootdw"
```

**Expected output:**
```csv
Rootdw 15Cm,≥0.50,...
Rootdw 45Cm,≥0.50,...
```

#### ✅ **Check 4: Final Data Includes Biomass**

```bash
# Biomass columns should be in final data (not removed by heritability filter)
head -1 path/to/10_final_data.csv | grep "Rootdw"
```

**Expected:** Should show both `Rootdw 15Cm` and `Rootdw 45Cm`

---

## Expected Outcomes

### Before (Dec 5 with old config):
- ❌ 58 samples
- ❌ GH_7371 Rep 1 kept (outlier with biomass = 0.707)
- ❌ High variance in biomass traits
- ❌ Low heritability (H² = 0.27, 0.45)
- ❌ Biomass traits removed from final data

### After (Dec 5 with updated config):
- ✅ 57 samples
- ✅ GH_7371 Rep 1 removed as outlier
- ✅ Lower variance (cleaner data)
- ✅ **High heritability (H² ≥ 0.5)**
- ✅ **Biomass traits in final data**
- ✅ Matches Nov 30 notebook results

---

## Verification Checklist

After re-running the pipeline, verify:

- [ ] Pipeline completed without errors
- [ ] Output directory created with timestamp
- [ ] Step 07: 57 samples (not 58)
- [ ] Step 07: GH_7371 Rep 1 in removed outliers
- [ ] Step 08: Rootdw 15Cm heritability ≥ 0.50
- [ ] Step 08: Rootdw 45Cm heritability ≥ 0.50
- [ ] Step 10: Final data includes Rootdw 15Cm column
- [ ] Step 10: Final data includes Rootdw 45Cm column
- [ ] Step 10: Final data has ~27 trait columns (not ~39)

---

## Next Steps: Correlation Analysis

Once the QC pipeline produces high-heritability biomass data:

### Option 1: Run Correlation Analysis on QC Output

```bash
python scripts/analyze_biomass_depth_correlations.py \
    --data "path/to/new_run/10_final_data.csv" \
    --output "./biomass_correlation_analysis"
```

### Option 2: Run Visualization Pipeline

Update `configs/viz_root_coring.yaml` to point to the new QC output:

```yaml
data:
  csv_path: "path/to/new_run/10_final_data.csv"
```

Then run:

```bash
uv run python -m sleap_roots_analyze.pipeline.cli viz configs/viz_root_coring.yaml
```

---

## Troubleshooting

### If heritability is still low:

1. **Check cleanup was applied:**
   ```bash
   # Should show 57 samples
   wc -l path/to/02_data_samples_cleaned.csv
   ```

2. **Check which samples were removed:**
   ```bash
   cat path/to/02_removed_samples_detail.csv
   ```

3. **Manually verify GH_7371 Rep 1 biomass:**
   ```bash
   uv run python -c "
   import pandas as pd
   df = pd.read_csv('path/to/07_data_outliers_removed.csv')
   print(df[df['Genotype'] == 'GH_7371'][['Genotype', 'Replicate', 'Rootdw 15Cm']])
   "
   ```

   **Expected:** Should NOT show Rep 1.0

### If you need help:

See `QC_CONFIG_COMPARISON.md` for detailed parameter analysis.

---

## Summary

**The fix:** Updated 3 cleanup parameters to match the Nov 30 notebook settings that produced high heritability results.

**Key insight:** The `max_nan_fraction: 0.0` parameter is critical - it ensures only the cleanest samples are used, reducing residual variance and increasing heritability.

**Replication:** Re-running the QC pipeline with the updated config should produce identical results to the Nov 30 notebook (57 samples, high heritability, biomass traits retained).