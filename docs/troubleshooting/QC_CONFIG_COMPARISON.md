# QC Configuration Comparison: Nov 30 Notebook vs Dec 5 Pipeline

## Executive Summary

The Nov 30 QC notebook (which produced high heritability) and the Dec 5 QC pipeline config have **3 critical parameter differences** in data cleanup that explain why heritability values differ.

---

## Detailed Comparison

### 1. CLEANUP PARAMETERS

| Parameter | Nov 30 Notebook | Dec 5 Pipeline | Match? | Impact |
|-----------|-----------------|----------------|--------|---------|
| `MAX_NAN_FRACTION` | **0.0** | **0.1** | ❌ **DIFFERENT** | Dec 5 allows samples with up to 10% NaN; Nov 30 removed ANY sample with NaN |
| `MAX_ZEROS_PER_TRAIT` | **0.5** | **0.3** | ❌ **DIFFERENT** | Dec 5 is MORE strict (removes traits with >30% zeros) |
| `MAX_NANS_PER_TRAIT` | **0.2** | **0.3** | ❌ **DIFFERENT** | Dec 5 is LESS strict (allows traits with up to 30% NaNs) |
| `MIN_SAMPLES_PER_TRAIT` | 10 | 10 | ✅ SAME | - |

### 2. OUTLIER DETECTION PARAMETERS

| Parameter | Nov 30 Notebook | Dec 5 Pipeline | Match? |
|-----------|-----------------|----------------|--------|
| Method | `mahalanobis` | `mahalanobis` | ✅ SAME |
| `MAHAL_VARIANCE_THRESHOLD` | 0.95 | 0.95 | ✅ SAME |
| `MAHAL_USE_CHI_SQUARED` | True | true | ✅ SAME |
| `MAHAL_CHI2_PERCENTILE` | 99 | 99.0 | ✅ SAME |
| Removal strategy | N/A | `single` | ✅ SAME |

### 3. HERITABILITY PARAMETERS

| Parameter | Nov 30 Notebook | Dec 5 Pipeline | Match? |
|-----------|-----------------|----------------|--------|
| Enabled | True | true | ✅ SAME |
| `HERITABILITY_THRESHOLD` | 0.50 | 0.5 | ✅ SAME |

### 4. PCA PARAMETERS

| Parameter | Nov 30 Notebook | Dec 5 Pipeline | Match? |
|-----------|-----------------|----------------|--------|
| `EXPLAINED_VARIANCE` / `n_components` | 0.95 | 0.95 | ✅ SAME |
| Standardize | Yes | true | ✅ SAME |

---

## Critical Differences Explained

### **Difference #1: `MAX_NAN_FRACTION` (0.0 → 0.1)**

**Nov 30 Behavior:**
- Removed ANY sample with even a single NaN value
- Very strict approach
- Result: 57 samples (cleaner data)

**Dec 5 Behavior:**
- Allows samples with up to 10% NaN values
- More permissive
- Result: 58 samples (potentially includes noisier samples)

**Impact on Heritability:**
- Including samples with NaN values can increase variance
- May reduce genetic signal-to-noise ratio

---

### **Difference #2: `MAX_ZEROS_PER_TRAIT` (0.5 → 0.3)**

**Nov 30 Behavior:**
- Allowed traits with up to 50% zero values
- More permissive for sparse traits

**Dec 5 Behavior:**
- Removes traits with >30% zero values
- Stricter filtering

**Impact on Heritability:**
- This is actually GOOD for Dec 5 (removes low-quality traits)
- But doesn't explain the heritability drop

---

### **Difference #3: `MAX_NANS_PER_TRAIT` (0.2 → 0.3)**

**Nov 30 Behavior:**
- Removed traits with >20% NaN values
- Stricter trait filtering

**Dec 5 Behavior:**
- Allows traits with up to 30% NaN values
- More permissive

**Impact on Heritability:**
- Traits with more missing values have less reliable estimates
- May reduce overall data quality

---

## Root Cause Analysis

### Why Did Nov 30 Remove GH_7371 Rep 1 as an Outlier?

**The outlier (biomass = 0.735) was correctly flagged by Mahalanobis distance**, BUT:

1. **Nov 30 had cleaner input data** (MAX_NAN_FRACTION = 0.0)
   - Started with 60 samples → removed 3 samples with NaNs → 57 samples
   - Then applied outlier detection on this cleaner dataset

2. **Dec 5 has noisier input data** (MAX_NAN_FRACTION = 0.1)
   - Kept samples with some NaN values
   - The added variance from these samples may have:
     - Shifted the Mahalanobis distance threshold
     - Made GH_7371 Rep 1 appear "less extreme" relative to other variance

### Why Heritability Differs

**Nov 30 (High H²):**
- Stricter sample filtering (0% NaN tolerance) → cleaner data
- Stricter trait filtering (20% NaN tolerance) → higher quality traits
- Lower residual variance → **higher heritability**

**Dec 5 (Low H²):**
- More permissive sample filtering (10% NaN tolerance) → noisier data
- More permissive trait filtering (30% NaN tolerance) → lower quality traits
- Higher residual variance → **lower heritability**

---

## Recommended Fix

Update `configs/qc_root_core_edpie.yaml` to match Nov 30 notebook settings:

```yaml
# Data cleanup thresholds
cleanup:
  max_zeros_per_trait: 0.5  # Change from 0.3 → 0.5 (match notebook)
  max_nans_per_trait: 0.2   # Change from 0.3 → 0.2 (match notebook)
  max_nan_fraction: 0.0     # Change from 0.1 → 0.0 (match notebook)
  min_samples_per_trait: 10 # Keep at 10
```

**Expected Outcome:**
- Will remove samples with any NaN values (stricter)
- Will produce 57 samples (matching Nov 30)
- Should remove GH_7371 Rep 1 as outlier
- **Biomass heritability should increase to H² ≥ 0.5**

---

## Verification Steps

After updating the config:

1. Re-run QC pipeline:
   ```bash
   python -m sleap_roots_analyze.pipeline.run_qc_pipeline configs/qc_root_core_edpie.yaml
   ```

2. Check outputs:
   - Step 07 should have **57 samples** (not 58)
   - GH_7371 Rep 1 should be in removed outliers
   - Step 08 heritability should show H² ≥ 0.5 for biomass columns
   - Step 10 final data should **include** Rootdw 15Cm and Rootdw 45Cm

3. Verify heritability:
   ```bash
   # Check if biomass columns have H² >= 0.5
   cat output_dir/08_heritability_results.csv | grep "Rootdw"
   ```
