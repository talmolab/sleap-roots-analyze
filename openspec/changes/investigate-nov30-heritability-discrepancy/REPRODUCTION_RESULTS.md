# Reproduction Results: Mean Aggregation + 50% Threshold Test

**Date:** 2025-12-09  
**Config:** `configs/test_nov30_reproduction.yaml`  
**Pipeline:** TEST Nov30 Reproduction (mean+50%)  
**Run ID:** 20251209_175507

## Executive Summary

**REPRODUCTION STATUS: FAILED ❌**

The test configuration with mean aggregation + 50% deviation threshold **did NOT successfully reproduce Nov 30 heritability results**.

## Configuration Changes Tested

```yaml
# Biomass source (line 24):
aggregation_method: "mean"  # Changed from "median"

# Core QC (lines 100-101):
detect_value_outliers: true   # Enabled (was false)
max_deviation_from_median: 0.50  # 50% threshold (was 0.30)
```

## Critical Validation Results

### 1. GH_7371 Rep 1 Aggregated Value ❌

**Target (Nov 30):** 0.7354 g (mean of cores 0 & 1)

**Actual (Test Run):** 0.5946 g

**Analysis:**
- Test value (0.5946g) = MEAN of all 3 cores [0.7636g, 0.7071g, 0.3132g]
- Expected: Core 2 (0.31g, 56% deviation) should have been flagged and excluded
- **Core 2 was NOT excluded** despite 56% deviation exceeding 50% threshold

**Conclusion:** Per-core value outlier detection did not flag the critical outlier core.

### 2. Root Biomass Heritability ❌

| Trait | Target (Nov 30) | Actual (Test) | Delta | Status |
|-------|-----------------|---------------|-------|--------|
| Rootdw 15Cm | 0.750 | **0.324** | -0.426 | ❌ FAIL |
| Rootdw 45Cm | 0.730 | **0.413** | -0.317 | ❌ FAIL |

**Both traits FAILED to meet H² ≥ 0.70 threshold.**

### 3. Sample Count ✅

**Target (Nov 30):** n = 57  
**Actual (Test):** n = 58  
**Delta:** +1 sample  
**Status:** ✅ ACCEPTABLE (within ±1)

### 4. Above-Ground Traits ✅

**Ph M Cm (Plant Height):**
- Target (Nov 30): 0.974
- Actual (Test): 0.970
- Delta: -0.004
- Status: ✅ STABLE

**Other traits:** Remain stable and consistent with baseline.

## Detailed Heritability Comparison

### Root Biomass Traits (PRIMARY INVESTIGATION FOCUS)

```
Rootdw 15Cm:
  Nov 30:    H² = 0.750 (n=57)
  Baseline:  H² = 0.270 (n=58, median, no core QC)
  TEST RUN:  H² = 0.324 (n=58, mean, 50% threshold)
  
  Improvement vs baseline: +0.054 (20% relative improvement)
  Gap to Nov 30:          -0.426 (FAILED)

Rootdw 45Cm:
  Nov 30:    H² = 0.730 (n=57)
  Baseline:  H² = 0.449 (n=58, median, no core QC)
  TEST RUN:  H² = 0.413 (n=58, mean, 50% threshold)
  
  Improvement vs baseline: -0.036 (WORSE!)
  Gap to Nov 30:          -0.317 (FAILED)
```

### All Traits Ranked by Heritability (Test Run)

```
HIGH (H² ≥ 0.9):
  Ph M Cm              0.970  ✅
  Mat Dto Day          0.951  ✅
  Gw M G1000Grn        0.941  ✅
  Gy Calc Gm2          0.930  ✅
  Gn Calc Grnm2        0.919  ✅
  Hd Dto Day           0.915  ✅
  Ant Dto Day          0.915  ✅
  Boot Dtoinit Day     0.904  ✅
  Gfr Calc Gm2Day      0.881  ✅
  Gwsp Calc (g)        0.877  ✅
  Sn Calc Spksm2       0.855  ✅
  Stmdw M (g)          0.805  ✅
  Sheathdw M (g)       0.741  ✅

MODERATE (0.5 ≤ H² < 0.9):
  Bm Calc Gm2          0.709  ✅
  Root Count 55cm      0.671  ✅
  Spkdw M (g)          0.662  ✅
  Spkdw Calc Gm2       0.620  ✅
  Grnspk Calc Grnspk   0.602  ✅
  Root Count 0cm       0.596  ✅
  Root Count 35cm      0.588  ✅
  Root Count 45cm      0.557  ✅
  Root Count 15cm      0.550  ✅
  Root Count 20cm      0.521  ✅
  Gfp Calc Pct         0.521  ✅

LOW (H² < 0.5) - FILTERED OUT:
  Rspkgp Calc Pct      0.499  ❌
  Plntstnd Calc Plntm2 0.490  ❌
  Stmdw M Gm2          0.465  ❌
  Root Count 10cm      0.451  ❌
  Root Count 30cm      0.417  ❌
  Rootdw 45Cm          0.413  ❌ (CRITICAL FAILURE)
  Root Count 40cm      0.389  ❌
  Rootdw 15Cm          0.324  ❌ (CRITICAL FAILURE)
  Root Count 5cm       0.301  ❌
  Root Count 25cm      0.172  ❌
  Root Count 50cm      0.000  ❌ (near-zero)
```

## Root Cause Analysis: Why Did Reproduction Fail?

### Hypothesis 1: 50% Threshold Too Conservative ❌

**Evidence:**
- GH_7371 Rep 1 core 2 has 56% deviation from median
- This EXCEEDS the 50% threshold
- Yet core 2 was NOT excluded (aggregated value = 0.5946g = mean of all 3)

**Possible explanations:**
1. **Calculation error in outlier detection code**
2. **Threshold applied AFTER aggregation** (should be BEFORE)
3. **Deviation calculated differently** than expected
4. **Outlier flagging not integrated with aggregation step**

### Hypothesis 2: Mean vs Median Aggregation Not Sufficient

**Evidence:**
- Switching from median to mean did NOT improve results
- Rootdw 45Cm actually got WORSE (0.449 → 0.413)
- Rootdw 15Cm improved slightly (0.270 → 0.324) but nowhere near target

**Conclusion:** Aggregation method alone does NOT explain the discrepancy.

### Hypothesis 3: Manual Core Exclusions in Nov 30 Were More Extensive

**Evidence:**
- Nov 30 sample count: n = 57
- Test run sample count: n = 58
- Only 1 sample difference suggests minimal exclusion

**Implication:** Nov 30 may have excluded MULTIPLE cores per plot, not just GH_7371.

## Technical Investigation Required

### 1. Verify Core QC Execution

**Check if core QC step actually ran:**
```bash
# Examine core QC metadata
cat "qc_runs/TEST Nov30 Reproduction (mean+50%)_20251209_175507/00c_core_qc_metadata.json"
```

**Expected:**
- List of flagged cores with reasons
- Statistics on flagging rates per depth/data type
- Confirmation that outlier detection was enabled

### 2. Verify Core-Level Data Before Aggregation

**Check raw core data with outlier flags:**
```bash
# Load core-level data from Step 00b
# Should have outlier_flag column with True/False values
```

**Expected for GH_7371 Rep 1:**
- Core 0 (0.7636g): outlier_flag = False
- Core 1 (0.7071g): outlier_flag = False
- Core 2 (0.3132g): outlier_flag = True (56% deviation)

### 3. Trace Aggregation Logic

**Verify aggregation code:**
```python
# src/sleap_roots_analyze/pipeline/steps/aggregate_cores.py
# Check if outlier-flagged cores are excluded before aggregation
```

**Expected behavior:**
```python
# Filter out flagged cores BEFORE aggregation
df_clean = df[df["outlier_flag"] == False]
# Then aggregate
df_agg = df_clean.groupby(group_cols).agg(agg_method)
```

## Comparison with Previous Configurations

| Config | Aggregation | Core QC | Threshold | Rootdw 15Cm H² | Rootdw 45Cm H² | n |
|--------|-------------|---------|-----------|----------------|----------------|---|
| Nov 30 (target) | mean (cores 0&1) | manual | N/A | **0.750** | **0.730** | 57 |
| Baseline (QC OFF) | median | disabled | N/A | 0.270 | 0.449 | 58 |
| Test (30% threshold) | median | enabled | 30% | 0.434 | 0.325 | 51 |
| **Test (50% threshold)** | **mean** | **enabled** | **50%** | **0.324** | **0.413** | **58** |

**Pattern Analysis:**
- 30% threshold was TOO AGGRESSIVE (removed 53% of data, n=51)
- 50% threshold appears to have NO EFFECT (n=58, same as baseline)
- Mean aggregation WORSENED Rootdw 45Cm vs baseline median

## Revised Hypotheses

### Primary Hypothesis: Per-Core QC Not Executing

**Evidence:**
1. GH_7371 Rep 1 aggregated value = 0.5946g (mean of all 3 cores)
2. This proves core 2 (0.31g) was NOT excluded
3. Sample count n=58 (same as baseline with QC disabled)

**Conclusion:** The `detect_value_outliers: true` setting may not be functioning as intended.

**Action Required:** Inspect core QC metadata file and aggregation step code.

### Secondary Hypothesis: Field_2024_clean.csv Used Different Raw Data

**Evidence:**
- Nov 30 notebook loaded `Field_2024_clean.csv` (pre-aggregated)
- We don't know HOW this file was created
- Manual processing may have corrected data errors BEYOND outlier removal

**Implications:**
- Source CSVs may have been edited/corrected
- Raw core values may differ from what we have
- Nov 30 may have used a DIFFERENT version of `rearranged_root_biomass_dw.csv`

**Action Required:** Hash-compare source CSVs, check git history, interview person who created Field_2024_clean.csv.

## Recommendations

### Immediate Actions (Priority 1)

1. **Verify Core QC Execution:**
   - Read `00c_core_qc_metadata.json`
   - Check if ANY cores were flagged
   - Verify outlier detection code path was executed

2. **Inspect Raw Core Data:**
   - Load `00b_root_core_biomass_long.csv` (before aggregation)
   - Check for `outlier_flag` column
   - Verify GH_7371 Rep 1 core 2 flagging status

3. **Trace Aggregation Code:**
   - Review `aggregate_cores.py` implementation
   - Verify outlier-flagged cores are excluded BEFORE aggregation
   - Check for bugs or logic errors

### Secondary Actions (Priority 2)

4. **Compare Source CSVs:**
   - Hash-compare `rearranged_root_biomass_dw.csv` (current vs Nov 30)
   - Check git history for any modifications
   - Document file provenance

5. **Interview Data Curator:**
   - Ask person who created `Field_2024_clean.csv`
   - Document manual QC steps applied
   - Get list of manually excluded cores (if available)

6. **Test Alternative Thresholds:**
   - Try 40% threshold (if QC is working)
   - Try 60-70% threshold (very conservative)
   - Document flagging rates vs heritability

### Long-Term Actions (Priority 3)

7. **Update Design Document:**
   - Document that 50% threshold test failed
   - Note that per-core QC may have implementation issues
   - Add findings to investigation summary

8. **Consider Alternative Approaches:**
   - Create manual core exclusion list (match Nov 30 exactly)
   - Implement alternative outlier detection methods
   - Accept empirical baseline and optimize from there

## Next Steps

**Before proceeding with additional tests, we must:**
1. Verify that per-core value outlier detection actually executed
2. Understand why GH_7371 core 2 was not flagged despite 56% deviation
3. Fix any bugs in the core QC implementation

**Only after confirming QC works correctly can we:**
- Tune thresholds to optimize heritability
- Compare different aggregation methods
- Attempt to fully reproduce Nov 30 results

## Conclusion

The mean aggregation + 50% threshold configuration **failed to reproduce Nov 30 heritability results**. The most likely explanation is that per-core value outlier detection **did not execute as expected**, as evidenced by:

1. GH_7371 Rep 1 aggregated value (0.5946g) includes the outlier core 2
2. Sample count (n=58) unchanged from baseline
3. Heritability values closer to baseline than to Nov 30 target

**Critical finding:** The per-core QC feature may have an implementation bug preventing it from flagging outliers, even when explicitly enabled with a reasonable threshold.

**Recommendation:** Investigate core QC implementation BEFORE attempting further reproduction experiments.
