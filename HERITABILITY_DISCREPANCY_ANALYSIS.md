# Heritability Discrepancy Analysis: Nov 30 Notebook vs Baseline Pipeline

## Executive Summary

**CRITICAL FINDING:** There are massive and systematic heritability discrepancies between the Nov 30, 2024 notebook results and the current baseline pipeline (with core-level QC disabled). This explains why the per-core value outlier detection feature failed to improve results - the baseline itself is fundamentally different from the Nov 30 "golden standard."

## Sample Counts
- **Nov 30 Notebook:** 57 samples
- **Baseline Pipeline:** 58 samples
- **Difference:** +1 sample in pipeline (negligible)

## Heritability Threshold Performance
- **Nov 30 Notebook:** 23/23 traits ≥ 0.5 threshold (100% pass rate)
- **Baseline Pipeline:** 20/23 traits ≥ 0.5 threshold (87% pass rate)
- **Net Loss:** 3 traits dropped below threshold

## Complete Trait-by-Trait Comparison

### ROOT BIOMASS TRAITS (CRITICAL - largest discrepancies)

| Trait | Nov 30 H² | Baseline H² | Difference | % Change | Status |
|-------|-----------|-------------|------------|----------|--------|
| **Rootdw 15Cm** (0-30cm) | **0.750** ✓ | **0.270** ✗ | **-0.480** | **-64.0%** | ⬇ **DROPPED BELOW 0.5** |
| **Rootdw 45Cm** (30-60cm) | **0.730** ✓ | **0.449** ✗ | **-0.281** | **-38.5%** | ⬇ **DROPPED BELOW 0.5** |

**Analysis:** ROOT BIOMASS traits show CATASTROPHIC heritability loss:
- 15cm depth: 75% → 27% (lost 64% of heritability)
- 45cm depth: 73% → 45% (lost 38% of heritability)
- Both traits REMOVED from final dataset (below 0.5 threshold)
- This is THE core problem that motivated the per-core QC feature

### ABOVE-GROUND TRAITS (Below 0.5 threshold in both datasets)

| Trait | Nov 30 H² | Baseline H² | Difference | % Change | Status |
|-------|-----------|-------------|------------|----------|--------|
| **Rspkgp Calc Pct** | **0.479** ✗ | **0.499** ✗ | **+0.020** | **+4.1%** | ⬆ **ROSE ABOVE 0.5** |
| **Plntstnd Calc Plntm2** | **0.461** ✗ | **0.490** ✗ | **+0.029** | **+6.2%** | Stable (both <0.5) |
| **Stmdw M Gm2** | **0.400** ✗ | **0.465** ✗ | **+0.065** | **+16.3%** | Stable (both <0.5) |

### ABOVE-GROUND TRAITS (≥0.5 in both datasets, small differences)

| Trait | Nov 30 H² | Baseline H² | Difference | % Change | Status |
|-------|-----------|-------------|------------|----------|--------|
| **Gfp Calc Pct** | 0.512 ✓ | 0.521 ✓ | +0.009 | +1.8% | Stable (✓→✓) |
| **Spkdw M (g)** | 0.660 ✓ | 0.662 ✓ | +0.002 | +0.3% | Stable (✓→✓) |
| **Spkdw Calc Gm2** | 0.617 ✓ | 0.620 ✓ | +0.003 | +0.5% | Stable (✓→✓) |
| **Bm Calc Gm2** | 0.705 ✓ | 0.709 ✓ | +0.004 | +0.6% | Stable (✓→✓) |
| **Sheathdw M (g)** | 0.743 ✓ | 0.741 ✓ | -0.002 | -0.3% | Stable (✓→✓) |
| **Stmdw M (g)** | 0.791 ✓ | 0.805 ✓ | +0.014 | +1.7% | Stable (✓→✓) |
| **Sn Calc Spksm2** | 0.848 ✓ | 0.855 ✓ | +0.007 | +0.8% | Stable (✓→✓) |
| **Gwsp Calc (g)** | 0.869 ✓ | 0.877 ✓ | +0.008 | +0.9% | Stable (✓→✓) |
| **Gfr Calc Gm2Day** | 0.877 ✓ | 0.881 ✓ | +0.004 | +0.5% | Stable (✓→✓) |
| **Boot Dtoinit Day** | 0.896 ✓ | 0.904 ✓ | +0.008 | +1.0% | Stable (✓→✓) |
| **Hd Dto Day** | 0.907 ✓ | 0.915 ✓ | +0.008 | +0.8% | Stable (✓→✓) |
| **Ant Dto Day** | 0.907 ✓ | 0.915 ✓ | +0.008 | +0.8% | Stable (✓→✓) |
| **Gn Calc Grnm2** | 0.915 ✓ | 0.919 ✓ | +0.004 | +0.4% | Stable (✓→✓) |
| **Gy Calc Gm2** | 0.927 ✓ | 0.930 ✓ | +0.003 | +0.3% | Stable (✓→✓) |
| **Gw M G1000Grn** | 0.941 ✓ | 0.941 ✓ | +0.000 | +0.0% | Stable (✓→✓) |
| **Mat Dto Day** | 0.957 ✓ | 0.951 ✓ | -0.006 | -0.7% | Stable (✓→✓) |
| **Ph M Cm** | 0.974 ✓ | 0.970 ✓ | -0.004 | -0.4% | Stable (✓→✓) |

### TRAITS WITH IMPROVED HERITABILITY (Grnspk Calc Grnspk)

| Trait | Nov 30 H² | Baseline H² | Difference | % Change | Status |
|-------|-----------|-------------|------------|----------|--------|
| **Grnspk Calc Grnspk** | **0.579** ✓ | **0.602** ✓ | **+0.023** | **+3.9%** | Stable (✓→✓) |

## Statistical Summary

### Discrepancy Distribution
- **Massive degradation (|Δ| > 0.25):** 2 traits (both root biomass)
- **Moderate degradation (-0.25 < Δ < -0.05):** 0 traits
- **Stable (|Δ| < 0.05):** 18 traits
- **Moderate improvement (0.05 < Δ < 0.25):** 3 traits
- **Massive improvement (Δ > 0.25):** 0 traits

### Root Cause Analysis

**Key Observations:**
1. **Above-ground traits are nearly identical** between Nov 30 and baseline (differences <2%)
2. **Root biomass traits show MASSIVE discrepancies** (38-64% heritability loss)
3. **Sample counts are nearly identical** (57 vs 58 samples)
4. **Root counting traits were NOT measured** in Nov 30 notebook (only biomass + above-ground)

**This pattern suggests:**
- Above-ground trait processing is consistent → both datasets use same source CSV
- Root biomass processing is FUNDAMENTALLY DIFFERENT → different aggregation or QC applied
- The difference is NOT due to sample exclusion (n=57 vs n=58)
- The difference IS due to core-level data processing

## Critical Questions

### 1. **What aggregation method did Nov 30 use?**
- Design doc claims: "mean of cores 0 & 1 only (excluded core 2)"
- Baseline pipeline: "median of all 3 cores"
- Need to verify: Was core 2 systematically excluded in Nov 30?

### 2. **Was core-level QC applied in Nov 30?**
- Nov 30 may have manually applied per-core QC
- Baseline pipeline (QC OFF) uses raw core aggregation
- Need to investigate notebook code

### 3. **Why is the sample count difference only 1?**
- Nov 30: 57 samples (n_observations)
- Baseline: 58 samples
- This suggests trait-level exclusion, not core-level

### 4. **What cleanup thresholds were used in Nov 30?**
- Current config: `max_nan_fraction=0.0` (VERY STRICT)
- Nov 30 may have used different thresholds
- Need to compare cleanup logs

## Impact on Per-Core Value Outlier Detection Feature

**Why the feature failed:**
1. Feature was designed to close gap from 0.27/0.45 → 0.75/0.73
2. Assumed the gap was due to gross measurement errors (like GH_7371 core 2)
3. Reality: The 30% threshold is too aggressive, removing 53% of biomass data
4. Result: H² went from 0.27/0.45 → 0.43/0.33 (WORSE for 45cm depth!)

**The feature CAN'T work until we understand:**
- What Nov 30 actually did to achieve H² = 0.75/0.73
- Whether manual core exclusion was applied
- What the "correct" baseline should be

## Recommendations

### Immediate Actions (Priority 1)
1. **Investigate Nov 30 notebook code** to determine:
   - Exact aggregation method used
   - Whether core 2 was systematically excluded
   - What cleanup thresholds were applied
   - Whether manual QC was performed

2. **Compare raw core-level data** between Nov 30 and baseline:
   - Check if same cores are present
   - Verify aggregation calculations
   - Identify any manual exclusions

3. **Reproduce Nov 30 results** using baseline pipeline:
   - Test mean vs median aggregation
   - Test excluding core 2 (as design doc claims)
   - Verify if this closes the heritability gap

### Secondary Actions (Priority 2)
4. **Document data provenance**:
   - Where did Nov 30 input data come from?
   - Was it pre-processed before notebook?
   - Are we using the same source CSVs?

5. **Validate design document claims**:
   - Verify GH_7371 Rep 1 case study
   - Check if "mean of cores 0&1" claim is accurate
   - Document any discrepancies

### Feature Development (Priority 3 - BLOCKED until above resolved)
6. **Pause per-core value outlier detection tuning**:
   - Current 30% threshold is too aggressive
   - Can't optimize until we understand Nov 30's success
   - Need correct baseline first

7. **Re-evaluate feature necessity**:
   - If Nov 30 used mean aggregation, feature may be unnecessary
   - If Nov 30 manually excluded cores, we need automated version
   - Need data-driven decision

## Conclusion

**The per-core value outlier detection feature was built on incorrect assumptions.** The Nov 30 notebook's high heritability (H² = 0.75/0.73 for root biomass) was NOT due to better handling of gross measurement errors like GH_7371 core 2. Instead, it appears to be due to fundamental differences in data processing that we don't yet understand.

**We cannot proceed with feature optimization until we:**
1. Understand what Nov 30 did differently
2. Reproduce Nov 30 results with the baseline pipeline
3. Establish the correct baseline for comparison

**This is a data provenance and pipeline validation issue, not a feature tuning issue.**
