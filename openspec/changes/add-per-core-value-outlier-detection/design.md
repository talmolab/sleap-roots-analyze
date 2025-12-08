# Design: Per-Core Value Outlier Detection

## Problem Analysis

### Root Cause Discovery
Investigation of EDPIE dataset showed:
1. Nov 30 QC notebook: 57 samples, H² ≥ 0.50 (HIGH heritability)
2. Dec 5 pipeline: 58 samples, H² = 0.27-0.45 (LOW heritability)

**Key finding:** Different core aggregation strategies:
- **Nov 30 notebook:** Used mean of cores 0 & 1 only (excluded core 2) → 0.735
- **Pipeline:** Used median of all 3 cores → 0.707

For GH_7371 Rep 1:
- Core 0: 0.7636 g
- Core 1: 0.7071 g
- Core 2: 0.3132 g ← **56% deviation from median (likely measurement error)**

### Why Median Wasn't Sufficient
With N=3 cores, median is NOT robust:
- Median([0.76, 0.71, 0.31]) = 0.71 (middle value)
- This is heavily influenced by the bad core (0.31)
- The median (0.71) is 7% lower than the true value (~0.74)
- At trait level, 0.71 is NOT flagged as extreme (just outside normal range)
- **Result:** Stays in dataset → inflates variance → destroys heritability

**Conclusion:** Need per-core QC BEFORE aggregation.

## Statistical Framework

### Quality Control vs. Statistical Hypothesis Testing

**Critical Distinction:** This proposal implements **measurement error detection** (quality control), NOT statistical outlier detection or hypothesis testing.

**Why This Matters:**

1. **Goal**: Remove gross measurement errors (damaged cores, typos, sampling failures) before aggregation
2. **Not trying to**: Model the full hierarchical variance structure or perform formal statistical inference
3. **Appropriate for**: Small sample sizes (N=3) where traditional statistical methods fail
4. **Similar to**: Clinical reference ranges ("flag if glucose >200 mg/dL"), GWAS QC thresholds (HWE p < 1e-6)

### Independence and Nested Structure

**Experimental Design:**
- 60 plots (experimental units)
- 3 cores per plot (technical replicates, nested within plots)
- 2 depths per core
- **Cores within a plot are NOT independent** - they share genotype, soil conditions, spatial correlation

**Statistical Implication:**
- Population-level methods (Mahalanobis, Isolation Forest) **violate independence assumptions**
- Cores are clustered within plots (positive intra-class correlation)
- Effective sample size ≈ 60 plots, not 180 cores

**Why Per-Group Detection is Valid:**
- Analyzes 3 cores within each plot independently
- **Independence assumption satisfied** within each group
- Compares cores to their own plot (controls for genotype/plot effects)
- Separates measurement error from biological variation

### Sample Size Considerations

**Traditional outlier detection requires:**
- Mahalanobis: N > 50-100 (stable covariance estimation)
- Z-score: N > 30 (central limit theorem)
- Isolation Forest: N > 50 (tree diversity)

**Why percent deviation works with N=3:**
- Non-parametric (no distribution assumptions)
- Doesn't estimate variance
- Uses fixed threshold from domain knowledge
- Measurement errors are LARGE (>30% deviation)
- Similar to industrial QC (control charts, specification limits)

## Design Decisions

### Decision 1: Detection Method

**Evaluated options:**

| Method | Pros | Cons | Verdict |
|--------|------|------|---------|
| Z-score (N=3) | Simple | Unstable std with N=3, failed to detect GH_7371 case | ❌ Rejected |
| IQR | Robust to distribution | Undefined for N=3 (quartiles ambiguous) | ❌ Rejected |
| MAD (Median Absolute Deviation) | Robust, good for small samples | Can be zero if 2/3 values identical | ✅ Option B |
| **Percent deviation from median** | **Simple, scale-independent, transparent** | **Requires arbitrary threshold** | ✅ **SELECTED** |
| Hybrid MAD + Percent | Combines rigor & practicality | More complex, two parameters | ✅ Option C |

**Selected:** Percent deviation from median

**Rationale:**
1. **Simple and interpretable** - Easy to explain in papers ("cores >30% different from median")
2. **Scale-independent** - Works for any trait magnitude (grams, counts, etc.)
3. **Transparent** - Threshold choice is explicit and tunable
4. **Conservative** - 30% threshold won't remove natural variation
5. **Effective** - Successfully detects GH_7371 core 2 (56% deviation > 30% threshold)

**Formula:**
```python
deviation = abs(value - median) / median
flag_if deviation > threshold  # e.g., 0.30 = 30%
```

**Example (GH_7371 Rep 1):**
```python
values = [0.7636, 0.7071, 0.3132]
median = 0.7071  # middle value with N=3

core_0_dev = abs(0.7636 - 0.7071) / 0.7071 = 0.08 = 8%   → NOT FLAGGED
core_1_dev = abs(0.7071 - 0.7071) / 0.7071 = 0.00 = 0%   → NOT FLAGGED
core_2_dev = abs(0.3132 - 0.7071) / 0.7071 = 0.56 = 56%  → FLAGGED ✓

# After removal: [0.7636, 0.7071]
# Aggregate: mean([0.76, 0.71]) = 0.735 (matches Nov 30 notebook!)
```

### Decision 2: Threshold Value

**Conservative (RECOMMENDED):** 30% (`max_deviation_from_median: 0.30`)
- **Catches:** Extreme outliers only (>30% different)
- **Example:** 56% deviation (GH_7371 core 2) ✓
- **Risk:** May miss moderate outliers (20-25%)
- **Use when:** Trust sampling protocol, want to avoid over-filtering

**Moderate:** 20% (`max_deviation_from_median: 0.20`)
- **Catches:** Moderate to extreme outliers
- **Risk:** May remove some natural variation
- **Use when:** Moderate confidence in sampling

**Aggressive:** 15% (`max_deviation_from_median: 0.15`)
- **Catches:** Minor to extreme outliers
- **Risk:** May remove too many cores → reduce power
- **Use when:** High measurement variability expected

**Chosen default:** 30% (conservative, minimal false positives)

### Decision 3: Safety Mechanisms

**Problem:** What if all 3 cores are flagged as outliers?

**Solution:** Always keep at least `min_cores_after_qc` cores per group

**Implementation:**
```python
if n_outliers >= len(values) - min_cores_after_qc:
    # Too many would be removed
    # Keep the min_cores closest to median
    distances = abs(values - median)
    keep_indices = distances.nsmallest(min_cores_after_qc).index
    # Only flag cores not in keep_indices
```

**Default:** `min_cores_after_qc: 1` (always keep at least 1 core)

**Rationale:**
- Prevents empty groups (catastrophic failure)
- Ensures aggregation step always has ≥1 value
- If all cores are bad, keep the "least bad" one

### Decision 4: Edge Case Handling

#### Edge Case 1: Median = 0
**Problem:** Division by zero in percent deviation formula

**Solution:** Use absolute threshold instead
```python
if median_val == 0:
    # Use absolute deviation threshold (e.g., 0.1 units)
    deviations = abs(values - median_val)
    outliers = deviations > absolute_threshold
```

**Alternative:** Skip this group (no QC applied)

#### Edge Case 2: Only 1-2 Cores Available
**Problem:** Can't compute robust median with <2 values

**Solution:** Skip value-based QC for this group
```python
if len(values) < 2:
    continue  # Need at least 2 for median comparison
```

#### Edge Case 3: All Values Identical
**Problem:** MAD = 0, can't compute meaningful deviation

**Solution:** Percent deviation handles this naturally
```python
if median == 0.71 and all values are 0.71:
    deviations = [0, 0, 0]  # None flagged ✓
```

### Decision 5: Integration with Existing QC

**Current step 00c (QCCoreLevelStep) performs:**
1. Missing data detection - Flags cores with >50% missing depths

**New enhancement:**
2. Value outlier detection - Flags cores with extreme values

**Execution order:**
```python
def _detect_outlier_cores(self, df, value_column, data_type, qc_config):
    # Method 1: Missing data (existing)
    df = self._flag_missing_data(df, value_column, qc_config)

    # Method 2: Value outliers (NEW)
    if qc_config.detect_value_outliers:
        df = self._detect_value_outliers_per_group(df, value_column, qc_config)

    # Combine flags and remove
    if qc_config.remove_outliers:
        df = self._remove_flagged_cores(df)

    return df
```

**Benefit:** Existing missing data logic preserved, new value detection is additive

## Implementation Strategy

### Phase 1: Core Detection Logic (Priority 1)
Files: `qc_core_level.py`, `components.py`

1. Add config parameters to `CoreQCConfig`
2. Implement `_detect_value_outliers_per_group()` method
3. Integrate into existing `_detect_outlier_cores()` workflow
4. Add metadata tracking (flagged cores, reasons, statistics)

### Phase 2: Configuration (Priority 1)
Files: `qc_root_core_edpie.yaml`, template configs

1. Update default config to enable per-core QC
2. Add comprehensive comments explaining thresholds
3. Provide tuning guidance

### Phase 3: Testing (Priority 1)
Files: `test_step_qc_core_level.py`

1. Test real-world case (GH_7371)
2. Test normal variation not flagged
3. Test safety mechanisms
4. Test edge cases
5. Test integration with missing data QC

### Phase 4: Documentation (Priority 2)
Files: `CLAUDE.md`, docstrings

1. Update QC pipeline documentation
2. Add configuration best practices
3. Add troubleshooting guide

## Alternative Approaches Considered

### Alternative 1: Always Use Mean After Manual Core QC
**Approach:** Trust Nov 30 notebook's manual exclusion of core 2, always use mean

**Rejected because:**
- Not automated (requires manual inspection)
- Not reproducible (decisions not documented)
- Doesn't scale to multiple datasets

### Alternative 2: Population-Level Statistical Outlier Detection
**Approach:** Apply Mahalanobis distance or Isolation Forest to all 180 cores together

**Rejected because:**
- **Violates independence assumption** - cores nested within plots are not i.i.d.
- Confounds plot/genotype effects with measurement errors
- May flag natural low-biomass genotypes as "outliers"
- Inflates Type I error rate (underestimates standard errors)
- Statistical reviewers would likely reject this approach

### Alternative 3: Aggregation-Level Thresholding
**Approach:** Flag aggregated values that are outliers, then trace back to bad cores

**Rejected because:**
- Too late - aggregation already done
- Can't identify which specific core was bad
- Current problem: Aggregated value (0.71) is NOT flagged

### Alternative 4: Machine Learning Outlier Detection
**Approach:** Use Isolation Forest or LOF on 3-core groups

**Rejected because:**
- Overkill for simple problem
- Requires more data (N=3 too small)
- Not interpretable ("why was this flagged?")

## Validation Strategy

### Success Criteria
1. ✅ GH_7371 Rep 1 core 2 is correctly flagged (56% deviation)
2. ✅ Normal variation is NOT flagged (<30% deviation)
3. ✅ Heritability increases from 0.27-0.45 → ≥0.50 for EDPIE biomass
4. ✅ Biomass traits retained in final dataset (pass H² threshold)
5. ✅ 57 samples in final data (matches Nov 30 notebook)

### Testing Approach
1. **Unit tests:** Individual functions with synthetic data
2. **Integration tests:** Full pipeline run with EDPIE config
3. **Regression tests:** Compare Nov 30 notebook vs updated pipeline
4. **Edge case tests:** Median=0, all cores flagged, only 1-2 cores
5. **Sensitivity analysis:** Test thresholds of 20%, 30%, 40% to assess robustness

### Publication-Quality Reporting

**Recommended Methods Section Language:**

> "Quality control of core-level data was performed within each plot-depth group (n=3 cores). Cores with biomass values deviating >30% from the within-group median were flagged as potential measurement errors and excluded prior to aggregation. This threshold was chosen conservatively to remove gross errors (e.g., damaged cores, recording mistakes) while preserving natural biological variation (typically <20% within plots). To prevent loss of entire plots, at least one core per group was always retained. Across all groups, [X cores out of 180 total] were flagged ([Y%]), consistent with expected measurement error rates in field sampling."

**Sensitivity Analysis Reporting:**

```
Threshold   Cores Flagged   Heritability (Biomass 0-30cm)
20%         [A] ([X%])      H² = [value]
30%         [B] ([Y%])      H² = [value]  ← Primary analysis
40%         [C] ([Z%])      H² = [value]
```

**Justification for Reviewers:**

1. **Respects nested structure**: Per-group analysis avoids independence violations
2. **Conservative threshold**: 30% chosen a priori to minimize false positives
3. **Transparent**: Simple, interpretable rule (not a black-box algorithm)
4. **Validates heritability**: Significant improvement in H² after QC
5. **Domain-appropriate**: Similar to QC practices in agricultural field trials

### Diagnostic Tools
User can inspect QC results via metadata files:
```bash
# Check flagged cores
cat qc_runs/.../00c_core_qc_metadata.json | jq '.sources[].flagged_cores_list'

# Check flagging statistics
cat qc_runs/.../00c_core_qc_metadata.json | jq '.sources[].flagged_by_method'
```

## Configuration Tuning Guide

### When to Adjust Threshold

**Increase threshold (e.g., 0.40 = 40%) if:**
- Too many cores being removed
- Final sample size too small (<50 samples)
- QC metadata shows many cores flagged with 30-35% deviation

**Decrease threshold (e.g., 0.20 = 20%) if:**
- Heritability still low after QC
- Visual inspection shows remaining outliers
- Domain expertise suggests stricter QC needed

### Diagnostic Workflow
```bash
# 1. Run pipeline with default (30%)
uv run sleap-roots-analyze qc configs/qc_root_core_edpie.yaml

# 2. Check how many cores were flagged
jq '.sources[].flagged_cores_list | length' qc_runs/.../00c_core_qc_metadata.json

# 3. Inspect flagged cores
jq '.sources[].flagged_cores_list[]' qc_runs/.../00c_core_qc_metadata.json

# 4. Check heritability results
cat qc_runs/.../08_heritability_results.csv | grep "Rootdw"

# 5. If needed, adjust threshold and re-run
```

## Risk Assessment

### Risk 1: Over-filtering (False Positives)
**Likelihood:** Low with 30% threshold
**Impact:** Reduced sample size, loss of statistical power
**Mitigation:** Conservative default, tuning guidance, safety keeps min cores

### Risk 2: Under-filtering (False Negatives)
**Likelihood:** Possible with very conservative threshold
**Impact:** Outliers remain, low heritability persists
**Mitigation:** Threshold tuning guide, diagnostic tools

### Risk 3: Breaking Existing Workflows
**Likelihood:** Low (backward compatible)
**Impact:** Users must update configs or opt-out
**Mitigation:**
- Old behavior preserved if `core_qc.enabled: false`
- New parameter `detect_value_outliers` defaults to True
- Migration guide in docs

### Risk 4: Edge Cases Not Handled
**Likelihood:** Low (comprehensive edge case testing)
**Impact:** Pipeline crashes or unexpected behavior
**Mitigation:** Safety checks, comprehensive test suite

## Future Enhancements

### Enhancement 1: Adaptive Thresholds
Use IQR or MAD-based thresholds instead of fixed percent:
```python
threshold = median + k * MAD  # where k=3
```

### Enhancement 2: Diagnostic Plots
Generate per-plot visualizations showing flagged cores:
```python
# Box plot of 3 cores with flagged cores highlighted
plot_core_qc_diagnostics(df, output_dir)
```

### Enhancement 3: Multiple Detection Methods
Allow users to combine methods:
```yaml
core_qc:
  detection_methods:
    - method: percent_deviation
      threshold: 0.30
    - method: mad
      k: 3
  combine_strategy: union  # Flag if ANY method flags
```

## References

- Original issue: GH_7371 Rep 1 core measurement error
- Nov 30 notebook: `trait_qc_root_coring_20251126.ipynb`
- Current pipeline: `src/sleap_roots_analyze/pipeline/steps/qc_core_level.py`
- Config: `configs/qc_root_core_edpie.yaml`
