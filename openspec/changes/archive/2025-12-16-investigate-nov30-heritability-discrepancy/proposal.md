# Proposal: Investigate Nov 30 Heritability Discrepancy

## Problem Statement

**CRITICAL DATA INTEGRITY ISSUE:** Systematic heritability discrepancies exist between the Nov 30, 2024 notebook results (used as design reference for per-core value outlier detection) and the current baseline pipeline with core-level QC disabled.

### Evidence of Discrepancy

**Root Biomass Traits (CATASTROPHIC):**
- Rootdw 15Cm (0-30cm depth): Nov 30 H² = 0.750 → Baseline H² = 0.270 (-64% loss)
- Rootdw 45Cm (30-60cm depth): Nov 30 H² = 0.730 → Baseline H² = 0.449 (-38% loss)
- **Impact:** Both traits dropped below 0.5 threshold, removed from final dataset

**Above-Ground Traits (STABLE):**
- 20 above-ground traits show <2% heritability difference
- Example: Ph M Cm: 0.974 → 0.970 (-0.4%)
- Example: Gw M G1000Grn: 0.941 → 0.941 (0.0%)

**Sample Counts (NEARLY IDENTICAL):**
- Nov 30 Notebook: 57 samples
- Baseline Pipeline: 58 samples
- Difference: +1 sample (negligible)

### Why This Matters

1. **Per-core value outlier detection was built on incorrect baseline**
   - Feature designed to close gap from H² = 0.27/0.45 → 0.75/0.73
   - Assumed gap was due to gross measurement errors (e.g., GH_7371 core 2 with 56% deviation)
   - Reality: 30% threshold too aggressive, removes 53% of biomass data
   - Result: H² degraded to 0.43/0.33 (WORSE for 45cm depth!)

2. **Root cause unknown**
   - Design document claims Nov 30 used "mean of cores 0 & 1 only (excluded core 2)"
   - Baseline uses "median of all 3 cores"
   - BUT: This claim needs verification from actual notebook code

3. **Data provenance unclear**
   - Are we using the same source CSVs as Nov 30?
   - Was manual pre-processing applied before notebook?
   - Were cleanup thresholds different?

4. **Cannot optimize feature until baseline is correct**
   - Current 30% threshold is demonstrably too aggressive
   - Can't tune threshold without understanding what Nov 30 actually did
   - Risk of chasing wrong optimization target

## Proposed Solution

**Forensic investigation and reproduction study** to:
1. Determine exact data processing steps used in Nov 30 notebook
2. Identify root cause of heritability discrepancy
3. Reproduce Nov 30 results using current pipeline infrastructure
4. Establish correct baseline for per-core QC feature validation

## Success Criteria

### Primary Goals (Must Have)
1. ✅ **Root cause identified**: Document exact differences between Nov 30 and baseline processing
2. ✅ **Nov 30 results reproduced**: Achieve H² ≥ 0.70 for both biomass traits using pipeline
3. ✅ **Correct baseline established**: Create validated reference configuration for future work

### Secondary Goals (Should Have)
4. ✅ **Data provenance documented**: CSV sources, cleanup thresholds, manual exclusions tracked
5. ✅ **Design document corrected**: Update add-per-core-value-outlier-detection with accurate claims

### Stretch Goals (Nice to Have)
6. ✅ **Automated validation**: Test that compares pipeline output to Nov 30 reference
7. ✅ **Pipeline refactoring**: If systematic differences found, update default behaviors

## Risks & Mitigation

### Risk 1: Nov 30 used manual, non-reproducible processing
- **Likelihood:** Medium
- **Impact:** High (feature may need complete redesign)
- **Mitigation:** Document manual steps, create automated equivalent

### Risk 2: Source data has changed since Nov 30
- **Likelihood:** Low (CSVs have May 2025 timestamps)
- **Impact:** High (invalidates entire comparison)
- **Mitigation:** Hash comparison of source files, check git history

### Risk 3: Heritability calculation method differs
- **Likelihood:** Low (both use statsmodels mixed models)
- **Impact:** Medium
- **Mitigation:** Compare variance components, not just final H²

### Risk 4: Investigation reveals no actionable differences
- **Likelihood:** Low
- **Impact:** Medium (time spent, no clear path forward)
- **Mitigation:** Document findings, pivot to empirical threshold optimization

## Out of Scope

- **Not investigating:** Root counting trait discrepancies (Nov 30 didn't measure these)
- **Not implementing:** New core-level QC methods (investigation only)
- **Not optimizing:** Per-core value outlier detection threshold (blocked until baseline correct)
- **Not refactoring:** Pipeline infrastructure (unless root cause demands it)

## Timeline & Dependencies

### Phase 1: Investigation (Priority 1)
- **Duration:** 1-2 days
- **Dependencies:** Access to Nov 30 notebook, source CSV files
- **Deliverables:** Root cause analysis document, reproduction strategy

### Phase 2: Reproduction (Priority 1)
- **Duration:** 1-2 days
- **Dependencies:** Phase 1 complete
- **Deliverables:** Pipeline configuration that reproduces Nov 30 results

### Phase 3: Validation & Documentation (Priority 2)
- **Duration:** 1 day
- **Dependencies:** Phase 2 complete
- **Deliverables:** Test suite, updated design docs, baseline reference config

### Total Estimated Effort: 3-5 days

## References

- **Per-core QC feature:** `openspec/changes/add-per-core-value-outlier-detection/`
- **Nov 30 notebook:** `c:/repos/sleap-roots-analyze/trait_qc_root_coring_20251126.ipynb`
- **Nov 30 results:** `c:/repos/runs/run_20251130_193257/analysis_outputs/heritability_results.json`
- **Baseline results:** `qc_runs/EDPIE Root Core Full QC_20251209_110820/09_heritability_diagnostics.csv`
- **Analysis document:** `HERITABILITY_DISCREPANCY_ANALYSIS.md`
- **Source CSVs:**
  - `C:/Users/Elizabeth/Desktop/phenotyping/EDPIE_paper/data/root_coring/rearranged_root_biomass_dw.csv`
  - `C:/Users/Elizabeth/Desktop/phenotyping/EDPIE_paper/data/field/Field_2024_aboveground.csv`

## Related Changes

- **Blocks:** Optimization of `add-per-core-value-outlier-detection` (can't tune without correct baseline)
- **Updates:** `add-per-core-value-outlier-detection/design.md` (Nov 30 claims need verification)
- **Informs:** Future root core QC enhancements (understanding of what works)
