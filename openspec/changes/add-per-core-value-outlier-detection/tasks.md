# Tasks: Per-Core Value Outlier Detection

## Phase 1: Core Implementation (Priority 1)

- [x] **Update CoreQCConfig dataclass** (`src/sleap_roots_analyze/pipeline/config/components.py`)
  - Add `detect_value_outliers: bool = False` field (opt-in, disabled by default)
  - Add `max_deviation_from_median: float = 0.30` field
  - Add `min_cores_after_qc: int = 1` field
  - Update docstring with new parameters, statistical rationale, and examples
  - Document that this is QC (measurement error detection), not statistical hypothesis testing
  - **Validation:** Fields present, defaults correct, types enforced

- [x] **Implement _detect_value_outliers_per_group() method** (`src/sleap_roots_analyze/pipeline/steps/qc_core_level.py`)
  - Calculate median for each (Plot, Rep, Geno, Depth) group
  - Compute percent deviation: `abs(value - median) / median`
  - Flag cores exceeding `max_deviation_from_median` threshold
  - Handle edge case: median == 0 (skip group)
  - Handle edge case: len(values) < 2 (skip group)
  - Apply safety: Keep at least `min_cores_after_qc` cores per group
  - Return DataFrame with outlier flags and reasons
  - **Validation:** Function exists, returns flagged cores, handles edge cases

- [x] **Integrate value detection into _detect_outlier_cores()** (`qc_core_level.py`)
  - Call `_flag_missing_data()` first (existing logic)
  - Call `_detect_value_outliers_per_group()` if `detect_value_outliers == True`
  - Combine flags from both methods
  - Remove flagged cores if `remove_outliers == True`
  - **Validation:** Both detection methods run, flags combined correctly

- [x] **Update metadata tracking** (`qc_core_level.py`)
  - Add `flagged_by_method` breakdown (missing_data vs value_outlier)
  - Include per-core details: value, median, deviation_pct, threshold
  - Ensure metadata JSON includes all diagnostic info
  - **Validation:** Metadata file contains new fields, values correct

## Phase 2: Configuration (Priority 1)

- [x] **Update qc_root_core_edpie.yaml config**
  - Change `core_qc.enabled: false` → `true` (enable core-level QC)
  - Add `detect_value_outliers: false` (opt-in, disabled by default for backward compatibility)
  - Add `max_deviation_from_median: 0.30` (conservative threshold when enabled)
  - Add `min_cores_after_qc: 1`
  - Update comments explaining:
    - Rationale (measurement error detection, not statistical testing)
    - Why per-group detection respects nested structure
    - Nov 30 comparison and GH_7371 example
    - Statistical considerations (N=3, independence, QC paradigm)
  - Add threshold tuning guidance
  - **Validation:** Config parses, parameters present, comments clear and accurate

- [ ] **Update qc_cylinder_edpie.yaml (if exists)**
  - Apply same changes as root_core config
  - **Validation:** Config parses correctly

- [ ] **Create config template** (`configs/templates/qc_root_core_with_value_qc.yaml`)
  - Copy from qc_root_core_edpie.yaml
  - Add comprehensive comments for each parameter
  - Include threshold tuning examples
  - **Validation:** Template is complete and well-documented

## Phase 3: Testing (Priority 1)

- [x] **Test GH_7371 real-world case** (`tests/test_step_qc_core_level.py`)
  - Create test data: cores = [0.7636, 0.7071, 0.3132]
  - Run QCCoreLevelStep with `max_deviation: 0.30`
  - Assert core 2 (56% deviation) is flagged
  - Assert cores 0 and 1 are NOT flagged
  - Assert metadata contains correct deviation percentages
  - **Validation:** Test passes, GH_7371 case detected

- [x] **Test normal variation not flagged**
  - Create test data: cores = [0.72, 0.68, 0.75] (±10% variation)
  - Run with `max_deviation: 0.30`
  - Assert NO cores are flagged
  - **Validation:** Test passes, normal variation preserved

- [x] **Test safety: keeps minimum cores**
  - Create test data: cores = [0.1, 0.5, 0.9] (all different)
  - Run with `max_deviation: 0.20` (strict) and `min_cores_after_qc: 1`
  - Assert at least 1 core is kept (closest to median)
  - Assert NOT all cores removed
  - **Validation:** Test passes, safety mechanism works

- [x] **Test edge case: median == 0**
  - Create test data with median = 0
  - Run value outlier detection
  - Assert group is skipped (no division by zero error)
  - **Validation:** Test passes, edge case handled

- [x] **Test edge case: only 1-2 cores available**
  - Create test data with len(values) < 2
  - Run value outlier detection
  - Assert detection is skipped
  - Assert remaining cores kept for aggregation
  - **Validation:** Test passes, insufficient data handled

- [x] **Test combined: missing data + value outliers**
  - Create test data with both NaN values and extreme values
  - Run both detection methods
  - Assert both types of outliers are flagged
  - Assert flags are combined correctly
  - **Validation:** Test passes, integration works

- [x] **Test backward compatibility**
  - Run with `core_qc.enabled: false`
  - Assert Step 00c is skipped
  - Assert behavior matches old pipeline
  - **Validation:** Test passes, backward compatible

- [x] **Test metadata structure**
  - Run pipeline and check `00c_core_qc_metadata.json`
  - Assert contains `flagged_by_method` breakdown
  - Assert contains per-core details (value, median, deviation_pct)
  - **Validation:** Metadata file format correct

## Phase 4: Integration Testing & Sensitivity Analysis (Priority 2)

- [ ] **Run full EDPIE pipeline with new QC**
  - Execute: `uv run sleap-roots-analyze qc configs/qc_root_core_edpie.yaml` (with `detect_value_outliers: true`)
  - Assert pipeline completes without errors
  - Assert GH_7371 Rep 1 core 2 is flagged in metadata
  - Assert Step 00d aggregation uses only remaining cores
  - **Validation:** Pipeline runs, core removed

- [ ] **Verify heritability improvement**
  - Check `08_heritability_results.csv` for biomass traits
  - Assert Rootdw 15Cm heritability ≥ 0.50
  - Assert Rootdw 45Cm heritability ≥ 0.50
  - Compare to Dec 5 run (0.27, 0.45) - should be higher
  - **Validation:** Heritability improved

- [ ] **Verify final sample count**
  - Check `07_data_outliers_removed.csv` for sample count
  - Assert 57 samples (not 58)
  - Assert GH_7371 Rep 1 removed at trait level
  - **Validation:** Sample count matches Nov 30

- [ ] **Verify biomass traits retained**
  - Check `10_final_data.csv` for column list
  - Assert "Rootdw 15Cm" column present
  - Assert "Rootdw 45Cm" column present
  - Assert biomass NOT removed by heritability filter
  - **Validation:** Biomass traits in final data

- [ ] **Sensitivity Analysis: Test multiple thresholds**
  - Run pipeline with `max_deviation_from_median: 0.20` (aggressive)
  - Run pipeline with `max_deviation_from_median: 0.30` (conservative, default)
  - Run pipeline with `max_deviation_from_median: 0.40` (very conservative)
  - For each threshold, record:
    - Number of cores flagged
    - Final heritability values
    - Final sample count
  - **Validation:** Results are robust across reasonable threshold range

- [ ] **Document sensitivity analysis results**
  - Create table showing threshold vs. cores flagged vs. heritability
  - Verify that 30% threshold is appropriate (balances false pos/neg)
  - Include in QC metadata or supplementary documentation
  - **Validation:** Sensitivity analysis demonstrates robustness

## Phase 5: Documentation (Priority 2)

- [ ] **Update qc_core_level.py docstrings**
  - Add explanation of value outlier detection as quality control
  - Clarify this is measurement error detection, not statistical testing
  - Document statistical considerations (nested structure, N=3, independence)
  - Explain why per-group detection is appropriate
  - Document when to use (measurement errors expected)
  - Provide formula and threshold guidance
  - Add examples (GH_7371 case)
  - Include publication-quality methods section language
  - **Validation:** Docstrings complete, accurate, and publication-ready

- [ ] **Update CoreQCConfig docstring**
  - Explain new parameters with statistical context
  - Document default values and rationale (opt-in, conservative threshold)
  - Provide usage recommendations
  - Include threshold tuning examples
  - Clarify QC vs. statistical testing paradigm
  - **Validation:** Docstring clear, helpful, and statistically accurate

- [ ] **Update CLAUDE.md configuration section**
  - Add per-core QC parameters to config docs
  - Explain when to enable value outlier detection (opt-in)
  - Document statistical framework (QC, not hypothesis testing)
  - Explain nested structure and independence considerations
  - Provide threshold tuning workflow with sensitivity analysis
  - Add publication-quality methods section template
  - Add troubleshooting section
  - **Validation:** Documentation complete and scientifically accurate

- [ ] **Update config template comments**
  - Explain rationale for per-core QC (Nov 30 comparison)
  - Provide threshold tuning guidance
  - Link to GH_7371 example case
  - **Validation:** Comments helpful for users

## Phase 6: Validation & Cleanup (Priority 3)

- [x] **Run black formatter**
  - Format: `uv run black src/sleap_roots_analyze/pipeline/steps/qc_core_level.py`
  - Format: `uv run black src/sleap_roots_analyze/pipeline/config/components.py`
  - Format: `uv run black tests/test_step_qc_core_level.py`
  - **Validation:** Code formatted correctly

- [x] **Run ruff linter**
  - Lint: `uv run ruff check src/sleap_roots_analyze/pipeline/`
  - Fix any issues
  - **Validation:** No linting errors

- [x] **Run full test suite**
  - Execute: `uv run pytest tests/test_step_qc_core_level.py -v`
  - Assert all tests pass
  - **Validation:** 100% test pass rate

- [ ] **Check test coverage**
  - Execute: `uv run pytest tests/test_step_qc_core_level.py --cov=src/sleap_roots_analyze/pipeline/steps/qc_core_level --cov-report=term-missing`
  - Assert coverage ≥95% for qc_core_level.py
  - **Validation:** High coverage achieved

- [ ] **Update CHANGELOG.md**
  - Add entry under "## [Unreleased]"
  - Document new feature and config changes
  - Reference issue/proposal
  - **Validation:** Changelog updated

- [ ] **Create summary document**
  - Update QC_ISSUE_SUMMARY.md or create new doc
  - Explain solution and validation results
  - Include before/after heritability comparison
  - **Validation:** Summary document complete

## Dependencies

### Sequential Dependencies
- Phase 1 (Implementation) must complete before Phase 3 (Testing)
- Phase 2 (Configuration) must complete before Phase 4 (Integration Testing)
- Phase 3 (Testing) must complete before Phase 6 (Validation)

### Parallel Work
- Phase 1 (Implementation) and Phase 2 (Configuration) can be done in parallel
- Phase 5 (Documentation) can be done in parallel with Phase 4 (Integration Testing)

## Acceptance Criteria

All tasks must be completed and all validation checks must pass before the change is considered complete. Key success metrics:

1. ✅ All unit tests pass (100% pass rate)
2. ✅ Test coverage ≥95% for modified code
3. ✅ GH_7371 Rep 1 core 2 correctly detected (56% deviation)
4. ✅ EDPIE biomass heritability ≥0.50 (up from 0.27-0.45)
5. ✅ Biomass traits retained in final dataset
6. ✅ Final sample count = 57 (matches Nov 30 notebook)
7. ✅ No linting errors
8. ✅ Documentation complete and clear
9. ✅ Backward compatibility preserved (old configs work)
10. ✅ Metadata provides diagnostic information for threshold tuning
