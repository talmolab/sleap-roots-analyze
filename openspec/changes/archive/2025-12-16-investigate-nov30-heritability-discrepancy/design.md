# Design: Investigate Nov 30 Heritability Discrepancy

## Background

The add-per-core-value-outlier-detection feature was designed based on a comparison between:
- **Nov 30, 2024 notebook:** Root biomass H² = 0.75/0.73 (HIGH, above 0.5 threshold)
- **Dec 5, 2024 pipeline baseline:** Root biomass H² = 0.27/0.45 (LOW, below 0.5 threshold)

The design document claimed this discrepancy was due to:
> "Nov 30 notebook used mean of cores 0 & 1 only (excluded core 2) → 0.735"
> "Pipeline used median of all 3 cores → 0.707"

However, **current investigation reveals this explanation is incomplete or incorrect:**
- Baseline pipeline (QC OFF) still shows H² = 0.27/0.45
- Above-ground traits are nearly identical (<2% difference)
- Sample counts differ by only 1 (57 vs 58)
- Pattern suggests root-specific processing difference, not global sample exclusion

## Investigation Strategy

### Phase 1: Code Archaeology (Nov 30 Notebook Analysis)

**Objective:** Extract exact processing steps from Nov 30 notebook

**Method:**
1. Parse notebook JSON to extract code cells in execution order
2. Identify root biomass data loading and processing
3. Document aggregation method (mean/median), core selection, exclusions
4. Extract cleanup thresholds (max_nan_fraction, max_zeros, etc.)
5. Check for manual data edits or pre-processing

**Key Questions:**
- What aggregation method was actually used? (mean vs median)
- Were any cores systematically excluded? (e.g., core 2 only, or per-plot decisions)
- What cleanup configuration was applied?
- Were there manual data corrections?

**Output:** `nov30_processing_steps.md` documenting exact workflow

### Phase 2: Data Forensics (Source CSV Comparison)

**Objective:** Verify we're using identical source data

**Method:**
1. Hash compare source CSVs between Nov 30 and current
2. Check CSV modification timestamps
3. Compare row counts, column names, value distributions
4. Identify any schema or content differences

**Key Questions:**
- Are source CSVs identical to Nov 30?
- Were CSVs edited/corrected after Nov 30?
- Do aggregated core values match before pipeline processing?

**Output:** `data_provenance_report.md` with hash comparisons and diff summary

### Phase 3: Variance Component Analysis

**Objective:** Understand heritability calculation differences

**Method:**
1. Compare variance components (Vg, Ve) between Nov 30 and baseline
2. Check between-genotype vs within-genotype variance
3. Analyze replicate structure (2.85 vs 2.9 mean reps per genotype)
4. Examine residual distributions

**Key Questions:**
- Is genetic variance (Vg) drastically different?
- Is residual variance (Ve) inflated in baseline?
- Does replicate imbalance explain differences?

**Output:** `variance_decomposition_comparison.csv` with side-by-side metrics

### Phase 4: Reproduction Experiments

**Objective:** Test hypotheses and reproduce Nov 30 results

**Experiment 1: Mean vs Median Aggregation**
```python
# Test if aggregation method explains discrepancy
config_mean = qc_config.copy()
config_mean.root_core.sources[0].aggregation_method = "mean"
# Run pipeline, compare H²
```

**Experiment 2: Core Exclusion (Cores 0&1 Only)**
```python
# Test design document claim
config_cores01 = qc_config.copy()
# Add filter: exclude core_id == 2 before aggregation
# Run pipeline, compare H²
```

**Experiment 3: Nov 30 Cleanup Thresholds**
```python
# Test if cleanup differences explain it
config_nov30_cleanup = qc_config.copy()
config_nov30_cleanup.cleanup.max_nan_fraction = <nov30_value>
config_nov30_cleanup.cleanup.max_zeros_per_trait = <nov30_value>
# Run pipeline, compare H²
```

**Experiment 4: Combined Processing**
```python
# Test combination of all Nov 30 processing steps
config_full_nov30 = apply_nov30_processing_steps(qc_config)
# Run pipeline, compare H²
# Target: Achieve H² ≥ 0.70 for both biomass traits
```

**Success Criterion:** Reproduce Nov 30 H² within ±0.05

**Output:** `reproduction_experiments.md` with results table

## Architectural Considerations

### Option 1: Configuration-Only Solution (Preferred)
**If:** Nov 30 differences are purely configuration-based (thresholds, aggregation method)

**Implementation:**
- Create `configs/templates/qc_root_core_nov30_reproduction.yaml`
- Document differences in config comments
- No code changes required

**Pros:** Simple, maintainable, backward compatible
**Cons:** None

### Option 2: Pipeline Enhancement
**If:** Nov 30 used processing steps not currently supported (e.g., per-plot core selection)

**Implementation:**
- Add new pipeline step: `00c2_apply_manual_core_exclusions`
- Read exclusion list from config or CSV
- Insert before aggregation step

**Pros:** Generalizable, supports manual QC workflows
**Cons:** Adds complexity, requires testing

### Option 3: Accept Discrepancy
**If:** Nov 30 used non-reproducible manual processing

**Implementation:**
- Document manual steps in design.md
- Update per-core QC feature to use empirical baseline (H² = 0.27/0.45)
- Adjust success criteria accordingly

**Pros:** Pragmatic, moves forward with feature
**Cons:** Doesn't achieve original H² = 0.75/0.73 target

## Design Decisions

### Decision 1: Investigation Scope
**Question:** How deep should investigation go?

**Options:**
A. **Quick scan** (1 day): Check notebook code, try mean aggregation
B. **Thorough forensics** (3 days): Full code archaeology, data forensics, reproduction experiments (CHOSEN)
C. **Complete audit** (1 week): Line-by-line notebook replay, variance component deep dive

**Choice:** B (Thorough forensics)

**Rationale:**
- Quick scan risks missing root cause
- Complete audit likely overkill given clear pattern (root biomass only affected)
- Thorough forensics balances speed and rigor

### Decision 2: Reproduction Strategy
**Question:** What defines successful reproduction?

**Options:**
A. **Exact match** (H² within ±0.01)
B. **Close match** (H² within ±0.05) (CHOSEN)
C. **Threshold match** (H² ≥ 0.50)

**Choice:** B (Close match)

**Rationale:**
- Exact match may be impossible due to floating-point variance in statsmodels
- Close match (±0.05) is scientifically meaningful
- Threshold match too lenient (0.50 vs 0.75 is large difference)

### Decision 3: Baseline Correction
**Question:** If Nov 30 used different processing, should we update baseline?

**Options:**
A. **Keep current baseline** (median, all cores)
B. **Adopt Nov 30 processing** (if superior and reproducible) (CHOSEN)
C. **Offer both** (create two reference configs)

**Choice:** B (Adopt Nov 30 processing) IF:
- Nov 30 processing is reproducible via configuration
- Results are scientifically justified (not arbitrary)
- Improvement is substantial (H² gain ≥ 0.20)

**Rationale:**
- Higher heritability enables better trait selection
- If Nov 30 found better processing, we should use it
- Must be justified, not just "tuned to get high H²"

## Testing Strategy

### Unit Tests
- Not applicable (investigation, not implementation)

### Integration Tests
**Test 1: Reproduction Validation**
```python
def test_nov30_reproduction():
    """Verify Nov 30 results are reproducible via pipeline."""
    config = load_config("configs/qc_root_core_nov30_reproduction.yaml")
    result = run_qc_pipeline(config)

    # Check biomass heritability
    assert result["Rootdw 15Cm"]["heritability"] >= 0.70
    assert result["Rootdw 45Cm"]["heritability"] >= 0.70

    # Check sample count matches Nov 30
    assert result["n_observations"] == 57

    # Check above-ground traits remain stable
    assert abs(result["Ph M Cm"]["heritability"] - 0.974) < 0.01
```

**Test 2: Aggregation Method Comparison**
```python
def test_aggregation_method_comparison():
    """Compare mean vs median aggregation impact."""
    config_mean = create_config(aggregation="mean")
    config_median = create_config(aggregation="median")

    result_mean = run_pipeline(config_mean)
    result_median = run_pipeline(config_median)

    # Document difference
    diff = result_mean["Rootdw 15Cm"]["heritability"] - \
           result_median["Rootdw 15Cm"]["heritability"]

    # Assert test passes regardless of which is higher
    # (documentation test only)
    assert diff is not None
```

### Manual Validation
1. **Visual comparison:** Side-by-side plots of Nov 30 vs reproduced PCA
2. **Variance components:** Check Vg and Ve match Nov 30
3. **Sample tracking:** Verify same samples included/excluded

## Success Metrics

### Quantitative
1. **Heritability reproduction:** |H²_reproduced - H²_nov30| < 0.05 for both biomass traits
2. **Sample count match:** n_reproduced = 57 (±1)
3. **Variance components:** |Vg_diff| < 10% AND |Ve_diff| < 10%

### Qualitative
1. **Root cause documented:** Clear explanation in design.md
2. **Reproduction method documented:** Step-by-step in tasks.md
3. **Design doc updated:** Correct claims in add-per-core-value-outlier-detection

## Rollback Plan

**If investigation fails to identify root cause:**
1. Document all attempted experiments in `investigation_log.md`
2. Escalate to user with findings
3. Pivot to empirical baseline (current H² = 0.27/0.45)
4. Update per-core QC feature to use realistic targets

**If reproduction fails:**
1. Document barriers in design.md
2. Accept that Nov 30 used non-reproducible manual processing
3. Create automated alternative that improves on baseline
4. Update success criteria to "H² improvement" not "H² = 0.75"

## Open Questions

1. **Q:** Did Nov 30 notebook use a different version of sleap-roots-analyze?
   **Investigation:** Check notebook imports, compare function signatures

2. **Q:** Were there any GH_7371-specific manual corrections?
   **Investigation:** Search notebook for "GH_7371", "7371", manual edits

3. **Q:** Did Nov 30 use different field data CSV?
   **Investigation:** Hash compare Field_2024_aboveground.csv

4. **Q:** Is the discrepancy due to cleanup order (before vs after merge)?
   **Investigation:** Check if Nov 30 cleaned root data before merging

5. **Q:** Could outlier removal explain differences?
   **Investigation:** Compare outlier detection results (Nov 30 vs baseline)

## Next Steps

After approval of this design:
1. Execute Phase 1 (code archaeology)
2. Execute Phase 2 (data forensics)
3. Execute Phase 3 (variance analysis)
4. Execute Phase 4 (reproduction experiments)
5. Update proposal.md with findings
6. Create tasks.md with concrete implementation steps
