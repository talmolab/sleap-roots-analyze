# Tasks: Investigate Nov 30 Heritability Discrepancy

## Phase 1: Code Archaeology (Investigation)

### Task 1.1: Extract Nov 30 Notebook Processing Steps
- [ ] Parse `trait_qc_root_coring_20251126.ipynb` to extract code cells
- [ ] Identify root biomass loading cell (look for "rearranged_root_biomass_dw.csv")
- [ ] Document aggregation method used (mean vs median)
- [ ] Check for core exclusion logic (search for "core" filtering)
- [ ] Extract cleanup threshold values (MAX_NAN_FRACTION, MAX_ZEROS, etc.)
- [ ] Search for manual data edits or corrections
- [ ] **Output:** Create `nov30_processing_steps.md` with findings
- [ ] **Validation:** Document includes exact Python code snippets from notebook

### Task 1.2: Compare Nov 30 Configuration to Current Baseline
- [ ] Extract all config parameters from Nov 30 notebook cells
- [ ] Compare to `configs/qc_root_core_edpie.yaml` (current baseline config)
- [ ] Create diff table showing parameter-by-parameter comparison
- [ ] Highlight parameters that differ significantly
- [ ] **Output:** Add comparison table to `nov30_processing_steps.md`
- [ ] **Validation:** Table includes at minimum: aggregation_method, max_nan_fraction, max_zeros_per_trait

### Task 1.3: Analyze GH_7371 Case Study
- [ ] Search notebook for "GH_7371" or "7371" mentions
- [ ] Check if core 2 was manually excluded for this genotype
- [ ] Verify claim: "cores [0.76g, 0.71g, 0.31g], core 2 flagged"
- [ ] Document whether exclusion was manual or automated
- [ ] **Output:** Add GH_7371 findings section to `nov30_processing_steps.md`
- [ ] **Validation:** Findings state explicitly whether manual exclusion was used

## Phase 2: Data Forensics (Validation)

### Task 2.1: Hash Compare Source CSVs
- [ ] Compute SHA256 hash of `rearranged_root_biomass_dw.csv` (current)
- [ ] Check git history for this file (any changes since Nov 30?)
- [ ] Compute hash of `Field_2024_aboveground.csv` (current)
- [ ] If possible, access Nov 30 versions and compare hashes
- [ ] **Output:** Create `data_provenance_report.md` with hash comparison
- [ ] **Validation:** Report confirms CSVs are identical OR documents differences

### Task 2.2: Compare Raw Core-Level Data
- [ ] Load Nov 30 intermediate CSV: `c:/repos/runs/run_20251130_193257/00d_root_core_biomass_aggregated.csv` (if exists)
- [ ] Load baseline intermediate CSV: `qc_runs/.../00d_root_core_biomass_aggregated.csv`
- [ ] Compare aggregated values for GH_7371 Rep 1 specifically
- [ ] Compare aggregated values for all plots
- [ ] Identify plots where aggregated values differ
- [ ] **Output:** Add comparison table to `data_provenance_report.md`
- [ ] **Validation:** Table shows per-plot aggregated biomass comparison

### Task 2.3: Verify Cleanup Step Consistency
- [ ] Load Nov 30 cleanup log: `c:/repos/runs/run_20251130_193257/02_trait_cleanup_log.json`
- [ ] Load baseline cleanup log: `qc_runs/.../02_cleanup_log.csv`
- [ ] Compare traits removed in each
- [ ] Compare samples removed in each
- [ ] Document any NaN/zero handling differences
- [ ] **Output:** Add cleanup comparison to `data_provenance_report.md`
- [ ] **Validation:** Report identifies which traits/samples differ between runs

## Phase 3: Variance Component Analysis (Deep Dive)

### Task 3.1: Extract Variance Components from Nov 30
- [ ] Load `c:/repos/runs/run_20251130_193257/analysis_outputs/heritability_results.json`
- [ ] Extract Vg (var_genetic) and Ve (var_residual) for Rootdw 15Cm and 45Cm
- [ ] Extract for all above-ground traits (for comparison)
- [ ] Calculate variance ratios and percent variance between genotypes
- [ ] **Output:** Create `variance_components_comparison.csv`
- [ ] **Validation:** CSV includes Vg, Ve, H², pct_var_between for all traits

### Task 3.2: Compare Variance Components
- [ ] Load baseline variance from `09_heritability_diagnostics.csv`
- [ ] Calculate Vg difference (baseline - Nov 30) for each trait
- [ ] Calculate Ve difference (baseline - Nov 30) for each trait
- [ ] Identify traits where Vg decreased (genetic variance lost)
- [ ] Identify traits where Ve increased (residual variance inflated)
- [ ] **Output:** Add analysis section to `variance_components_comparison.csv`
- [ ] **Validation:** Analysis identifies primary driver of H² loss (Vg decrease vs Ve increase)

### Task 3.3: Replicate Structure Analysis
- [ ] Compare mean_reps_per_geno: Nov 30 = 2.85, Baseline = 2.9
- [ ] Check if different samples were excluded at trait-level cleanup
- [ ] Verify replicate imbalance matches (both have range 2-3 reps)
- [ ] Check if unbalanced design could explain H² differences
- [ ] **Output:** Add replicate analysis to `variance_components_comparison.csv`
- [ ] **Validation:** Explains +0.05 difference in mean_reps_per_geno

## Phase 4: Reproduction Experiments (Hypothesis Testing)

### Task 4.1: Test Aggregation Method Hypothesis
- [ ] Create `configs/experiments/test_mean_aggregation.yaml`
- [ ] Set `aggregation_method: "mean"` for biomass source
- [ ] Run pipeline: `uv run sleap-roots-analyze qc configs/experiments/test_mean_aggregation.yaml`
- [ ] Extract Rootdw 15Cm and 45Cm heritability from output
- [ ] Compare to Nov 30 (H² = 0.75/0.73) and baseline (H² = 0.27/0.45)
- [ ] **Output:** Add result to `reproduction_experiments.md`
- [ ] **Validation:** Result shows whether mean aggregation closes the gap

### Task 4.2: Test Core Exclusion Hypothesis (Cores 0&1 Only)
- [ ] Modify `src/sleap_roots_analyze/pipeline/steps/aggregate_cores.py` (TEMPORARY)
- [ ] Add filter: `df = df[df["Core_Replicate"] != 2]` before aggregation
- [ ] Run pipeline with this modification
- [ ] Extract heritability results
- [ ] Compare to Nov 30 target
- [ ] Revert code changes (this is temporary test only)
- [ ] **Output:** Add result to `reproduction_experiments.md`
- [ ] **Validation:** Result shows whether excluding core 2 closes the gap

### Task 4.3: Test Cleanup Threshold Hypothesis
- [ ] Create `configs/experiments/test_nov30_cleanup.yaml`
- [ ] Set cleanup thresholds to match Nov 30 (from Task 1.2)
- [ ] Run pipeline
- [ ] Extract heritability and sample count
- [ ] Check if n=57 (matches Nov 30)
- [ ] Compare heritability to Nov 30
- [ ] **Output:** Add result to `reproduction_experiments.md`
- [ ] **Validation:** Result shows impact of cleanup threshold differences

### Task 4.4: Test Combined Nov 30 Processing
- [ ] Create `configs/qc_root_core_nov30_reproduction.yaml`
- [ ] Apply ALL processing differences identified in Phase 1
- [ ] Run pipeline
- [ ] Extract full heritability results
- [ ] Calculate |H²_reproduced - H²_nov30| for both biomass traits
- [ ] **Target:** Achieve H² ≥ 0.70 for both traits
- [ ] **Output:** Add final result to `reproduction_experiments.md`
- [ ] **Validation:** |ΔH²| < 0.05 for biomass traits, n=57 samples

## Phase 5: Documentation & Validation (Finalization)

### Task 5.1: Document Root Cause
- [ ] Review all findings from Phases 1-4
- [ ] Write executive summary in `investigation_summary.md`
- [ ] Identify THE primary cause of discrepancy (single most important factor)
- [ ] Explain mechanism: Why does this processing difference change H² so drastically?
- [ ] **Output:** Create `investigation_summary.md`
- [ ] **Validation:** Summary is < 500 words, identifies specific root cause

### Task 5.2: Update Per-Core QC Design Document
- [ ] Open `openspec/changes/add-per-core-value-outlier-detection/design.md`
- [ ] Correct any inaccurate claims about Nov 30 processing
- [ ] Update GH_7371 case study with verified facts
- [ ] Add reference to this investigation
- [ ] Update baseline heritability values if new baseline established
- [ ] **Output:** Modified design.md with corrections
- [ ] **Validation:** All claims about Nov 30 are factually accurate

### Task 5.3: Create Reference Configuration
- [ ] Finalize `configs/qc_root_core_nov30_reproduction.yaml` (from Task 4.4)
- [ ] Add comprehensive comments explaining each parameter choice
- [ ] Document differences from baseline config
- [ ] Add warning: "This config reproduces Nov 30 results. For new analyses, use..."
- [ ] **Output:** Finalized reference config
- [ ] **Validation:** Config achieves H² ≥ 0.70 when run

### Task 5.4: Create Reproduction Test
- [ ] Create `tests/test_nov30_reproduction.py`
- [ ] Add test that loads nov30_reproduction config
- [ ] Add test that runs pipeline
- [ ] Add assertions: H² ≥ 0.70, n=57
- [ ] Add pytest markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- [ ] **Output:** New test file
- [ ] **Validation:** Test passes when run with `uv run pytest tests/test_nov30_reproduction.py`

### Task 5.5: Update Analysis Document
- [ ] Update `HERITABILITY_DISCREPANCY_ANALYSIS.md` with final findings
- [ ] Add "Resolution" section with root cause
- [ ] Add "Reproduction Method" section with config details
- [ ] Update "Recommendations" based on findings
- [ ] **Output:** Updated analysis document
- [ ] **Validation:** Document includes concrete reproduction steps

## Phase 6: Cleanup & Archive (Optional)

### Task 6.1: Clean Up Experiment Configs
- [ ] Move experiment configs to `configs/experiments/archive/nov30_investigation/`
- [ ] Add README explaining each experiment
- [ ] Remove temporary code modifications (if any)
- [ ] **Output:** Organized experiment archive
- [ ] **Validation:** No temporary files in main configs/ directory

### Task 6.2: Update OpenSpec Change Status
- [ ] Mark investigation as "COMPLETED" in proposal.md
- [ ] Add summary of findings to proposal.md
- [ ] Update `openspec/changes/add-per-core-value-outlier-detection/` with references
- [ ] Run `openspec validate investigate-nov30-heritability-discrepancy`
- [ ] **Output:** Updated proposal with completion status
- [ ] **Validation:** OpenSpec validation passes

## Success Criteria Checklist

### Must Have (Phase 1-4)
- [ ] Root cause identified and documented
- [ ] Nov 30 results reproduced (H² ≥ 0.70 for both biomass traits)
- [ ] Reproduction config created and tested

### Should Have (Phase 5)
- [ ] Data provenance documented (CSV sources, modification dates)
- [ ] Per-core QC design document corrected
- [ ] Analysis document updated with findings

### Nice to Have (Phase 6)
- [ ] Automated reproduction test created
- [ ] Experiment archive organized
- [ ] OpenSpec change marked complete

## Notes

- **Estimated Time:** 3-5 days total
- **Priority:** HIGH (blocks optimization of per-core QC feature)
- **Dependencies:** Access to Nov 30 notebook, source CSVs
- **Risks:** Nov 30 may have used manual, non-reproducible steps
