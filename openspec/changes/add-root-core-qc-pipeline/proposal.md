# Proposal: Root Core Data QC Pipeline Extension

## Summary

Extend the existing QC Pipeline to handle root core experimental data (biomass and counting) with proper data transformation, core-level QC, aggregation to biological replicates, and integration with above-ground traits without trait duplication.

## Problem Statement

The current workflow for processing root core data is:
1. **Untracked and non-reproducible**: Collaborator manually aggregated 3 cores per biological replicate using unknown methods (mean? median? how were NaNs handled?)
2. **No QC before aggregation**: Outlier cores may skew biological replicate means
3. **No documentation**: The final combined CSV lacks metadata explaining processing steps
4. **Duplicate trait risk**: When merging root and above-ground traits, column name conflicts could cause data loss or errors
5. **Manual and error-prone**: Depth data transformations done ad-hoc in notebooks

## Proposed Solution

### Phase 1: Core Data Processing (New Pipeline Steps)

Add a new pre-processing phase to QC Pipeline that handles root core data:

1. **Step 0a: Load Raw Root Core Data**
   - Load biomass CSV (2 depths, 3 cores/rep)
   - Load counting CSV (12 depths, 3 cores/rep)
   - Validate data structure and required columns
   - Create unique sample identifiers per core

2. **Step 0b: Transform to Long Format**
   - Use existing `melt_depth_data()` function
   - Parse depth from column names (biomass: manual mapping, counting: automatic)
   - Output: Long-format DataFrames with `Depth_cm` column

3. **Step 0c: Core-Level QC**
   - Detect outlier cores within each biological replicate
   - Flag cores with excessive missing data
   - Generate visualizations of individual core profiles
   - Optional: Remove flagged cores before aggregation

4. **Step 0d: Aggregate to Biological Replicates**
   - Use existing `aggregate_by_replicate()` function
   - Configurable aggregation method (mean/median)
   - Explicit NaN handling strategy
   - Output: One row per Plot-Rep-Depth

5. **Step 0e: Generate Pre-QC Depth Visualizations**
   - Use existing `plot_depth_profile_faceted()` and `plot_depth_profile_replicates()`
   - Bar plots for biomass (2 depths)
   - Line plots for counting (12 depths)

6. **Step 0f: Reshape for Trait-Level QC**
   - Pivot long → wide format
   - Treat each depth as a separate trait
   - Add prefix to column names: `RootDW_15cm`, `RootCount_5cm`, etc.
   - This prevents column name conflicts when merging later

### Phase 2: Trait-Level QC (Existing Pipeline)

Connect to existing QC pipeline steps:

7. **Step 1: LoadData** - Load the wide-format root data from Step 0f
8. **Steps 2-10**: Run existing QC pipeline as normal
   - Cleanup traits (remove high-NaN depths)
   - Detect and remove outlier replicates
   - Calculate heritability by depth
   - Filter low-heritability depths

### Phase 3: Integration with Above-Ground Traits (New Steps)

Add post-QC merge functionality:

9. **Step 11: Load and Validate Above-Ground Traits**
   - Load phenology, biomass, yield CSV
   - Validate Plot-Rep combinations match root data
   - Check for column name conflicts

10. **Step 12: Merge All Trait Sources**
    - Merge on Plot-Rep-geno keys
    - Validate no duplicate columns (fail if conflicts found)
    - Preserve all metadata columns
    - Output: Final wide-format CSV with all traits

11. **Step 13: Generate Processing Metadata**
    - JSON file with complete provenance:
      - Input file paths and timestamps
      - Processing parameters (agg method, QC thresholds)
      - Outliers removed (sample IDs)
      - Traits filtered (column names and reasons)
      - Final sample size
      - Software versions

12. **Step 14: Post-QC Visualizations**
    - Depth profiles with clean data
    - Cross-correlations (root vs above-ground)
    - Heritability by depth bar plots
    - PCA with all traits

## Key Design Decisions

### 1. Prefixing Strategy to Prevent Duplicates

**Problem**: Both root biomass and above-ground data might have columns like `BM_Calc_gm2` (biomass)

**Solution**: Add descriptive prefixes during wide-format conversion:
- Root biomass depths: `RootDW_15cm`, `RootDW_45cm`
- Root counting depths: `RootCount_0cm`, `RootCount_5cm`, ..., `RootCount_55cm`
- Above-ground traits: Keep original names (or prefix with `AG_` if needed)

**Implementation**: Add `column_prefix` parameter to reshape function

### 2. Two-Level QC Approach

**Core-level QC (before aggregation)**:
- Purpose: Remove bad individual cores
- Method: Mahalanobis distance within Plot-Rep groups
- Rationale: One bad core shouldn't skew the biological replicate mean

**Replicate-level QC (after aggregation)**:
- Purpose: Remove outlier Plot-Reps across all genotypes
- Method: Existing QC pipeline (PCA-based Mahalanobis)
- Rationale: Some plots may be damaged/diseased

### 3. Configuration-Driven Processing

All processing decisions configurable via YAML:
```yaml
root_core_processing:
  biomass:
    csv_path: "rearranged_root_biomass_dw.csv"
    depth_column_prefix: "RootDW"
    aggregation_method: "mean"

  counting:
    csv_path: "root_counting_cimmyt_edited.csv"
    depth_column_prefix: "RootCount"
    aggregation_method: "mean"

  core_qc:
    enabled: true
    outlier_method: "mahalanobis"
    contamination: 0.1

merge_traits:
  above_ground_csv: "above_ground_phenotypes.csv"
  join_keys: ["Plot", "Rep", "geno"]
  validate_no_duplicates: true
  output_path: "Field_2024_final.csv"
```

### 4. Metadata Generation

Every output file paired with metadata JSON:
- `Field_2024_final.csv` → `Field_2024_final_metadata.json`
- Includes: data sources, processing steps, QC decisions, sample sizes
- Enables reproducibility and troubleshooting

## Impact Assessment

### Benefits
1. **Reproducibility**: Every step documented and configurable
2. **Data quality**: Two-level QC catches both core and replicate outliers
3. **Prevents errors**: Automated duplicate detection prevents data loss
4. **Transparency**: Metadata files explain every decision
5. **Reusability**: Works for any root core experiment with similar structure

### Risks
- **Complexity**: Adds 6 new pipeline steps
- **Validation needed**: Must test that aggregated data matches collaborator's results
- **Breaking changes**: None (purely additive)

### Compatibility
- **Backward compatible**: Existing QC pipeline unchanged
- **Opt-in**: Root core processing only runs if configured
- **Modular**: New steps can be used standalone or integrated

## Implementation Plan

See `tasks.md` for detailed implementation checklist.

**Estimated effort**:
- Core pipeline steps (0a-0f): ~2-3 days
- Integration steps (11-14): ~1-2 days
- Testing and documentation: ~1 day
- **Total**: 4-6 days

## Success Criteria

1. ✅ Pipeline processes both biomass and counting data from raw CSVs
2. ✅ Core-level outliers identified and optionally removed
3. ✅ Aggregated data matches collaborator's values (when using same method)
4. ✅ Final merged CSV has no duplicate columns
5. ✅ Metadata JSON generated with complete provenance
6. ✅ All depth profile visualizations generated automatically
7. ✅ Integration tests pass with real EDPIE data
8. ✅ Documentation explains every configuration option

## References

- Existing root_core_analysis module: `src/sleap_roots_analyze/root_core_analysis.py`
- Existing depth profile plots: `src/sleap_roots_analyze/depth_profile_plots.py`
- QC Pipeline: `src/sleap_roots_analyze/pipeline/pipelines/qc_pipeline.py`
- Example notebook: `20250512_charlotte_rambla_edpie/field_biomass_and_root_core_analysis_20240729/first_step_root_counting_20240729.v006.ipynb`