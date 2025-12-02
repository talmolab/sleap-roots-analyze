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
   - Validate data structure and required columns (Plot, Rep, core ID)
   - Map genotype column name (e.g., `salk_geno` → `geno`) per source configuration
   - Create unique sample identifiers per core

2. **Step 0b: Transform to Long Format**
   - Use existing `melt_depth_data()` function
   - Parse depth from column names (biomass: manual mapping, counting: automatic)
   - Output: Long-format DataFrames with `Depth_cm` column

3. **Step 0c: Core-Level QC**
   - **IMPORTANT: Statistical outlier detection is NOT performed at core level** due to insufficient sample sizes
   - Root core datasets typically have ~3 cores per plot, but statistical methods (e.g., Mahalanobis distance) require 30+ samples for reliable detection
   - Core-level QC only flags cores with excessive missing data (>50% NaN depths)
   - Optional: Remove flagged cores before aggregation
   - **Recommended approach**: Disable core-level QC (`core_qc.enabled: false`) and use median aggregation for robustness to outliers

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

### Phase 2: Integration with Above-Ground Traits (NEW WORKFLOW)

**CRITICAL WORKFLOW CHANGE**: Merge happens BEFORE trait-level QC to enable outlier detection and heritability analysis on the full trait manifold.

6. **Step 11: Load and Validate Above-Ground Traits**
   - Load phenology, biomass, yield CSV
   - Validate join key columns match config exactly (case-sensitive)
   - Check for duplicate samples (one row per join key combination)

7. **Step 12: Merge All Trait Sources**
   - Merge root + above-ground on configurable join keys (e.g., ["Rep", "geno"])
   - Validate no duplicate column names (fail if conflicts found)
   - Output: Merged dataset with all traits (root + above-ground)

### Phase 3: Trait-Level QC on Merged Dataset (Existing Pipeline)

**Run QC on combined trait manifold** (root + above-ground traits together):

8. **Step 1: LoadData** - Uses merged data from Step 12
9. **Steps 2-10**: Run QC pipeline on full trait set
   - Cleanup traits (remove high-NaN traits from either source)
   - **Detect outliers using Mahalanobis distance on FULL manifold** (root + above-ground)
   - Remove outlier plots
   - **Calculate heritability on FULL trait set** (root + above-ground)
   - Filter low-heritability traits
   - Output: Final QC'd dataset with both root and above-ground traits

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

## Implementation Status

### Phase 1: Core Data Processing (Steps 0a-0e) - ✅ IMPLEMENTED
- Step 0a: LoadRootCoreData - Load biomass/counting from multiple sources
- Step 0b: TransformDepthData - Convert to long format with depth_cm
- Step 0c: QCCoreLevel - **Missing data detection only** (statistical outlier detection removed due to insufficient sample sizes)
- Step 0d: AggregateCores - Aggregate 3 cores → replicate level (median recommended)
- Step 0e: ReshapeForTraitQC - Pivot to wide format with prefixes

**Output**: `00e_root_core_merged.csv` containing root core traits only (biomass + counting)

**IMPORTANT CHANGE**: Core-level QC no longer performs statistical outlier detection (e.g., Mahalanobis distance) because root core datasets have insufficient samples (~3 cores per plot, need 30+ for reliability). Instead, use median aggregation for robustness to outliers and let trait-level QC (Step 5) detect outliers on plot-level data (60+ samples)

### Phase 2: Above-Ground Integration (Steps 11-12) - ✅ IMPLEMENTED
**IMPORTANT**: These steps now run BEFORE trait-level QC to enable analysis on full manifold

### Phase 3: Trait-Level QC on Merged Data (Steps 1-10) - ✅ IMPLEMENTED
**WORKFLOW CHANGE NEEDED**: Pipeline needs reordering to run Steps 11-12 before Steps 1-10
**Status**: Core merge functionality implemented, visualization steps not yet added.

Implemented steps:
- Step 11: LoadAboveGroundTraitsStep - Load and validate above-ground CSV
- Step 12: MergeAllTraitsStep - Merge root + above-ground on configurable join keys with duplicate handling and metadata generation

**Output**: `final_merged_traits.csv` (or custom path via `merge_traits.output_path`)

**CRITICAL**: Join Key Requirements
- Root core output (Step 0e) has columns: `Plot`, `Rep`, `geno` (one row per plot after core aggregation)
- Above-ground CSV **must** have matching columns for the configured `join_keys`
- Common scenarios:
  - **Plot-level design**: `join_keys: ["Plot", "Rep", "geno"]` - Requires Plot column in both datasets. Use when above-ground data is also collected per plot.
  - **Replicate-only design**: `join_keys: ["Rep", "geno"]` - When above-ground lacks Plot column (e.g., one measurement per rep-genotype). This is the most common scenario when root cores are plot-level but above-ground is replicate-level.
  - **Genotype-only design**: `join_keys: ["geno"]` - Matches all reps/plots per genotype (creates many-to-many merge)
- **Key insight**: Both datasets must have exactly **one row per join key combination** to avoid duplicate rows. If root core data has multiple plots per Rep-Geno (after Step 0d aggregation), you cannot use `["Rep", "geno"]` join keys alone.

**CRITICAL: Column Name Validation to Prevent Metadata Contamination**

**Metadata columns MUST be excluded from statistical analyses**. The pipeline uses case-sensitive column matching, so config values must EXACTLY match data column names:

**Common configuration errors:**
- Config: `replicate: "rep"` but data has `"Rep"` → Rep column NOT excluded! ❌
- Config: `barcode: "barcode"` but data has `"Barcode"` → Barcode column NOT excluded! ❌
- Result: Metadata contaminate PCA, outlier detection, heritability calculations

**Validation checklist before running pipeline:**
1. Check actual data columns: `pd.read_csv("your_data.csv").columns.tolist()`
2. Update config to match EXACTLY (case-sensitive):
   ```yaml
   columns:
     barcode: "Barcode"  # Match case from data
     genotype: "geno"     # Match case from data  
     replicate: "Rep"     # Match case from data (NOT "rep")
   ```
3. **NEVER rely on function defaults** - always specify column names in config
4. Root core pipeline uses: `Plot`, `Rep`, `geno`, `Barcode` (mixed case)

**Consequences of misconfiguration:**
- Plot/Rep/geno values included in PCA → contaminated outlier detection ❌
- Heritability calculated FOR metadata columns (nonsensical) ❌
- Statistical tests use replicate IDs as trait values (invalid) ❌

**CRITICAL: Why Median Aggregation is Recommended Over Core-Level Outlier Detection**

Root core experiments present a unique challenge: each plot has only ~3 cores, but statistical outlier detection methods require 30+ samples for reliability. This fundamental limitation means:

1. **Statistical methods don't work at core level**: 
   - Mahalanobis distance with PCA requires large samples for stable covariance estimation
   - Chi-squared threshold calculation assumes asymptotic distribution (violated with n=3)
   - With only 2-12 depth measurements per core, PCA compression loses outlier signal

2. **Median aggregation solves the problem without detection**:
   - Median is inherently robust to outliers (breaks down only when >50% of values are outliers)
   - Handles typos, miscounts, and measurement errors automatically
   - Works with any sample size (including n=3)
   - Simpler, faster, and more reliable than statistical detection with insufficient samples

3. **Trait-level QC still detects outliers**:
   - After aggregation, you have 60+ plots (sufficient for statistical methods)
   - Step 5 (DetectOutliers) uses Mahalanobis distance on plot-level data
   - This catches biological outliers (e.g., genotypes with extreme root phenotypes)

**RECOMMENDATION**: Use `aggregation_method: "median"` and disable core-level QC (`core_qc.enabled: false`).

**When to use MEAN (use with caution)**:
- Data has very low within-plot variance (CV < 0.3)
- Manual inspection confirms no outlier cores
- Differences between mean and median are negligible (< 5% of typical values)
- You need maximum precision for subtle genetic effects

**When to use MEDIAN (recommended default)**:
- Any dataset where you haven't manually inspected all cores
- Data with potential typos, miscounts, or measurement errors
- High within-plot variance (CV > 0.3)
- You want robust analysis without manual data curation

**Example analysis** (EDPIE root biomass data):
- Within-plot CV: 0.3-0.4 (moderate)
- Plots with high skewness: 55% have abs(skew) > 1
- IQR outliers: 0 plots
- Mean vs median difference: < 0.2g (< 10% of values)
- **Recommendation**: MEAN is appropriate - differences are minimal, no outliers detected

To analyze your own data before choosing:
```python
# Check within-plot variance and outliers
plot_stats = df.groupby('Plot')['depth_column'].agg(['mean', 'median', 'std'])
cv = plot_stats['std'] / plot_stats['mean']
print(f"Mean CV: {cv.mean():.3f}")  # < 0.3 = low, 0.3-0.5 = moderate, > 0.5 = high
```

Future enhancements (not yet implemented):
- Step 13/14: Post-merge visualizations (cross-correlations, PCA with combined traits)

**Impact**: Users can now:
- ✅ Process root core data through full QC pipeline
- ✅ Get clean root trait outputs with prefixes (RootDW_15cm, RootCount_5cm)
- ✅ Automatically merge with above-ground phenotypes
- ✅ Handle duplicate columns with configurable strategies (fail, skip, suffix)
- ✅ Generate merge metadata JSON with statistics

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
    genotype_column: "salk_geno"  # Map to standard 'geno' column

  counting:
    csv_path: "root_counting_cimmyt_edited.csv"
    depth_column_prefix: "RootCount"
    aggregation_method: "mean"
    genotype_column: "geno"  # Already uses standard name

  core_qc:
    enabled: true
    outlier_method: "mahalanobis"
    contamination: 0.1

merge_traits:
  above_ground_csv: "above_ground_phenotypes.csv"
  join_keys: ["Rep", "geno"]  # IMPORTANT: Must match columns in above-ground CSV
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

### Phase 1 & 2 (Steps 0a-10) - Current Implementation
1. ✅ Pipeline processes both biomass and counting data from raw CSVs
2. ✅ Core-level outliers identified and optionally removed
3. ✅ Aggregated data matches expected values (using mean/median)
4. ✅ Root core traits merged with prefixes (no duplicate columns between biomass/counting)
5. ✅ Metadata JSON generated for each step
6. ✅ All depth profile visualizations generated automatically
7. ✅ Integration tests pass with real EDPIE data
8. ✅ Documentation explains configuration options

### Phase 3 (Steps 11-12) - Implemented
9. ✅ Above-ground traits merged with root traits
10. ✅ Duplicate column handling with configurable strategies (fail, skip, suffix)
11. ✅ Merge metadata JSON with statistics generated
12. ✅ Integration test with 17-step pipeline passes
13. ✅ 17 unit tests for Steps 11-12 (all pass)

### Phase 3 Future Enhancements (Steps 13-14)
14. ⏳ Cross-correlation visualizations (root vs above-ground)
15. ⏳ PCA with combined trait set

## References

- Existing root_core_analysis module: `src/sleap_roots_analyze/root_core_analysis.py`
- Existing depth profile plots: `src/sleap_roots_analyze/depth_profile_plots.py`
- QC Pipeline: `src/sleap_roots_analyze/pipeline/pipelines/qc_pipeline.py`
- Example notebook: `20250512_charlotte_rambla_edpie/field_biomass_and_root_core_analysis_20240729/first_step_root_counting_20240729.v006.ipynb`