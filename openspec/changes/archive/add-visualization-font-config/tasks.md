# Implementation Tasks

## 1. Update VisualizationConfig Dataclass

**File**: `src/sleap_roots_analyze/pipeline/config/components.py`

**Changes**:
- Add font size fields: `title_fontsize`, `label_fontsize`, `tick_fontsize`, `legend_fontsize`
- Add format fields: `figure_format`, `figure_formats`
- Add savefig parameter fields: `bbox_inches`, `facecolor`, `edgecolor`, `transparent`
- Update docstring with new parameter descriptions

**Estimate**: 1 hour

## 2. Update Pipeline Steps

### 2.1 ExploratoryAnalysisStep

**File**: `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py`

**Changes**:
- Pass font config to all visualization functions
- Update savefig call to use `config.visualization.figure_format` and `config.visualization.bbox_inches`
- Update matplotlib calls to use configured font sizes

**Estimate**: 1 hour

### 2.2 VisualizeOutliersStep

**File**: `src/sleap_roots_analyze/pipeline/steps/visualize_outliers.py`

**Changes**:
- Pass font config to all outlier visualization functions (~9 savefig calls)
- Update format and bbox_inches parameters
- Ensure transparent/facecolor/edgecolor are applied

**Estimate**: 1 hour

### 2.3 FilterHeritabilityStep

**File**: `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py`

**Changes**:
- Update hardcoded `dpi=300` to use `config.visualization.dpi` (line 173)
- Apply format and bbox_inches configuration
- Pass font config to `create_heritability_plot`

**Estimate**: 0.5 hours

### 2.4 GenerateStaticFiguresStep

**File**: `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py`

**Changes**:
- Update `_save_figure` method to use `config.visualization.bbox_inches`, `transparent`, `facecolor`, `edgecolor`
- Ensure all visualization functions receive font config parameters
- Update format handling to use `config.static_viz.figure_formats`

**Estimate**: 0.5 hours

## 3. Update Visualization Modules

### 3.1 visualization.py

**File**: `src/sleap_roots_analyze/visualization.py`

**Changes**:
- Add optional font size parameters to all plotting functions with defaults
- Replace ~50 hardcoded fontsize values with parameter usage
- Key functions to update:
  - `create_trait_histograms_batched` (lines 86, 212)
  - `create_trait_boxplots_by_genotype_batched` (line 256)
  - `create_heritability_plot` (lines 812, 813, 836, 837)
  - `create_pca_biplot` (lines 1777, 1768)
  - `create_feature_contribution_heatmap` (lines 1292, 1293, 1296, 1304)
  - `create_umap_facet_plot` (lines 1905, 1906, 1918, 1934)
  - `create_umap_single_trait` (lines 2031, 2033, 2034, 2037, 2039, 2078, 2079, 2080)
  - `create_pc_genotype_boxplots` (lines 2186, 2199, 2300, 2320)
  - `visualize_genotype_root_images` (lines 2815, 2816, 2838, 2839, 2967, 3021)
  - `plot_trait_regression` (lines 3243, 3244, 3247, 3249)
- Update savefig call (line 337) to accept format/bbox_inches parameters

**Estimate**: 3 hours

### 3.2 outlier_visualization.py

**File**: `src/sleap_roots_analyze/outlier_visualization.py`

**Changes**:
- Add optional font size parameters to all functions
- Replace ~30 hardcoded fontsize values
- Key functions to update:
  - `create_outlier_overlap_venn` (lines 117, 119)
  - `plot_outliers_by_genotype` (lines 242, 288)
  - `create_mahalanobis_outlier_plot` (lines 355, 435, 492, 532, 543, 550, 555, 589, 617, 644, 647, 746, 797, 819, 1006, 1024, 1051, 1053, 1056)
  - `create_pca_outlier_plot` (line 1070, 1082, 1260)

**Estimate**: 2 hours

### 3.3 depth_profile_plots.py

**File**: `src/sleap_roots_analyze/depth_profile_plots.py`

**Changes**:
- Add optional savefig parameters to both functions
- Update savefig calls (lines 142, 229) to use parameters instead of hardcoded values
- Add optional font size parameters if needed

**Estimate**: 0.5 hours

## 4. Update Tests

### 4.1 Test Config Changes

**File**: `tests/pipeline/config/test_components.py`

**Changes**:
- Add tests for new VisualizationConfig fields
- Test default values match old hardcoded values
- Test validation of font sizes (must be positive)
- Test validation of figure_format (must be valid format)
- Test validation of bbox_inches (must be "tight" or None)

**Estimate**: 1 hour

### 4.2 Update Pipeline Step Tests

**Files**: 
- `tests/pipeline/steps/test_qc_step_exploratory_analysis.py`
- `tests/pipeline/steps/test_qc_step_visualize_outliers.py`
- `tests/pipeline/steps/test_qc_step_filter_heritability.py`
- `tests/pipeline/steps/test_viz_step_generate_static_figures.py`

**Changes**:
- Update test configs to include new parameters
- Verify figures use configured values (check file format, etc.)
- Add tests for non-default configurations

**Estimate**: 1.5 hours

### 4.3 Update Visualization Tests

**Files**:
- `tests/test_visualization.py`
- `tests/test_outlier_visualization.py`

**Changes**:
- Add parametric tests for font size configuration
- Test backward compatibility (functions work without font params)
- Test that configured values are actually applied

**Estimate**: 1 hour

## 5. Update Documentation

### 5.1 Update Config Examples

**Files**:
- `configs/qc_turface_150genotypes.yaml` - Add font/format parameters
- Update any other example configs in `configs/`

**Estimate**: 0.5 hours

### 5.2 Update README/Docs

**Files**:
- `TURFACE_QC_README.md` - Document new parameters
- Add docstring examples showing font configuration

**Estimate**: 0.5 hours

## Total Effort Estimate

| Category | Hours |
|----------|-------|
| Config changes | 1.0 |
| Pipeline steps | 3.0 |
| Visualization modules | 5.5 |
| Testing | 3.5 |
| Documentation | 1.0 |
| **Total** | **14.0** |

## Implementation Order

1. **Config changes** (task 1) - Foundation for all other changes
2. **Visualization modules** (task 3) - Update functions to accept parameters
3. **Pipeline steps** (task 2) - Wire config through to visualization functions
4. **Tests** (task 4) - Verify everything works
5. **Documentation** (task 5) - Update examples and docs

## Validation Checklist

- [ ] All new config fields have defaults matching old hardcoded values
- [ ] Existing tests pass without modification (backward compatible)
- [ ] New tests verify configured values are applied
- [ ] Turface QC config runs successfully with new parameters
- [ ] Can generate figures in PDF, SVG, and PNG formats
- [ ] Font sizes are consistent across all plot types
- [ ] Documentation includes examples of new parameters
- [ ] `openspec validate` passes
