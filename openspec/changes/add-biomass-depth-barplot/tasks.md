# Implementation Tasks (TDD Approach)

## 1. Test Setup
- [ ] 1.1 Create `tests/test_depth_profile_plots.py` test file
- [ ] 1.2 Add biomass fixture data with 2 depth intervals (15cm, 45cm) and 3 genotypes
- [ ] 1.3 Add biomass fixture data with variable replicates per genotype

## 2. Test: Basic Barplot Generation (Red → Green → Refactor)
- [ ] 2.1 **RED**: Write failing test `test_plot_biomass_depth_barplot_basic()`
  - Assert function exists and accepts required parameters
  - Assert returns matplotlib Figure object
  - Assert figure has correct number of axes (1 for non-faceted barplot)
- [ ] 2.2 **GREEN**: Implement `plot_biomass_depth_barplot()` in `depth_profile_plots.py`
  - Minimal implementation to pass test
  - Signature: `plot_biomass_depth_barplot(df, depth_col, value_col, genotype_col, output_path=None)`
  - Return empty figure initially
- [ ] 2.3 **REFACTOR**: Clean up implementation, add docstring

## 3. Test: Depth Interval Detection (Red → Green → Refactor)
- [ ] 3.1 **RED**: Write test `test_depth_interval_detection()`
  - Given df with Depth_cm=[15, 45], assert depth labels are ['0-30cm', '30-60cm']
  - Given df with Depth_cm=[15, 30, 45], assert 3 depth labels detected
- [ ] 3.2 **GREEN**: Implement depth interval detection logic
  - Extract unique depth values
  - Generate descriptive labels based on depth ranges
- [ ] 3.3 **REFACTOR**: Extract to helper function `_generate_depth_labels()`

## 4. Test: Genotype Ordering (Red → Green → Refactor)
- [ ] 4.1 **RED**: Write test `test_genotype_ordering_control_first()`
  - Given genotypes=['GH_001', 'Control', 'GH_002'] with varying biomass
  - Assert x-axis order places Control first, then sorted by mean biomass at shallowest depth
- [ ] 4.2 **GREEN**: Implement genotype ordering
  - Check for 'Control' genotype
  - Calculate mean biomass at shallowest depth per genotype
  - Sort ascending, with Control prepended
- [ ] 4.3 **REFACTOR**: Extract to helper function `_order_genotypes_by_shallow_biomass()`

## 5. Test: Barplot Visual Elements (Red → Green → Refactor)
- [ ] 5.1 **RED**: Write test `test_barplot_has_correct_elements()`
  - Assert barplot has grouped bars (hue='Depth')
  - Assert error bars are present (errorbar='se')
  - Assert x-axis labels are rotated 90°
  - Assert grid is enabled
- [ ] 5.2 **GREEN**: Implement barplot generation with seaborn
  - Use `sns.barplot()` with dodge=True, errorbar='se'
  - Set x-tick rotation to 90°
  - Enable grid with `ax.grid(True)`
- [ ] 5.3 **REFACTOR**: Apply consistent styling theme (darkgrid, white facecolor)

## 6. Test: Stripplot Overlay (Red → Green → Refactor)
- [ ] 6.1 **RED**: Write test `test_stripplot_overlay_optional()`
  - Assert stripplot overlay is added when `include_points=True`
  - Assert no stripplot when `include_points=False` (default)
- [ ] 6.2 **GREEN**: Add optional stripplot overlay
  - Add `include_points` parameter (default False)
  - Use `sns.stripplot()` with dodge=True, alpha=0.5, jitter=True when enabled
- [ ] 6.3 **REFACTOR**: Clean up parameter handling

## 7. Test: File Saving (Red → Green → Refactor)
- [ ] 7.1 **RED**: Write test `test_barplot_saves_to_file()`
  - Assert file is created at `output_path` when specified
  - Assert file has correct PNG format
  - Assert file has DPI=300
- [ ] 7.2 **GREEN**: Implement file saving
  - Check if `output_path` is provided
  - Save using `fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')`
- [ ] 7.3 **REFACTOR**: Extract DPI and facecolor as function parameters with defaults

## 8. Integration: Step 00f Pipeline Integration
- [ ] 8.1 **RED**: Write test `test_visualize_depth_profiles_step_generates_barplot_for_biomass()`
  - Mock biomass data source
  - Assert Step 00f generates 3 files for biomass: mean line, reps line, barplot
  - Assert metadata includes barplot path
- [ ] 8.2 **GREEN**: Integrate into `VisualizeDepthProfilesStep.execute()`
  - After generating line plots, check if `data_type == 'biomass'`
  - If yes, call `plot_biomass_depth_barplot()` with aggregated data
  - Save to `figures/00f_depth_profile_{data_type}_barplot.png`
  - Add barplot path to metadata
- [ ] 8.3 **REFACTOR**: Extract barplot generation to helper method `_generate_barplot_if_biomass()`

## 9. Integration: Skip Barplot for Counting Data
- [ ] 9.1 **RED**: Write test `test_visualize_depth_profiles_step_skips_barplot_for_counting()`
  - Mock counting data source
  - Assert Step 00f generates only 2 files: mean line, reps line (NO barplot)
  - Assert metadata does not include barplot_path key
- [ ] 9.2 **GREEN**: Add conditional check
  - Only generate barplot if `source.data_type == 'biomass'`
  - Skip barplot for counting sources
- [ ] 9.3 **REFACTOR**: Add clear comments explaining why barplot is biomass-only

## 10. Edge Cases & Error Handling
- [ ] 10.1 Test: Handle missing 'Control' genotype gracefully (sort without prepending)
- [ ] 10.2 Test: Handle single depth interval (barplot still works, single bar per genotype)
- [ ] 10.3 Test: Handle empty DataFrame (raise ValueError with clear message)
- [ ] 10.4 Test: Handle missing required columns (raise ValueError)
- [ ] 10.5 Implement validation logic for all edge cases

## 11. Documentation & Polish
- [ ] 11.1 Add comprehensive docstring to `plot_biomass_depth_barplot()` with Google format
- [ ] 11.2 Add usage example to docstring showing reference notebook use case
- [ ] 11.3 Update `depth_profile_plots.py` module docstring to mention barplot
- [ ] 11.4 Add inline comments explaining genotype ordering and depth detection logic

## 12. End-to-End Validation
- [ ] 12.1 Run full QC pipeline on EDPIE config: `uv run sleap-roots-analyze qc configs/qc_root_core_edpie.yaml`
- [ ] 12.2 Verify biomass barplot is generated in output figures directory
- [ ] 12.3 Manually inspect barplot:
  - Control genotype is first
  - Genotypes sorted by shallow-layer biomass
  - Two depth bars per genotype (0-30cm, 30-60cm)
  - Error bars visible
  - Stripplot points overlay (if enabled)
- [ ] 12.4 Run counting data through pipeline, verify NO barplot generated

## 13. Test Coverage
- [ ] 13.1 Run pytest with coverage: `uv run pytest --cov=src/sleap_roots_analyze/depth_profile_plots --cov-branch`
- [ ] 13.2 Ensure `plot_biomass_depth_barplot()` has >95% coverage
- [ ] 13.3 Ensure Step 00f biomass branch has >90% coverage

## 14. Code Quality
- [ ] 14.1 Run black formatter: `uv run black src/sleap_roots_analyze/depth_profile_plots.py tests/test_depth_profile_plots.py`
- [ ] 14.2 Run ruff linter: `uv run ruff check src/sleap_roots_analyze/depth_profile_plots.py tests/test_depth_profile_plots.py`
- [ ] 14.3 Fix any linter warnings or errors