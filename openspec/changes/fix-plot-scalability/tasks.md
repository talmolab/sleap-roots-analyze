## 1. Config Template & Validation Fixes

- [ ] 1.1 Write test: config validation emits warning when `umap.enabled: true` and UMAP is not implemented
- [ ] 1.2 Implement config validation warning for UMAP stub
- [ ] 1.3 Write test: pipeline summary includes UMAP skipped reason when UMAP enabled but stubbed
- [ ] 1.4 Ensure pipeline summary clearly indicates UMAP was skipped with reason
- [ ] 1.5 Update config templates to set `static_viz.pca_biplot_top_features: 1` to match intended `extreme` selection behavior
- [ ] 1.6 Add comments in config templates clarifying `pca.n_top_features` (analysis) vs `static_viz.pca_biplot_top_features` (visualization)

## 2. Infrastructure

- [x] 2.1 Write test: verify `plt.close(fig)` is called after figure save in a representative plotting function (covered by 7.1)
- [x] 2.2 Add `plt.close(fig)` to all plotting functions that don't already close figures (covered by 7.3)
- [x] 2.3 Write test: verify `gc.collect()` is called periodically during batch generation (covered by 7.4)
- [x] 2.4 Add `gc.collect()` calls in batch generation loops in `generate_static_figures.py` (covered by 7.4)
- [x] 2.5 Add `adjustText` to project dependencies (or implement simple label repulsion fallback) (already in pyproject.toml)

## 3. Heritability Plot Readability (P0)

- [x] 3.1 Write test: `create_heritability_plot` with <50 traits returns single figure, appearance unchanged
- [x] 3.2 Write test: `create_heritability_plot` with 200+ traits returns list of paginated figures
- [x] 3.3 Write test: paginated heritability figures have sequential batch numbering and readable labels
- [x] 3.4 Implement pagination in `create_heritability_plot` when `n_traits > threshold`
- [x] 3.5 Write test: `generate_static_figures.py` correctly handles paginated heritability output (saves multiple files)
- [x] 3.6 Update `generate_static_figures.py` to handle paginated heritability output

### 3b. Heritability Bar Value Label Overlap (discovered during integration)
- [x] 3.7 Write test: heritability bar value labels use adaptive font size based on trait count (fontsize <= 6 when traits > 30)
- [x] 3.8 Write test: heritability bar value labels use rotation when traits > 15 to prevent horizontal overlap
- [x] 3.9 Implement adaptive font size and rotation for bar value labels in `_create_single_heritability_figure()`
- [x] 3.10 Write test: x-axis tick labels use smaller font when traits > 30
- [x] 3.11 Implement adaptive x-axis tick label font size

## 4. Correlation Heatmap Readability (P0)

- [x] 4.1 Write test: `create_correlation_heatmap` with 19 traits produces figure with original dimensions
- [x] 4.2 Write test: `create_correlation_heatmap` with 200+ traits scales figure size adaptively
- [x] 4.3 Write test: heatmap label font is >= 6pt for all trait counts up to 500
- [x] 4.4 Implement adaptive figure sizing in `create_correlation_heatmap` that scales with trait count
- [x] 4.5 Add font size floor (6pt minimum) and label omission when traits exceed readability threshold
- [x] 4.6 Verify full_correlation_heatmap renders correctly for both 19-trait and 300-trait datasets

## 5. EDA & Variance Decomposition Readability (P0)

- [x] 5.1 Write test: `create_trait_eda_plots` with <50 traits returns figures with current dimensions
- [x] 5.2 Write test: `create_trait_eda_plots` with 200+ traits has non-overlapping x-axis labels (figure width scales or panels paginate)
- [x] 5.3 Implement EDA overview fix: scale figure width with trait count or paginate panels
- [x] 5.4 Write test: `create_variance_decomposition_plot` with <50 traits returns figure with current dimensions
- [x] 5.5 Write test: `create_variance_decomposition_plot` with 200+ traits paginates or shows top-N per panel
- [x] 5.6 Implement variance decomposition fix: paginate or show sorted top-N per panel

## 6. PCA Biplot Label Overlap (P0)

- [x] 6.1 Write test: `create_pca_biplot` with <10 features has unchanged label placement
- [x] 6.2 Write test: `create_pca_biplot` with 10+ features has non-overlapping labels (adjustText or manual repulsion)
- [x] 6.3 Integrate adjustText or manual repulsion for feature loading labels
- [x] 6.4 Write test: biplot renders without error for both 19-genotype and 100+ genotype datasets
- [x] 6.5 Verify biplot label readability across dataset sizes

## 6b. PCA PC Boxplots Sizing and PC Selection (P0, discovered during integration)

Uses existing `adaptive_sizing` config. PC selection uses same variance threshold logic as feature contribution plot.

- [x] 6b.1 Write test: `create_pc_genotype_boxplots()` uses adaptive figsize based on genotype count (width) and n_components (height)
- [x] 6b.2 Write test: `generate_static_figures` uses variance threshold from pca.n_components (if <1) for PC selection
- [x] 6b.3 Write test: PCA PC boxplots with 150 genotypes have width scaled appropriately using adaptive_sizing config
- [x] 6b.4 Implement auto-sizing in `create_pc_genotype_boxplots()` using adaptive_sizing params
- [x] 6b.5 Update `generate_static_figures.py` to pass variance_threshold and use adaptive_sizing for figsize

## 6c. Trait Boxplots by Genotype Sizing (P0, discovered during integration)

Uses existing `adaptive_sizing` config. Fix hardcoded figsize assumption.

- [x] 6c.1 Write test: `create_trait_boxplots_by_genotype_batched()` calculates figsize based on actual batch_size and n_cols
- [x] 6c.2 Write test: trait boxplots with batch_size=6 have proportional subplot dimensions (not vertically stretched)
- [x] 6c.3 Write test: subplot dimensions use adaptive_sizing config params when available
- [x] 6c.4 Fix `create_trait_boxplots_by_genotype_batched()` to compute figsize from batch_size, not hardcoded 16
- [x] 6c.5 Update `generate_static_figures.py` to pass adaptive_sizing config to batched boxplot function

## 7. Memory Management (P1)

- [x] 7.1 Write test: generating 50+ figures in sequence does not accumulate open figure handles (check `plt.get_fignums()`)
- [x] 7.2 Audit all plotting functions for missing `plt.close()` calls
- [x] 7.3 Add `plt.close(fig)` after every `fig.savefig()` call across all modules
- [x] 7.4 Add periodic `gc.collect()` in batch generation loops
- [x] 7.5 Write test: batch generation of 50+ figures keeps memory within bounds (figure count stays low)

## 8. Batch File Reduction (P2)

- [ ] 8.1 Write test: adaptive batch size increases from default when trait count > 100 (e.g., 16→36 subplots per page)
- [x] 8.2 Write test: `save_pdf` config option controls whether PDF files are generated alongside PNG
- [ ] 8.3 Write test: cylinder-scale experiment (300+ traits) generates < 30 batch files per plot type
- [ ] 8.4 Implement adaptive `traits_per_page` in `create_trait_histograms_batched` and `create_trait_boxplots_by_genotype_batched`
- [x] 8.5 Add `save_pdf` config option (default: True) to `StaticVisualizationConfig`
- [x] 8.6 Update batch generation to respect `save_pdf` toggle
- [ ] 8.7 Verify cylinder experiment generates reasonable batch count

## 9. Label Formatting Consistency (P3)

- [x] 9.1 Write test: cross-platform joint plot axis labels match `sanitize_trait_names()` output
- [x] 9.2 Write test: cross-platform boxplot axis labels match `sanitize_trait_names()` output
- [x] 9.3 Write test: cross-platform heatmap axis labels match `sanitize_trait_names()` output
- [x] 9.4 Remove ad-hoc `.replace('_', ' ').title()` in `cross_experiment_analysis.py` and reuse `sanitize_trait_names()` from `data_utils.py`
- [x] 9.5 Apply consistent label formatting to cross-platform boxplot axis labels
- [x] 9.6 Apply consistent label formatting to heatmap axis labels in `reduce_trait_redundancy.py`
- [x] 9.7 Verify cross-platform plots use the same sanitized names as QC pipeline outputs

### 9b. Sanitize Trait Names Improvements (discovered during review)
- [x] 9.8 Write test: units like `cm`, `mm`, `g` are formatted as `(cm)`, `(mm)`, `(g)` not `Cm`, `Mm`, `G`
- [x] 9.9 Write test: `degrees` is formatted as `(°)`
- [x] 9.10 Write test: `DW` (dry weight) is preserved or expanded appropriately
- [x] 9.11 Fix sanitize_trait_names to handle unit suffixes without incorrect capitalization
- [x] 9.12 Add `degrees` → `(°)` conversion to unit handling

## 10. Missing Static Plots — Wire into Pipeline (P6)

### 10a. PCA Feature Contribution Bar Chart
- [x] 10.1 Write test: `generate_static_figures` produces `pca_feature_contributions.{fmt}` when PCA results exist
- [x] 10.2 Write test: feature contribution plot is NOT generated when PCA results are missing
- [x] 10.3 Add `create_feature_contribution_plot()` call to `generate_static_figures.py` PCA section

### 10a2. PCA Feature Contribution Config (discovered during integration)
- [x] 10.3a Write test: `StaticVisualizationConfig` accepts `feature_contribution_variance_threshold` field (default: use pca.n_components)
- [x] 10.3b Write test: `StaticVisualizationConfig` accepts `feature_contribution_top_n` field (default: 20)
- [x] 10.3c Add config fields to `StaticVisualizationConfig`: `feature_contribution_variance_threshold` (Optional[float], default None to inherit from pca.n_components), `feature_contribution_top_n` (int, default 20)
- [x] 10.3d Write test: `generate_static_figures` passes variance_threshold from config to `create_feature_contribution_plot()`
- [x] 10.3e Write test: `generate_static_figures` passes top_n from config to `create_feature_contribution_plot()`
- [x] 10.3f Update `generate_static_figures.py` to pass variance_threshold and top_n from config

### 10b. Phenotype Variation Plots
- [x] 10.4 Write test: config `StaticVisualizationConfig` accepts `create_phenotype_variation_plots` and `phenotype_variation_top_n` fields
- [x] 10.5 Add config fields to `StaticVisualizationConfig`: `create_phenotype_variation_plots` (default: True), `phenotype_variation_top_n` (default: 10)
- [x] 10.6 Write test: `generate_static_figures` produces phenotype variation plots for top N traits when heritability results exist
- [x] 10.7 Write test: phenotype variation plots are skipped with log message when heritability results missing
- [x] 10.8 Write test: each phenotype variation figure is closed after saving (plt.close called)
- [x] 10.9 Add `create_phenotype_variation_plot()` loop to `generate_static_figures.py`, iterating over top N traits by heritability

### 10c. Regression Plots
- [x] 10.10 Write test: config `StaticVisualizationConfig` accepts `regression_trait_pairs` field (list of [x, y] pairs)
- [x] 10.11 Add config field to `StaticVisualizationConfig`: `regression_trait_pairs` (default: empty list)
- [x] 10.12 Write test: `generate_static_figures` produces regression plots for each configured pair
- [x] 10.13 Write test: no regression plots generated when `regression_trait_pairs` is empty
- [x] 10.14 Add `create_regression_plot()` loop to `generate_static_figures.py` for configured trait pairs

### 10d. Genotype Image Grids
- [x] 10.15 Write test: config `StaticVisualizationConfig` accepts `create_genotype_image_grids` field
- [x] 10.16 Add config field to `StaticVisualizationConfig`: `create_genotype_image_grids` (default: True)
- [x] 10.17 Write test: `generate_static_figures` produces image grids when image paths and PCA results are available
- [x] 10.18 Write test: image grids are skipped with log message when image paths are not available
- [x] 10.19 Write test: each image grid figure is closed after saving (plt.close called)
- [x] 10.20 Add `identify_extreme_genotypes_by_pc()` + `create_genotype_image_grid()` loop to `generate_static_figures.py`, guarded by image path availability

## 11. Missing Interactive Plots — Wire into Pipeline (P6)

- [x] 11.1 Write test: config `InteractiveVisualizationConfig` accepts `create_scatter_with_images`, `create_image_viewer`, `create_image_gallery` fields
- [x] 11.2 Add config flags to `InteractiveVisualizationConfig`: `create_scatter_with_images` (default: True), `create_image_viewer` (default: True), `create_image_gallery` (default: True)
- [x] 11.3 Write test: `generate_interactive` produces `scatter_with_images.html` when image paths available and config enabled
- [x] 11.4 Write test: `generate_interactive` produces `pca_image_viewer.html` when image paths available and PCA plot exists
- [x] 11.5 Write test: `generate_interactive` produces `image_gallery.html` when image paths available and config enabled
- [x] 11.6 Write test: all three interactive plots are skipped with log message when image paths not available
- [x] 11.7 Add `create_interactive_scatter_with_images()` call to `generate_interactive.py`, guarded by image path availability
- [x] 11.8 Add `create_html_with_image_viewer()` call to `generate_interactive.py`, using PCA plot as base figure
- [x] 11.9 Add `create_interactive_image_gallery()` call to `generate_interactive.py`, guarded by image path availability

## 12. Missing QC Plot — Outlier Method Comparison (P6)

- [x] 12.1 Write test: `create_outlier_method_comparison_plot()` returns figure with correct bar count matching number of methods
- [x] 12.2 Write test: bar chart includes value labels on each bar
- [x] 12.3 Extract inline outlier method comparison bar chart from notebooks into `create_outlier_method_comparison_plot()` in `outlier_visualization.py`
- [x] 12.4 Write test: `visualize_outliers` step generates comparison bar chart when 2+ methods run
- [x] 12.5 Write test: comparison bar chart is NOT generated when only 1 method run
- [x] 12.6 Add call to `create_outlier_method_comparison_plot()` in `visualize_outliers.py` when multiple methods have been run

## 13. Integration Testing

- [ ] 13.1 Run full pipeline on turface 19-genotype dataset and verify all plots unchanged/improved
- [ ] 13.2 Run full pipeline on cylinder EDPIE dataset and verify all labels readable
- [ ] 13.3 Verify PCA biplot shows correct number of feature arrows per config
- [ ] 13.4 Verify new plots (feature contribution, phenotype variation, image grids, interactive plots) are generated
- [ ] 13.5 Visually inspect all generated figures for both datasets
- [ ] 13.6 Verify no memory issues during cylinder pipeline run
- [ ] 13.7 Verify image-dependent plots are gracefully skipped when no image paths configured
