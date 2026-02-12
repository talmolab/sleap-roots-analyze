## 1. Bug Fix: Image-dependent plots not generated (P0)

### 1a. Interactive image plots

- [ ] 1.1 Write test: `_create_image_dependent_plots()` handles dict format `{barcode: {img_type: Path}}`
- [ ] 1.2 Write test: `_create_image_dependent_plots()` generates scatter_with_images.html when valid image_paths dict provided
- [ ] 1.3 Write test: `_create_image_dependent_plots()` generates image_gallery.html when valid image_paths dict provided
- [ ] 1.4 Fix `generate_interactive.py:_create_image_dependent_plots()` to handle dict format from `link_rhizovision_images_to_samples()`
- [ ] 1.5 Verify interactive image plots generated for turface_19genotypes (has image_dir configured)

### 1b. Static genotype image grids

- [ ] 1.6 Write test: `_create_genotype_image_grids()` handles dict format `{barcode: {img_type: Path}}`
- [ ] 1.7 Write test: genotype image grids generated when valid image_paths dict provided
- [ ] 1.8 Fix `generate_static_figures.py:_create_genotype_image_grids()` to handle dict format
- [ ] 1.9 Verify static image grids generated for turface_19genotypes

## 2. Bug Fix: Genotype boxplot x-axis labels illegible (P0)

- [ ] 2.1 Write test: `create_trait_boxplots_by_genotype()` accepts `orientation` parameter ("horizontal", "vertical", "auto")
- [ ] 2.2 Write test: horizontal boxplots put genotype names on y-axis (readable)
- [ ] 2.3 Write test: `orientation="auto"` uses horizontal when n_genotypes > 15
- [ ] 2.4 Write test: `orientation="auto"` uses vertical when n_genotypes <= 15
- [ ] 2.5 Add `boxplot_orientation` config field to `StaticVisualizationConfig` (default: "auto")
- [ ] 2.6 Implement horizontal boxplot rendering in `create_trait_boxplots_by_genotype()`
- [ ] 2.7 Update `create_trait_boxplots_by_genotype_batched()` to pass orientation parameter
- [ ] 2.8 Update `generate_static_figures.py` to use config.static_viz.boxplot_orientation
- [ ] 2.9 Verify turface_150genotypes boxplots are readable (horizontal orientation)
- [ ] 2.10 Verify turface_19genotypes boxplots are readable (should auto-select horizontal since 19 > 15)

## 3. Bug Fix: Final batch wrong aspect ratio (P1)

- [ ] 3.1 Write test: batch with 2 traits has narrower figure than batch with 6 traits
- [ ] 3.2 Write test: figure width scales with actual trait count, not batch_size
- [ ] 3.3 Fix `create_trait_boxplots_by_genotype_batched()` to size figures based on actual traits in batch
- [ ] 3.4 Fix `create_trait_histograms_batched()` to size figures based on actual traits in batch
- [ ] 3.5 Verify no excessive whitespace in final batch figures

## 4. Bug Fix: Duplicate axis labels on boxplots (P2)

- [ ] 4.1 Write test: boxplot subplots do NOT have both y-label and title showing same trait name
- [ ] 4.2 Fix `create_trait_boxplots_by_genotype()` to remove duplicate labeling
- [ ] 4.3 Verify boxplots show trait name only once (either title or axis label)

## 5. Integration Testing

- [ ] 5.1 Run full pipeline on turface_19genotypes with image_dir configured
- [ ] 5.2 Verify scatter_with_images.html generated in figures/interactive/
- [ ] 5.3 Verify image_gallery.html generated in figures/interactive/
- [ ] 5.4 Verify genotype image grids generated in figures/
- [ ] 5.5 Verify boxplots are readable with appropriate orientation
- [ ] 5.6 Run full pipeline on turface_150genotypes
- [ ] 5.7 Verify 150-genotype boxplots are horizontal and readable
- [ ] 5.8 Upload verified outputs to Box
