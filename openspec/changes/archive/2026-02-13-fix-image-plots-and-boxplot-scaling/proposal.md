# Proposal: Fix Image Plots and Boxplot Scaling

## Problem Statement

The fix-plot-scalability changes introduced bugs and left some visualization issues unresolved:

### Bug 1: Image-dependent plots not generated
**Severity: High**

When `image_dir` is configured, image-dependent plots (scatter_with_images.html, image_gallery.html, genotype image grids) are not generated despite images being successfully linked.

**Root cause**: Type mismatch in `generate_interactive.py:315`. The code expects `image_paths` to be a `pd.Series` indexed by DataFrame row indices, but `link_rhizovision_images_to_samples()` returns a `Dict[barcode, Dict[img_type, Path]]`.

The condition `if idx in df.index:` always fails because `idx` is a barcode string (e.g., "0H28413AIU") while `df.index` is the default integer index (0, 1, 2, ...).

### Bug 2: Genotype boxplot x-axis labels illegible
**Severity: High**

Boxplots by genotype are unreadable when n_genotypes > ~10-15. With 19 genotypes, labels completely overlap. With 150 genotypes, the x-axis is a solid black bar.

**Root cause**: Vertical boxplots with genotypes on x-axis don't scale. Need horizontal orientation or label rotation/truncation strategy.

### Bug 3: Final batch has wrong aspect ratio
**Severity: Medium**

When the last batch has fewer traits than batch_size (e.g., 2 traits in a batch configured for 6), the figure maintains the full width, creating excessive whitespace.

**Root cause**: Figure size calculation uses full batch_size even for partial batches.

### Bug 4: Duplicate axis labels on boxplots
**Severity: Low**

Boxplots show trait name both as y-axis label AND as subplot title, creating redundancy.

## Proposed Solution

### For Bug 1 (Image plots):
- Fix `_create_image_dependent_plots()` to handle both dict and Series formats
- The dict format is `{barcode: {img_type: Path}}` - use this directly since barcode is what we need

### For Bug 2 (Genotype labels):
- Add config option `boxplot_orientation: "auto" | "horizontal" | "vertical"`
- When "auto": use horizontal orientation if n_genotypes > 15
- Horizontal boxplots put genotypes on y-axis where labels can be read

### For Bug 3 (Aspect ratio):
- Calculate figure size based on actual number of traits in batch, not batch_size
- Already partially implemented but not working correctly

### For Bug 4 (Duplicate labels):
- Remove subplot title when y-axis label is set, or vice versa

## Scope

- `generate_interactive.py` - Fix dict/Series handling
- `generate_static_figures.py` - Fix dict/Series handling for image grids
- `visualization.py` - Add horizontal boxplot support, fix label duplication
- `components.py` - Add boxplot_orientation config
- Tests for all changes (TDD approach)

## Out of Scope

- Other visualization improvements not related to these bugs
- Cross-platform analysis changes
