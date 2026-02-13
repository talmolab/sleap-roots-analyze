## Why

The current interactive UMAP visualization shows all data points in grey and requires clicking on legend items to highlight individual genotypes. This is inconsistent with the interactive PCA visualization, which colors points by genotype and shows sample Barcode on hover. Users cannot easily distinguish genotypes or identify individual samples in the UMAP plot.

## What Changes

- Interactive UMAP points colored by genotype (matching interactive PCA style)
- Barcode displayed on hover for sample identification
- UMAP parameters shown in plot title
- Consistent style with interactive PCA plot
- Metadata preservation verified (umap_results, image_paths, trait_names)
- Data alignment using clean_indices for correct sample identification

## Impact

- Affected specs: `visualization-pipeline` (interactive plots)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/generate_interactive.py` (_create_interactive_umap method)
  - No changes needed to `interactive_visualization.py` (existing functions support this)
- Tests:
  - Unit tests for coloring and hover behavior
  - Integration tests for metadata flow and data alignment
