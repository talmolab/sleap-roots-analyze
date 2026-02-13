## Why

The UMAPAnalysisStep in the Viz pipeline is currently a stub that skips analysis entirely. UMAP (Uniform Manifold Approximation and Projection) provides non-linear dimensionality reduction that complements PCA, revealing clusters and local structure that linear methods miss. The notebooks use UMAP extensively for trait space exploration and publication figures.

## What Changes

- **Implement UMAPAnalysisStep**: Replace stub with full implementation calling `perform_umap_analysis()` from `umap.py`
- **Enable UMAP visualizations**: Static UMAP plots colored by genotype and top traits
- **Enable interactive UMAP**: Interactive plots with image hover (using existing `create_interactive_umap_with_images()`)
- **Metadata propagation**: Ensure `umap_results` flows through pipeline for downstream visualization steps
- **Data export**: Save UMAP embeddings to `data/umap/` directory for reproducibility

## Impact

- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/umap_analysis.py` - Main implementation
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py` - Add UMAP plots
  - `src/sleap_roots_analyze/pipeline/steps/generate_interactive.py` - Already has UMAP code, just needs results
  - `tests/test_step_umap_analysis.py` - New test file (TDD)
  - `tests/test_umap_metadata_flow.py` - Integration tests for metadata flow

- Affected configs:
  - `configs/active/viz/*.yaml` - Set `umap.enabled: true` to use

## Feature Parity with Notebooks

The following notebook features MUST be implemented:

| Notebook Feature | Implementation |
|-----------------|----------------|
| `perform_umap_analysis()` | Call from UMAPAnalysisStep |
| UMAP projection plot by genotype | `GenerateStaticFiguresStep` |
| `create_umap_colored_by_top_traits()` | `GenerateStaticFiguresStep` |
| `create_interactive_umap_with_images()` | `GenerateInteractiveStep` (already wired) |
| `create_interactive_umap_with_hover_highlight()` | `GenerateInteractiveStep` (already wired) |

## Success Criteria

1. All existing tests pass
2. New TDD tests pass (written before implementation)
3. Metadata flows correctly: `umap_results` available in `GenerateInteractiveStep`
4. UMAP embeddings reproducible (same `random_state` → same results)
5. Feature parity with notebooks verified by visual inspection
