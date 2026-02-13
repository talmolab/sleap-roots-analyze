## Context

UMAP analysis is a standard part of the root trait visualization workflow in notebooks. The pipeline infrastructure already has:
- `UMAPConfig` dataclass with all necessary parameters
- `perform_umap_analysis()` function in `umap.py`
- Interactive UMAP visualization functions in `interactive_visualization.py`
- Static UMAP visualization functions in `visualization.py`

The missing piece is the pipeline step that connects these components.

## Goals / Non-Goals

**Goals:**
- Implement UMAPAnalysisStep following PCAAnalysisStep pattern
- Achieve feature parity with Jupyter notebooks
- Preserve metadata flow (especially `image_paths`)
- Export reproducible artifacts (embeddings, parameters)
- Enable both static and interactive UMAP visualizations

**Non-Goals:**
- Optimizing UMAP performance (use default scikit-learn/umap-learn settings)
- Adding new UMAP features not in notebooks
- Implementing UMAP for QC pipeline (Viz pipeline only)

## Decisions

### Decision 1: Follow PCAAnalysisStep Pattern
**What**: Mirror the structure of `pca_analysis.py` for consistency
**Why**: Reduces cognitive load, ensures metadata propagation follows established patterns, makes code review easier

Key patterns to follow:
```python
# 1. Preserve previous metadata
metadata = {
    **prev_result.metadata,
    "umap_results": umap_results,
    ...
}

# 2. Get trait columns from multiple possible keys
trait_cols = (
    prev_result.metadata.get("trait_names")
    or prev_result.metadata.get("valid_trait_names")
    or prev_result.metadata.get("trait_cols")
)

# 3. Save artifacts to data/umap/ directory
umap_dir = run_dir / "data" / "umap"
```

### Decision 2: Store Serializable Results Only
**What**: Store embedding array and parameters, not the UMAP model object
**Why**: The fitted UMAP model is not JSON-serializable and not needed downstream. Only the embedding coordinates are used for visualization.

```python
umap_results_serializable = {
    "embedding": embedding,  # np.ndarray
    "n_neighbors": n_neighbors,
    "min_dist": min_dist,
}
```

### Decision 3: Use Existing Visualization Functions
**What**: Don't create new visualization code; use existing functions from `visualization.py` and `interactive_visualization.py`
**Why**: DRY principle, functions are already tested in notebooks

Functions to reuse:
- `create_umap_colored_by_top_traits()` - static matplotlib plot
- `create_interactive_umap_with_images()` - interactive plotly with images
- `create_interactive_umap_with_hover_highlight()` - interactive with genotype highlighting

### Decision 4: Handle NaN Values Identically to PCA
**What**: Drop rows with NaN in trait columns before UMAP, track indices
**Why**: UMAP cannot handle NaN values; matching indices needed for metadata alignment

```python
data_clean = data[trait_cols].dropna()
logger.info(f"Using {len(data_clean)} samples after dropping NaN values")
```

### Decision 5: Preserve Image Paths Metadata
**What**: Explicitly preserve `image_paths` in output metadata
**Why**: Critical bug was fixed where `StatisticalAnalysisStep` dropped `image_paths`. Must ensure UMAP step doesn't reintroduce this issue.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| UMAP not installed | ImportError at runtime | Check `UMAP_AVAILABLE` flag, skip gracefully with warning |
| Large datasets slow | Long pipeline runtime | Log progress, document expected time |
| Non-deterministic results | Reproducibility concerns | Always use `random_state` from config |
| Metadata lost | Interactive plots broken | TDD tests for metadata flow |

## Data Flow

```
LoadDataAndImagesStep
    ↓ metadata: {image_paths, trait_names, n_samples}
StatisticalAnalysisStep
    ↓ metadata: {**prev, heritability_results, anova_results}
PCAAnalysisStep
    ↓ metadata: {**prev, pca_results, top_features}
UMAPAnalysisStep  ← NEW
    ↓ metadata: {**prev, umap_results}
GenerateStaticFiguresStep
    ↓ reads: pca_results, umap_results, heritability_results
GenerateInteractiveStep
    ↓ reads: pca_results, umap_results, image_paths
```

## Test Strategy (TDD)

### Phase 1: Unit Tests (Red → Green)
1. Write failing test for UMAPAnalysisStep basic execution
2. Write failing test for metadata preservation
3. Write failing test for artifact export
4. Implement UMAPAnalysisStep to pass tests

### Phase 2: Integration Tests
1. Test metadata flow from LoadData → UMAP → Interactive
2. Test that GenerateInteractiveStep receives umap_results
3. Test static UMAP plot generation

### Phase 3: Validation Tests
1. Verify UMAP embedding shape matches sample count
2. Verify reproducibility (same random_state → same results)
3. Verify feature parity with notebook output

## Open Questions

1. **Q: Should UMAP run before or after PCA?**
   A: After PCA (current position). Both are independent analyses, but PCA provides top features used to color UMAP plots.

2. **Q: What if umap-learn is not installed?**
   A: Log warning and skip, setting `umap_status: "skipped_not_installed"` in metadata.

3. **Q: Should we support n_components > 2?**
   A: No, keep it simple. Notebooks only use 2D UMAP for visualization.
