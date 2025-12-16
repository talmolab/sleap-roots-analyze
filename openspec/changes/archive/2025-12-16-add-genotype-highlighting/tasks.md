# Implementation Tasks

## 1. Write Tests (TDD - Red Phase)
- [x] 1.1 Write test for `StaticVisualizationConfig` with genotype highlighting parameters (`tests/test_viz_config.py`)
- [x] 1.2 Write test for config validation with invalid genotype names
- [x] 1.3 Write test for `GenerateStaticFiguresStep` passing highlighting params to `create_pca_biplot`
- [x] 1.4 Write test for `GenerateStaticFiguresStep` passing highlighting params to `create_pc_genotype_boxplots`
- [x] 1.5 Run tests - verify they fail (red phase)

## 2. Update Configuration (TDD - Green Phase)
- [x] 2.1 Add `genotypes_to_color: Optional[List[str]] = None` to `StaticVisualizationConfig` in `components.py:377-428`
- [x] 2.2 Add `highlight_genotypes: Optional[List[str]] = None` to `StaticVisualizationConfig`
- [x] 2.3 Update docstring for `StaticVisualizationConfig` with parameter descriptions
- [x] 2.4 Run config tests - verify they pass

## 3. Update Pipeline Step (TDD - Green Phase)
- [x] 3.1 Pass `genotypes_to_color` parameter to `create_pca_biplot` call in `generate_static_figures.py:234-240`
- [x] 3.2 Pass `highlight_genotypes` parameter to `create_pca_biplot` call
- [x] 3.3 Pass `highlight_genotypes` parameter to `create_pc_genotype_boxplots` call in `generate_static_figures.py:285-290`
- [x] 3.4 Run pipeline step tests - verify they pass

## 4. Add Example Configuration
- [x] 4.1 Add `genotypes_to_color` list to `configs/viz_turface_150genotypes.yaml` under `static_viz` section
- [x] 4.2 Add `highlight_genotypes` list to `configs/viz_turface_150genotypes.yaml`
- [x] 4.3 Add comments explaining the feature and providing example genotype names from the notebook

## 5. Integration Testing
- [x] 5.1 Run full viz pipeline with highlighting enabled - verify plots generated correctly
- [x] 5.2 Run full viz pipeline with highlighting disabled (None values) - verify backward compatibility
- [x] 5.3 Verify PCA biplot shows colored genotypes distinctly
- [x] 5.4 Verify PC boxplots show highlighted genotypes in gold with bold labels

## 6. Documentation
- [x] 6.1 Update config YAML comments with usage examples
- [x] 6.2 Add inline code comments explaining parameter flow
- [x] 6.3 Verify all tests pass with `uv run pytest tests/ -x --tb=short`
  - All 187 visualization-related tests pass (test_visualization.py, test_viz_pipeline_config.py, test_step_generate_static_figures.py)
  - 2 pre-existing failures in unrelated tests (test_depth_profile_plots.py, test_step_filter_heritability.py)
