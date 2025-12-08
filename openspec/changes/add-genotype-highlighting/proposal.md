# Genotype Highlighting in Visualization Pipeline

## Why

Researchers need to visually compare specific genotypes of interest against the full panel in PCA biplots and PC boxplots. Currently, all genotypes are colored equally, making it difficult to identify and track specific genotypes across visualizations. The trait_viz notebooks manually implement genotype highlighting using `GENOTYPES_TO_COLOR` and `GENOTYPES_TO_HIGHLIGHT` lists, but this functionality is not available in the automated visualization pipeline.

## What Changes

- Add `genotypes_to_color` and `highlight_genotypes` configuration parameters to `StaticVisualizationConfig`
- Update `GenerateStaticFiguresStep` to pass genotype highlighting parameters to plotting functions (`create_pca_biplot`, `create_pc_genotype_boxplots`)
- Add example genotype lists to `viz_turface_150genotypes.yaml` config
- Add tests to verify genotype highlighting behavior

**Note**: The underlying visualization functions (`create_pca_biplot`, `create_pc_genotype_boxplots` in `visualization.py`) already support these parameters. This change exposes them through the pipeline configuration.

## Impact

- **Affected capability**: `visualization-pipeline`
- **Affected code**:
  - `src/sleap_roots_analyze/pipeline/config/components.py:375-428` - Add config fields
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py:234-290` - Pass parameters
  - `configs/viz_turface_150genotypes.yaml` - Add example configuration
- **Breaking changes**: None (new optional parameters with backward-compatible defaults)
- **Tests needed**:
  - Config validation tests
  - Pipeline step integration tests
  - End-to-end visualization tests with highlighting enabled