## Why

The `create_genotype_image_grid` function is used in the pipeline but currently hardcodes `"features.png"` as the image type. Cylinder scanner experiments use numbered rotation images (e.g., "1.jpg", "36.jpg") instead of RhizoVision feature images. Users also need to configure which trait columns to show statistics for in the image grids.

## What Changes

- Add `genotype_image_grid_image_type` config option to `StaticVizConfig` (default: "features.png")
- Add `genotype_image_grid_trait_cols` config option to `StaticVizConfig` (default: None)
- Update `_create_genotype_image_grids()` in `generate_static_figures.py` to use these config options
- Update active viz configs to use cylinder-appropriate settings

## Impact

- Affected specs: `visualization-pipeline`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/components.py` (StaticVizConfig)
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py` (_create_genotype_image_grids)
  - `configs/active/viz/viz_cylinder_edpie.yaml` and other active configs
