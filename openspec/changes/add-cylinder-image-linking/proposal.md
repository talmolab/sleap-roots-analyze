## Why

The visualization pipeline's `LoadDataAndImagesStep` only supports RhizoVision flatbed scanner images (`{barcode}_c1_p1_features.png` in a flat directory). Cylinder scanner experiments use a different image structure: subdirectories named by barcode containing numbered rotation images (`1.jpg` to `72.jpg`). The notebook `trait_viz_cylinder_20251105.ipynb` works because it uses `link_cylinder_images_from_scan_path()` which reads the `scan_path` column and builds correct paths. The pipeline currently uses `link_rhizovision_images_to_samples()` for all datasets, causing cylinder image grids to fail with "No valid images found."

## What Changes

- Add `image_linking_method` config option to `DataConfig`: "rhizovision" (default) or "cylinder"
- Add `scan_path_col` config option to `DataConfig` for cylinder linking (default: "scan_path")
- Update `LoadDataAndImagesStep` to use correct linking function based on config
- Ensure image_paths metadata flows correctly through pipeline steps to GenerateStaticFiguresStep
- Fix `_create_genotype_image_grids` to handle nested dict format from link_* functions
  (both `link_rhizovision_images_to_samples` and `link_cylinder_images_from_scan_path` return
  `Dict[barcode, Dict[image_type, Path]]`, not the legacy Series format)

## Impact

- Affected specs: `visualization-pipeline`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/components.py` (DataConfig)
  - `src/sleap_roots_analyze/pipeline/steps/load_data_images.py` (LoadDataAndImagesStep)
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py` (format handling)
  - `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` (metadata flow)
  - `src/sleap_roots_analyze/pipeline/steps/umap_analysis.py` (metadata flow)
  - `configs/active/viz/viz_cylinder_edpie.yaml`
