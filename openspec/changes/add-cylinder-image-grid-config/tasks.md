## 1. Unit Tests (TDD)

- [x] 1.1 Write test: `StaticVizConfig` accepts `genotype_image_grid_image_type` field (default: "features.png")
- [x] 1.2 Write test: `StaticVizConfig` accepts `genotype_image_grid_trait_cols` field (default: None)
- [x] 1.3 Write test: `_create_genotype_image_grids()` uses `image_type` from config
- [x] 1.4 Write test: `_create_genotype_image_grids()` passes `trait_cols` to `create_genotype_image_grid()`

## 1b. Pipeline Integration Tests (TDD)

- [ ] 1.5 Write test: metadata flows correctly from StatisticalAnalysisStep to GenerateStaticFiguresStep
- [ ] 1.6 Write test: image_paths metadata is preserved through UMAP analysis step
- [ ] 1.7 Write test: full viz pipeline passes image_type and trait_cols to visualization function

## 2. Implementation

- [x] 2.1 Add `genotype_image_grid_image_type: str = "features.png"` to `StaticVizConfig`
- [x] 2.2 Add `genotype_image_grid_trait_cols: Optional[List[str]] = None` to `StaticVizConfig`
- [x] 2.3 Update `_create_genotype_image_grids()` to read `image_type` from config
- [x] 2.4 Update `_create_genotype_image_grids()` to pass `trait_cols` to the function
- [x] 2.5 Run tests to verify implementation

## 3. Config Updates

- [x] 3.1 Update `viz_cylinder_edpie.yaml` with cylinder image settings (image_type: "1.jpg")
- [x] 3.2 Update all active viz configs with explicit settings for traceability
- [x] 3.3 Verify all configs load correctly

## 4. Validation

- [x] 4.1 Run `openspec validate add-cylinder-image-grid-config --strict`
- [ ] 4.2 Run full test suite