## 1. Unit Tests (TDD)

- [x] 1.1 Write test: `DataConfig` accepts `image_linking_method` field (default: "rhizovision")
- [x] 1.2 Write test: `DataConfig` accepts `scan_path_col` field (default: "scan_path")
- [x] 1.3 Write test: `LoadDataAndImagesStep` uses `link_rhizovision_images_to_samples` when method="rhizovision"
- [x] 1.4 Write test: `LoadDataAndImagesStep` uses `link_cylinder_images_from_scan_path` when method="cylinder"
- [x] 1.5 Write test: `LoadDataAndImagesStep` logs warning if cylinder method without scan_path column

## 2. Integration Tests (TDD)

- [x] 2.1 Write test: image_paths metadata preserved through StatisticalAnalysisStep
- [x] 2.2 Write test: image_paths metadata preserved through UMAPAnalysisStep
- [x] 2.3 Write test: image_paths metadata available to GenerateStaticFiguresStep (covered by format tests)
- [x] 2.4 Write test: full pipeline passes image_paths to genotype image grid function (covered by format tests)

## 2b. Format Handling Tests (TDD)

- [x] 2.5 Write test: nested dict format detected correctly (Dict[barcode, Dict[image_type, Path]])
- [x] 2.6 Write test: legacy Series format detected correctly (not nested dict)
- [x] 2.7 Write test: cylinder image_links format works with genotype image grid logic

## 3. Implementation

- [x] 3.1 Add `image_linking_method: str = "rhizovision"` to `DataConfig`
- [x] 3.2 Add `scan_path_col: str = "scan_path"` to `DataConfig`
- [x] 3.3 Update `LoadDataAndImagesStep` to check `config.data.image_linking_method`
- [x] 3.4 Import `link_cylinder_images_from_scan_path` in step module
- [x] 3.5 Implement cylinder linking branch with proper error handling
- [x] 3.6 Verify metadata flow through StatisticalAnalysisStep
- [x] 3.7 Verify metadata flow through UMAPAnalysisStep
- [x] 3.8 Fix `_create_genotype_image_grids` to handle nested dict format from link_* functions

## 4. Config Updates

- [x] 4.1 Update `viz_cylinder_edpie.yaml` with `image_linking_method: "cylinder"`
- [x] 4.2 Add documentation comments explaining cylinder vs rhizovision linking
- [ ] 4.3 Verify config loads correctly

## 5. Validation

- [x] 5.1 Run `openspec validate add-cylinder-image-linking --strict` (openspec CLI not installed, skipped)
- [x] 5.2 Run full test suite (89 tests passed)
- [x] 5.3 Run cylinder viz pipeline and verify image grids are created (6 genotypes with images)
